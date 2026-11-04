#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <type_traits>
#define OPTIX_INCLUDE_COOPERATIVE_VECTOR 1
#include <optix.h>
#include <optix_device.h>
#include "coop_backend.h"
#include "feature_sampling.h"
#define MPMAT_COOP_DEVICE
#include "rgba16f.h"
#include "lobe_math.h"
#include "lobe_decode.h"

extern "C" { __constant__ __align__(16) unsigned char coop_params[sizeof(CoopLaunch)]; }
#define params (*reinterpret_cast<const CoopLaunch*>(coop_params))

template<class V> __device__ V activate(V x){
    // Same LeakySmeLU function, expressed using vector operations so OptiX can
    // retain its internal cooperative layout between affine layers.
    V t=optixCoopVecMin(optixCoopVecMax(optixCoopVecAdd(x,V(__half(.5f))),__half(0.f)),__half(1.f));
    V curve=optixCoopVecMul(optixCoopVecMul(V(__half(.495f)),t),t);
    V smooth=optixCoopVecAdd(curve,optixCoopVecMul(V(__half(.01f)),x));
    V positive=optixCoopVecStep(x,V(__half(.5f)));
    return optixCoopVecAdd(optixCoopVecMul(positive,x),optixCoopVecMul(optixCoopVecSub(V(__half(1.f)),positive),smooth));
}
template<int N,int K> __device__ OptixCoopVec<__half,N> affine(OptixCoopVec<__half,K> x,int layer){
    return optixCoopVecMatMul<OptixCoopVec<__half,N>,OptixCoopVec<__half,K>,
        OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,
        false,N,K,OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16>(
            x,reinterpret_cast<CUdeviceptr>(params.weights),params.matrix[layer],
            reinterpret_cast<CUdeviceptr>(params.weights),params.bias[layer]);
}
__device__ __forceinline__ unsigned pack_half2(__half a,__half b){
    return unsigned(__half_as_ushort(a))|(unsigned(__half_as_ushort(b))<<16);
}
template<int LOBES> __device__ __forceinline__ void store_packed_lobes(
    const LobeNetwork& l,const __half* values,__half alpha,int id){
    constexpr int stride=(LOBES*7+1+7)&~7;
    // Constant component indices and unrolled stores let this small temporary
    // remain scalar registers instead of a dynamically indexed local array.
    __half decoded[stride];
    #pragma unroll
    for(int j=0;j<stride;++j)decoded[j]=__half(0.f);
    lobe_decode_component(l,values,decoded,0);
    if constexpr(LOBES>1)lobe_decode_component(l,values,decoded,1);
    if constexpr(LOBES>2)lobe_decode_component(l,values,decoded,2);
    if constexpr(LOBES>3)lobe_decode_component(l,values,decoded,3);
    decoded[LOBES*7]=alpha;
    uint4* output=reinterpret_cast<uint4*>(params.outputs)+size_t(id)*(stride/8);
    #pragma unroll
    for(int j=0;j<stride/8;++j)output[j]=make_uint4(
        pack_half2(decoded[j*8],decoded[j*8+1]),pack_half2(decoded[j*8+2],decoded[j*8+3]),
        pack_half2(decoded[j*8+4],decoded[j*8+5]),pack_half2(decoded[j*8+6],decoded[j*8+7]));
}
__device__ __forceinline__ void zero_lobes(const LobeNetwork& l,int id){
    if(params.packed_lobes){
        int groups=(l.stride+7)/8;
        uint4* output=reinterpret_cast<uint4*>(params.outputs)+size_t(id)*groups;
        #pragma unroll
        for(int j=0;j<4;++j)if(j<groups)output[j]=make_uint4(0,0,0,0);
    }else{
        float* output=params.outputs+size_t(id)*l.stride;
        for(int j=0;j<l.stride;++j)output[j]=0;
    }
}

template<int WIDTH,int INPUTS> __device__ void infer(){
    int id=optixGetLaunchIndex().x;
    const Local& ray=params.local.ray(id);const Model& m=params.model;const LobeNetwork& l=params.lobe;
    if(!ray.valid){zero_lobes(l,id);return;}
    OptixCoopVec<__half,INPUTS> input(__half(0.f));bool any=false;
    for(int p=0;p<m.k;++p){bool valid;Half4 value=sample_rgba16f(m,ray,p,params.periodic,params.crop,valid);
        for(int c=0;c<m.c;++c)input[p*m.c+c]=value.v[c];
        input[m.k*m.c+p]=__half(valid?1.f:0.f);any=any||valid;
    }
    float length=1.f/fmaxf(sqrtf(ray.view.x*ray.view.x+ray.view.y*ray.view.y+ray.view.z*ray.view.z),1e-8f);
    int v=m.k*(m.c+1);input[v]=__half(ray.view.x*length);input[v+1]=__half(ray.view.y*length);input[v+2]=__half(ray.view.z*length);
    for(int j=0;j<l.inputs;++j)input[j]=(input[j]-l.half_normalization[j])/l.half_normalization[l.inputs+j];
    auto hidden=activate(affine<WIDTH,INPUTS>(input,0));
    if(l.network.depth==2)hidden=activate(affine<WIDTH,WIDTH>(hidden,1));
    auto result=affine<32,WIDTH>(hidden,l.network.depth);
    __half alpha=any?sigmoid(result[l.stride-1]):__half(0.f);
    if(params.packed_lobes){
        // Cooperative heads have at most 32 outputs, i.e. at most four lobes.
        switch(l.lobes){
        case 1:store_packed_lobes<1>(l,result.data(),alpha,id);break;
        case 2:store_packed_lobes<2>(l,result.data(),alpha,id);break;
        case 3:store_packed_lobes<3>(l,result.data(),alpha,id);break;
        case 4:store_packed_lobes<4>(l,result.data(),alpha,id);break;
        }
    }else{
        float* out=params.outputs+size_t(id)*l.stride;
        for(int c=0;c<l.lobes;++c)lobe_decode_component(l,result.data(),out,c);
        out[l.stride-1]=float(alpha);
    }
}
#define COOP_ENTRY(W,I) extern "C" __global__ void __raygen__coop##W##_##I(){infer<W,I>();}
COOP_ENTRY(16,16) COOP_ENTRY(16,32) COOP_ENTRY(16,48) COOP_ENTRY(16,64)
COOP_ENTRY(32,16) COOP_ENTRY(32,32) COOP_ENTRY(32,48) COOP_ENTRY(32,64)
COOP_ENTRY(64,16) COOP_ENTRY(64,32) COOP_ENTRY(64,48) COOP_ENTRY(64,64)

// Four half values per naturally aligned store instead of four independent
// strided scalar stores. The common latent16 cache has a 40-byte ray stride.
__device__ __forceinline__ void store_code4(__half* target,__half a,__half b,__half c,__half d){
    *reinterpret_cast<uint2*>(target)=make_uint2(unsigned(__half_as_ushort(a))|(unsigned(__half_as_ushort(b))<<16),
        unsigned(__half_as_ushort(c))|(unsigned(__half_as_ushort(d))<<16));
}

// The opaque mip path encodes the ray first, then evaluates the latent IBL
// head in the same cooperative-vector invocation. 21-wide heads are padded
// to 32 only in converted matrices; zero columns exclude padded activations.
template<int RAY_WIDTH,int HEAD_WIDTH,int HEAD_INPUTS> __device__ void infer_mip(){
    int id=optixGetLaunchIndex().x;
    const Local& ray=params.local.ray(id);const Model& m=params.model;const LobeNetwork& l=params.lobe;
    if(!ray.valid){zero_lobes(l,id);
        if(params.appearance_codes)params.appearance_codes[size_t(id)*m.stride+m.z]=__half(0.f);
        return;}
    OptixCoopVec<__half,32> input(__half(0.f));
    float lod=fminf(fmaxf(m.feature_lod<0?ray.lod:m.feature_lod,0.f),float(m.mip_levels-1));
    for(int p=0;p<10;++p){
        if(m.profiled_inputs){input[p*2]=m.profiled_inputs[size_t(id)*23+p*2];input[p*2+1]=m.profiled_inputs[size_t(id)*23+p*2+1];}
        else{Half4 value=sample_mip_texture(m,ray,p,params.periodic,params.crop,lod);
            input[p*2]=value.v[0];input[p*2+1]=value.v[1];}
    }
    float length=1.f/fmaxf(sqrtf(ray.view.x*ray.view.x+ray.view.y*ray.view.y+ray.view.z*ray.view.z),1e-8f);
    __half view[3]={__half(ray.view.x*length),__half(ray.view.y*length),__half(ray.view.z*length)};
    if(m.profiled_inputs)for(int j=0;j<3;++j)view[j]=m.profiled_inputs[size_t(id)*23+20+j];
    for(int j=0;j<3;++j)input[20+j]=view[j];
    auto hidden=activate(affine<RAY_WIDTH,32>(input,0));
    if(m.depth==2)hidden=activate(affine<RAY_WIDTH,RAY_WIDTH>(hidden,1));
    auto latent=affine<16,RAY_WIDTH>(hidden,m.depth);
    if(params.appearance_codes){
        __half* code=params.appearance_codes+size_t(id)*m.stride;
        #pragma unroll
        for(int j=0;j<16;j+=4)store_code4(code+j,latent[j],latent[j+1],latent[j+2],latent[j+3]);
        store_code4(code+16,__half(1.f),view[0],view[1],view[2]);
    }
    OptixCoopVec<__half,HEAD_INPUTS> head_input(__half(0.f));
    for(int j=0;j<l.inputs;++j){__half value=j<16?latent[j]:view[j-16];
        head_input[j]=(value-l.half_normalization[j])/l.half_normalization[l.inputs+j];}
    int start=m.depth+1;
    auto head_hidden=activate(affine<HEAD_WIDTH,HEAD_INPUTS>(head_input,start));
    if(l.network.depth==2)head_hidden=activate(affine<HEAD_WIDTH,HEAD_WIDTH>(head_hidden,start+1));
    auto result=affine<32,HEAD_WIDTH>(head_hidden,start+l.network.depth);
    __half alpha=l.learned_alpha?sigmoid(result[21]):__half(1.f);
    if(params.packed_lobes)store_packed_lobes<3>(l,result.data(),alpha,id);
    else{
        float* out=params.outputs+size_t(id)*l.stride;
        for(int c=0;c<3;++c)lobe_decode_component(l,result.data(),out,c);
        out[21]=float(alpha);
    }
}
#define MIP_COOP_ENTRY(R,W,I) extern "C" __global__ void __raygen__mip##R##_##W##_##I(){infer_mip<R,W,I>();}
MIP_COOP_ENTRY(16,32,16) MIP_COOP_ENTRY(16,32,32)
MIP_COOP_ENTRY(16,64,16) MIP_COOP_ENTRY(16,64,32)
MIP_COOP_ENTRY(64,32,16) MIP_COOP_ENTRY(64,32,32)
MIP_COOP_ENTRY(64,64,16) MIP_COOP_ENTRY(64,64,32)

// Source-network references use the same texture sampling and angular half
// arithmetic as the native oracle, with cooperative affine layers.
template<int WIDTH,int INPUTS> __device__ void encode_appearance_ray(){
    int id=optixGetLaunchIndex().x;const Model& m=params.model;
    Local ray=params.local.ray(id);__half* code=params.appearance_codes+size_t(id)*m.stride;
    if(!ray.valid){code[m.z]=__half(0.f);return;}
    OptixCoopVec<__half,INPUTS> input(__half(0.f));bool any=m.kind==8;
    float lod=fminf(fmaxf(m.feature_lod<0?ray.lod:m.feature_lod,0.f),float(m.mip_levels-1));
    for(int p=0;p<m.k;++p){bool valid=true;Half4 value;
        if(m.kind==8)value=sample_mip_texture(m,ray,p,params.periodic,params.crop,lod);
        else value=sample_rgba16f(m,ray,p,params.periodic,params.crop,valid);
        for(int c=0;c<m.c;++c)input[p*m.c+c]=value.v[c];
        if(m.kind!=8)input[m.k*m.c+p]=__half(valid?1.f:0.f);any=any||valid;
    }
    float inv=1.f/fmaxf(sqrtf(ray.view.x*ray.view.x+ray.view.y*ray.view.y+ray.view.z*ray.view.z),1e-8f);
    __half view[3]={__half(ray.view.x*inv),__half(ray.view.y*inv),__half(ray.view.z*inv)};
    int offset=m.k*(m.c+(m.kind==8?0:1));
    for(int j=0;j<3;++j)input[offset+j]=view[j];
    auto hidden=activate(affine<WIDTH,INPUTS>(input,0));
    if(m.depth==2)hidden=activate(affine<WIDTH,WIDTH>(hidden,1));
    auto result=affine<32,WIDTH>(hidden,m.depth);
    __half alpha=m.kind==8?__half(1.f):(any?sigmoid(result[m.z]):__half(0.f));
    if(m.z==16){
        #pragma unroll
        for(int j=0;j<16;j+=4)store_code4(code+j,result[j],result[j+1],result[j+2],result[j+3]);
        store_code4(code+16,alpha,view[0],view[1],view[2]);
    }else{
        for(int j=0;j<m.z;++j)code[j]=result[j];code[m.z]=alpha;
        for(int j=0;j<3;++j)code[m.z+1+j]=view[j];
    }
}

template<int WIDTH,int INPUTS> __device__ void decode_appearance(){
    int id=optixGetLaunchIndex().x;const Model& m=params.model;
    const __half* code=params.appearance_codes+size_t(id)*m.stride;
    float4* out=reinterpret_cast<float4*>(params.outputs)+id;float alpha=float(code[m.z]);
    if(alpha==0.f){*out=make_float4(0,0,0,0);return;}
    float4 direction=params.directions[id];float3 v=make_float3(direction.x,direction.y,direction.z);
    if(!isfinite(v.x)||!isfinite(v.y)||!isfinite(v.z)){*out=make_float4(0,0,0,alpha);return;}
    float inv=1.f/fmaxf(sqrtf(v.x*v.x+v.y*v.y+v.z*v.z),1e-8f);
    Half3 wi=hnormalize(hvec(make_float3(v.x*inv,v.y*inv,v.z*inv)));
    Half3 wo=hnormalize({code[m.z+1],code[m.z+2],code[m.z+3]});
    __half zero=__half(0.f),one=__half(1.f);
    Half3 h=hnormalize({wi.x+wo.x,wi.y+wo.y,wi.z+wo.z});
    if(h.x==zero&&h.y==zero&&h.z==zero)h={zero,zero,one};
    Half3 axis={h.y,-h.x,zero},cr=hcross(axis,wi);
    __half scale=hdot(axis,wi)/__hmax(one+h.z,__half(1e-6f));
    Half3 d={wi.x*h.z+cr.x+scale*axis.x,wi.y*h.z+cr.y+scale*axis.y,wi.z*h.z+cr.z+scale*axis.z};
    Half3 vectors[4]={wi,wo,h,d};
    OptixCoopVec<__half,INPUTS> input(zero);
    for(int j=0;j<m.z;++j)input[j]=code[j];
    for(int j=0;j<12;++j)input[m.z+j]=hcomponent(vectors[j/3],j%3);
    auto hidden=activate(affine<WIDTH,INPUTS>(input,0));
    int depth=((m.kind==4&&m.depth==1)||m.kind==7)?1:2;
    if(depth==2)hidden=activate(affine<WIDTH,WIDTH>(hidden,1));
    auto result=affine<16,WIDTH>(hidden,depth);float rgb[3];
    for(int j=0;j<3;++j){float encoded=float(softplus(result[j]));
        rgb[j]=(m.mu>0?expm1f(fminf(encoded*log1pf(m.mu),80.f))/m.mu:encoded)*alpha;}
    *out=make_float4(rgb[0],rgb[1],rgb[2],alpha);
}
#define APPEARANCE_ENTRY(W,I) \
    extern "C" __global__ void __raygen__encode##W##_##I(){encode_appearance_ray<W,I>();} \
    extern "C" __global__ void __raygen__appearance##W##_##I(){decode_appearance<W,I>();}
#define APPEARANCE_WIDTH(W) APPEARANCE_ENTRY(W,16) APPEARANCE_ENTRY(W,32) APPEARANCE_ENTRY(W,48) APPEARANCE_ENTRY(W,64)
APPEARANCE_WIDTH(16) APPEARANCE_WIDTH(32) APPEARANCE_WIDTH(64) APPEARANCE_WIDTH(128)
