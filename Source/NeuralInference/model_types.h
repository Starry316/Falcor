#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
struct FeatureTexture {cudaArray_t array=nullptr;cudaTextureObject_t clamp=0,wrap=0;};
struct FeatureMipTexture {cudaMipmappedArray_t array=nullptr;cudaTextureObject_t clamp=0,wrap=0,clamp_point=0,wrap_point=0;};
struct Model {
    int k,r,c,w,aw,z,bands,angular,stride,kind=0;
    int angle_mode=0,angle_r=0,angle_c=0,frames=0,appearance_inputs=0,depth=2;
    int mip_levels=1;float feature_lod=-1,feature_lod_bias=0;size_t mip_offsets[11]={};FeatureMipTexture mip_texture;
    float mu,epsilon,bounds[6];
    float *features,*heights,*weights;
    int synthesis_mode=0;uint32_t synthesis_seed=17;
    float synthesis_scale=2,synthesis_power=7,*gaussian=nullptr,*inverse_histogram=nullptr;
    float synthesis_rotation=0,synthesis_border=.5f;
    const float *matrix[9],*bias[9];
    __half *half_weights=nullptr;
    __half *half_features=nullptr,*half_gaussian=nullptr,*half_inverse_histogram=nullptr;
    const __half *half_matrix[9],*half_bias[9];
    bool nearest_fetch=false;
    bool fp16=false,half_compatible=true,half_features_compatible=true;
    const __half* profiled_inputs=nullptr; // Diagnostic-only split feature/MLP pass.
    FeatureTexture feature_texture,gaussian_texture;
};
struct LobeNetwork {
    bool latent=false,learned_alpha=false;
    Model network={};int inputs=0,lobes=2,stride=15,parameterization=0;float min_r=.12f,log_min_r=-2.1202635f,scale[3]={1,1,1};float* normalization=nullptr;__half* half_normalization=nullptr;
    __half* tensor_weights=nullptr;
};
struct Local { float3 o,d,view,tangent,bitangent,normal; int valid; float lod=0; };
// External renderer's exact FP32 ray layout; avoids expanding a 48-byte record
// into the 80-byte native full geometry cache merely to run inference.
struct ExternalMaterialRay {float3 o,d,view;float lod;uint32_t valid,reserved;};
static_assert(sizeof(ExternalMaterialRay)==48,"External material ray stride");

// Production FP16 geometry: keep sampling coordinates FP32. Unit vectors use
// SNORM16 (higher angular precision than half), and bitangent is reconstructed.
struct alignas(16) PackedLocal {float3 o,d;int16_t view[3];__half lod;};
struct alignas(16) PackedFrame {int16_t directions[6];uint32_t flags;};
static_assert(sizeof(Local)==80&&sizeof(PackedLocal)==32&&sizeof(PackedFrame)==16,"Geometry cache strides");
struct LocalCache {
    Local* full;PackedLocal* packed;PackedFrame* frames;
    const ExternalMaterialRay* external;
    __host__ __device__ explicit operator bool()const{return full||packed||external;}
#ifdef __CUDACC__
    __device__ __forceinline__ static float dot3(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
    __device__ __forceinline__ static float3 cross3(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
    __device__ __forceinline__ static float3 normalize(float3 a){float s=rsqrtf(fmaxf(dot3(a,a),1e-30f));return make_float3(a.x*s,a.y*s,a.z*s);}
    __device__ __forceinline__ Local ray(int id)const{
        if(external){const auto& p=external[id];Local v={};
            if(!p.valid||!isfinite(p.lod)||!isfinite(p.o.x)||!isfinite(p.o.y)||!isfinite(p.o.z)||
                !isfinite(p.d.x)||!isfinite(p.d.y)||!isfinite(p.d.z)||!isfinite(p.view.x)||!isfinite(p.view.y)||!isfinite(p.view.z))return v;
            v.o=p.o;v.d=p.d;v.view=p.view;v.lod=fmaxf(p.lod,0.f);v.valid=1;return v;}
        if(full)return full[id];const PackedLocal& p=packed[id];Local v={};
        if(__half_as_ushort(p.lod)&0x8000)return v;v.o=p.o;v.d=p.d;v.valid=1;v.lod=float(p.lod);
        v.view=make_float3(p.view[0]*(1.f/32767),p.view[1]*(1.f/32767),p.view[2]*(1.f/32767));return v;
    }
    __device__ __forceinline__ Local frame(int id)const{
        if(full)return full[id];const PackedFrame& p=frames[id];Local v={};if(!(p.flags&1))return v;v.valid=1;
        const int16_t* q=p.directions;float3 t=make_float3(q[0]*(1.f/32767),q[1]*(1.f/32767),q[2]*(1.f/32767));
        v.normal=normalize(make_float3(q[3]*(1.f/32767),q[4]*(1.f/32767),q[5]*(1.f/32767)));
        float projection=dot3(t,v.normal);
        v.tangent=normalize(make_float3(t.x-projection*v.normal.x,t.y-projection*v.normal.y,t.z-projection*v.normal.z));
        v.bitangent=cross3(v.normal,v.tangent);if(p.flags&2){v.bitangent.x=-v.bitangent.x;v.bitangent.y=-v.bitangent.y;v.bitangent.z=-v.bitangent.z;}
        return v;
    }
    __device__ __forceinline__ Local operator[](int id)const{
        if(full)return full[id];Local v=ray(id),f=frame(id);v.tangent=f.tangent;v.bitangent=f.bitangent;v.normal=f.normal;return v;
    }
    __device__ __forceinline__ void store(int id,const Local& v)const{
        if(full){full[id]=v;return;}PackedLocal p={};PackedFrame f={};p.lod=__half(-1.f);
        if(v.valid){p.o=v.o;p.d=v.d;p.lod=__half(v.lod);f.flags=1;
            if(dot3(cross3(v.normal,v.tangent),v.bitangent)<0)f.flags|=2;
            float view[]={v.view.x,v.view.y,v.view.z},axes[]={v.tangent.x,v.tangent.y,v.tangent.z,v.normal.x,v.normal.y,v.normal.z};
            for(int j=0;j<3;++j)p.view[j]=int16_t(__float2int_rn(fminf(fmaxf(view[j],-1.f),1.f)*32767));
            for(int j=0;j<6;++j)f.directions[j]=int16_t(__float2int_rn(fminf(fmaxf(axes[j],-1.f),1.f)*32767));
        }packed[id]=p;frames[id]=f;
    }
    __device__ __forceinline__ bool valid(int id)const{return full?full[id].valid!=0:(frames[id].flags&1)!=0;}
    __device__ __forceinline__ void set_lod(int id,float lod)const{if(full)full[id].lod=lod;else packed[id].lod=__half(lod);}
    __device__ __forceinline__ float lod(int id)const{return full?full[id].lod:float(packed[id].lod);}
#endif
};
