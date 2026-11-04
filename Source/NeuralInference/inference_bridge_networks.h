#pragma once
// Shared native/bridge feature encoding and appearance inference kernels.
__device__ float angle_value(float3 d,int index,int bands){
    const float xyz[3]={d.x,d.y,d.z};if(index<3)return xyz[index];
    int pos=index-3;bool cosine=pos>=3*bands;pos%=3*bands;
    float angle=xyz[pos/bands]*(3.14159265358979323846f*float(1<<(pos%bands)));
    return cosine?cosf(angle):sinf(angle);
}

#include "feature_sampling.h"

#include "rgba16f.h"

__device__ void dense(Model m,int layer,const float* in,float* out,int ni,int no,bool silu){
    int lane=threadIdx.x&31;
    for(int neuron=lane;neuron<no;neuron+=32){
        float value;
        value=m.bias[layer][neuron];
        for(int j=0;j<ni;++j)value+=in[j]*m.matrix[layer][j*no+neuron];
        if(silu&&(m.kind==2||m.kind==4||m.kind==5||m.kind==6||m.kind==7||m.kind==8)){
            out[neuron]=value<=-.5f?.01f*value:(value>=.5f?value:.495f*(value+.5f)*(value+.5f)+.01f*value);
        }else out[neuron]=silu?((m.kind==1||m.kind==3)?fmaxf(value,0):value/(1+expf(-value))):value;
    }
    __syncwarp();
}

#include "lobe_math.h"
__device__ void dense(Model m,int layer,const __half* in,__half* out,int ni,int no,bool activate){
    int lane=threadIdx.x&31;
    for(int neuron=lane;neuron<no;neuron+=32){
        __half value=m.half_bias[layer][neuron];
        for(int j=0;j<ni;++j)value=__hfma(in[j],m.half_matrix[layer][j*no+neuron],value);
        if(activate){
            if(m.kind==2||m.kind==4||m.kind==5||m.kind==6||m.kind==7||m.kind==8){
                __half half=__float2half_rn(.5f),leak=__float2half_rn(.01f);
                value=value<=-half?leak*value:(value>=half?value:__float2half_rn(.495f)*(value+half)*(value+half)+leak*value);
            }else value=(m.kind==1||m.kind==3)?__hmax(value,__float2half_rn(0.f)):value*sigmoid(value);
        }
        out[neuron]=value;
    }
    __syncwarp();
}

template<class T> __device__ bool neumip_lookup(Model m,int slot,float ux,float uy,bool periodic,float crop,T* out){
    bool valid=isfinite(ux)&&isfinite(uy);
    if(periodic){ux=ux-floorf(ux);uy=uy-floorf(uy);ux=crop+(1-2*crop)*ux;uy=crop+(1-2*crop)*uy;}
    else valid=valid&&ux>=0&&ux<=1&&uy>=0&&uy<=1;
    float gx=2*ux-1,gy=2*uy-1;int r=m.r;
    bool circular=periodic&&crop==0;
    if(circular){gx=gx*m.r/(m.r+2);gy=gy*m.r/(m.r+2);r+=2;}
    float x=fminf(fmaxf(((gx+1)*r-1)*.5f,0),float(r-1)),y=fminf(fmaxf(((gy+1)*r-1)*.5f,0),float(r-1));
    if(m.nearest_fetch){x=floorf(x+.5f);y=floorf(y+.5f);}
    int ix=int(floorf(x)),iy=int(floorf(y));float fx=x-ix,fy=y-iy;
    int lane=threadIdx.x&31;
    if(lane<7){T value=T(0.f),tx=T(fx),ty=T(fy),one=T(1.f);
        if(valid&&m.nearest_fetch)value=texel<T>(m,slot,lane,ix,iy,circular);
        else if(valid){value=texel<T>(m,slot,lane,ix,iy,circular)*((one-tx)*(one-ty));
            value+=texel<T>(m,slot,lane,ix+1,iy,circular)*(tx*(one-ty));
            value+=texel<T>(m,slot,lane,ix,iy+1,circular)*((one-tx)*ty);
            value+=texel<T>(m,slot,lane,ix+1,iy+1,circular)*(tx*ty);}
        out[lane]=value;
    }
    __syncwarp();return valid;
}

template<class T> __device__ void encode_neumip(Model m,const Local& ray,bool periodic,float crop,T* a,T* b,T* code,float* samples){
    int lane=threadIdx.x&31;float3 view=fnorm(ray.view);
    bool usable=fabsf(ray.d.z)>m.epsilon;float dz=usable?ray.d.z:1.f;
    float t=(m.bounds[5]-ray.o.z)/dz,ex=m.bounds[3]-m.bounds[0],ey=m.bounds[4]-m.bounds[1];
    float ux=(ray.o.x+t*ray.d.x-m.bounds[0])/ex,uy=(ray.o.y+t*ray.d.y-m.bounds[1])/ey;
    bool valid=neumip_lookup(m,0,ux,uy,periodic,crop,a);
    T feature=lane<7?a[lane]:T(0.f);
    if(samples&&lane<7)samples[lane]=feature;
    if(lane==7)a[7]=view.x;if(lane==8)a[8]=view.y;__syncwarp();
    dense(m,0,a,b,9,25,true);dense(m,1,b,a,25,25,true);dense(m,2,a,b,25,25,true);dense(m,3,b,a,25,1,false);
    float depth=a[0];__syncwarp();
    if(lane<7)a[lane]=feature;if(lane==7)a[7]=view.x;if(lane==8)a[8]=view.y;__syncwarp();
    dense(m,8,a,b,9,1,false);T alpha=sigmoid(b[0]);__syncwarp();
    float factor=depth*(m.kind==3?.05f:1.f)*(view.z/fmaxf(view.z,.6f));
    valid=neumip_lookup(m,1,ux+factor*ray.d.x/dz,uy+factor*(ex/ey)*ray.d.y/dz,periodic,crop,a)&&valid;
    if(lane<7){code[lane]=a[lane];if(samples)samples[7+lane]=a[lane];}
    if(lane==0){code[7]=valid&&usable&&t>=0&&view.z>0?alpha:T(0.f);code[8]=view.x;code[9]=view.y;}
}

template<class T> __device__ void decode_material(Model m,T* a,T* b){
    if(m.kind==1||m.kind==3){dense(m,4,a,b,11,25,true);dense(m,5,b,a,25,25,true);dense(m,6,a,b,25,25,true);dense(m,7,b,a,25,m.kind==3?6:3,false);
        int lane=threadIdx.x&31;if(lane<3){T value=a[lane];if(m.kind==3){T y=a[0],u=a[1],v=a[2];
            T logarithm=lane==0?y+T(1.13983f)*v:(lane==1?y-T(.39465f)*u-T(.58060f)*v:y+T(2.03211f)*u);
            if constexpr(std::is_same_v<T,float>)value=expm1f(logarithm)*(T(.1f)+sigmoid(a[3+lane]));
            else value=(hexp(logarithm)-T(1.f))*(T(.1f)+sigmoid(a[3+lane]));}b[lane]=value;}__syncwarp();}
    else if(m.kind==8){int start=m.depth+1;dense(m,start,a,b,m.appearance_inputs,m.aw,true);
        dense(m,start+1,b,a,m.aw,m.aw,true);dense(m,start+2,a,b,m.aw,3,false);}
    else if(m.kind==4&&m.depth==1){dense(m,2,a,b,m.appearance_inputs,m.aw,true);dense(m,3,b,a,m.aw,3,false);
        int lane=threadIdx.x&31;if(lane<3)b[lane]=a[lane];__syncwarp();}
    else if(m.kind==7){dense(m,3,a,b,m.appearance_inputs,m.aw,true);dense(m,4,b,a,m.aw,3,false);
        int lane=threadIdx.x&31;if(lane<3)b[lane]=a[lane];__syncwarp();}
    else{dense(m,3,a,b,m.appearance_inputs,m.aw,true);dense(m,4,b,a,m.aw,m.aw,true);dense(m,5,a,b,m.aw,3,false);}
}

__device__ float component(float3 v,int j){return j==0?v.x:(j==1?v.y:v.z);}
__device__ float3 fcross(float3 a,float3 b){return make_float3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
__device__ float2 hemisphere_square(float3 v){
    float r=fminf(sqrtf(v.x*v.x+v.y*v.y),1.f),pi=CUDART_PI_F;
    float phi=r>1e-12f?atan2f(v.y,v.x):0.f;if(phi<-.25f*pi)phi+=2*pi;
    float x,y;
    if(phi<.25f*pi){x=r;y=phi*r/(pi*.25f);}
    else if(phi<.75f*pi){y=r;x=-(phi-.5f*pi)*r/(pi*.25f);}
    else if(phi<1.25f*pi){x=-r;y=-(phi-pi)*r/(pi*.25f);}
    else{y=-r;x=(phi-1.5f*pi)*r/(pi*.25f);}
    return make_float2(fminf(fmaxf((x+1)*.5f,0.f),1.f),fminf(fmaxf((y+1)*.5f,0.f),1.f));
}
template<class T=float> __device__ T angular_lookup(Model m,int slot,int channel,float2 uv){
    float gx=2*uv.x-1,gy=2*uv.y-1;
    float x=fminf(fmaxf(((gx+1)*m.angle_r-1)*.5f,0.f),float(m.angle_r-1));
    float y=fminf(fmaxf(((gy+1)*m.angle_r-1)*.5f,0.f),float(m.angle_r-1));
    int ix=int(floorf(x)),iy=int(floorf(y)),jx=min(ix+1,m.angle_r-1),jy=min(iy+1,m.angle_r-1);
    float fx=x-ix,fy=y-iy;
    size_t base=size_t(m.k)*m.c*m.r*m.r+size_t(slot*m.angle_c+channel)*m.angle_r*m.angle_r;
    T tx=T(fx),ty=T(fy),one=T(1.f);
    T value=feature_at<T>(m,base+iy*m.angle_r+ix)*((one-tx)*(one-ty));value+=feature_at<T>(m,base+iy*m.angle_r+jx)*(tx*(one-ty));
    value+=feature_at<T>(m,base+jy*m.angle_r+ix)*((one-tx)*ty);value+=feature_at<T>(m,base+jy*m.angle_r+jx)*(tx*ty);return value;
}
__device__ void prepare_appearance(Model m,const float* code,float3 direction,float* a){
    int lane=threadIdx.x&31;
    for(int j=lane;j<m.z;j+=32)a[j]=code[j];
    if(m.kind!=2&&m.kind!=4&&m.kind!=5&&m.kind!=6&&m.kind!=7&&m.kind!=8){
        for(int j=lane;j<m.angular;j+=32){a[m.z+j]=code[m.z+1+j];a[m.z+m.angular+j]=angle_value(direction,j,m.bands);}
    }else{
        float3 wi=fnorm(direction),wo=fnorm(make_float3(code[m.z+1],code[m.z+2],code[m.z+3]));
        float3 sum=make_float3(wi.x+wo.x,wi.y+wo.y,wi.z+wo.z),h=fdot(sum,sum)>1e-12f?fnorm(sum):make_float3(0,0,1);
        float3 axis=make_float3(h.y,-h.x,0),cr=fcross(axis,wi);float scale=fdot(axis,wi)/fmaxf(1+h.z,1e-6f);
        float3 d=make_float3(wi.x*h.z+cr.x+scale*axis.x,wi.y*h.z+cr.y+scale*axis.y,wi.z*h.z+cr.z+scale*axis.z);
        float3 vectors[4]={wi,wo,h,d};
        if(m.angle_mode==0){
            for(int j=lane;j<12*(1+m.frames);j+=32){
                int f=j/12,v=(j%12)/3,c=j%3;
                float value=component(vectors[v],c);
                if(f){const float* frame=code+m.z+4+(f-1)*9+c*3;value=frame[0]*vectors[v].x+frame[1]*vectors[v].y+frame[2]*vectors[v].z;}
                a[m.z+j]=value;
            }
        }else{
            int raw=m.angle_mode==2?6:0;
            if(raw&&lane<6)a[m.z+lane]=component(vectors[lane/3],lane%3);
            float2 uv[2]={hemisphere_square(h),hemisphere_square(d)};
            for(int j=lane;j<2*m.angle_c;j+=32)a[m.z+raw+j]=angular_lookup(m,j/m.angle_c,j%m.angle_c,uv[j/m.angle_c]);
        }
    }
    __syncwarp();
}

__device__ void prepare_appearance(Model m,const __half* code,float3 direction,__half* a){
    int lane=threadIdx.x&31;
    for(int j=lane;j<m.z;j+=32)a[j]=code[j];
    if(m.kind!=2&&m.kind!=4&&m.kind!=5&&m.kind!=6&&m.kind!=7&&m.kind!=8){
        for(int j=lane;j<m.angular;j+=32){a[m.z+j]=code[m.z+1+j];a[m.z+m.angular+j]=__float2half_rn(angle_value(direction,j,m.bands));}
    }else{
        __half zero=__float2half_rn(0.f),one=__float2half_rn(1.f);
        Half3 wi=hnormalize(hvec(direction)),wo=hnormalize({code[m.z+1],code[m.z+2],code[m.z+3]});
        Half3 sum={wi.x+wo.x,wi.y+wo.y,wi.z+wo.z},h=hnormalize(sum);
        if(h.x==zero&&h.y==zero&&h.z==zero)h={zero,zero,one};
        Half3 axis={h.y,-h.x,zero},cr=hcross(axis,wi);__half scale=hdot(axis,wi)/__hmax(one+h.z,__float2half_rn(1e-6f));
        Half3 d={wi.x*h.z+cr.x+scale*axis.x,wi.y*h.z+cr.y+scale*axis.y,wi.z*h.z+cr.z+scale*axis.z};
        Half3 vectors[4]={wi,wo,h,d};
        if(m.angle_mode==0){
            for(int j=lane;j<12*(1+m.frames);j+=32){int f=j/12,v=(j%12)/3,c=j%3;
                __half value=hcomponent(vectors[v],c);
                if(f){const __half* frame=code+m.z+4+(f-1)*9+c*3;value=frame[0]*vectors[v].x+frame[1]*vectors[v].y+frame[2]*vectors[v].z;}
                a[m.z+j]=value;
            }
        }else{
            int raw=m.angle_mode==2?6:0;
            if(raw&&lane<6)a[m.z+lane]=hcomponent(vectors[lane/3],lane%3);
            // Angular-chart coordinates, like spatial UVs, retain FP32 precision.
            float2 uv[2]={hemisphere_square(make_float3(float(h.x),float(h.y),float(h.z))),hemisphere_square(make_float3(float(d.x),float(d.y),float(d.z)))};
            for(int j=lane;j<2*m.angle_c;j+=32)a[m.z+raw+j]=angular_lookup<__half>(m,j/m.angle_c,j%m.angle_c,uv[j/m.angle_c]);
        }
    }
    __syncwarp();
}

#include "lobe_decode.h"

template<class T> __device__ void encode_lobe(LobeNetwork lobe,T* a,T* b,float* out,bool covered){
    int lane=threadIdx.x&31;int outputs=lobe.stride-((lobe.latent&&!lobe.learned_alpha)?1:0);
    for(int j=lane;j<lobe.inputs;j+=32){
        if constexpr(std::is_same_v<T,float>)a[j]=(a[j]-lobe.normalization[j])/lobe.normalization[lobe.inputs+j];
        else a[j]=(a[j]-lobe.half_normalization[j])/lobe.half_normalization[lobe.inputs+j];
    }
    __syncwarp();dense(lobe.network,0,a,b,lobe.inputs,lobe.network.w,true);
    if(lobe.network.depth==1){dense(lobe.network,1,b,a,lobe.network.w,outputs,false);
        for(int j=lane;j<outputs;j+=32)b[j]=a[j];__syncwarp();}
    else{dense(lobe.network,1,b,a,lobe.network.w,lobe.network.w,true);dense(lobe.network,2,a,b,lobe.network.w,outputs,false);}
    if(lane<lobe.lobes)lobe_decode_component(lobe,b,out,lane);
    if(lane==0)out[lobe.stride-1]=(lobe.latent&&!lobe.learned_alpha)?1.f:(covered?float(sigmoid(b[lobe.stride-1])):0.f);
}

template<class T> __global__ void encode_kernel(Model m,int count,LocalCache local,bool periodic,float crop,T* codes,float* samples,
    LobeNetwork lobe={},float* lobe_codes=nullptr){
    int warp=threadIdx.x/32,lane=threadIdx.x&31,id=blockIdx.x*WARPS+warp;
    if(id>=count)return;
    __shared__ T aa[WARPS][MAX_INPUT],bb[WARPS][MAX_INPUT];
    T *a=aa[warp],*b=bb[warp];const Local& ray=local.ray(id);
    T* code=codes+size_t(id)*m.stride;
    if(!ray.valid){for(int j=lane;j<m.stride;j+=32)code[j]=0;if(lobe_codes)for(int j=lane;j<lobe.stride;j+=32)lobe_codes[size_t(id)*lobe.stride+j]=0;return;}
    if(m.kind==1||m.kind==3){encode_neumip(m,ray,periodic,crop,a,b,code,samples?samples+size_t(id)*14:nullptr);return;}
    if constexpr(std::is_same_v<T,__half>){
        if(m.profiled_inputs){for(int j=lane;j<m.k*m.c;j+=32)a[j]=m.profiled_inputs[size_t(id)*(m.k*m.c+3)+j];}
        else load_spatial_inputs(m,ray,periodic,crop,a,samples?samples+size_t(id)*m.k*m.c:nullptr,b);
    }else load_spatial_inputs(m,ray,periodic,crop,a,samples?samples+size_t(id)*m.k*m.c:nullptr,b);
    float3 view=fnorm(ray.view);
    for(int j=lane;j<m.angular;j+=32){float v=m.profiled_inputs?float(m.profiled_inputs[size_t(id)*(m.k*m.c+3)+m.k*m.c+j]):angle_value(view,j,m.bands);a[m.k*(m.c+(m.kind==8?0:1))+j]=v;code[m.z+1+j]=v;}
    __syncwarp();
    if(lobe_codes&&!lobe.latent){
        bool any=false;for(int p=0;p<m.k;++p)any=any||a[m.k*m.c+p]>T(0.f);
        __syncwarp();
        float* out=lobe_codes+size_t(id)*lobe.stride;
        encode_lobe(lobe,a,b,out,any);
        return;
    }
    if(m.kind==6){
        // Keep bottom-to-top feature slots, but accumulate visibility along
        // the actual ray's near-to-far order. All gating stays in FP32.
        if(lane==0){T transmittance=T(1.f);
            for(int step=0;step<m.k;++step){int p=ray.d.z<0?m.k-1-step:step;
                T alpha=a[m.k*m.c+p]>T(0.f)?sigmoid(a[p*4+3]):T(0.f);
                T weight=transmittance*alpha;
                for(int channel=0;channel<3;++channel)a[p*4+channel]*=weight;
                a[p*4+3]=weight;transmittance*=T(1.f)-alpha;
                if(samples)for(int channel=0;channel<4;++channel)samples[size_t(id)*m.k*4+p*4+channel]=a[p*4+channel];
            }
        }
        __syncwarp();
    }
    bool any=m.kind==8;if(m.kind!=8)for(int p=0;p<m.k;++p)any=any||a[m.k*m.c+p]>T(0.f);
    int input=m.k*(m.c+(m.kind==8?0:1))+m.angular,outputs=m.z+(m.kind==8?0:1);
    dense(m,0,a,b,input,m.w,true);
    if((m.kind==4||m.kind==8)&&m.depth==1){dense(m,1,b,a,m.w,outputs,false);for(int j=lane;j<outputs;j+=32)b[j]=a[j];__syncwarp();}
    else{dense(m,1,b,a,m.w,m.w,true);dense(m,2,a,b,m.w,outputs,false);}
    for(int j=lane;j<m.z;j+=32)code[j]=b[j];
    if(lane==0)code[m.z]=m.kind==8?T(1.f):(any?sigmoid(b[m.z]):T(0.f));
    __syncwarp();
    if(lobe_codes&&lobe.latent){
        for(int j=lane;j<lobe.inputs;j+=32)a[j]=j<m.z?code[j]:code[m.z+1+j-m.z];
        __syncwarp();encode_lobe(lobe,a,b,lobe_codes+size_t(id)*lobe.stride,true);return;
    }
    if(m.frames){
        Local top=ray;float dz=fabsf(ray.d.z)>m.epsilon?ray.d.z:1.f,t=(m.bounds[5]-ray.o.z)/dz;
        top.o=make_float3(ray.o.x+t*ray.d.x,ray.o.y+t*ray.d.y,ray.o.z+t*ray.d.z);top.d=make_float3(0,0,-1);
        load_spatial_inputs(m,top,periodic,crop,a);
        __syncwarp();dense(m,6,a,b,m.k*m.c,6*m.frames,false);
        for(int f=0;f<m.frames;++f){
            if constexpr(std::is_same_v<T,float>){
                float3 n=make_float3(b[f*3],b[f*3+1],b[f*3+2]+1),tangent=make_float3(b[3*m.frames+f*3]+1,b[3*m.frames+f*3+1],b[3*m.frames+f*3+2]);
                float ns=1/fmaxf(sqrtf(fdot(n,n)),1e-6f),ts=1/fmaxf(sqrtf(fdot(tangent,tangent)),1e-6f);
                n=make_float3(n.x*ns,n.y*ns,n.z*ns);tangent=make_float3(tangent.x*ts,tangent.y*ts,tangent.z*ts);
                float3 axes[3]={tangent,fcross(n,tangent),n};if(lane<9)code[m.z+4+f*9+lane]=component(axes[lane/3],lane%3);
            }
            else{
                Half3 hn=hnormalize({b[f*3],b[f*3+1],b[f*3+2]+T(1.f)}),ht=hnormalize({b[3*m.frames+f*3]+T(1.f),b[3*m.frames+f*3+1],b[3*m.frames+f*3+2]});
                Half3 ha[3]={ht,hcross(hn,ht),hn};if(lane<9)code[m.z+4+f*9+lane]=hcomponent(ha[lane/3],lane%3);
            }
        }
    }
}

