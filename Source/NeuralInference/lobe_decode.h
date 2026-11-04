#pragma once
__device__ float lobe_exp(float v){return expf(v);}
__device__ __half lobe_exp(__half v){return hexp(v);}
template<class T,class Output> __device__ __forceinline__ void lobe_decode_component(const LobeNetwork& l,const T* b,Output* out,int component_index){
    int j=component_index*7;
    if constexpr(std::is_same_v<T,float>){float3 axis=make_float3(b[j],b[j+1],b[j+2]);float length=sqrtf(fdot(axis,axis));
        if(length<1e-6f)axis=make_float3(0,0,1);else{axis.x/=length;axis.y/=length;axis.z/=length;}
        out[j]=axis.x;out[j+1]=axis.y;out[j+2]=axis.z;
    }else{Half3 axis=hnormalize({b[j],b[j+1],b[j+2]});
        if(__hmax(__habs(b[j]),__hmax(__habs(b[j+1]),__habs(b[j+2])))<T(1e-6f))axis={T(0.f),T(0.f),T(1.f)};
        out[j]=float(axis.x);out[j+1]=float(axis.y);out[j+2]=float(axis.z);}
    if(!l.parameterization){
        out[j+3]=float(T(l.min_r)+(T(1.f)-T(l.min_r))*sigmoid(b[j+3]));
        #pragma unroll
        for(int c=0;c<3;++c)out[j+4+c]=float(softplus(b[j+4+c])*T(l.scale[c]));
    }else{
        out[j+3]=float(lobe_exp(T(l.log_min_r)*(T(1.f)-sigmoid(b[j+3]))));
        #pragma unroll
        for(int c=0;c<3;++c){
            T largest=T(0.f);for(int k=1;k<l.lobes;++k)if(b[k*7+4+c]>largest)largest=b[k*7+4+c];
            T sum=lobe_exp(-largest);for(int k=1;k<l.lobes;++k)sum+=lobe_exp(b[k*7+4+c]-largest);
            T logit=component_index==l.lobes-1?T(0.f):b[(component_index+1)*7+4+c];
            T energy=softplus(b[4+c])*T(l.scale[c])*T(float(l.lobes));
            out[j+4+c]=float(energy*(lobe_exp(logit-largest)/sum));
        }
    }
}

