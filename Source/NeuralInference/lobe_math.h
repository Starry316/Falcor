#pragma once
// Entire dot product and activation stay half. No float activation/product buffer.
__device__ __half sigmoid(__half x){
    __half z=hexp(-__habs(x)),one=__float2half_rn(1.f);
    return x>=__float2half_rn(0.f)?one/(one+z):z/(one+z);
}
__device__ float sigmoid(float x){return 1.f/(1.f+expf(-x));}
__device__ __half softplus(__half x){return __hmax(x,__float2half_rn(0.f))+hlog(__float2half_rn(1.f)+hexp(-__habs(x)));}
__device__ float softplus(float x){return x>20?x:log1pf(expf(x));}
// Half vector arithmetic for direction inputs and learned frame intermediates.
struct Half3 {__half x,y,z;};
__device__ Half3 hvec(float3 v){return {__float2half_rn(v.x),__float2half_rn(v.y),__float2half_rn(v.z)};}
__device__ __half hdot(Half3 a,Half3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ Half3 hcross(Half3 a,Half3 b){return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
__device__ Half3 hnormalize(Half3 v){
    __half zero=__float2half_rn(0.f),scale=__hmax(__habs(v.x),__hmax(__habs(v.y),__habs(v.z)));
    if(scale==zero)return {zero,zero,zero};
    Half3 n={v.x/scale,v.y/scale,v.z/scale};__half length=hsqrt(hdot(n,n));
    return {n.x/length,n.y/length,n.z/length};
}
__device__ __half hcomponent(Half3 v,int j){return j==0?v.x:(j==1?v.y:v.z);}
