import Scene.Scene;
import Rendering.Lights.EnvMapSampler;
#include "NeuralLighting.slangh"

StructuredBuffer<NeuralFrame> gFrames;
Texture2D<float4> gPosW;
StructuredBuffer<PolygonLight> gLights;
Texture2D<float4> gAreaSources[8];
SamplerState gLinearSampler;
ParameterBlock<EnvMapSampler> gEnvSampler;
RWStructuredBuffer<float4> gLocalDirections;
RWStructuredBuffer<float4> gRadianceWeights;
StructuredBuffer<float4> gAppearanceRGBA;
// Current frame only: RGB is the sum of decoded HDR contributions, A is
// the actual number of attempted samples (including rejected/zero samples).
RWTexture2D<float4> gFrameSamples;
cbuffer gReferenceCB
{
    uint2 gFrameDim;
    uint gSampleIndex;
    uint gLightCount;
    uint gDomain; // 0 environment, 1 polygons
    uint gSeed;
    float gSubstrate;
};

uint referenceHash(uint x)
{
    x=(x^(x>>16))*0x7feb352du;x=(x^(x>>15))*0x846ca68bu;return x^(x>>16);
}
float referenceRandom(uint pixel,uint sample,uint dimension)
{
    uint x=referenceHash(pixel^gSeed)^referenceHash(sample+0x9e3779b9u)^referenceHash(dimension+0x68bc21ebu);
    return min((float(referenceHash(x)>>8)+.5f)*5.9604644775390625e-8f,.99999994f);
}
float3 toLocal(NeuralFrame f,float3 wi)
{
    float3 n=normalize(f.normal.xyz),t=normalize(f.tangent.xyz),b=normalize(cross(n,t))*f.tangent.w;
    return float3(dot(wi,t),dot(wi,b),dot(wi,n));
}
float3 toWorld(NeuralFrame f,float3 wi)
{
    float3 n=normalize(f.normal.xyz),t=normalize(f.tangent.xyz),b=normalize(cross(n,t))*f.tangent.w;
    return normalize(wi.x*t+wi.y*b+wi.z*n);
}
float3 referenceEmission(uint index,PolygonLight light,float2 uv)
{
    uint kind=light.params.y;
    if(kind==1||kind==2)uv=(min(floor(saturate(uv)*12.f),11.f)+.5f)/12.f;
    float3 value=kind==3?gAreaSources[index].SampleLevel(gLinearSampler,saturate(uv),0).rgb:neuralEmission(uv,kind);
    return max(value,0.f)*light.emission.rgb;
}

[numthreads(8,8,1)]
void generate(uint3 tid:SV_DispatchThreadID)
{
    if(any(tid.xy>=gFrameDim))return;uint id=tid.y*gFrameDim.x+tid.x;
    gLocalDirections[id]=0;gRadianceWeights[id]=0;
    NeuralFrame frame=gFrames[id];if(frame.normal.w<=0)return;
    float r0=referenceRandom(id,gSampleIndex,4*gDomain),r1=referenceRandom(id,gSampleIndex,4*gDomain+1),r2=referenceRandom(id,gSampleIndex,4*gDomain+2);
    float3 wi=0,localWi=0,weight=0;
    if(gDomain==0)
    {
        // Balance environment importance with cosine-hemisphere sampling.
        // The original response already contains its cosine: only divide by PDF.
        float average=gEnvSampler.importanceMap.Load(int3(0,0,gEnvSampler.importanceBaseMip));
        if(!(average>0))return;
        if(r0<.5f)
        {
            EnvMapSample sample;
            bool valid=gEnvSampler.sample(float2(r1,r2),sample);
            if(!valid)return;wi=sample.dir;localWi=toLocal(frame,wi);
        }
        else
        {
            float phi=2.f*kNeuralPi*r2,s=sqrt(r1);
            localWi=float3(s*cos(phi),s*sin(phi),sqrt(1.f-r1));wi=toWorld(frame,localWi);
        }
        if(localWi.z<=0)return;
        float pdf=.5f*gEnvSampler.evalPdf(wi)+.5f*localWi.z/kNeuralPi;
        if(!(pdf>0)||!isfinite(pdf))return;
        weight=gScene.envMap.eval(wi)/pdf;
    }
    else
    {
        if(gLightCount==0)return;
        uint index=min(uint(r0*gLightCount),gLightCount-1);PolygonLight light=gLights[index];
        if(light.params.z==0)return;float2 uv=float2(r1,r2);
        if(!neuralInside(uv,min(light.params.x,16u),light.uvVertices))return;
        float3 delta=light.center.xyz+(uv.x-.5f)*light.fullU.xyz+(uv.y-.5f)*light.fullV.xyz-gPosW[tid.xy].xyz;
        float d2=max(dot(delta,delta),1e-20f);wi=delta*rsqrt(d2);localWi=toLocal(frame,wi);
        if(localWi.z<=0)return;
        float3 areaNormal=cross(light.fullU.xyz,light.fullV.xyz);
        weight=referenceEmission(index,light,uv)*float(gLightCount)*max(-dot(wi,areaNormal),0.f)/d2;
    }
    gLocalDirections[id]=float4(localWi,1);
    gRadianceWeights[id]=float4(weight,1);
}

[numthreads(8,8,1)]
void sumFrameSamples(uint3 tid:SV_DispatchThreadID)
{
    if(any(tid.xy>=gFrameDim))return;uint id=tid.y*gFrameDim.x+tid.x;
    float4 response=gAppearanceRGBA[id];
    float3 value=(response.rgb+(1.f-saturate(response.w))*max(gSubstrate,0.f)*max(gLocalDirections[id].z,0.f))*gRadianceWeights[id].rgb;
    gFrameSamples[tid.xy]+=float4(value,1.f);
}
