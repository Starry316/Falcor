import Scene.Scene;
import Rendering.Lights.LightHelpers;
#include "NeuralTypes.slangh"
#include "NeuralDelta.slangh"

StructuredBuffer<NeuralFrame> gFrames;
Texture2D<float4> gPosW;
RWStructuredBuffer<float4> gLocalDirections;
StructuredBuffer<float4> gAppearanceRGBA;
RWStructuredBuffer<float4> gReferenceDirect;
cbuffer gDeltaCB
{
    uint2 gFrameDim;
    uint gLightIndex;
    float gSubstrate;
};

bool deltaQuery(uint2 pixel,uint id,out float3 localWi,out float3 Li)
{
    localWi=0;Li=0;
    NeuralFrame frame=gFrames[id];
    if(frame.normal.w<=0||gLightIndex>=gScene.getLightCount()) return false;
    LightData light=gScene.getLight(gLightIndex);
    if(light.type!=LightType::Point && light.type!=LightType::Directional) return false;
    AnalyticLightSample ls=neuralSampleDelta(gPosW[pixel].xyz,light);
    float3 n=normalize(frame.normal.xyz),t=normalize(frame.tangent.xyz);
    float3 b=normalize(cross(n,t))*frame.tangent.w;
    localWi=float3(dot(ls.dir,t),dot(ls.dir,b),dot(ls.dir,n));Li=ls.Li;
    return localWi.z>0;
}

[numthreads(8,8,1)]
void generate(uint3 tid:SV_DispatchThreadID)
{
    if(any(tid.xy>=gFrameDim))return;
    uint id=tid.y*gFrameDim.x+tid.x;float3 wi,Li;
    // Store the completed query. CUDA uses xyz; w retains validity for captures.
    bool valid = deltaQuery(tid.xy,id,wi,Li);
    if(!valid) wi=float3(0);
    gLocalDirections[id]=float4(wi,valid?1.f:0.f);
}

[numthreads(8,8,1)]
void accumulate(uint3 tid:SV_DispatchThreadID)
{
    if(any(tid.xy>=gFrameDim))return;
    uint id=tid.y*gFrameDim.x+tid.x;float3 wi,Li;
    if(deltaQuery(tid.xy,id,wi,Li))
    {
        float4 appearance=gAppearanceRGBA[id];
        float3 response=appearance.rgb+(1.f-saturate(appearance.w))*max(gSubstrate,0.f)*wi.z;
        gReferenceDirect[id]+=float4(response*Li,0);
    }
}
