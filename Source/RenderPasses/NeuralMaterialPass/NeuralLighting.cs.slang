import Scene.Scene;
import Rendering.Lights.LightHelpers;
#include "NeuralLighting.slangh"
#include "NeuralDelta.slangh"

StructuredBuffer<NeuralRay> gRays;
StructuredBuffer<NeuralFrame> gFrames;
#ifndef NEURAL_PACKED_LOBES
#define NEURAL_PACKED_LOBES 0
#endif
#if NEURAL_PACKED_LOBES
StructuredBuffer<uint> gLobes;
float loadLobe(uint index)
{
    uint pair=gLobes[index>>1];
    return f16tof32((pair>>((index&1u)*16u))&0xffffu);
}
#else
StructuredBuffer<float> gLobes;
float loadLobe(uint index) { return gLobes[index]; }
#endif
StructuredBuffer<PolygonLight> gLights;
StructuredBuffer<float> gLtcLut;
StructuredBuffer<float> gHorizonLut;
StructuredBuffer<float4> gReferenceDirect;
StructuredBuffer<float4> gAppearanceRGBA;
Texture2D<float4> gPosW;
TextureCube<float4> gEnvPrefilter;
Texture2D<float4> gAreaSources[8];
Texture3D<float4> gAreaPrefilters[8];
SamplerState gLinearSampler;
RWTexture2D<float4> gForeground;
// RGB background or a display override; W: -1 override, otherwise bit 0 is
// opaque environment background and bit 1 marks a neural receiver.
RWTexture2D<float4> gDisplay;

cbuffer gLightingCB
{
    uint2 gFrameDim;
    uint gLobeCount;
    uint gLobeStride;
    uint gLightCount;
    uint gEnvironmentEnabled;
    uint gDeltaEnabled;
    uint gAreaEnabled;
    uint gBackgroundEnabled;
    uint gHorizonOcclusion;
    uint gAreaMode; // 0 LTC + filtered texture; 1 same-lobe QMC; 2 original appearance
    uint gReferenceSamples;
    uint gDebugView; // 0 radiance,1 LOD,2 lobe0 axis,3 roughness,4 alpha,5 frame normal
    uint gEnvMipCount;
    uint gHasEnvironment;
    uint gMipLevels;
    uint gDeltaReference;
    uint gEnvironmentReference;
    float gEnvMinR;
    float gSubstrate;
    float gBackgroundGain;
    float gMaterialOpacity;
};

float3 localToWorld(NeuralFrame frame,float3 direction)
{
    float3 n=normalize(frame.normal.xyz),t=normalize(frame.tangent.xyz);
    float3 b=normalize(cross(n,t))*frame.tangent.w;
    return normalize(t*direction.x+b*direction.y+n*direction.z);
}

float3 worldToLocal(NeuralFrame frame,float3 direction)
{
    float3 n=normalize(frame.normal.xyz),t=normalize(frame.tangent.xyz);
    float3 b=normalize(cross(n,t))*frame.tangent.w;
    return float3(dot(direction,t),dot(direction,b),dot(direction,n));
}

float3 lobeAxis(uint first)
{
    float3 axis=float3(loadLobe(first),loadLobe(first+1),loadLobe(first+2));
    float norm2=dot(axis,axis);
    return norm2>1e-20f?axis*rsqrt(norm2):float3(0,0,1);
}
float3 lobeEnergy(uint first) { return float3(loadLobe(first+4),loadLobe(first+5),loadLobe(first+6)); }

float horizonVisibility(float3 axis,float roughness)
{
    float at=saturate(roughness)*256.f; uint lo=min((uint)at,255u);
    float beta=lerp(gHorizonLut[lo],gHorizonLut[lo+1],at-float(lo));
    float theta=acos(clamp(normalize(axis).z,-1.f,1.f));
    float x=saturate((theta+beta-.5f*kNeuralPi)/max(2.f*beta,1e-12f));
    if(beta==0&&axis.z==0) x=.5f;
    return 1.f-.95f*x*x*(3.f-2.f*x);
}

float3 filteredEnvironment(float3 worldDirection,float roughness)
{
    float lod=log(clamp(roughness,gEnvMinR,1.f)/gEnvMinR)*float(max(gEnvMipCount,1u)-1u)/max(log(1.f/gEnvMinR),1e-10f);
    return gEnvPrefilter.SampleLevel(gLinearSampler,gScene.envMap.toLocal(worldDirection),lod).rgb*gScene.envMap.getIntensity();
}

float ltcScale(float roughness)
{
    float at=saturate(log(clamp(roughness,.08f,1.f)/.08f)/log(12.5f))*511.f;
    uint lo=min((uint)at,510u);
    return exp(lerp(gLtcLut[lo],gLtcLut[lo+1],at-float(lo)));
}

float3 transformLtc(float3 vector,float3 t,float3 b,float3 axis,float scale)
{
    return float3(dot(vector,t)/scale,dot(vector,b)/scale,dot(vector,axis));
}

float3 sourceEmission(uint lightIndex,PolygonLight light,float2 uv)
{
    uint kind=light.params.y;
    // Built-in sources are the same 12x12 scene-linear texels as the Python
    // reference. Arbitrary loaded images retain their own source resolution.
    if(kind==1||kind==2) uv=(min(floor(saturate(uv)*12.f),11.f)+.5f)/12.f;
    float3 color=kind==3?gAreaSources[lightIndex].SampleLevel(gLinearSampler,saturate(uv),0).rgb:neuralEmission(uv,kind);
    return max(color,0.f)*light.emission.rgb;
}

struct NeuralLtcFrame
{
    float3 normal;
    float3 axis;
    float3 t;
    float3 b;
    float scale;
};

NeuralLtcFrame prepareLtcFrame(NeuralFrame frame,float3 axisLocal,float roughness)
{
    NeuralLtcFrame result;
    result.normal=normalize(frame.normal.xyz);
    result.axis=localToWorld(frame,axisLocal);
    float3 t,b;neuralBasis(result.axis,t,b);result.t=t;result.b=b;
    result.scale=roughness>=1.f?1.f:ltcScale(roughness);
    return result;
}

float3 ltcLight(uint lightIndex,PolygonLight light,float3 position,NeuralLtcFrame frame)
{
    float3 lightNormal=cross(light.fullU.xyz,light.fullV.xyz);
    if(dot(position-light.center.xyz,lightNormal)<=0) return 0;
    float3 axis=frame.axis,t=frame.t,b=frame.b;float scale=frame.scale;
    float integral=neuralStreamingIntegral(light,position,frame.normal,t/scale,b/scale,axis);
    float3 color=light.emission.rgb;
    if(light.params.y!=0&&integral>0)
    {
        float3 center=transformLtc(light.center.xyz-position,t,b,axis,scale);
        float3 u=transformLtc(light.fullU.xyz,t,b,axis,scale),v=transformLtc(light.fullV.xyz,t,b,axis,scale);
        float3 normal=cross(u,v); float area2=max(dot(normal,normal),1e-30f);
        float2 uv=float2(dot(cross(-center,v),normal),dot(cross(u,-center),normal))/area2+.5f;
        float distance2=dot(center,normal)*dot(center,normal)/area2;
        float area=sqrt(area2)*neuralPolygonArea(min(light.params.x,16u),light.uvVertices);
        float sigma=sqrt(distance2/max(2.f*area,1e-30f));
        float3 query=float3((clamp(uv,-1.f,2.f)+1.f)/3.f,saturate((log2(max(sigma,exp2(-10.f)))+10.f)/13.f));
        uint width,height,depth; gAreaPrefilters[lightIndex].GetDimensions(width,height,depth);
        float3 dimensions=float3(width,height,depth);
        // Match align_corners=True in the independent Python oracle.
        query=(query*(dimensions-1.f)+.5f)/dimensions;
        color*=gAreaPrefilters[lightIndex].SampleLevel(gLinearSampler,query,0).rgb;
    }
    return color*integral;
}

float3 areaReference(uint lightIndex,PolygonLight light,float3 position,NeuralFrame frame,uint base,out float3 diffuseResponse)
{
    diffuseResponse=0;
    float3 n=cross(light.fullU.xyz,light.fullV.xyz); float area=length(n);
    if(area<1e-15f||dot(position-light.center.xyz,n)<=0) return 0;
    n/=area; float3 total=0; uint count=max(1u,gReferenceSamples);
    for(uint i=0;i<count;++i)
    {
        float2 uv=float2((float(i)+.5f)/count,neuralRadical(i,0x61c88647u));
        if(!neuralInside(uv,min(light.params.x,16u),light.uvVertices)) continue;
        float3 delta=light.center.xyz+(uv.x-.5f)*light.fullU.xyz+(uv.y-.5f)*light.fullV.xyz-position;
        float d2=max(dot(delta,delta),1e-20f); float3 wi=delta*rsqrt(d2);
        float3 localWi=worldToLocal(frame,wi);
        if(localWi.z<=0) continue;
        float3 response=0;
        for(uint k=0;k<gLobeCount;++k)
        {
            uint first=base+7*k;
            response+=lobeEnergy(first)*neuralKernel(dot(lobeAxis(first),localWi),loadLobe(first+3));
        }
        float3 radianceWeight=sourceEmission(lightIndex,light,uv)*max(-dot(wi,n),0.f)*area/d2;
        total+=response*radianceWeight;
        if(gSubstrate>0) diffuseResponse+=localWi.z*radianceWeight;
    }
    diffuseResponse/=float(count);
    return total/float(count);
}

bool visibleEmitter(uint2 pixel,float3 receiverPosition,bool receiverHit,out float3 radiance)
{
    radiance=0;
    Ray ray=gScene.camera.computeRayPinhole(pixel,gFrameDim);
    float nearest=receiverHit?length(receiverPosition-ray.origin):1e30f;
    bool hit=false;
    for(uint i=0;i<min(gLightCount,8u);++i)
    {
        PolygonLight light=gLights[i];
        if(light.params.z==0||light.params.x<3) continue;
        float3 n=cross(light.fullU.xyz,light.fullV.xyz);
        float denominator=dot(ray.dir,n);
        if(abs(denominator)<1e-12f) continue;
        float distance=dot(light.center.xyz-ray.origin,n)/denominator;
        if(distance<=0||distance>=nearest) continue;
        float3 offset=ray.origin+distance*ray.dir-light.center.xyz;
        float norm2=max(dot(n,n),1e-30f);
        float2 uv=float2(dot(cross(offset,light.fullV.xyz),n),dot(cross(light.fullU.xyz,offset),n))/norm2+.5f;
        if(!neuralInside(uv,min(light.params.x,16u),light.uvVertices)) continue;
        nearest=distance; hit=true;
        radiance=denominator<0?sourceEmission(i,light,uv):float3(0);
    }
    return hit;
}

[numthreads(8,8,1)]
void main(uint3 tid:SV_DispatchThreadID)
{
    if(any(tid.xy>=gFrameDim)) return;
    uint id=tid.y*gFrameDim.x+tid.x,base=id*gLobeStride;
    NeuralFrame frame=gFrames[id];
    float4 position=gPosW[tid.xy];
    float3 background=0;
    if(gBackgroundEnabled!=0&&gHasEnvironment!=0)
        background=gBackgroundGain*gScene.envMap.eval(gScene.camera.computeRayPinhole(tid.xy,gFrameDim).dir);
    float3 rgb=0; float alpha=0;
    if(frame.normal.w>0)
    {
        alpha=saturate(loadLobe(base+7*gLobeCount));
        float3 environment=0,area=0,delta=0,areaSubstrate=0;
        for(uint k=0;k<gLobeCount;++k)
        {
            uint first=base+7*k;
            float3 axis=lobeAxis(first); float roughness=loadLobe(first+3);
            if(gEnvironmentEnabled!=0&&gHasEnvironment!=0&&gEnvironmentReference==0)
            {
                float visibility=gHorizonOcclusion!=0?horizonVisibility(axis,roughness):1.f;
                environment+=lobeEnergy(first)*filteredEnvironment(localToWorld(frame,axis),roughness)*visibility;
            }
            if(gAreaEnabled!=0&&gAreaMode==0)
            {
                NeuralLtcFrame ltcFrame=prepareLtcFrame(frame,axis,roughness);
                for(uint i=0;i<min(gLightCount,8u);++i)
                    if(gLights[i].params.z!=0)
                        area+=lobeEnergy(first)*ltcLight(i,gLights[i],position.xyz,ltcFrame);
            }
        }
        if(gSubstrate>0&&gAreaEnabled!=0&&gAreaMode==0)
        {
            // r=1 is exactly cosine/pi; prepare its identity LTC once.
            NeuralLtcFrame ltcFrame=prepareLtcFrame(frame,float3(0,0,1),1.f);
            for(uint i=0;i<min(gLightCount,8u);++i)
                if(gLights[i].params.z!=0)
                    // Existing substrate direct response is cos(theta), so
                    // cosine/pi LTC integration needs this factor of pi.
                    areaSubstrate+=kNeuralPi*ltcLight(i,gLights[i],position.xyz,ltcFrame);
        }
        if(gAreaEnabled!=0&&gAreaMode==1)
            for(uint i=0;i<min(gLightCount,8u);++i)
                if(gLights[i].params.z!=0)
                {
                    float3 diffuseResponse;
                    area+=areaReference(i,gLights[i],position.xyz,frame,base,diffuseResponse);
                    areaSubstrate+=diffuseResponse;
                }
        if(gDeltaEnabled!=0&&gDeltaReference==0)
        {
            for(uint i=0;i<gScene.getLightCount();++i)
            {
                LightData light=gScene.getLight(i);
                if(light.type!=LightType::Point && light.type!=LightType::Directional) continue;
                AnalyticLightSample ls=neuralSampleDelta(position.xyz,light);
                float3 wi=worldToLocal(frame,ls.dir);
                if(wi.z<=0) continue;
                for(uint k=0;k<gLobeCount;++k)
                {
                    uint first=base+7*k;
                    delta+=lobeEnergy(first)*neuralKernel(dot(lobeAxis(first),wi),loadLobe(first+3))*ls.Li;
                }
                if(gSubstrate>0) rgb+=(1.f-alpha)*gSubstrate*wi.z*ls.Li;
            }
        }
        rgb+=alpha*(environment+area+delta);
        rgb+=(1.f-alpha)*max(gSubstrate,0.f)*areaSubstrate;
        if(gDeltaEnabled!=0&&gDeltaReference!=0) rgb+=gReferenceDirect[id].rgb;
        if(gSubstrate>0&&gEnvironmentEnabled!=0&&gHasEnvironment!=0&&gEnvironmentReference==0)
            rgb+=(1.f-alpha)*gSubstrate*filteredEnvironment(normalize(frame.normal.xyz),1.f);
        // Coverage is material data, separate from the accumulator's sample
        // count. The source network's light-independent alpha survives a hold.
        if((gEnvironmentEnabled!=0&&gHasEnvironment!=0&&gEnvironmentReference!=0)||(gAreaEnabled!=0&&gAreaMode==2))
            alpha=saturate(gAppearanceRGBA[id].w);
        if(gDebugView==1) {float lod=gRays[id].lod/max(1.f,float(gMipLevels-1));rgb=float3(lod,1.f-lod,.15f);alpha=1;}
        if(gDebugView==2) {rgb=lobeAxis(base)*.5f+.5f;alpha=1;}
        if(gDebugView==3) {rgb=float3(loadLobe(base+3));alpha=1;}
        if(gDebugView==4) {rgb=float3((gSubstrate>0?1.f:alpha)*saturate(gMaterialOpacity));alpha=1;}
        if(gDebugView==5) {rgb=normalize(frame.normal.xyz)*.5f+.5f;alpha=1;}
    }
    else if(position.w>0) alpha=1; // Non-neural foreground remains occluding.
    float displayFlags=float((gBackgroundEnabled!=0&&gHasEnvironment!=0?1u:0u)|(frame.normal.w>0?2u:0u));
    float4 display=float4(background,displayFlags);
    if(gDebugView!=0&&frame.normal.w>0) display=float4(rgb,-1.f);
    if(gAreaEnabled!=0&&gDebugView==0)
    {
        float3 emitter;
        if(visibleEmitter(tid.xy,position.xyz,position.w>0,emitter)) display=float4(emitter,-1.f);
    }
    gForeground[tid.xy]=float4(rgb,alpha);
    gDisplay[tid.xy]=display;
}
