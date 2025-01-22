
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
import Rendering.Lights.EnvMapSampler;
import Scene.Lights.EnvMap;
import Utils.Sampling.SampleGenerator;
import Utils.Math.ShadingFrame;
import BTFUtils;
import Scene.Scene;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    EnvMapSampler envMapSampler;
    EnvMap envMap;
    uint gFrameCount;
}
SamplerState gSampler;
RWTexture2D<float4> gLightSampleMap;
Texture2D<float4> gAngleMap;
Texture2D<float4> gNMap;
Texture2D<float4> gTMap;

RWBuffer<int> gNNInput;
Buffer<int> cudaVaildBuffer;
Buffer<int> gSelectBuffer;

bool generateEnvMapSample(inout SampleGenerator sg, inout float3 Li, inout float3 dir)
{
    EnvMapSample lightSample;
    envMapSampler.sampleLocal(sampleNext2D(sg), lightSample);
    dir = envMap.toWorld(lightSample.dir);
    Li = lightSample.pdf > 0.f ? envMap.eval(dir) / lightSample.pdf : float3(0);
    return true;
}

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{

    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;

    uint2 pixel = dispatchThreadId.xy;
    if (cudaVaildBuffer[pixel.y * gRenderTargetDim.x + pixel.x] == 0)
        return;

    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);
    ShadingFrame sf = ShadingFrame(gNMap[pixel].xyz, gTMap[pixel]);
    float3 wo = gAngleMap[pixel].xyz;
    float3 wi = float3(1, 0, 0);

    float3 Li = 0;
    // gLightSampleMap[pixel].xyz = gScene.envMap.eval(pixel / gRenderTargetDim.x);
    // gLightSampleMap[pixel].xyz = envMap.evalTexel(pixel);
    generateEnvMapSample(sg, Li, wi);
    wi = sf.toLocal(wi);
    float NdotL = wi.z > 0 ? wi.z : 0;
    uint2 newAngle = packAngularInput(wi, wo);
    gNNInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0] = newAngle.x;
    gNNInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = newAngle.y;
    gLightSampleMap[pixel].xyz = Li * NdotL;
    // if (generateEnvMapSample(sg, Li, wi)) {
    //
    //
    //     uint2 newAngle = packAngularInput(wi, wo);
    //     gNNInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0] = newAngle.x;
    //     gNNInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = newAngle.y;
    //     //


    // }

    // else
    //     gLightSampleMap[pixel] = float4(1, 0, 0, 0);

}
