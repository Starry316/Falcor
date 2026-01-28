#include "Utils/Math/MathConstants.slangh"
#include "utils.slangh"
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import Scene.Intersection;
import Utils.Geometry.GeometryHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGenerator;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    uint gShowBillboardID;
    uint gBillboardCount;
    // uint gDebugViewID;
    bool gDebugMode;
    bool gShowDiff;
    float gInvalidThreshold;
};
RWTexture2D<float4> gOutput;
Texture2D<float4> gRtOut;
Texture2DArray<float4> gBillboards;
Texture2D<float4> gReference;
StructuredBuffer<float4x4> gViewProjBuffer;
StructuredBuffer<float4x4> gViewProjInvBuffer;
StructuredBuffer<float3> gBillboardNormalBuffer;

SamplerState gSampler;

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gFrameDim))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    // printSetPixel(pixel);
    if (gShowDiff)
    {
        float4 rtColor = gRtOut.Load(int3(pixel, 0));
        float4 refColor = gReference.Load(int3(pixel, 0));
        gOutput[pixel] = abs(rtColor - refColor);
        return;
    }
    gOutput[pixel] = gRtOut.Load(int3(pixel, 0));
}
