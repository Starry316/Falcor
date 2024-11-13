cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    float gDisplayMipLevel;

}
SamplerState gSampler;

RWTexture2D<float4> gOutputColor;
Texture2D<float4> gOutputNormalMap;
Texture2D<float4> gOutputTangentMap;


[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;

    const uint2 pixelPos = dispatchThreadId.xy;
    float2 uv = (float2(pixelPos)) / gRenderTargetDim;
    // gOutputColor[pixelPos.xy] = float4(1, 0, 0, 0);
    if (uv.x > 0.5)
    {
        uv.x = 2 * uv.x - 1;
        gOutputColor[pixelPos.xy] = 0.5 * (gOutputNormalMap.SampleLevel(gSampler, uv, gDisplayMipLevel) + 1);
        // gOutputColor[pixelPos.xy] = float4(1,0,0,0);
        return;
    }
    else {
        uv.x = 2 * uv.x;
        gOutputColor[pixelPos.xy] = 0.5 * (gOutputTangentMap.SampleLevel(gSampler, uv, gDisplayMipLevel) + 1);
        return;
    }



}
