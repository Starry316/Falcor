cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
}
RWTexture2D<float4> gOutputColor;
Buffer<float> gInputColor;

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    float4 color = gOutputColor[pixel];
    if (color.w > 0.5f) {
        float3 btfVal = float3(gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
                        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
                        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2]);

        gOutputColor[pixel].xyz *= btfVal;
    }

}
