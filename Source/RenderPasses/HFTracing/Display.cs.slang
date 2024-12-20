import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    bool showCUDA;
}
RWTexture2D<float4> gOutputColor;
Texture2D<float4> uvWoyz;
Buffer<float> gInputColor;
Buffer<int> cudaVaildBuffer;
[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    // printSetPixel(pixel);

    if (cudaVaildBuffer[pixel.y * gRenderTargetDim.x + pixel.x] == 0)
        return;
    float4 btfVal = float4(
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
        1.0
    );
    // print("Li", uvWoyz[pixel]);
    // if (showCUDA) gOutputColor[pixel].xyz = uvWoyz[pixel].xyz * gOutputColor[pixel].xyz;
    // else gOutputColor[pixel].xyz = uvWoyz[pixel].xyz * btfVal.xyz;
    gOutputColor[pixel] *= btfVal;
}
