import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
}
RWTexture2D<float4> gOutputColor;

Buffer<float> gInputColor;
Buffer<int> cudaVaildBuffer;
[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);

    if (cudaVaildBuffer[pixel.y * gRenderTargetDim.x + pixel.x] == 0)
        return;
    float4 btfVal = float4(
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
        1.0
    );

    if (any(isnan(btfVal)) || any(isinf(btfVal)))
        btfVal = float4(0);
#ifdef HDR_COMPRESSED
    btfVal = exp(btfVal) - 1;
#endif
    print("btf", btfVal);
    gOutputColor[pixel] *= btfVal;
}
