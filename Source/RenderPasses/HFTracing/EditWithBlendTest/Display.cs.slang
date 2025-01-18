import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
}
RWTexture2D<float4> gOutputColor;



Buffer<float> gInputColor;
Buffer<int> cudaVaildBuffer;
Buffer<int> gSelectBuffer;

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
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]
        //1.0
    );

    if (any(isnan(btfVal)) || any(isinf(btfVal)))
        btfVal = float4(0);

    // print("Li", uvWoyz[pixel]);
    // print("Li", gOutputColor[pixel]);
    // print("btf", btfVal);

    //float2 t0 = float2(0.3f, 0.5f);
    //float2 t1 = float2(2.1f, 4.7f);
    //float4 test_hash = float4(rnd21(t0), rnd21(t1), 0, 1);
    //float4 test_hash = float4(floor(rnd21(t0) * 2048), floor(rnd21(t1) * 2048), 0, 1);


    gOutputColor[pixel] *= btfVal;
    //gOutputColor[pixel] *= btfVal * (1 - gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] + 3]);

    //if(gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] == 0)
    //{
    //    gOutputColor[pixel] = 1;
    //}
}
