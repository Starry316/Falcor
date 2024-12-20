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


    if(cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)]>0)
    {
        float4 btfVal = float4(
            gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
            gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
            gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
            1.0
        );

        // print("Li", uvWoyz[pixel]);
        // print("ref", gOutputColor[pixel]);
        // print("BTF", btfVal);
        // print("dff", btfVal - gOutputColor[pixel]);
        // if (showCUDA) gOutputColor[pixel] = abs(btfVal - gOutputColor[pixel]);
        // if (showCUDA) gOutputColor[pixel] = btfVal;
        if (showCUDA) gOutputColor[pixel].xyz = uvWoyz[pixel].xyz * gOutputColor[pixel].xyz;

        else gOutputColor[pixel].xyz = uvWoyz[pixel].xyz * btfVal.xyz;
        // uvWoyz[pixel] *
        // print("Li * BTF", gOutputColor[pixel]);
        // gOutputColor[pixel] *= float4(1, 0, 0, 0);
    }
}
