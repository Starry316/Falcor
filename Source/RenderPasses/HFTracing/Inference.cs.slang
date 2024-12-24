import Utils.Neural.NBTF;
import BTFUtils;
import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    bool gApplySyn;
    NBTF nbtf;
    float4 gCurvatureParas;
    bool gDebugMLP;
}
RWTexture2D<float4> gOutputColor;
Buffer<uint> btfInput;
RWBuffer<int> cudaVaildBuffer;

#define UV_SCALE gCurvatureParas.z

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    if (cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] == 1)
    {
        uint4 packed = uint4(btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)],
                             btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
                             btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
                             btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]);
        float2 h, d, uv, ddx, ddy;
        unpackInput(packed, h, d, uv, ddx, ddy);
        uv = uv.yx * UV_SCALE;
        if (gDebugMLP)
            gOutputColor[pixel].xyz *= nbtf.evalDebug(h, d, uv, ddx, ddy, gApplySyn).xyz;
        else
            gOutputColor[pixel].xyz *= nbtf.evalDebugFP32(h, d, uv, ddx, ddy, gApplySyn).xyz;


        print("uv", uv);
        print("h", h);
        print("d", d);
    }
}
