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
RWBuffer<float> gInputColor;
#define UV_SCALE gCurvatureParas.z
#define MIP_SCALE gCurvatureParas.w

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    // printSetPixel(pixel);
    if (cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] == 1)
    {
        uint4 packed;
        if (gApplySyn)
            packed = uint4(
                btfInput[5 * (pixel.y * gRenderTargetDim.x + pixel.x)],
                btfInput[5 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
                btfInput[5 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
                btfInput[5 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]
            );
        else
            packed = uint4(
                btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)],
                btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
                btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
                btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]
            );
        float2 h, d, uv, ddx, ddy;
        unpackInput(packed, h, d, uv, ddx, ddy);
        uv = uv.yx * UV_SCALE;
        float4 btfVal = float4(1.0f);
        if (gDebugMLP)
            btfVal.xyz = nbtf.evalDebug(h, d, uv, float2(0.0001f), float2(0.0001f), gApplySyn);
        else
            btfVal = nbtf.evalDebugFP32(h, d, uv, ddx, ddy, gApplySyn);
        // btfVal = nbtf.evalDebugFP32(h, d, uv, 0.1 * MIP_SCALE, gApplySyn);

        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0] = btfVal.x;
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = btfVal.y;
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = btfVal.z;
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = btfVal.w;

        // gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0] = frac(uv.x);
        // gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = frac(uv.y);
        // gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] =0;
        // gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] =0;
    }
}

// btfVal = nbtf.evalDebugFP32(h, d, uv, ddx * MIP_SCALE, ddy * MIP_SCALE, gApplySyn);
