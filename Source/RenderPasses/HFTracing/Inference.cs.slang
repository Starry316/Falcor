import Utils.Neural.NBTF;
import BTFUtils;
import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    bool gApplySyn;
    NBTF nbtf;
    NBTF nbtfInt8;
    float4 gCurvatureParas;

}
SamplerState gSampler;
//
RWTexture2D<float4> gOutputColor;
Buffer<uint> btfInput;
Texture2D<float4> uvWoyz;

Texture2D<float4> dfDxy;

RWBuffer<uint> cudaInputUIntBuffer;
RWBuffer<int> cudaInputIntBuffer;
RWBuffer<int> cudaVaildBuffer;
RWBuffer<float> cudaInputBuffer;
#define UV_SCALING gCurvatureParas.z * 10
void bindInputInt8(uint2 pixel, float2 uv, float2 h, float2 d, float2 ddx, float2 ddy)
{
    uint offset = 6 * (pixel.y * gRenderTargetDim.x + pixel.x);
    nbtfInt8.bindInputInt8(cudaInputUIntBuffer, offset, h, d, uv, ddx, ddy, gApplySyn);
}
void bindInput(uint2 pixel, float2 uv, float2 h, float2 d, float2 ddx, float2 ddy)
{
    uint offset = 24 * (pixel.y * gRenderTargetDim.x + pixel.x);
    nbtf.bindInput(cudaInputBuffer, offset, h, d, uv, ddx, ddy, gApplySyn);
}

void bindInputFP16(uint2 pixel, float2 uv, float2 h, float2 d, float2 ddx, float2 ddy)
{
    uint offset = 16 * (pixel.y * gRenderTargetDim.x + pixel.x);

    // nbtf.bindInputFP16(cudaInputUIntBuffer, offset, h, d, uv, ddx, ddy, gApplySyn);
}

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;

    if (cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] == 1)
    {
        uint4 packed = uint4(btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)],
                             btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
                             btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
                             btfInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]);
        float2 h, d, uv, ddx, ddy;
        unpackInput(packed, h, d, uv, ddx, ddy);
        uv *= UV_SCALING;
// #ifdef CUDA
//         bindInput(pixel, uv, h, d, ddx, ddy);
// #endif

// #ifdef CUDAFP16
//         bindInputFP16(pixel, uv, h, d, ddx, ddy);
// #endif

// #ifdef CUDAINT8
//         bindInputInt8(pixel, uv, h, d, ddx, ddy);
// #endif

// #ifdef SHADER

// #endif
        gOutputColor[pixel].xyz *= nbtf.evalDebug(h, d, uv, ddx, ddy, gApplySyn);
    }
}
