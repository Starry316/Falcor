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
    bool gMLPDebug;
}
SamplerState gSampler;
//
RWTexture2D<float4> gOutputColor;
Texture2D<uint4> btfInput;
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
    // printSetPixel(pixel);
    // float2 uv1 = (float2)pixel / (float2)gRenderTargetDim;
    // float2 h1 = float2(0.951858, 0.01944);
    // float2 d1 = float2(0.875318, 0.58938);

    // gOutputColor[pixel].xyz = nbtf.evalDebug_relu(h1, d1, uv1, 0.000001, 0.00001, gApplySyn);
    // bindInput(pixel, uv1, h1, d1, 0.000001, 0.00001);

    int isVaild = 0;
    if (gOutputColor[pixel].w > 0.5f)
    {
        uint4 packed = btfInput[pixel];
        float2 h, d, uv, ddx, ddy;

        unpackInput(packed, h, d, uv, ddx, ddy);
        // print("h", h);
        // print("d", d);

        // h = float2(0.951858, 0.01944);
        // d = float2(0.875318, 0.58938);

        uv *= UV_SCALING;

        ddx = float2(0.0000001);
        ddy = float2(0.0000001);



#ifdef CUDA
        bindInput(pixel, uv, h, d, ddx, ddy);
#endif

#ifdef CUDAFP16
        bindInputFP16(pixel, uv, h, d, ddx, ddy);
#endif

#ifdef CUDAINT8
        bindInputInt8(pixel, uv, h, d, ddx, ddy);
#endif

        // #ifdef SHADER
        if (!gMLPDebug)
            gOutputColor[pixel].xyz = nbtf.evalDebug(h, d, uv, ddx, ddy, gApplySyn);
        else
            gOutputColor[pixel].xyz = nbtf.evalDebug_realDebug(h, d, uv, ddx, ddy, gApplySyn).xyz;
        // #endif

        isVaild = 1;
    }
    cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] = isVaild;
    // cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] = 1;
}
