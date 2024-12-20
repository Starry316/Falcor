import Utils.Texture.HistogramBlending;
import Utils.Neural.NBTF;
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;

cbuffer ToyCB
{
    float2 iResolution;
    float iGlobalTime;
    HistogramBlendingHFData hfData;
    float gUVScaling;
    NBTF nbtf;
    bool gSynthesis;
    bool gDebugMLP;
    float2 gWo;
    float2 gWi;
};
Buffer<float> gInputColor;
RWBuffer<uint> cudaInputUIntBuffer;
RWBuffer<float> cudaInputBuffer;
RWTexture2D<float4> ouputColor;
SamplerState gSampler;


[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= iResolution))
        return;

    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    float2 uv = (float2(pixel)) / iResolution.y;
    float2 ddx = 0.0000001;
    float2 ddy = 0.0000001;

    float3 wi = sample_hemisphere(gWi);
    float3 wo = sample_hemisphere(gWo);
    float2 h = float2(0.951858, 0.01944);
    float2 d = float2(0.875318, 0.58938);
    if (!gDebugMLP)
    ouputColor[pixel] = float4(nbtf.evalDebug(wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis), 1.0);
    else ouputColor[pixel] = nbtf.evalDebug_realDebug(wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);



}
