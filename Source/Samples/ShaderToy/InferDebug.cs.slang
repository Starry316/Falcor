import Utils.Texture.HistogramBlending;
import Utils.Neural.NBTF;
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;

cbuffer ToyCB
{
    float2 iResolution;
    float iGlobalTime;
    // HistogramBlendingHFData hfData;
    float gUVScaling;
    NBTF nbtf;
    bool gSynthesis;
    bool gDebugMLP;
    float2 gWo;
    float2 gWi;
};
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

    float2 h, d;
    dirToRusinkiewicz(sample_hemisphere(gWi), sample_hemisphere(gWo), h, d);
    float3 vh = spherical_to_cartesian_rad(h);
    float3 vd = spherical_to_cartesian_rad(d);

    h = hemisphereDirToDiskSquare(vh);
    d = hemisphereDirToDiskSquare(vd);
    if (!gDebugMLP)
        ouputColor[pixel] = float4(nbtf.evalDebug(h, d, uv.yx * gUVScaling, ddx, ddy, gSynthesis), 1.0);
    else ouputColor[pixel] = nbtf.evalDebugFP32(h, d, uv.yx * gUVScaling, ddx, ddy, gSynthesis);



}
