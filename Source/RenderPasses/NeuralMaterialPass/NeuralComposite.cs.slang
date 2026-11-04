Texture2D<float4> gForeground;
Texture2D<float4> gDisplay;
Texture2D<float4> gEnvironment;
Texture2D<float4> gArea;
RWTexture2D<float4> gColor;
cbuffer gCompositeCB
{
    uint2 gFrameDim;
    uint gEnvironmentEnabled;
    uint gAreaEnabled;
    float gMaterialOpacity;
    uint gOpaqueSubstrate;
};

float3 referenceMean(float4 accumulated)
{
    // Both RGB and count were divided by the SAME number of accumulated
    // frames. Their ratio is therefore a per-sample mean, even for variable
    // batches and a partial final batch. Coverage never occupies this count.
    return accumulated.w > 0.f ? accumulated.rgb / accumulated.w : float3(0.f);
}

[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (any(tid.xy >= gFrameDim)) return;
    float4 display = gDisplay[tid.xy];
    if (display.w < 0.f) { gColor[tid.xy] = float4(max(display.rgb,0.f),1.f); return; }
    float4 foreground = gForeground[tid.xy];
    uint flags = uint(display.w);
    if ((flags & 2u) != 0)
    {
        if (gEnvironmentEnabled != 0) foreground.rgb += referenceMean(gEnvironment[tid.xy]);
        if (gAreaEnabled != 0) foreground.rgb += referenceMean(gArea[tid.xy]);
        if (gOpaqueSubstrate != 0) foreground.a = 1.f;
        foreground *= saturate(gMaterialOpacity);
    }
    foreground.rgb += (1.f - foreground.a) * display.rgb;
    if ((flags & 1u) != 0) foreground.a = 1.f;
    gColor[tid.xy] = float4(max(foreground.rgb,0.f),foreground.a);
}
