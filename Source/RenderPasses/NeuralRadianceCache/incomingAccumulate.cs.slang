// A separate linear HDR average: predictions always show the current cache.
Texture2D<float4> referenceSample;
RWTexture2D<float4> referenceMean;
cbuffer Params { uint sampleCount; };
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint w, h;
    referenceMean.GetDimensions(w, h);
    if (tid.x >= w || tid.y >= h) return;
    float4 value = referenceSample[tid.xy];
    referenceMean[tid.xy] = sampleCount == 0 ? value :
        float4(lerp(referenceMean[tid.xy].rgb, value.rgb, 1.f / (sampleCount + 1.f)), value.a);
}
