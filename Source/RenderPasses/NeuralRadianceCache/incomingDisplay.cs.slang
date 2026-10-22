// Two orthographic spheres with identical orientation and exposure.
Texture2D<float4> prediction;
Texture2D<float4> referenceMean;
Texture2D<float4> directRadiance;
RWTexture2D<float4> spheres;
cbuffer Params { float exposure; float2 rotation; uint includeDirect; };
float3 displayColor(float3 c)
{
    c = max(c, 0.f) * exp2(exposure);
    c = c / (1.f + c);
    return pow(c, float3(1.f / 2.2f));
}
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= 512 || tid.y >= 256) return;
    float2 xy = (float2(tid.x % 256, tid.y) + .5f - 128.f) / 120.f;
    xy.y = -xy.y;
    float r2 = dot(xy, xy);
    float3 c = float3(.035f);
    if (r2 <= 1.f)
    {
        float3 d = float3(xy, sqrt(1.f - r2));
        float2 a = rotation * (3.14159265359f / 180.f);
        d.yz = float2(cos(a.y)*d.y - sin(a.y)*d.z, sin(a.y)*d.y + cos(a.y)*d.z);
        d.xz = float2(cos(a.x)*d.x + sin(a.x)*d.z, -sin(a.x)*d.x + cos(a.x)*d.z);
        float2 uv = float2(atan2(d.x, d.z) / (2.f * 3.14159265359f) + .5f,
            acos(clamp(d.y, -1.f, 1.f)) / 3.14159265359f);
        uint2 p = min(uint2(frac(uv.x) * 512, uv.y * 256), uint2(511,255));
        float4 ref = referenceMean[p];
        float3 radiance = tid.x < 256 ? prediction[p].rgb : ref.rgb;
        if (includeDirect != 0) radiance += directRadiance[p].rgb;
        c = displayColor(radiance);
        // Grey stripes indicate directions outside the trained surface hemisphere.
        if (ref.a < .75f) c = lerp(c, float3(.22f), ((tid.x + tid.y) % 12 < 5) ? .85f : .35f);
        if (ref.a == 0.f) c = float3(.12f); // Background selection: no surface.
    }
    spheres[tid.xy] = float4(c, 1.f);
}
