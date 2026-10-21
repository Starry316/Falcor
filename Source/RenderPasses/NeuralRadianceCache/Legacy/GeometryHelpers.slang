// Original Falcor 4.4 ray offset, retained for estimator alignment.
float3 computeLegacyRayOrigin(float3 pos, float3 normal)
{
    const float origin = 1.f / 32.f;
    const float fScale = 1.f / 65536.f;
    const float iScale = 256.f;

    // Per-component integer offset to bit representation of fp32 position.
    int3 iOff = int3(normal * iScale);
    float3 iPos = asfloat(asint(pos) + (pos < 0.f ? -iOff : iOff));

    // Select per-component between small fixed offset or above variable offset depending on distance to origin.
    float3 fOff = normal * fScale;
    return abs(pos) < origin ? pos + fOff : iPos;
}
