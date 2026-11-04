// Raster visibility -> original tangent-plane material coordinates.
// The UV metric comes from the current deformed triangle, not a unit TBN.
import Scene.Scene;
#include "NeuralTypes.slangh"

Texture2D<float4> gPosW;
Texture2D<float4> gNormW;
Texture2D<float4> gTangentW;
Texture2D<float4> gFaceNormalW;
Texture2D<float2> gTexC;
Texture2D<float4> gTexGrads;
Texture2D<uint4> gMtlData;
Texture2D<PackedHitInfo> gVBuffer;
RWStructuredBuffer<NeuralRay> gRays;
RWStructuredBuffer<NeuralFrame> gFrames;

cbuffer gMappingCB
{
    uint2 frameDim;
    float3 boundsMin;
    float3 boundsMax;
    uint baseResolution;
    uint mipLevels;
    uint automaticLod;
    float fixedLod;
    float lodBias;
    float uvScale;
    float crop;
    float thickness;
    float maxParallax;
    uint smoothNormals;
    uint twoSided;
    uint referenceTop;
    uint stackCorrection;
    uint flipV;
    int materialId;
};

float3 safeUnit(float3 v)
{
    return v * rsqrt(max(dot(v, v), 1e-30f));
}

// Reference-surface footprint from raster pixel derivatives, before periodic
// wrapping. Repeat and source crop are each applied exactly once. All planes
// share this level. A non-mip asset is simply mipLevels == 1.
float featureLod(float4 gradients)
{
    float last = float(max(mipLevels, 1u) - 1u);
    if (automaticLod == 0) return clamp(fixedLod, 0.f, last);
    float rho = max(length(gradients.xy), length(gradients.zw));
    rho *= float(baseResolution) * uvScale * (1.f - 2.f * crop);
    float level = log2(max(rho, 1e-20f)) + lodBias;
    return all(isfinite(gradients)) && isfinite(level) ? clamp(level, 0.f, last) : last;
}

struct NeuralMappingResult
{
    NeuralRay ray;
    NeuralFrame frame;
};

NeuralMappingResult mapMaterialPixel(uint2 pixel)
{
    NeuralMappingResult result = {};
    NeuralRay ray = {};
    NeuralFrame frame = {};

    float4 pos = gPosW[pixel];
    if (pos.w == 0.f || !all(isfinite(pos))) return result;
    HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.getType() != HitType::Triangle) return result;
    TriangleHit triangle = hit.getTriangleHit();
    if (materialId >= 0 && gScene.getMaterialID(triangle.instanceID) != uint(materialId)) return result;

    StaticVertexData vertices[3];
    VertexData vertex = gScene.getVertexData(triangle.instanceID, triangle.primitiveIndex,
                                           triangle.getBarycentricWeights(), vertices);
    float4x4 world = gScene.getWorldMatrix(triangle.instanceID);
    float3 p[3];
    float2 uv[3];
    [unroll] for (uint i = 0; i < 3; ++i)
    {
        p[i] = mul(world, float4(vertices[i].position, 1.f)).xyz;
        uv[i] = vertices[i].texCrd;
        if (flipV != 0) uv[i].y = 1.f - uv[i].y;
    }
    float3 e1 = p[1] - p[0], e2 = p[2] - p[0];
    float2 a = uv[1] - uv[0], b = uv[2] - uv[0];
    float det = a.x * b.y - a.y * b.x;
    if (!isfinite(det) || abs(det) <= 1e-12f) return result;
    float3 dpdu = (e1 * b.y - e2 * a.y) / det;
    float3 dpdv = (-e1 * b.x + e2 * a.x) / det;

    // Raw animated vertex normals preserve the original two-sided convention;
    // stock material normal maps/face-forwarding must not silently remap the
    // learned material. Falcor still owns all vertex animation/transforms.
    float3 n = smoothNormals != 0 ? vertex.normalW : vertex.faceNormalW;
    n = safeUnit(n);
    float3 cameraDelta = pos.xyz - gScene.camera.getPosition();
    float distance = length(cameraDelta);
    if (distance <= 1e-12f) return result;
    float3 d = cameraDelta / distance;
    float side = twoSided != 0 && dot(d, n) > 0.f ? -1.f : 1.f;
    n *= side;
    dpdu -= n * dot(dpdu, n);
    dpdv -= n * dot(dpdv, n);
    float3 tangent = safeUnit(dpdu);
    float3 bitangent = safeUnit(cross(n, tangent));
    float handedness = dot(bitangent, dpdv) < 0.f ? -1.f : 1.f;
    bitangent *= handedness;
    float3 dualU = cross(dpdv, n);
    float denom = dot(dpdu, dualU);
    if (!isfinite(denom) || abs(denom) <= 1e-12f) return result;
    dualU /= denom;
    float3 dualV = cross(n, dpdu) / denom;
    float3 extent = boundsMax - boundsMin;
    float3 slope = float3(dot(d, dualU) * uvScale * extent.x,
                          dot(d, dualV) * uvScale * extent.y,
                          dot(d, n) * extent.z / max(thickness, 1e-12f));

    // Same normal-field midpoint correction as mpmat/cuda/mapping.h. This
    // follows animated vertex normals, so no private animation system is used.
    if (stackCorrection != 0 && smoothNormals != 0)
    {
        float3x3 normalMatrix = gScene.getInverseTransposeWorldMatrix(triangle.instanceID);
        float3 n0 = safeUnit(mul(normalMatrix, vertices[0].normal)) * side;
        float3 n1 = safeUnit(mul(normalMatrix, vertices[1].normal)) * side;
        float3 n2 = safeUnit(mul(normalMatrix, vertices[2].normal)) * side;
        float3 bary = triangle.getBarycentricWeights();
        float3 raw = n0 * bary.x + n1 * bary.y + n2 * bary.z;
        float rawLength = length(raw);
        float3 nu = ((n1 - n0) * b.y - (n2 - n0) * a.y) / det;
        float3 nv = (-(n1 - n0) * b.x + (n2 - n0) * a.x) / det;
        float dn = dot(d, n);
        if (rawLength > 1e-8f && abs(dn) > 1e-6f)
        {
            nu = (nu - n * dot(n, nu)) / rawLength;
            nv = (nv - n * dot(n, nv)) / rawLength;
            float reference = referenceTop != 0 ? boundsMax.z : 0.f;
            float depth = (reference - .5f * (boundsMin.z + boundsMax.z)) * thickness / extent.z;
            float du = -depth * dot(d, dualU) / dn;
            float dv = -depth * dot(d, dualV) / dn;
            float3 delta = nu * du + nv * dv;
            float3 midRaw = n + delta;
            float midLength = length(midRaw);
            float3 mid = midRaw / max(midLength, 1e-15f);
            float3 midNu = (nu - mid * dot(mid, nu)) / max(midLength, 1e-15f);
            float3 midNv = (nv - mid * dot(mid, nv)) / max(midLength, 1e-15f);
            float3 offsetU = dpdu - mid * dot(dpdu, mid) - midNu * depth;
            float3 offsetV = dpdv - mid * dot(dpdv, mid) - midNv * depth;
            float midDenom = dot(offsetU, cross(offsetV, mid)), midDn = dot(d, mid);
            float turn2 = dot(delta, delta), roundoff = 1e-14f * dn * dn;
            if (turn2 > roundoff && turn2 <= .25f && midDenom * denom > 0.f &&
                abs(midDenom) > .1f * abs(denom) && midDn * dn > 0.f && abs(midDn) > .25f * abs(dn))
            {
                float3 midU = cross(offsetV, mid) / midDenom;
                float3 midV = cross(mid, offsetU) / midDenom;
                float2 corrected = float2(dot(d, midU) * extent.x, dot(d, midV) * extent.y) * (dn / midDn * uvScale);
                if (all(isfinite(corrected))) slope.xy = corrected;
            }
        }
    }

    if (maxParallax > 0.f)
    {
        float span = max(abs(slope.x / extent.x), abs(slope.y / extent.y)) * extent.z / max(abs(slope.z), 1e-15f);
        float knee = .5f * maxParallax;
        if (span > knee)
        {
            float limited = knee + knee * (1.f - exp(-(span - knee) / knee));
            slope.xy *= limited / span;
        }
    }
    if (referenceTop != 0 && dot(d, n) >= 0.f) return result;
    float reference = referenceTop != 0 ? boundsMax.z : 0.f;
    // Falcor's OBJ/Assimp import flips V. Undo it for the original Blender/OBJ
    // neural chart in both the source coordinate and its world-space Jacobian.
    float2 sourceUV = gTexC[pixel];
    if (flipV != 0) sourceUV.y = 1.f - sourceUV.y;
    float3 at = float3(boundsMin.xy + sourceUV * uvScale * extent.xy, reference);
    float margin = 1e-5f * extent.z;
    float backExtent = max(abs(boundsMin.z - reference), abs(boundsMax.z - reference)) + margin;
    float back = min(distance, backExtent / max(abs(slope.z), 1e-15f));
    ray.origin = at - slope * back;
    ray.direction = slope;
    ray.view = -float3(dot(d, tangent), dot(d, bitangent), dot(d, n));
    ray.lod = featureLod(gTexGrads[pixel]);
    ray.valid = all(isfinite(ray.origin)) && all(isfinite(ray.direction)) && all(isfinite(ray.view)) ? 1u : 0u;
    frame.tangent = float4(tangent, handedness);
    frame.normal = float4(n, float(ray.valid));
    result.ray = ray;
    result.frame = frame;
    return result;
}

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadID.xy;
    if (any(pixel >= frameDim)) return;
    uint id = pixel.y * frameDim.x + pixel.x;
    // Commit each output once. Misses still receive zero records, while hits
    // avoid an initial full-record clear followed by the final geometry write.
    NeuralMappingResult result = mapMaterialPixel(pixel);
    gRays[id] = result.ray;
    gFrames[id] = result.frame;
}
