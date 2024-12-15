#include "Utils/Math/MathConstants.slangh"
import Rendering.Materials.TexLODHelpers;
import Scene.Material.TextureSampler;

inline float inversesqrt(float x) { return 1.0 / sqrt(x); }
float2 MakeCenST(int2 Vertex)
{
    float2x2 invSkewMat = float2x2(1.0, 0.5, 0.0, 1.0 / 1.15470054);
    return mul(invSkewMat, Vertex) / (2 * sqrt(3));
}
float2x2 LoadRot2x2(int2 idx, float rotStrength)
{
    float angle = abs(idx.x * idx.y) + abs(idx.x + idx.y) + M_PI;
    // Remap to +/-pi.
    angle = fmod(angle, 2 * M_PI);
    if (angle < 0)
        angle += 2 * M_PI;
    if (angle > M_PI)
        angle -= 2 * M_PI;
    angle *= rotStrength;
    float cs = cos(angle), si = sin(angle);
    return float2x2(cs, -si, si, cs);
}
float3 ProduceHexWeights(float3 W, int2 vertex1, int2 vertex2, int2 vertex3)
{
    float3 res = 0.0;
    int v1 = (vertex1.x - vertex1.y) % 3;
    if (v1 < 0)
        v1 += 3;
    int vh = v1 < 2 ? (v1 + 1) : 0;
    int vl = v1 > 0 ? (v1 - 1) : 2;
    int v2 = vertex1.x < vertex3.x ? vl : vh;
    int v3 = vertex1.x < vertex3.x ? vh : vl;
    res.x = v3 == 0 ? W.z : (v2 == 0 ? W.y : W.x);
    res.y = v3 == 1 ? W.z : (v2 == 1 ? W.y : W.x);
    res.z = v3 == 2 ? W.z : (v2 == 2 ? W.y : W.x);
    return res;
}
float3 Gain3(float3 x, float r)
{
    // Increase contrast when r > 0.5 and
    // reduce contrast if less.
    float k = log(1 - r) / log(0.5);
    float3 s = 2 * step(0.5, x);
    float3 m = 2 * (1 - s);
    float3 res = 0.5 * s + 0.25 * m * pow(max(0.0, s + x * m), k);
    return res.xyz / (res.x + res.y + res.z);
}
// Output: weights associated with each hex tile and integer centers
void TriangleGrid(out float w1, out float w2, out float w3, out int2 vertex1, out int2 vertex2, out int2 vertex3, float2 st)
{
    // Scaling of the input
    st *= 2 * sqrt(3);
    // Skew input space into simplex triangle grid.
    const float2x2 gridToSkewedGrid = float2x2(1.0, -0.57735027, 0.0, 1.15470054);
    float2 skewedCoord = mul(gridToSkewedGrid, st);
    int2 baseId = int2(floor(skewedCoord));
    float3 temp = float3(frac(skewedCoord), 0);
    temp.z = 1.0 - temp.x - temp.y;
    float s = step(0.0, -temp.z);
    float s2 = 2 * s - 1;
    w1 = -temp.z * s2;
    w2 = s - temp.y * s2;
    w3 = s - temp.x * s2;
    vertex1 = baseId + int2(s, s);
    vertex2 = baseId + int2(s, 1 - s);
    vertex3 = baseId + int2(1 - s, s);
}
float2 hash(float2 p)
{
    float2 r = mul(float2x2(127.1, 311.7, 269.5, 183.3), p);
    return frac(sin(r) * 43758.5453);
}
