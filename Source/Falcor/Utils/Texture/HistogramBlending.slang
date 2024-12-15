import SynthesisUtils;

struct HistogramBlendingHFData
{
    Texture2D<float> tex;
    Texture2D<float> invTex;
};

struct HistogramBlendingColorData
{
    Texture2D<float3> tex;
    Texture2D<float3> invTex;
};
struct HistogramBlendingFeatureData
{
    Texture2DArray<float4> tex;
    Texture2DArray<float4> invTex;
    Texture2D<float> acf;
};

float HistorgramBlending(
    HistogramBlendingHFData data,
    SamplerState sampler,
    float2 ddx,
    float2 ddy,
    float2 uv
)
{
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

    float2 st1 = uv + hash(vertex1);
    float2 st2 = uv + hash(vertex2);
    float2 st3 = uv + hash(vertex3);

    float g1 = data.tex.SampleGrad(sampler, frac(st1), ddx, ddy);
    float g2 = data.tex.SampleGrad(sampler, frac(st2), ddx, ddy);
    float g3 = data.tex.SampleGrad(sampler, frac(st3), ddx, ddy);

    // float G = w1 * g1 + w2 * g2 + w3 * g3;
    float G = max(max(w1 * g1, w2 * g2), w3 * g3);
    G -= 0.5f;
    G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
    G += 0.5f;
    G = abs(G);
    G = clamp(G, 0.001, 0.999);
    float res;
    res = data.invTex.SampleLevel(sampler, float2(G, 0.0f), 0.f);
    return res;
}

float4 HistorgramBlending(
    HistogramBlendingFeatureData data,
    SamplerState sampler,
    float2 uv,
    uint layer,
    float2 ddx,
    float2 ddy
)
{
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

    float2 st1 = uv + hash(vertex1);
    float2 st2 = uv + hash(vertex2);
    float2 st3 = uv + hash(vertex3);
    float4 g1 = data.tex.SampleGrad(sampler, float3(st1, layer), ddx, ddy);
    float4 g2 = data.tex.SampleGrad(sampler, float3(st2, layer), ddx, ddy);
    float4 g3 = data.tex.SampleGrad(sampler, float3(st3, layer), ddx, ddy);

    float4 G = w1 * g1 + w2 * g2 + w3 * g3;
    G -= 0.5f;
    G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
    G += 0.5f;
    G = abs(G);
    G = clamp(G, 0.0001, 0.999);
    float4 res;
    res.x = data.invTex.SampleLevel(sampler, float3(G.x, 0.0f, layer), 0.f).x;
    res.y = data.invTex.SampleLevel(sampler, float3(G.y, 0.0f, layer), 0.f).y;
    res.z = data.invTex.SampleLevel(sampler, float3(G.z, 0.0f, layer), 0.f).z;
    res.w = data.invTex.SampleLevel(sampler, float3(G.w, 0.0f, layer), 0.f).w;
    return res;
}











float3 HistorgramBlending(
    HistogramBlendingColorData data,
    SamplerState sampler,
    float2 ddx,
    float2 ddy,
    float2 uv
)
{
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

    float2 st1 = uv + hash(vertex1);
    float2 st2 = uv + hash(vertex2);
    float2 st3 = uv + hash(vertex3);

    float3 g1 = data.tex.SampleGrad(sampler, frac(st1), ddx, ddy);
    float3 g2 = data.tex.SampleGrad(sampler, frac(st2), ddx, ddy);
    float3 g3 = data.tex.SampleGrad(sampler, frac(st3), ddx, ddy);

    float3 G = w1 * g1 + w2 * g2 + w3 * g3;
    G -= 0.5f;
    G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
    G += 0.5f;
    G = abs(G);
    G = clamp(G, 0.001, 0.999);
    float3 res;
    res.r = data.invTex.SampleLevel(sampler, float2(G.x, 0.0f), 0.f).r;
    res.g = data.invTex.SampleLevel(sampler, float2(G.x, 0.0f), 0.f).g;
    res.g = data.invTex.SampleLevel(sampler, float2(G.x, 0.0f), 0.f).b;
    return res;
}

// float HistorgramBlendingFunc(
//     Texture2D<float> tex,
//     Texture2D<float> invTex,
//     SamplerState sampler,
//     float2 ddx,
//     float2 ddy,
//     float2 uv
// )
// {
//     float w1, w2, w3;
//     int2 vertex1, vertex2, vertex3;
//     TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);

//     float2 st1 = uv  + hash(vertex1);
//     float2 st2 = uv  + hash(vertex2);
//     float2 st3 = uv  + hash(vertex3);

//     float g1 = tex.SampleGrad(sampler, frac(st1), ddx, ddy);
//     float g2 = tex.SampleGrad(sampler, frac(st2), ddx, ddy);
//     float g3 = tex.SampleGrad(sampler, frac(st3), ddx, ddy);

//     float G = w1 * g1 + w2 * g2 + w3 * g3;
//     G -= 0.5f;
//     G *= inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
//     G += 0.5f;
//     G = abs(G);
//     G = clamp(G, 0.001, 0.999);
//     float res;

//     res = invTex.SampleLevel(sampler, float2(G, 0.0f), 0.f);

//     return res;
// }
