import SynthesisUtils;
float3 HexTiling(
    Texture2D<float3> tex,
    SamplerState samp,
    float2 dfdx,
    float2 dfdy,
    float2 uv,
    float rotStrength = 0,
    float r = 0.5,
    float fallOffContrast = 0.6,
    float exp = 7
)
{
    // Get triangle info.
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);
    float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
    float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
    float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
    float2 cen1 = MakeCenST(vertex1);
    float2 cen2 = MakeCenST(vertex2);
    float2 cen3 = MakeCenST(vertex3);
    float2 st1 = mul(uv - cen1, rot1) + cen1 + hash(vertex1);
    float2 st2 = mul(uv - cen2, rot2) + cen2 + hash(vertex2);
    float2 st3 = mul(uv - cen3, rot3) + cen3 + hash(vertex3);
    // Fetch input.
    float3 c1 = tex.SampleGrad(samp, frac(st1), mul(dfdx, rot1), mul(dfdy, rot1));
    float3 c2 = tex.SampleGrad(samp, frac(st2), mul(dfdx, rot2), mul(dfdy, rot2));
    float3 c3 = tex.SampleGrad(samp, frac(st3), mul(dfdx, rot3), mul(dfdy, rot3));
    // Use luminance as weight.
    float3 Lw = float3(0.299, 0.587, 0.114);
    float3 Dw = float3(dot(c1.xyz, Lw), dot(c2.xyz, Lw), dot(c3.xyz, Lw));
    Dw = lerp(1.0, Dw, fallOffContrast);          // 0.6
    float3 W = Dw * pow(float3(w1, w2, w3), exp); // 7
    W /= (W.x + W.y + W.z);
    if (r != 0.5)
        W = Gain3(W, r);
    return W.x * c1 + W.y * c2 + W.z * c3;
}
float HexTiling(
    Texture2D<float> tex,
    SamplerState samp,
    float2 dfdx,
    float2 dfdy,
    float2 uv,
    float rotStrength = 0,
    float r = 0.5,
    float fallOffContrast = 0.6,
    float exp = 7
)
{
    // Get triangle info.
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);
    float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
    float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
    float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
    float2 cen1 = MakeCenST(vertex1);
    float2 cen2 = MakeCenST(vertex2);
    float2 cen3 = MakeCenST(vertex3);
    float2 st1 = mul(uv - cen1, rot1) + cen1 + hash(vertex1);
    float2 st2 = mul(uv - cen2, rot2) + cen2 + hash(vertex2);
    float2 st3 = mul(uv - cen3, rot3) + cen3 + hash(vertex3);
    // Fetch input.
    float c1 = tex.SampleGrad(samp, frac(st1), mul(dfdx, rot1), mul(dfdy, rot1));
    float c2 = tex.SampleGrad(samp, frac(st2), mul(dfdx, rot2), mul(dfdy, rot2));
    float c3 = tex.SampleGrad(samp, frac(st3), mul(dfdx, rot3), mul(dfdy, rot3));
    float3 Dw = float3(c1, c2, c3);
    Dw = lerp(1.0, Dw, fallOffContrast);        // 0.6
    float3 W = Dw * pow(float3(w1, w2, w3), exp); // 7
    W /= (W.x + W.y + W.z);
    if (r != 0.5)
        W = Gain3(W, r);
    return max(max(W.x * c1, W.y * c2),  W.z * c3);
}
float4 HexTiling(
    Texture2DArray<float4> tex,
    SamplerState samp,
    float2 dfdx,
    float2 dfdy,
    float2 uv,
    uint layer,
    float rotStrength = 0,
    float r = 0.5,
    float fallOffContrast = 0.6,
    float exp = 7
)
{
    // Get triangle info.
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);
    float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
    float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
    float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
    float2 cen1 = MakeCenST(vertex1);
    float2 cen2 = MakeCenST(vertex2);
    float2 cen3 = MakeCenST(vertex3);
    float2 st1 = mul(uv - cen1, rot1) + cen1 + hash(vertex1);
    float2 st2 = mul(uv - cen2, rot2) + cen2 + hash(vertex2);
    float2 st3 = mul(uv - cen3, rot3) + cen3 + hash(vertex3);
    // Fetch input.
    float4 c1 = tex.SampleGrad(samp, float3(st1, layer), mul(dfdx, rot1), mul(dfdy, rot1));
    float4 c2 = tex.SampleGrad(samp, float3(st2, layer), mul(dfdx, rot2), mul(dfdy, rot2));
    float4 c3 = tex.SampleGrad(samp, float3(st3, layer), mul(dfdx, rot3), mul(dfdy, rot3));
    float4 Lw = float4(0.25);
    float3 Dw = float3(dot(c1, Lw), dot(c2, Lw), dot(c3, Lw));
    Dw = lerp(1.0, Dw, fallOffContrast);          // 0.6
    float3 W = Dw * pow(float3(w1, w2, w3), exp); // 7
    W /= (W.x + W.y + W.z);
    if (r != 0.5)
        W = Gain3(W, r);
    return W.x * c1 + W.y * c2 + W.z * c3;
}
float HexTilingHeight(
    Texture2D<float4> tex,
    SamplerState samp,
    float2 dfdx,
    float2 dfdy,
    float2 uv,
    float rotStrength = 0,
    float r = 0.5,
    float fallOffContrast = 0.6,
    float exp = 7
)
{
    uv *= 1;
    // return tex.SampleGrad(samp, frac(uv), dfdx, dfdy).x;
    // Get triangle info.
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);
    float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
    float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
    float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
    float2 cen1 = MakeCenST(vertex1);
    float2 cen2 = MakeCenST(vertex2);
    float2 cen3 = MakeCenST(vertex3);
    float2 st1 = mul(uv - cen1, rot1) + cen1 + hash(vertex1);
    float2 st2 = mul(uv - cen2, rot2) + cen2 + hash(vertex2);
    float2 st3 = mul(uv - cen3, rot3) + cen3 + hash(vertex3);
    // Fetch input.
    float c1 = tex.SampleGrad(samp, frac(st1), mul(dfdx, rot1), mul(dfdy, rot1)).x;
    float c2 = tex.SampleGrad(samp, frac(st2), mul(dfdx, rot2), mul(dfdy, rot2)).x;
    float c3 = tex.SampleGrad(samp, frac(st3), mul(dfdx, rot3), mul(dfdy, rot3)).x;
    float3 Dw = float3(c1, c2, c3);
    Dw = lerp(1.0, Dw, fallOffContrast);          // 0.6
    float3 W = Dw * pow(float3(w1, w2, w3), exp); // 7
    W /= (W.x + W.y + W.z);
    if (r != 0.5)
        W = Gain3(W, r);
    return W.x * c1 + W.y * c2 + W.z * c3;
}
