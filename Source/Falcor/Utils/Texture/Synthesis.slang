import HexTiling;



// struct HistogramPreserved
// {
//     float2 hash(in int2 p)
//     {
//         float2x2 hashMat = float2x2(float2(127.1, 311.7), float2(269.5, 183.3));

//         float2 q = mul(hashMat, float2(p));
//         q[0] = sin(q[0]);
//         q[1] = sin(q[1]);
//         q *= 43758.5453;
//         return float2(q[0] - floor(q[0]), q[1] - floor(q[1]));
//     }

//     // Dual tiling and blending (Lutz and colleagues, 2023)
//     void DualGrid(in float2 p_uv, inout float v1, inout float v2, inout int2 vertex1, inout int2 vertex2)
//     {
//         float2 uv = p_uv * 2.0 * sqrt(3.0);
//         vertex1 = int2(uv - 0.5);
//         vertex2 = int2(uv) + int2(10);

//         float2 localuv1 = frac(uv - 0.5) * 2.0 - 1.0;
//         float2 localuv2 = frac(uv) * 2.0 - 1.0;

//         v1 = min(1.0 - abs(localuv1.x), 1.0 - abs(localuv1.y));
//         v2 = min(1.0 - abs(localuv2.x), 1.0 - abs(localuv2.y));
//     }

//     // Compute local triangle barycentric coordinates and vertex IDs
//     void TriangleGrid(float2 uv, out float w1, out float w2, out float w3, out int2 vertex1, out int2 vertex2, out int2 vertex3)
//     {
//         // Scaling of the input
//         uv *= 3.464; // 2 * sqrt(3)

//         // Skew input space into simplex triangle grid
//         const float2x2 gridToSkewedGrid = float2x2(1.0, -0.57735027, 0.0, 1.15470054);
//         float2 skewedCoord = mul(gridToSkewedGrid, uv);

//         // Compute local triangle vertex IDs and local barycentric coordinates
//         int2 baseId = int2(floor(skewedCoord));
//         float3 temp = float3(frac(skewedCoord), 0);
//         temp.z = 1.0 - temp.x - temp.y;
//         if (temp.z > 0.0)
//         {
//             w1 = temp.z;
//             w2 = temp.y;
//             w3 = temp.x;
//             vertex1 = baseId;
//             vertex2 = baseId + int2(0, 1);
//             vertex3 = baseId + int2(1, 0);
//         }
//         else
//         {
//             w1 = -temp.z;
//             w2 = 1.0 - temp.y;
//             w3 = 1.0 - temp.x;
//             vertex1 = baseId + int2(1, 1);
//             vertex2 = baseId + int2(1, 0);
//             vertex3 = baseId + int2(0, 1);
//         }
//     }

//     float inversesqrt(float x) { return 1.0 / sqrt(x); }

//     float4 variancePresevered(float2 uv, Texture2D<float4> inputTex, SamplerState sampler)
//     {
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         // uv *= 2.0f;
//         TriangleGrid(uv, w1, w2, w3, vertex1, vertex2, vertex3);

//         // Assign random offset to each triangle vertex
//         float2 uv1 = uv + hash(int2(vertex1));
//         float2 uv2 = uv + hash(int2(vertex2));
//         float2 uv3 = uv + hash(int2(vertex3));

//         float4 G11 = inputTex.SampleLevel(sampler, frac(uv1), 0.f);
//         float4 G12 = inputTex.SampleLevel(sampler, frac(uv2), 0.f);
//         float4 G13 = inputTex.SampleLevel(sampler, frac(uv3), 0.f);

//         float4 res = w1 * G11 + w2 * G12 + w3 * G13;
//         res = res * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
//         return res;
//     }
//     float4 variancePresevered(float2 uv, Texture2DArray<float4> inputTex, int arrayIndx, SamplerState sampler)
//     {
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         // uv *= 2.0f;
//         TriangleGrid(uv, w1, w2, w3, vertex1, vertex2, vertex3);

//         // Assign random offset to each triangle vertex
//         float2 uv1 = uv + hash(int2(vertex1));
//         float2 uv2 = uv + hash(int2(vertex2));
//         float2 uv3 = uv + hash(int2(vertex3));

//         float4 G11 = inputTex.SampleLevel(sampler, float3(frac(uv1), arrayIndx), 0.f);
//         float4 G12 = inputTex.SampleLevel(sampler, float3(frac(uv2), arrayIndx), 0.f);
//         float4 G13 = inputTex.SampleLevel(sampler, float3(frac(uv3), arrayIndx), 0.f);

//         float4 res = w1 * G11 + w2 * G12 + w3 * G13;
//         res = res * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
//         return res;
//     }

//     float4 histogramPresevered( float2 uv, Texture2D<float3> inputTex, Texture2D<float3> invTex, SamplerState sampler)
//     {
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         // uv *= 2.0f;
//         TriangleGrid(uv, w1, w2, w3, vertex1, vertex2, vertex3);

//         // Assign random offset to each triangle vertex
//         float2 uv1 = uv + hash(int2(vertex1));
//         float2 uv2 = uv + hash(int2(vertex2));
//         float2 uv3 = uv + hash(int2(vertex3));

//         float3 G11 = inputTex.SampleLevel(sampler, frac(uv1), 0.f);
//         float3 G12 = inputTex.SampleLevel(sampler, frac(uv2), 0.f);
//         float3 G13 = inputTex.SampleLevel(sampler, frac(uv3), 0.f);

//         float3 G = w1 * G11 + w2 * G12 + w3 * G13;
//         G = G - float3(0.5);
//         G = G * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
//         G = G + float3(0.5);
//         G = abs(G);
//         G = clamp(G, 0.01, 0.99);
//         float4 res;

//         res.r = invTex.SampleLevel(sampler, float2(G.r, 0.0f), 0.f).r;
//         res.g = invTex.SampleLevel(sampler, float2(G.g, 0.0f), 0.f).g;
//         res.b = invTex.SampleLevel(sampler, float2(G.b, 0.0f), 0.f).b;
//         return res;
//     }

//     float4 histogramPresevered(float2 uv, Texture2DArray<float4> inputTex, Texture2DArray<float4> invTex, int arrayIndx, SamplerState
//     sampler)
//     {
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         // uv *= 2.0f;
//         TriangleGrid(uv, w1, w2, w3, vertex1, vertex2, vertex3);

//         // Assign random offset to each triangle vertex
//         float2 uv1 = uv + hash(int2(vertex1));
//         float2 uv2 = uv + hash(int2(vertex2));
//         float2 uv3 = uv + hash(int2(vertex3));

//         float4 G11 = inputTex.SampleLevel(sampler, float3(frac(uv1), arrayIndx), 0.f);
//         float4 G12 = inputTex.SampleLevel(sampler, float3(frac(uv2), arrayIndx), 0.f);
//         float4 G13 = inputTex.SampleLevel(sampler, float3(frac(uv3), arrayIndx), 0.f);

//         float4 G = w1 * G11 + w2 * G12 + w3 * G13;
//         G = G - float4(0.5);
//         G = G * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
//         G = G + float4(0.5);
//         G = abs(G);
//         G = clamp(G, 0.01, 0.99);
//         float4 res;

//         res.r = invTex.SampleLevel(sampler, float3(G.r, 0.0f, arrayIndx), 0.f).r;
//         res.g = invTex.SampleLevel(sampler, float3(G.g, 0.0f, arrayIndx), 0.f).g;
//         res.b = invTex.SampleLevel(sampler, float3(G.b, 0.0f, arrayIndx), 0.f).b;
//         res.a = invTex.SampleLevel(sampler, float3(G.a, 0.0f, arrayIndx), 0.f).a;
//         return res;
//     }
//     float4 histogramPresevered( float2 uv, Texture2DArray<float4> inputTex, Texture2DArray<float4> invTex, int arrayIndx, SamplerState
//     sampler, ITextureSampler lodSampler )
//     {
//         // float w1, w2, w3;
//         // int2 vertex1, vertex2, vertex3;
//         // // uv *= 2.0f;
//         // TriangleGrid(uv, w1, w2, w3, vertex1, vertex2, vertex3);

//         // // Assign random offset to each triangle vertex
//         // float2 uv1 = uv + hash(int2(vertex1));
//         // float2 uv2 = uv + hash(int2(vertex2));
//         // float2 uv3 = uv + hash(int2(vertex3));

//         // // float4 G11 = lodSampler.sampleTexture(inputTex, sampler, float3(frac(float2(0.0f)), arrayIndx));
//         // // float4 G12 = lodSampler.sampleTexture(inputTex, sampler, float3(frac(float2(0.0f)), arrayIndx));
//         // // float4 G13 = lodSampler.sampleTexture(inputTex, sampler, float3(frac(float2(0.0f)), arrayIndx));

//         // float4 G11 = lodSampler.sampleTexture(inputTex, sampler, float3(frac(uv1), arrayIndx));
//         // float4 G12 = lodSampler.sampleTexture(inputTex, sampler, float3(frac(uv2), arrayIndx));
//         // float4 G13 = lodSampler.sampleTexture(inputTex, sampler, float3(frac(uv3), arrayIndx));

//         // // float4 G11 = inputTex.SampleLevel(sampler, float3(frac(uv1), arrayIndx), 0);
//         // // float4 G12 = inputTex.SampleLevel(sampler, float3(frac(uv2), arrayIndx), 0);
//         // // float4 G13 = inputTex.SampleLevel(sampler, float3(frac(uv3), arrayIndx), 0);

//         // float4 G = w1 * G11 + w2 * G12 + w3 * G13;
//         // G = G - float4(0.5);
//         // G = G * inversesqrt(w1 * w1 + w2 * w2 + w3 * w3);
//         // G = G + float4(0.5);
//         // G = abs(G);
//         // G = clamp(G, 0.01, 0.99);
//         float4 res;

//         // res.r = invTex.SampleLevel(sampler, float3(G.r, 0.0f, arrayIndx), 0.f).r;
//         // res.g = invTex.SampleLevel(sampler, float3(G.g, 0.0f, arrayIndx), 0.f).g;
//         // res.b = invTex.SampleLevel(sampler, float3(G.b, 0.0f, arrayIndx), 0.f).b;
//         // res.a = invTex.SampleLevel(sampler, float3(G.a, 0.0f, arrayIndx), 0.f).a;
//         return res;
//     }

// };
// struct HexTile
// {
//     // Output: weights associated with each hex tile and integer centers
//     void TriangleGrid(out float w1, out float w2, out float w3, out int2 vertex1, out int2 vertex2, out int2 vertex3, float2 st)
//     {
//         // Scaling of the input
//         st *= 2 * sqrt(3);
//         // Skew input space into simplex triangle grid.
//         const float2x2 gridToSkewedGrid = float2x2(1.0, -0.57735027, 0.0, 1.15470054);
//         float2 skewedCoord = mul(gridToSkewedGrid, st);
//         int2 baseId = int2(floor(skewedCoord));
//         float3 temp = float3(frac(skewedCoord), 0);
//         temp.z = 1.0 - temp.x - temp.y;
//         float s = step(0.0, -temp.z);
//         float s2 = 2 * s - 1;
//         w1 = -temp.z * s2;
//         w2 = s - temp.y * s2;
//         w3 = s - temp.x * s2;
//         vertex1 = baseId + int2(s, s);
//         vertex2 = baseId + int2(s, 1 - s);
//         vertex3 = baseId + int2(1 - s, s);
//     }
//     float2 hash(float2 p)
//     {
//         float2 r = mul(float2x2(127.1, 311.7, 269.5, 183.3), p);
//         return frac(sin(r) * 43758.5453);
//     }
//     // Input: tex is a texture with color
//     // Input: r increase contrast when r > 0.5
//     // Output: color is the blended result
//     // Output: weights shows the weight of each hex tile
//     void hex2colTex( out float3 color, Texture2D<float3> tex, SamplerState samp, float2 st, float rotStrength = 0, float r = 0.5, float
//     g_fallOffContrast = 0.6, float g_exp = 7)
//     {
//         float2 dSTdx, dSTdy;

//         dSTdx = float2(1.0f / 1920.0f, 0.0);
//         dSTdy = float2(0.0, 1.0f / 1080.0f);

//         // Get triangle info.
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
//         float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
//         float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
//         float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
//         float2 cen1 = MakeCenST(vertex1);
//         float2 cen2 = MakeCenST(vertex2);
//         float2 cen3 = MakeCenST(vertex3);
//         float2 st1 = mul(st - cen1, rot1) + cen1 + hash(vertex1);
//         float2 st2 = mul(st - cen2, rot2) + cen2 + hash(vertex2);
//         float2 st3 = mul(st - cen3, rot3) + cen3 + hash(vertex3);
//         // Fetch input.
//         float3 c1 = tex.SampleGrad(samp, frac(st1), mul(dSTdx, rot1), mul(dSTdy, rot1));
//         float3 c2 = tex.SampleGrad(samp, frac(st2), mul(dSTdx, rot2), mul(dSTdy, rot2));
//         float3 c3 = tex.SampleGrad(samp, frac(st3), mul(dSTdx, rot3), mul(dSTdy, rot3));
//         // Use luminance as weight.
//         float3 Lw = float3(0.299, 0.587, 0.114);
//         float3 Dw = float3(dot(c1.xyz, Lw), dot(c2.xyz, Lw), dot(c3.xyz, Lw));
//         Dw = lerp(1.0, Dw, g_fallOffContrast);          // 0.6
//         float3 W = Dw * pow(float3(w1, w2, w3), g_exp); // 7
//         W /= (W.x + W.y + W.z);
//         if (r != 0.5)
//             W = Gain3(W, r);
//         color = W.x * c1 + W.y * c2 + W.z * c3;
//     }
//     void hex2colTex( out float color, Texture2D<float> tex, SamplerState samp, float2 st, float rotStrength = 0, float r = 0.5, float
//     g_fallOffContrast = 0.6, float g_exp = 7 )
//     {
//         float2 dSTdx, dSTdy;

//         dSTdx = float2(1.0f / 1920.0f, 0.0);
//         dSTdy = float2(0.0, 1.0f / 1080.0f);

//         // Get triangle info.
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
//         float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
//         float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
//         float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
//         float2 cen1 = MakeCenST(vertex1);
//         float2 cen2 = MakeCenST(vertex2);
//         float2 cen3 = MakeCenST(vertex3);
//         float2 st1 = mul(st - cen1, rot1) + cen1 + hash(vertex1);
//         float2 st2 = mul(st - cen2, rot2) + cen2 + hash(vertex2);
//         float2 st3 = mul(st - cen3, rot3) + cen3 + hash(vertex3);
//         // Fetch input.
//         float c1 = tex.SampleGrad(samp, frac(st1), mul(dSTdx, rot1), mul(dSTdy, rot1));
//         float c2 = tex.SampleGrad(samp, frac(st2), mul(dSTdx, rot2), mul(dSTdy, rot2));
//         float c3 = tex.SampleGrad(samp, frac(st3), mul(dSTdx, rot3), mul(dSTdy, rot3));
//         // Use luminance as weight.
//         float3 Dw = float3(c1, c2, c3);
//         Dw = lerp(1.0, Dw, g_fallOffContrast);          // 0.6
//         float3 W = Dw * pow(float3(w1, w2, w3), g_exp); // 7
//         W /= (W.x + W.y + W.z);
//         if (r != 0.5)
//             W = Gain3(W, r);
//         color = W.x * c1 + W.y * c2 + W.z * c3;
//     }
//     float3 texSyn( Texture2D<float3> tex, SamplerState samp, float2 dSTdx, float2 dSTdy, float2 st, float rotStrength = 0, float r = 0.5,
//     float g_fallOffContrast = 0.6, float g_exp = 7 )
//     {
//         // Get triangle info.
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
//         float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
//         float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
//         float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
//         float2 cen1 = MakeCenST(vertex1);
//         float2 cen2 = MakeCenST(vertex2);
//         float2 cen3 = MakeCenST(vertex3);
//         float2 st1 = mul(st - cen1, rot1) + cen1 + hash(vertex1);
//         float2 st2 = mul(st - cen2, rot2) + cen2 + hash(vertex2);
//         float2 st3 = mul(st - cen3, rot3) + cen3 + hash(vertex3);
//         // Fetch input.
//         float3 c1 = tex.SampleGrad(samp, frac(st1), mul(dSTdx, rot1), mul(dSTdy, rot1));
//         float3 c2 = tex.SampleGrad(samp, frac(st2), mul(dSTdx, rot2), mul(dSTdy, rot2));
//         float3 c3 = tex.SampleGrad(samp, frac(st3), mul(dSTdx, rot3), mul(dSTdy, rot3));
//         // Use luminance as weight.
//         float3 Lw = float3(0.299, 0.587, 0.114);
//         float3 Dw = float3(dot(c1.xyz, Lw), dot(c2.xyz, Lw), dot(c3.xyz, Lw));
//         Dw = lerp(1.0, Dw, g_fallOffContrast);          // 0.6
//         float3 W = Dw * pow(float3(w1, w2, w3), g_exp); // 7
//         W /= (W.x + W.y + W.z);
//         if (r != 0.5)
//             W = Gain3(W, r);
//         return W.x * c1 + W.y * c2 + W.z * c3;
//     }
//     float texSyn( Texture2D<float> tex, SamplerState samp, float2 dSTdx, float2 dSTdy, float2 st, float rotStrength = 0, float r = 0.5,
//     float g_fallOffContrast = 0.6, float g_exp = 7 )
//     {
//         // Get triangle info.
//         float w1, w2, w3;
//         int2 vertex1, vertex2, vertex3;
//         TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, st);
//         float2x2 rot1 = LoadRot2x2(vertex1, rotStrength);
//         float2x2 rot2 = LoadRot2x2(vertex2, rotStrength);
//         float2x2 rot3 = LoadRot2x2(vertex3, rotStrength);
//         float2 cen1 = MakeCenST(vertex1);
//         float2 cen2 = MakeCenST(vertex2);
//         float2 cen3 = MakeCenST(vertex3);
//         float2 st1 = mul(st - cen1, rot1) + cen1 + hash(vertex1);
//         float2 st2 = mul(st - cen2, rot2) + cen2 + hash(vertex2);
//         float2 st3 = mul(st - cen3, rot3) + cen3 + hash(vertex3);
//         // Fetch input.
//         float c1 = tex.SampleGrad(samp, frac(st1), mul(dSTdx, rot1), mul(dSTdy, rot1));
//         float c2 = tex.SampleGrad(samp, frac(st2), mul(dSTdx, rot2), mul(dSTdy, rot2));
//         float c3 = tex.SampleGrad(samp, frac(st3), mul(dSTdx, rot3), mul(dSTdy, rot3));
//         // Use luminance as weight.
//         float3 Dw = float3(c1, c2, c3);
//         Dw = lerp(1.0, Dw, g_fallOffContrast);          // 0.6
//         float3 W = Dw * pow(float3(w1, w2, w3), g_exp); // 7
//         W /= (W.x + W.y + W.z);
//         if (r != 0.5)
//             W = Gain3(W, r);
//         return W.x * c1 + W.y * c2 + W.z * c3;
//     }
//     float2 MakeCenST(int2 Vertex)
//     {
//         float2x2 invSkewMat = float2x2(1.0, 0.5, 0.0, 1.0 / 1.15470054);
//         return mul(invSkewMat, Vertex) / (2 * sqrt(3));
//     }
//     float2x2 LoadRot2x2(int2 idx, float rotStrength)
//     {
//         float angle = abs(idx.x * idx.y) + abs(idx.x + idx.y) + M_PI;
//         // Remap to +/-pi.
//         angle = fmod(angle, 2 * M_PI);
//         if (angle < 0)
//             angle += 2 * M_PI;
//         if (angle > M_PI)
//             angle -= 2 * M_PI;
//         angle *= rotStrength;
//         float cs = cos(angle), si = sin(angle);
//         return float2x2(cs, -si, si, cs);
//     }
//     float3 ProduceHexWeights(float3 W, int2 vertex1, int2 vertex2, int2 vertex3)
//     {
//         float3 res = 0.0;
//         int v1 = (vertex1.x - vertex1.y) % 3;
//         if (v1 < 0)
//             v1 += 3;
//         int vh = v1 < 2 ? (v1 + 1) : 0;
//         int vl = v1 > 0 ? (v1 - 1) : 2;
//         int v2 = vertex1.x < vertex3.x ? vl : vh;
//         int v3 = vertex1.x < vertex3.x ? vh : vl;
//         res.x = v3 == 0 ? W.z : (v2 == 0 ? W.y : W.x);
//         res.y = v3 == 1 ? W.z : (v2 == 1 ? W.y : W.x);
//         res.z = v3 == 2 ? W.z : (v2 == 2 ? W.y : W.x);
//         return res;
//     }
//     float3 Gain3(float3 x, float r)
//     {
//         // Increase contrast when r > 0.5 and
//         // reduce contrast if less.
//         float k = log(1 - r) / log(0.5);
//         float3 s = 2 * step(0.5, x);
//         float3 m = 2 * (1 - s);
//         float3 res = 0.5 * s + 0.25 * m * pow(max(0.0, s + x * m), k);
//         return res.xyz / (res.x + res.y + res.z);
//     }
// };
