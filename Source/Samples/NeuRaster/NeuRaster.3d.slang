/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

// Full-screen NeuLobes inference with two INDEPENDENT blending toggles, for A/B perf testing:
//
//   NEU_VS_INTERP (MLP weights): 1 = the vertex shader loads each corner probe's weights and the
//       RASTERIZER barycentric-blends them (many hardware interpolants); 0 = the pixel shader fetches
//       the 3 corner probes' weights from the pool buffers and blends them itself (no weight interpolants).
//
//   NEU_FEAT_VS_INTERP (neural feature): 1 = the vertex shader computes each corner probe's feature and
//       the RASTERIZER interpolates it; 0 = the pixel shader fetches the 3 corner probes' features and
//       blends by barycentrics.
//
// The two are orthogonal (any of the 4 combinations is valid). All combinations produce the same image;
// only the work distribution differs. The quad's four corners map to probes via gCornerProbe. Interpolant
// sizes are fixed by the NEU_* macros (injected from the model dimensions on the host).

#include "Utils/Math/MathConstants.slangh"

// --- Compile-time MLP dimensions (injected via program defines; defaults = reference config). ------
#ifndef NEU_PE_BANDS
#define NEU_PE_BANDS 2
#endif
#ifndef NEU_PLANE_DIM
#define NEU_PLANE_DIM 4
#endif
#ifndef NEU_L0_IN_BLK
#define NEU_L0_IN_BLK 4
#endif
#ifndef NEU_L0_OUT_BLK
#define NEU_L0_OUT_BLK 1
#endif
#ifndef NEU_L1_IN_BLK
#define NEU_L1_IN_BLK 1
#endif
#ifndef NEU_L1_OUT_BLK
#define NEU_L1_OUT_BLK 1
#endif
#ifndef NEU_L2_IN_BLK
#define NEU_L2_IN_BLK 1
#endif
#ifndef NEU_L2_OUT_BLK
#define NEU_L2_OUT_BLK 1
#endif

// Feature-plane storage backend: 0 = StructuredBuffer with manual bilinear interp (baseline),
// 1 = Texture2D with hardware linear filtering. The texture path requires NEU_PLANE_DIM == 4
// (the plane channels are packed into one RGBA texel); the host only enables it in that case.
#ifndef NEU_USE_TEXTURES
#define NEU_USE_TEXTURES 0
#endif

// Weight-blending strategy: 1 = interpolate MLP weights in the vertex shader; 0 = blend in the pixel shader.
#ifndef NEU_VS_INTERP
#define NEU_VS_INTERP 1
#endif

// Feature-fetch location: 1 = compute the feature in the vertex shader and interpolate it (rasterizer
// blends), 0 = fetch the 3 corner probes' features in the pixel shader and blend by barycentrics.
#ifndef NEU_FEAT_VS_INTERP
#define NEU_FEAT_VS_INTERP 0
#endif

// Derived sizes. Each 4x4 weight block is 4 float4 (rows); each bias block is 1 float4.
#define NEU_W0_VEC4 (4 * NEU_L0_IN_BLK * NEU_L0_OUT_BLK)
#define NEU_W1_VEC4 (4 * NEU_L1_IN_BLK * NEU_L1_OUT_BLK)
#define NEU_W2_VEC4 (4 * NEU_L2_IN_BLK * NEU_L2_OUT_BLK)
#define NEU_FEAT_VEC4 ((NEU_PLANE_DIM + 3) / 4)
#define NEU_INPUT_DIM (6 * NEU_PE_BANDS + NEU_PLANE_DIM)
#define NEU_L0_IN_FLOATS (4 * NEU_L0_IN_BLK)

cbuffer NeuCB
{
    float3 gInputDir;    ///< Query direction fed to the MLP (already normalized on the host).
    uint gIterations;    ///< Inference repeats per pixel (throughput scaling for the perf test).
    uint4 gCornerProbe;  ///< Probe pool index for quad corners 0..3 (blended by corner).
    uint gRes;           ///< Feature-plane resolution (runtime; must equal the model's res).
    uint gRank;          ///< Low-rank factor count (runtime; must equal the model's rank).
}

// NeuLobes weight tensors (same layout as PTTest::loadNeuLobesModel).
#if NEU_USE_TEXTURES
// Feature planes as textures: width = res, height = numMLPs*rank (one row per (probe,rank) pair).
// gNeuVecXTex is scalar; gNeuMatYZTex packs the NEU_PLANE_DIM(=4) channels into RGBA.
Texture2D<float> gNeuVecXTex;      ///< [numMLPs*rank] rows x [res] cols (u axis).
Texture2D<float4> gNeuMatYZTex;    ///< [numMLPs*rank] rows x [res] cols (v axis), RGBA = planeDim.
SamplerState gNeuFeatSampler;      ///< Linear filtering, clamp addressing.
#else
StructuredBuffer<float> gNeuVecX;      ///< [numMLPs][rank][res].
StructuredBuffer<float> gNeuMatYZ;     ///< [numMLPs][rank][res][planeDim].
#endif
StructuredBuffer<float4x4> gNeuW0;     ///< Layer 0 weights, [numMLPs][outBlk][inBlk] 4x4 blocks.
StructuredBuffer<float4x4> gNeuW1;     ///< Layer 1 weights.
StructuredBuffer<float4x4> gNeuW2;     ///< Layer 2 weights.
StructuredBuffer<float4> gNeuB0;       ///< Layer 0 biases, [numMLPs][outBlk] float4 blocks.
StructuredBuffer<float4> gNeuB1;       ///< Layer 1 biases.
StructuredBuffer<float4> gNeuB2;       ///< Layer 2 biases.

float4 neuLeakyRelu(float4 v)
{
    return max(v, 0.0f) + 0.01f * min(v, 0.0f);
}

// Direction -> equal-area octahedral UV in [0,1]^2 (inverse map, matches the export spec).
float2 neuDirToOctUV(float3 dir)
{
    float3 d = normalize(dir);
    float r = sqrt(max(1.0f - abs(d.z), 0.0f));
    float phi = atan2(abs(d.y), abs(d.x)); // [0, pi/2]
    float rt = r * (4.0f * phi / M_PI - 1.0f);
    float s = (d.z >= 0.0f) ? r : (2.0f - r);
    float apx = max((s - rt) * 0.5f, 0.0f);
    float apy = max((s + rt) * 0.5f, 0.0f);
    float px = (d.x >= 0.0f) ? apx : -apx; // copysign(apx, d.x)
    float py = (d.y >= 0.0f) ? apy : -apy; // copysign(apy, d.y)
    return 0.5f * (float2(px, py) + 1.0f);
}

// Shared per-fragment octahedral sample coordinates (independent of which probe is sampled).
struct NeuOctSetup
{
#if NEU_USE_TEXTURES
    float su;   ///< Normalized u sample coord for gNeuVecXTex / MatYZ (x axis).
    float sv;   ///< Normalized v sample coord for gNeuMatYZTex (x axis).
    float invH; ///< 1 / texture height (rows = numMLPs*rank).
#else
    int i0, i1, j0, j1; ///< Bilinear texel indices.
    float fu, fv;       ///< Bilinear fractions.
#endif
};

// Compute the direction-dependent sample coords once (octahedral map + bilinear setup). Shared by all
// probes at a fragment, so the FS blend computes this once instead of per probe.
NeuOctSetup neuOctSetup(float3 dir)
{
    const uint res = gRes;
    float2 uv = neuDirToOctUV(dir);
    float tu = uv.x * (res - 1);
    float tv = uv.y * (res - 1);

    NeuOctSetup s;
#if NEU_USE_TEXTURES
    // Texel-center convention: sampling at (t+0.5)/res reproduces lerp(texel_i0, texel_i0+1, frac(t)),
    // matching the manual clamp-to-edge interpolation of the buffer path.
    uint dimX, dimY;
    gNeuMatYZTex.GetDimensions(dimX, dimY);
    float invW = 1.0f / (float)res;
    s.su = (tu + 0.5f) * invW;
    s.sv = (tv + 0.5f) * invW;
    s.invH = 1.0f / (float)dimY;
#else
    s.i0 = (int)clamp(floor(tu), 0.0f, (float)(res - 1));
    s.i1 = min(s.i0 + 1, (int)res - 1);
    s.fu = clamp(tu - s.i0, 0.0f, 1.0f);
    s.j0 = (int)clamp(floor(tv), 0.0f, (float)(res - 1));
    s.j1 = min(s.j0 + 1, (int)res - 1);
    s.fv = clamp(tv - s.j0, 0.0f, 1.0f);
#endif
    return s;
}

// Accumulate weight * (probe p's low-rank feature) into featS, using precomputed sample coords.
// The rank loop is outer so the VecX term (independent of channel) is fetched once per rank, not per
// channel; the barycentric weight is fused in so callers avoid per-probe temp arrays.
void neuFeatureAccum(uint p, NeuOctSetup s, float weight, inout float featS[NEU_PLANE_DIM])
{
    const uint rank = gRank;
#if NEU_USE_TEXTURES
    // The row (y) coordinate is point-sampled (row+0.5)/H so the Linear sampler never blends across rows.
    float4 acc = float4(0.0f);
    for (uint r = 0; r < rank; ++r)
    {
        float yv = ((float)(p * rank + r) + 0.5f) * s.invH;
        float vx = gNeuVecXTex.SampleLevel(gNeuFeatSampler, float2(s.su, yv), 0);
        float4 my = gNeuMatYZTex.SampleLevel(gNeuFeatSampler, float2(s.sv, yv), 0);
        acc += vx * my;
    }
    [unroll]
    for (uint c = 0; c < NEU_PLANE_DIM; ++c)
        featS[c] += weight * acc[c];
#else
    const uint res = gRes;
    for (uint r = 0; r < rank; ++r)
    {
        uint base = (p * rank + r) * res;
        float vx = lerp(gNeuVecX[base + s.i0], gNeuVecX[base + s.i1], s.fu); // once per rank
        [unroll]
        for (uint c = 0; c < NEU_PLANE_DIM; ++c)
        {
            float my = lerp(
                gNeuMatYZ[(base + s.j0) * NEU_PLANE_DIM + c],
                gNeuMatYZ[(base + s.j1) * NEU_PLANE_DIM + c],
                s.fv);
            featS[c] += weight * vx * my;
        }
    }
#endif
}

// Low-rank feature for a single probe `p` at direction `dir` (used by the VS-interp feature path).
void neuFeature(uint p, float3 dir, out float featS[NEU_PLANE_DIM])
{
    [unroll]
    for (uint c = 0; c < NEU_PLANE_DIM; ++c)
        featS[c] = 0.0f;
    neuFeatureAccum(p, neuOctSetup(dir), 1.0f, featS);
}

// Build the layer-0 input blocks: concat(positional encoding of dir, feature), zero-padded.
void neuBuildInput(float3 dir, float featS[NEU_PLANE_DIM], out float4 h0[NEU_L0_IN_BLK])
{
    float inp[NEU_L0_IN_FLOATS];
    [unroll]
    for (uint z = 0; z < NEU_L0_IN_FLOATS; ++z)
        inp[z] = 0.f;

    float3 d = normalize(dir);
    float axis[3] = {d.x, d.y, d.z};
    uint idx = 0;
    [unroll]
    for (uint a = 0; a < 3; ++a)
    {
        [unroll]
        for (uint kb = 0; kb < NEU_PE_BANDS; ++kb)
            inp[idx++] = sin(M_PI * axis[a] * exp2((float)kb));
        [unroll]
        for (uint kb = 0; kb < NEU_PE_BANDS; ++kb)
            inp[idx++] = cos(M_PI * axis[a] * exp2((float)kb));
    }
    [unroll]
    for (uint fc = 0; fc < NEU_PLANE_DIM; ++fc)
        inp[idx++] = featS[fc];

    [unroll]
    for (uint k = 0; k < NEU_L0_IN_BLK; ++k)
        h0[k] = float4(inp[4 * k + 0], inp[4 * k + 1], inp[4 * k + 2], inp[4 * k + 3]);
}

// Output lanes are mu-law compressed radiance; decompress with mu = 16.
float3 neuDecode(float4 h3last)
{
    const float mu = 16.0f;
    return (pow(float3(1.0f + mu), h3last.xyz) - 1.0f) / mu;
}

// One MLP layer applied to interpolated (already bary-blended) weights held in registers.
#define NEU_MATVEC_LAYER(WARR, BARR, INBLK, OUTBLK, HIN, HOUT)                           \
    {                                                                                    \
        [unroll] for (uint _oi = 0; _oi < (OUTBLK); ++_oi)                               \
        {                                                                                \
            float4 _acc = BARR[_oi];                                                     \
            [unroll] for (uint _ii = 0; _ii < (INBLK); ++_ii)                            \
            {                                                                            \
                uint _k = (_oi * (INBLK) + _ii) * 4;                                     \
                float4x4 _Wm = float4x4(WARR[_k + 0], WARR[_k + 1], WARR[_k + 2], WARR[_k + 3]); \
                _acc += mul(_Wm, HIN[_ii]);                                              \
            }                                                                            \
            HOUT[_oi] = neuLeakyRelu(_acc);                                              \
        }                                                                                \
    }

// One MLP layer that fetches the three corner probes' weights from the pool buffers and blends them
// by barycentric weights in the pixel shader (the PTTest-style manual blend).
#define NEU_MATVEC_LAYER_PS(WBUF, BBUF, INBLK, OUTBLK, P0, P1, P2, BARY, HIN, HOUT)      \
    {                                                                                    \
        [unroll] for (uint _oi = 0; _oi < (OUTBLK); ++_oi)                               \
        {                                                                                \
            float4 _acc = BARY.x * BBUF[(P0) * (OUTBLK) + _oi]                           \
                        + BARY.y * BBUF[(P1) * (OUTBLK) + _oi]                           \
                        + BARY.z * BBUF[(P2) * (OUTBLK) + _oi];                          \
            [unroll] for (uint _ii = 0; _ii < (INBLK); ++_ii)                            \
            {                                                                            \
                float4x4 _Wm = BARY.x * WBUF[((P0) * (OUTBLK) + _oi) * (INBLK) + _ii]    \
                             + BARY.y * WBUF[((P1) * (OUTBLK) + _oi) * (INBLK) + _ii]    \
                             + BARY.z * WBUF[((P2) * (OUTBLK) + _oi) * (INBLK) + _ii];   \
                _acc += mul(_Wm, HIN[_ii]);                                              \
            }                                                                            \
            HOUT[_oi] = neuLeakyRelu(_acc);                                              \
        }                                                                                \
    }

// Copy probe P's weights/biases into per-vertex interpolant arrays (each 4x4 block as its 4 rows).
#define NEU_LOAD_WEIGHTS(WBUF, BBUF, INBLK, OUTBLK, WARR, BARR, P)                       \
    {                                                                                    \
        const uint _n = (INBLK) * (OUTBLK);                                              \
        [unroll] for (uint _k = 0; _k < _n; ++_k)                                        \
        {                                                                                \
            float4x4 _m = WBUF[(P) * _n + _k];                                           \
            WARR[_k * 4 + 0] = _m[0];                                                    \
            WARR[_k * 4 + 1] = _m[1];                                                    \
            WARR[_k * 4 + 2] = _m[2];                                                    \
            WARR[_k * 4 + 3] = _m[3];                                                    \
        }                                                                                \
        [unroll] for (uint _b = 0; _b < (OUTBLK); ++_b)                                  \
            BARR[_b] = BBUF[(P) * (OUTBLK) + _b];                                         \
    }

struct VSIn
{
    float2 pos : POSITION;
    uint corner : CORNER; ///< Quad corner index 0..3 (selects the probe via gCornerProbe).
    float3 bary : BARY;   ///< This vertex's barycentric basis within its triangle.
};

struct VSOut
{
    float4 pos : SV_POSITION;
    float3 bary : BARYW; ///< Interpolated barycentric coords (used by the PS-blend paths).
#if NEU_VS_INTERP
    float4 wA[NEU_W0_VEC4] : WLA;     ///< Layer 0 weight rows.
    float4 bA[NEU_L0_OUT_BLK] : BLA;  ///< Layer 0 biases.
    float4 wB[NEU_W1_VEC4] : WLB;     ///< Layer 1 weight rows.
    float4 bB[NEU_L1_OUT_BLK] : BLB;  ///< Layer 1 biases.
    float4 wC[NEU_W2_VEC4] : WLC;     ///< Layer 2 weight rows.
    float4 bC[NEU_L2_OUT_BLK] : BLC;  ///< Layer 2 biases.
#endif
#if NEU_FEAT_VS_INTERP
    float4 feat[NEU_FEAT_VEC4] : FEAT; ///< Neural feature (planeDim channels, zero-padded).
#endif
};

VSOut vsMain(VSIn vIn)
{
    VSOut o;
    o.pos = float4(vIn.pos, 0.f, 1.f);
    o.bary = vIn.bary;

#if NEU_VS_INTERP || NEU_FEAT_VS_INTERP
    // This corner's probe; used for whichever quantities are emitted as interpolants below.
    const uint p = gCornerProbe[min(vIn.corner, 3u)];
#endif

#if NEU_VS_INTERP
    // Emit this corner probe's weights; the rasterizer barycentric-blends them.
    NEU_LOAD_WEIGHTS(gNeuW0, gNeuB0, NEU_L0_IN_BLK, NEU_L0_OUT_BLK, o.wA, o.bA, p);
    NEU_LOAD_WEIGHTS(gNeuW1, gNeuB1, NEU_L1_IN_BLK, NEU_L1_OUT_BLK, o.wB, o.bB, p);
    NEU_LOAD_WEIGHTS(gNeuW2, gNeuB2, NEU_L2_IN_BLK, NEU_L2_OUT_BLK, o.wC, o.bC, p);
#endif

#if NEU_FEAT_VS_INTERP
    // Compute this corner probe's feature and emit it; the rasterizer barycentric-blends it.
    float featS[NEU_PLANE_DIM];
    neuFeature(p, gInputDir, featS);
    [unroll]
    for (uint k = 0; k < NEU_FEAT_VEC4; ++k)
    {
        float4 v = float4(0.f);
        [unroll]
        for (uint e = 0; e < 4; ++e)
        {
            uint c = k * 4 + e;
            if (c < NEU_PLANE_DIM)
                v[e] = featS[c];
        }
        o.feat[k] = v;
    }
#endif

    return o;
}

// Unified evaluation. The feature is either read from a VS interpolant (NEU_FEAT_VS_INTERP) or blended
// here from the 3 corner probes; the MLP weights are either read from VS interpolants (NEU_VS_INTERP)
// or fetched+blended here. `probes` are the triangle's 3 corner probes; `i.bary` are the barycentrics.
float3 neuEval(VSOut i, uint3 probes, float3 dir)
{
    // --- Feature ---
    float featS[NEU_PLANE_DIM];
#if NEU_FEAT_VS_INTERP
    [unroll]
    for (uint c = 0; c < NEU_PLANE_DIM; ++c)
        featS[c] = i.feat[c / 4][c % 4];
#else
    // Fetch + blend the 3 corner probes' features in the pixel shader. The octahedral/bilinear setup
    // is direction-only, so compute it once and accumulate each probe weighted by its barycentric.
    [unroll]
    for (uint c = 0; c < NEU_PLANE_DIM; ++c)
        featS[c] = 0.0f;
    NeuOctSetup setup = neuOctSetup(dir);
    neuFeatureAccum(probes.x, setup, i.bary.x, featS);
    neuFeatureAccum(probes.y, setup, i.bary.y, featS);
    neuFeatureAccum(probes.z, setup, i.bary.z, featS);
#endif

    float4 h0[NEU_L0_IN_BLK];
    neuBuildInput(dir, featS, h0);

    // --- MLP layers ---
    float4 h1[NEU_L0_OUT_BLK];
    float4 h2[NEU_L1_OUT_BLK];
    float4 h3[NEU_L2_OUT_BLK];
#if NEU_VS_INTERP
    NEU_MATVEC_LAYER(i.wA, i.bA, NEU_L0_IN_BLK, NEU_L0_OUT_BLK, h0, h1);
    NEU_MATVEC_LAYER(i.wB, i.bB, NEU_L1_IN_BLK, NEU_L1_OUT_BLK, h1, h2);
    NEU_MATVEC_LAYER(i.wC, i.bC, NEU_L2_IN_BLK, NEU_L2_OUT_BLK, h2, h3);
#else
    NEU_MATVEC_LAYER_PS(gNeuW0, gNeuB0, NEU_L0_IN_BLK, NEU_L0_OUT_BLK, probes.x, probes.y, probes.z, i.bary, h0, h1);
    NEU_MATVEC_LAYER_PS(gNeuW1, gNeuB1, NEU_L1_IN_BLK, NEU_L1_OUT_BLK, probes.x, probes.y, probes.z, i.bary, h1, h2);
    NEU_MATVEC_LAYER_PS(gNeuW2, gNeuB2, NEU_L2_IN_BLK, NEU_L2_OUT_BLK, probes.x, probes.y, probes.z, i.bary, h2, h3);
#endif
    return neuDecode(h3[0]);
}

float4 psMain(VSOut i, uint primID : SV_PrimitiveID) : SV_TARGET
{
    const uint iters = max(gIterations, 1u);
    float3 color;
    float sink = 0.f;

    // The quad is two triangles: prim 0 = corners {0,1,2}, prim 1 = corners {0,2,3}. bary.{x,y,z}
    // weight the triangle's local vertices in that order, matching the CPU-assigned barycentric basis.
    // probes are needed whenever the feature or the weights are blended in the pixel shader.
    uint3 corners = (primID == 0u) ? uint3(0, 1, 2) : uint3(0, 2, 3);
    uint3 probes = uint3(gCornerProbe[corners.x], gCornerProbe[corners.y], gCornerProbe[corners.z]);

    color = neuEval(i, probes, gInputDir);
    [loop]
    for (uint it = 1u; it < iters; ++it)
    {
        float3 c = neuEval(i, probes, gInputDir + float(it) * 1e-4f);
        sink += c.x + c.y + c.z;
    }

    // Keep the extra perf-loop work alive (see notes) without visibly changing the displayed color.
    color += sink * 1e-20f;
    return float4(color, 1.f);
}
