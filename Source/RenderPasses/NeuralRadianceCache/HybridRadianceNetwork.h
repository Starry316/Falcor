#pragma once
// Joint geodesic position-direction encoding (Weier et al., arXiv:2603.05079)
// and spatially interpolated MLP parameters, adapted from neulobes'
// HashSphereHemiMLP. Spatial banks live on a regular world-space grid rather
// than mesh vertices, so the cache works at arbitrary ray intersections.
#include <cstdlib>
#include <fstream>
#include <map>
#include <random>
#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/network.h>

namespace nirc_hybrid
{
constexpr int Levels = 8, Features = 2, Input = Levels * Features + 6;
constexpr int Hidden = 8;
constexpr int Layer0 = (Input + 1) * Hidden;
constexpr int Layer1 = (Hidden + 1) * Hidden;
constexpr int MlpParams = Layer0 + Layer1 + (Hidden + 1) * 3;

struct Vec
{
    float x, y, z;
};
__host__ __device__ inline Vec add(Vec a, Vec b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__host__ __device__ inline Vec cross(Vec a, Vec b)
{
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__host__ __device__ inline float dot(Vec a, Vec b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec unit(Vec v)
{
    float r = 1.f / sqrtf(dot(v, v));
    return {v.x * r, v.y * r, v.z * r};
}
struct Face
{
    Vec a, b, c;
    uint32_t id[3];
    uint32_t hash[3];
};
struct Settings
{
    int cells, banks, mlpCount;
    bool tetrahedral;
    // Experimental MLP-only override; hash-feature interpolation is unchanged.
    bool mlpTrilinear = false;
    uint32_t offset[Levels], size[Levels];
    uint32_t faceOffset[4];
    const Face* faces;
};
struct Inputs
{
    const float* data[5];
    const int* map[5];
    int stride[5];
    __device__ const float* at(int channel, int sample) const
    {
        return data[channel] + (map[channel] ? map[channel][sample] : sample) * stride[channel];
    }
};
struct Banks
{
    int id[8];
    float w[8];
    int count;
};

struct Tetrahedron
{
    int corner[4];
    float weight[4];
};
__host__ __device__ inline Tetrahedron tetrahedron(float x, float y, float z)
{
    // Freudenthal triangulation: sort fractional coordinates and follow the
    // cube's 000 -> 111 diagonal. Adjacent tetrahedra share vertex values,
    // making the interpolation continuous, including across cube faces.
    float a = x, b = y, c = z;
    int ia = 1, ib = 2, ic = 4;
    if (a < b)
    {
        float t = a;
        a = b;
        b = t;
        int j = ia;
        ia = ib;
        ib = j;
    }
    if (b < c)
    {
        float t = b;
        b = c;
        c = t;
        int j = ib;
        ib = ic;
        ic = j;
    }
    if (a < b)
    {
        float t = a;
        a = b;
        b = t;
        int j = ia;
        ia = ib;
        ib = j;
    }
    return {{0, ia, ia | ib, 7}, {1 - a, a - b, b - c, c}};
}

__device__ inline Vec oct(float x, float y)
{
    Vec n{x, y, 1 - fabsf(x) - fabsf(y)};
    if (n.z < 0)
    {
        n.x = (1 - fabsf(y)) * (x >= 0 ? 1.f : -1.f);
        n.y = (1 - fabsf(x)) * (y >= 0 ? 1.f : -1.f);
    }
    return unit(n);
}
__device__ inline Vec bary(const Face& f, Vec d)
{
    Vec w{dot(d, cross(f.b, f.c)), dot(d, cross(f.c, f.a)), dot(d, cross(f.a, f.b))};
    float sum = w.x + w.y + w.z;
    return {w.x / sum, w.y / sum, w.z / sum};
}
__device__ inline int locate(const Settings& cfg, Vec dir, int depth, Vec& b)
{
    // Test oriented cone planes, not normalized barycentrics: antipodal faces
    // have three negative coefficients and must never qualify as a hit.
    int face = 0;
    float best = -1.e30f;
    for (int i = 0; i < 20; ++i)
    {
        const auto& f = cfg.faces[i];
        float score = fminf(dot(dir, cross(f.b, f.c)), fminf(dot(dir, cross(f.c, f.a)), dot(dir, cross(f.a, f.b))));
        if (score > best)
        {
            best = score;
            face = i;
        }
    }
    for (int l = 0; l < depth; ++l)
    {
        // Reprojection changes midpoint locations. Test all four spherical
        // child cones instead of using a flat-parent beta>=0.5 shortcut.
        int selected = 0;
        best = -1.e30f;
        for (int k = 0; k < 4; ++k)
        {
            const auto& f = cfg.faces[cfg.faceOffset[l + 1] + 4 * face + k];
            float score = fminf(dot(dir, cross(f.b, f.c)), fminf(dot(dir, cross(f.c, f.a)), dot(dir, cross(f.a, f.b))));
            if (score > best)
            {
                best = score;
                selected = k;
            }
        }
        face = 4 * face + selected;
    }
    int index = cfg.faceOffset[depth] + face;
    b = bary(cfg.faces[index], dir);
    b = {fmaxf(b.x, 0.f), fmaxf(b.y, 0.f), fmaxf(b.z, 0.f)};
    float sum = b.x + b.y + b.z;
    b = {b.x / sum, b.y / sum, b.z / sum};
    return index;
}
template<bool Tetra = false>
__device__ inline Banks banksAtImpl(const Settings& cfg, Vec p)
{
    Banks b{};
    b.count = cfg.cells ? (Tetra ? 4 : 8) : 1;
    if (!cfg.cells)
    {
        b.id[0] = 0;
        b.w[0] = 1;
        return b;
    }
    float x = fminf(fmaxf(p.x, 0.f), 1.f) * cfg.cells;
    float y = fminf(fmaxf(p.y, 0.f), 1.f) * cfg.cells;
    float z = fminf(fmaxf(p.z, 0.f), 1.f) * cfg.cells;
    int ix = min(int(x), cfg.cells - 1), iy = min(int(y), cfg.cells - 1), iz = min(int(z), cfg.cells - 1);
    x -= ix;
    y -= iy;
    z -= iz;
    int n = cfg.cells + 1;
    Tetrahedron tetra;
    if constexpr (Tetra)
        tetra = tetrahedron(x, y, z);
    for (int k = 0; k < (Tetra ? 4 : 8); ++k)
    {
        int corner = Tetra ? tetra.corner[k] : k;
        int a = corner & 1, c = (corner >> 1) & 1, d = corner >> 2;
        b.id[k] = ix + a + n * (iy + c + n * (iz + d));
        b.w[k] = Tetra ? tetra.weight[k] : (a ? x : 1 - x) * (c ? y : 1 - y) * (d ? z : 1 - z);
    }
    return b;
}
__device__ inline Banks banksAt(const Settings& cfg, Vec p)
{
    return cfg.tetrahedral && !cfg.mlpTrilinear ? banksAtImpl<true>(cfg, p) : banksAtImpl<false>(cfg, p);
}
__device__ inline float blend(const __half* p, const Banks& b, int index)
{
    float v = 0;
    for (int k = 0; k < b.count; ++k)
        v += b.w[k] * float(p[b.id[k] * MlpParams + index]);
    return v;
}
__device__ inline void scatter(float* g, const Banks& b, int index, float v)
{
    if (v == 0)
        return;
    for (int k = 0; k < b.count; ++k)
        if (b.w[k] != 0)
            atomicAdd(g + b.id[k] * MlpParams + index, b.w[k] * v);
}
template<bool Tetra>
__device__ inline float featureImpl(
    const Settings& cfg,
    const __half* params,
    Vec p,
    Vec d,
    int level,
    int channel,
    float* gradients = nullptr,
    float upstream = 0
)
{
    Vec beta;
    int face = locate(cfg, d, level / 2, beta);
    const Face& f = cfg.faces[face];
    const int n = 4 << level;
    float x = fminf(fmaxf(p.x, 0.f), 1.f) * (n - 1), y = fminf(fmaxf(p.y, 0.f), 1.f) * (n - 1), z = fminf(fmaxf(p.z, 0.f), 1.f) * (n - 1);
    int ix = min(int(x), n - 2), iy = min(int(y), n - 2), iz = min(int(z), n - 2);
    x -= ix;
    y -= iy;
    z -= iz;
    uint64_t denseSize = uint64_t(n) * n * n * (10 * (1u << (2 * (level / 2))) + 2);
    float result = 0;
    Tetrahedron tetra;
    if constexpr (Tetra)
        tetra = tetrahedron(x, y, z);
    for (int k = 0; k < (Tetra ? 4 : 8); ++k)
    {
        int corner = Tetra ? tetra.corner[k] : k;
        uint32_t a = ix + (corner & 1), b = iy + ((corner >> 1) & 1), c = iz + (corner >> 2);
        float w = Tetra ? tetra.weight[k] : ((k & 1) ? x : 1 - x) * ((k & 2) ? y : 1 - y) * ((k & 4) ? z : 1 - z);
        for (int v = 0; v < 3; ++v)
        {
            uint32_t slot;
            if (denseSize <= cfg.size[level])
                slot = (a + n * (b + n * c)) * (10 * (1u << (2 * (level / 2))) + 2) + f.id[v];
            else
                slot = ((a * 1u) ^ (b * 2654435761u) ^ (c * 805459861u) ^ f.hash[v]) % cfg.size[level];
            uint32_t index = cfg.offset[level] + slot * Features + channel;
            float weight = w * (v == 0 ? beta.x : v == 1 ? beta.y : beta.z);
            if (gradients)
            {
                if (weight * upstream != 0)
                    atomicAdd(gradients + index, weight * upstream);
            }
            else
                result += weight * float(params[index]);
        }
    }
    return result;
}
__device__ inline float feature(
    const Settings& cfg,
    const __half* params,
    Vec p,
    Vec d,
    int level,
    int channel,
    float* gradients = nullptr,
    float upstream = 0
)
{
    return cfg.tetrahedral ? featureImpl<true>(cfg, params, p, d, level, channel, gradients, upstream)
                           : featureImpl<false>(cfg, params, p, d, level, channel, gradients, upstream);
}
__device__ inline float layer(const __half* p, const Banks& b, float x, int in, int out, int offset)
{
    int lane = threadIdx.x & 31;
    float y = 0;
    for (int k = 0; k <= in; ++k)
    {
        float value = k == in ? 1.f : __shfl_sync(0xffffffff, x, k);
        if (lane < out)
            y += blend(p, b, offset + k * out + lane) * value;
    }
    return y;
}
__device__ inline float backwardLayer(const __half* p, float* g, const Banks& b, float x, float dy, int in, int out, int offset)
{
    int lane = threadIdx.x & 31;
    float dx = 0;
    for (int j = 0; j < out; ++j)
    {
        float delta = __shfl_sync(0xffffffff, dy, j);
        if (lane < in)
            dx += blend(p, b, offset + lane * out + j) * delta;
    }
    for (int k = 0; k <= in; ++k)
    {
        float value = k == in ? 1.f : __shfl_sync(0xffffffff, x, k);
        if (lane < out)
            scatter(g, b, offset + k * out + lane, value * dy);
    }
    return dx;
}
__device__ inline float activate(float x)
{
    return x >= 0 ? x : .01f * x;
}
__device__ inline float derivative(float x)
{
    return x >= 0 ? 1.f : .01f;
}

template<bool Backward>
__global__ void evaluate(
    Settings cfg,
    Inputs inputs,
    int count,
    const __half* params,
    __half* output,
    int outputStride,
    float* prediction,
    tcnn::FusedOutputData fused,
    const __half* dy,
    float* gradients
)
{
    int sample = (blockIdx.x * blockDim.x + threadIdx.x) / 32, lane = threadIdx.x & 31;
    if (sample >= count)
        return;
    const float* pos = inputs.at(1, sample);
    Vec p{pos[0], pos[1], pos[2]};
    // Preserve the research fork's half2 order (low half first) for both modes.
    float2 uv = __half22float2(*reinterpret_cast<const __half2*>(inputs.at(4, sample)));
    Vec direction = oct(uv.x, uv.y);
    Banks banks = banksAt(cfg, p);
    float x = 0;
    if (lane < Levels * Features)
        x = feature(cfg, params, p, direction, lane / Features, lane % Features);
    else if (lane < Input)
    {
        float2 normal = __half22float2(*reinterpret_cast<const __half2*>(inputs.at(0, sample)));
        int a = lane - Levels * Features;
        x = a == 0 ? normal.x : a == 1 ? normal.y : a == 2 ? *inputs.at(2, sample) : inputs.at(3, sample)[a - 3];
    }
    float h0 = activate(layer(params, banks, x, Input, Hidden, 0));
    float h1 = activate(layer(params, banks, h0, Hidden, Hidden, Layer0));
    float y = layer(params, banks, h1, Hidden, 3, Layer0 + Layer1);
    if constexpr (Backward)
    {
        float delta = lane < 3 ? float(dy[sample * outputStride + lane]) : 0;
        float d1 = backwardLayer(params, gradients, banks, h1, delta, Hidden, 3, Layer0 + Layer1) * derivative(h1);
        float d0 = backwardLayer(params, gradients, banks, h0, d1, Hidden, Hidden, Layer0) * derivative(h0);
        float df = backwardLayer(params, gradients, banks, x, d0, Input, Hidden, 0);
        if (lane < Levels * Features)
            feature(cfg, params, p, direction, lane / Features, lane % Features, gradients, df);
    }
    else
    {
        if (output && lane < outputStride)
            output[sample * outputStride + lane] = __float2half(lane < 3 ? y : 0);
        if (prediction && lane < 3)
        {
            if (fused.arr0)
            {
                int id = reinterpret_cast<const uint32_t*>(fused.arr1)[sample];
                float weight = reinterpret_cast<const float*>(fused.arr0)[sample * 3 + lane];
                atomicAdd(prediction + 3 * id + lane, fmaxf(y, 0.f) * weight);
            }
            else
                prediction[sample * 3 + lane] = y;
        }
    }
}
__global__ void castGradients(size_t n, const float* source, __half* target)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        target[i] = __float2half(source[i]);
}

// Inference uses eight lanes per query. Share the spherical search across
// those lanes and reuse its four depths for all sixteen encoded features.
// Training retains the reference path above, including its gradient ordering.
__device__ inline float coneScore(const Face& f, Vec d)
{
    return fminf(dot(d, cross(f.b, f.c)), fminf(dot(d, cross(f.c, f.a)), dot(d, cross(f.a, f.b))));
}
__device__ inline int bestFace8(unsigned mask, float score, int id)
{
    // Resolve exact edge ties in the same order as locate().
#pragma unroll
    for (int delta = 4; delta; delta >>= 1)
    {
        float other = __shfl_xor_sync(mask, score, delta, 8);
        int otherId = __shfl_xor_sync(mask, id, delta, 8);
        if (other > score || (other == score && otherId < id))
        {
            score = other;
            id = otherId;
        }
    }
    return id;
}
template<bool Tetra>
__device__ inline float lookupFeatureImpl(const Settings& cfg, const __half* params, Vec p, Vec direction, int face, int level, int channel)
{
    const Face& f = cfg.faces[face];
    Vec beta = bary(f, direction);
    beta = {fmaxf(beta.x, 0.f), fmaxf(beta.y, 0.f), fmaxf(beta.z, 0.f)};
    float sum = beta.x + beta.y + beta.z;
    beta = {beta.x / sum, beta.y / sum, beta.z / sum};
    int n = 4 << level;
    float x = fminf(fmaxf(p.x, 0.f), 1.f) * (n - 1), y = fminf(fmaxf(p.y, 0.f), 1.f) * (n - 1), z = fminf(fmaxf(p.z, 0.f), 1.f) * (n - 1);
    int ix = min(int(x), n - 2), iy = min(int(y), n - 2), iz = min(int(z), n - 2);
    x -= ix;
    y -= iy;
    z -= iz;
    uint32_t vertices = 10 * (1u << (2 * (level / 2))) + 2;
    bool dense = uint64_t(n) * n * n * vertices <= cfg.size[level];
    float value = 0;
    Tetrahedron tetra;
    if constexpr (Tetra)
        tetra = tetrahedron(x, y, z);
#pragma unroll
    for (int k = 0; k < (Tetra ? 4 : 8); ++k)
    {
        int corner = Tetra ? tetra.corner[k] : k;
        uint32_t a = ix + (corner & 1), b = iy + ((corner >> 1) & 1), c = iz + (corner >> 2);
        float w = Tetra ? tetra.weight[k] : ((k & 1) ? x : 1 - x) * ((k & 2) ? y : 1 - y) * ((k & 4) ? z : 1 - z);
#pragma unroll
        for (int v = 0; v < 3; ++v)
        {
            // Non-dense tables are allocated at exactly a power of two.
            uint32_t slot = dense ? (a + n * (b + n * c)) * vertices + f.id[v]
                                  : ((a * 1u) ^ (b * 2654435761u) ^ (c * 805459861u) ^ f.hash[v]) & (cfg.size[level] - 1);
            float weight = w * (v == 0 ? beta.x : v == 1 ? beta.y : beta.z);
            value += weight * float(params[cfg.offset[level] + slot * Features + channel]);
        }
    }
    return value;
}
__device__ inline float lookupFeature(const Settings& cfg, const __half* params, Vec p, Vec direction, int face, int level, int channel)
{
    return cfg.tetrahedral ? lookupFeatureImpl<true>(cfg, params, p, direction, face, level, channel)
                           : lookupFeatureImpl<false>(cfg, params, p, direction, face, level, channel);
}
// Vector loads require the even hash offsets established in Network.
// Each feature retains the scalar kernel's accumulation order.
__device__ inline Vec directionWeights(const Face& f, Vec direction)
{
    Vec beta = bary(f, direction);
    beta = {fmaxf(beta.x, 0.f), fmaxf(beta.y, 0.f), fmaxf(beta.z, 0.f)};
    float sum = beta.x + beta.y + beta.z;
    return {beta.x / sum, beta.y / sum, beta.z / sum};
}
template<bool Tetra>
__device__ inline float2 lookupFeaturePairWithWeights(
    const Settings& cfg, const __half* params, Vec p, const Face& f, Vec beta, int level)
{
    int n = 4 << level;
    float x = fminf(fmaxf(p.x, 0.f), 1.f) * (n - 1), y = fminf(fmaxf(p.y, 0.f), 1.f) * (n - 1), z = fminf(fmaxf(p.z, 0.f), 1.f) * (n - 1);
    int ix = min(int(x), n - 2), iy = min(int(y), n - 2), iz = min(int(z), n - 2);
    x -= ix;
    y -= iy;
    z -= iz;
    uint32_t vertices = 10 * (1u << (2 * (level / 2))) + 2;
    bool dense = uint64_t(n) * n * n * vertices <= cfg.size[level];
    float2 value{0, 0};
    Tetrahedron tetra;
    if constexpr (Tetra)
        tetra = tetrahedron(x, y, z);
#pragma unroll
    for (int k = 0; k < (Tetra ? 4 : 8); ++k)
    {
        int corner = Tetra ? tetra.corner[k] : k;
        uint32_t a = ix + (corner & 1), b = iy + ((corner >> 1) & 1), c = iz + (corner >> 2);
        float w = Tetra ? tetra.weight[k] : ((k & 1) ? x : 1 - x) * ((k & 2) ? y : 1 - y) * ((k & 4) ? z : 1 - z);
#pragma unroll
        for (int v = 0; v < 3; ++v)
        {
            uint32_t slot = dense ? (a + n * (b + n * c)) * vertices + f.id[v]
                                  : ((a * 1u) ^ (b * 2654435761u) ^ (c * 805459861u) ^ f.hash[v]) & (cfg.size[level] - 1);
            float weight = w * (v == 0 ? beta.x : v == 1 ? beta.y : beta.z);
            const __half* address = params + cfg.offset[level] + slot * Features;
            float2 pair = __half22float2(*reinterpret_cast<const __half2*>(address));
            value.x += weight * pair.x;
            value.y += weight * pair.y;
        }
    }
    return value;
}

template<bool Tetra>
__device__ inline float2 lookupFeaturePair(const Settings& cfg, const __half* params, Vec p, Vec direction, int face, int level)
{
    const Face& f = cfg.faces[face];
    return lookupFeaturePairWithWeights<Tetra>(cfg, params, p, f, directionWeights(f, direction), level);
}

template<int Width>
__device__ inline int bestFaceWide(unsigned mask, float score, int id)
{
#pragma unroll
    for (int step = Width / 2; step; step >>= 1)
    {
        float other = __shfl_xor_sync(mask, score, step, Width);
        int otherId = __shfl_xor_sync(mask, id, step, Width);
        if (other > score || (other == score && otherId < id))
        {
            score = other;
            id = otherId;
        }
    }
    return id;
}
template<bool Tetra>
__device__ inline Banks inferenceBanksImpl(const Settings& cfg, Vec p)
{
    Banks b{};
    b.count = cfg.cells ? (Tetra ? 4 : 8) : 1;
    if (!cfg.cells)
    {
        b.w[0] = 1;
        return b;
    }
    float x = fminf(fmaxf(p.x, 0.f), 1.f) * cfg.cells;
    float y = fminf(fmaxf(p.y, 0.f), 1.f) * cfg.cells;
    float z = fminf(fmaxf(p.z, 0.f), 1.f) * cfg.cells;
    int ix = min(int(x), cfg.cells - 1), iy = min(int(y), cfg.cells - 1), iz = min(int(z), cfg.cells - 1);
    x -= ix;
    y -= iy;
    z -= iz;
    int n = cfg.cells + 1;
    Tetrahedron tetra;
    if constexpr (Tetra)
        tetra = tetrahedron(x, y, z);
#pragma unroll
    for (int k = 0; k < (Tetra ? 4 : 8); ++k)
    {
        int corner = Tetra ? tetra.corner[k] : k;
        int a = corner & 1, c = (corner >> 1) & 1, d = corner >> 2;
        b.id[k] = ix + a + n * (iy + c + n * (iz + d));
        b.w[k] = Tetra ? tetra.weight[k] : (a ? x : 1 - x) * (c ? y : 1 - y) * (d ? z : 1 - z);
    }
    return b;
}
__device__ inline Banks inferenceBanks(const Settings& cfg, Vec p)
{
    return cfg.tetrahedral ? inferenceBanksImpl<true>(cfg, p) : inferenceBanksImpl<false>(cfg, p);
}
template<int Corners = 8>
__device__ inline float inferenceBlend(const __half* p, const Banks& b, int index)
{
    float value = 0;
#pragma unroll
    for (int k = 0; k < Corners; ++k)
        if (k < b.count)
            value += b.w[k] * float(p[b.id[k] * MlpParams + index]);
    return value;
}
template<int Stage = 0, bool Tetra = false>
__global__ void infer8(
    const __grid_constant__ Settings cfg,
    Inputs inputs,
    int count,
    const __half* params,
    float* prediction,
    tcnn::FusedOutputData fused,
    float* encoded = nullptr
)
{
    int sample = (blockIdx.x * blockDim.x + threadIdx.x) / 8, lane = threadIdx.x & 7;
    if (sample >= count)
        return;
    unsigned mask = 0xffu << (threadIdx.x & 24);
    const float* pos = inputs.at(1, sample);
    Vec p{pos[0], pos[1], pos[2]};
    float x0 = 0, x1 = 0, x2 = 0;
    if constexpr (Stage != 2)
    {
        float2 uv = __half22float2(*reinterpret_cast<const __half2*>(inputs.at(4, sample)));
        Vec direction = oct(uv.x, uv.y);
        float best = -1.e30f;
        int face = 0;
#pragma unroll
        for (int k = 0; k < 3; ++k)
        {
            int candidate = lane + 8 * k;
            if (candidate < 20)
            {
                float score = coneScore(cfg.faces[candidate], direction);
                if (score > best)
                {
                    best = score;
                    face = candidate;
                }
            }
        }
        face = bestFace8(mask, best, face);
        int first = 0, second = 0;
#pragma unroll
        for (int depth = 0; depth < 4; ++depth)
        {
            if (depth == lane / 4)
                first = cfg.faceOffset[depth] + face;
            if (depth == 2 + lane / 4)
                second = cfg.faceOffset[depth] + face;
            if (depth < 3)
            {
                float score = lane < 4 ? coneScore(cfg.faces[cfg.faceOffset[depth + 1] + 4 * face + lane], direction) : -1.e30f;
                face = 4 * face + bestFace8(mask, score, lane);
            }
        }
        x0 = lookupFeatureImpl<Tetra>(cfg, params, p, direction, first, lane / 2, lane % 2);
        x1 = lookupFeatureImpl<Tetra>(cfg, params, p, direction, second, 4 + lane / 2, lane % 2);
        if (lane < 6)
        {
            float2 normal = __half22float2(*reinterpret_cast<const __half2*>(inputs.at(0, sample)));
            x2 = lane == 0 ? normal.x : lane == 1 ? normal.y : lane == 2 ? *inputs.at(2, sample) : inputs.at(3, sample)[lane - 3];
        }
    }
    if constexpr (Stage == 1)
    {
        encoded[sample * Input + lane] = x0;
        encoded[sample * Input + lane + 8] = x1;
        if (lane < 6)
            encoded[sample * Input + lane + 16] = x2;
        return;
    }
    if constexpr (Stage == 2)
    {
        x0 = encoded[sample * Input + lane];
        x1 = encoded[sample * Input + lane + 8];
        if (lane < 6)
            x2 = encoded[sample * Input + lane + 16];
    }
    Banks banks = inferenceBanksImpl<Tetra>(cfg, p);
    float h0 = 0;
#pragma unroll
    for (int k = 0; k <= Input; ++k)
    {
        float x = k == Input ? 1.f : __shfl_sync(mask, k < 8 ? x0 : k < 16 ? x1 : x2, k % 8, 8);
        h0 += inferenceBlend<Tetra ? 4 : 8>(params, banks, k * Hidden + lane) * x;
    }
    h0 = activate(h0);
    float h1 = 0;
#pragma unroll
    for (int k = 0; k <= Hidden; ++k)
    {
        float x = k == Hidden ? 1.f : __shfl_sync(mask, h0, k, 8);
        h1 += inferenceBlend<Tetra ? 4 : 8>(params, banks, Layer0 + k * Hidden + lane) * x;
    }
    h1 = activate(h1);
    float y = 0;
#pragma unroll
    for (int k = 0; k <= Hidden; ++k)
    {
        float x = k == Hidden ? 1.f : __shfl_sync(mask, h1, k, 8);
        if (lane < 3)
            y += inferenceBlend<Tetra ? 4 : 8>(params, banks, Layer0 + Layer1 + k * 3 + lane) * x;
    }
    if (lane < 3)
    {
        if (fused.arr0)
        {
            int id = reinterpret_cast<const uint32_t*>(fused.arr1)[sample];
            float weight = reinterpret_cast<const float*>(fused.arr0)[sample * 3 + lane];
            atomicAdd(prediction + 3 * id + lane, fmaxf(y, 0.f) * weight);
        }
        else
            prediction[sample * 3 + lane] = y;
    }
}

#include "HybridMlpTexture.cuh"

// A thread keeps a complete small MLP in registers. Consecutive threads
// walk the same feature levels, improving locality versus lanes split across
// levels. Width=2 remains available to the replay profiler for comparison.
template<int Width, bool ReuseDirectionWeights = false, int MlpBackend = 0>
__global__ void inferSmall(
    const __grid_constant__ Settings cfg,
    Inputs inputs,
    int count,
    const __half* params,
    float* prediction,
    tcnn::FusedOutputData fused,
    cudaTextureObject_t texture = 0
)
{
    int sample = (blockIdx.x * blockDim.x + threadIdx.x) / Width, lane = threadIdx.x % Width;
    if (sample >= count)
        return;
    unsigned mask = ((1u << Width) - 1) << (threadIdx.x & (32 - Width));
    const float* pos = inputs.at(1, sample);
    Vec p{pos[0], pos[1], pos[2]};
    float2 uv = __half22float2(*reinterpret_cast<const __half2*>(inputs.at(4, sample)));
    Vec direction = oct(uv.x, uv.y);
    float best = -1.e30f;
    int face = 0;
#pragma unroll
    for (int k = 0; k < 20 / Width; ++k)
    {
        int candidate = lane + Width * k;
        float score = coneScore(cfg.faces[candidate], direction);
        if (score > best)
        {
            best = score;
            face = candidate;
        }
    }
    face = bestFaceWide<Width>(mask, best, face);
    int selected[4];
#pragma unroll
    for (int depth = 0; depth < 4; ++depth)
    {
        selected[depth] = cfg.faceOffset[depth] + face;
        if (depth < 3)
        {
            int child = 0;
            float scoreBest = -1.e30f;
#pragma unroll
            for (int k = 0; k < 4 / Width; ++k)
            {
                int candidate = lane + k * Width;
                float score = coneScore(cfg.faces[cfg.faceOffset[depth + 1] + 4 * face + candidate], direction);
                if (score > scoreBest)
                {
                    scoreBest = score;
                    child = candidate;
                }
            }
            face = 4 * face + bestFaceWide<Width>(mask, scoreBest, child);
        }
    }
    float2 features[8 / Width];
    if constexpr (ReuseDirectionWeights)
    {
        static_assert(Width == 1, "Paired-level direction reuse expects one thread per query");
        // Adjacent levels use the same spherical triangle. Reuse only this
        // query's directional weights; spatial queries and MLPs are unchanged.
#pragma unroll
        for (int depth = 0; depth < 4; ++depth)
        {
            const Face& f = cfg.faces[selected[depth]];
            Vec beta = directionWeights(f, direction);
            features[2 * depth] = lookupFeaturePairWithWeights<true>(cfg, params, p, f, beta, 2 * depth);
            features[2 * depth + 1] = lookupFeaturePairWithWeights<true>(cfg, params, p, f, beta, 2 * depth + 1);
        }
    }
    else
    {
#pragma unroll
        for (int j = 0; j < 8 / Width; ++j)
        {
            int level = j * Width + lane;
            features[j] = lookupFeaturePair<true>(cfg, params, p, direction, selected[level / 2], level);
        }
    }
    float2 normal = __half22float2(*reinterpret_cast<const __half2*>(inputs.at(0, sample)));
    float surface[(6 + Width - 1) / Width];
#pragma unroll
    for (int j = 0; j < (6 + Width - 1) / Width; ++j)
    {
        int k = j * Width + lane;
        surface[j] = k == 0 ? normal.x : k == 1 ? normal.y : k == 2 ? *inputs.at(2, sample) : k < 6 ? inputs.at(3, sample)[k - 3] : 0;
    }
    if constexpr (MlpBackend != 0)
    {
        static_assert(Width == 1, "Experimental MLP backend requires one thread per query");
        float encoded[Input];
#pragma unroll
        for (int k = 0; k < 16; ++k) encoded[k] = k % 2 ? features[k / 2].y : features[k / 2].x;
#pragma unroll
        for (int k = 0; k < 6; ++k) encoded[16 + k] = surface[k];
        float3 y = decodeMlp<MlpBackend>(cfg, params, texture, p, encoded);
        float values[3] = {y.x, y.y, y.z};
#pragma unroll
        for (int c = 0; c < 3; ++c)
            if (fused.arr0)
            {
                int id = reinterpret_cast<const uint32_t*>(fused.arr1)[sample];
                float w = reinterpret_cast<const float*>(fused.arr0)[3 * sample + c];
                atomicAdd(prediction + 3 * id + c, fmaxf(values[c], 0.f) * w);
            }
            else prediction[3 * sample + c] = values[c];
        return;
    }
    Banks banks = inferenceBanksImpl<true>(cfg, p);
    float a[8 / Width] = {}, b[8 / Width] = {};
#pragma unroll
    for (int k = 0; k <= Input; ++k)
    {
        float v = 1;
        if (k < 16)
            v = k % 2 ? features[k / (2 * Width)].y : features[k / (2 * Width)].x;
        else if (k < Input)
            v = surface[(k - 16) / Width];
        float x = k == Input ? 1.f : __shfl_sync(mask, v, k < 16 ? (k / 2) % Width : (k - 16) % Width, Width);
#pragma unroll
        for (int j = 0; j < 8 / Width; ++j)
            a[j] += inferenceBlend<4>(params, banks, k * Hidden + j * Width + lane) * x;
    }
#pragma unroll
    for (int j = 0; j < 8 / Width; ++j)
        a[j] = activate(a[j]);
#pragma unroll
    for (int k = 0; k <= Hidden; ++k)
    {
        float x = k == Hidden ? 1.f : __shfl_sync(mask, a[k / Width], k % Width, Width);
#pragma unroll
        for (int j = 0; j < 8 / Width; ++j)
            b[j] += inferenceBlend<4>(params, banks, Layer0 + k * Hidden + j * Width + lane) * x;
    }
#pragma unroll
    for (int j = 0; j < 8 / Width; ++j)
        b[j] = activate(b[j]);
    float y[(3 + Width - 1) / Width] = {};
#pragma unroll
    for (int k = 0; k <= Hidden; ++k)
    {
        float x = k == Hidden ? 1.f : __shfl_sync(mask, b[k / Width], k % Width, Width);
#pragma unroll
        for (int j = 0; j < (3 + Width - 1) / Width; ++j)
            if (j * Width + lane < 3)
                y[j] += inferenceBlend<4>(params, banks, Layer0 + Layer1 + k * 3 + j * Width + lane) * x;
    }
#pragma unroll
    for (int j = 0; j < (3 + Width - 1) / Width; ++j)
    {
        int c = j * Width + lane;
        if (c < 3)
        {
            if (fused.arr0)
            {
                int id = reinterpret_cast<const uint32_t*>(fused.arr1)[sample];
                float w = reinterpret_cast<const float*>(fused.arr0)[3 * sample + c];
                atomicAdd(prediction + 3 * id + c, fmaxf(y[j], 0.f) * w);
            }
            else
                prediction[3 * sample + c] = y[j];
        }
    }
}

class Network final : public tcnn::Network<float, __half>
{
public:
    explicit Network(const tcnn::json& config)
    {
        // Diagnostic fallback for matched production comparisons. The old
        // eight-lane path remains available, with identical model semantics.
        if (const char* value = std::getenv("NIRC_HYBRID_LEGACY_INFERENCE"))
            m_legacyInference = std::string(value) == "1";
        if (const char* value = std::getenv("NIRC_HYBRID_REUSE_DIRECTION_WEIGHTS"))
            m_reuseDirectionWeights = std::string(value) == "1";
        const std::string interpolation = config.value("spatial_interpolation", std::string("trilinear"));
        if (interpolation != "trilinear" && interpolation != "tetrahedral")
            throw std::runtime_error("Unknown hybrid spatial interpolation");
        m_cfg.tetrahedral = interpolation == "tetrahedral";
        if (const char* value = std::getenv("NIRC_HYBRID_MLP_BACKEND"))
        {
            std::string backend(value);
            if (backend != "default" && backend != "trilinear" && backend != "texture")
                throw std::runtime_error("NIRC_HYBRID_MLP_BACKEND must be default, trilinear or texture");
            m_cfg.mlpTrilinear = backend != "default";
            m_textureInference = backend == "texture";
        }
        // Captured models retain their training stencil independently of the
        // replay backend. Only MLP interpolation changes in this experiment.
        if (m_cfg.tetrahedral && config.value("mlp_interpolation", std::string()) == "trilinear") m_cfg.mlpTrilinear = true;
        if (m_cfg.mlpTrilinear && !m_cfg.tetrahedral)
            throw std::runtime_error("MLP-only experiment requires tetrahedral hash features");
        m_cfg.cells = config.value("mlp_grid_cells", 4);
        if (m_cfg.cells < 0 || m_cfg.cells > 16)
            throw std::runtime_error("mlp_grid_cells must be 0..16");
        m_cfg.banks = m_cfg.cells ? int(pow(m_cfg.cells + 1, 3)) : 1;
        m_cfg.mlpCount = m_cfg.banks * MlpParams;
        if (m_textureInference) m_mlpTexture.initialize(m_cfg.cells);
        int logCapacity = config.value("log2_hashmap_size", 18);
        if (logCapacity < 10 || logCapacity > 22)
            throw std::runtime_error("Invalid hybrid hash capacity");
        uint32_t capacity = 1u << logCapacity;
        // Align paired FP16 hash features without changing any bank or hash index.
        m_count = (m_cfg.mlpCount + 1u) & ~1u;
        for (int l = 0; l < Levels; ++l)
        {
            uint64_t n = 4ull << l, total = n * n * n * (10 * (1u << (2 * (l / 2))) + 2);
            m_cfg.offset[l] = uint32_t(m_count);
            m_cfg.size[l] = uint32_t(std::min<uint64_t>(total, capacity));
            m_count += m_cfg.size[l] * Features;
        }
        makeHierarchy();
        m_floatGradients.resize(m_count);
    }
    size_t n_params() const override { return m_count; }
    const Settings& settings() const { return m_cfg; }
    size_t texturePayloadBytes() const { return m_textureInference ? m_mlpTexture.payloadBytes() : 0; }
    const char* mlpBackend() const { return m_textureInference ? "texture" : m_cfg.mlpTrilinear ? "trilinear" : "default"; }
    void setOptimizedInference(bool enabled) { m_optimizedInference = enabled; }
    void setLegacyInference(bool enabled) { m_legacyInference = enabled; }
    bool hasCapturedInference() const { return m_captured; }
    void captureInference(
        cudaStream_t stream,
        const tcnn::NeuralInputData<float>& input,
        uint32_t count,
        const char* directory,
        tcnn::FusedOutputData fused = {},
        size_t outputBytes = 0,
        uint32_t outputPixelCount = 0,
        size_t outputAllocationBytes = 0
    )
    {
        if (m_captured)
            return;
        CUDA_CHECK_THROW(cudaStreamSynchronize(stream));
        std::filesystem::path path(directory);
        std::filesystem::create_directories(path);
        auto save = [&](const std::string& name, const void* data, size_t bytes)
        {
            std::vector<char> host(bytes);
            CUDA_CHECK_THROW(cudaMemcpy(host.data(), data, bytes, cudaMemcpyDeviceToHost));
            std::ofstream file(path / name, std::ios::binary);
            file.write(host.data(), bytes);
            if (!file)
                throw std::runtime_error("Cannot write hybrid inference capture");
        };
        tcnn::json meta = {
            {"count", count},
            {"params", m_count},
            {"mlp_grid_cells", m_cfg.cells},
            {"log2_hashmap_size", int(std::log2(m_cfg.size[Levels - 1]))}
        };
        save("params.bin", m_inference, m_count * sizeof(__half));
        meta["spatial_interpolation"] = m_cfg.tetrahedral ? "tetrahedral" : "trilinear";
        meta["mlp_interpolation"] = m_cfg.mlpTrilinear ? "trilinear" : (m_cfg.tetrahedral ? "tetrahedral" : "trilinear");
        meta["mlp_backend"] = mlpBackend();
        meta["mlp_texture_payload_bytes"] = texturePayloadBytes();
        meta["hash_parameter_padding"] = m_cfg.offset[0] - m_cfg.mlpCount;
        meta["inference_lanes"] = m_cfg.tetrahedral && !m_legacyInference ? 1 : 8;
        meta["tcnn_tracked_cuda_bytes"] = tcnn::total_n_bytes_allocated().load();
        meta["output_bytes"] = outputBytes;
        meta["output_pixel_count"] = outputPixelCount;
        meta["output_allocation_bytes"] = outputAllocationBytes;
        meta["fused_output"] = fused.arr0 != nullptr;
        if (fused.arr0)
        {
            save("throughput.bin", fused.arr0, size_t(count) * 3 * sizeof(float));
            save("pixels.bin", fused.arr1, size_t(count) * sizeof(uint32_t));
        }
        for (int c = 0; c < 5; ++c)
        {
            int mapGroup = c;
            for (int previous = 0; previous < c; ++previous)
                if (input.mappings[c] && input.mappings[c] == input.mappings[previous])
                {
                    mapGroup = previous;
                    break;
                }
            meta["channels"].push_back(
                {{"rows", input.data[c].rows()},
                 {"cols", input.data[c].cols()},
                 {"mapped", input.mappings[c] != nullptr},
                 {"mapping_storage_group", mapGroup}}
            );
            save("input" + std::to_string(c) + ".bin", input.data[c].data(), input.data[c].n_bytes());
            if (input.mappings[c])
                save("map" + std::to_string(c) + ".bin", input.mappings[c], count * sizeof(int));
        }
        std::ofstream(path / "metadata.json") << meta.dump(2);
        m_captured = true;
    }
    std::vector<std::pair<uint32_t, uint32_t>> layer_sizes() const override { return {{uint32_t(m_cfg.mlpCount), 1}}; }
    uint32_t padded_output_width() const override { return 16; }
    uint32_t output_width() const override { return 3; }
    uint32_t required_input_alignment() const override { return 1; }
    uint32_t width(uint32_t) const override { return Hidden; }
    uint32_t num_forward_activations() const override { return 0; }
    const __half* forward_activations(uint32_t) const override { return nullptr; }
    void initialize_params(tcnn::pcg32& random, float* full, __half* params, __half* inference, __half*, __half* gradients, float scale = 1)
        override
    {
        m_params = params;
        m_inference = inference;
        m_gradients = gradients;
        std::vector<float> values(m_count);
        for (size_t i = m_cfg.offset[0]; i < m_count; ++i)
            values[i] = (random.next_float() * 2 - 1) * 1.e-4f;
        // Shared initial basis prevents independently initialized neighbouring
        // MLPs from cancelling under parameter interpolation. Banks train apart.
        int sizes[3] = {Input, Hidden, Hidden}, outs[3] = {Hidden, Hidden, 3};
        int offset = 0;
        for (int l = 0; l < 3; ++l)
        {
            for (int i = 0; i <= sizes[l]; ++i)
                for (int j = 0; j < outs[l]; ++j)
                {
                    float v = i == sizes[l] ? 0.f : (random.next_float() * 2 - 1) * sqrtf(6.f / (sizes[l] + outs[l])) * scale;
                    for (int b = 0; b < m_cfg.banks; ++b)
                        values[b * MlpParams + offset + i * outs[l] + j] = v;
                }
            offset += (sizes[l] + 1) * outs[l];
        }
        CUDA_CHECK_THROW(cudaMemcpy(full, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    void inference(
        cudaStream_t stream,
        tcnn::NeuralInputData<float>& input,
        tcnn::GPUMatrix<float>& output,
        uint32_t n,
        int*,
        float*,
        float*,
        tcnn::FusedOutputData fused = {}
    ) override
    {
        if (!n)
            return;
        if (m_cfg.mlpTrilinear)
        {
            if (m_textureInference)
            {
                // Refresh from EMA inference weights on the same stream before
                // every call. Includes updates in measured inference cost and
                // avoids stale parameters after optimizer/CUDA-graph execution.
                m_mlpTexture.refresh(stream, m_inference);
                inferSmall<1, true, 2><<<(n + 255) / 256, 256, 0, stream>>>(m_cfg, describe(input), n, m_inference, output.data(), fused, m_mlpTexture.object());
            }
            else
                inferSmall<1, true, 1><<<(n + 255) / 256, 256, 0, stream>>>(m_cfg, describe(input), n, m_inference, output.data(), fused);
            CUDA_CHECK_THROW(cudaGetLastError());
            return;
        }
        if (m_optimizedInference)
        {
            if (m_cfg.tetrahedral && !m_legacyInference)
            {
                if (m_reuseDirectionWeights)
                    inferSmall<1, true><<<(n + 255) / 256, 256, 0, stream>>>(m_cfg, describe(input), n, m_inference, output.data(), fused);
                else
                    inferSmall<1, false><<<(n + 255) / 256, 256, 0, stream>>>(m_cfg, describe(input), n, m_inference, output.data(), fused);
            }
            else if (m_cfg.tetrahedral)
                infer8<0, true><<<(n + 15) / 16, 128, 0, stream>>>(m_cfg, describe(input), n, m_inference, output.data(), fused);
            else
                infer8<0, false><<<(n + 15) / 16, 128, 0, stream>>>(m_cfg, describe(input), n, m_inference, output.data(), fused);
        }
        else
            evaluate<false><<<(n + 3) / 4, 128, 0, stream>>>(
                m_cfg, describe(input), n, m_inference, nullptr, 0, output.data(), fused, nullptr, nullptr
            );
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    void forward(
        cudaStream_t stream,
        tcnn::NeuralInputData<float>& input,
        uint32_t n,
        int*,
        float*,
        float*,
        tcnn::GPUMatrixDynamic<__half>* output = nullptr,
        bool inference = false,
        bool inputGrad = false
    ) override
    {
        if (inputGrad)
            throw std::runtime_error("Hybrid input derivatives are not supported");
        if (!output || output->layout() != tcnn::CM)
            throw std::runtime_error("Hybrid requires column-major training output");
        m_lastInput = describe(input);
        m_lastCount = n;
        CUDA_CHECK_THROW(cudaMemsetAsync(output->data(), 0, output->n_bytes(), stream));
        evaluate<false><<<(n + 3) / 4, 128, 0, stream>>>(
            m_cfg, m_lastInput, n, inference ? m_inference : m_params, output->data(), output->m(), nullptr, {}, nullptr, nullptr
        );
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    void backward(
        cudaStream_t stream,
        tcnn::NeuralInputData<float>&,
        const tcnn::GPUMatrixDynamic<__half>&,
        const tcnn::GPUMatrixDynamic<__half>& dy,
        tcnn::GPUMatrix<float>* inputGrad = nullptr,
        bool inference = false,
        bool compute = true
    ) override
    {
        if (inputGrad)
            throw std::runtime_error("Hybrid input derivatives are not supported");
        if (!compute)
            return;
        CUDA_CHECK_THROW(cudaMemsetAsync(m_floatGradients.data(), 0, m_count * sizeof(float), stream));
        evaluate<true><<<(m_lastCount + 3) / 4, 128, 0, stream>>>(
            m_cfg,
            m_lastInput,
            m_lastCount,
            inference ? m_inference : m_params,
            nullptr,
            dy.m(),
            nullptr,
            {},
            dy.data(),
            m_floatGradients.data()
        );
        castGradients<<<(m_count + 255) / 256, 256, 0, stream>>>(m_count, m_floatGradients.data(), m_gradients);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    void inference(cudaStream_t, const tcnn::GPUMatrix<float>&, tcnn::GPUMatrix<float>&, tcnn::GPUMatrix<float>* = nullptr) override
    {
        throw std::runtime_error("Hybrid expects five NIRC input channels");
    }
    void forward(cudaStream_t, const tcnn::GPUMatrix<float>&, tcnn::GPUMatrixDynamic<__half>* = nullptr, bool = false, bool = false)
        override
    {
        throw std::runtime_error("Hybrid expects five NIRC input channels");
    }
    void backward(
        cudaStream_t,
        const tcnn::GPUMatrix<float>&,
        const tcnn::GPUMatrixDynamic<__half>&,
        const tcnn::GPUMatrixDynamic<__half>&,
        tcnn::GPUMatrix<float>* = nullptr,
        bool = false,
        bool = true
    ) override
    {
        throw std::runtime_error("Hybrid expects five NIRC input channels");
    }

private:
    Inputs describe(const tcnn::NeuralInputData<float>& input)
    {
        if (input.num_channels() != 5)
            throw std::runtime_error("Hybrid requires five NIRC channels");
        Inputs result{};
        for (int i = 0; i < 5; ++i)
        {
            result.data[i] = input.data[i].data();
            result.map[i] = input.mappings[i];
            result.stride[i] = input.data[i].rows();
        }
        return result;
    }
    void makeHierarchy()
    {
        float t = (1 + sqrtf(5.f)) * .5f;
        std::vector<Vec> vertices = {
            {-1, t, 0},
            {1, t, 0},
            {-1, -t, 0},
            {1, -t, 0},
            {0, -1, t},
            {0, 1, t},
            {0, -1, -t},
            {0, 1, -t},
            {t, 0, -1},
            {t, 0, 1},
            {-t, 0, -1},
            {-t, 0, 1}
        };
        for (auto& v : vertices)
            v = unit(v);
        using Tri = std::array<uint32_t, 3>;
        std::vector<Tri> faces = {{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                  {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
                                  {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};
        std::vector<Face> all;
        for (int l = 0; l < 4; ++l)
        {
            m_cfg.faceOffset[l] = uint32_t(all.size());
            for (auto tri : faces)
            {
                Face f{vertices[tri[0]], vertices[tri[1]], vertices[tri[2]], {tri[0], tri[1], tri[2]}, {}};
                for (int k = 0; k < 3; ++k)
                {
                    auto v = vertices[tri[k]];
                    f.hash[k] = (uint32_t(floor((1. + v.x) * 2097152.)) * 3674653429u) ^
                                (uint32_t(floor((1. + v.y) * 2097152.)) * 2097192037u) ^
                                (uint32_t(floor((1. + v.z) * 2097152.)) * 1434869437u);
                }
                all.push_back(f);
            }
            if (l == 3)
                break;
            std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
            auto midpoint = [&](uint32_t a, uint32_t b)
            {
                auto key = std::make_pair(std::min(a, b), std::max(a, b));
                auto it = edges.find(key);
                if (it != edges.end())
                    return it->second;
                uint32_t id = uint32_t(vertices.size());
                vertices.push_back(unit(add(vertices[a], vertices[b])));
                edges[key] = id;
                return id;
            };
            std::vector<Tri> next;
            for (auto f : faces)
            {
                auto ab = midpoint(f[0], f[1]), bc = midpoint(f[1], f[2]), ca = midpoint(f[2], f[0]);
                next.push_back({f[0], ab, ca});
                next.push_back({ab, f[1], bc});
                next.push_back({ca, bc, f[2]});
                next.push_back({ab, bc, ca});
            }
            faces = std::move(next);
        }
        m_faces.resize(all.size());
        CUDA_CHECK_THROW(cudaMemcpy(m_faces.data(), all.data(), all.size() * sizeof(Face), cudaMemcpyHostToDevice));
        m_cfg.faces = m_faces.data();
    }
    Settings m_cfg{};
    size_t m_count = 0;
    uint32_t m_lastCount = 0;
    bool m_captured = false;
    bool m_optimizedInference = true;
    bool m_legacyInference = false;
    // Reuse spherical barycentrics across adjacent levels. Set the diagnostic
    // environment switch to 0 to replay the previous arithmetic schedule.
    bool m_reuseDirectionWeights = true;
    bool m_textureInference = false;
    MlpTexture m_mlpTexture;
    Inputs m_lastInput{};
    __half *m_params = nullptr, *m_inference = nullptr, *m_gradients = nullptr;
    tcnn::GPUMemory<Face> m_faces;
    tcnn::GPUMemory<float> m_floatGradients;
};
} // namespace nirc_hybrid
