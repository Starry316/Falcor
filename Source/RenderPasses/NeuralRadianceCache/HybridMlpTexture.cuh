#pragma once
// Included inside nirc_hybrid after the scalar interpolation helpers.
// Each x-brick stores one RGBA group on an (cells+1)^3 lattice. Coordinates
// stay between texel centers inside that brick, never blending parameter groups.
constexpr int MlpTextureGroups = (MlpParams + 3) / 4;

__global__ void packMlpTexture(cudaSurfaceObject_t surface, const __half* parameters, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int width = n * MlpTextureGroups;
    if (index >= width * n * n) return;
    int x = index % width, y = (index / width) % n, z = index / (width * n);
    int group = x / n, bank = x % n + n * (y + n * z);
    ushort4 value{};
    unsigned short* components = reinterpret_cast<unsigned short*>(&value);
#pragma unroll
    for (int c = 0; c < 4; ++c)
        components[c] = group * 4 + c < MlpParams ? __half_as_ushort(parameters[bank * MlpParams + group * 4 + c]) : 0;
    surf3Dwrite(value, surface, x * int(sizeof(ushort4)), y, z);
}

class MlpTexture
{
public:
    MlpTexture() = default;
    MlpTexture(const MlpTexture&) = delete;
    MlpTexture& operator=(const MlpTexture&) = delete;
    ~MlpTexture()
    {
        if (m_texture) cudaDestroyTextureObject(m_texture);
        if (m_surface) cudaDestroySurfaceObject(m_surface);
        if (m_array) cudaFreeArray(m_array);
    }
    void initialize(int cells)
    {
        if (m_array) throw std::runtime_error("MLP texture already initialized");
        m_n = cells + 1;
        cudaChannelFormatDesc format{16, 16, 16, 16, cudaChannelFormatKindFloat};
        CUDA_CHECK_THROW(cudaMalloc3DArray(&m_array, &format, make_cudaExtent(m_n * MlpTextureGroups, m_n, m_n), cudaArraySurfaceLoadStore));
        cudaResourceDesc resource{};
        resource.resType = cudaResourceTypeArray;
        resource.res.array.array = m_array;
        CUDA_CHECK_THROW(cudaCreateSurfaceObject(&m_surface, &resource));
        cudaTextureDesc sampling{};
        sampling.addressMode[0] = sampling.addressMode[1] = sampling.addressMode[2] = cudaAddressModeClamp;
        sampling.filterMode = cudaFilterModeLinear;
        sampling.readMode = cudaReadModeElementType;
        sampling.normalizedCoords = 0;
        CUDA_CHECK_THROW(cudaCreateTextureObject(&m_texture, &resource, &sampling, nullptr));
    }
    void refresh(cudaStream_t stream, const __half* parameters)
    {
        int texels = m_n * m_n * m_n * MlpTextureGroups;
        packMlpTexture<<<(texels + 127) / 128, 128, 0, stream>>>(m_surface, parameters, m_n);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    cudaTextureObject_t object() const { return m_texture; }
    size_t payloadBytes() const { return size_t(m_n) * m_n * m_n * MlpTextureGroups * sizeof(ushort4); }
private:
    int m_n = 0;
    cudaArray_t m_array = nullptr;
    cudaTextureObject_t m_texture = 0;
    cudaSurfaceObject_t m_surface = 0;
};

template<int Backend>
__device__ inline float4 mlpWeights4(const Settings& cfg, const __half* params, cudaTextureObject_t texture,
                                   Vec p, const Banks& banks, int group)
{
    if constexpr (Backend == 2)
        return tex3D<float4>(texture,
            group * (cfg.cells + 1) + fminf(fmaxf(p.x, 0.f), 1.f) * cfg.cells + .5f,
            fminf(fmaxf(p.y, 0.f), 1.f) * cfg.cells + .5f,
            fminf(fmaxf(p.z, 0.f), 1.f) * cfg.cells + .5f);
    else
    {
        float v[4]{};
#pragma unroll
        for (int c = 0; c < 4; ++c)
            if (group * 4 + c < MlpParams) v[c] = inferenceBlend<Backend == 0 ? 4 : 8>(params, banks, group * 4 + c);
        return make_float4(v[0], v[1], v[2], v[3]);
    }
}

template<int Backend>
__device__ inline float3 decodeMlp(const Settings& cfg, const __half* params, cudaTextureObject_t texture, Vec p, const float* encoded)
{
    Banks banks{};
    if constexpr (Backend != 2) banks = inferenceBanksImpl<Backend == 0>(cfg, p);
    float a[8]{}, b[8]{}, y[3]{};
#pragma unroll
    for (int k = 0; k <= Input; ++k)
    {
        float x = k == Input ? 1.f : encoded[k];
#pragma unroll
        for (int g = 0; g < 2; ++g)
        {
            float4 w = mlpWeights4<Backend>(cfg, params, texture, p, banks, k * 2 + g);
            a[g * 4] += w.x * x; a[g * 4 + 1] += w.y * x;
            a[g * 4 + 2] += w.z * x; a[g * 4 + 3] += w.w * x;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = activate(a[j]);
#pragma unroll
    for (int k = 0; k <= 8; ++k)
    {
        float x = k == 8 ? 1.f : a[k];
#pragma unroll
        for (int g = 0; g < 2; ++g)
        {
            float4 w = mlpWeights4<Backend>(cfg, params, texture, p, banks, Layer0 / 4 + k * 2 + g);
            b[g * 4] += w.x * x; b[g * 4 + 1] += w.y * x;
            b[g * 4 + 2] += w.z * x; b[g * 4 + 3] += w.w * x;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = activate(b[j]);
#pragma unroll
    for (int g = 0; g < 7; ++g)
    {
        float4 w = mlpWeights4<Backend>(cfg, params, texture, p, banks, (Layer0 + Layer1) / 4 + g);
        float v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            int k = (g * 4 + c) / 3, j = (g * 4 + c) % 3;
            if (g * 4 + c < 27) y[j] += v[c] * (k == 8 ? 1.f : b[k]);
        }
    }
    return make_float3(y[0], y[1], y[2]);
}
