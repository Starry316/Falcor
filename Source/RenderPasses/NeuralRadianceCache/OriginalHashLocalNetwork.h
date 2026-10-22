#pragma once
// Original NIRC surface encodings and SH direction basis with interpolated local
// 64/32/23 -> 8 -> 8 -> 3 MLPs. The original position HashGrid is reused.
#include "HybridRadianceNetwork.h"
#include <tiny-cuda-nn/encoding.h>

namespace nirc_local
{
template<int N> struct Layout {
    static_assert(N == 64 || N == 32 || N == 23, "Unsupported local MLP input layout");
    static constexpr int Input = N, Hidden = 8;
    static constexpr int Normal = N == 23 ? 0 : 8;
    static constexpr int Hash = N == 64 ? 24 : 8;
    static constexpr int Roughness = N == 23 ? 1 : 4;
    static constexpr int RawNormal = N == 23 ? 2 : 0;
    static constexpr int EncodedSurface = Normal + Hash + Roughness + 3;
    static constexpr int Surface = EncodedSurface + RawNormal;
    static constexpr int SHDegree = N == 64 ? 5 : 3;
    static constexpr int Stride = (N + 1) & ~1; // half2-aligned training rows
    static constexpr int Layer0 = (Input + 1) * Hidden, Layer1 = 72;
    static constexpr int MlpParams = Layer0 + Layer1 + 27;
};
// Retain the original constants for compatibility with existing diagnostic tools.
constexpr int Surface = Layout<64>::Surface, Input = 64, Hidden = 8;
constexpr int Layer0 = Layout<64>::Layer0, Layer1 = 72, MlpParams = Layout<64>::MlpParams;
template<int N> struct Encoded {
    const __half* data[4]{};
    const int* map[4]{};
    int stride[4]{};
    const __half* merged = nullptr;
    __device__ float at(int q, int k) const {
        using L = Layout<N>;
        if (merged) return float(merged[q * L::Stride + k]);
        int c = k < L::Normal ? 0 : k < L::Normal+L::Hash ? 1 : k < L::Normal+L::Hash+L::Roughness ? 2 : 3;
        int offset = c == 0 ? 0 : c == 1 ? L::Normal : c == 2 ? L::Normal+L::Hash : L::Normal+L::Hash+L::Roughness;
        return float(data[c][(map[c] ? map[c][q] : q) * stride[c] + k - offset]);
    }
};

// Same 25 SH polynomials and FP16 feature rounding as the released NIRC.
template<int Degree = 5>
__device__ inline void directionFeatures(nirc_hybrid::Vec d, float* out)
{
    float x=d.x,y=d.y,z=d.z;
    float xy=x*y,xz=x*z,yz=y*z,x2=x*x,y2=y*y,z2=z*z;
    float x4=x2*x2,y4=y2*y2,z4=z2*z2;
    out[0] = 0.28209479177387814f;
    out[1] = -0.48860251190291987f * y;
    out[2] = 0.48860251190291987f * z;
    out[3] = -0.48860251190291987f * x;
    out[4] = 1.0925484305920792f * xy;
    out[5] = -1.0925484305920792f * yz;
    out[6] = (0.94617469575755997f * z2 - 0.31539156525251999f);
    out[7] = -1.0925484305920792f * xz;
    out[8] = (0.54627421529603959f * x2 - 0.54627421529603959f * y2);
    if constexpr (Degree == 5) {
    out[9] = 0.59004358992664352f * y * (-3.0f * x2 + y2);
    out[10] = 2.8906114426405538f * xy * z;
    out[11] = 0.45704579946446572f * y * (1.0f - 5.0f * z2);
    out[12] = 0.3731763325901154f * z * (5.0f * z2 - 3.0f);
    out[13] = 0.45704579946446572f * x * (1.0f - 5.0f * z2);
    out[14] = 1.4453057213202769f * z * (x2 - y2);
    out[15] = 0.59004358992664352f * x * (-x2 + 3.0f * y2);
    out[16] = 2.5033429417967046f * xy * (x2 - y2);
    out[17] = 1.7701307697799304f * yz * (-3.0f * x2 + y2);
    out[18] = 0.94617469575756008f * xy * (7.0f * z2 - 1.0f);
    out[19] = 0.66904654355728921f * yz * (3.0f - 7.0f * z2);
    out[20] = (-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f);
    out[21] = (0.66904654355728921f * xz * (3.0f - 7.0f * z2));
    out[22] = 0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f);
    out[23] = 1.7701307697799304f * xz * (-x2 + 3.0f * y2);
    out[24] = (-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4);
    }
#pragma unroll
    for(int k=0;k<Degree*Degree;++k) out[k]=float(__float2half(out[k]));
}

template<int N>
__device__ inline float blendLocal(const __half* p, const nirc_hybrid::Banks& banks, int index)
{
    float v = 0;
    for (int k = 0; k < banks.count; ++k) v += banks.w[k] * float(p[banks.id[k] * Layout<N>::MlpParams + index]);
    return v;
}
template<int N>
__device__ inline void scatterLocal(float* g, const nirc_hybrid::Banks& banks, int index, float v)
{
    if (v == 0) return;
    for (int k = 0; k < banks.count; ++k)
        if (banks.w[k] != 0) atomicAdd(g + banks.id[k] * Layout<N>::MlpParams + index, banks.w[k] * v);
}

template<int N, bool Backward>
__global__ void evaluate(nirc_hybrid::Settings cfg, nirc_hybrid::Inputs input, Encoded<N> encoded,
    int count, const __half* params, float* prediction, __half* output, int stride,
    tcnn::FusedOutputData fused, const __half* dy, float* gradients, __half* dencoded)
{
    using L = Layout<N>;
    constexpr int Input = L::Input, Surface = L::Surface, Layer0 = L::Layer0;
    int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count) return;
    const float* p = input.at(1, q);
    auto banks = nirc_hybrid::banksAt(cfg, {p[0], p[1], p[2]});
    float x[Input], a[8] = {}, b[8] = {}, y[3] = {};
    float2 uv = __half22float2(*reinterpret_cast<const __half2*>(input.at(4, q)));
    directionFeatures<L::SHDegree>(nirc_hybrid::oct(uv.x, uv.y), x + Surface);
#pragma unroll
    for (int k = 0; k < L::EncodedSurface; ++k) x[k+L::RawNormal] = encoded.at(q, k);
    if constexpr (L::RawNormal) {
        float2 normal = __half22float2(*reinterpret_cast<const __half2*>(input.at(0, q)));
        x[0] = normal.x; x[1] = normal.y;
    }
#pragma unroll
    for (int k = 0; k <= Input; ++k) {
#pragma unroll
        for (int j = 0; j < 8; ++j) a[j] += blendLocal<N>(params, banks, k * 8 + j) * (k == Input ? 1.f : x[k]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) a[j] = nirc_hybrid::activate(a[j]);
#pragma unroll
    for (int k = 0; k <= 8; ++k) {
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] += blendLocal<N>(params, banks, Layer0 + k * 8 + j) * (k == 8 ? 1.f : a[k]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = nirc_hybrid::activate(b[j]);
#pragma unroll
    for (int k = 0; k <= 8; ++k) {
#pragma unroll
        for (int j = 0; j < 3; ++j) y[j] += blendLocal<N>(params, banks, Layer0 + Layer1 + k * 3 + j) * (k == 8 ? 1.f : b[k]);
    }
    if constexpr (Backward)
    {
        float d2[3], d1[8] = {}, d0[8] = {};
        for (int j = 0; j < 3; ++j) d2[j] = float(dy[q * stride + j]);
        for (int k = 0; k <= 8; ++k)
            for (int j = 0; j < 3; ++j) {
                int index = Layer0 + Layer1 + k * 3 + j;
                if (k < 8) d1[k] += blendLocal<N>(params, banks, index) * d2[j];
                scatterLocal<N>(gradients, banks, index, (k == 8 ? 1.f : b[k]) * d2[j]);
            }
        for (int k = 0; k < 8; ++k) d1[k] *= nirc_hybrid::derivative(b[k]);
        for (int k = 0; k <= 8; ++k)
            for (int j = 0; j < 8; ++j) {
                int index = Layer0 + k * 8 + j;
                if (k < 8) d0[k] += blendLocal<N>(params, banks, index) * d1[j];
                scatterLocal<N>(gradients, banks, index, (k == 8 ? 1.f : a[k]) * d1[j]);
            }
        for (int k = 0; k < 8; ++k) d0[k] *= nirc_hybrid::derivative(a[k]);
        for (int k = 0; k <= Input; ++k) {
            float dx = 0;
            for (int j = 0; j < 8; ++j) {
                if (k < Surface) dx += blendLocal<N>(params, banks, k * 8 + j) * d0[j];
                scatterLocal<N>(gradients, banks, k * 8 + j, (k == Input ? 1.f : x[k]) * d0[j]);
            }
            if (k >= L::RawNormal && k < Surface) dencoded[q * L::Stride + k-L::RawNormal] = __float2half(dx);
        }
    }
    else {
      for (int j = 0; j < 3; ++j) {
        if (output) output[q * stride + j] = __float2half(y[j]);
        if (prediction) {
            if (fused.arr0) {
                auto id = reinterpret_cast<const uint32_t*>(fused.arr1)[q];
                float w = reinterpret_cast<const float*>(fused.arr0)[q * 3 + j];
                atomicAdd(prediction + id * 3 + j, fmaxf(y[j], 0.f) * w);
            } else prediction[q * 3 + j] = y[j];
        }
    }
    }
}

class LocalNetworkBase : public tcnn::Network<float, __half> {
public:
    virtual size_t extraGradientBytes() const = 0;
    virtual int inputWidth() const = 0;
};

template<int N> class NetworkImpl final : public LocalNetworkBase
{
public:
    using L = Layout<N>;
    static constexpr int Input = L::Input, Surface = L::Surface, MlpParams = L::MlpParams;
    explicit NetworkImpl(const tcnn::json& config)
    {
        auto surface = config.at("encoding");
        // Direction is evaluated in the decoder, just as in original inference.
        if (surface["nested"].size() != 5 || surface["nested"][4]["degree"] != L::SHDegree)
            throw std::runtime_error("OriginalHashLocal expects the original five-channel encoding");
        surface["nested"].erase(4);
        m_encoding.reset(tcnn::create_encoding<__half>(9, surface));
        if (m_encoding->offsets().back() != L::EncodedSurface)
            throw std::runtime_error("OriginalHashLocal surface layout mismatch, got " + std::to_string(m_encoding->num_encoded_dims()) + " offsets " + tcnn::json(m_encoding->offsets()).dump());
        auto local = config.at("local_mlp");
        m_cfg.cells = local.value("mlp_grid_cells", 4);
        if (m_cfg.cells < 0 || m_cfg.cells > 16) throw std::runtime_error("Invalid local MLP grid");
        auto interpolation = local.value("spatial_interpolation", std::string("tetrahedral"));
        if (interpolation != "tetrahedral" && interpolation != "trilinear") throw std::runtime_error("Invalid local MLP interpolation");
        m_cfg.tetrahedral = interpolation == "tetrahedral";
        m_cfg.banks = m_cfg.cells ? (m_cfg.cells+1)*(m_cfg.cells+1)*(m_cfg.cells+1) : 1;
        m_cfg.mlpCount = m_cfg.banks * MlpParams;
        m_encodingOffset = (m_cfg.mlpCount + 1u) & ~1u;
        m_gradFloat.resize(m_cfg.mlpCount);
    }
    size_t n_params() const override { return m_encodingOffset + m_encoding->n_params(); }
    int inputWidth() const override { return Input; }
    size_t extraGradientBytes() const override { return m_gradFloat.get_bytes(); }
    const nirc_hybrid::Settings& settings() const { return m_cfg; }
    size_t encodingOffset() const { return m_encodingOffset; }
    uint32_t padded_output_width() const override { return 16; }
    uint32_t output_width() const override { return 3; }
    uint32_t required_input_alignment() const override { return 1; }
    uint32_t width(uint32_t) const override { return Hidden; }
    uint32_t num_forward_activations() const override { return 0; }
    const __half* forward_activations(uint32_t) const override { return nullptr; }
    std::vector<std::pair<uint32_t,uint32_t>> layer_sizes() const override { return {{uint32_t(m_cfg.mlpCount),1}}; }
    void initialize_params(tcnn::pcg32& rng, float* full, __half* params, __half* inference, __half* backward,
        __half* gradients, float scale = 1) override
    {
        m_params=params; m_inference=inference; m_gradients=gradients;
        std::vector<float> v(m_encodingOffset);
        int sizes[3]={Input,8,8}, outs[3]={8,8,3}, offset=0;
        for(int l=0;l<3;++l) {
            for(int i=0;i<=sizes[l];++i) for(int j=0;j<outs[l];++j) {
                float w=i==sizes[l]?0.f:(rng.next_float()*2-1)*sqrtf(6.f/(sizes[l]+outs[l]))*scale;
                for(int bank=0;bank<m_cfg.banks;++bank) v[bank*MlpParams+offset+i*outs[l]+j]=w;
            }
            offset+=(sizes[l]+1)*outs[l];
        }
        CUDA_CHECK_THROW(cudaMemcpy(full,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));
        auto n=m_encodingOffset;
        m_encoding->initialize_params(rng,full+n,params+n,inference+n,backward+n,gradients+n,scale);
    }
    void inference(cudaStream_t stream,tcnn::NeuralInputData<float>& input,tcnn::GPUMatrix<float>& out,
        uint32_t n,int*,float* bb0,float* bb1,tcnn::FusedOutputData fused={}) override
    {
        if(!n) return;
        m_encoding->encode(stream,n,input,&m_separate,{},nullptr,true,nullptr,bb0,bb1);
        Encoded<N> encoded;
        for(int c=0;c<4;++c) { encoded.data[c]=m_separate.data[c].data(); encoded.map[c]=m_separate.mappings[c]; encoded.stride[c]=m_separate.data[c].rows(); }
        evaluate<N, false><<<(n+127)/128,128,0,stream>>>(m_cfg,describe(input),encoded,n,m_inference,out.data(),nullptr,0,fused,nullptr,nullptr,nullptr);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    void forward(cudaStream_t stream,tcnn::NeuralInputData<float>& input,uint32_t n,int*,float* bb0,float* bb1,
        tcnn::GPUMatrixDynamic<__half>* out=nullptr,bool inference=false,bool inputGrad=false) override
    {
        if(inputGrad || !out || out->layout()!=tcnn::CM) throw std::runtime_error("Invalid local MLP forward request");
        m_count=n; m_inputs=describe(input);
        if(m_encoded.size()<size_t(n)*L::Stride) {m_encoded.resize(size_t(n)*L::Stride);m_dencoded.resize(size_t(n)*L::Stride);}
        CUDA_CHECK_THROW(cudaMemsetAsync(out->data(),0,out->n_bytes(),stream));
        if(!n) return;
        m_encoding->encode(stream,n,input,nullptr,{m_encoded.data(),L::Stride},nullptr,inference,nullptr,bb0,bb1);
        Encoded<N> encoded; encoded.merged=m_encoded.data();
        evaluate<N, false><<<(n+127)/128,128,0,stream>>>(m_cfg,m_inputs,encoded,n,inference?m_inference:m_params,nullptr,out->data(),out->m(),{},nullptr,nullptr,nullptr);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    void backward(cudaStream_t stream,tcnn::NeuralInputData<float>& input,const tcnn::GPUMatrixDynamic<__half>&,
        const tcnn::GPUMatrixDynamic<__half>& dy,tcnn::GPUMatrix<float>* inputGrad=nullptr,bool inference=false,bool compute=true) override
    {
        if(inputGrad) throw std::runtime_error("Local MLP input derivatives are unsupported");
        if(!compute || !m_count) return;
        CUDA_CHECK_THROW(cudaMemsetAsync(m_gradFloat.data(),0,m_gradFloat.get_bytes(),stream));
        Encoded<N> encoded; encoded.merged=m_encoded.data();
        evaluate<N, true><<<(m_count+127)/128,128,0,stream>>>(m_cfg,m_inputs,encoded,m_count,inference?m_inference:m_params,nullptr,nullptr,dy.m(),{},dy.data(),m_gradFloat.data(),m_dencoded.data());
        nirc_hybrid::castGradients<<<(m_cfg.mlpCount+255)/256,256,0,stream>>>(m_cfg.mlpCount,m_gradFloat.data(),m_gradients);
        if(m_encodingOffset>size_t(m_cfg.mlpCount)) CUDA_CHECK_THROW(cudaMemsetAsync(m_gradients+m_cfg.mlpCount,0,sizeof(__half),stream));
        m_encoding->backward(stream,m_count,{m_dencoded.data(),L::Stride},nullptr,{},input,false);
        CUDA_CHECK_THROW(cudaGetLastError());
    }
    void inference(cudaStream_t,const tcnn::GPUMatrix<float>&,tcnn::GPUMatrix<float>&,tcnn::GPUMatrix<float>* =nullptr) override {throw std::runtime_error("Expected five NIRC channels");}
    void forward(cudaStream_t,const tcnn::GPUMatrix<float>&,tcnn::GPUMatrixDynamic<__half>* =nullptr,bool=false,bool=false) override {throw std::runtime_error("Expected five NIRC channels");}
    void backward(cudaStream_t,const tcnn::GPUMatrix<float>&,const tcnn::GPUMatrixDynamic<__half>&,const tcnn::GPUMatrixDynamic<__half>&,tcnn::GPUMatrix<float>* =nullptr,bool=false,bool=true) override {throw std::runtime_error("Expected five NIRC channels");}
private:
    static nirc_hybrid::Inputs describe(const tcnn::NeuralInputData<float>& input)
    {
        if(input.num_channels()!=5) throw std::runtime_error("Expected five NIRC channels");
        nirc_hybrid::Inputs result{};
        for(int c=0;c<5;++c) {result.data[c]=input.data[c].data();result.map[c]=input.mappings[c];result.stride[c]=input.data[c].rows();}
        return result;
    }
    nirc_hybrid::Settings m_cfg{};
    nirc_hybrid::Inputs m_inputs{};
    size_t m_encodingOffset=0;
    uint32_t m_count=0;
    std::unique_ptr<tcnn::Encoding<__half>> m_encoding;
    tcnn::NeuralInputData<__half> m_separate;
    tcnn::GPUMemory<__half> m_encoded,m_dencoded;
    tcnn::GPUMemory<float> m_gradFloat;
    __half *m_params=nullptr,*m_inference=nullptr,*m_gradients=nullptr;
};

using Network = NetworkImpl<64>;
inline std::shared_ptr<LocalNetworkBase> createNetwork(const tcnn::json& config) {
    int input = config.at("local_mlp").value("input_dims", 64);
    if (input == 64) return std::make_shared<NetworkImpl<64>>(config);
    if (input == 32) return std::make_shared<NetworkImpl<32>>(config);
    if (input == 23) return std::make_shared<NetworkImpl<23>>(config);
    throw std::runtime_error("Local MLP input_dims must be 64, 32, or 23");
}
}
