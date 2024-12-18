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
#pragma once
#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/FullScreenPass.h"
#include "Utils/Texture/Synthesis.h"
#include "Utils/Neural/NBTF.h"
#include "Utils/Debug/PixelDebug.h"
#include "cuda/MLPInference.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace Falcor;
enum class RenderType : uint32_t
{
    // RT,
    // WAVEFRONT_SHADER_NN,
    SHADER_NN,
    CUDA,
    CUDAFP16
    // TRT,
    // DEBUG_MIP
};

FALCOR_ENUM_INFO(
    RenderType,
    {
        // { RenderType::RT, "RT" },
        // { RenderType::WAVEFRONT_SHADER_NN, "Wavefront Inference" },
        { RenderType::SHADER_NN, "Shader Inference" },
        { RenderType::CUDA, "CUDA Inference" },
        { RenderType::CUDAFP16, "CUDA FP16 Inference" }
        // { RenderType::TRT, "TensorRT Inference" },
        // { RenderType::DEBUG_MIP, "DEBUG MIP" }
    }
);
FALCOR_ENUM_REGISTER(RenderType);

class ShaderToy : public SampleApp
{
public:
    ShaderToy(const SampleAppConfig& config);
    ~ShaderToy();

    void onLoad(RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return mpPixelDebug->onMouseEvent(mouseEvent); }
    void shaderInfer(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void cudaInfer(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void bindInput(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void display(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
private:
    std::unique_ptr<PixelDebug> mpPixelDebug;
    ref<Sampler> mpLinearSampler;
    float mAspectRatio = 0;
    ref<RasterizerState> mpNoCullRastState;
    ref<DepthStencilState> mpNoDepthDS;
    ref<BlendState> mpOpaqueBS;
    ref<FullScreenPass> mpMainPass;
    ref<FullScreenPass> mpDisplayPass;
    ref<FullScreenPass> mpBindInputPass;
    float mUVScale = 1.0f;
    std::unique_ptr<NBTF> mpNBTF;
    std::unique_ptr<NBTF> mpNBTFInt8;
    std::unique_ptr<TextureSynthesis> mpTextureSynthesis;
    std::string mNetName = "leather11_m24u8h8d8";
    std::string mNetInt8Name = "leather11_m24u8h8d8_int8";
    uint mFrames = 1;

    bool mSynthesis = false;
    bool mFP16 = true;

    RenderType mRenderType = RenderType::CUDA;

    // cuda
    float mCudaTime = 0.0;
    double mCudaAvgTime = 0.0;
    cudaEvent_t mCudaStart, mCudaStop;
    ref<Buffer> mpInputBuffer;
    ref<Buffer> mpOutputBuffer;

    ref<Buffer> mpWeightBuffer;
    ref<Buffer> mpBiasBuffer;


    ref<Buffer> mpWeightFP16Buffer;
    ref<Buffer> mpBiasFP16Buffer;

    cudaTextureObject_t mTexObj;

    ref<Buffer> mpQInt8Buffer;

    int mDebugOffset = 0;

};
