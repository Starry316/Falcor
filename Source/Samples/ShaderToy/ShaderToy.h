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
#include "Core/Pass/ComputePass.h"
#include "Utils/Texture/Synthesis.h"
#include "Utils/Neural/NBTF.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Neural/cuda/CUDADefines.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace Falcor;
enum class RenderType : uint32_t
{
    // RT,
    // WAVEFRONT_SHADER_NN,
    SHADER_NN,
    CUDA,
    CUDAFP16,
    CUDAINT8
    // TEST
    // DEBUG_MIP
};

FALCOR_ENUM_INFO(
    RenderType,
    {
        { RenderType::SHADER_NN, "Shader Inference" },
        { RenderType::CUDA, "CUDA FP32 Inference" },
        { RenderType::CUDAFP16, "CUDA FP16 Inference" },
        { RenderType::CUDAINT8, "CUDA INT8 Inference" }
        // { RenderType::TEST, "CUDA TEST" }
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
    ref<ComputePass> mpBindInputPass;
    ref<ComputePass> mpDebugPass;
    float mUVScale = 1.0f;
    std::unique_ptr<NBTF> mpNBTFInt8;
    std::unique_ptr<TextureSynthesis> mpTextureSynthesis;

#ifdef PEBBLE
    std::string mNetInt8Name = "pebble_m32u8h8d8_int8";
    std::string mShellHFFileName = "ganges_river_pebbles_disp_4k.png";
    std::string mHFFileName = "ganges_river_pebbles_disp_4k.png";
    bool mHDRBTF = false;
#endif

#ifdef LEATHER
    std::string mNetInt8Name = "leather11_m32u8h8d8_int8";
    std::string mShellHFFileName = "ubo/leather11.png";
    std::string mHFFileName = "ubo/leather11.png";
    bool mHDRBTF = false;
#endif

#ifdef LEATHER10
    std::string mNetInt8Name = "leather10_m32u8h8d8_int8";
    std::string mShellHFFileName = "ubo/leather11.png";
    std::string mHFFileName = "ubo/leather11.png";
    bool mHDRBTF = false;
#endif

#ifdef LEATHER_04R
    std::string mNetInt8Name = "leather04r_m32u8h8d8_int8";
    std::string mShellHFFileName = "ubo/leather04.png";
    std::string mHFFileName = "ubo/leather04.png";
    bool mHDRBTF = false;
#endif



#ifdef METAL
    std::string mNetInt8Name = "metal_m32u8h8d8_int8";
    std::string mShellHFFileName = "metal_grate_rusty_disp_4k.png";
    std::string mHFFileName = "metal_grate_rusty_disp_4k.png";
    bool mHDRBTF = true;
#endif


#ifdef METAL2
    std::string mNetInt8Name = "metal2_m32u8h8d8_int8";
    std::string mShellHFFileName = "MetalGoldHammered001_DISP_6K.png";
    std::string mHFFileName = "MetalGoldHammered001_DISP_6K.png";
    bool mHDRBTF = true;
#endif


#ifdef BRICK
    std::string mNetInt8Name = "brick_m32u8h8d8_int8";
    std::string mShellHFFileName = "castle_brick_02_red_disp_4k.png";
    std::string mHFFileName = "castle_brick_02_red_disp_4k.png";
    bool mHDRBTF = false;
#endif


#ifdef TILE
    std::string mNetInt8Name = "tile_m32u8h8d8_int8";
    std::string mShellHFFileName = "TilesCeramicFishscale002_DISP_6k.jpg";
    std::string mHFFileName = "TilesCeramicFishscale002_DISP_6k.jpg";
    bool mHDRBTF = false;
#endif



#ifdef TILE2
    std::string mNetInt8Name = "tile2_m32u8h8d8_int8";
    std::string mShellHFFileName = "roof_tiles_14_disp_1k.png";
    std::string mHFFileName = "roof_tiles_14_disp_1k.png";
    bool mHDRBTF = false;
#endif

#ifdef FABRIC
    std::string mNetInt8Name = "fabric_m32u8h8d8_int8";
    std::string mShellHFFileName = "FabricWeaveWooly001_DISP_4K.jpg";
    std::string mHFFileName = "FabricWeaveWooly001_DISP_4K.jpg";
    bool mHDRBTF = false;
#endif


#ifdef WEAVE
    std::string mNetInt8Name = "weave_m32u8h8d8_int8";
    std::string mShellHFFileName = "WickerWeavesBrownRattan001_DISP_6K.jpg";
    std::string mHFFileName = mShellHFFileName;
    bool mHDRBTF = false;
#endif


#ifdef DUMMY
    std::string mNetInt8Name = "Dummy";
    std::string mShellHFFileName = "roof_tiles_14_disp_1k.png";
    std::string mHFFileName = "roof_tiles_14_disp_1k.png";
    bool mHDRBTF = false;
#endif


    uint mFrames = 1;

    bool mSynthesis = false;
    bool mFP16 = true;
    bool mShowShader = false;
    bool mDebugMLP = true;

    RenderType mRenderType = RenderType::CUDA;

    // cuda
    float mCudaTime = 0.0;
    double mCudaAvgTime = 0.0;
    cudaEvent_t mCudaStart, mCudaStop;
    ref<Texture> mpOutColor;
    ref<Buffer> mpTestInput;
    ref<Buffer> mpOutputBuffer;




    int mCudaInferTimes = 1;
    Falcor::float2 mWo = { 0.0f, 0.0f };
    Falcor::float2 mWi = { 0.0f, 0.0f };

};
