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
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Texture/Synthesis.h"
#include "Utils/Neural/MLP.h"
#include "Utils/Neural/NBTF.h"
#include "Utils/Neural/MLPCuda.h"
#include "Utils/Neural/cuda/CUDADefines.h"
#include "Rendering/Lights/EnvMapSampler.h"

// #include "NvInfer.h"
// #include "NvOnnxParser.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// #include <nvrtc.h>

// using namespace nvinfer1;
// using namespace nvonnxparser;
using namespace Falcor;

enum class InferType : uint32_t
{
    SHADERTP,
    SHADER,
    CUDA,
    CUDAFP16,
    CUDAINT8
};

FALCOR_ENUM_INFO(
    InferType,
    {{InferType::SHADERTP, "Shader TP Inference"},
     {InferType::SHADER, "Shader Inference"},
     {InferType::CUDA, "CUDA FP32 Inference"},
     {InferType::CUDAFP16, "CUDA FP16 Inference"},
     {InferType::CUDAINT8, "CUDA INT8 Inference"}}
);
FALCOR_ENUM_REGISTER(InferType);

enum class RenderType : uint32_t
{
    RT,
    WAVEFRONT_SHADER_NN
    // SHADER_NN,
    // CUDA,
    // TRT,
    // DEBUG_MIP
};

FALCOR_ENUM_INFO(
    RenderType,
    {
        {RenderType::RT, "RT"},
        {RenderType::WAVEFRONT_SHADER_NN, "Wavefront Inference"}
        // { RenderType::SHADER_NN, "Shader Inference" },
        // { RenderType::CUDA, "CUDA Inference" },
        // { RenderType::TRT, "TensorRT Inference" },
        // { RenderType::DEBUG_MIP, "DEBUG MIP" }
    }
);
FALCOR_ENUM_REGISTER(RenderType);

/**
 * Minimal path tracer.
 *
 * This pass implements a minimal brute-force path tracer. It does purposely
 * not use any importance sampling or other variance reduction techniques.
 * The output is unbiased/consistent ground truth images, against which other
 * renderers can be validated.
 *
 * Note that transmission and nested dielectrics are not yet supported.
 */
class HFTracing : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HFTracing, "HFTracing", "Minimal path tracer.");

    static ref<HFTracing> create(ref<Device> pDevice, const Properties& props) { return make_ref<HFTracing>(pDevice, props); }

    HFTracing(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderHF(RenderContext* pRenderContext, const RenderData& renderData);
    void nnInferPass(RenderContext* pRenderContext, const RenderData& renderData);
    void cudaInferPass(RenderContext* pRenderContext, const RenderData& renderData);

    void displayPass(RenderContext* pRenderContext, const RenderData& renderData);
    void handleOutput();
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    // virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return mpPixelDebug->onMouseEvent(mouseEvent); }
    bool onMouseEvent(const MouseEvent& mouseEvent);
    // virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    bool onKeyEvent(const KeyboardEvent& keyEvent) ;

private:
    void parseProperties(const Properties& props);
    void prepareVars();

    // Internal state

    /// Current scene.
    ref<Scene> mpScene;
    /// GPU sample generator.
    ref<SampleGenerator> mpSampleGenerator;

    // Configuration

    /// Max number of indirect bounces (0 = none).
    uint mMaxBounces = 3;
    /// Compute direct illumination (otherwise indirect only).
    bool mComputeDirect = true;
    /// Use importance sampling for materials.
    bool mUseImportanceSampling = true;

    // Runtime data

    /// Frame count since scene was loaded.
    uint mFrameCount = 0;
    bool mOptionsChanged = false;

    // Ray tracing program.
    struct
    {
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;
    } mTracer;

    ref<ComputePass> mpVisualizeMapsPass;
    ref<ComputePass> mpCreateMaxMipPass;
    ref<ComputePass> mpInferPass;
    ref<ComputePass> mpDisplayPass;
    ref<ComputePass> mpGenerateGeometryMapPass;
    // Texture inputs
    std::string mMediaPath = getProjectDirectory().string();

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

#ifdef TEST_MULTI
    std::string mNetInt8Name[2] = {"tile4_small_m32u8h8d8_int8" , "weave_small_m32u8h8d8_int8"};
    std::string mShellHFFileName[2] = { "Tiles11_DISP_3K_sml.jpg","WickerWeavesBrownRattan001_DISP_6K_small.jpg"};

    // std::string mNetInt8Name[2] = {"tile4_small_m32u8h8d8_int8" , "leather11_tile_m32u8h8d8_int8"};
    // std::string mShellHFFileName[2] = { "Tiles11_DISP_3K_sml.jpg","ubo/leather11_tile.png"};


    //std::string mNetInt8Name[2] = {"leather11_m32u8h8d8_int8", "metal2_m32u8h8d8_int8"};
    //std::string mShellHFFileName[2] = {"ubo/leather11.png", "MetalGoldHammered001_DISP_6K.png"};
    //std::string mNetInt8Name[2] = {"metal2_m32u8h8d8_int8", "leather11_m32u8h8d8_int8"};
    //std::string mShellHFFileName[2] = {"MetalGoldHammered001_DISP_6K.png", "ubo/leather11.png"};
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

#ifdef FABRIC09
    std::string mNetInt8Name = "fabric09_m32u8h8d8_int8";
    std::string mShellHFFileName = "ubo/fabric09.png";
    std::string mHFFileName = "ubo/fabric09.png";
    bool mHDRBTF = false;
#endif

#ifdef FABRIC10
    std::string mNetInt8Name = "fabric10_m32u8h8d8_int8";
    std::string mShellHFFileName = "ubo/fabric09.png";
    std::string mHFFileName = "ubo/fabric09.png";
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

#ifdef METAL3
    std::string mNetInt8Name = "metal3_m32u8h8d8_int8";
    std::string mShellHFFileName = "MetalPlateDiamondQuad001_DISP_4K_SPECULAR.png";
    std::string mHFFileName = "MetalPlateDiamondQuad001_DISP_4K_SPECULAR.png";
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

#ifdef TILE3
    std::string mNetInt8Name = "tile3_m32u8h8d8_int8";
    std::string mShellHFFileName = "TilesCeramicChevron001_DISP_6K.jpg";
    std::string mHFFileName = "TilesCeramicChevron001_DISP_6K.jpg";
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

    // std::string mNetName = "leather11_m32u16h8d8";
    // std::string mNetName = "tile2_m32u16h8d8";
    // std::string mNetName = "metal2_m32u16h8d8";
    std::string mNetName = "weave_m32u16h8d8";
    ref<Texture> mpHitBuffer;

    ref<Texture> mpHF;
    ref<Texture> mpShellHF[2];
    ref<Texture> mpHFMaxMip;
    ref<Texture> mpColor;
    ref<Texture> mpNormalMap;
    ref<Texture> mpTangentMap;
    ref<Texture> mpPosMap;
    ref<Texture> mpEditMap;

    ref<Sampler> mpMaxSampler;

    std::unique_ptr<PixelDebug> mpPixelDebug;

    Falcor::float4 mControlParas = Falcor::float4(1, 0.6, 0, 0.099);
    Falcor::float4 mCurvatureParas = Falcor::float4(0.15, 0.1, 10, 0.3); // z - 10
    Falcor::float4 mLightZPR = Falcor::float4(0.056, 1, 0.15, 0.1);
    float mPatchScale = 1.0f;
    uint mTriID = 0;
    uint mMaxSteps = 1000;
    uint mMaxTriCount = 1000;
    Falcor::uint2 mSelectedPixel = {19200, 10800};
    Falcor::uint2 mFrameDim = {0, 0};
    uint mMatId = 0;
    ACFCurve mCurveType = ACFCurve::X;
    RenderType mRenderType = RenderType::WAVEFRONT_SHADER_NN;
    InferType mInferType = InferType::CUDAINT8;

    Falcor::float2 point_data[5] = {
        Falcor::float2(0.0f, 1.0f), Falcor::float2(0.0f, 1.0f), Falcor::float2(1.0f, 0.0f), Falcor::float2(1.0f, 0.0f), Falcor::float2(0.0f, 0.0f)
    };

    bool mContactRefinement = false;
    bool mMipGenerated = false;
    bool mApplySyn = true;
    bool mNNInfer = true;
    bool mScaleUV = false;
    bool mHFBound = true;
    bool mLocalFrame = true;
    bool mCudaInfer = true;
    bool mUseFP16 = false;
    bool mMLPDebug = false;
    bool mOutputingVideo = false;
    bool mEnableEdit = false;
    bool mUseEditMap = false;

    bool mRefresh = false;
    bool mPaint = false;
    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;
    ref<Fence> mpFence1;
    ref<Fence> mpFence2;

    std::unique_ptr<TextureSynthesis> mpTextureSynthesis[2];
    std::unique_ptr<NBTF> mpNBTFInt8[2];
    std::unique_ptr<NBTF> mpNBTF;

    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    uint mDebugPrism = 0;
    bool mShowTracedHF = false;
    bool mTracedShadowRay = true;
    bool mUseMIS = false;

    // output
    uint32_t mOutputSPP = 100;
    uint32_t mOutputIndx = 0;
    std::string mOutputPath = "D:/video/{}.png";
    Falcor::float3 mEnvRotAngle = Falcor::float3(0.0f, 0.0f, 0.0f);
    Falcor::float3 mOriginEnvRotAngle = Falcor::float3(0.0f, 0.0f, 0.0f);
    uint mOutputStep = 4;

    // camera controll
    Falcor::float3 mCameraPos = Falcor::float3(0.0f, 0.0f, 0.0f);
    Falcor::float3 mCameraTarget = Falcor::float3(0.0f, 0.0f, 0.0f);


    // cuda
    float mCudaTime = 0.0;
    float mPhi = 0.0;
    float mBrushSize = 10;
    double mCudaAvgTime = 0.0;
    int cudaInferTimes = 1;
    cudaEvent_t mCudaStart, mCudaStop;
    ref<Buffer> mpOutputBuffer;
    ref<Buffer> mpVaildBuffer;
    ref<Buffer> mpPackedInputBuffer;
    ref<Buffer> mpHashedUVBuffer;
    ref<Buffer> mpSelectBuffer;

    ref<Buffer> mpSelectUVBuffer;

    uint mCudaAccumulatedFrames = 1;
};
