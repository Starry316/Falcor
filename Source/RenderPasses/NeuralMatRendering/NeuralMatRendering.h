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
#include "Rendering/Lights/EnvMapSampler.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Texture/Synthesis.h"
#include "Utils/Neural/MLP.h"
#include "Utils/Neural/NBTF.h"
#include "Utils/Neural/MLPCuda.h"
#include "Utils/Neural/NNMat.h"
#include "Utils/Neural/cuda/CUDADefines.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

using namespace Falcor;

enum class ModelName : uint32_t
{
    LEATHER11,
    WEAVE,
    TILE,
    CERAMIC_TILE
};

FALCOR_ENUM_INFO(
    ModelName,
    {{ModelName::LEATHER11, "UBO Leather11"},
     {ModelName::WEAVE, "Weave"},
     {ModelName::TILE, "Tile"},
     {ModelName::CERAMIC_TILE, "Ceramic Tile"}}
);
FALCOR_ENUM_REGISTER(ModelName);

enum class NeuMat : uint32_t
{
    LEATHER11BL,
    LEATHER11,
    FABRIC12,
    CARPET11
};

FALCOR_ENUM_INFO(
    NeuMat,
    {{NeuMat::LEATHER11BL, "Leather11 Baseline"},
     {NeuMat::LEATHER11, "Leather11"},
     {NeuMat::FABRIC12, "Fabric12"},
     {NeuMat::CARPET11, "Carpet11"}}
);
FALCOR_ENUM_REGISTER(NeuMat);

struct ModelInfo
{
    std::string name;
    std::string hfName;
    bool HDRBTF;
    float scales[8]; // quantization scales
};

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
class NeuralMatRendering : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralMatRendering, "NeuralMatRendering", "Minimal path tracer.");

    static ref<NeuralMatRendering> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<NeuralMatRendering>(pDevice, props);
    }

    NeuralMatRendering(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return mpPixelDebug->onMouseEvent(mouseEvent); }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    void tracingPass(RenderContext* pRenderContext, const RenderData& renderData);
    void displayPass(RenderContext* pRenderContext, const RenderData& renderData);
    void loadNetwork(RenderContext* pRenderContext);

private:
    NeuMat mNeuMat = NeuMat::LEATHER11;

    const std::string mNeuMatPath[7] = {
        "leather11_Validation",
        "leather11_XYZ_BTFNetXYZHU72x2",
        "fabric12_XYZ_BTFNetXYZHU72x2",
        "carpet11_XYZ_BTFNetXYZHU72x2",
        "",
        "",
        ""};

    const std::string mNeuIBLPath[7] = {
        "leather11_45_IBL_BTFNetIBL2x2",
        "leather11_45_IBL_BTFNetIBL2x2",
        "fabric12_45_IBL_BTFNetIBL2x2",
        "fabric12_45_IBL_BTFNetIBL2x2",
        "",
        "",
        ""};

    const bool mIsHisto[7] = {0, 0, 0, 0, 0, 0, 0};
    const bool mIsWi[7] = {0, 1, 1, 1, 1, 1, 1};

    // std::string mNNIBLName = "leather11_45_IBL_BTFNetIBL2x2";
    std::string mNNIBLName = "fabric12_45_IBL_BTFNetIBL2x2";

    // const std::string mNeuMatPath[7] = {
    //    "leather11_Validation","leather11_Validation_BTFNetWi2x2",
    //     "leather11_Histo_wi","leather11_XYZ_BTFNetXYZH2x2","fabric12_XYZ_BTFNetXYZHU72x2",
    //     "carpet11_XYZ_BTFNetXYZHU72x2", "fabric07_remote"
    // };
    // const bool mIsHisto[7] = {0, 0, 1, 0, 0, 0, 0};
    // const bool mIsWi[7] =    {0, 1, 0, 1, 1, 1, 0};

    bool mUseIBL = false;
    void prepareVars();

    /// Current scene.
    ref<Scene> mpScene;
    /// GPU sample generator.
    ref<SampleGenerator> mpSampleGenerator;
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
    ref<ComputePass> mpDisplayPass;
    std::string mProjectPath = getProjectDirectory().string();

    ModelName mModelName = ModelName::LEATHER11;

    ModelInfo mModelInfo[4] = {

        // {"leather11_m32u8h8d8_int8",
        {"leather11_int8",
         "leather11.png",
         false,
         {0.003400295041501522,
          1.1354546586517245e-05,
          0.0024283595848828554,
          1.047514069796307e-05,
          0.0021721271332353354,
          1.9848570445901714e-05,
          0.0016346105840057135,
          1.605643228685949e-05}},

        {"leather11_int8",
         "leather11.png",
         false,
         {0.003400295041501522,
          1.1354546586517245e-05,
          0.0024283595848828554,
          1.047514069796307e-05,
          0.0021721271332353354,
          1.9848570445901714e-05,
          0.0016346105840057135,
          1.605643228685949e-05}},

        // {"weave_int8",
        //  "weave.jpg",
        //  false,
        //  {0.002025123918429017,
        //   7.385711796814576e-06,
        //   0.0017646728083491325,
        //   1.3128001228324138e-05,
        //   0.0012104109628126025,
        //   1.130689270212315e-05,
        //   0.001690503559075296,
        //   2.1813480998389423e-05}},

        {"tile2_int8",
         "tile2.png",
         true,
         {0.0021086351480334997,
          6.1559362620755564e-06,
          0.0015336197102442384,
          5.158955445949687e-06,
          0.0009424724266864359,
          4.815707598027075e-06,
          0.0011210687225684524,
          6.463145837187767e-06}},

        {"tile_int8",
         "tile.jpg",
         true,
         {0.0025666167493909597,
          7.262530743901152e-06,
          0.0012831123312935233,
          4.6556156121368986e-06,
          0.0010884717339649796,
          5.779493676527636e-06,
          0.000891408184543252,
          4.09871927331551e-06}}};

    // displacement map
    ref<Texture> mpHF;
    // max filter sampler for HF texel fetch.
    ref<Sampler> mpPointSampler;
    std::unique_ptr<PixelDebug> mpPixelDebug;
    // cuda inference output buffer
    ref<Buffer> mpOutputBuffer;
    ref<Buffer> mpValidBuffer;
    ref<Buffer> mpPackedInputBuffer;
    ref<Buffer> mpScaleBuffer;

    Falcor::float4 mControlParas = Falcor::float4(0.1, 1, 0.3, 0.76);

    ACFCurve mCurveType = ACFCurve::X;
    Falcor::float2 point_data[5] = {
        Falcor::float2(0.0f, 1.0f),
        Falcor::float2(0.0f, 1.0f),
        Falcor::float2(1.0f, 0.0f),
        Falcor::float2(1.0f, 0.0f),
        Falcor::float2(0.0f, 0.0f)};

    float point_data_curve[1] = {0};
    Falcor::float3 lightPos = Falcor::float3(2, 15, 2);
    float lightIntensity = 200.0f;
    float lightR = 8.0f;

    float lightPhi = 0.0;

    bool mApplySyn = true;
    bool mShowGT = false;
    bool mUsePointLight = true;

    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;

    std::unique_ptr<TextureSynthesis> mpTextureSynthesis;
    std::shared_ptr<NBTF> mpNBTFInt8;

    std::shared_ptr<NBTF> mpNBTF[4];

    std::shared_ptr<NNMat> mpNNMat;
    std::shared_ptr<NNMat> mpNNMatIBL;

    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    bool mShowTracedHF = false;
    bool mTracedShadowRay = true;
    bool mHDRBTF = false;

    bool mShowFeatureMap = false;
    int  mFeatureLevel = 0;

    Falcor::float3 mEnvRotAngle = Falcor::float3(0.0f, 0.0f, 0.0f);
    // cuda
    float mCudaTime = 0.0;
    double mCudaAvgTime = 0.0;
    int cudaInferTimes = 1;
    cudaEvent_t mCudaStart, mCudaStop;

    uint mCudaAccumulatedFrames = 1;


    std::shared_ptr<NNMat> mpNNMatT;
    std::shared_ptr<NNMat> mpNNMatBTF;
    // std::string mNeuBTFName = "NeuBTF_MultiPlane5LReLURegFliterXYZDirXYZ_fur_U400-4_H100-8_D100-8_h32_bt1_L1_pretrain";
    // std::string mNeuBTFName = "NeuBTF_MultiPlane5LReLURegFliterXYZDirXYZ_fur_U400-8_H100-8_D100-8_h32_bt1_L1_pretrain";
    // std::string mNeuBTFName = "NeuBTF_MultiPlane5LReLURegFliterXYZDirXYZTest_test_U400-8_H100-8_D100-8_h32_bt1_L1_pretrain";
    // std::string mNeuBTFName = "NeuBTF_MultiPlane5LReLURegFliterXYZ_fur_U400-8_H100-8_D100-8_h32_bt1_L1_pretrain";
    // std::string mNeuBTFName = "NeuLF_P5TB_peb_U400-8_H100-8_D100-8_h32_bt1_L1_Filter";
    // std::string mNeuBTFName = "NeuLF_P5TB_peball_U800-8_H100-8_D100-8_h32_bt1_L1_Filter";
    // std::string mNeuBTFName = "NeuLF_P5TB_bunnysp_U800-8_H100-8_D100-8_h48_bt1_L1_Filter";
    // std::string mNeuBTFName = "NeuLF_P5TB_bunnydisk_U800-8_H100-8_D100-8_h48_bt1_L1_Filter";
    // std::string mNeuBTFName = "NeuLF_P5TPos_furball_U600-8_H100-8_D100-8_h48_bt1_L1_Filter";
    // std::string mNeuBTFName = "NeuLF_P5TPos_bunnydisk_U800-8_H100-8_D100-8_h48_bt1_L1_Filter";
    // std::string mNeuBTFName = "NeuLF_P5T3L_fur_hemi2_U400-8_H100-8_D100-8_h32_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuLF_P5T3LPos_fur_hemi3_U600-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuBTF_MultiPlane5LReLURegFliterXYZ_leather11_U400-8_H100-8_D100-8_h32_bt1_L1_pretrain";
    // std::string mNeuBTFName = "NeuBTF_MultiPlane5LReLURegFliterXYZDirXYZWIO_leather11_U400-8_H100-8_D100-8_h32_bt1_L1_pretrain";



    // std::string mNeuBTFName = "NeuLF_P5T3LPos_wire_U400-8_H100-8_D100-8_h32_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuLF_P5T3LPos_fur_hemi3_U600-8_H100-8_D100-8_h32_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuLF_P5T3LPos_fur_hemi4_U1000-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuLF_P5T3LPos_fur_ss_al1_U1500-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuBTF_P5T3LPos_fur_hemi_btf_800_U800-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg_HDR";
    // std::string mNeuBTFName = "NeuLF_3DP2Pos_fur_hemi4_U1000-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg";

    // std::string mNeuBTFName = "NeuLF_3DP2Pos_wire_U1000-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg";
    // std::string mNeuBTFName = "NeuBTF_3DP2Pos_fur_hemi_btf_800_U800-12_H100-8_D100-8_h64_bt1_L1_Filter_Reg_HDR";

    // std::string mNeuBTFName = "NeuLF_3DP2PosDoulbeFaceBlend_wirefull_U600-8_H100-8_D100-8_h64_bt1_L1_Filter_finetune_grad_zero";
    // std::string mNeuBTFName = "NeuLF_3DP2PosDoulbeFaceBlend_bunny_U600-8_H100-8_D100-8_h64_bt1_L1_Filter_grad";
    // std::string mNeuBTFName = "NeuLF_3DP2PosDoulbeFaceBlend_fur_ss_al1_U1500-8_H100-8_D100-8_h64_bt1_L1_Filter_grad";
    // std::string mNeuBTFName = "NeuLF_3DP2PosDoulbeFaceBlendFullOct_wireoct_U800-8_H100-8_D100-8_h64_bt1_L1_Filter_grad";
    // std::string mNeuBTFName = "NeuLF_3DP2PosDoulbeFaceBlendFull_fur_hemi5_U1000-8_H100-8_D100-8_h64_bt1_L1_Filter_grad";
    std::string mNeuBTFName = "NeuLF_DualTriP2_fur_hemi5_U1000-8_H100-8_D100-8_h64_bt1_L1_Filter_grad";

};
