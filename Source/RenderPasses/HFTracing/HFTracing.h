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
#include "Rendering/Lights/EnvMapSampler.h"
#include "cuda/MLPInference.h"

using namespace Falcor;
enum class RenderType {
    HF,
    MAP
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
class HFTracing : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(HFTracing, "HFTracing", "Minimal path tracer.");

    static ref<HFTracing> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<HFTracing>(pDevice, props);
    }

    HFTracing(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void generateGeometryMap(RenderContext* pRenderContext, const RenderData& renderData);
    void renderHF(RenderContext* pRenderContext, const RenderData& renderData);
    void visualizeMaps(RenderContext* pRenderContext, const RenderData& renderData);
    void createMaxMip(RenderContext* pRenderContext, const RenderData& renderData);
    void nnInferPass(RenderContext* pRenderContext, const RenderData& renderData);
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override {   return mpPixelDebug->onMouseEvent(mouseEvent); }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

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

    ref<ComputePass> mpGenerateGeometryMapPass;
    ref<ComputePass> mpVisualizeMapsPass;
    ref<ComputePass> mpCreateMaxMipPass;
    ref<ComputePass> mpInferPass ;
    // Texture inputs
    std::string mTexturePath =getProjectDirectory().string();
    // std::string mHFFileName = "ganges_river_pebbles_disp_4k.png";
    std::string mHFFileName = "castle_brick_02_red_cut_disp_4k";
    // std::string mHFFileName = "dirty_carpet_cut_disp_4k";
    // std::string mColorFileName = "ganges_river_pebbles_diff_4k.jpg";
    std::string mColorFileName = "castle_brick_02_red_cut_diff_4k";
    // std::string mColorFileName = "dirty_carpet_cut_diff_4k.jpg";
    ref<Texture> mpHF;
    ref<Texture> mpHFMaxMip;
    ref<Texture> mpColor;
    ref<Texture> mpNormalMap;
    ref<Texture> mpTangentMap;
    ref<Texture> mpPosMap;

    ref<Texture> mpWiWox;
    ref<Texture> mpUVWoyz;
    ref<Texture> mpDfDxy;

    std::unique_ptr<PixelDebug> mpPixelDebug;

    Falcor::float4 mControlParas = Falcor::float4(1, 0.6, 0, 0.045);
    Falcor::float4 mCurvatureParas = Falcor::float4(0.056, 1, 0.65, 0.6);
    Falcor::float4 mLightZPR = Falcor::float4(0.056, 1, 0.15, 0.1);
    uint mTriID = 0;
    uint mMaxSteps = 1000;
    uint mMaxTriCount = 1000;
    RenderType mRenderType = RenderType::HF;

    bool mContactRefinement = true;
    bool mMipGenerated = false;
    bool mApplySyn = false;
    bool mNNInfer = true;
    bool mHFBound = true;
    bool mLocalFrame = true;

    /// GPU fence for synchronizing readback.
    ref<Fence> mpFence;
    /// Buffer for data for the selected pixel.
    ref<Buffer> mpPixelDataBuffer;
    /// Staging buffer for readback of pixel data.
    ref<Buffer> mpPixelStagingBuffer;
    /// Pixel data for the selected pixel (if valid).
    // PixelData mPixelData;
    bool mPixelDataValid = false;
    bool mPixelDataAvailable = false;


    std::unique_ptr<TextureSynthesis> mpTextureSynthesis;
    std::unique_ptr<MLP> mpMLP;
    std::unique_ptr<NBTF> mpNBTF;


    std::unique_ptr<EnvMapSampler>  mpEnvMapSampler;
};
