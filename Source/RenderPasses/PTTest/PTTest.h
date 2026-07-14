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
#include "Rendering/Lights/EnvMapSampler.h"
#include "Utils/Neural/NNMat.h"
using namespace Falcor;

class PTTest : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(PTTest, "PTTest", "Insert pass description here.");

    static ref<PTTest> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<PTTest>(pDevice, props);
    }

    PTTest(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override {}
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }
    void handleOutput();

    /// NeuLobes light-probe neural model (PETwoLVTensorFMLPInterp).
    /// Holds hyperparameters read from manifest.json plus the raw weight tensors uploaded to the GPU.
    struct NeuLobesModel
    {
        // Hyperparameters (read from manifest.json, fall back to the reference config on missing keys).
        uint32_t peBands = 2;   ///< Positional-encoding bands.
        uint32_t res = 6;       ///< Feature-plane resolution (planeRes).
        uint32_t rank = 2;      ///< Low-rank factor count.
        uint32_t planeDim = 4;  ///< Feature channels per basis (== MLP feature tail length).
        uint32_t hiddenDim = 4; ///< MLP hidden width.
        uint32_t numBasis = 3;  ///< Number of blended weight/feature sets (== length of bary).
        uint32_t inputDim = 16; ///< MLP input width (6*peBands + planeDim).
        uint32_t outputDim = 3; ///< MLP output width (RGB).

        // Per-layer 4x4 block counts, one entry per MLP layer (3 layers).
        uint32_t inBlk[3] = {};
        uint32_t outBlk[3] = {};

        // GPU buffers (CPU tensors uploaded to device-local memory).
        ref<Buffer> pVecX;  ///< StructuredBuffer<float>    [numBasis*rank*res].
        ref<Buffer> pMatYZ; ///< StructuredBuffer<float>    [numBasis*rank*res*planeDim].
        ref<Buffer> pW[3];  ///< StructuredBuffer<float4x4> per layer, mul-ready 4x4 blocks.
        ref<Buffer> pB[3];  ///< StructuredBuffer<float4>   per layer, bias blocks.

        bool loaded = false;
    };

    /// Reads manifest.json + the float32 weight binaries from `dir` and uploads them to GPU buffers.
    void loadNeuLobesModel(const std::string& dir);
    /// Binds the loaded NeuLobes buffers and config constants to the ray tracing program.
    void bindNeuLobesData(const ShaderVar& var);

private:
    void parseProperties(const Properties& props);
    void prepareVars();
    /// Reads back the selected instance/triangle vertices and computes their world-space positions.
    void computeSelectedTriangleWorldPositions();


    ref<Texture> mpBarycentric;
    ref<Texture> mpWi;
    ref<Texture> mpRadiance;

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

    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;

    float mLightTheta = 0;
    float mLightPhi = 0;

    float mViewTheta = 0.3f;
    float mViewPhi = 0;
    float mViewSize = 1;
    float mViewHeight = 1;
    float mViewHeightBot = 0;
    bool mBTFViewMode = true;
    bool mPluckerMode = true;
    bool mShowSelectedTri = false;
    bool mChangeLight = false;

    float4 mXYUV = float4(0.0);


    bool mIsOutputing = false;
    uint mOutputStep = 0;

    uint mOutputIndx = 0;
    uint mOutputOffsetIndx = 0;
    uint mOutputSPP = 32;
    std::string mOutputPath = "C:/Data/Probe/test/{:06}_{:.6f}_{:.6f}.exr";
    std::string mOutputBTFPath = "D:/Data/BTF/test/{:06}_{:.6f}_{:.6f}_{:.6f}_{:.6f}.exr";

    float mSampleTheta = 0;
    float mSampleLightTheta = 0;

    std::shared_ptr<NNMat> mpNNMatT;
    std::shared_ptr<NNMat> mpNNMatBTF;
    std::string mNeuBTFName = "NeuBTF_P5Dir_carpet02_U400-8_H100-8_D100-8_h32_bt1_L1_Filter";

    // NeuLobes neural light-probe model.
    NeuLobesModel mNeuLobes;
    std::string mNeuLobesDir = "C:/projects/neulobes/outputs/neulobes_bin";
    bool mUseNeuLobes = false;
    /// Barycentric probe-blend weights used for the demo inference query (should sum to 1).
    float3 mNeuBary = float3(1.0f, 0.0f, 0.0f);

    float phiCount = 60.0f;
    float thetaCount = 30.0f;

    float phiStep   = 1.0f / phiCount;
    float thetaStep = 0.98f / (thetaCount - 1.0f);

    float3 mProbePos = float3(0.0f, 0.1f, 0.0f);

    // Scene stats (populated in setScene).
    uint32_t mMeshCount = 0;
    uint32_t mInstanceCount = 0;

    // Triangle picking: select a triangle by instance + triangle ID and store its world-space vertices.
    uint32_t mSelectedInstanceID = 0;
    uint32_t mSelectedTriangleID = 0;
    bool mSelectedTriangleValid = false;
    float3 mSelectedTrianglePosW[3] = {float3(0.f), float3(0.f), float3(0.f)};
    uint32_t mSelectedTriVertexIDs[3] = {0, 0, 0};
    // Barycentric sample coordinates for uniform triangle sampling of the primary ray origin.
    float mTriSampleU = 0.5f;
    float mTriSampleV = 0.5f;
    const float2 vertexUV[3] = {float2(0.0f, 0.0f), float2(0.999f, 0.0f), float2(0.999f, 0.999f)};
    // const float startingUV = 0.05f;
    const float startingUV = 0.05f;
    float numberOfInterals = 10;
    float intervalUV = (1.0f - 2 * startingUV) / numberOfInterals;
};
