/***************************************************************************
 # Copyright (c) 2015-21, NVIDIA CORPORATION. All rights reserved.
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


#include "PortCompat.h"
#include "Legacy/PathTracer/PathTracer.h"



namespace Falcor {
    class NNLaplacian;

    class SimpleNNModel;


    struct NeuralRequestData {

        NeuralRequestData() {
            bInit = false;
            mpRadianceMerge = nullptr;
            mpAtomicCounterCPURead = nullptr;
            mpPredictionResults = nullptr;
            mpAtomicCounterCPURead = nullptr;

        }

        NeuralRequestData(std::string _name, uint32_t numbuff, uint32_t numelements, int32_t numuniqueelements = -1, bool bForcedMapping=false) {
            name = std::move(_name);
            numBuffers = numbuff;
            numElements = numelements;
            bMapped = bForcedMapping;
            if (numuniqueelements == -1) {
                numUniqueElements = numElements;
            }
            else {
                numUniqueElements = numuniqueelements;
                bMapped = true;
            }
            bInit = true;
            mpRadianceMerge = nullptr;
            mpAtomicCounterCPURead = nullptr;
            mpRadianceMerge = nullptr;
            mpAtomicCounterCPURead = nullptr;
            mpPredictionResults = nullptr;
            mpAtomicCounterCPURead = nullptr;
        }

        bool bInit;
        bool bMapped;

        Falcor::ref<Falcor::Texture> mpRadianceMerge;
        Falcor::ref<Falcor::Buffer> mpPredictionResults;
        Falcor::ref<Falcor::Buffer> mpAtomicCounterCPURead;
        std::string name;
        uint32_t numBuffers;
        uint32_t numElements;
        uint32_t numUniqueElements;
        uint32_t numElementsActive = 0;
        uint32_t numUniqueElementsActive = 0;
        Falcor::uint2 res;

        std::vector<bool> isChannelMapped;

        bool isMapped() const {
            if (!bInit)
                return false;
            return bMapped;
        }

        bool allocateBuffers(Falcor::PortBuffers& buffers);
        bool bindBuffers(Falcor::RenderContext* pRenderContext, Falcor::ref<Falcor::RtProgramVars> vars, const Falcor::PortRenderData& renderData, Falcor::uint2 resolution);

        void requestNumRequests(Falcor::RenderContext* pRenderContext, const Falcor::PortRenderData& renderData);
        void updateNumActiveElements();
    };

class DummyNeuralNetwork : public Falcor::LegacyPathTracer
{
public:
    using SharedPtr = ref<DummyNeuralNetwork>;
    FALCOR_PLUGIN_CLASS(DummyNeuralNetwork, "NeuralRadianceCache", "Original NRC and NIRC with two-level Monte Carlo correction.");

    /** Create a new render pass object.
        \param[in] pRenderContext The render context.
        \param[in] dict Properties of serialized parameters.
        \return A new object, or an exception is thrown if creation failed.
    */
    static SharedPtr create(ref<Device> pDevice, const Properties& props) { return make_ref<DummyNeuralNetwork>(pDevice, props); }

    std::string getDesc();
     virtual Falcor::Properties getProperties() const override;
     void parseDictionary(const Falcor::Properties& dict);
    virtual Falcor::RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(Falcor::RenderContext* pContext, const CompileData& compileData) override;
    virtual void execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& renderData) override;
    virtual void renderUI(Falcor::Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const Falcor::ref<Falcor::Scene>& pScene) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;
    bool onKeyEvent(const Falcor::KeyboardEvent& keyEvent) override;
    void inspectPixel(uint32_t x, uint32_t y);
    uint32_t getInspectorSamples() const { return mInspectorSamples; }

     void resetNN();

     bool isStDevEstimating() const {
        return bStDevEstimation;
    }

     void setModel(bool NIRC) {
        bIncidentNRC = NIRC;
    }
     void enablestats();

     void nrc_experiment_setup();
     void nirc_experiment_setup(int samples);

     void enableTraining(bool bEnable);

     void enableInference(bool bEnable);


     float getAveragePath() const;

     float getLoss() const;

     int getTrainingStep() const;

    DummyNeuralNetwork(ref<Device> pDevice, const Properties& dict);
    
     void reloadModel();
     float getTrainingError() const;
     int32_t getTrainingEpoch() const;
    float getDiffuseTime() const;
    void setDiffuseTime(float t);

    float m_training_error = 0.f;
    int32_t m_training_epoch = 0;
    float m_diffuse_time;

    int32_t mMaxTrainingSteps = -1;

    PortBuffers mInternalBuffers;
    uint2 mScreenDim = { 1920, 1080 };
    uint32_t mTrainingSampleCapacity = 0;
    std::vector<uint32_t> lodSizes;
    uint32_t m_LOD;
    static const char* sDesc;

    uint32_t mNumNRCSamplesPerDepth[3];

    std::shared_ptr<Falcor::NNLaplacian> model;
    std::shared_ptr<Falcor::NNLaplacian> modelEstimator;

     Falcor::ref<Falcor::Texture> getOutput() {
        return output;
    }

private:

    Falcor::ref<Falcor::Texture> output;
    void recreateVars() override { mTracer.pVars = nullptr; }
    void prepareVars();
    void setTracerData(const Falcor::PortRenderData& renderData);
    
    Falcor::ref<Falcor::Buffer> predColorQLearningBuffer;
    Falcor::ref<Falcor::Buffer> predColorResError;
    Falcor::ref<Falcor::Buffer> predColorQLearningBufferTmp;
    Falcor::ref<Falcor::Buffer> predColorVarianceLossTmp;
    Falcor::ref<Falcor::Buffer> predColorVarianceLoss;
    Falcor::ref<Falcor::Buffer> predColorVarianceLossPart;
    Falcor::ref<Falcor::Buffer> estimationsBuffer;
    Falcor::ref<Falcor::Buffer> ourestimationsBuffer;
    Falcor::ref<Falcor::Buffer> YTrainingBuffer;
    Falcor::ref<Falcor::Buffer> YTrainingEstimatorNNBuffer;
    Falcor::ref<Falcor::ComputePass>          mpEnvMapRequests;
    Falcor::ref<Falcor::ComputePass>          mpGenerateTrainingDataPass;
    Falcor::ref<Falcor::ComputePass>          mpConvertNNOutputPass;
    Falcor::ref<Falcor::ComputePass>          mpConvertNNOutputPassFused;
    Falcor::ref<Falcor::ComputePass>          mpGatherOutEstimations;
    Falcor::ref<Falcor::ComputePass>          mpFinalOutputPass;
    Falcor::ref<Falcor::ComputePass>          mpYTrainAssemblyPass;
    bool skipFirstPred = true;
    uint64_t ourestimationSignal;
    Falcor::ref<Falcor::Buffer> mpCounterTrainBuffer;
    Falcor::ref<Falcor::Buffer> mpCounterEstimatorNNTrainBuffer;
    Falcor::ref<Falcor::Buffer> mpCounterPredBuffer;
    Falcor::ref<Falcor::Buffer> mpCounterPredPositionBuffer;
    Falcor::ref<Falcor::Buffer> mpCounterNetworkEstimator;
    Falcor::ref<Falcor::Buffer> mpCounterLThpSelfTrainingBuffer;

    Falcor::ref<Falcor::Buffer> mpCounterQLearningPredPosBuffer;
    Falcor::ref<Falcor::Buffer> mpCounterQLearningPredBuffer;
    
    Falcor::ref<Falcor::Scene> mpScene;

    Falcor::ref<Falcor::Fence> mpFenceForOurEstimations;
    Falcor::ref<Falcor::Fence> mpFenceForPredictionData;
    Falcor::ref<Falcor::Fence> mpFenceForTrainingData;
    Falcor::ref<Falcor::Fence> mpFenceOut;
    Falcor::ref<Falcor::Fence> mpFenceOutDebug;


    bool bErrorBased = false;
    float errorThreashold;
    bool bInitCudaFence = false;
    uint32_t MAX_SAMPLES = 40;
    // Ray tracing program.
    struct
    {
        Falcor::ref<Falcor::Program> pProgram;
        Falcor::ref<Falcor::RtBindingTable> pBindingTable;
        Falcor::ref<Falcor::RtProgramVars> pVars;
        Falcor::ref<Falcor::ParameterBlock> pParameterBlock;      ///< ParameterBlock for all data.
    } mTracer;

    bool bDispatchTrainingRays;
    bool bGradientDenoising;
    bool bFilteringNetwork = false;
    bool bNRCEnabled;
    bool bOnlyIndirect;
    bool bIncidentNRC;
    bool bQLearning;
    bool bNRCDebugMode;
    bool bStopByMetric = false;
    bool bStopByOurMetric = false;
    bool bErrorTest = false;
    bool bNRCBiased;
    bool bForceNonSeparateInput = false;
    bool bDirectLightCache;
    bool mSync = true;
    bool bEvalDirectOnlyOnce = true;
    bool bTonemapWeight = false;
    bool bDirectLightCacheNEE;
    bool bNIRCBasedOnEnvMap = true;
    bool bResErrorForLoss;
    bool bDebugHemispherical;
    bool bVarianceLoss = false;
    bool bStDevEstimation = false;
    bool bFusedOutput = false;

    bool bResidualVarianceLoss = false;
    bool bNNEstimatorDebugRender = false;
    bool bFreezeDebugMouse = true;
    uint32_t mInspectorSamples = 0;
    uint32_t mInspectorSeed = 0;
    float mInspectorExposure = 0.f;
    uint32_t mInspectorComponent = 0; // Cached continuation, or full incoming with traced direct term.
    float2 mInspectorRotation = {0.f, 0.f};
    ref<ComputePass> mpInspectorAccumulate;
    ref<ComputePass> mpInspectorDisplay;
    ref<Texture> mpInspectorPreview;

    float roughnessThreshold = 0.45f;
    float roulleteProb = 0.0f;
    bool bJIT = true;

    uint32_t mNumGTSamples = 1;
    uint32_t mNumVarianceLossSamples = 2;
    
    uint32_t mNumQLearningSamples;
    uint32_t gradientHashGridRes;
    uint32_t maxBounceWithCV;

    cudaEvent_t eventNetworksSync;
    cudaEvent_t eventNetworksSyncResVar;
    bool bEventIsCreatedResVar = false;
    bool bEventIsCreated = false;
    
    bool bSelfLearningForFilterNetwork = false;
    bool bUseNNAsEstimator = false;
    uint32_t gradientHashGridSize;

    uint32_t mNumSamplesForExpectedValues = 32;
    uint2  mSelectedPixel = { 0, 0 };      ///< Currently selected pixel.

    bool bResetModel = false;
    uint32_t mNircArchitecture = 0; // Original, DirectionalHash, Hybrid, OriginalHashLocal
    uint32_t mNircInterpolation = 1; // Trilinear, Tetrahedral (default for custom models)
    NeuralRequestData mainRenderData;
    NeuralRequestData debugRenderData;
};
}
