#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include <memory>

using namespace Falcor;

/** Raster primary visibility, GPU-resident cooperative FP16 inference, deferred lighting. */
class NeuralMaterialPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralMaterialPass, "NeuralMaterialPass", "Raster neural materials with CUDA FP16 cooperative inference and deferred IBL/LTC.");
    static ref<NeuralMaterialPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<NeuralMaterialPass>(pDevice, props);
    }
    NeuralMaterialPass(ref<Device> pDevice, const Properties& props);
    ~NeuralMaterialPass();
    Properties getProperties() const override;
    void setProperties(const Properties& props) override;
    RenderPassReflection reflect(const CompileData& compileData) override;
    void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    void renderUI(Gui::Widgets& widget) override;
    void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;

private:
    struct Runtime;
    struct Lighting;
    void ensureModel(RenderContext* pRenderContext);
    void ensureBuffers(RenderContext* pRenderContext, uint2 dimensions);
    void createPrograms();
    void bindInputs(const ShaderVar& root, const RenderData& renderData) const;
    void mapAndInfer(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareAppearance(RenderContext* pRenderContext);
    void prepareAppearanceReference(RenderContext* pRenderContext, const RenderData& renderData);
    void shade(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareDeltaReference(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareLighting(RenderContext* pRenderContext);
    void updateLightControls();
    void configureLights(const Properties& props);
    void renderLightControls(Gui::Widgets& widget);
    void dumpDebug(RenderContext* pRenderContext, const RenderData& renderData);
    void invalidateGeometry() { mGeometryDirty = true; }

    std::unique_ptr<Runtime> mpRuntime;
    std::unique_ptr<Lighting> mpLighting;
    ref<Scene> mpScene;
    ref<ComputePass> mpMappingPass;
    ref<ComputePass> mpShadingPass;
    ref<Buffer> mpFrames;
    uint2 mFrameDim = {0, 0};
    std::string mModelPath;
    std::string mRuntimePath;
    std::string mLoadedModelPath;
    std::string mError;
    bool mReloadModel = true;
    bool mGeometryDirty = true;
    bool mRenderingFailed = false;
    bool mCacheInference = true;
    bool mUsePackedLobes = true;
    uint64_t mInferenceFrames = 0;
    uint64_t mShadingFrames = 0;
    int32_t mMaterialId = -1;
    float mUvScale = 1.f;
    float mSourceCrop = 0.f;
    float mThickness = .025f;
    float mMaxParallax = 0.f;
    bool mSmoothNormals = true;
    bool mFlipV = true;
    bool mReferenceTop = true;
    bool mStackCorrection = false;
    bool mAutomaticLod = true;
    float mFeatureLod = 0.f;
    float mLodBias = 0.f;
    bool mPeriodic = true;
    bool mNearestFeatures = false;
    uint32_t mSynthesisMode = 0;
    int32_t mSynthesisSeed = 17;
    float mSynthesisScale = 2.f, mSynthesisPower = 7.f, mSynthesisRotation = 0.f, mSynthesisBorder = .5f;
    bool mSynthesisDirty = true;
    bool mTwoSided = false;
    bool mEnvironmentEnabled = true;
    bool mDeltaEnabled = true;
    bool mDeltaReference = false;
    bool mEnvironmentReference = false;
    bool mReferenceDirty = true;
    std::string mReferenceControlPrefix = "neuralReference";
    bool mReferenceOutputsInitialized = false;
    uint32_t mLastBatchSamples = 0;
    uint32_t mAppearanceSamplesPerFrame = 1;
    uint32_t mAppearanceMaxSamples = 4096;
    uint32_t mAppearanceSamples = 0;
    bool mDeltaDirty = true;
    bool mAreaEnabled = true;
    bool mBackgroundEnabled = true;
    float mBackgroundExposure = 0.f;
    bool mHorizonOcclusion = false;
    uint32_t mDebugView = 0;
    uint32_t mAreaMode = 0;
    uint32_t mReferenceSamples = 64;
    float mSubstrate = 0.f;
    float mMaterialOpacity = 1.f;
    std::string mDumpPath;
    uint32_t mCaptureFrame = 0;
    uint64_t mFrameIndex = 0;
    bool mDumpRequested = false;
};
