#include "NeuralMaterialPass.h"
#include "NeuralCompositePass.h"
#include "RenderGraph/AccumulatePassControl.h"
#include "NeuralLightMotion.h"
#include "Core/Platform/OS.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Utils/CudaUtils.h"
#include "Utils/Math/Float16.h"
#include "Utils/Image/Bitmap.h"
#include "inference_bridge.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace
{
constexpr uint32_t kMaxLights = 8;
constexpr float kPi = 3.14159265358979323846f;
const std::string kShaderDirectory = "RenderPasses/NeuralMaterialPass/";
const ChannelList kInputs = {
    {"posW", "gPosW", "Raster position"}, {"normW", "gNormW", "Raster normal", true},
    {"tangentW", "gTangentW", "Raster tangent and handedness", true}, {"faceNormalW", "gFaceNormalW", "Face normal", true},
    {"texC", "gTexC", "Unwrapped UV"}, {"texGrads", "gTexGrads", "Raster ddx/ddy UV"},
    {"mtlData", "gMtlData", "Material identity", true}, {"vbuffer", "gVBuffer", "Current deformed triangle"},
};

struct GpuPolygonLight
{
    float4 center, fullU, fullV, emission;
    uint4 params;
    float2 uvVertices[16];
};
static_assert(sizeof(GpuPolygonLight) == 208);

// Resource ownership is explicit: mapped pointer -> external memory -> Falcor buffer.
// The caller drains old work only at resize/model replacement/destruction.
struct SharedBuffer
{
    ref<Buffer> buffer;
    cudaExternalMemory_t memory = nullptr;
    void* pointer = nullptr;
    ~SharedBuffer()
    {
        if (pointer) cudaFree(pointer);
        if (memory) cudaDestroyExternalMemory(memory);
    }
    static std::unique_ptr<SharedBuffer> create(ref<Device> device, uint32_t stride, uint32_t count)
    {
        auto result = std::make_unique<SharedBuffer>();
        result->buffer = device->createStructuredBuffer(stride, count,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Shared);
        result->memory = cuda_utils::importExternalMemory(result->buffer.get());
        result->pointer = cuda_utils::externalMemoryGetMappedBuffer(result->memory, 0, result->buffer->getSize());
        return result;
    }
};

template<typename T> T loadFunction(SharedLibraryHandle module, const char* name)
{
    auto result = reinterpret_cast<T>(getProcAddress(module, name));
    FALCOR_CHECK(result, "Inference runtime is missing {}", name);
    return result;
}

float3 rotateXYZ(float3 p, float3 degrees)
{
    const float3 a = degrees * (kPi / 180.f);
    p = float3(p.x, std::cos(a.x) * p.y - std::sin(a.x) * p.z, std::sin(a.x) * p.y + std::cos(a.x) * p.z);
    p = float3(std::cos(a.y) * p.x + std::sin(a.y) * p.z, p.y, -std::sin(a.y) * p.x + std::cos(a.y) * p.z);
    return float3(std::cos(a.z) * p.x - std::sin(a.z) * p.y, std::sin(a.z) * p.x + std::cos(a.z) * p.y, p.z);
}

std::vector<float> readFloatTable(const std::filesystem::path& path, size_t expected)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    FALCOR_CHECK(input.good(), "Cannot open neural lighting table {}", path.string());
    FALCOR_CHECK(size_t(input.tellg()) == expected * sizeof(float), "Invalid neural table size {}", path.string());
    std::vector<float> values(expected);
    input.seekg(0); input.read(reinterpret_cast<char*>(values.data()), std::streamsize(expected * sizeof(float)));
    return values;
}
}

struct NeuralMaterialPass::Runtime
{
    SharedLibraryHandle library = nullptr;
    void* context = nullptr;
    cudaStream_t stream = nullptr;
    MpmatInferenceInfo info = {};
    decltype(&mpmat_infer_create) create = nullptr;
    decltype(&mpmat_infer_get_info) getInfo = nullptr;
    decltype(&mpmat_infer_reserve) reserve = nullptr;
    decltype(&mpmat_infer_lobes) infer = nullptr;
    decltype(&mpmat_infer_lobes_fp16) inferFp16 = nullptr;
    decltype(&mpmat_infer_prepare_appearance) prepareAppearance = nullptr;
    decltype(&mpmat_infer_appearance) appearance = nullptr;
    decltype(&mpmat_infer_set_synthesis) setSynthesis = nullptr;
    decltype(&mpmat_infer_destroy) destroy = nullptr;
    decltype(&mpmat_infer_error) error = nullptr;
    std::unique_ptr<SharedBuffer> rays, lobes;
    std::unique_ptr<SharedBuffer> localDirections, appearanceRgba;
    bool appearancePrepared = false;
    bool packedLobes = false;
    uint32_t lobeStride() const { return packedLobes ? ((info.output_stride_floats + 7u) & ~7u) : info.output_stride_floats; }
    uint32_t lobeWords() const { return packedLobes ? lobeStride() / 2u : lobeStride(); }

    ~Runtime()
    {
        if (stream) cudaStreamSynchronize(stream);
        localDirections.reset(); appearanceRgba.reset(); rays.reset(); lobes.reset();
        if (context && destroy) destroy(context);
        if (library) releaseSharedLibrary(library);
    }
    void check(int result) const
    {
        FALCOR_CHECK(result != 0, "MPMat FP16 cooperative inference: {}", error ? error() : "unknown error");
    }
};

struct NeuralMaterialPass::Lighting
{
    struct Light
    {
        bool enabled = true;
        uint32_t shape = 1;
        uint32_t textureKind = 0;
        float3 center = float3(0.f, 0.f, 1.8f);
        float3 rotation = float3(90.f, 0.f, 0.f);
        float2 size = float2(1.4f);
        float3 color = float3(1.f);
        float intensity = 6.f;
        int32_t sceneLight = -1;
        std::string texturePath;
        std::string textureError;
        uint32_t bakedTextureKind = 0;
        uint32_t bakedShape = 1;
        neural_controls::LightMotion motion;
        bool bakeDirty = true;
        ref<Texture> source, prefilter;
    };
    std::vector<Light> lights{Light{}};
    struct DeltaControl
    {
        ref<Falcor::Light> light;
        float3 position = float3(0.f, 0.f, 2.f);
        float3 direction = float3(0.f, 0.f, -1.f);
        neural_controls::LightMotion motion;
    };
    std::vector<DeltaControl> deltaControls;
    Properties pendingAnalyticLights;
    bool hasPendingAnalyticLights = false;
    neural_controls::LightMotion environmentMotion;
    float3 environmentRotation = float3(0.f);
    bool overrideEnvironmentRotation = false;
    ref<EnvMap> controlledEnvironment;
    bool controlsDirty = false;
    ref<Buffer> lightBuffer, ltcTable, horizonTable;
    ref<Sampler> sampler;
    ref<ComputePass> envBake, lightBake;
    ref<ComputePass> deltaGenerate, deltaAccumulate;
    ref<ComputePass> referenceGenerate, referenceSum;
    ref<Buffer> referenceWeights;
    ref<Buffer> referenceDirect, dummyDirect;
    ref<Texture> envPrefilter, dummyCube, dummySource, dummyPrefilter;
    ref<ShaderResourceView> envPrefilterSrv, dummyCubeSrv;
    ref<EnvMap> environment;
    std::unique_ptr<EnvMapSampler> envSampler;
    bool envDirty = true;
    uint32_t envSize = 256, envLevels = 7, envSamples = 2048;
    float envMinimumRoughness = .12f;
    std::array<GpuPolygonLight, kMaxLights> records = {};
    uint32_t activePolygonLights = 0;
};

extern "C" FALCOR_API_EXPORT void registerPlugin(PluginRegistry& registry)
{
    registry.registerClass<RenderPass, NeuralMaterialPass>();
    registry.registerClass<RenderPass, NeuralCompositePass>();
}

NeuralMaterialPass::NeuralMaterialPass(ref<Device> device, const Properties& props)
    : RenderPass(device), mpLighting(std::make_unique<Lighting>())
{
    FALCOR_CHECK(device->getType() == Device::Type::D3D12, "NeuralMaterialPass requires D3D12 GPU-fence/CUDA interoperability");
    mModelPath = props.get<std::string>("modelPath", "");
    mRuntimePath = props.get<std::string>("runtimePath", "");
    mMaterialId = props.get<int32_t>("materialId", -1);
    mUvScale = props.get<float>("uvScale", 1.f);
    mSourceCrop = props.get<float>("sourceCrop", 0.f);
    mThickness = props.get<float>("thickness", .025f);
    mMaxParallax = props.get<float>("maxParallax", 0.f);
    mSmoothNormals = props.get<bool>("smoothNormals", true);
    mFlipV = props.get<bool>("flipV", true);
    mTwoSided = props.get<bool>("twoSided", false);
    mReferenceTop = props.get<bool>("referenceTop", true);
    mStackCorrection = props.get<bool>("stackCorrection", false);
    mPeriodic = props.get<bool>("periodic", true);
    mNearestFeatures = props.get<bool>("nearestFeatures", false);
    mSynthesisMode = props.get<uint32_t>("synthesisMode", 0);
    mSynthesisSeed = props.get<int32_t>("synthesisSeed", 17);
    mSynthesisScale = props.get<float>("synthesisScale", 2.f);
    mSynthesisPower = props.get<float>("synthesisPower", 7.f);
    mSynthesisRotation = props.get<float>("synthesisRotation", 0.f);
    mSynthesisBorder = props.get<float>("synthesisBorder", .5f);
    mAutomaticLod = props.get<bool>("automaticLod", true);
    mFeatureLod = props.get<float>("featureLod", 0.f);
    mLodBias = props.get<float>("lodBias", 0.f);
    mCacheInference = props.get<bool>("cacheInference", true);
    mUsePackedLobes = props.get<bool>("packedLobes", true);
    mEnvironmentEnabled = props.get<bool>("environment", true);
    mDeltaEnabled = props.get<bool>("deltaLights", true);
    mDeltaReference = props.get<bool>("deltaAppearanceReference", false);
    mAreaEnabled = props.get<bool>("areaLights", true);
    mBackgroundEnabled = props.get<bool>("background", true);
    mBackgroundExposure = props.get<float>("backgroundExposure", 0.f);
    mEmitterExposure = props.get<float>("emitterExposure", -6.f);
    FALCOR_CHECK(std::isfinite(mEmitterExposure) && mEmitterExposure >= -16.f && mEmitterExposure <= 8.f, "Invalid emitterExposure");
    mHorizonOcclusion = props.get<bool>("horizonOcclusion", false);
    mAreaMode = props.get<uint32_t>("areaMode", 0);
    mEnvironmentReference = props.get<bool>("environmentAppearanceReference", false);
    mReferenceControlPrefix = props.get<std::string>("referenceControlPrefix", "neuralReference");
    mAppearanceSamplesPerFrame = props.get<uint32_t>("appearanceSamplesPerFrame", 1);
    mAppearanceMaxSamples = props.get<uint32_t>("appearanceMaxSamples", 4096);
    mReferenceSamples = props.get<uint32_t>("referenceSamples", 64);
    mDebugView = props.get<uint32_t>("debugView", 0);
    mSubstrate = props.get<float>("substrate", 0.f);
    mDumpPath = props.get<std::string>("dumpPath", "");
    mCaptureFrame = props.get<uint32_t>("captureFrame", 0);
    mpLighting->envSamples = props.get<uint32_t>("environmentBakeSamples", 2048);
    configureLights(props);
    FALCOR_CHECK(std::isfinite(mUvScale) && mUvScale > 0 && std::isfinite(mSourceCrop) && mSourceCrop >= 0 && mSourceCrop < .5f,
        "Invalid UV scale/source crop");
    FALCOR_CHECK(std::isfinite(mThickness) && mThickness > 0, "Thickness must be finite and positive");
    FALCOR_CHECK(mAreaMode <= 2 && mReferenceSamples >= 1 && mReferenceSamples <= 4096, "Invalid area reference settings");
    FALCOR_CHECK(mAppearanceSamplesPerFrame >= 1 && mAppearanceSamplesPerFrame <= 64 && mAppearanceMaxSamples >= 1 &&
        mAppearanceMaxSamples <= 1048576, "Invalid original appearance reference sample settings");
}

NeuralMaterialPass::~NeuralMaterialPass()
{
    // Exceptional teardown is allowed to drain queues; normal frames use GPU fences only.
    try { mpDevice->getRenderContext()->submit(true); } catch (...) {}
    mpRuntime.reset();
}

void NeuralMaterialPass::configureLights(const Properties& props)
{
    auto& lighting = *mpLighting;
    if (props.has("lights"))
    {
        std::vector<Lighting::Light> lights;
        const Properties entries = props.get<Properties>("lights");
        for (const auto& [key, value] : entries)
        {
            FALCOR_CHECK(lights.size() < kMaxLights, "At most {} polygon lights are supported", kMaxLights);
            const Properties p = value;
            Lighting::Light light;
            if (lights.size() < lighting.lights.size()) light = lighting.lights[lights.size()];
            const auto oldPath = light.texturePath;
            const auto oldKind = light.textureKind, oldShape = light.shape;
            light.enabled = p.get<bool>("enabled", light.enabled);
            light.shape = p.get<uint32_t>("shape", light.shape);
            light.textureKind = p.get<uint32_t>("texture", light.textureKind);
            light.center = p.get<float3>("center", light.center);
            light.rotation = p.get<float3>("rotation", light.rotation);
            light.size = p.get<float2>("size", light.size);
            light.color = p.get<float3>("color", light.color);
            light.intensity = p.get<float>("intensity", light.intensity);
            light.sceneLight = p.get<int32_t>("sceneLight", light.sceneLight);
            light.texturePath = p.get<std::string>("texturePath", light.texturePath);
            if (p.has("motion")) light.motion = neural_controls::LightMotion::fromProperties(p.get<Properties>("motion"));
            light.bakeDirty |= oldPath != light.texturePath || oldKind != light.textureKind || oldShape != light.shape;
            FALCOR_CHECK(light.shape <= 1 && light.textureKind <= 3 && all(light.size > float2(0.f)), "Invalid polygon light");
            lights.push_back(light);
        }
        lighting.lights = std::move(lights);
    }
    if (props.has("analyticLights"))
    {
        lighting.pendingAnalyticLights = props.get<Properties>("analyticLights");
        lighting.hasPendingAnalyticLights = true;
    }
    if (props.has("environmentRotation"))
    {
        lighting.environmentRotation = props.get<float3>("environmentRotation");
        lighting.overrideEnvironmentRotation = true;
    }
    if (props.has("environmentMotion"))
        lighting.environmentMotion = neural_controls::LightMotion::fromProperties(props.get<Properties>("environmentMotion"));
}

void NeuralMaterialPass::setProperties(const Properties& props)
{
    configureLights(props);
    auto change = [&](const char* key, auto& dst, bool& dirty)
    {
        if (!props.has(key)) return;
        const auto value = props.get<std::decay_t<decltype(dst)>>(key);
        if (value != dst) { dst = value; dirty = true; }
    };
    change("environment", mEnvironmentEnabled, mReferenceDirty);
    change("environmentAppearanceReference", mEnvironmentReference, mReferenceDirty);
    change("areaLights", mAreaEnabled, mReferenceDirty);
    change("areaMode", mAreaMode, mReferenceDirty);
    change("deltaLights", mDeltaEnabled, mDeltaDirty);
    change("deltaAppearanceReference", mDeltaReference, mDeltaDirty);
    change("substrate", mSubstrate, mReferenceDirty);
    mDeltaDirty |= props.has("substrate");
    mBackgroundEnabled = props.get<bool>("background", mBackgroundEnabled);
    mBackgroundExposure = props.get<float>("backgroundExposure", mBackgroundExposure);
    if (props.has("emitterExposure"))
    {
        const float exposure = props.get<float>("emitterExposure");
        FALCOR_CHECK(std::isfinite(exposure) && exposure >= -16.f && exposure <= 8.f, "Invalid emitterExposure");
        mEmitterExposure = exposure;
    }
    mAppearanceSamplesPerFrame = props.get<uint32_t>("appearanceSamplesPerFrame", mAppearanceSamplesPerFrame);
    mAppearanceMaxSamples = props.get<uint32_t>("appearanceMaxSamples", mAppearanceMaxSamples);
    if (mAppearanceMaxSamples < mAppearanceSamples) mReferenceDirty = true;
    FALCOR_CHECK(mAppearanceSamplesPerFrame >= 1 && mAppearanceSamplesPerFrame <= 64 && mAppearanceMaxSamples >= 1 &&
        mAppearanceMaxSamples <= 1048576 && mAreaMode <= 2, "Invalid reference settings");
    mReferenceDirty |= props.get<bool>("resetAppearance", false);
    mDumpPath = props.get<std::string>("dumpPath", mDumpPath);
    mDumpRequested |= props.get<bool>("dumpRequested", false);
    mRenderingFailed = false;
}

void NeuralMaterialPass::updateLightControls()
{
    auto& lighting = *mpLighting;
    const double time = mpScene->getTime();
    bool changed = lighting.controlsDirty;
    lighting.controlsDirty = false;
    if (lighting.hasPendingAnalyticLights)
    {
        for (const auto& [key, value] : lighting.pendingAnalyticLights)
        {
            const Properties p = value;
            const auto name = p.get<std::string>("name", "Neural delta " + key);
            const auto type = p.get<std::string>("type", "point");
            FALCOR_CHECK(type == "point" || type == "spot" || type == "directional", "Unknown delta light type {}", type);
            auto light = mpScene->getLightByName(name);
            const bool point = type != "directional";
            if (!light)
            {
                light = point ? ref<Falcor::Light>(PointLight::create(name)) : ref<Falcor::Light>(DirectionalLight::create(name));
                mpScene->addLight(light);
            }
            FALCOR_CHECK(light->getType() == (point ? LightType::Point : LightType::Directional), "Delta light {} has a different type", name);
            light->setActive(p.get<bool>("active", true));
            light->setIntensity(p.get<float3>("intensity", light->getIntensity()));
            auto it = std::find_if(lighting.deltaControls.begin(), lighting.deltaControls.end(), [&](const auto& c) { return c.light == light; });
            if (it == lighting.deltaControls.end())
            {
                Lighting::DeltaControl control;
                control.light = light; control.position = light->getData().posW; control.direction = light->getData().dirW;
                lighting.deltaControls.push_back(control); it = std::prev(lighting.deltaControls.end());
            }
            it->position = p.get<float3>("position", it->position);
            it->direction = p.get<float3>("direction", it->direction);
            FALCOR_CHECK(dot(it->direction, it->direction) > 1e-12f, "Delta direction cannot be zero");
            if (p.has("motion")) it->motion = neural_controls::LightMotion::fromProperties(p.get<Properties>("motion"));
            if (p.has("sceneAnimated")) light->setIsAnimated(p.get<bool>("sceneAnimated"));
            if (it->motion.autoRotate || it->motion.autoOrbit) light->setIsAnimated(false);
            if (point)
            {
                auto* source = static_cast<PointLight*>(light.get());
                source->setOpeningAngle(p.get<float>("openingAngle", type == "spot" ? 30.f : source->getOpeningAngle() * 180.f / kPi) * kPi / 180.f);
                source->setPenumbraAngle(p.get<float>("penumbraAngle", source->getPenumbraAngle() * 180.f / kPi) * kPi / 180.f);
                source->setWorldPosition(it->motion.position(it->position, time));
                source->setWorldDirection(rotateXYZ(it->direction, it->motion.spin(time)));
            }
            else static_cast<DirectionalLight*>(light.get())->setWorldDirection(rotateXYZ(it->direction, it->motion.spin(time)));
            changed = true;
        }
        lighting.hasPendingAnalyticLights = false;
    }
    const auto& sceneLights = mpScene->getLights();
    lighting.deltaControls.erase(std::remove_if(lighting.deltaControls.begin(), lighting.deltaControls.end(),
        [&](const auto& c) { return std::find(sceneLights.begin(), sceneLights.end(), c.light) == sceneLights.end(); }), lighting.deltaControls.end());
    for (const auto& source : sceneLights)
    {
        if (source->getType() != LightType::Point && source->getType() != LightType::Directional) continue;
        auto it = std::find_if(lighting.deltaControls.begin(), lighting.deltaControls.end(), [&](const auto& c) { return c.light == source; });
        if (it == lighting.deltaControls.end())
        {
            Lighting::DeltaControl c; c.light = source; c.position = source->getData().posW; c.direction = source->getData().dirW;
            lighting.deltaControls.push_back(c); it = std::prev(lighting.deltaControls.end());
        }
        auto& c = *it;
        if (!c.motion.autoRotate) c.direction = source->getData().dirW;
        if (!c.motion.autoOrbit) c.position = source->getData().posW;
        if (!c.motion.autoRotate && !c.motion.autoOrbit) continue;
        source->setIsAnimated(false);
        const float3 direction = normalize(rotateXYZ(c.direction, c.motion.spin(time)));
        const float3 position = c.motion.position(c.position, time);
        const bool moved = any(direction != source->getData().dirW) || (source->getType() == LightType::Point && any(position != source->getData().posW));
        if (!moved) continue;
        if (source->getType() == LightType::Point)
        {
            auto* point = static_cast<PointLight*>(source.get());
            point->setWorldPosition(position); point->setWorldDirection(direction);
        }
        else static_cast<DirectionalLight*>(source.get())->setWorldDirection(direction);
        changed = true;
    }
    if (const auto env = mpScene->getEnvMap())
    {
        if (lighting.controlledEnvironment != env)
        {
            lighting.controlledEnvironment = env;
            if (!lighting.overrideEnvironmentRotation) lighting.environmentRotation = env->getRotation();
        }
        if (lighting.overrideEnvironmentRotation || lighting.environmentMotion.autoRotate)
        {
            const auto rotation = lighting.environmentMotion.rotation(lighting.environmentRotation, time);
            if (any(rotation != env->getRotation())) { env->setRotation(rotation); changed = true; }
            lighting.overrideEnvironmentRotation = false;
        }
        else lighting.environmentRotation = env->getRotation();
    }
    if (changed) mpScene->refreshLighting();
}

Properties NeuralMaterialPass::getProperties() const
{
    Properties p;
    p["modelPath"] = mModelPath; p["runtimePath"] = mRuntimePath;
    p["materialId"] = mMaterialId; p["uvScale"] = mUvScale; p["sourceCrop"] = mSourceCrop;
    p["thickness"] = mThickness; p["maxParallax"] = mMaxParallax;
    p["smoothNormals"] = mSmoothNormals; p["twoSided"] = mTwoSided; p["referenceTop"] = mReferenceTop;
    p["flipV"] = mFlipV;
    p["stackCorrection"] = mStackCorrection; p["periodic"] = mPeriodic; p["nearestFeatures"] = mNearestFeatures;
    p["synthesisMode"] = mSynthesisMode; p["synthesisSeed"] = mSynthesisSeed;
    p["synthesisScale"] = mSynthesisScale; p["synthesisPower"] = mSynthesisPower;
    p["synthesisRotation"] = mSynthesisRotation; p["synthesisBorder"] = mSynthesisBorder;
    p["automaticLod"] = mAutomaticLod; p["featureLod"] = mFeatureLod; p["lodBias"] = mLodBias;
    p["cacheInference"] = mCacheInference; p["environment"] = mEnvironmentEnabled; p["deltaLights"] = mDeltaEnabled;
    p["packedLobes"] = mUsePackedLobes;
    p["packedLobesActive"] = mpRuntime && mpRuntime->packedLobes;
    p["deltaAppearanceReference"] = mDeltaReference;
    p["environmentAppearanceReference"] = mEnvironmentReference;
    p["appearanceSamplesPerFrame"] = mAppearanceSamplesPerFrame; p["appearanceMaxSamples"] = mAppearanceMaxSamples;
    p["appearanceAccumulatedSamples"] = mAppearanceSamples;
    p["referenceControlPrefix"] = mReferenceControlPrefix;
    p["referenceAccumulator"] = std::string("Falcor AccumulatePass");
    p["referenceBatchSamples"] = mLastBatchSamples;
    p["areaLights"] = mAreaEnabled; p["background"] = mBackgroundEnabled; p["horizonOcclusion"] = mHorizonOcclusion;
    p["backgroundExposure"] = mBackgroundExposure;
    p["emitterExposure"] = mEmitterExposure;
    p["activePolygonLights"] = mpLighting->activePolygonLights;
    p["areaMode"] = mAreaMode; p["referenceSamples"] = mReferenceSamples; p["debugView"] = mDebugView;
    p["substrate"] = mSubstrate; p["environmentBakeSamples"] = mpLighting->envSamples;
    p["dumpPath"] = mDumpPath; p["captureFrame"] = mCaptureFrame;
    p["inferenceFrames"] = mInferenceFrames; p["shadingFrames"] = mShadingFrames;
    p["lastError"] = mError;
    Properties lights;
    for (size_t i = 0; i < mpLighting->lights.size(); ++i)
    {
        const auto& light = mpLighting->lights[i];
        Properties item;
        item["enabled"] = light.enabled; item["shape"] = light.shape; item["texture"] = light.textureKind;
        item["center"] = light.center; item["rotation"] = light.rotation; item["size"] = light.size;
        item["color"] = light.color; item["intensity"] = light.intensity; item["sceneLight"] = light.sceneLight;
        item["texturePath"] = light.texturePath;
        item["textureError"] = light.textureError;
        item["motion"] = light.motion.toProperties();
        const double time = mpScene ? mpScene->getTime() : 0.0;
        item["effectiveCenter"] = light.motion.position(light.center, time);
        item["effectiveRotation"] = light.motion.rotation(light.rotation, time);
        lights[std::to_string(i)] = item;
    }
    p["lights"] = lights;
    p["environmentRotation"] = mpLighting->environmentRotation;
    p["environmentMotion"] = mpLighting->environmentMotion.toProperties();
    if (mpScene && mpScene->getEnvMap()) p["effectiveEnvironmentRotation"] = mpScene->getEnvMap()->getRotation();
    Properties analytic;
    for (size_t i = 0; i < mpLighting->deltaControls.size(); ++i)
    {
        const auto& control = mpLighting->deltaControls[i];
        const auto& light = control.light;
        Properties d;
        d["name"] = light->getName(); d["type"] = light->getType() == LightType::Point ? "point" : "directional";
        d["active"] = light->isActive(); d["intensity"] = light->getIntensity();
        d["sceneAnimated"] = light->isAnimated();
        d["position"] = control.position; d["direction"] = control.direction;
        d["effectivePosition"] = light->getData().posW; d["effectiveDirection"] = light->getData().dirW;
        d["motion"] = control.motion.toProperties();
        if (light->getType() == LightType::Point)
        {
            const auto* point = static_cast<const PointLight*>(light.get());
            d["openingAngle"] = point->getOpeningAngle() * (180.f / kPi);
            d["penumbraAngle"] = point->getPenumbraAngle() * (180.f / kPi);
        }
        analytic[std::to_string(i)] = d;
    }
    p["analyticLights"] = mpLighting->hasPendingAnalyticLights ? mpLighting->pendingAnalyticLights : analytic;
    return p;
}

RenderPassReflection NeuralMaterialPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflection;
    addRenderPassInputs(reflection, kInputs);
    for (const char* channel : {"foreground", "display", "environmentSamples", "areaSamples"})
    {
        auto& field = reflection.addOutput(channel, "Neural HDR graph data (reference RGBA = frame sample sum and count)")
            .format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        if (std::string(channel).find("Samples") != std::string::npos)
            field.flags(RenderPassReflection::Field::Flags::Persistent);
    }
    return reflection;
}

void NeuralMaterialPass::compile(RenderContext* context, const CompileData& compileData)
{
    mReferenceOutputsInitialized = false;
    mGeometryDirty = true;
    mRenderingFailed = false;
}

void NeuralMaterialPass::setScene(RenderContext* context, const ref<Scene>& scene)
{
    mpScene = scene;
    mpLighting->deltaControls.clear();
    mpLighting->controlledEnvironment = nullptr;
    mpMappingPass = nullptr; mpShadingPass = nullptr; mpLighting->envBake = nullptr;
    mpLighting->deltaGenerate = nullptr; mpLighting->deltaAccumulate = nullptr;
    mpLighting->referenceGenerate = nullptr; mpLighting->referenceSum = nullptr;
    mReferenceDirty = true;
    mpLighting->envDirty = true;
    mGeometryDirty = true; mDeltaDirty = true; mRenderingFailed = false; mFrameIndex = 0;
}

void NeuralMaterialPass::ensureModel(RenderContext* renderContext)
{
    if (!mReloadModel) return;
    mReloadModel = false;
    if (mModelPath.empty()) { mError = "Choose an exported .mpmat model/head asset, then press Load model."; return; }
    try
    {
        FALCOR_CHECK(mpDevice->initCudaDevice(), "CUDA initialization failed");
        auto runtime = std::make_unique<Runtime>();
        runtime->stream = reinterpret_cast<cudaStream_t>(mpDevice->getCudaDevice()->getStream());
        const std::filesystem::path dll = mRuntimePath.empty() ? getRuntimeDirectory() / "mpmat_inference.dll" : std::filesystem::path(mRuntimePath);
        runtime->library = loadSharedLibrary(dll);
        FALCOR_CHECK(runtime->library, "Could not load {}", dll.string());
        runtime->create = loadFunction<decltype(runtime->create)>(runtime->library, "mpmat_infer_create");
        runtime->getInfo = loadFunction<decltype(runtime->getInfo)>(runtime->library, "mpmat_infer_get_info");
        runtime->reserve = loadFunction<decltype(runtime->reserve)>(runtime->library, "mpmat_infer_reserve");
        runtime->infer = loadFunction<decltype(runtime->infer)>(runtime->library, "mpmat_infer_lobes");
        runtime->inferFp16 = reinterpret_cast<decltype(runtime->inferFp16)>(getProcAddress(runtime->library, "mpmat_infer_lobes_fp16"));
        runtime->prepareAppearance = reinterpret_cast<decltype(runtime->prepareAppearance)>(getProcAddress(runtime->library, "mpmat_infer_prepare_appearance"));
        runtime->appearance = reinterpret_cast<decltype(runtime->appearance)>(getProcAddress(runtime->library, "mpmat_infer_appearance"));
        runtime->setSynthesis = loadFunction<decltype(runtime->setSynthesis)>(runtime->library, "mpmat_infer_set_synthesis");
        runtime->destroy = loadFunction<decltype(runtime->destroy)>(runtime->library, "mpmat_infer_destroy");
        runtime->error = loadFunction<decltype(runtime->error)>(runtime->library, "mpmat_infer_error");
        runtime->check(runtime->create(mModelPath.c_str(), static_cast<int>(mpDevice->getCudaDevice()->getDevice()), &runtime->context));
        runtime->info.struct_size = sizeof(MpmatInferenceInfo);
        runtime->check(runtime->getInfo(runtime->context, &runtime->info));
        const auto& info = runtime->info;
        FALCOR_CHECK(info.abi_version == MPMAT_INFER_ABI_VERSION && info.ray_bytes == sizeof(MpmatInferenceRay)
            && info.output_stride_floats == 7 * info.lobes + 1 && (info.capabilities & 1), "Incompatible inference SDK ABI/capabilities");
        runtime->packedLobes = mUsePackedLobes && runtime->inferFp16 && (info.capabilities & 16u);
        runtime->check(runtime->setSynthesis(runtime->context, int(mSynthesisMode), mSynthesisSeed,
            mSynthesisScale, mSynthesisPower, mSynthesisRotation, mSynthesisBorder));
        renderContext->submit(true);
        mpRuntime = std::move(runtime);
        mpShadingPass = nullptr;
        mLoadedModelPath = mModelPath; mFrameDim = uint2(0); mGeometryDirty = true;
        mpLighting->envDirty = true;
        mSynthesisDirty = false;
        mRenderingFailed = false; mError.clear();
        logInfo("NeuralMaterialPass: FP16 coop, {} lobes, {} learned mip level(s), model kind {}", info.lobes, info.mip_levels, info.model_kind);
    }
    catch (const std::exception& e) { mError = e.what(); logError("{}", mError); }
}

void NeuralMaterialPass::ensureBuffers(RenderContext* context, uint2 dimensions)
{
    if (math::all(dimensions == mFrameDim) && mpRuntime->rays) return;
    const uint64_t count64 = uint64_t(dimensions.x) * dimensions.y;
    FALCOR_CHECK(count64 > 0 && count64 <= std::numeric_limits<uint32_t>::max() / mpRuntime->lobeStride(),
        "Unsupported neural framebuffer dimensions");
    const uint32_t count = uint32_t(count64);
    context->submit(true);
    FALCOR_CUDA_CHECK(cudaStreamSynchronize(mpRuntime->stream));
    auto rays = SharedBuffer::create(mpDevice, sizeof(MpmatInferenceRay), count);
    auto lobes = SharedBuffer::create(mpDevice, sizeof(uint32_t), count * mpRuntime->lobeWords());
    auto frames = mpDevice->createStructuredBuffer(32, count);
    mpRuntime->check(mpRuntime->reserve(mpRuntime->context, count));
    mpRuntime->rays = std::move(rays); mpRuntime->lobes = std::move(lobes); mpFrames = frames;
    mpRuntime->localDirections.reset(); mpRuntime->appearanceRgba.reset();
    mpLighting->referenceDirect = nullptr;
    mpLighting->referenceWeights = nullptr;
    mReferenceOutputsInitialized = false;
    mReferenceDirty = true;
    mFrameDim = dimensions;
    mGeometryDirty = true; mDeltaDirty = true;
}

void NeuralMaterialPass::createPrograms()
{
    auto make = [&](const char* file, bool lighting = false)
    {
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderDirectory + file).csEntry("main");
        desc.addTypeConformances(mpScene->getTypeConformances());
        auto defines = mpScene->getSceneDefines();
        if (lighting && mpRuntime->packedLobes) defines.add("NEURAL_PACKED_LOBES", "1");
        return ComputePass::create(mpDevice, desc, defines, true);
    };
    if (!mpMappingPass) mpMappingPass = make("NeuralMapping.cs.slang");
    if (!mpShadingPass) mpShadingPass = make("NeuralLighting.cs.slang", true);
    if (!mpLighting->envBake) mpLighting->envBake = make("NeuralEnvBake.cs.slang");
    if (!mpLighting->lightBake) mpLighting->lightBake = ComputePass::create(mpDevice, kShaderDirectory + "NeuralLightBake.cs.slang", "main");
}

void NeuralMaterialPass::bindInputs(const ShaderVar& root, const RenderData& data) const
{
    for (const auto& channel : kInputs)
    {
        auto var = root.findMember(channel.texname);
        if (var.isValid()) var = data.getTexture(channel.name);
    }
    auto scene = root.findMember("gScene");
    if (scene.isValid()) mpScene->bindShaderData(scene);
}

void NeuralMaterialPass::mapAndInfer(RenderContext* context, const RenderData& data)
{
    FALCOR_PROFILE(context, "Neural mapping and cooperative inference");
    const auto& info = mpRuntime->info;
    const bool cacheAppearance = ((mDeltaReference && mDeltaEnabled) || (mEnvironmentReference && mEnvironmentEnabled) ||
        (mAreaMode == 2 && mAreaEnabled)) && (info.capabilities & 8u);
    auto root = mpMappingPass->getRootVar();
    bindInputs(root, data);
    root["gRays"] = mpRuntime->rays->buffer;
    root["gFrames"] = mpFrames;
    auto cb = root["gMappingCB"];
    cb["frameDim"] = mFrameDim;
    cb["boundsMin"] = float3(info.bounds[0], info.bounds[1], info.bounds[2]);
    cb["boundsMax"] = float3(info.bounds[3], info.bounds[4], info.bounds[5]);
    cb["baseResolution"] = info.base_resolution; cb["mipLevels"] = info.mip_levels;
    cb["automaticLod"] = uint32_t(mAutomaticLod); cb["fixedLod"] = mFeatureLod; cb["lodBias"] = mLodBias;
    cb["uvScale"] = mUvScale; cb["crop"] = mSourceCrop; cb["thickness"] = mThickness;
    cb["maxParallax"] = mMaxParallax; cb["smoothNormals"] = uint32_t(mSmoothNormals); cb["twoSided"] = uint32_t(mTwoSided);
    cb["flipV"] = uint32_t(mFlipV);
    cb["referenceTop"] = uint32_t(mReferenceTop); cb["stackCorrection"] = uint32_t(mStackCorrection); cb["materialId"] = mMaterialId;
    {
        FALCOR_PROFILE(context, "Raster material mapping");
        mpMappingPass->execute(context, mFrameDim.x, mFrameDim.y);
    }
    {
    FALCOR_PROFILE(context, "CUDA lobe inference and GPU handoff");
    context->uavBarrier(mpRuntime->rays->buffer.get());
    context->resourceBarrier(mpRuntime->rays->buffer.get(), Resource::State::Common);
    context->resourceBarrier(mpRuntime->lobes->buffer.get(), Resource::State::Common);
    context->waitForFalcor(mpRuntime->stream);
    MpmatInferenceOptions options = {sizeof(MpmatInferenceOptions),
        (mPeriodic ? MPMAT_INFER_PERIODIC : 0u) | (mNearestFeatures ? MPMAT_INFER_NEAREST_FEATURE : 0u) |
            (cacheAppearance ? MPMAT_INFER_CACHE_APPEARANCE : 0u),
        mSourceCrop, mAutomaticLod ? -1.f : mFeatureLod};
    if (mpRuntime->packedLobes)
        mpRuntime->check(mpRuntime->inferFp16(mpRuntime->context, static_cast<MpmatInferenceRay*>(mpRuntime->rays->pointer),
            mFrameDim.x * mFrameDim.y, &options, static_cast<uint16_t*>(mpRuntime->lobes->pointer), mpRuntime->stream));
    else
        mpRuntime->check(mpRuntime->infer(mpRuntime->context, static_cast<MpmatInferenceRay*>(mpRuntime->rays->pointer),
            mFrameDim.x * mFrameDim.y, &options, static_cast<float*>(mpRuntime->lobes->pointer), mpRuntime->stream));
    context->waitForCuda(mpRuntime->stream);
    context->resourceBarrier(mpRuntime->lobes->buffer.get(), Resource::State::ShaderResource);
    }
    mGeometryDirty = false;
    mpRuntime->appearancePrepared = cacheAppearance; mDeltaDirty = true;
    ++mInferenceFrames;
}

void NeuralMaterialPass::execute(RenderContext* context, const RenderData& data)
{
    auto& dict = data.getDictionary();
    dict[mReferenceControlPrefix + ".dumpPath"] = std::string();
    dict[mReferenceControlPrefix + ".substrate"] = mSubstrate > 0.f;
    auto clearFailedFrame = [&]()
    {
        for (const char* channel : {"foreground", "display", "environmentSamples", "areaSamples"})
            context->clearUAV(data.getTexture(channel)->getUAV().get(), float4(0.f));
        dict[mReferenceControlPrefix + ".environment"] = kAccumulateReset | kAccumulateHold;
        dict[mReferenceControlPrefix + ".area"] = kAccumulateReset | kAccumulateHold;
        dict[mReferenceControlPrefix + ".environmentEnabled"] = false;
        dict[mReferenceControlPrefix + ".areaEnabled"] = false;
    };
    ensureModel(context);
    if (!mpScene || !mpRuntime || mRenderingFailed)
    {
        clearFailedFrame();
        return;
    }
    try
    {
        if (mSynthesisDirty)
        {
            context->submit(true);
            FALCOR_CUDA_CHECK(cudaStreamSynchronize(mpRuntime->stream));
            mpRuntime->check(mpRuntime->setSynthesis(mpRuntime->context, int(mSynthesisMode), mSynthesisSeed,
                mSynthesisScale, mSynthesisPower, mSynthesisRotation, mSynthesisBorder));
            mSynthesisDirty = false; mGeometryDirty = true;
        }
        updateLightControls();
        const auto updates = mpScene->getUpdates();
        if (is_set(updates, IScene::UpdateFlags::RecompileNeeded) || is_set(updates, IScene::UpdateFlags::GeometryChanged))
        {
            mpMappingPass = nullptr; mpShadingPass = nullptr; mpLighting->envBake = nullptr; mGeometryDirty = true;
            mpLighting->deltaGenerate = nullptr; mpLighting->deltaAccumulate = nullptr;
            mpLighting->referenceGenerate = nullptr; mpLighting->referenceSum = nullptr;
        }
        const auto geometryUpdates = IScene::UpdateFlags::GeometryMoved | IScene::UpdateFlags::MeshesChanged |
            IScene::UpdateFlags::CameraMoved | IScene::UpdateFlags::CameraPropertiesChanged | IScene::UpdateFlags::CameraSwitched |
            IScene::UpdateFlags::MaterialsChanged | IScene::UpdateFlags::DisplacementChanged;
        if ((uint32_t(updates) & uint32_t(geometryUpdates)) != 0) mGeometryDirty = true;
        const auto lightUpdates = IScene::UpdateFlags::LightsMoved | IScene::UpdateFlags::LightIntensityChanged |
            IScene::UpdateFlags::LightPropertiesChanged | IScene::UpdateFlags::LightCountChanged;
        if ((uint32_t(updates) & uint32_t(lightUpdates)) != 0) mDeltaDirty = true;
        if (is_set(updates, IScene::UpdateFlags::EnvMapChanged) || is_set(updates, IScene::UpdateFlags::EnvMapPropertiesChanged))
            mReferenceDirty |= mEnvironmentReference;
        const auto cameraUpdates = mpScene->getCamera()->getChanges();
        if (is_set(cameraUpdates, Camera::Changes::Jitter) || is_set(cameraUpdates, Camera::Changes::Frustum)) mGeometryDirty = true;
        const auto output = data.getTexture("foreground");
        ensureBuffers(context, uint2(output->getWidth(), output->getHeight()));
        createPrograms();
        if (mGeometryDirty) mReferenceDirty = true;
        if (mGeometryDirty || !mCacheInference) mapAndInfer(context, data);
        prepareLighting(context);
        shade(context, data);
        if (mDumpRequested || (!mDumpPath.empty() && mFrameIndex == mCaptureFrame)) dumpDebug(context, data);
        ++mFrameIndex;
    }
    catch (const std::exception& e)
    {
        mError = e.what(); mRenderingFailed = true;
        logError("NeuralMaterialPass: {}", mError);
        clearFailedFrame();
    }
}

void NeuralMaterialPass::dumpDebug(RenderContext* context, const RenderData& data)
{
    mDumpRequested = false;
    if (mDumpPath.empty()) return;
    const std::filesystem::path directory(mDumpPath);
    std::filesystem::create_directories(directory);
    context->submit(true);
    auto dump = [&](const char* name, const ref<Buffer>& buffer, size_t bytes)
    {
        std::vector<char> host(bytes);
        buffer->getBlob(host.data(), 0, bytes);
        std::ofstream file(directory / name, std::ios::binary);
        FALCOR_CHECK(file.good(), "Cannot create debug buffer {}", (directory / name).string());
        file.write(host.data(), std::streamsize(bytes));
        FALCOR_CHECK(file.good(), "Failed writing debug buffer {}", name);
    };
    const size_t pixels = size_t(mFrameDim.x) * mFrameDim.y;
    dump("rays.bin", mpRuntime->rays->buffer, pixels * 48);
    dump("frames.bin", mpFrames, pixels * 32);
    if (mpRuntime->packedLobes)
    {
        // Explicit debug readback retains the original FP32 capture schema.
        // Normal display consumes packed values directly on the GPU.
        const uint32_t logicalStride = mpRuntime->info.output_stride_floats, storageStride = mpRuntime->lobeStride();
        std::vector<uint16_t> packed(pixels * storageStride);
        mpRuntime->lobes->buffer->getBlob(packed.data(), 0, packed.size() * sizeof(uint16_t));
        std::vector<float> unpacked(pixels * logicalStride);
        for (size_t pixel = 0; pixel < pixels; ++pixel)
            for (uint32_t j = 0; j < logicalStride; ++j)
                unpacked[pixel * logicalStride + j] = float(math::float16_t::fromBits(packed[pixel * storageStride + j]));
        std::ofstream file(directory / "lobes.bin", std::ios::binary);
        file.write(reinterpret_cast<const char*>(unpacked.data()), std::streamsize(unpacked.size() * sizeof(float)));
        FALCOR_CHECK(file.good(), "Failed writing packed lobe debug capture");
    }
    else dump("lobes.bin", mpRuntime->lobes->buffer, pixels * mpRuntime->info.output_stride_floats * sizeof(float));
    if (mpRuntime->localDirections)
    {
        dump("localDirections.bin", mpRuntime->localDirections->buffer, pixels * 16);
        dump("appearanceRGBA.bin", mpRuntime->appearanceRgba->buffer, pixels * 16);
        if (mpLighting->referenceDirect) dump("referenceDirect.bin", mpLighting->referenceDirect, pixels * 16);
    }
    if (mpLighting->referenceWeights) dump("referenceWeights.bin", mpLighting->referenceWeights, pixels * 16);
    for (const char* channel : {"posW", "texC", "texGrads", "foreground", "display", "environmentSamples", "areaSamples"})
    {
        const auto bytes = context->readTextureSubresource(data.getTexture(channel).get(), 0);
        std::ofstream output(directory / (std::string(channel) + ".bin"), std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    }
    // Final color and normalized reference captures belong to the downstream
    // Composite node, after the real AccumulatePass nodes have executed.
    data.getDictionary()[mReferenceControlPrefix + ".dumpPath"] = directory.string();
    Properties metadata = getProperties();
    const auto& info = mpRuntime->info;
    metadata["width"] = mFrameDim.x; metadata["height"] = mFrameDim.y;
    metadata["rayBytes"] = info.ray_bytes; metadata["frameBytes"] = 32u;
    metadata["outputStrideFloats"] = info.output_stride_floats;
    metadata["lobeStorageBytesPerPixel"] = mpRuntime->lobeWords() * uint32_t(sizeof(uint32_t));
    metadata["mipLevels"] = info.mip_levels; metadata["baseResolution"] = info.base_resolution;
    metadata["boundsMin"] = float3(info.bounds[0], info.bounds[1], info.bounds[2]);
    metadata["boundsMax"] = float3(info.bounds[3], info.bounds[4], info.bounds[5]);
    metadata["lobes"] = info.lobes; metadata["modelKind"] = info.model_kind;
    metadata["headInputKind"] = info.head_input_kind; metadata["coverageKind"] = info.coverage_kind;
    metadata["inferenceFrames"] = mInferenceFrames; metadata["shadingFrames"] = mShadingFrames;
    metadata["actualModelPath"] = mLoadedModelPath;
    metadata["cameraPosition"] = mpScene->getCamera()->getPosition();
    Properties sceneLights;
    const auto& activeLights = mpScene->getActiveAnalyticLights();
    for (uint32_t index = 0; index < activeLights.size(); ++index)
    {
        const auto& light = activeLights[index]->getData();
        Properties item;
        item["type"] = light.type; item["position"] = light.posW; item["direction"] = light.dirW;
        item["intensity"] = light.intensity; item["openingAngle"] = light.openingAngle; item["penumbraAngle"] = light.penumbraAngle;
        sceneLights[std::to_string(index)] = item;
    }
    metadata["activeLights"] = sceneLights;
    metadata["precision"] = std::string("fp16-cooperative-vector");
    std::ofstream file(directory / "metadata.json"); file << metadata.dump(2);
    logInfo("NeuralMaterialPass debug buffers saved to {}", directory.string());
}

void NeuralMaterialPass::prepareLighting(RenderContext* context)
{
    auto& lighting = *mpLighting;
    if (!lighting.sampler)
    {
        Sampler::Desc desc;
        desc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear);
        desc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        lighting.sampler = mpDevice->createSampler(desc);
        const float4 white(1.f), black(0.f);
        std::array<float4, 6> blackFaces = {};
        lighting.dummySource = mpDevice->createTexture2D(1, 1, ResourceFormat::RGBA32Float, 1, 1, &white);
        lighting.dummyPrefilter = mpDevice->createTexture3D(1, 1, 1, ResourceFormat::RGBA32Float, 1, &white);
        lighting.dummyCube = mpDevice->createTextureCube(1, 1, ResourceFormat::RGBA32Float, 1, 1, blackFaces.data());
        // Falcor 8's default cube SRV passes a cube count where GFX expects
        // individual layers. Specify all six faces explicitly.
        lighting.dummyCubeSrv = ShaderResourceView::create(mpDevice.get(), lighting.dummyCube.get(), 0, 1, 0, 6);
        lighting.dummyDirect = mpDevice->createStructuredBuffer(16, 1, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, &black);
        lighting.lightBuffer = mpDevice->createStructuredBuffer(sizeof(GpuPolygonLight), kMaxLights);
        const auto tableDir = getRuntimeDirectory() / "data" / "NeuralMaterial";
        auto ltc = readFloatTable(tableDir / "ltc_log_scale.bin", 512);
        auto horizon = readFloatTable(tableDir / "horizon_cone.bin", 257);
        lighting.ltcTable = mpDevice->createStructuredBuffer(sizeof(float), uint32_t(ltc.size()), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, ltc.data());
        lighting.horizonTable = mpDevice->createStructuredBuffer(sizeof(float), uint32_t(horizon.size()), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, horizon.data());
    }
    const auto env = mpScene->getEnvMap();
    if (lighting.environment != env || is_set(mpScene->getUpdates(), IScene::UpdateFlags::EnvMapChanged)) lighting.envDirty = true;
    if (lighting.envDirty && mEnvironmentEnabled)
    {
        FALCOR_PROFILE(context, "Neural environment prefilter bake (load only)");
        if (env)
        {
            auto sampler = std::make_unique<EnvMapSampler>(mpDevice, env);
            auto texture = mpDevice->createTextureCube(lighting.envSize, lighting.envSize, ResourceFormat::RGBA32Float, 1,
                lighting.envLevels, nullptr, ResourceBindFlags::ShaderResource);
            // Falcor cube views count cubes, whereas RWTexture2DArray requires
            // six individual layers. Bake into an array and copy once on GPU.
            auto bakeTarget = mpDevice->createTexture2D(lighting.envSize, lighting.envSize, ResourceFormat::RGBA32Float, 6,
                lighting.envLevels, nullptr, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
            const float minR = std::max(1e-4f, std::min(mpRuntime->info.minimum_roughness, 1.f));
            auto root = lighting.envBake->getRootVar();
            mpScene->bindShaderData(root["gScene"]);
            sampler->bindShaderData(root["gEnvSampler"]);
            for (uint32_t level = 0; level < lighting.envLevels; ++level)
            {
                const uint32_t size = lighting.envSize >> level;
                root["gEnvOutput"].setUav(bakeTarget->getUAV(level, 0, 6));
                auto cb = root["gEnvBakeCB"];
                cb["gBakeSize"] = size; cb["gBakeSamples"] = lighting.envSamples;
                cb["gBakeRoughness"] = minR * std::pow(1.f / minR, float(level) / float(lighting.envLevels - 1));
                cb["gUseEnvImportance"] = 1u;
                for (uint32_t face = 0; face < 6; ++face)
                {
                    cb["gBakeFace"] = face;
                    lighting.envBake->execute(context, size, size, 1);
                }
            }
            context->uavBarrier(bakeTarget.get());
            for (uint32_t face = 0; face < 6; ++face)
                for (uint32_t level = 0; level < lighting.envLevels; ++level)
                    context->copySubresource(texture.get(), texture->getSubresourceIndex(face, level),
                        bakeTarget.get(), bakeTarget->getSubresourceIndex(face, level));
            root["gEnvOutput"].setUav(nullptr);
            lighting.envSampler = std::move(sampler); lighting.envPrefilter = texture;
            lighting.envPrefilterSrv = ShaderResourceView::create(mpDevice.get(), texture.get(), 0, lighting.envLevels, 0, 6);
            lighting.envMinimumRoughness = minR;
        }
        else { lighting.envSampler.reset(); lighting.envPrefilter = nullptr; lighting.envPrefilterSrv = nullptr; }
        lighting.environment = env; lighting.envDirty = false;
    }

    std::array<GpuPolygonLight, kMaxLights> records = {};
    for (size_t index = 0; index < lighting.lights.size(); ++index)
    {
        auto& light = lighting.lights[index];
        auto& gpu = records[index];
        // Keep pending texture edits for re-enabling, without baking or
        // animating lights that cannot contribute to this frame.
        if (!mAreaEnabled || !light.enabled) continue;
        FALCOR_CHECK(light.shape <= 1 && light.textureKind <= 3 && light.size.x > 0 && light.size.y > 0,
            "Invalid polygon light shape/texture/size");
        const double time = mpScene->getTime();
        const float3 rotation = light.motion.rotation(light.rotation, time);
        gpu.center = float4(light.motion.position(light.center, time), 1.f);
        gpu.fullU = float4(rotateXYZ(float3(light.size.x, 0.f, 0.f), rotation), 0.f);
        gpu.fullV = float4(rotateXYZ(float3(0.f, light.size.y, 0.f), rotation), 0.f);
        gpu.emission = float4(light.color * light.intensity, 0.f);
        gpu.params = uint4(light.shape == 0 ? 4 : 10, light.textureKind, light.enabled ? 1 : 0, 1);
        auto setOutline = [&]()
        {
            for (auto& uv : gpu.uvVertices) uv = float2(0.f);
            gpu.params.x = light.shape == 0 ? 4 : 10;
            if (light.shape == 0)
            {
                gpu.uvVertices[0] = float2(0.f, 0.f); gpu.uvVertices[1] = float2(1.f, 0.f);
                gpu.uvVertices[2] = float2(1.f, 1.f); gpu.uvVertices[3] = float2(0.f, 1.f);
            }
            else
            {
                for (uint32_t vertex = 0; vertex < 10; ++vertex)
                {
                    const float angle = kPi * .5f + 2.f * kPi * float(vertex) / 10.f;
                    const float radius = vertex % 2 == 0 ? .5f : .225f;
                    gpu.uvVertices[vertex] = float2(.5f) + radius * float2(std::cos(angle), std::sin(angle));
                }
            }
        };
        setOutline();
        if (light.sceneLight >= 0)
        {
            FALCOR_CHECK(uint32_t(light.sceneLight) < mpScene->getLightCount(), "Bound polygon sceneLight is out of range");
            const auto source = mpScene->getLight(uint32_t(light.sceneLight));
            FALCOR_CHECK(source->getType() == LightType::Rect, "Polygon sceneLight binding requires a Falcor RectLight");
            const auto& data = source->getData();
            gpu.center = math::mul(data.transMat, float4(0.f, 0.f, 0.f, 1.f));
            gpu.fullU = math::mul(data.transMat, float4(2.f, 0.f, 0.f, 0.f));
            gpu.fullV = math::mul(data.transMat, float4(0.f, 2.f, 0.f, 0.f));
            gpu.emission = float4(light.color * light.intensity * source->getIntensity(), 0.f);
            if (!source->isActive()) gpu.params.z = 0;
        }
        if (gpu.params.z == 0) { gpu = {}; continue; }
        if (light.bakeDirty && light.textureKind == 0)
        {
            light.source = nullptr; light.prefilter = nullptr;
            light.bakedTextureKind = 0; light.bakedShape = light.shape; light.textureError.clear(); light.bakeDirty = false;
        }
        if (light.bakeDirty && light.textureKind != 0)
        {
            try
            {
                FALCOR_PROFILE(context, "Polygon emission prefilter bake (load only)");
                auto source = lighting.dummySource;
                if (light.textureKind == 3)
                {
                    FALCOR_CHECK(!light.texturePath.empty(), "Choose a texture path for the image polygon light");
                    source = Texture::createFromFile(mpDevice, light.texturePath, false, true);
                    FALCOR_CHECK(source, "Cannot load polygon light texture {}", light.texturePath);
                }
                auto volume = mpDevice->createTexture3D(257, 257, 27, ResourceFormat::RGBA32Float, 1, nullptr,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
                auto root = lighting.lightBake->getRootVar();
                root["gBakeSource"] = source; root["gBakeSampler"] = lighting.sampler; root["gBakeOutput"] = volume;
                auto cb = root["gLightBakeCB"];
                cb["gBakeDimensions"] = uint3(257, 257, 27);
                cb["gBakeCells"] = light.textureKind == 3 ? 32u : light.shape == 1 ? 48u : 12u;
                cb["gBakeTextureKind"] = light.textureKind; cb["gBakeVertexCount"] = gpu.params.x;
                cb["gBakeLogMin"] = -10.f; cb["gBakeLogMax"] = 3.f;
                for (uint32_t vertex = 0; vertex < 16; ++vertex) cb["gBakeVertices"][vertex] = gpu.uvVertices[vertex];
                for (uint32_t slice = 0; slice < 27; ++slice)
                {
                    cb["gBakeSlice"] = slice;
                    lighting.lightBake->execute(context, 257, 257, 1);
                }
                context->uavBarrier(volume.get());
                light.source = source; light.prefilter = volume;
                light.bakedTextureKind = light.textureKind; light.bakedShape = light.shape; light.textureError.clear();
                mReferenceDirty |= mAreaMode == 2;
            }
            catch (const std::exception& e)
            {
                // Keep the last complete source/prefilter pair. A failed file selection
                // is a local UI error, not a reason to black out the render graph.
                light.textureError = e.what();
                if (light.bakedTextureKind != 0 && light.shape != light.bakedShape)
                {
                    light.shape = light.bakedShape;
                    setOutline();
                }
                logWarning("Polygon emission image: {}", light.textureError);
            }
            light.bakeDirty = false;
            // The recorded GPU commands retain in-flight resources. The reusable
            // bake pass must not retain the last light after it is removed.
            auto root = lighting.lightBake->getRootVar();
            root["gBakeSource"].setTexture(nullptr);
            root["gBakeOutput"].setTexture(nullptr);
        }
        gpu.params.y = light.bakedTextureKind;
    }
    lighting.activePolygonLights = uint32_t(std::count_if(records.begin(), records.end(),
        [](const auto& light) { return light.params.z != 0; }));
    // Refresh bindings even when reference accumulation is stopped or complete;
    // otherwise that pass keeps deleted lights' source textures alive indefinitely.
    if (lighting.referenceGenerate)
    {
        auto root = lighting.referenceGenerate->getRootVar();
        for (uint32_t i = 0; i < kMaxLights; ++i)
            root["gAreaSources"][i] = i < lighting.lights.size() && records[i].params.z && lighting.lights[i].source ?
                lighting.lights[i].source : lighting.dummySource;
    }
    if (std::memcmp(records.data(), lighting.records.data(), sizeof(records)) != 0 || mShadingFrames == 0)
    {
        lighting.lightBuffer->setBlob(records.data(), 0, sizeof(records));
        lighting.records = records;
        mReferenceDirty |= mAreaMode == 2;
    }
}

void NeuralMaterialPass::prepareAppearance(RenderContext* context)
{
    FALCOR_CHECK(mpRuntime->prepareAppearance && mpRuntime->appearance && (mpRuntime->info.capabilities & 2u),
        "This inference SDK/model does not provide the original appearance reference");
    const uint32_t pixels = mFrameDim.x * mFrameDim.y;
    if (!mpRuntime->localDirections)
    {
        mpRuntime->localDirections = SharedBuffer::create(mpDevice, 16, pixels);
        mpRuntime->appearanceRgba = SharedBuffer::create(mpDevice, 16, pixels);
    }
    if (!mpRuntime->appearancePrepared)
    {
        FALCOR_PROFILE(context, "Appearance ray encoding and GPU handoff");
        context->resourceBarrier(mpRuntime->rays->buffer.get(), Resource::State::Common);
        context->waitForFalcor(mpRuntime->stream);
        const MpmatInferenceOptions options = {sizeof(MpmatInferenceOptions),
            (mPeriodic ? MPMAT_INFER_PERIODIC : 0u) | (mNearestFeatures ? MPMAT_INFER_NEAREST_FEATURE : 0u),
            mSourceCrop, mAutomaticLod ? -1.f : mFeatureLod};
        mpRuntime->check(mpRuntime->prepareAppearance(mpRuntime->context,
            static_cast<const MpmatInferenceRay*>(mpRuntime->rays->pointer), pixels, &options, mpRuntime->stream));
        context->waitForCuda(mpRuntime->stream);
        mpRuntime->appearancePrepared = true;
    }
}

void NeuralMaterialPass::prepareDeltaReference(RenderContext* context, const RenderData& data)
{
    if (!mDeltaReference || !mDeltaEnabled || !mDeltaDirty) return;
    prepareAppearance(context);
    auto& lighting = *mpLighting;
    const uint32_t pixels = mFrameDim.x * mFrameDim.y;
    if (!lighting.referenceDirect) lighting.referenceDirect = mpDevice->createStructuredBuffer(16, pixels);
    if (!lighting.deltaGenerate)
    {
        auto make = [&](const char* entry)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kShaderDirectory + "NeuralDeltaReference.cs.slang").csEntry(entry);
            desc.addTypeConformances(mpScene->getTypeConformances());
            return ComputePass::create(mpDevice, desc, mpScene->getSceneDefines(), true);
        };
        lighting.deltaGenerate = make("generate"); lighting.deltaAccumulate = make("accumulate");
    }
    FALCOR_PROFILE(context, "Original appearance delta reference");
    context->clearUAV(lighting.referenceDirect->getUAV().get(), float4(0.f));
    const auto& lights = mpScene->getActiveAnalyticLights();
    for (uint32_t index = 0; index < lights.size(); ++index)
    {
        if (lights[index]->getType() != LightType::Point && lights[index]->getType() != LightType::Directional) continue;
        auto root = lighting.deltaGenerate->getRootVar();
        bindInputs(root, data); root["gFrames"] = mpFrames; root["gLocalDirections"] = mpRuntime->localDirections->buffer;
        root["gDeltaCB"]["gFrameDim"] = mFrameDim; root["gDeltaCB"]["gLightIndex"] = index;
        lighting.deltaGenerate->execute(context, mFrameDim.x, mFrameDim.y);
        context->uavBarrier(mpRuntime->localDirections->buffer.get());
        context->resourceBarrier(mpRuntime->localDirections->buffer.get(), Resource::State::Common);
        context->resourceBarrier(mpRuntime->appearanceRgba->buffer.get(), Resource::State::Common);
        context->waitForFalcor(mpRuntime->stream);
        mpRuntime->check(mpRuntime->appearance(mpRuntime->context, pixels,
            static_cast<const float*>(mpRuntime->localDirections->pointer), static_cast<float*>(mpRuntime->appearanceRgba->pointer), mpRuntime->stream));
        context->waitForCuda(mpRuntime->stream);
        context->resourceBarrier(mpRuntime->appearanceRgba->buffer.get(), Resource::State::ShaderResource);
        root = lighting.deltaAccumulate->getRootVar();
        bindInputs(root, data); root["gFrames"] = mpFrames; root["gAppearanceRGBA"] = mpRuntime->appearanceRgba->buffer;
        root["gReferenceDirect"] = lighting.referenceDirect;
        root["gDeltaCB"]["gFrameDim"] = mFrameDim; root["gDeltaCB"]["gLightIndex"] = index;
        auto substrate = root["gDeltaCB"].findMember("gSubstrate");
        if (substrate.isValid()) substrate = mSubstrate;
        lighting.deltaAccumulate->execute(context, mFrameDim.x, mFrameDim.y);
        context->uavBarrier(lighting.referenceDirect.get());
    }
    mDeltaDirty = false;
}

void NeuralMaterialPass::prepareAppearanceReference(RenderContext* context, const RenderData& data)
{
    const bool environment = mEnvironmentReference && mEnvironmentEnabled && bool(mpScene->getEnvMap());
    const bool area = mAreaMode == 2 && mAreaEnabled && mpLighting->activePolygonLights > 0;
    auto& lighting = *mpLighting;
    auto& dict = data.getDictionary();
    for (const char* domain : {"environment", "area"})
    {
        const auto restart = mReferenceControlPrefix + "." + domain + ".restart";
        mReferenceDirty |= dict.getValue(restart, false);
        dict[restart] = false;
    }
    const bool reset = mReferenceDirty;
    if (reset) { mAppearanceSamples = 0; mReferenceDirty = false; }
    const uint32_t batch = (environment || area) && mAppearanceSamples < mAppearanceMaxSamples ?
        std::min(mAppearanceSamplesPerFrame, mAppearanceMaxSamples - mAppearanceSamples) : 0u;
    dict[mReferenceControlPrefix + ".environmentEnabled"] = environment;
    dict[mReferenceControlPrefix + ".areaEnabled"] = area;
    dict[mReferenceControlPrefix + ".environment"] = (reset ? kAccumulateReset : 0u) | (environment && batch ? 0u : kAccumulateHold);
    dict[mReferenceControlPrefix + ".area"] = (reset ? kAccumulateReset : 0u) | (area && batch ? 0u : kAccumulateHold);
    if (!mReferenceOutputsInitialized || batch || mLastBatchSamples)
    {
        context->clearUAV(data.getTexture("environmentSamples")->getUAV().get(), float4(0.f));
        context->clearUAV(data.getTexture("areaSamples")->getUAV().get(), float4(0.f));
        mReferenceOutputsInitialized = true;
    }
    mLastBatchSamples = batch;
    if (!batch) return;
    const uint32_t pixels = mFrameDim.x * mFrameDim.y;
    if (!lighting.referenceWeights) lighting.referenceWeights = mpDevice->createStructuredBuffer(16, pixels);
    prepareAppearance(context);
    if (!lighting.referenceGenerate)
    {
        auto make = [&](const char* entry)
        {
            ProgramDesc desc;
            desc.addShaderModules(mpScene->getShaderModules());
            desc.addShaderLibrary(kShaderDirectory + "NeuralAppearanceReference.cs.slang").csEntry(entry);
            desc.addTypeConformances(mpScene->getTypeConformances());
            return ComputePass::create(mpDevice, desc, mpScene->getSceneDefines(), true);
        };
        lighting.referenceGenerate = make("generate"); lighting.referenceSum = make("sumFrameSamples");
    }
    FALCOR_PROFILE(context, "Original appearance IBL and area reference");
    const uint32_t end = mAppearanceSamples + batch;
    for (; mAppearanceSamples < end; ++mAppearanceSamples)
    {
        for (uint32_t domain = 0; domain < 2; ++domain)
        {
            if (domain == 0 ? !environment : !area) continue;
            auto root = lighting.referenceGenerate->getRootVar();
            bindInputs(root, data); root["gFrames"] = mpFrames; root["gLights"] = lighting.lightBuffer;
            root["gLocalDirections"] = mpRuntime->localDirections->buffer; root["gRadianceWeights"] = lighting.referenceWeights;
            root["gLinearSampler"] = lighting.sampler;
            if (lighting.envSampler) lighting.envSampler->bindShaderData(root["gEnvSampler"]);
            else
            {
                root["gEnvSampler"]["importanceMap"] = lighting.dummySource;
                root["gEnvSampler"]["importanceSampler"] = lighting.sampler;
                root["gEnvSampler"]["importanceInvDim"] = float2(1);
                root["gEnvSampler"]["importanceBaseMip"] = 0u;
            }
            for (uint32_t i = 0; i < kMaxLights; ++i)
                root["gAreaSources"][i] = i < lighting.lights.size() && lighting.records[i].params.z && lighting.lights[i].source ?
                    lighting.lights[i].source : lighting.dummySource;
            auto cb = root["gReferenceCB"];
            cb["gFrameDim"] = mFrameDim; cb["gSampleIndex"] = mAppearanceSamples;
            cb["gLightCount"] = uint32_t(lighting.lights.size()); cb["gDomain"] = domain; cb["gSeed"] = 17u;
            lighting.referenceGenerate->execute(context, mFrameDim.x, mFrameDim.y);
            context->uavBarrier(mpRuntime->localDirections->buffer.get());
            context->uavBarrier(lighting.referenceWeights.get());
            context->resourceBarrier(mpRuntime->localDirections->buffer.get(), Resource::State::Common);
            context->resourceBarrier(mpRuntime->appearanceRgba->buffer.get(), Resource::State::Common);
            context->waitForFalcor(mpRuntime->stream);
            mpRuntime->check(mpRuntime->appearance(mpRuntime->context, pixels,
                static_cast<const float*>(mpRuntime->localDirections->pointer), static_cast<float*>(mpRuntime->appearanceRgba->pointer), mpRuntime->stream));
            context->waitForCuda(mpRuntime->stream);
            context->resourceBarrier(mpRuntime->appearanceRgba->buffer.get(), Resource::State::ShaderResource);
            root = lighting.referenceSum->getRootVar();
            root["gAppearanceRGBA"] = mpRuntime->appearanceRgba->buffer;
            root["gLocalDirections"] = mpRuntime->localDirections->buffer; root["gRadianceWeights"] = lighting.referenceWeights;
            const auto frameSamples = data.getTexture(domain == 0 ? "environmentSamples" : "areaSamples");
            root["gFrameSamples"] = frameSamples;
            cb = root["gReferenceCB"]; cb["gFrameDim"] = mFrameDim; cb["gSubstrate"] = mSubstrate;
            lighting.referenceSum->execute(context, mFrameDim.x, mFrameDim.y);
            context->uavBarrier(frameSamples.get());
        }
    }
}

void NeuralMaterialPass::shade(RenderContext* context, const RenderData& data)
{
    prepareDeltaReference(context, data);
    prepareAppearanceReference(context, data);
    FALCOR_PROFILE(context, "Neural deferred IBL delta and LTC");
    const auto& lighting = *mpLighting;
    auto root = mpShadingPass->getRootVar();
    bindInputs(root, data);
    root["gRays"] = mpRuntime->rays->buffer; root["gFrames"] = mpFrames; root["gLobes"] = mpRuntime->lobes->buffer;
    root["gLights"] = lighting.lightBuffer; root["gLtcLut"] = lighting.ltcTable; root["gHorizonLut"] = lighting.horizonTable;
    root["gReferenceDirect"] = lighting.referenceDirect ? lighting.referenceDirect : lighting.dummyDirect;
    root["gAppearanceRGBA"] = mpRuntime->appearanceRgba ? mpRuntime->appearanceRgba->buffer : lighting.dummyDirect;
    root["gEnvPrefilter"].setSrv(lighting.envPrefilterSrv ? lighting.envPrefilterSrv : lighting.dummyCubeSrv);
    root["gLinearSampler"] = lighting.sampler; root["gForeground"] = data.getTexture("foreground");
    root["gDisplay"] = data.getTexture("display");
    for (uint32_t i = 0; i < kMaxLights; ++i)
    {
        const bool active = i < lighting.lights.size() && lighting.records[i].params.z != 0;
        const auto source = active ? lighting.lights[i].source : nullptr;
        const auto prefilter = active ? lighting.lights[i].prefilter : nullptr;
        root["gAreaSources"][i] = source ? source : lighting.dummySource;
        root["gAreaPrefilters"][i] = prefilter ? prefilter : lighting.dummyPrefilter;
    }
    auto cb = root["gLightingCB"];
    cb["gFrameDim"] = mFrameDim; cb["gLobeCount"] = mpRuntime->info.lobes;
    cb["gLobeStride"] = mpRuntime->lobeStride(); cb["gLightCount"] = uint32_t(lighting.lights.size());
    cb["gEnvironmentEnabled"] = uint32_t(mEnvironmentEnabled); cb["gDeltaEnabled"] = uint32_t(mDeltaEnabled);
    cb["gAreaEnabled"] = uint32_t(mAreaEnabled && lighting.activePolygonLights > 0); cb["gBackgroundEnabled"] = uint32_t(mBackgroundEnabled);
    cb["gBackgroundGain"] = std::exp2(mBackgroundExposure);
    cb["gEmitterGain"] = std::exp2(mEmitterExposure);
    cb["gHorizonOcclusion"] = uint32_t(mHorizonOcclusion); cb["gAreaMode"] = mAreaMode;
    cb["gReferenceSamples"] = mReferenceSamples; cb["gDebugView"] = mDebugView;
    cb["gEnvMipCount"] = lighting.envLevels; cb["gHasEnvironment"] = uint32_t(bool(mpScene->getEnvMap()));
    cb["gEnvMinR"] = lighting.envMinimumRoughness; cb["gSubstrate"] = mSubstrate;
    cb["gMipLevels"] = mpRuntime->info.mip_levels; cb["gDeltaReference"] = uint32_t(mDeltaReference);
    cb["gEnvironmentReference"] = uint32_t(mEnvironmentReference);
    mpShadingPass->execute(context, mFrameDim.x, mFrameDim.y);
    ++mShadingFrames;
}

void NeuralMaterialPass::renderLightControls(Gui::Widgets& widget)
{
    auto& lighting = *mpLighting;
    const double time = mpScene ? mpScene->getTime() : 0.0;
    if (auto group = widget.group("Environment controls"))
    {
        if (mpScene && mpScene->getEnvMap())
        {
            auto env = mpScene->getEnvMap();
            auto rotation = env->getRotation();
            auto tint = env->getTint();
            float intensity = env->getIntensity();
            bool changed = group.var("Rotation XYZ (degrees)", rotation, -360.f, 360.f, .5f);
            if (changed)
            {
                lighting.environmentRotation = rotation;
                lighting.environmentMotion.startTime = time;
                env->setRotation(rotation);
            }
            if (group.var("Intensity", intensity, 0.f, 100000.f, .05f)) { env->setIntensity(intensity); changed = true; }
            if (group.rgbColor("Tint", tint)) { env->setTint(tint); changed = true; }
            if (lighting.environmentMotion.renderUI(group, false))
            {
                lighting.environmentRotation = env->getRotation();
                lighting.environmentMotion.startTime = time;
            }
            lighting.controlsDirty |= changed;
        }
        else group.text("Load an environment through Falcor's Scene controls first.");
        group.textWrapped("Rotation uses the Falcor timeline and does not rebake the environment texture.");
    }
    if (auto group = widget.group("Delta light controls"))
    {
        if (!mpScene) group.text("Load a scene to add lights.");
        else
        {
            auto add = [&](bool directional, bool spot)
            {
                std::string name;
                uint32_t suffix = mpScene->getLightCount();
                do { name = std::string("Neural ") + (directional ? "directional " : spot ? "spot " : "point ") + std::to_string(suffix++); }
                while (mpScene->getLightByName(name));
                ref<Falcor::Light> source;
                if (directional)
                {
                    auto light = DirectionalLight::create(name); light->setWorldDirection(float3(0.f, 0.f, -1.f)); source = light;
                }
                else
                {
                    auto light = PointLight::create(name); light->setWorldPosition(float3(0.f, 0.f, 2.f));
                    light->setWorldDirection(float3(0.f, 0.f, -1.f));
                    if (spot) { light->setOpeningAngle(kPi / 6.f); light->setPenumbraAngle(kPi / 36.f); }
                    source = light;
                }
                source->setIntensity(float3(directional ? 1.f : 6.f));
                mpScene->addLight(source); lighting.controlsDirty = true;
                mDeltaEnabled = true; mDeltaDirty = true;
            };
            if (group.button("Add point light")) add(false, false);
            if (group.button("Add spot light", true)) add(false, true);
            if (group.button("Add directional light")) add(true, false);
            for (auto& c : lighting.deltaControls)
            {
                auto source = c.light;
                auto item = group.group(source->getName());
                if (!item) continue;
                // Reuse Falcor's native power, color, position, direction and spot controls.
                const auto before = source->getData();
                source->renderUI(item);
                const auto after = source->getData();
                const bool poseEdited = any(before.posW != after.posW) || any(before.dirW != after.dirW);
                const bool motionEdited = c.motion.renderUI(item, source->getType() == LightType::Point);
                if (motionEdited || poseEdited)
                {
                    c.position = source->getData().posW;
                    c.direction = source->getData().dirW;
                    c.motion.startTime = time;
                    if (c.motion.autoRotate || c.motion.autoOrbit) source->setIsAnimated(false);
                }
                lighting.controlsDirty |= std::memcmp(&before, &after, sizeof(before)) != 0;
                if (item.button("Remove delta light"))
                {
                    const auto& lights = mpScene->getLights();
                    const auto found = std::find(lights.begin(), lights.end(), source);
                    if (found != lights.end())
                    {
                        const int32_t id = int32_t(std::distance(lights.begin(), found));
                        mpScene->removeLight(uint32_t(id));
                        for (auto& polygon : lighting.lights)
                        {
                            if (polygon.sceneLight == id) polygon.sceneLight = -1;
                            else if (polygon.sceneLight > id) --polygon.sceneLight;
                        }
                    }
                    lighting.controlsDirty = true;
                    break;
                }
            }
        }
    }
    if (auto group = widget.group("Polygon lights"))
    {
        if (lighting.lights.size() < kMaxLights)
        {
            if (group.button("Add star light")) lighting.lights.push_back(Lighting::Light{});
            if (lighting.lights.size() < kMaxLights && group.button("Add rectangle light", true))
            { Lighting::Light light; light.shape = 0; lighting.lights.push_back(light); }
            if (lighting.lights.size() < kMaxLights && group.button("Add textured light..."))
            {
                std::filesystem::path path;
                if (openFileDialog(Bitmap::getFileDialogFilters(), path))
                {
                    Lighting::Light light; light.shape = 0; light.textureKind = 3; light.texturePath = path.string();
                    lighting.lights.push_back(light);
                }
            }
        }
        for (size_t index = 0; index < lighting.lights.size(); ++index)
        {
            auto item = group.group("Light " + std::to_string(index));
            if (!item) continue;
            auto& light = lighting.lights[index];
            item.checkbox("Enabled", light.enabled);
            light.bakeDirty |= item.dropdown("Shape", {{0, "Rectangle"}, {1, "Star"}}, light.shape);
            light.bakeDirty |= item.dropdown("Emission texture", {{0, "White"}, {1, "Rainbow"}, {2, "Sunset"}, {3, "Image"}}, light.textureKind);
            Gui::DropdownList bindings = {{0, "Manual polygon transform"}};
            if (mpScene)
                for (uint32_t id = 0; id < mpScene->getLightCount(); ++id)
                    if (mpScene->getLight(id)->getType() == LightType::Rect)
                        bindings.push_back({id + 1, mpScene->getLight(id)->getName()});
            uint32_t binding = uint32_t(light.sceneLight + 1);
            if (item.dropdown("Transform source", bindings, binding)) light.sceneLight = int32_t(binding) - 1;
            if (light.sceneLight < 0)
            {
                auto center = light.motion.position(light.center, time);
                auto rotation = light.motion.rotation(light.rotation, time);
                bool changed = item.var("Center", center, -10000.f, 10000.f, .02f);
                changed |= item.var("Rotation XYZ (degrees)", rotation, -360.f, 360.f, .5f);
                changed |= light.motion.renderUI(item);
                if (changed)
                {
                    light.center = center; light.rotation = rotation; light.motion.startTime = time;
                }
                item.var("Full width / height", light.size, .001f, 10000.f, .02f);
            }
            else item.text("Transform/animation follow the bound Falcor RectLight.");
            item.rgbColor("Tint", light.color);
            item.var(light.sceneLight < 0 ? "Radiance scale" : "Falcor radiance multiplier", light.intensity, 0.f, 100000.f, .1f);
            if (light.textureKind == 3)
            {
                item.textbox("Emission image", light.texturePath);
                if (item.button("Browse emission image..."))
                {
                    std::filesystem::path path;
                    if (openFileDialog(Bitmap::getFileDialogFilters(), path))
                    { light.texturePath = path.string(); light.bakeDirty = true; }
                }
                if (item.button("Load emission image")) light.bakeDirty = true;
            }
            if (!light.textureError.empty()) item.textWrapped("Image load failed; previous emission retained. " + light.textureError);
            if (item.button("Remove light")) { lighting.lights.erase(lighting.lights.begin() + index); break; }
        }
        group.textWrapped("Automatic motion follows Falcor's timeline. Each X/Y/Z axis has its own enable and speed; orbit uses a world-space center.");
    }
}

void NeuralMaterialPass::renderUI(Gui::Widgets& widget)
{
    widget.text("FP16 cooperative vectors | raster visibility | no shadows");
    if (!mError.empty()) widget.textWrapped(mError);
    const Gui::DropdownList presets = {{0, "Fuzzy silk learned mips"}, {1, "Fuzzy silk single level"}, {2, "Shot silk | 4 LOD | BCE IBL"}, {3, "Velvet trained LOD IBL"}, {4, "Chainmail | 4 LOD | BCE IBL"}, {5, "Blue gray pile | full ray | LOD0 | BCE IBL"}, {6, "Custom asset"}};
    const std::array<const char*, 6> names = {"fuzzy_mip", "fuzzy_silk", "shot_silk", "velvet_mip", "chainmail_jitter", "blue_gray_pile_fullray_lod0"};
    uint32_t preset = uint32_t(names.size());
    for (uint32_t i = 0; i < names.size(); ++i) if (std::filesystem::path(mModelPath).stem() == names[i]) preset = i;
    if (widget.dropdown("Material", presets, preset) && preset < names.size())
    {
        mModelPath = (std::filesystem::path(mModelPath).parent_path() / (std::string(names[preset]) + ".mpmat")).string();
        mReloadModel = true;
    }
    widget.textbox("Model asset", mModelPath);
    if (widget.button("Load model")) mReloadModel = true;
    if (widget.button("Reload shaders / retry"))
    {
        mpMappingPass = nullptr; mpShadingPass = nullptr; mpLighting->envBake = nullptr; mpLighting->lightBake = nullptr;
        mpLighting->referenceGenerate = nullptr; mpLighting->referenceSum = nullptr;
        mpLighting->deltaGenerate = nullptr; mpLighting->deltaAccumulate = nullptr;
        mRenderingFailed = false; mError.clear(); mGeometryDirty = true; mDeltaDirty = true;
    }
    if (mpRuntime)
    {
        widget.text(fmt::format("{} lobes | {} mip level(s) | {} planes x {} channels", mpRuntime->info.lobes,
            mpRuntime->info.mip_levels, mpRuntime->info.planes, mpRuntime->info.channels));
    }
    bool mappingChanged = false;
    if (auto group = widget.group("Neural material mapping"))
    {
        mappingChanged |= group.var("Material ID (-1 = all)", mMaterialId, -1, std::numeric_limits<int32_t>::max());
        mappingChanged |= group.var("UV repeats", mUvScale, .001f, 1000.f, .01f);
        mappingChanged |= group.var("Source crop", mSourceCrop, 0.f, .499f, .001f);
        mappingChanged |= group.var("Thickness (world units)", mThickness, .00001f, 10.f, .001f);
        mappingChanged |= group.var("Maximum parallax (0 = off)", mMaxParallax, 0.f, 100.f, .01f);
        mappingChanged |= group.checkbox("Smooth normals", mSmoothNormals);
        mappingChanged |= group.checkbox("Source V follows Blender/OBJ", mFlipV);
        mappingChanged |= group.checkbox("Two-sided material", mTwoSided);
        mappingChanged |= group.checkbox("Top reference surface", mReferenceTop);
        mappingChanged |= group.checkbox("Normal-derived stack correction", mStackCorrection);
        mappingChanged |= group.checkbox("Periodic source tiles", mPeriodic);
        mappingChanged |= group.checkbox("Nearest feature XY", mNearestFeatures);
        mappingChanged |= group.checkbox("Automatic feature LOD", mAutomaticLod);
        const float lastMip = mpRuntime ? float(mpRuntime->info.mip_levels - 1) : 0.f;
        if (mAutomaticLod) mappingChanged |= group.var("Feature LOD bias", mLodBias, -8.f, 8.f, .05f);
        else mappingChanged |= group.var("Fixed feature LOD", mFeatureLod, 0.f, lastMip, .05f);
        group.text("Raster ddx/ddy at reference surface; one LOD shared by every plane.");
        mappingChanged |= group.checkbox("Cache inference", mCacheInference);
    }
    if (mappingChanged) { mGeometryDirty = true; mDeltaDirty = true; }
    if (auto group = widget.group("Texture synthesis"))
    {
        if (mpRuntime && (mpRuntime->info.capabilities & 4))
        {
            mSynthesisDirty |= group.dropdown("Mode", {{0, "Off"}, {1, "Hex"}, {2, "Histogram"}}, mSynthesisMode);
            mSynthesisDirty |= group.var("Seed", mSynthesisSeed, 0, std::numeric_limits<int32_t>::max());
            mSynthesisDirty |= group.var("Scale", mSynthesisScale, .25f, 8.f, .01f);
            mSynthesisDirty |= group.var("Blend sharpness", mSynthesisPower, 1.f, 16.f, .1f);
            mSynthesisDirty |= group.var("Rotation strength", mSynthesisRotation, 0.f, 25.f, .1f);
            mSynthesisDirty |= group.var("Border contrast", mSynthesisBorder, .01f, .99f, .01f);
        }
        else group.textWrapped("This learned mip model has no synthesis path. Disable synthesis before loading it.");
    }
    if (auto group = widget.group("Lighting"))
    {
        mReferenceDirty |= group.checkbox("Environment lighting", mEnvironmentEnabled);
        mReferenceDirty |= group.checkbox("Original appearance IBL reference", mEnvironmentReference);
        group.checkbox("Environment background", mBackgroundEnabled);
        group.var("Background exposure (EV)", mBackgroundExposure, -10.f, 10.f, .1f);
        group.var("Visible light exposure (EV)", mEmitterExposure, -16.f, 8.f, .1f);
        group.tooltip("Exposure of directly visible polygon light surfaces only. Lower it to reveal emission textures. Does not change illumination, background or reference accumulation. Zero restores unscaled emission.");
        mDeltaDirty |= group.checkbox("Delta lights", mDeltaEnabled);
        uint32_t deltaMode = mDeltaReference ? 1u : 0u;
        if (group.dropdown("Delta response", {{0, "Cached lobes"}, {1, "Original appearance reference (FP16)"}}, deltaMode))
        { mDeltaReference = deltaMode != 0; mDeltaDirty = true; }
        mReferenceDirty |= group.checkbox("Polygon lights", mAreaEnabled);
        mReferenceDirty |= group.dropdown("Polygon integration", {{0, "Whole-light LTC + prefilter"},
            {1, "Same-lobe numerical reference"}, {2, "Original appearance progressive reference"}}, mAreaMode);
        if (mAreaMode == 1) group.var("Reference samples", mReferenceSamples, 1u, 4096u);
        group.checkbox("Lobe horizon correction", mHorizonOcclusion);
        if (group.var("Diffuse substrate", mSubstrate, 0.f, 1.f, .01f)) { mDeltaDirty = true; mReferenceDirty = true; }
        group.tooltip("Adds a lit diffuse backing where learned coverage is incomplete. Any positive value makes the surface opaque; zero uses learned coverage.");
        if (group.var("Environment bake samples", mpLighting->envSamples, 32u, 16384u)) mpLighting->envDirty = true;
        if (mEnvironmentReference || mAreaMode == 2)
        {
            group.var("Appearance samples per frame", mAppearanceSamplesPerFrame, 1u, 64u);
            if (group.var("Appearance sample limit", mAppearanceMaxSamples, 1u, 1048576u) && mAppearanceMaxSamples < mAppearanceSamples)
                mReferenceDirty = true;
            group.text(fmt::format("Accumulated appearance samples: {}", mAppearanceSamples));
            if (group.button("Reset appearance accumulation")) mReferenceDirty = true;
            group.textWrapped("Falcor AccumulatePass nodes accumulate frame sample sums and counts. Motion, lighting and substrate edits restart them; exposure and batch size preserve them.");
        }
        group.textWrapped("Falcor owns scene lights, environment rotation/intensity, camera, animation and tone mapping.");
    }
    renderLightControls(widget);
    if (auto group = widget.group("Diagnostics"))
    {
        group.dropdown("View", {{0, "Radiance"}, {1, "Feature LOD"}, {2, "First lobe axis"}, {3, "Roughness"}, {4, "Coverage"}, {5, "Normal"}}, mDebugView);
        group.text(fmt::format("Inference frames: {} | shading frames: {}", mInferenceFrames, mShadingFrames));
        if (mpRuntime) group.text(fmt::format("Lobe buffer: {} | {} bytes/pixel",
            mpRuntime->packedLobes ? "FP16 packed" : "FP32 transport", mpRuntime->lobeWords() * 4u));
        group.textbox("Explicit buffer dump directory", mDumpPath);
        if (group.button("Dump rays / frames / lobes")) mDumpRequested = true;
        group.textWrapped("Buffer dumps synchronize and read GPU data only when requested. Normal display stays on the GPU; Falcor's profiler measures GPU stages.");
    }
}
