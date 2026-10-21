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
#include "PortCompat.h"
#include "NNLaplacian.h"
#include "DummyNeuralNetwork.h"
#include "RenderGraph/RenderPassHelpers.h"

#include "Scene/HitInfo.h"
#include "json/json.hpp"
#include "cub/cub.cuh"
#include "tiny-cuda-nn/common.h"
#include <tiny-cuda-nn/common_device.h>
#include "NNLaplacian.h"
#include "Legacy/NeuralNetwork/NeuralStructures.slang"



#define DEMO 0

namespace
{

    const char kShaderFile[] = "RenderPasses/NeuralRadianceCache/PathTracer.rt.slang";
    const char kParameterBlockName[] = "gData";

    // Ray tracing settings that affect the traversal stack size.
    // These should be set as small as possible.
    // The payload for the scatter rays is 8-12B.
    // The payload for the shadow rays is 4B.
    const uint32_t kMaxPayloadSizeBytes = Falcor::HitInfo::kMaxPackedSizeInBytes;
    const uint32_t kMaxAttributeSizeBytes = 8;
    const uint32_t kMaxRecursionDepth = 1;

    const Falcor::uint2 DEBUG_RES = Falcor::uint2(256, 256);

    const std::string kIsNRCEnable = "IsNRCEnable";
    const std::string kIsBiased = "IsBiased";
    const std::string kNumNRCSamples = "NumNRCSamples";

    const std::string kMullerHeuristic = "MullerHeuristic";
    const std::string kOurHeuristic = "OurHeuristic";
    const std::string kErrorBasedHeuristic = "ErrorBasedHeuristic";
    const std::string kDoErrorTest = "DoErrorTest";
    const std::string kErrorThreshold = "ErrorThreshold";

    const std::string kIsIncidentCache = "IsIncidentCache";
    const std::string kDebugHemispherical = "DebugHemispherical";
    const std::string kQLearning = "QLearning";
    const std::string kNumGTSamples = "NumGTSamples";

    // Render pass output channels.
    const std::string kColorOutput = "color";
    const std::string kAlbedoOutput = "albedo";
    const std::string kNormalOutput = "normal";
    const std::string kTimeOutput = "time";

    const Falcor::ChannelList kOutputChannels =
    {

        { kAlbedoOutput,    "gOutputAlbedo",              "Surface albedo (base color) or background color", true /* optional */   },
        { kNormalOutput,    "gOutputNormal",              "Normal output", true /* optional */    },
        { kTimeOutput,      "gOutputTime",                "Per-pixel execution time", true /* optional */, Falcor::ResourceFormat::R32Uint  }
    };

    const char kDesc[] = "Insert pass description here";

    // Note: Path is now determined dynamically at runtime
    // const std::string addPath = "D://GradientDiffusion//gradientdiffusion//Falcor//Source//";

    const std::string kColorOutputFinal = "colorOutputFinal";

    const std::string kColor2MomentOutput = "color2MomentOutput";
    const std::string kColorOutputDebug = "colorOutputDebug";
    const std::string kColorOutputDebugGT = "colorOutputDebugGT";
    const std::string kPredictedRadianceMerged = "gPredictedRadianceMerged";
    const std::string kConvertBufferToTexture = "RenderPasses/NeuralRadianceCache/buffToTexDynamicReal.cs.slang";
    const std::string kConvertBufferToTextureFused = "RenderPasses/NeuralRadianceCache/buffToTexDynamicRealFused.cs.slang";
    const std::string kFinalOutput = "RenderPasses/NeuralRadianceCache/combine.cs.slang";
    const std::string kEnvMapRequest = "RenderPasses/NeuralRadianceCache/envMapRequest.cs.slang";
    const std::string kGatherOurEstimations = "RenderPasses/NeuralRadianceCache/gatherOurEstimations.cs.slang";

    const std::string kYTrainBufferAssembly = "RenderPasses/NeuralRadianceCache/assembler.cs.slang";

    const std::string kNetworkConfigNIRC = "RenderPasses/NeuralRadianceCache/network_config_nirc.json";
    const std::string kNetworkConfigNIRCEnvMap = "RenderPasses/NeuralRadianceCache/network_config_nirc_env_map.json";
    const std::string kNetworkConfigNRC = "RenderPasses/NeuralRadianceCache/network_config_nrc.json";
    const std::string kNetworkConfigEstimator = "RenderPasses/NeuralRadianceCache/network_estimator_config.json";

    const std::string kPyramidConfig = "RenderPasses/NeuralRadianceCache/pyramid_config.json";

    const std::string kNetworkEstimatorPredBuffer = "gNetworkEstimatorPredBuffer";

    const std::string kNetworkEstimatorThroughput = "gNetworkEstimatorThroughput";

    const std::string kQLearningPredBuffer = "gQLearningPredBuffer";
    const std::string kQLearningPredBufferMapping = "gQLearningPredBufferMapping";
    const std::string kQLearningMapping = "gQLearningMapping";
    const std::string kQLearningThroughput = "gQLearningThroughput";

    const std::string kResErrorID = "gResErrorID";
    const std::string kXTrainResError = "gXTrainResErrorID";
    const std::string kPredBuffer = "gPredBuffer";
    const std::string kPredBufferMapping = "gPredBufferMapping";
    const std::string kPredPixelID = "gPredBufferPixelID";
    const std::string kPredThroughput = "gPredBufferThroughput";

    const std::string kPredBufferOutActAddData = "gPredBufferOutActAddData";
    const std::string kXTrainBuffer = "gXTrainBuffer";
    const std::string kPredictedRadianceSTYBuffer = "gPredictedRadianceSTYBuffer";
    const std::string kLThpSelfTrainingBuffer = "gLThpSelfTrainingBuffer";
    const std::string kXTrainAdditionalBuffer = "gXTrainAdditionalBuffer";
    const std::string kYTrainBuffer = "gYTrainBuffer";
    const std::string kExpectedValuesTmp = "gExpectedValueTmp";
    const std::string kLossVarianceScatterIDBuffer = "gLossVarianceScatterID";
    const std::string kLossVarianceThroughputBuffer = "gLossVarianceThroughput";
    const std::string kEstimatorNNTrainBuffer = "gEstimatorNNXTrainBuffer";
    const std::string kTrainPixel = "gTrainPixel";
    const std::string kOurEstimations = "ourEstimations";
    const std::string kGTEstimations = "gtEstimations";
    const std::string kEstimatorNNXTrainAdditionalBuffer = "gEstimatorNNXTrainAdditionalBuffer";


    // Mapping of idNN to index in X Buffers
    /*const std::string kNNMapPredBufferKeys = "gNNMapPredBufferKeys";
    const std::string kNNMapPredBufferValues = "gNNMapPredBufferValues";

    const std::string kNNMapTrainBufferKeys = "gNNMapTrainBufferKeys";
    const std::string kNNMapTrainBufferValues = "gNNMapTrainBufferValues";*/
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<Falcor::RenderPass, Falcor::DummyNeuralNetwork>();
    Falcor::ScriptBindings::registerBinding([](pybind11::module& m) {
        pybind11::class_<Falcor::DummyNeuralNetwork, Falcor::RenderPass, Falcor::ref<Falcor::DummyNeuralNetwork>> pass(m, "NeuralRadianceCache");
        pass.def("resetLearning", &Falcor::DummyNeuralNetwork::resetNN);
        pass.def("setTraining", &Falcor::DummyNeuralNetwork::enableTraining);
        pass.def_property_readonly("trainingEpoch", &Falcor::DummyNeuralNetwork::getTrainingEpoch);
        pass.def_property_readonly("trainingLoss", &Falcor::DummyNeuralNetwork::getLoss);
        pass.def("getSettings", [](Falcor::DummyNeuralNetwork& p) { return p.getProperties().toPython(); });
        pass.def("setSettings", [](Falcor::DummyNeuralNetwork& p, const pybind11::dict& changes) {
            auto settings = p.getProperties().toPython();
            settings.attr("update")(changes);
            p.parseDictionary(Falcor::Properties(settings));
        });
    });
}

Falcor::DummyNeuralNetwork::DummyNeuralNetwork(ref<Device> pDevice, const Falcor::Properties& dict)
    : LegacyPathTracer(pDevice, dict, kOutputChannels)
{
    mpConvertNNOutputPass = ComputePass::create(portDevice(), kConvertBufferToTexture);
    mpConvertNNOutputPassFused = ComputePass::create(portDevice(), kConvertBufferToTextureFused);
    mpGatherOutEstimations = ComputePass::create(portDevice(), kGatherOurEstimations);
    mpFinalOutputPass = ComputePass::create(portDevice(), kFinalOutput);
    mpYTrainAssemblyPass = ComputePass::create(portDevice(), kYTrainBufferAssembly);
    mpEnvMapRequests = ComputePass::create(portDevice(), kEnvMapRequest);

    uint32_t resolutionPerBounds = mScreenDim.x * mScreenDim.y * MAX_SAMPLES;
    resolutionPerBounds = (resolutionPerBounds + 255) / 256 * 256;

    uint32_t resolutionPerBoundsBase = mScreenDim.x * mScreenDim.y * 3;
    resolutionPerBoundsBase = (resolutionPerBoundsBase + 255) / 256 * 256;

#if 1
    mainRenderData = Falcor::NeuralRequestData("main", 3, resolutionPerBounds, resolutionPerBoundsBase, true);
    debugRenderData = Falcor::NeuralRequestData("debug", 3, DEBUG_RES.x * DEBUG_RES.y, -1, true);
#else
    // Supporting Direct NRC without recompiling by sacrificing some memory
    mainRenderData = Falcor::NeuralRequestData("main", 3, resolutionPerBounds, -1, true);
    debugRenderData = Falcor::NeuralRequestData("debug", 3, DEBUG_RES.x* DEBUG_RES.y, -1, true);
#endif

    bResidualVarianceLoss = false;

    mNumNRCSamplesPerDepth[0] = 4;
    mNumNRCSamplesPerDepth[1] = 4;
    mNumNRCSamplesPerDepth[2] = 0;

    mNumQLearningSamples = 3;
    bNRCBiased = false;
    bDirectLightCache = false;
    bResErrorForLoss = false;
    bDebugHemispherical = false;
    bNRCEnabled = true;
    bNRCDebugMode = false;
    bOnlyIndirect = false;
    bVarianceLoss = true;
    mNumVarianceLossSamples = 4;
    bQLearning = false;
    bIncidentNRC = true;

    bDispatchTrainingRays = true;
    maxBounceWithCV = 1;

    bDirectLightCacheNEE = false;

    bFusedOutput = true;
    mSharedParams.maxBounces = 5;
    mSharedParams.maxNonSpecularBounces = 5;
    roughnessThreshold = 0;
    mSharedParams.useEnvLightNEE = 0;

    bVarianceLoss = false;
    bDebugHemispherical = false;
    //bFilteringNetwork = true;
#if 1
    mSharedParams.useNEE  = 1;
#endif

#if 0
    bIncidentNRC = true;
    bVarianceLoss = true;
    mNumNRCSamples = 1;
    bOnlyIndirect = true;
    bUseNNAsEstimator = false;
#endif

#if 0
    mNumNRCSamples = 3;
    bDebugHemispherical = true;
    bDirectLightCacheNEE = false;
    bIncidentNRC = true;
    bDirectLightCache = true;
    bOnlyIndirect = false;
    //mNumVarianceLossSamples = 4;
    maxBounceWithCV = 1;
    mSharedParams.maxBounces = 1;
    mSharedParams.useNEE = 0;
    mSharedParams.maxNonSpecularBounces = 1;
#endif

    //bNRCEnabled = false;
    maxBounceWithCV = 2;
    bStopByMetric = 1;
    mNumNRCSamplesPerDepth[0] = 1;
    mNumNRCSamplesPerDepth[1] = 0;


#if 0
    bNRCBiased = true;
    bStopByMetric = false;
    bFusedOutput = false;
    bForceNonSeparateInput = true;
#endif

#if 0
    bStopByMetric = 0;
    bDebugHemispherical = true;
    bDirectLightCacheNEE = false;
    bIncidentNRC = true;
    bDirectLightCache = true;
    bOnlyIndirect = false;
    //mNumVarianceLossSamples = 4;
    maxBounceWithCV = 1;
    mSharedParams.maxBounces = 1;
    mSharedParams.useNEE = 0;
    mSharedParams.maxNonSpecularBounces = 1;
    bNRCBiased = true;
#endif

#if DEMO
    bStopByMetric = 1;
    bIncidentNRC = true;
    bNRCBiased = true;
    mSharedParams.maxBounces = 15;
    mSharedParams.maxNonSpecularBounces = 15;
    bOnlyIndirect = false;
    bDirectLightCache = false;
    maxBounceWithCV = 3;

    mNumNRCSamplesPerDepth[0] = 1;
    mNumNRCSamplesPerDepth[1] = 1;
    mNumNRCSamplesPerDepth[2] = 1;
#endif

#if 1
    mSharedParams.useEnvLightNEE = false;
    bStopByMetric = 1;
    bIncidentNRC = true;
    bNRCBiased = true;
    bOnlyIndirect = false;
    bDirectLightCache = false;
    maxBounceWithCV = 3;

    mNumNRCSamplesPerDepth[0] = 3;
    mNumNRCSamplesPerDepth[1] = 1;
    mNumNRCSamplesPerDepth[2] = 1;
#endif

    bStopByOurMetric = true;

    std::ifstream networkConfigStream;
    std::string networkConfigFullPath;

    std::string dummy_path = std::string();
    dummy_path.clear();

    // Note: Old hardcoded path removed for privacy

    if (portFindFile(dummy_path + kPyramidConfig, networkConfigFullPath)) {
        networkConfigStream.open(networkConfigFullPath);
    }
    else {
        throw std::runtime_error("Config for " + dummy_path + kPyramidConfig + " not found");
    }



    nlohmann::json config;
    networkConfigStream >> config;
    networkConfigStream.close();

    m_LOD = config["numLODs"] + 1;
    lodSizes.push_back(1);

    bGradientDenoising = false;
    for (uint32_t i = 0; i < m_LOD - 1; i++) {
        lodSizes.push_back(config["lodSize"][i]);
    }

    if (config["gradient_denoising"]["enable"]) {
        bGradientDenoising = true;
        gradientHashGridRes = config["gradient_denoising"]["hash_grid_res"];
        gradientHashGridSize = config["gradient_denoising"]["hash_grid_size"];
    }
    else {
        gradientHashGridRes = 256;
        gradientHashGridSize = 1024;
    }

    parseDictionary(dict);
}

std::string Falcor::DummyNeuralNetwork::getDesc() { return kDesc; }

 Falcor::Properties Falcor::DummyNeuralNetwork::getProperties() const
{
    Properties p = LegacyPathTracer::getProperties();
    p["architecture"] = mNircArchitecture == 0 ? "Original" : mNircArchitecture == 1 ? "DirectionalHash" : "Hybrid";
    p["interpolation"] = mNircInterpolation == 0 ? "Trilinear" : "Tetrahedral";
    p["mode"] = bNRCEnabled ? (bIncidentNRC ? "NIRC" : "NRC") : "PathTracing";
    p["correction"] = !bNRCBiased;
    p["training"] = bDispatchTrainingRays;
    p["referenceSamples"] = mNumGTSamples;
    p["cacheSamples"] = mNumNRCSamplesPerDepth[0];
    p["cacheSamplesBounce2"] = mNumNRCSamplesPerDepth[1];
    p["cacheSamplesBounce3"] = mNumNRCSamplesPerDepth[2];
    p["lastCorrectionBounce"] = maxBounceWithCV;
    p["fusedOutput"] = bFusedOutput;
    p["indirectOnly"] = bOnlyIndirect;
    p["directionalInspector"] = bDebugHemispherical;
    return p;
}

void Falcor::DummyNeuralNetwork::nrc_experiment_setup() {
    bIncidentNRC = false;
    bDirectLightCache = false;

    mNumNRCSamplesPerDepth[0] = 1;
    mNumNRCSamplesPerDepth[1] = 0;
    mNumNRCSamplesPerDepth[2] = 0;
}

 void Falcor::DummyNeuralNetwork::nirc_experiment_setup(int samples) {
    bIncidentNRC = true;
    bDirectLightCache = false;

    mNumNRCSamplesPerDepth[0] = samples;
    mNumNRCSamplesPerDepth[1] = 0;
    mNumNRCSamplesPerDepth[2] = 0;
}

 void Falcor::DummyNeuralNetwork::enableTraining(bool bEnable) {
    if(model)
        model->enableTraining(bEnable);

    if(modelEstimator)
        modelEstimator->enableTraining(bEnable);

    bDispatchTrainingRays = bEnable;
}

 float Falcor::DummyNeuralNetwork::getAveragePath() const {
    PixelStats::Stats stats;
    mpPixelStats->getStats(stats);
    return stats.avgPathLength;
}

 void Falcor::DummyNeuralNetwork::enableInference(bool bEnable) {
    if(model)
        model->enablePrediction(bEnable);

    if(modelEstimator)
        modelEstimator->enablePrediction(bEnable);
}

 float Falcor::DummyNeuralNetwork::getLoss() const {
    return model ? model->m_loss : 0.f;
}

 int Falcor::DummyNeuralNetwork::getTrainingStep() const {
    return model ? model->mTrainingStep : 0;
}

 void Falcor::DummyNeuralNetwork::resetNN() {
    bResetModel = true;
    mOptionsChanged = true;
}

 void Falcor::DummyNeuralNetwork::parseDictionary(const Falcor::Properties& props)
{
    const bool previousIncident = bIncidentNRC;
    const uint32_t previousArchitecture = mNircArchitecture;
    const uint32_t previousInterpolation = mNircInterpolation;
    const std::string interpolation = props.get("interpolation", std::string("Tetrahedral"));
    if (interpolation != "Trilinear" && interpolation != "Tetrahedral") FALCOR_THROW("interpolation must be Trilinear or Tetrahedral");
    mNircInterpolation = interpolation == "Trilinear" ? 0u : 1u;
    const std::string architecture = props.get("architecture", std::string("Original"));
    if (architecture != "Original" && architecture != "DirectionalHash" && architecture != "Hybrid")
        FALCOR_THROW("architecture must be Original, DirectionalHash, or Hybrid");
    mNircArchitecture = architecture == "Original" ? 0u : architecture == "DirectionalHash" ? 1u : 2u;
    LegacyPathTracer::parseDictionary(props);
    std::string mode = props.get("mode", std::string("NIRC"));
    if (mode != "NRC" && mode != "NIRC" && mode != "PathTracing") FALCOR_THROW("mode must be NRC, NIRC, or PathTracing");
    bNRCEnabled = mode != "PathTracing";
    bIncidentNRC = mode == "NIRC";
    bNRCBiased = !props.get("correction", true);
    bDispatchTrainingRays = props.get("training", true);
    mNumGTSamples = std::clamp(props.get("referenceSamples", 1u), 1u, 10u);
    mNumNRCSamplesPerDepth[0] = props.get("cacheSamples", bIncidentNRC ? 4u : 1u);
    mNumNRCSamplesPerDepth[1] = props.get("cacheSamplesBounce2", 0u);
    mNumNRCSamplesPerDepth[2] = props.get("cacheSamplesBounce3", 0u);
    if (mNumNRCSamplesPerDepth[0] + mNumNRCSamplesPerDepth[1] + mNumNRCSamplesPerDepth[2] > MAX_SAMPLES-3)
        FALCOR_THROW("Total cache samples must be at most {}", MAX_SAMPLES-3);
    maxBounceWithCV = props.get("lastCorrectionBounce", 3u);
    bFusedOutput = props.get("fusedOutput", true);
    bOnlyIndirect = props.get("indirectOnly", false);
    bDebugHemispherical = props.get("directionalInspector", false);
    if (model && (previousIncident != bIncidentNRC || previousArchitecture != mNircArchitecture || previousInterpolation != mNircInterpolation)) bResetModel = true;
    enableTraining(bDispatchTrainingRays);
    mOptionsChanged = true;
    recreateVars();
}

void Falcor::DummyNeuralNetwork::prepareVars()
{
    assert(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = Falcor::RtProgramVars::create(portDevice(), mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    bool success = (mpSampleGenerator->bindShaderData(var), true);
    if (!success) throw std::exception("Failed to bind sample generator");

    // Create parameter block for shared data.
    auto pReflection = mTracer.pProgram->getReflector();
    auto pBlockReflection = pReflection->getParameterBlock(kParameterBlockName);
    assert(pBlockReflection);
    mTracer.pParameterBlock = Falcor::ParameterBlock::create(portDevice(), pBlockReflection);
    assert(mTracer.pParameterBlock);

    // Bind static resources to the parameter block here. No need to rebind them every frame if they don't change.
    // Bind the light probe if one is loaded.
    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(mTracer.pParameterBlock->getRootVar()["envMapSampler"]);

    // Bind the parameter block to the global program variables.
    mTracer.pVars->setParameterBlock(kParameterBlockName, mTracer.pParameterBlock);
}


bool Falcor::NeuralRequestData::allocateBuffers(Falcor::PortBuffers& buffers) {
    if (!bInit)
        return false;

    portStructuredBuffer(buffers, kPredBuffer + "0"+name, numUniqueElements, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredBuffer + "1" + name, numUniqueElements, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredBuffer + "2" + name, numUniqueElements, sizeof(float), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredBuffer + "3" + name, numUniqueElements, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredBuffer + "4" + name, numElements, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredBufferMapping + "1" + name, numElements, sizeof(uint32_t), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredPixelID + name, numElements, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(buffers, kPredThroughput + name, numElements, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));


    isChannelMapped.resize(5);
    if(!mpPredictionResults)
        mpPredictionResults = Falcor::PortBuffer::createStructured(3 * 4, numElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));

    return true;
}

void Falcor::NeuralRequestData::requestNumRequests(Falcor::RenderContext* pRenderContext, const Falcor::PortRenderData& renderData) {
    if (mpAtomicCounterCPURead == nullptr)
        mpAtomicCounterCPURead = PortBuffer::create(2 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);

    pRenderContext->copyBufferRegion(mpAtomicCounterCPURead.get(), 0, renderData[kPredBuffer + "4" + name]->asBuffer()->getUAVCounter().get(), 0, 4);
    pRenderContext->copyBufferRegion(mpAtomicCounterCPURead.get(), 4, renderData[kPredBuffer + "1" + name]->asBuffer()->getUAVCounter().get(), 0, 4);
}


void Falcor::NeuralRequestData::updateNumActiveElements() {
    uint32_t* uavCounters = (uint32_t*)mpAtomicCounterCPURead->map();
    numElementsActive = uavCounters[0];
    numUniqueElementsActive = uavCounters[1];
    mpAtomicCounterCPURead->unmap();
}

bool Falcor::NeuralRequestData::bindBuffers(Falcor::RenderContext* pRenderContext, Falcor::ref<Falcor::RtProgramVars> vars, const Falcor::PortRenderData& renderData, Falcor::uint2 resolution) {
    if (!bInit)
        return false;

    if (mpRadianceMerge == nullptr) {
        mpRadianceMerge = PortTexture::create2D(resolution.x, resolution.y, ResourceFormat::RGBA16Float, 1, 1, nullptr, ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource);
        res = resolution;
    }

    vars->getRootVar()[kPredBuffer+"0"] = renderData[kPredBuffer + "0" + name]->asBuffer();
    vars->getRootVar()[kPredBuffer + "1"] = renderData[kPredBuffer + "1" + name]->asBuffer();
    vars->getRootVar()[kPredBuffer + "2"] = renderData[kPredBuffer + "2" + name]->asBuffer();
    vars->getRootVar()[kPredBuffer + "3"] = renderData[kPredBuffer + "3" + name]->asBuffer();
    vars->getRootVar()[kPredBuffer + "4"] = renderData[kPredBuffer + "4" + name]->asBuffer();


    if (isMapped()) {
        isChannelMapped[0] = true;
        isChannelMapped[1] = true;
        isChannelMapped[2] = true;
        isChannelMapped[3] = true;
        isChannelMapped[4] = false;
    }

    vars->getRootVar()[kPredBufferMapping+"1"] = renderData[kPredBufferMapping + "1" + name]->asBuffer();


    vars->getRootVar()[kPredPixelID] = renderData[kPredPixelID + name]->asBuffer();
    vars->getRootVar()[kPredThroughput] = renderData[kPredThroughput + name]->asBuffer();

    pRenderContext->clearUAVCounter(renderData[kPredBuffer + "4"+name]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kPredBuffer + "1"+name]->asBuffer(), 0);
    return true;
}

Falcor::RenderPassReflection Falcor::DummyNeuralNetwork::reflect(const CompileData& compileData)
{
    auto reflector = LegacyPathTracer::reflect(compileData);
    reflector.addInternal(kExpectedValuesTmp, "Expected Values Tmp").texture2D().format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource);
    reflector.addOutput(kColorOutputFinal, "Output FrameBuffer Predicted by Neural Network").format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kColor2MomentOutput, "Output 2nd Moment").format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kColorOutputDebug, "Output FrameBuffer Predicted by Neural Network").texture2D(DEBUG_RES.x, DEBUG_RES.y).format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addOutput(kColorOutputDebugGT, "Output FrameBuffer Predicted by Neural Network").texture2D(DEBUG_RES.x, DEBUG_RES.y).format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::UnorderedAccess);
    reflector.addInput(kOurEstimations, "Optional diagnostic image").flags(RenderPassReflection::Field::Flags::Optional).texture2D().format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::ShaderResource);
    reflector.addInput(kGTEstimations, "Optional diagnostic image").flags(RenderPassReflection::Field::Flags::Optional).texture2D().format(ResourceFormat::RGBA32Float).bindFlags(Falcor::ResourceBindFlags::ShaderResource);
    return reflector;
}

void Falcor::DummyNeuralNetwork::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    LegacyPathTracer::compile(pRenderContext, compileData);
    const auto dimensions = compileData.defaultTexDims;
    if (any(mScreenDim != dimensions) && !mInternalBuffers.empty())
    {
        pRenderContext->submit(true);
        CUDA_CHECK_THROW(cudaDeviceSynchronize());
        bResetModel = true;
        estimationsBuffer = nullptr;
        ourestimationsBuffer = nullptr;
        YTrainingBuffer = nullptr;
        YTrainingEstimatorNNBuffer = nullptr;
    }
    if (any(mScreenDim != dimensions))
    {
        mScreenDim = dimensions;
        const uint32_t pixels = mScreenDim.x * mScreenDim.y;
        mainRenderData = NeuralRequestData("main", 3, (pixels * MAX_SAMPLES + 255) / 256 * 256, (pixels * 3 + 255) / 256 * 256, true);
        mInternalBuffers.clear();
    }
    uint32_t dims = (mScreenDim.x / 8) * (mScreenDim.y / 4) * 2;

    uint32_t num_bounces = 1;

    uint32_t numPredElementsNotPadded = mScreenDim.x * mScreenDim.y;
    numPredElementsNotPadded = (numPredElementsNotPadded + numPredElementsNotPadded / (8 * 4));

    uint32_t resolutionPerBounds = mScreenDim.x * mScreenDim.y * 4;
    resolutionPerBounds = (resolutionPerBounds + 255) / 256 * 256;

    uint32_t resolutionPerBoundsBase = mScreenDim.x * mScreenDim.y * 2;
    resolutionPerBoundsBase = (resolutionPerBoundsBase + 255) / 256 * 256;

    mainRenderData.allocateBuffers(mInternalBuffers);
    debugRenderData.allocateBuffers(mInternalBuffers);


    uint32_t selfLearningDims = (dims + 255) / 256 * 256;



    //portStructuredBuffer(mInternalBuffers, kFilterNetPredBuffer + "0", dims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kFilterNetPredBuffer + "1", dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kFilterNetPredBuffer + "2", dims, sizeof(float), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kFilterNetPredBuffer + "3", dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kFilterNetPredBuffer + "4", dims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    //portStructuredBuffer(mInternalBuffers, kFilterNetPredThroughput, dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    const uint32_t max_num_loss_variance_samples = 0;
    const uint32_t num_training_number_multiplier = 1 + max_num_loss_variance_samples;

    portStructuredBuffer(mInternalBuffers, kXTrainBuffer + "0", dims*num_training_number_multiplier, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kXTrainBuffer + "1", dims*num_training_number_multiplier, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kXTrainBuffer + "2", dims*num_training_number_multiplier, sizeof(float), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kXTrainBuffer + "3", dims * num_training_number_multiplier, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kXTrainBuffer + "4", dims * num_training_number_multiplier, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    portStructuredBuffer(mInternalBuffers, kTrainPixel, dims, sizeof(uint2), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kLThpSelfTrainingBuffer, dims, sizeof(LThp), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kXTrainAdditionalBuffer, dims * num_training_number_multiplier, sizeof(TrainingAdditionalData), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));


#if 1
    dims = 1;
    selfLearningDims = 1;
#endif
    portStructuredBuffer(mInternalBuffers, kQLearningPredBuffer + "0", selfLearningDims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kQLearningPredBuffer + "1", selfLearningDims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kQLearningPredBuffer + "2", selfLearningDims, sizeof(float), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kQLearningPredBufferMapping + "1", selfLearningDims, sizeof(uint32_t), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kQLearningPredBuffer + "3", selfLearningDims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kQLearningPredBuffer + "4", selfLearningDims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    portStructuredBuffer(mInternalBuffers, kQLearningMapping, selfLearningDims, sizeof(uint32_t), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kQLearningThroughput, selfLearningDims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    portStructuredBuffer(mInternalBuffers, kNetworkEstimatorPredBuffer + "0", dims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kNetworkEstimatorPredBuffer + "1", dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kNetworkEstimatorPredBuffer + "2", dims, sizeof(float), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kNetworkEstimatorPredBuffer + "3", dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kNetworkEstimatorPredBuffer + "4", dims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    portStructuredBuffer(mInternalBuffers, kNetworkEstimatorThroughput, dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    portStructuredBuffer(mInternalBuffers, kEstimatorNNXTrainAdditionalBuffer, dims , sizeof(TrainingAdditionalData), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kEstimatorNNTrainBuffer+ "0", dims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kEstimatorNNTrainBuffer + "1", dims , sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kEstimatorNNTrainBuffer + "2", dims , sizeof(float), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kEstimatorNNTrainBuffer + "3", dims, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kEstimatorNNTrainBuffer + "4", dims, sizeof(uint), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));

    //portStructuredBuffer(mInternalBuffers, kXTrainResError, dims, sizeof(int), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kResErrorID, dims, sizeof(float4), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));


    portStructuredBuffer(mInternalBuffers, kLossVarianceScatterIDBuffer, dims* num_training_number_multiplier, sizeof(uint32_t), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    portStructuredBuffer(mInternalBuffers, kLossVarianceThroughputBuffer, dims* num_training_number_multiplier, sizeof(float3), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    uint s = (m_LOD == 1) ? 1 : numPredElementsNotPadded * (m_LOD - 1);
    //portStructuredBuffer(mInternalBuffers, kNNMapPredBufferKeys, 4, sizeof(int), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kNNMapPredBufferValues, 4, sizeof(int), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));*/
    //portStructuredBuffer(mInternalBuffers, kNNMapTrainBufferKeys, dims * m_LOD, sizeof(int), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));
    //portStructuredBuffer(mInternalBuffers, kNNMapTrainBufferValues, dims * m_LOD, sizeof(int), (Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource));*/




}

void Falcor::DummyNeuralNetwork::setTracerData(const Falcor::PortRenderData& renderData)
{
    auto pBlock = mTracer.pParameterBlock->getRootVar();

    // Upload parameters struct.
    pBlock["params"].setBlob(mSharedParams);

    // Bind emissive light sampler.
    if (mUseEmissiveSampler)
    {
        assert(mpEmissiveSampler);
        bool success = (mpEmissiveSampler->bindShaderData(pBlock["emissiveSampler"]), true);
        if (!success) throw std::exception("Failed to bind emissive light sampler");
    }
}

void Falcor::DummyNeuralNetwork::setScene(Falcor::RenderContext* pRenderContext, const Falcor::ref<Falcor::Scene>& pScene)
{
    LegacyPathTracer::setScene(pRenderContext, pScene);

    mpScene = pScene;

    if (mpScene)
    {
        if (mpScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("This render pass only supports triangles. Other types of geometry will be ignored.");
        }

        // Create ray tracing program.
        auto sceneDefines = mpScene->getSceneDefines();
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(kMaxAttributeSizeBytes);
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);
        
        sceneDefines.add("MAX_BOUNCES", std::to_string(mSharedParams.maxBounces));
        sceneDefines.add("SAMPLES_PER_PIXEL", std::to_string(mSharedParams.samplesPerPixel));

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(kRayTypeScatter, desc.addMiss("scatterMiss"));
        sbt->setMiss(kRayTypeShadow, desc.addMiss("shadowMiss"));
        sbt->setHitGroup(kRayTypeScatter, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("scatterClosestHit", "scatterAnyHit"));
        sbt->setHitGroup(kRayTypeShadow, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowAnyHit"));

        sceneDefines.add("CONTINUE_RAY", "0");
        mTracer.pProgram = Program::create(portDevice(), desc, sceneDefines);
    }
}

Falcor::NeuralInputDataFalcor setupNeuralInput(const Falcor::PortRenderData& renderData, const std::string& predBufferName, uint32_t numPredElements, const std::string& predBufferMappingName = "", uint32_t numPredMappingElements = 0, std::string postFix = "", const std::vector<bool>* isMappedChannel=nullptr) {
    Falcor::NeuralInputDataFalcor neuralInputData;

    for (int i = 0; i < 5; i++) {
        const Falcor::ref<Falcor::Resource>& res = renderData.getResource(predBufferName + std::to_string(i) + postFix);
        if (res == nullptr)
            break;

        bool isForcedMapped = false;
        if (isMappedChannel) {
            isForcedMapped = (*isMappedChannel)[i];
        }

        if ((isMappedChannel && !isForcedMapped) || (!isMappedChannel && (predBufferMappingName.empty())) || numPredMappingElements == 0) {
            neuralInputData.addInput(res->asBuffer(), nullptr, numPredElements);
        }
        else {
            const Falcor::ref<Falcor::Resource>& resMap = renderData.getResource(predBufferMappingName + "1" + postFix);
            uint32_t numElements = (resMap == nullptr) ? numPredElements : numPredMappingElements;
            neuralInputData.addInput(res->asBuffer(), (resMap == nullptr) ? nullptr : resMap->asBuffer(), numElements);
        }
    }

    return neuralInputData;
}


#define SYNC_EVERYTHING 0
void Falcor::DummyNeuralNetwork::execute(Falcor::RenderContext* pRenderContext, const Falcor::RenderData& graphData)
{
    const PortRenderData renderData{graphData, mInternalBuffers};

    output = renderData.getTexture(kColorOutputFinal);

    if (!mpScene)
    {
        return;
    }

    // Call shared pre-render code.
    if (!beginFrame(pRenderContext, renderData)) return;


    // Set compile-time constants.
    Falcor::ref<Falcor::Program> pProgram = mTracer.pProgram;

    /*if (bNRCEnabled)
        mSharedParams.samplesPerPixel = 1;
    else
        mSharedParams.samplesPerPixel = mNumGTSamples;*/


    bool bPrevNEE = mSharedParams.useNEE;
    if (bNRCEnabled) {
        /*if (!bDirectLightCache)
            mSharedParams.useNEE = bPrevNEE;
        else {
            mSharedParams.useNEE = bDirectLightCacheNEE;
        }*/
    }
    else {
        mSharedParams.useNEE = bPrevNEE;
    }

    setStaticParams(pProgram.get());


    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    pProgram->addDefines(getValidResourceDefines(mInputChannels, renderData));
    pProgram->addDefines(getValidResourceDefines(mOutputChannels, renderData));

    if (mUseEmissiveSampler)
    {
        // Specialize program for the current emissive light sampler options.
        assert(mpEmissiveSampler);
        if (pProgram->addDefines(mpEmissiveSampler->getDefines())) mTracer.pVars = nullptr;
    }


    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars) prepareVars();
    assert(mTracer.pVars);


    // Set shared data into parameter block.
    setTracerData(renderData);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            auto var = mTracer.pVars->getRootVar();
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : mInputChannels) bind(channel);
    for (auto channel : mOutputChannels) bind(channel);

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    assert(targetDim.x > 0 && targetDim.y > 0);
    mScreenDim = targetDim;


    const std::string buffersNames[] = {
        kXTrainBuffer + "0",
        kXTrainBuffer + "1",
        kXTrainBuffer + "2",
        kXTrainBuffer + "3",
        kXTrainBuffer + "4",
        /*kXTrainResError,
        kResErrorID,*/
        kLThpSelfTrainingBuffer,
        kXTrainAdditionalBuffer,
        kLossVarianceScatterIDBuffer,
        kLossVarianceThroughputBuffer,
        kEstimatorNNTrainBuffer + "0",
        kEstimatorNNTrainBuffer + "1",
        kEstimatorNNTrainBuffer + "2",
        kEstimatorNNTrainBuffer + "3",
        kEstimatorNNTrainBuffer + "4",
        kEstimatorNNXTrainAdditionalBuffer,
        kTrainPixel,
    };

    for (auto buffChannel : buffersNames)
    {
        mTracer.pVars->getRootVar()[buffChannel] = renderData[buffChannel]->asBuffer();
    }
    pRenderContext->clearUAVCounter(renderData[kLossVarianceScatterIDBuffer]->asBuffer(), 0);

    mainRenderData.bindBuffers(pRenderContext, mTracer.pVars, renderData, targetDim);
    mTracer.pVars->getRootVar()["gOutputColor"] = renderData.getTexture(kColorOutputFinal);
    mTracer.pVars->getRootVar()[kGTEstimations] = renderData.getTexture(kGTEstimations);
    mTracer.pVars->getRootVar()[kOurEstimations] = renderData.getTexture(kOurEstimations);
    mTracer.pVars->getRootVar()[kExpectedValuesTmp] = renderData.getTexture(kExpectedValuesTmp);
    mTracer.pVars->getRootVar()["gOutputM2"] = renderData.getTexture(kColor2MomentOutput);


    pRenderContext->clearUAVCounter(renderData[kLThpSelfTrainingBuffer]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kXTrainBuffer + "4"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kEstimatorNNTrainBuffer + "4"]->asBuffer(), 0);

    pRenderContext->clearUAVCounter(renderData[kQLearningPredBuffer + "4"]->asBuffer(), 0);
    pRenderContext->clearUAVCounter(renderData[kQLearningPredBuffer + "1"]->asBuffer(), 0);

    pRenderContext->clearUAVCounter(renderData[kNetworkEstimatorPredBuffer + "4"]->asBuffer(), 0);

    mpPixelDebug->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    mpPixelStats->prepareProgram(pProgram, mTracer.pVars->getRootVar());
    pProgram->addDefine("RENDER_ONLY_INDIRECT", std::to_string(bOnlyIndirect));
    pProgram->addDefine("HEMISPHERICAL_RENDER", "0");
    pProgram->addDefine("RES_VARIANCE_LOSS", std::to_string(bResidualVarianceLoss));
    pProgram->addDefine("DEBUG_RENDER", std::to_string(bNNEstimatorDebugRender));
    pProgram->addDefine("ENABLE_NRC", std::to_string(bNRCEnabled));
    pProgram->addDefine("CONTINUE_RAY", "0");
    pProgram->addDefine("DEBUG_VIEW", std::to_string(bNRCDebugMode));
    pProgram->addDefine("EVAL_DIRECT_ONLY_ONCE", std::to_string(bEvalDirectOnlyOnce));


    bool bJustStarted = false;
    if (!model || bResetModel || model->getTrainingEpoch() < 2) {
        bJustStarted = true;
    }

    if (bNRCEnabled || 1) {
        pProgram->addDefine("GRADIENT_DENOISING", std::to_string(bGradientDenoising));
        pProgram->addDefine("MLOD", std::to_string(m_LOD));
        pProgram->addDefine("STOP_BY_METRIC", std::to_string(bStopByMetric));
        pProgram->addDefine("STOP_BY_OUR_METRIC", std::to_string(bStopByOurMetric));
        pProgram->addDefine("ERROR_BASED_METRIC", std::to_string(bErrorBased));
        pProgram->addDefine("DO_ERROR_TEST", std::to_string(bErrorTest));
        pProgram->addDefine("BIASED", std::to_string(bNRCBiased));
        pProgram->addDefine("INCIDENT_NRC", std::to_string(bIncidentNRC));
        pProgram->addDefine("TRAIN_DIRECT", std::to_string(bDirectLightCache));
        pProgram->addDefine("RESIDUAL_ERROR_ESTIMATION", std::to_string(bResErrorForLoss));
        pProgram->addDefine("VARIANCE_LOSS_MODE", std::to_string(bVarianceLoss));
        pProgram->addDefine("NUM_VARIANCE_LOSS_SAMPLES", std::to_string(mNumVarianceLossSamples));
        pProgram->addDefine("ESTIMATOR_NETWORK", std::to_string(bUseNNAsEstimator));
        pProgram->addDefine("SUPPORT_NEE", std::to_string(bDirectLightCacheNEE && bDirectLightCache));
        pProgram->addDefine("USE_GUIDANCE_OF_ENV_MAP", std::to_string(bNIRCBasedOnEnvMap));
        pProgram->addDefine("FILTERING_NET", std::to_string(bFilteringNetwork));
    }

    mTracer.pVars->getRootVar()["GeneralData"]["bbStart"] = mpScene->getSceneBounds().minPoint;
    mTracer.pVars->getRootVar()["GeneralData"]["bbEnd"] = mpScene->getSceneBounds().maxPoint;
    mTracer.pVars->getRootVar()["GeneralData"]["original_dim"] = targetDim;
    if (bNRCEnabled)
        mTracer.pVars->getRootVar()["GeneralData"]["samples_per_pixel"] = 1;
    else
        mTracer.pVars->getRootVar()["GeneralData"]["samples_per_pixel"] = mNumGTSamples;
    mTracer.pVars->getRootVar()["GeneralData"]["max_bounce_with_cv"] = maxBounceWithCV;
    mTracer.pVars->getRootVar()["GeneralData"]["gd_hash_grid_res"] = gradientHashGridRes;
    mTracer.pVars->getRootVar()["GeneralData"]["gd_hash_grid_res"] = gradientHashGridRes;
    mTracer.pVars->getRootVar()["GeneralData"]["g_num_samples"][size_t(0)] = (bJustStarted) ? 1 : mNumNRCSamplesPerDepth[0];
    mTracer.pVars->getRootVar()["GeneralData"]["g_num_samples"][size_t(1)] = (bJustStarted) ? 0 : mNumNRCSamplesPerDepth[1];
    mTracer.pVars->getRootVar()["GeneralData"]["g_num_samples"][size_t(2)] = (bJustStarted) ? 0 : mNumNRCSamplesPerDepth[2];
    mTracer.pVars->getRootVar()["GeneralData"]["roughnessThreshold"] = roughnessThreshold;
    mTracer.pVars->getRootVar()["GeneralData"]["gRoulleteProb"] = roulleteProb;
    mTracer.pVars->getRootVar()["GeneralData"]["expected_value_num_samples"] = mNumSamplesForExpectedValues;
    mTracer.pVars->getRootVar()["GeneralData"]["num_training_id"] = (model) ? getTrainingStep() : 0;
    mTracer.pVars->getRootVar()["GeneralData"]["errorThreshold"] = errorThreashold;
    if(bDebugHemispherical)
        mTracer.pVars->getRootVar()["GeneralData"]["debug_pixel_id"] = mSelectedPixel;
    else
        mTracer.pVars->getRootVar()["GeneralData"]["debug_pixel_id"] = int2(-1, -1);

    bool prevBRDFSampling = mSharedParams.useBRDFSampling;

    for (uint32_t i = 0; i < lodSizes.size(); i++)
    {
        mTracer.pVars->getRootVar()["GeneralData"]["lodSize" + std::to_string(i)] = lodSizes[i];
    }
    // Spawn the rays.
    {
        FALCOR_PROFILE(pRenderContext, "MegakernelPathTracer::execute()_RayTrace - Predicting");
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));
    }


    if (!bNRCEnabled) {
        endFrame(pRenderContext, renderData);
        return;
    }

    if (bDebugHemispherical) {
        FALCOR_PROFILE(pRenderContext, "MegakernelPathTracer::execute()_RayTrace - Debug Hemisphere");

        pProgram->addDefine("HEMISPHERICAL_RENDER", "1");
        pProgram->addDefine("NUM_NRC_SAMPLES", std::to_string(1));
        mTracer.pVars->getRootVar()["GeneralData"]["debug_pixel_id"] = mSelectedPixel;
        mTracer.pVars->getRootVar()["GeneralData"]["samples_per_pixel"] = 1;
        mTracer.pVars->getRootVar()["gOutputColor"] = renderData.getTexture(kColorOutputDebugGT);

        debugRenderData.bindBuffers(pRenderContext, mTracer.pVars, renderData, DEBUG_RES);

        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(DEBUG_RES.x, DEBUG_RES.y, 1));

        pProgram->addDefine("HEMISPHERICAL_RENDER", "0");
    }


    uint32_t maxTrainingRays = 0;

    // Spawn the rays.
    if (bDispatchTrainingRays)
    {
        mTracer.pVars->getRootVar()["GeneralData"]["debug_pixel_id"] = int2(-1, -1);

        uint2 train_dim_window;
        if (targetDim.x == 1920) {
            train_dim_window = uint2(8, 8);
        }
        else {
            train_dim_window = uint2(8, 4);
        }

        if (bDirectLightCache && bDirectLightCacheNEE) {
            train_dim_window.y *= 2;
        }

        mTracer.pVars->getRootVar()["GeneralData"]["samples_per_pixel"] = 1;
        mTracer.pVars->getRootVar()["GeneralData"]["train_dim_window"] = train_dim_window;
        setTracerData(renderData);
        pProgram->addDefine("CONTINUE_RAY", "1");
        pProgram->addDefine("Q_LEARNING", std::to_string(bQLearning || bFilteringNetwork));
        pProgram->addDefine("NUM_NRC_SAMPLES", std::to_string(mNumQLearningSamples));


        if (bVarianceLoss || bResidualVarianceLoss)
        {
            mTracer.pVars->getRootVar()["gPredBuffer0"] = renderData[kNetworkEstimatorPredBuffer + "0"]->asBuffer();
            mTracer.pVars->getRootVar()["gPredBuffer1"] = renderData[kNetworkEstimatorPredBuffer + "1"]->asBuffer();
            mTracer.pVars->getRootVar()["gPredBuffer2"] = renderData[kNetworkEstimatorPredBuffer + "2"]->asBuffer();
            mTracer.pVars->getRootVar()["gPredBuffer3"] = renderData[kNetworkEstimatorPredBuffer + "3"]->asBuffer();
            mTracer.pVars->getRootVar()["gPredBuffer4"] = renderData[kNetworkEstimatorPredBuffer + "4"]->asBuffer();
            mTracer.pVars->getRootVar()[kPredThroughput] = renderData[kNetworkEstimatorThroughput]->asBuffer();
        }
        else {


                mTracer.pVars->getRootVar()["gPredBuffer0"] = renderData[kQLearningPredBuffer + "0"]->asBuffer();
                mTracer.pVars->getRootVar()["gPredBuffer1"] = renderData[kQLearningPredBuffer + "1"]->asBuffer();
                mTracer.pVars->getRootVar()["gPredBuffer2"] = renderData[kQLearningPredBuffer + "2"]->asBuffer();
                mTracer.pVars->getRootVar()["gPredBufferMapping1"] = renderData[kQLearningPredBufferMapping + "1"]->asBuffer();
                mTracer.pVars->getRootVar()["gPredBuffer3"] = renderData[kQLearningPredBuffer + "3"]->asBuffer();
                mTracer.pVars->getRootVar()["gPredBuffer4"] = renderData[kQLearningPredBuffer + "4"]->asBuffer();
                mTracer.pVars->getRootVar()[kPredPixelID] = renderData[kQLearningMapping]->asBuffer();
                mTracer.pVars->getRootVar()[kPredThroughput] = renderData[kQLearningThroughput]->asBuffer();


        }

        FALCOR_PROFILE(pRenderContext, "MegakernelPathTracer::execute()_RayTrace - Training");

        maxTrainingRays = (targetDim.x / train_dim_window.x) * (targetDim.y / train_dim_window.y);
        mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim.x/train_dim_window.x, targetDim.y/train_dim_window.y, 1));
    }

    mSharedParams.useNEE = bPrevNEE;

    //// Call shared post-render code.
    //endFrame(pRenderContext, renderData);

    //if (!mpScene)
    //{
    //    return;
    //}

    if (!mpFenceForPredictionData) mpFenceForPredictionData = portDevice()->createFence(true);
    if (!mpFenceOut) mpFenceOut = portDevice()->createFence(true);
    if (!mpFenceOutDebug) mpFenceOutDebug = portDevice()->createFence(true);
    if (!mpFenceForOurEstimations) mpFenceForOurEstimations = portDevice()->createFence(true);

    // Wait getting data for predicting

    if (!mpCounterTrainBuffer) {
        mpCounterTrainBuffer = PortBuffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);
        mpCounterEstimatorNNTrainBuffer = PortBuffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);
        mpCounterQLearningPredBuffer = PortBuffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);
        mpCounterQLearningPredPosBuffer = PortBuffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);
        mpCounterNetworkEstimator = PortBuffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);
        mpCounterLThpSelfTrainingBuffer = PortBuffer::create(1 * sizeof(uint32_t), Falcor::ResourceBindFlags::None, MemoryType::ReadBack);
    }

    Falcor::ref<Falcor::Buffer> xTrainBuff = renderData[kXTrainBuffer + "4"]->asBuffer();
    Falcor::ref<Falcor::Buffer> xEstimatorBuff = renderData[kEstimatorNNTrainBuffer + "4"]->asBuffer();

    Falcor::ref<Falcor::Buffer> xLThpSelfTrainBuff = renderData[kLThpSelfTrainingBuffer]->asBuffer();
    Falcor::ref<Falcor::Buffer> xXTrainAddBuff = renderData[kXTrainAdditionalBuffer]->asBuffer();

    Falcor::ref<Falcor::Buffer> xXTrainAddBuffEstimatorNN = renderData[kEstimatorNNXTrainAdditionalBuffer]->asBuffer();

    Falcor::ref<Falcor::Buffer> xPredPosBuff = renderData[kPredBuffer + "1"]->asBuffer();
    Falcor::ref<Falcor::Buffer> xPredBuff = renderData[kPredBuffer + "4"]->asBuffer();


    Falcor::ref<Falcor::Buffer> xQLearningPredBuffer = renderData[kQLearningPredBuffer + "4"]->asBuffer();
    Falcor::ref<Falcor::Buffer> xQLearningPredPosBuffer = renderData[kQLearningPredBuffer + "1"]->asBuffer();

    Falcor::ref<Falcor::Buffer> xNetworkEstimatorPredBuffer = renderData[kNetworkEstimatorPredBuffer+ "4"]->asBuffer();



    pRenderContext->copyBufferRegion(mpCounterQLearningPredPosBuffer.get(), 0, xQLearningPredPosBuffer->getUAVCounter().get(), 0, 4);
    pRenderContext->copyBufferRegion(mpCounterQLearningPredBuffer.get(), 0, xQLearningPredBuffer->getUAVCounter().get(), 0, 4);

    uint64_t signalValue;
    uint32_t NumTrainingElements = 0;
    uint32_t NumQLearningPredictPositions = 0;
    uint32_t NumQLearningPredictElements = 0;
    uint32_t NumNetworkEstimatorRequests = 0;
    uint32_t NumTrainingBuffersForNNEstimator = 0;


    bool bSync = mSync;
    if (bSync) {
#if 1
        pRenderContext->copyBufferRegion(mpCounterTrainBuffer.get(), 0, xTrainBuff->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mpCounterNetworkEstimator.get(), 0, xNetworkEstimatorPredBuffer->getUAVCounter().get(), 0, 4);
        pRenderContext->copyBufferRegion(mpCounterEstimatorNNTrainBuffer.get(), 0, xEstimatorBuff->getUAVCounter().get(), 0, 4);

        mainRenderData.requestNumRequests(pRenderContext, renderData);
        if (bDebugHemispherical)
            debugRenderData.requestNumRequests(pRenderContext, renderData);
#endif

        //pRenderContext->flush(false || SYNC_EVERYTHING);
        pRenderContext->submit(true);

        signalValue = pRenderContext->signal(mpFenceForPredictionData.get());
        mpFenceForPredictionData->wait();


#if 1

        uint32_t* uavCounters = (uint32_t*)mpCounterTrainBuffer->map();
        NumTrainingElements = uavCounters[0];
        mpCounterTrainBuffer->unmap();

        mainRenderData.updateNumActiveElements();

        if (bDebugHemispherical)
            debugRenderData.updateNumActiveElements();

        uint32_t* uavCounters4 = (uint32_t*)mpCounterQLearningPredBuffer->map();
        NumQLearningPredictElements = uavCounters4[0];
        mpCounterQLearningPredBuffer->unmap();


        uint32_t* uavCounters5 = (uint32_t*)mpCounterQLearningPredPosBuffer->map();
        NumQLearningPredictPositions = uavCounters5[0];
        mpCounterQLearningPredPosBuffer->unmap();

        uint32_t* uavCounters6 = (uint32_t*)mpCounterNetworkEstimator->map();
        NumNetworkEstimatorRequests = uavCounters6[0];
        mpCounterNetworkEstimator->unmap();

        uint32_t* uavCounters7 = (uint32_t*)mpCounterEstimatorNNTrainBuffer->map();
        NumTrainingBuffersForNNEstimator = uavCounters7[0];
        mpCounterEstimatorNNTrainBuffer->unmap();

#endif
    }

    uint32_t w = mScreenDim.x;
    uint32_t h = mScreenDim.y;
    uint32_t nElements = w * h * MAX_SAMPLES;
    uint32_t nrcElements = w * h / (8 * 4);


    if (!model || bResetModel)
    {
#if 1
        // Note: Path removed for privacy - using dynamic path resolution instead
        std::string dummy_path = std::string();
        dummy_path.clear();
#else
        const std::string dummy_path = "";
#endif
        std::string p = dummy_path;
        if (bIncidentNRC) {
            if (bDirectLightCache && bNIRCBasedOnEnvMap)
                p += kNetworkConfigNIRCEnvMap;
            else if (mNircArchitecture == 1)
                p += mNircInterpolation == 0 ? "RenderPasses/NeuralRadianceCache/network_config_nirc_directional.json"
                    : "RenderPasses/NeuralRadianceCache/network_config_nirc_directional_tetrahedral.json";
            else if (mNircArchitecture == 2)
                p += mNircInterpolation == 0 ? "RenderPasses/NeuralRadianceCache/network_config_nirc_hybrid.json"
                    : "RenderPasses/NeuralRadianceCache/network_config_nirc_hybrid_tetrahedral.json";
            else
                p += kNetworkConfigNIRC;
        }
        else
            p += kNetworkConfigNRC;
        std::vector<std::string> configs = {
            p
        };
        std::cout << dummy_path << std::endl;
        std::cout << dummy_path + kPyramidConfig << std::endl;
        model.reset(new Falcor::NNLaplacian("Model", configs, dummy_path + kPyramidConfig, 2, w * h * MAX_SAMPLES + nrcElements, Falcor::AABB(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f)), &m_diffuse_time, bIncidentNRC && !bForceNonSeparateInput));

        //model->m_denoising_level = &m_diffuse_time;
    }

    if (!modelEstimator || bResetModel) {
#if 1
        // Note: Path removed for privacy - using dynamic path resolution instead
        std::string dummy_path = std::string();
        dummy_path.clear();
#else
        const std::string dummy_path = "";
#endif
        std::vector<std::string> configs = {
            dummy_path + kNetworkConfigEstimator
        };
        std::cout << dummy_path << std::endl;
        std::cout << dummy_path + kPyramidConfig << std::endl;
        modelEstimator.reset(new Falcor::NNLaplacian("Model Estimator", configs, dummy_path + kPyramidConfig, 2, w * h * MAX_SAMPLES + nrcElements, Falcor::AABB(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 1.0f, 1.0f)), &m_diffuse_time));
        bResetModel = false;
    }

    enableTraining(bDispatchTrainingRays);

    nrcElements = ((w * h) / (8 * 4))*2;


    if (estimationsBuffer == nullptr) {
#if 1
        nrcElements = 1;
#endif
        estimationsBuffer = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        ourestimationsBuffer = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorVarianceLoss = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorVarianceLossTmp = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorVarianceLossPart = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorQLearningBufferTmp = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        predColorQLearningBuffer = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        //predColorResError = Falcor::PortBuffer::createStructured(3 * 4, nrcElements, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
    }

    // Wait prediction results

    if (NumTrainingElements > 0 && model->getTrainFlag()) {
        FALCOR_PROFILE(pRenderContext, "NeuralNetwork Train Sorting IDs");
    }

    auto inferencePredict = [&](std::shared_ptr<Falcor::NNLaplacian> network, NeuralRequestData& reqdata, Falcor::ref<Falcor::Texture> renderFrame, uint64_t inputSignalValue, Falcor::ref<Falcor::Fence> fenIn = nullptr, Falcor::ref<Falcor::Fence> fenOut = nullptr, bool bCancelMapping=false, bool bFusedOutput=false) {
        if (reqdata.numElementsActive > 0)
        {
            pRenderContext->clearTexture(reqdata.mpRadianceMerge.get());

            FALCOR_PROFILE(pRenderContext, "NeuralNetwork Predict Start");
            // Just prediction

            if (0 && bNIRCBasedOnEnvMap && bIncidentNRC && bDirectLightCache) {
                FALCOR_PROFILE(pRenderContext, "EnvMap Requests");

                mpEnvMapRequests->getRootVar()["PerFrameCB"]["gElems"] = reqdata.numElementsActive;
                mpScene->getEnvMap()->bindShaderData(mpEnvMapRequests->getRootVar()["gEnv"]);
                mpEnvMapRequests->getRootVar()["gDirections"] = renderData[kPredBuffer + "4" + reqdata.name]->asBuffer();
                mpEnvMapRequests->getRootVar()["gThroughput"] = renderData[kPredThroughput + reqdata.name]->asBuffer();
                mpEnvMapRequests->execute(pRenderContext, Falcor::uint3(reqdata.numElementsActive, uint32_t(1), uint32_t(1)));
            }

            NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kPredBuffer, reqdata.numElementsActive, kPredBufferMapping, reqdata.numUniqueElementsActive, reqdata.name, (bCancelMapping) ? nullptr : &reqdata.isChannelMapped);
            bool predictedSucessfuly = false;

            Falcor::ref<Falcor::Buffer> thp = nullptr;
            Falcor::ref<Falcor::Buffer> pixel_mapping = nullptr;
            if (bFusedOutput) {
                thp = renderData.getResource(kPredThroughput + reqdata.name)->asBuffer();
                pixel_mapping = renderData.getResource(kPredPixelID + reqdata.name)->asBuffer();
            }

            predictedSucessfuly = network->predict(reqdata.numElementsActive, neuralInputData, reqdata.mpPredictionResults, (bSync) ? fenIn : nullptr, inputSignalValue, fenOut, thp, pixel_mapping, false, reqdata.res.x * reqdata.res.y);

            return predictedSucessfuly;
        }

        return false;
    };

    auto inferencePredictMerge = [&](NeuralRequestData& reqdata, Falcor::ref<Falcor::Texture> renderFrame, uint64_t inputSignalValue, Falcor::ref<Falcor::Fence> fenIn = nullptr, Falcor::ref<Falcor::Fence> fenOut = nullptr, Falcor::ref<Falcor::Texture> secondMomentFrame = nullptr, bool bFusedOutput=false) {
        if (reqdata.numElementsActive > 0)
        {
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();


                if (fenOut && bSync)
                    pRenderContext->wait(fenOut.get());
                {
                    if (!bFusedOutput) {


                        FALCOR_PROFILE(pRenderContext, "Merge predicted CV for all bounces to a intermediate frame buffer");
                        mpConvertNNOutputPass->getRootVar()["PerFrameCB"]["gElems"] = reqdata.numElementsActive;
                        mpConvertNNOutputPass->getRootVar()["PerFrameCB"]["outScaleInv"] = float(1.0 / model->getOutputScale());
                        mpConvertNNOutputPass->addDefine("CLAMP_RAD", std::to_string(!bVarianceLoss));
#if 0
                        mpConvertNNOutputPass->addDefine("THRESHOLD", std::to_string(bNIRCBasedOnEnvMap && bDirectLightCache));
#endif
                        // Run mapping predicted data from structure buffer to final texture
                        mpConvertNNOutputPass->getRootVar()["gPredictedColor"] = reqdata.mpPredictionResults;
                        mpConvertNNOutputPass->getRootVar()[kPredPixelID] = renderData[kPredPixelID + reqdata.name]->asBuffer();
                        mpConvertNNOutputPass->getRootVar()[kPredThroughput] = renderData[kPredThroughput + reqdata.name]->asBuffer();
                        mpConvertNNOutputPass->getRootVar()[kPredictedRadianceMerged] = reqdata.mpRadianceMerge;



                        mpConvertNNOutputPass->execute(pRenderContext, Falcor::uint3(reqdata.numElementsActive, uint32_t(1), uint32_t(1)));
                    }
                    else {
                        FALCOR_PROFILE(pRenderContext, "Merge predicted CV for all bounces to a intermediate frame buffer");
                        // Run mapping predicted data from structure buffer to final texture
                        mpConvertNNOutputPassFused->getRootVar()["gPredictedColor"] = reqdata.mpPredictionResults;
                        mpConvertNNOutputPassFused->getRootVar()[kPredictedRadianceMerged] = reqdata.mpRadianceMerge;

                        mpConvertNNOutputPassFused->execute(pRenderContext, Falcor::uint3(mScreenDim.x*mScreenDim.y, uint32_t(1), uint32_t(1)));
                    }
                }
        }

        {
            FALCOR_PROFILE(pRenderContext, "Merge estimated radiance and predicted one");
            mpFinalOutputPass->getRootVar()[kPredictedRadianceMerged] = reqdata.mpRadianceMerge;
            mpFinalOutputPass->getRootVar()["colorOutput"] = renderFrame;


            if (secondMomentFrame)
                mpFinalOutputPass->getRootVar()["secondMoment"] = secondMomentFrame->asTexture();

            mpFinalOutputPass->addDefine("SECOND_MOMENT", std::to_string(bStDevEstimation && secondMomentFrame != nullptr));
            mpFinalOutputPass->execute(pRenderContext, Falcor::uint3(reqdata.res.x, reqdata.res.y, uint32_t(1)));
        }
    };



    if (bNNEstimatorDebugRender || bNRCDebugMode) {
        bool havePredictions = inferencePredict((bIncidentNRC || bFilteringNetwork) ? modelEstimator : model, mainRenderData, renderData.getTexture(kColorOutputFinal), signalValue, mpFenceForPredictionData, mpFenceOut, true, bFusedOutput && bIncidentNRC && !bJustStarted);
        if (havePredictions)
            inferencePredictMerge(mainRenderData, renderData.getTexture(kColorOutputFinal), signalValue, mpFenceForPredictionData, mpFenceOut, nullptr, bFusedOutput && bIncidentNRC && model->getTrainingEpoch() > 1);
    }
    else{
        if (bDebugHemispherical) {
            bool havePredictions = inferencePredict(model, mainRenderData, renderData.getTexture(kColorOutputFinal), signalValue, mpFenceForPredictionData, nullptr, false, bFusedOutput && bIncidentNRC && !bJustStarted);
            pRenderContext->clearTexture(renderData.getTexture(kColorOutputDebug).get());
            inferencePredict(model, debugRenderData, renderData.getTexture(kColorOutputDebug), 0, nullptr, mpFenceOut);

            if (havePredictions) {
                inferencePredictMerge(mainRenderData, renderData.getTexture(kColorOutputFinal), signalValue, mpFenceForPredictionData, mpFenceOut, renderData.getTexture(kColor2MomentOutput), bFusedOutput&& bIncidentNRC&& !bJustStarted);
                inferencePredictMerge(debugRenderData, renderData.getTexture(kColorOutputDebug), 0, nullptr, nullptr, nullptr);
            }
        }
        else {
            bool havePredictions = inferencePredict(model, mainRenderData, renderData.getTexture(kColorOutputFinal), signalValue, mpFenceForPredictionData, mpFenceOut, false, bFusedOutput && bIncidentNRC && !bJustStarted);
            if (havePredictions)
                inferencePredictMerge(mainRenderData, renderData.getTexture(kColorOutputFinal), signalValue, mpFenceForPredictionData, mpFenceOut, renderData.getTexture(kColor2MomentOutput), bFusedOutput && bIncidentNRC && !bJustStarted);
        }
    }



    if (NumTrainingElements > 0 && model->getTrainFlag() && bTonemapWeight && bFilteringNetwork && bIncidentNRC) {
        mpGatherOutEstimations->getRootVar()["PerFrameCB"]["gElems"] = maxTrainingRays;
        mpGatherOutEstimations->getRootVar()["gInput"] = renderData.getTexture(kColorOutputFinal);
        mpGatherOutEstimations->getRootVar()[kTrainPixel] = renderData[kTrainPixel]->asBuffer();
        mpGatherOutEstimations->getRootVar()["gOutput"] = ourestimationsBuffer;

        mpGatherOutEstimations->execute(pRenderContext, Falcor::uint3(maxTrainingRays, uint32_t(1), uint32_t(1)));
        pRenderContext->submit(false);

        ourestimationSignal = pRenderContext->signal(mpFenceForOurEstimations.get());

    }

    //if (NumTrainingElements > 0 && ((bVarianceLoss && bUseNNAsEstimator) || (bFilteringNetwork && bIncidentNRC) || bResidualVarianceLoss) && model->getTrainFlag()) {
    //    if (!bEventIsCreated)
    //    {
    //        CUDA_CHECK_THROW(cudaEventCreate(&eventNetworksSync));
    //        bEventIsCreated = true;
    //    }
    //
    //    NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kNetworkEstimatorPredBuffer, NumTrainingElements, "", 0);
    //    modelEstimator->predict(NumTrainingElements, neuralInputData, estimationsBuffer, mpFenceForPredictionData, signalValue, nullptr, nullptr, true);

    //    Falcor::ref<Falcor::Buffer> throughputBuffer = renderData[kNetworkEstimatorThroughput]->asBuffer();
    //    modelEstimator->transformNetworkOutput(NumTrainingElements, NumTrainingElements, estimationsBuffer, estimationsBuffer, throughputBuffer, eventNetworksSync, model->getOutputScale(), false, !bResidualVarianceLoss);
    //}


    Falcor::ref<Falcor::Buffer> qLearnPredictBuffer = nullptr;
    std::shared_ptr<Falcor::NNLaplacian> executableModel = (bFilteringNetwork) ? modelEstimator : model;
    if (NumQLearningPredictElements != 0) {
        if (!bEventIsCreated)
        {
            CUDA_CHECK_THROW(cudaEventCreate(&eventNetworksSync));
            bEventIsCreated = true;
        }


        // QLearning
        FALCOR_PROFILE(pRenderContext, "NeuralNetwork QLearning Predict");
        // Just prediction

        std::vector<bool> isChannelMapped(5);
        isChannelMapped[0] = true;
        isChannelMapped[1] = true;
        isChannelMapped[2] = true;
        isChannelMapped[3] = true;
        isChannelMapped[4] = false;

        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();
        NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kQLearningPredBuffer, NumQLearningPredictElements, kQLearningPredBufferMapping, NumQLearningPredictPositions, "", &isChannelMapped);
        executableModel->predict(NumQLearningPredictElements, neuralInputData, predColorQLearningBufferTmp, nullptr, 0, nullptr, nullptr, nullptr, true);
        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();
        Falcor::ref<Falcor::Buffer> QLearningMapping = renderData[kQLearningMapping]->asBuffer();
        Falcor::ref<Falcor::Buffer> QLearningThroughput = renderData[kQLearningThroughput]->asBuffer();

        executableModel->accumulatedRadianceForSelfLearning(NumQLearningPredictElements, nrcElements, predColorQLearningBufferTmp, predColorQLearningBuffer, QLearningMapping, QLearningThroughput, executableModel->getOutputScale());
        if(executableModel == modelEstimator)
            executableModel->pushEvent(eventNetworksSync);

        qLearnPredictBuffer = predColorQLearningBuffer;
    }



    if (NumTrainingBuffersForNNEstimator > 0 && ((bVarianceLoss) || (bFilteringNetwork && bIncidentNRC)) && modelEstimator->getTrainFlag() && model->getTrainFlag()) {
        uint32_t dims = (w / 8) * (h / 4) * 2;

        if (YTrainingEstimatorNNBuffer == nullptr) {
            YTrainingEstimatorNNBuffer = Falcor::PortBuffer::createStructured(sizeof(Falcor::YData), dims, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        }

        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();


        modelEstimator->assemblyTrainingData(NumTrainingBuffersForNNEstimator, (bSelfLearningForFilterNetwork) ? qLearnPredictBuffer : nullptr, xXTrainAddBuffEstimatorNN, xLThpSelfTrainBuff, YTrainingEstimatorNNBuffer, nullptr, nullptr, nullptr, nullptr, nullptr, modelEstimator->getOutputScale());

        NeuralInputDataFalcor neuralInputData2 = setupNeuralInput(renderData, kEstimatorNNTrainBuffer, NumTrainingBuffersForNNEstimator, "", 0);
        if (SYNC_EVERYTHING)
            cudaDeviceSynchronize();

        modelEstimator->fit(NumTrainingBuffersForNNEstimator, neuralInputData2, YTrainingEstimatorNNBuffer, nullptr, 0, !bJIT);
    }


    if (NumTrainingElements > 0 && model->getTrainFlag()) {
        uint32_t dims = (w / 8) * (h / 4) * 4;

        if (YTrainingBuffer == nullptr) {
            YTrainingBuffer = Falcor::PortBuffer::createStructured(sizeof(Falcor::YData), dims*2, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
        }


        //Falcor::ref<Falcor::Buffer> qLearnPredictBuffer = nullptr;
        //if (NumQLearningPredictElements != 0) {
        //    // QLearning
        //    FALCOR_PROFILE(pRenderContext, "NeuralNetwork QLearning Predict");
        //    // Just prediction

        //    std::vector<bool> isChannelMapped(5);
        //    isChannelMapped[0] = true;
        //    isChannelMapped[1] = true;
        //    isChannelMapped[2] = true;
        //    isChannelMapped[3] = true;
        //    isChannelMapped[4] = false;


        //    if (SYNC_EVERYTHING)
        //        cudaDeviceSynchronize();
        //    NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kQLearningPredBuffer, NumQLearningPredictElements, kQLearningPredBufferMapping, NumQLearningPredictPositions, "", &isChannelMapped);
        //    model->predict(NumQLearningPredictElements, neuralInputData, predColorQLearningBufferTmp, nullptr, 0, nullptr, nullptr, true);
        //    if (SYNC_EVERYTHING)
        //        cudaDeviceSynchronize();
        //    Falcor::ref<Falcor::Buffer> QLearningMapping = renderData[kQLearningMapping]->asBuffer();
        //    Falcor::ref<Falcor::Buffer> QLearningThroughput = renderData[kQLearningThroughput]->asBuffer();
        //    model->accumulatedRadianceForSelfLearning(NumQLearningPredictElements, nrcElements, predColorQLearningBufferTmp, predColorQLearningBuffer, QLearningMapping, QLearningThroughput, model->getOutputScale());

        //
        //    qLearnPredictBuffer = predColorQLearningBuffer;
        //}

        Falcor::ref<Falcor::Buffer> VarianceLossMapping = renderData[kLossVarianceScatterIDBuffer]->asBuffer();

        if (bVarianceLoss) {
            FALCOR_PROFILE(pRenderContext, "Predict For Variance Loss");
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kXTrainBuffer, NumTrainingElements, "", 0);
            model->predict(NumTrainingElements, neuralInputData, predColorVarianceLossTmp, nullptr, 0, nullptr, nullptr, nullptr, true);

            Falcor::ref<Falcor::Buffer> VarianceLossThroughput = renderData[kLossVarianceThroughputBuffer]->asBuffer();


            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();


            model->accumulatedRadianceForSelfLearning(NumTrainingElements, NumTrainingElements, predColorVarianceLossTmp, predColorVarianceLoss, VarianceLossMapping, VarianceLossThroughput, model->getOutputScale(), true);
            model->accumulatedRadianceOnlyPart(NumTrainingElements, NumTrainingElements, predColorVarianceLossTmp, predColorVarianceLossPart, VarianceLossMapping, VarianceLossThroughput, model->getOutputScale(), true);
        }

        if (bResidualVarianceLoss) {

            if (!bEventIsCreatedResVar)
            {
                CUDA_CHECK_THROW(cudaEventCreate(&eventNetworksSyncResVar));
                bEventIsCreatedResVar = true;
            }

            NeuralInputDataFalcor neuralInputData = setupNeuralInput(renderData, kXTrainBuffer, NumTrainingElements, "", 0);
            model->predict(NumTrainingElements, neuralInputData, predColorVarianceLossTmp, nullptr, 0, nullptr, nullptr, nullptr, true);

            Falcor::ref<Falcor::Buffer> VarianceLossThroughput = renderData[kLossVarianceThroughputBuffer]->asBuffer();


            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();


            model->transformNetworkOutput(NumTrainingElements, NumTrainingElements, predColorVarianceLossTmp, predColorVarianceLossTmp, VarianceLossThroughput, eventNetworksSyncResVar, model->getOutputScale(), false, true);

            uint32_t dims = (w / 8) * (h / 4) * 2;

            if (YTrainingEstimatorNNBuffer == nullptr) {
                YTrainingEstimatorNNBuffer = Falcor::PortBuffer::createStructured(sizeof(Falcor::YData), dims, ((Falcor::ResourceBindFlags::Shared | Falcor::ResourceBindFlags::UnorderedAccess | Falcor::ResourceBindFlags::ShaderResource)));
            }

            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            modelEstimator->syncEvent(eventNetworksSyncResVar);


            modelEstimator->assemblyTrainingData(NumTrainingElements, nullptr, xXTrainAddBuffEstimatorNN, xLThpSelfTrainBuff, YTrainingEstimatorNNBuffer, predColorVarianceLossTmp, nullptr, nullptr, nullptr, nullptr, modelEstimator->getOutputScale());

            NeuralInputDataFalcor neuralInputData2 = setupNeuralInput(renderData, kEstimatorNNTrainBuffer, NumTrainingElements, "", 0);
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            modelEstimator->fit(NumTrainingElements, neuralInputData2, YTrainingEstimatorNNBuffer, nullptr, 0, !bJIT);
        }

         if (SYNC_EVERYTHING)
             cudaDeviceSynchronize();

         if ((bUseNNAsEstimator || bFilteringNetwork || bResidualVarianceLoss ) && bEventIsCreated && executableModel == modelEstimator) {
            // We must wait for estimations from the supporting NN
            model->syncEvent(eventNetworksSync);
         }

         if (bTonemapWeight && bFilteringNetwork) {
             model->assemblyTrainingData(NumTrainingElements, (bQLearning || bFilteringNetwork) ? qLearnPredictBuffer : nullptr, xXTrainAddBuff, xLThpSelfTrainBuff, YTrainingBuffer, nullptr, (bVarianceLoss) ? predColorVarianceLoss : nullptr, (bVarianceLoss) ? VarianceLossMapping : nullptr, estimationsBuffer, ourestimationsBuffer, model->getOutputScale(), mpFenceForOurEstimations, ourestimationSignal);
         }
         else {
            model->assemblyTrainingData(NumTrainingElements, (bQLearning || bFilteringNetwork) ? qLearnPredictBuffer : nullptr, xXTrainAddBuff, xLThpSelfTrainBuff, YTrainingBuffer, nullptr, (bVarianceLoss) ? predColorVarianceLoss : nullptr, (bVarianceLoss) ? VarianceLossMapping : nullptr, ((bVarianceLoss&& bUseNNAsEstimator) || bFilteringNetwork || bResidualVarianceLoss) ? estimationsBuffer : nullptr, (bVarianceLoss) ? predColorVarianceLossPart : nullptr, model->getOutputScale());
         }



        {
            if (SYNC_EVERYTHING)
                cudaDeviceSynchronize();
            NeuralInputDataFalcor neuralInputData2 = setupNeuralInput(renderData, kXTrainBuffer, NumTrainingElements, "", 0);
            model->fit(NumTrainingElements, neuralInputData2, YTrainingBuffer, nullptr, 0, !bJIT);
        }
    }





#if 1
    m_training_epoch = model->getTrainingEpoch();
    m_training_error = model->getTrainingError();
#endif

   // mpPixelStats->setEnabled(true);
    endFrame(pRenderContext, renderData);
}


bool Falcor::DummyNeuralNetwork::onMouseEvent(const MouseEvent& mouseEvent) {
    Falcor::LegacyPathTracer::onMouseEvent(mouseEvent);

    if (bDebugHemispherical)
    {
        if (mouseEvent.type == Falcor::MouseEvent::Type::Move && !bFreezeDebugMouse)
        {
            mSelectedPixel = uint2(mouseEvent.pos * float2(mScreenDim));
            return false;
        }
    }

    return false;
}

bool Falcor::DummyNeuralNetwork::onKeyEvent(const Falcor::KeyboardEvent& keyEvent) {
    if (bDebugHemispherical) {
        if (keyEvent.type == Falcor::KeyboardEvent::Type::KeyPressed && keyEvent.key == Falcor::Input::Key::F) {
            bFreezeDebugMouse = !bFreezeDebugMouse;
            return true;
        }
    }

    return false;
}

 void Falcor::DummyNeuralNetwork::reloadModel() {
    bResetModel = true;
    mOptionsChanged = true;
}


 void Falcor::DummyNeuralNetwork::enablestats() {
    mpPixelStats->setEnabled(true);
}

void Falcor::DummyNeuralNetwork::renderUI(Falcor::Gui::Widgets& widget)
{
    static const Gui::DropdownList modes = {{0, "Path tracing reference"}, {1, "NRC - outgoing radiance"}, {2, "NIRC - incoming radiance"}};
    uint32_t mode = bNRCEnabled ? (bIncidentNRC ? 2u : 1u) : 0u;
    bool dirty = false;
    if (widget.dropdown("Mode", modes, mode))
    {
        bNRCEnabled = mode != 0;
        bIncidentNRC = mode == 2;
        bDirectLightCache = false;
        if (mode == 1) nrc_experiment_setup();
        if (mode == 2) nirc_experiment_setup(4);
        bResetModel = true;
        dirty = true;
    }
    widget.tooltip("NRC predicts outgoing light after a ray hit. NIRC predicts incoming light before tracing the ray.");
    if (bNRCEnabled)
    {
        if (bIncidentNRC) {
            static const Gui::DropdownList architectures = {{0, "Original NIRC"}, {1, "Directional hash + shared MLP"}, {2, "Directional hash + interpolated MLPs"}};
            if (widget.dropdown("Cache model", architectures, mNircArchitecture)) { bResetModel = true; dirty = true; }
            widget.tooltip("Changing the model restarts learning. Hybrid blends weights of nearby local MLPs in world space. The two-level estimator is unchanged.");
            if (mNircArchitecture != 0)
            {
                static const Gui::DropdownList interpolations = {{1, "4 corners (tetrahedral)"}, {0, "8 corners (trilinear)"}};
                if (widget.dropdown("Spatial interpolation", interpolations, mNircInterpolation)) { bResetModel = true; dirty = true; }
                widget.tooltip("Four corners halve spatial blending work in the directional grid and local MLP banks. Both modes interpolate continuously. Changing this restarts learning.");
            }
        }
        bool correction = !bNRCBiased;
        if (widget.checkbox("Two-level MC correction", correction)) { bNRCBiased = !correction; dirty = true; }
        widget.tooltip("Adds traced residual samples to correct cache error. Disabling this shows the cache approximation and can introduce bias.");
        if (auto sampling = widget.group("Sampling", false))
        {
            if (correction) sampling.text("1 traced residual path/pixel");
            if (bIncidentNRC)
            {
                for (uint32_t depth = 0; depth < 3; ++depth)
                {
                    uint32_t used = 0;
                    for (uint32_t k = 0; k < 3; ++k) if (k != depth) used += mNumNRCSamplesPerDepth[k];
                    const std::string label = "Cache @ bounce " + std::to_string(depth + 1);
                    dirty |= sampling.var(label.c_str(), mNumNRCSamplesPerDepth[depth], 0u, MAX_SAMPLES - 3u - std::min(used, MAX_SAMPLES - 3u));
                }
                sampling.tooltip("Cheap neural evaluations used to integrate incoming light at each supported bounce.");
                dirty |= sampling.var("Correct through", maxBounceWithCV, 0u, 5u);
            }
        }
        if (auto training = widget.group("Training", false))
        {
            bool enabled = bDispatchTrainingRays;
            if (training.checkbox("Train continuously", enabled)) enableTraining(enabled);
            training.tooltip("Pause to inspect a fixed cache. Rendering continues with the current learned weights.");
            if (training.button("Reset cache")) { bResetModel = true; dirty = true; }
            training.text("Training frame: " + std::to_string(model ? model->getTrainingEpoch() : 0));
            training.text("Loss: " + std::to_string(model ? model->m_loss : 0.f));
        }
    }
    if (auto path = widget.group("Path tracing", false))
    {
        if (!bNRCEnabled) dirty |= path.var("Reference spp", mNumGTSamples, 1u, 10u);
        dirty |= path.var("Maximum bounces", mSharedParams.maxBounces, 0u, 32u);
        mSharedParams.maxNonSpecularBounces = mSharedParams.maxBounces;
        dirty |= path.checkbox("Direct light sampling", mSharedParams.useNEE);
        dirty |= path.checkbox("Multiple importance sampling", mSharedParams.useMIS);
        dirty |= path.checkbox("Russian roulette", mSharedParams.useRussianRoulette);
        dirty |= path.checkbox("Indirect lighting only", bOnlyIndirect);
        dirty |= path.checkbox("Fixed random seed", mSharedParams.useFixedSeed);
    }
    if (auto debug = widget.group("Diagnostics", false))
    {
        dirty |= debug.checkbox("Show directional inspector", bDebugHemispherical);
        if (bDebugHemispherical) debug.var("Inspector pixel", mSelectedPixel);
        dirty |= debug.checkbox("Record second moment", bStDevEstimation);
        dirty |= debug.checkbox("Fused NIRC output", bFusedOutput);
        debug.tooltip("Original fused CUDA integration. Keep enabled for normal NIRC rendering.");
    }
    if (dirty) { mOptionsChanged = true; recreateVars(); }
}

int32_t Falcor::DummyNeuralNetwork::getTrainingEpoch() const {
    return m_training_epoch;
}

 float Falcor::DummyNeuralNetwork::getTrainingError() const {
    return m_training_error;
}

float Falcor::DummyNeuralNetwork::getDiffuseTime() const {
    return m_diffuse_time;
}


void Falcor::DummyNeuralNetwork::setDiffuseTime(float t){
    m_diffuse_time = t;
}

