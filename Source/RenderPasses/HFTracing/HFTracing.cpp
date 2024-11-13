/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "HFTracing.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, HFTracing>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/HFTracing/MinimalPathTracer.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" },
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";
} // namespace

HFTracing::HFTracing(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    FALCOR_ASSERT(mpSampleGenerator);
}

void HFTracing::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
            logWarning("Unknown property '{}' in HFTracing properties.", key);
    }
}

Properties HFTracing::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

RenderPassReflection HFTracing::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void HFTracing::generateGeometryMap(RenderContext* pRenderContext, const RenderData& renderData){
    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    mMaxTriCount = mpScene->getMesh(MeshID{0}).getTriangleCount();


    // Get dimensions of ray dispatch.
    uint2 targetDim = renderData.getDefaultTextureDims();
    targetDim = uint2(2048);
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);


    auto precomputeVar = mpGenerateGeometryMapPass->getRootVar();

    // set mesh data
    const auto& meshDesc = mpScene->getMesh(MeshID{ 0 });
    precomputeVar["PerFrameCB"]["vertexCount"] = meshDesc.vertexCount;
    precomputeVar["PerFrameCB"]["vbOffset"] = meshDesc.vbOffset;
    precomputeVar["PerFrameCB"]["triangleCount"] = meshDesc.getTriangleCount();
    precomputeVar["PerFrameCB"]["ibOffset"] = meshDesc.ibOffset;
    precomputeVar["PerFrameCB"]["use16BitIndices"] = meshDesc.use16BitIndices();
    precomputeVar["PerFrameCB"]["gTriID"] = mTriID;





    // mpScene->getMesh(0)->setIntoConstantBuffer(precomputeVar["PerFrameCB"]["gMeshData"]);
    precomputeVar["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    precomputeVar["PerFrameCB"]["gControlParas"] = mControlParas;
    precomputeVar["gOutputNormalMap"].setUav(mpNormalMap->getUAV());
    precomputeVar["gOutputTangentMap"].setUav(mpTangentMap->getUAV());
    precomputeVar["gOutputPosMap"].setUav(mpPosMap->getUAV());
    precomputeVar["gOutputColor"] = renderData.getTexture("color");
    mpScene->bindShaderData(precomputeVar["scene"]);

    mpGenerateGeometryMapPass->execute(pRenderContext, targetDim.x, targetDim.y);
    mTriID++;


}


void HFTracing::visualizeMaps(RenderContext* pRenderContext, const RenderData& renderData){
     // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }
    // Get dimensions of ray dispatch.
    uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    auto var = mpVisualizeMapsPass->getRootVar();
    var["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    var["PerFrameCB"]["gDisplayMipLevel"] = mCurvatureParas.w;
    var["gOutputNormalMap"].setSrv(mpNormalMap->getSRV());
    var["gOutputTangentMap"].setSrv(mpTangentMap->getSRV());
    var["gOutputColor"] = renderData.getTexture("color");

    mpVisualizeMapsPass->execute(pRenderContext, targetDim.x, targetDim.y);
    // refresh the frame
    mOptionsChanged = true;
}


void HFTracing::createMaxMip(RenderContext* pRenderContext, const RenderData& renderData){
     // If we have no scene, just clear the outputs and return.
    if(mpHF.get() == nullptr){
        return;
    }
    if(mpHFMaxMip.get() != nullptr){
        return;
    }
    int windowSize = pow(2, 5);
    int mipHeight = mpHF->getHeight()/windowSize;
    int mipWidth = mpHF->getWidth()/windowSize;
    mpHFMaxMip = mpDevice->createTexture2D(
                mipWidth,
                mipHeight,
                ResourceFormat::R32Float,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
              );
    // Get dimensions of ray dispatch.
    uint2 targetDim = uint2(mipWidth, mipHeight);
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    auto var = mpCreateMaxMipPass->getRootVar();
    var["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    var["PerFrameCB"]["gLod"] = 5;
    var["gInputHeightMap"].setSrv(mpHF->getSRV());
    var["gOutputHeightMap"].setUav(mpHFMaxMip->getUAV());

    mpCreateMaxMipPass->execute(pRenderContext, targetDim.x, targetDim.y);

}


void HFTracing::nnInferPass(RenderContext* pRenderContext, const RenderData& renderData){

    uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    auto var = mpInferPass->getRootVar();
    var["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    var["PerFrameCB"]["gApplySyn"] = mApplySyn;
    mpNBTF->bindShaderData(var["PerFrameCB"]["nbtf"]);

    var["gOutputColor"] = renderData.getTexture("color");
    var["wiWox"] = mpWiWox;
    var["uvWoyz"]= mpUVWoyz;
    var["dfDxy"] = mpDfDxy;


    mpInferPass->execute(pRenderContext, targetDim.x, targetDim.y);

}

void createTex(ref<Texture> &tex, ref<Device> device, uint2 targetDim){
    if(tex.get() == nullptr){
        tex = device->createTexture2D(
            targetDim.x,
            targetDim.y,
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
        );
    }
    else{
        if (tex.get()->getWidth() != targetDim.x || tex.get()->getHeight() != targetDim.y)
        {
            tex = device->createTexture2D(
                targetDim.x,
                targetDim.y,
                ResourceFormat::RGBA32Float,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
            );
        }
    }
}

void HFTracing::renderHF(RenderContext* pRenderContext, const RenderData& renderData){
    auto& dict = renderData.getDictionary();
    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }
    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            // lightingChanged = true;
            // mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            // lightingChanged = true;
            // mRecompile = true;
        }
    }



    auto light = mpScene->getLightByName("MyLight0");
    if (light)
    {
        if (light->getType() == LightType::Sphere)
        {
            ref<SphereLight> sl = static_ref_cast<SphereLight>(light);
            sl->setScaling(float3(1.0f * mLightZPR.w));
            float phi = M_2PI * mLightZPR.y;
            float3 pos = float3(0, mLightZPR.x, 0);
            float r = mLightZPR.z;
            pos.x = r * cos(phi);
            pos.z = r * sin(phi);
            auto transf = Transform();
            transf.setTranslation(pos);
            sl->setTransformMatrix(transf.getMatrix());
        }
    }

    createTex(mpWiWox, mpDevice, targetDim);
    createTex(mpUVWoyz, mpDevice, targetDim);
    createTex(mpDfDxy, mpDevice, targetDim);




    // if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
    //     is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    // {
    //     FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    // }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    if (mHFBound)
        mTracer.pProgram->addDefine("HF_BOUND");
    else
        mTracer.pProgram->removeDefine("HF_BOUND");

    if (mNNInfer)
        mTracer.pProgram->addDefine("NN_INFER");
    else
        mTracer.pProgram->removeDefine("NN_INFER");
    if (mContactRefinement)
        mTracer.pProgram->addDefine("CONTACT_REFINEMENT");
    else
        mTracer.pProgram->removeDefine("CONTACT_REFINEMENT");
    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gControlParas"] = mControlParas;
    var["CB"]["gCurvatureParas"] = mCurvatureParas;
    var["CB"]["gApplySyn"] = mApplySyn;
    var["CB"]["gLocalFrame"] = mLocalFrame;

    if (mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["CB"]["envMapSampler"]);

    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels)
        bind(channel);
    for (auto channel : kOutputChannels)
        bind(channel);

    // Bind textures
    var["gColor"].setSrv(mpColor->getSRV());
    var["gHF"].setSrv(mpHF->getSRV());
    var["gHFMaxMip"].setSrv(mpHFMaxMip->getSRV());
    var["gNormalMap"].setSrv(mpNormalMap->getSRV());
    var["gTangentMap"].setSrv(mpTangentMap->getSRV());
    var["gPosMap"].setSrv(mpPosMap->getSRV());



    var["wiWox"].setUav(mpWiWox->getUAV());
    var["uvWoyz"].setUav(mpUVWoyz->getUAV());
    var["dfDxy"].setUav(mpDfDxy->getUAV());



    mpPixelDebug->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    mpPixelDebug->prepareProgram(mTracer.pProgram, mTracer.pVars->getRootVar());


    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

   // Copy pixel data to staging buffer for readback.
    // This is to avoid a full flush and the associated perf warning.
    // pRenderContext->copyResource(mpPixelStagingBuffer.get(), mpPixelDataBuffer.get());
    // pRenderContext->submit(false);
    // pRenderContext->signal(mpFence.get());
    // mPixelDataAvailable = true;
    // mPixelDataValid = false;

    mpPixelDebug->endFrame(pRenderContext);



    mFrameCount++;
}

void HFTracing::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    if (mTriID < mMaxTriCount){
        if(mTriID ==1) logInfo("[HF Tracing] Start to generate normal, tangent and position maps");
        generateGeometryMap(pRenderContext, renderData);
        visualizeMaps(pRenderContext, renderData);
    }
    else{
        if (!mMipGenerated){
            mpNormalMap->generateMips(pRenderContext);
            mpTangentMap->generateMips(pRenderContext);
            createMaxMip(pRenderContext, renderData);
            mMipGenerated = true;
        }

        if (mRenderType == RenderType::HF){
               renderHF(pRenderContext, renderData);
            if(mNNInfer)
            nnInferPass(pRenderContext, renderData);
        }
        else
            visualizeMaps(pRenderContext, renderData);
    }


    mFrameCount++;
}

void HFTracing::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    // widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    // dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    // widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    // dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    // widget.tooltip("Use importance sampling for materials", true);


    dirty |= widget.slider("HF Footprint Scale", mControlParas.x, 0.1f, 10.0f);
    widget.tooltip("Increse = less marching steps", true);
    dirty |= widget.slider("LoD Scale", mControlParas.y, 1.0f, 100.0f);
    dirty |= widget.slider("HF Offset", mControlParas.z, 0.0f, 1.0f);
    widget.tooltip("height = Scale * h + Offset", true);
    dirty |= widget.slider("HF Scale", mControlParas.w, 0.0f, 1.0f);
    widget.tooltip("height = Scale * h + Offset", true);

    dirty |= widget.slider("D", mCurvatureParas.x, 0.0f, 1.0f);
    widget.tooltip("Distance to mesh surface", true);
    dirty |= widget.slider("UV Scale", mCurvatureParas.z, 0.0f, 1.0f);

    dirty |= widget.slider("Z Min", mCurvatureParas.w, 0.0f, 1.0f);
    dirty |= widget.slider("Z Max", mCurvatureParas.y, 0.0f, 1.0f);

    dirty |= widget.var("Max Steps", mMaxSteps);

    dirty |= widget.checkbox("Contact Refinement", mContactRefinement);
    dirty |= widget.checkbox("Apply Syn", mApplySyn);
    dirty |= widget.checkbox("NN Infer", mNNInfer);
    dirty |= widget.checkbox("HF Bound", mHFBound);
    dirty |= widget.checkbox("Local Frame", mLocalFrame);


    // dirty |= widget.var("Light-Phi", mLightZPR.y);
    dirty |= widget.slider("Light Z", mLightZPR.x, 0.0f, 10.0f);
    dirty |= widget.slider("Light Phi", mLightZPR.y, 0.0f, 1.0f);
    dirty |= widget.slider("Light R", mLightZPR.z, 0.0f, 10.0f);
    dirty |= widget.slider("Light Scaling", mLightZPR.w, 0.01f, 10.0f);


    if (widget.button("Render Type")){
        mRenderType = (RenderType)(1 - (int)mRenderType);
    };

    widget.text("Triangle ID: " + std::to_string(mTriID));
    widget.text("Max Triangle ID: " + std::to_string(mMaxTriCount));

    mpPixelDebug->renderUI(widget);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void HFTracing::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("HFTracing: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
                desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        logInfo("[HF Tracing] Total mesh num: {} ", mpScene->getMeshCount());
        for(uint i = 0; i < mpScene->getMeshCount(); i++){
            logInfo("[HF Tracing] Mesh: #{} Triangle: {}", i, mpScene->getMesh(MeshID{i}).getTriangleCount());
        }

        // mpScene->getMeshVerticesAndIndices(MeshID meshID, const std::map<std::string, ref<Buffer>>& buffers);

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    // Read in textures, we use a constant texture now
    mpHF    = Texture::createFromFile(mpDevice, mTexturePath+mHFFileName+".png", true, false, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);
    mpColor = Texture::createFromFile(mpDevice, mTexturePath+mColorFileName, true, true, ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget);




    mpNormalMap = mpDevice->createTexture2D(
                2048,
                2048,
                ResourceFormat::RGBA32Float,
                1,
                Resource::kMaxPossible,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
              );
    mpTangentMap = mpDevice->createTexture2D(
                2048,
                2048,
                ResourceFormat::RGBA32Float,
                1,
                Resource::kMaxPossible,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
              );

    mpPosMap = mpDevice->createTexture2D(
                2048,
                2048,
                ResourceFormat::RGBA32Float,
                1,
                1,
                nullptr,
                ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
              );
    // mpTextureSynthesis = std::make_unique<TextureSynthesis>(mpDevice);
    mpMLP = std::make_unique<MLP>(mpDevice, "block_io");
    mpNBTF = std::make_unique<NBTF>(mpDevice, "block_io");
    // Create a precompute pass.

    DefineList defines = mpScene->getSceneDefines();
    mpGenerateGeometryMapPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/GenerateGeometryMap.cs.slang", "csMain", defines);
    mpVisualizeMapsPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/VisualizeMaps.cs.slang", "csMain", defines);
    mpCreateMaxMipPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/CreateMaxMip.cs.slang", "csMain", defines);
    mpInferPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/Inference.cs.slang", "csMain", defines);

}

void HFTracing::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);


}
