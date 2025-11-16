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
#include "PTTest.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#define pX mXYUV.x
#define pY mXYUV.y
#define pU mXYUV.z
#define pV mXYUV.w
extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PTTest>();
}

namespace
{
const char kShaderFile[] = "RenderPasses/PTTest/MinimalPathTracer.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" , true},
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

PTTest::PTTest(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);
}

void PTTest::parseProperties(const Properties& props)
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
            logWarning("Unknown property '{}' in PTTest properties.", key);
    }
}

Properties PTTest::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

RenderPassReflection PTTest::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void PTTest::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

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

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->getLightCount() > 0)
    {
        ref<Light> light = mpScene->getLight(0);
        DirectionalLight* dirlight = (DirectionalLight*)light.get();
        if (light->getType() == LightType::Directional)
        {
            ref<DirectionalLight> dl = static_ref_cast<DirectionalLight>(light);
            float phi = M_2PI * mLightPhi;
            float theta = M_PI_2 * mLightTheta;
            float3 dir;
            dir.x = -(cos(phi - M_PI)) * sin(theta);
            dir.z = -(sin(phi - M_PI)) * sin(theta);
            dir.y = cos(theta);
            dl->setWorldDirection(-dir);
        }
    }

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    // mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

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
    var["CB"]["gViewTheta"] = mViewTheta;
    var["CB"]["gViewPhi"] = mViewPhi;
    var["CB"]["gViewSize"] = mViewSize;
    var["CB"]["gBTFViewMode"] = mBTFViewMode;
    var["CB"]["gViewHeight"] = mViewHeight;
    var["CB"]["gViewHeightBot"] = mViewHeightBot;
    var["CB"]["gMaxBounces"] = mMaxBounces;
    var["CB"]["gXYUV"] = mXYUV;
    var["CB"]["gPluckerMode"] = mPluckerMode;
    var["CB"]["gShowOffset"] = mShowOffset;
    if (mpEnvMapSampler)
        mpEnvMapSampler->bindShaderData(var["CB"]["gEnvMapSampler"]);
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

    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    mFrameCount++;
}



void PTTest::handleOutput()
{
    auto camera = mpScene->getCamera();
    if (!mChangeLight)
    {
        camera->setOutputPath(fmt::format(mOutputPath, mOutputIndx, pV, pY, mViewTheta, mViewPhi));
    }
    else
    {
        camera->setOutputPath(fmt::format(mOutputBTFPath, mOutputIndx, mLightTheta, mLightPhi, mViewTheta, mViewPhi));
    }

    if (!camera->isNextStep())
    {
        return;
    }
    camera->setNextStep(false);
    camera->setAccumulating(mIsOutputing);
    camera->setOutputFrameCount(mOutputSPP);
    mOutputIndx++;

    // Steps (match original increments)
    const float phiLightStep = 1.0f / 1.0f;
    const float thetaLightStep = 1.0f / 1.0f;

    // const float phiStep = 1.0f / 100.0f;
    // const float thetaStep = 1.0f / 200.0f;

    // const float phiStep = 1.0f / 15.0f;
    // const float thetaStep = 1.0f / 20.0f;

    const float phiStep = 1.0f / 200.0f;
    const float thetaStep = 1.0f / 100.0f;

    // const float yStep = 1.0f / 100.0f;
    // const float vStep = 1.0f / 100.0f;

    if (!mChangeLight)
    {
        if (mPluckerMode)
        {
            mViewPhi += phiStep;

            if (mViewPhi >= 1.0f)
            {
                // Reset theta to original small value and carry to phi
                mViewPhi = 0.0f;
                mViewTheta += thetaStep;
                // If phi wrapped past end, we've finished the full nested iteration
                if (mViewTheta >= 1.0f)
                {
                    // finalize/stop outputing
                    mViewTheta = 0.0f;
                    mOutputStep = 0;
                    mOutputIndx = 0;
                    mpScene->getCamera()->setResetFlag(true);
                    mpScene->getCamera()->setNextStep(false);
                    mIsOutputing = false;
                    mpScene->getCamera()->setAccumulating(false);
                }
            }
        }

        else
        {
            mViewPhi += phiStep;
            mSampleTheta += thetaStep * phiStep;

            mViewTheta = 2 * acos(1 - mSampleTheta) / M_PI;

            if (mViewPhi >= 1.0f)
            {
                // Reset theta to original small value and carry to phi
                mViewPhi = 0.0f;
                // mViewTheta += thetaStep;
                // If phi wrapped past end, we've finished the full nested iteration
                if (mViewTheta >= 0.95f)
                {
                    // finalize/stop outputing
                    mViewTheta = 0.0f;
                    mOutputStep = 0;
                    mOutputIndx = 0;
                    mpScene->getCamera()->setResetFlag(true);
                    mpScene->getCamera()->setNextStep(false);
                    mIsOutputing = false;
                    mpScene->getCamera()->setAccumulating(false);
                }
            }
        }
    }

    // if (!mChangeLight)
    // {
    //     mViewPhi += phiStep;
    //     if (mViewPhi >= 1.0f)
    //     {
    //         // Reset theta to original small value and carry to phi
    //         mViewPhi = 0.0f;
    //         mViewTheta += thetaStep;
    //         // If phi wrapped past end, we've finished the full nested iteration
    //         if (mViewTheta >= 0.6f)
    //         {
    //             // finalize/stop outputing
    //             mViewTheta = 0.0f;
    //             mOutputStep = 0;
    //             mOutputIndx = 0;
    //             mpScene->getCamera()->setResetFlag(true);
    //             mpScene->getCamera()->setNextStep(false);
    //             mIsOutputing = false;
    //             mpScene->getCamera()->setAccumulating(false);
    //         }
    //     }
    // }
    else
    {
        mLightPhi += phiLightStep;
        mSampleLightTheta += thetaLightStep * phiLightStep;
        mLightTheta = 2 * acos(1 - mSampleLightTheta) / M_PI;

        if (mLightPhi >= 1.0f)
        {
            mLightPhi = 0.0f;
            if (mLightTheta >= 0.6f)
            {
                mLightTheta = 0.01f;
                mOutputOffsetIndx++;

                mViewPhi += phiStep;
                mSampleTheta += thetaStep * phiStep;
                mViewTheta = 2 * acos(1 - mSampleTheta) / M_PI;
                if (mViewPhi >= 1.0f)
                {
                    // Reset theta to original small value and carry to phi
                    mViewPhi = 0.0f;
                    // mViewTheta += thetaStep;
                    // If phi wrapped past end, we've finished the full nested iteration
                    if (mViewTheta >= 0.8f)
                    {
                        // finalize/stop outputing
                        mViewTheta = 0.0f;
                        mOutputStep = 0;
                        mOutputIndx = 0;
                        mpScene->getCamera()->setResetFlag(true);
                        mpScene->getCamera()->setNextStep(false);
                        mIsOutputing = false;
                        mpScene->getCamera()->setAccumulating(false);
                    }
                }
            }
        }
    }

    // else
    // {
    //     mLightPhi += phiLightStep;
    //     if (mLightPhi >= 1.0f)
    //     {
    //         mLightPhi = 0.0f;
    //         mLightTheta += thetaLightStep;
    //         if (mLightTheta >= 0.4f)
    //         {
    //             mLightTheta = 0.05f;
    //             mOutputOffsetIndx++;

    //             mViewPhi += phiStep;
    //             if (mViewPhi >= 1.0f)
    //             {
    //                 // Reset theta to original small value and carry to phi
    //                 mViewPhi = 0.0f;
    //                 mViewTheta += thetaStep;
    //                 // If phi wrapped past end, we've finished the full nested iteration
    //                 if (mViewTheta >= 0.3f)
    //                 {
    //                     // finalize/stop outputing
    //                     mViewTheta = 0.0f;
    //                     mOutputStep = 0;
    //                     mOutputIndx = 0;
    //                     mpScene->getCamera()->setResetFlag(true);
    //                     mpScene->getCamera()->setNextStep(false);
    //                     mIsOutputing = false;
    //                     mpScene->getCamera()->setAccumulating(false);
    //                 }
    //             }
    //         }
    //     }
    // }

    // pY += yStep;
    // if (pY >= 1.0f)
    // {
    //     // Reset theta to original small value and carry to phi
    //     pY = 0.0f;
    //     pV += vStep;
    //     // If phi wrapped past end, we've finished the full nested iteration
    //     if (pV >= 1.0f)
    //     {
    //         // finalize/stop outputing
    //         pV = 0.0f;
    //         mOutputStep = 0;
    //         mOutputIndx = 0;
    //         mpScene->getCamera()->setResetFlag(true);
    //         mpScene->getCamera()->setNextStep(false);
    //         mIsOutputing = false;
    //         mpScene->getCamera()->setAccumulating(false);
    //     }
    // }
}
void PTTest::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.slider("light theta", mLightTheta, 0.0f, 1.0f);
    dirty |= widget.slider("light phi", mLightPhi, 0.0f, 1.0f);

    dirty |= widget.slider("view theta", mViewTheta, 0.0f, 1.0f);
    dirty |= widget.slider("view phi", mViewPhi, 0.0f, 1.0f);
    dirty |= widget.slider("view size", mViewSize, 0.0f, 10.0f);
    dirty |= widget.slider("view height", mViewHeight, 0.0f, 10.0f);
    dirty |= widget.var("view height_", mViewHeight);
    dirty |= widget.slider("view height bot", mViewHeightBot, 0.0f, mViewHeight);

    dirty |= widget.slider("x", pX, 0.0f, 1.0f);
    dirty |= widget.slider("y", pY, 0.0f, 1.0f);
    dirty |= widget.slider("u", pU, 0.0f, 1.0f);
    dirty |= widget.slider("v", pV, 0.0f, 1.0f);

    dirty |= widget.checkbox("btf mode", mBTFViewMode);
    dirty |= widget.checkbox("Plucker mode", mPluckerMode);
    dirty |= widget.checkbox("Show offset", mShowOffset);
    dirty |= widget.checkbox("Change Light", mChangeLight);

    if (!mChangeLight)
    {
        widget.textbox("Output Path", mOutputPath);
    }
    else
    {
        widget.textbox("Output Path", mOutputBTFPath);
    }

    widget.var("OutputSPP", mOutputSPP);
    if (mIsOutputing)
    {
        handleOutput();
        if (widget.button("Stop", true))
        {
            auto camera = mpScene->getCamera();
            camera->setOutputFrameCount(mOutputSPP);
            camera->setAccumulating(false);
            mIsOutputing = false;
            dirty = true;
        }
    }
    else
    {
        if (widget.button("Start Output"))
        {
            mIsOutputing = true;
            dirty = true;
            mLightTheta = 0.01f;
            // mViewTheta = 0.05f;
            mViewTheta = 0.01f;
            pY = 0.0f;
            pV = 0.0f;

            auto camera = mpScene->getCamera();
            camera->setOutputFrameCount(mOutputSPP);
            camera->setAccumulating(true);
        }

        if (widget.button("Rest"))
        {
            mIsOutputing = false;
            dirty = true;

            auto camera = mpScene->getCamera();
            camera->setOutputFrameCount(mOutputSPP);
            camera->setAccumulating(false);
            mIsOutputing = false;
            dirty = true;
            mOutputStep = 0;
            mOutputOffsetIndx = 0;
            mOutputIndx = 0;
            mViewTheta = 0;
            mViewPhi = 0;
            mLightTheta = 0;
            mLightPhi = 0;
        }
    }

    //

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void PTTest::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
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
            logWarning("PTTest: This render pass does not support custom primitives.");
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

        if (mpScene->useEnvLight())
        {
            if (!mpEnvMapSampler)
            {
                mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            }
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }
#ifdef NN_PRECOMPUTE
    mpNNMatT   = std::make_shared<NNMat>(mpDevice, mNeuBTFName, 1, 0);
    mpNNMatBTF = std::make_shared<NNMat>(mpDevice, mNeuBTFName, 0, 1);
#endif
}

void PTTest::prepareVars()
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
