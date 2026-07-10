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
#include "Tools/Utils.h"
#define pX mXYUV.x
#define pY mXYUV.y
#define pU mXYUV.z
#define pV mXYUV.w
extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PTTest>();
}
float3 spherical_to_cartesian_radXZY(float2 sph)
{
    float3 p;
    p.x = -cos(sph.y - M_PI) * sin(sph.x);
    p.z = -sin(sph.y - M_PI) * sin(sph.x);
    p.y = cos(sph.x);
    return p;
}

float4x4 MakeViewFromCameraPos(float3 cameraPos)
{
    // Forward: from camera to origin
    float3 fwd = normalize(-cameraPos);

    // Stable Y-up basis
    float3 worldUp = float3(0.0f, 1.0f, 0.0f);
    if (abs(dot(fwd, worldUp)) > 0.99999999f)
        worldUp = float3(0.0f, 0.0f, 1.0f);

    float3 right = normalize(cross(worldUp, fwd));
    float3 up = cross(fwd, right);

    // World -> Camera:
    // x = dot(right, P) - dot(right, C)
    // y = dot(up,    P) - dot(up,    C)
    // z = dot(fwd,   P) - dot(fwd,   C)
    return float4x4{
        right.x,
        right.y,
        right.z,
        -dot(right, cameraPos),
        up.x,
        up.y,
        up.z,
        -dot(up, cameraPos),
        fwd.x,
        fwd.y,
        fwd.z,
        -dot(fwd, cameraPos),
        0.f,
        0.f,
        0.f,
        1.f};
}
float4x4 makeOrthoProjection(float halfSize)
{
    float inv = (halfSize != 0.f) ? (1.f / halfSize) : 0.f;

    return float4x4{inv, 0.f, 0.f, 0.f, 0.f, inv, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
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
    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    createTex(mpBarycentric, mpDevice, targetDim);
    createTex(mpWi, mpDevice, targetDim);
    createTex(mpRadiance, mpDevice, targetDim);

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
    var["CB"]["gShowSelectedTri"] = mShowSelectedTri;
    var["CB"]["gSelectedInstanceID"] = mSelectedInstanceID;
    var["CB"]["gSelectedTriangleID"] = mSelectedTriangleID;
    var["CB"]["gTriSampleUV"] = float2(mTriSampleU, mTriSampleV);
    var["CB"]["gProbePos"] = mProbePos;

    var["gBarycentric"] = mpBarycentric;
    var["gWi"] = mpWi;
    var["gRadiance"] = mpRadiance;

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



    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    mFrameCount++;
}

void PTTest::handleOutput()
{
        auto camera = mpScene->getCamera();
        camera->setOutputPath(fmt::format(mOutputPath, mOutputIndx, mTriSampleU, mTriSampleV));
        if (!camera->isNextStep())
        {
            return;
        }
        camera->setNextStep(false);
        camera->setAccumulating(mIsOutputing);
        camera->setOutputFrameCount(mOutputSPP);
        mOutputIndx++;


        if (mOutputIndx < 3){
            mTriSampleU = vertexUV[mOutputIndx].x;
            mTriSampleV = vertexUV[mOutputIndx].y;
            return;
        }

        if(mOutputIndx == 3){
            mTriSampleU = startingUV;
            mTriSampleV = startingUV;
        }


        mTriSampleU += intervalUV;

        if (mTriSampleU >= startingUV + numberOfInterals * intervalUV - 0.01f)
        {
            mTriSampleU = startingUV;
            mTriSampleV+= intervalUV;
        }

        if (mTriSampleV >= startingUV + numberOfInterals * intervalUV - 0.01f)
        {
            mOutputStep = 0;
            mOutputIndx = 0;
            mpScene->getCamera()->setResetFlag(true);
            mpScene->getCamera()->setNextStep(false);
            mIsOutputing = false;
            mpScene->getCamera()->setAccumulating(false);

            mTriSampleU = startingUV;
            mTriSampleV = startingUV;
        }
        return;
}
void PTTest::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    widget.text(fmt::format("Meshes: {}", mMeshCount));
    widget.text(fmt::format("Instances: {}", mInstanceCount));
    widget.separator();

    // Triangle picking. An instance maps to exactly one mesh (gi.geometryID), so instance ID alone
    // determines the geometry; we resolve and show the mesh ID and its triangle count below.
    dirty |= widget.var("Pick instance ID", mSelectedInstanceID, 0u, mInstanceCount > 0 ? mInstanceCount - 1 : 0u);

    uint32_t selectedTriangleCount = 0;
    if (mpScene && mSelectedInstanceID < mpScene->getGeometryInstanceCount())
    {
        const auto& gi = mpScene->getGeometryInstance(mSelectedInstanceID);
        if (gi.getType() == GeometryType::TriangleMesh)
        {
            selectedTriangleCount = mpScene->getMesh(MeshID{gi.geometryID}).getTriangleCount();
            widget.text(fmt::format("  -> mesh ID: {}, triangles: {}", gi.geometryID, selectedTriangleCount));
        }
        else
        {
            widget.text("  -> instance is not a triangle mesh");
        }
    }

    dirty |= widget.var("Pick triangle ID", mSelectedTriangleID, 0u, selectedTriangleCount > 0 ? selectedTriangleCount - 1 : 0u);
    if (widget.button("Compute triangle world positions"))
        computeSelectedTriangleWorldPositions();
    if (mSelectedTriangleValid)
    {
        widget.text(fmt::format("Vertex IDs: {}, {}, {}", mSelectedTriVertexIDs[0], mSelectedTriVertexIDs[1], mSelectedTriVertexIDs[2]));
        for (int i = 0; i < 3; ++i)
        {
            const float3& p = mSelectedTrianglePosW[i];
            widget.text(fmt::format("v{} (id {}): ({:.4f}, {:.4f}, {:.4f})", i, mSelectedTriVertexIDs[i], p.x, p.y, p.z));
        }
    }
    dirty |= widget.checkbox("Show select Tri", mShowSelectedTri);

    // Uniform triangle sampling coordinates for the primary ray origin.
    dirty |= widget.var("Tri sample u", mTriSampleU, 0.f, 1.f);
    dirty |= widget.var("Tri sample v", mTriSampleV, 0.f, 1.f);

    widget.separator();

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.slider("light theta", mLightTheta, 0.0f, 2.0f);
    dirty |= widget.slider("light phi", mLightPhi, 0.0f, 1.0f);

    dirty |= widget.slider("view theta", mViewTheta, 0.0f, 1.0f);
    dirty |= widget.slider("view phi", mViewPhi, 0.0f, 1.0f);
    dirty |= widget.slider("view size", mViewSize, 0.0f, 10.0f);
    dirty |= widget.slider("view height", mViewHeight, 0.0f, 10.0f);
    dirty |= widget.var("view height_", mViewHeight);
    dirty |= widget.slider("view height bot", mViewHeightBot, 0.0f, mViewHeight);

    dirty |= widget.slider("probe pos", mProbePos, 0.0f, 1.0f);
    // dirty |= widget.slider("x", pX, 0.0f, 1.0f);
    // dirty |= widget.slider("y", pY, 0.0f, 1.0f);
    // dirty |= widget.slider("u", pU, 0.0f, 1.0f);
    // dirty |= widget.slider("v", pV, 0.0f, 1.0f);

    dirty |= widget.checkbox("btf mode", mBTFViewMode);
    dirty |= widget.checkbox("Plucker mode", mPluckerMode);

    dirty |= widget.checkbox("Change Light", mChangeLight);

    if (widget.button("output frame"))
    {
        mpBarycentric->captureToFile(
                    0, 0, "C:/Data/Render/bary.exr", Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Lossy
                );
        mpWi->captureToFile(
                    0, 0, "C:/Data/Render/wi.exr", Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Lossy
                );
        mpRadiance->captureToFile(
                    0, 0, "C:/Data/Render/rad.exr", Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Lossy
                );

    }

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

            mTriSampleU = vertexUV[0].x;
            mTriSampleV = vertexUV[0].y;

            // mTriSampleU = 0.05f;
            // mTriSampleV = 0.05f;

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

void PTTest::computeSelectedTriangleWorldPositions()
{
    mSelectedTriangleValid = false;

    if (!mpScene)
    {
        logWarning("PTTest: no scene loaded.");
        return;
    }
    if (mSelectedInstanceID >= mpScene->getGeometryInstanceCount())
    {
        logWarning("PTTest: instance ID {} out of range ({} instances).", mSelectedInstanceID, mpScene->getGeometryInstanceCount());
        return;
    }

    const GeometryInstanceData& gi = mpScene->getGeometryInstance(mSelectedInstanceID);
    if (gi.getType() != GeometryType::TriangleMesh)
    {
        logWarning("PTTest: instance {} is not a triangle mesh.", mSelectedInstanceID);
        return;
    }

    const MeshID meshID{gi.geometryID};
    const MeshDesc& desc = mpScene->getMesh(meshID);
    const uint32_t vertexCount = desc.vertexCount;
    const uint32_t triangleCount = desc.getTriangleCount();
    if (mSelectedTriangleID >= triangleCount)
    {
        logWarning("PTTest: triangle ID {} out of range ({} triangles in mesh {}).", mSelectedTriangleID, triangleCount, meshID.get());
        return;
    }

    // GPU output buffers filled by getMeshVerticesAndIndices (positions/texcrds are float3, indices are uint3).
    // Positions are in object/local space; indices are local vertex indices for the mesh.
    const ResourceBindFlags uavFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto pPositions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);
    auto pTexcrds = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);
    auto pIndices = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);

    mpScene->getMeshVerticesAndIndices(
        meshID, {{"positions", pPositions}, {"texcrds", pTexcrds}, {"triangleIndices", pIndices}}
    );

    // Copy the data we need to CPU-readable staging buffers.
    auto pIdxStaging = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
    auto pPosStaging = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);

    RenderContext* pRenderContext = mpDevice->getRenderContext();
    pRenderContext->copyBufferRegion(pIdxStaging.get(), 0, pIndices.get(), 0, sizeof(uint3) * triangleCount);
    pRenderContext->copyBufferRegion(pPosStaging.get(), 0, pPositions.get(), 0, sizeof(float3) * vertexCount);
    pRenderContext->submit(true); // Wait for GPU work to complete.

    const uint3* pIdxData = reinterpret_cast<const uint3*>(pIdxStaging->map());
    const float3* pPosData = reinterpret_cast<const float3*>(pPosStaging->map());

    const uint3 tri = pIdxData[mSelectedTriangleID];

    // Object-to-world transform for this instance.
    const float4x4 objectToWorld = mpScene->getAnimationController()->getGlobalMatrices()[gi.globalMatrixID];

    for (int i = 0; i < 3; ++i)
    {
        mSelectedTriVertexIDs[i] = tri[i];
        const float3 posO = pPosData[tri[i]];
        const float4 posW = mul(objectToWorld, float4(posO, 1.f));
        mSelectedTrianglePosW[i] = float3(posW.x, posW.y, posW.z) / posW.w;
    }

    pIdxStaging->unmap();
    pPosStaging->unmap();

    mSelectedTriangleValid = true;

    logInfo(
        "PTTest: instance {} (mesh {}) triangle {} world positions: v0=({}, {}, {}), v1=({}, {}, {}), v2=({}, {}, {})",
        mSelectedInstanceID, meshID.get(), mSelectedTriangleID,
        mSelectedTrianglePosW[0].x, mSelectedTrianglePosW[0].y, mSelectedTrianglePosW[0].z,
        mSelectedTrianglePosW[1].x, mSelectedTrianglePosW[1].y, mSelectedTrianglePosW[1].z,
        mSelectedTrianglePosW[2].x, mSelectedTrianglePosW[2].y, mSelectedTrianglePosW[2].z
    );
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

    // Reset scene stats; repopulated below if we have a scene.
    mMeshCount = 0;
    mInstanceCount = 0;
    mSelectedTriangleValid = false;

    if (mpScene)
    {
        mMeshCount = mpScene->getMeshCount();
        mInstanceCount = mpScene->getGeometryInstanceCount();
        logInfo("PTTest: scene has {} meshes and {} geometry instances.", mMeshCount, mInstanceCount);

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
