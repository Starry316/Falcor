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
#include "HelloDXR.h"
#include "Utils/Math/FalcorMath.h"
#include "Utils/UI/TextRenderer.h"
#include "SSRDefs.slangh"

FALCOR_EXPORT_D3D12_AGILITY_SDK

// static const float4 kClearColor(0.38f, 0.52f, 0.10f, 1);
static const float4 kClearColor(0.38f, 0.38f, 0.38f, 1);
// static const float4 kClearColor(0.0f, 0.0f, 0.0f, 1);
static const std::string kDefaultScene = "neural_materials/scene/PTFullObjTest.pyscene";
// static const std::string kDefaultScene = "neural_materials/scene/plane.pyscene";

void createTex(ref<Texture>& tex, ref<Device> device, Falcor::uint2 targetDim, uint arraySize)
{
    ResourceBindFlags flag = ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess;

    if (tex.get() == nullptr)
    {
        tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, arraySize, 1, nullptr, flag);
    }
    else
    {
        if (tex.get()->getWidth() != targetDim.x || tex.get()->getHeight() != targetDim.y)
        {
            logInfo("Recreating texture");
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, arraySize, 1, nullptr, flag);
        }
    }
};
float3 spherical_to_cartesian_radXZY(float2 sph)
{
    float3 p;
    p.x = -cos(sph.y - M_PI) * sin(sph.x);
    p.z = -sin(sph.y - M_PI) * sin(sph.x);
    p.y = cos(sph.x);
    return p;
}
float3 spherical_fibonacci_hemi(int i, int N)
{
    constexpr float golden_ratio = 1.61803398875f;
    float phi = M_2PI * ((float)i / golden_ratio - std::floor((float)i / golden_ratio));
    float z = (1.0f - (float)(i + 1) / (float)N) * 2.0f - 1.0f;
    // float z = ( (float)i + 0.5f ) / (float)N;
    return spherical_to_cartesian_radXZY(float2(std::acos(z), phi));
}
HelloDXR::HelloDXR(const SampleAppConfig& config) : SampleApp(config) {}

HelloDXR::~HelloDXR() {}

void HelloDXR::onLoad(RenderContext* pRenderContext)
{
    if (getDevice()->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        FALCOR_THROW("Device does not support raytracing!");
    }

    loadScene(kDefaultScene, getTargetFbo().get());
    std::vector<float4x4> viewProjMatrices(TOTAL_VIEWS);
    std::vector<float4x4> viewProjInvMatrices(TOTAL_VIEWS);
    std::vector<float3> billboardNormal(TOTAL_VIEWS);
    for (uint i = 0; i < TOTAL_VIEWS; i++)
    {
        float3 cameraPos = spherical_fibonacci_hemi(i, TOTAL_VIEWS);
        float4x4 viewMat = math::matrixFromLookAt(cameraPos, float3(0, 0, 0), float3(0, 1, 0));
        float4x4 projMat = math::ortho(-1.f, 1.f, -1.f, 1.f, 0.0f, 2.f);
        viewProjMatrices[i] = mul(projMat, viewMat);
        viewProjInvMatrices[i] = inverse(mul(projMat, viewMat));
        billboardNormal[i] = normalize(-cameraPos);
    }

    mpViewProjBuffer = getDevice()->createStructuredBuffer(
        uint32_t(sizeof(float4x4)),
        TOTAL_VIEWS,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        viewProjMatrices.data(),
        false
    );

    mpViewProjInvBuffer = getDevice()->createStructuredBuffer(
        uint32_t(sizeof(float4x4)),
        TOTAL_VIEWS,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        viewProjInvMatrices.data(),
        false
    );

    mpBillboardNormalBuffer = getDevice()->createStructuredBuffer(
        uint32_t(sizeof(float3)),
        TOTAL_VIEWS,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        billboardNormal.data(),
        false
    );

    mpPixelDebug = std::make_unique<PixelDebug>(getDevice());

    mpBillboardFbo =
        Fbo::create2D(getDevice(), BILLBOARD_RESOLUTION, BILLBOARD_RESOLUTION, ResourceFormat::RGBA32Float, ResourceFormat::D32Float);
    Sampler::Desc samplerDesc = Sampler::Desc();
    samplerDesc.setReductionMode(TextureReductionMode::Max);
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    mpMaxSampler = getDevice()->createSampler(samplerDesc);
}

void HelloDXR::onResize(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    if (mpCamera)
    {
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }
    mpRtOut = getDevice()->createTexture2D(
        width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, ResourceBindFlags::UnorderedAccess | ResourceBindFlags::ShaderResource
    );
    createTex(mpBillboards, getDevice(), Falcor::uint2(BILLBOARD_RESOLUTION, BILLBOARD_RESOLUTION), TOTAL_VIEWS);
    createTex(mpBillboardColors, getDevice(), Falcor::uint2(BILLBOARD_RESOLUTION, BILLBOARD_RESOLUTION), TOTAL_VIEWS);
    createTex(mpBillboardPosWs, getDevice(), Falcor::uint2(BILLBOARD_RESOLUTION, BILLBOARD_RESOLUTION), TOTAL_VIEWS);
    createTex(mpBillboardNormalWs, getDevice(), Falcor::uint2(BILLBOARD_RESOLUTION, BILLBOARD_RESOLUTION), TOTAL_VIEWS);
    createTex(mpReference, getDevice(), Falcor::uint2(width, height), 1);
    createTex(mpPostOut, getDevice(), Falcor::uint2(width, height), 1);
    mFrameCount = 0;
}

void HelloDXR::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);
    pRenderContext->clearFbo(mpBillboardFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpScene)
    {
        IScene::UpdateFlags updates = mpScene->update(pRenderContext, getGlobalClock().getTime());
        if (is_set(updates, IScene::UpdateFlags::GeometryChanged))
            FALCOR_THROW("This sample does not support scene geometry changes.");
        if (is_set(updates, IScene::UpdateFlags::RecompileNeeded))
            FALCOR_THROW("This sample does not support scene changes that require shader recompilation.");

        if (mFrameCount < TOTAL_VIEWS)
        {
            mCreateBillboards = true;
            if (mBillboardRenderID == 0)
            {
                pRenderContext->clearUAV(mpBillboards->getUAV().get(), float4(5.0f));
                pRenderContext->clearUAV(mpBillboardColors->getUAV().get(), float4(0.0f));
                pRenderContext->clearUAV(mpBillboardPosWs->getUAV().get(), float4(0.0f));
                pRenderContext->clearUAV(mpBillboardNormalWs->getUAV().get(), float4(0.0f));
            }

            logInfo("Rendering billboard {}", mBillboardRenderID);
            renderRaster(pRenderContext, pTargetFbo);
            if (++mBillboardRenderID == TOTAL_VIEWS)
            {
                mCreateBillboards = false;
                mBillboardRenderID = 0;
            }
        }
        else
        {
            renderRaster(pRenderContext, pTargetFbo);
            if (mRayTrace)
            {
                renderRT(pRenderContext, pTargetFbo);
                // postprocess(pRenderContext, pTargetFbo);
            }
        }
    }

    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
    mFrameCount++;
}

void HelloDXR::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Hello DXR Settings", {300, 400}, {10, 80});

    w.checkbox("Billboard Trace", mRayTrace);
    // w.checkbox("Use Depth of Field", mUseDOF);
    if (w.checkbox("Show Color", mShowColor))
    {
        mFrameCount = 0;
    };
    w.slider("Thickness", mThickness, 0.0f, 1.0f);
    if (w.button("Load Scene"))
    {
        std::filesystem::path path;
        if (openFileDialog(Scene::getFileExtensionFilters(), path))
        {
            loadScene(path, getTargetFbo().get());
        }
    }
    w.checkbox("Show Billboard", mShowBillboard);
    w.checkbox("Debug Mode", mDebugMode);
    w.checkbox("Show Diff", mShowDiff);
    w.checkbox("View Render Mask", mViewRenderMask);
    w.checkbox("Use Sorted Billboards", mUseSortedBillboards);

    w.slider("Billboard ID", mShowBillboardID, 0u, uint(TOTAL_VIEWS - 1));
    w.slider("Trace Step Count", mTraceStepCount, 1u, 5000u);
    w.var("Trace Step Count_", mTraceStepCount);

    w.slider("Forward Billboard Count", mFwdBillboardCount,  1u, uint(TOTAL_VIEWS));
    w.slider("Forward Billboard Radius", mFwdBillboardRadius, 0u, 20u);
    // w.slider("Forward Screen Radius", mFwdScreenRadius, 0u, 20u);
    w.slider("Forward Screen Radius", mFwdScreenRadius, 0.9f, 20.0f);

    w.slider("View Theta", mViewTheta, 0.0f, 1.0f);
    w.slider("View Phi", mViewPhi, 0.0f, 1.0f);
    w.slider("Invalid Threshold", mInvalidThreshold, 0.0f, 0.5f);
    w.slider("Invalid Max Threshold", mMaxInvalidThreshold, 0.0f, 1.5f);

    w.slider("Billboard Trace Count", mBillboardTraceCount, 1u, uint(TOTAL_VIEWS));

    w.separator();
    if (w.button("Output world pos billboards"))
    {
        for (size_t i = 0; i < TOTAL_VIEWS; i++)
        {
            mpBillboardColors->captureToFile(
                0,
                i,
                "D:/Data/Billboard/billboard_pos_" + std::to_string(i) + ".exr",
                Bitmap::FileFormat::ExrFile,
                Bitmap::ExportFlags::Uncompressed,
                true
            );
        }
    }
    if (w.button("Print Camera proj mat"))
    {
        float4x4 V = mpScene->getCamera()->getViewProjMatrixNoJitter();
        float3 cameraDir = normalize(mpScene->getCamera()->getData().target - mpScene->getCamera()->getData().posW);
        logInfo("cameraDir: \n{}", to_string(cameraDir));
        logInfo("V Matrix: \n{}", to_string(V));
        logInfo(
            "\n[[{}, {}, {}, {}], \n[{}, {}, {}, {}], \n[{}, {}, {}, {}], \n[{}, {}, {}, {}]],",
            V[0][0],
            V[0][1],
            V[0][2],
            V[0][3],
            V[1][0],
            V[1][1],
            V[1][2],
            V[1][3],
            V[2][0],
            V[2][1],
            V[2][2],
            V[2][3],
            V[3][0],
            V[3][1],
            V[3][2],
            V[3][3]
        );
    }
    mpPixelDebug->renderUI(w);

    // mpScene->renderUI(w);
}

bool HelloDXR::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (keyEvent.key == Input::Key::Space && keyEvent.type == KeyboardEvent::Type::KeyPressed)
    {
        mRayTrace = !mRayTrace;
        return true;
    }

    if (mpScene && mpScene->onKeyEvent(keyEvent))
        return true;

    return false;
}

bool HelloDXR::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent) && mpPixelDebug->onMouseEvent(mouseEvent);
}

void HelloDXR::loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo)
{
    mpScene = Scene::create(getDevice(), path);
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 100;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to use Falcor's material system.
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    // Get scene defines. These need to be set on any program using the scene.
    auto defines = mpScene->getSceneDefines();

    // Create raster pass.
    // This utility wraps the creation of the program and vars, and sets the necessary scene defines.
    ProgramDesc rasterProgDesc;
    rasterProgDesc.addShaderModules(shaderModules);
    rasterProgDesc.addShaderLibrary("Samples/HelloDXR/HelloDXR.3d.slang").vsEntry("vsMain").psEntry("psMain");
    rasterProgDesc.addTypeConformances(typeConformances);
    rasterProgDesc.setShaderModel(ShaderModel::SM6_3);
    mpRasterPass = RasterPass::create(getDevice(), rasterProgDesc, defines);

    // We'll now create a raytracing program. To do that we need to setup two things:
    // - A program description (ProgramDesc). This holds all shader entry points, compiler flags, macro defintions,
    // etc.
    // - A binding table (RtBindingTable). This maps shaders to geometries in the scene, and sets the ray generation and
    // miss shaders.
    //
    // After setting up these, we can create the Program and associated RtProgramVars that holds the variable/resource
    // bindings. The Program can be reused for different scenes, but RtProgramVars needs to binding table which is
    // Scene-specific and needs to be re-created when switching scene. In this example, we re-create both the program
    // and vars when a scene is loaded.

    ProgramDesc rtProgDesc;
    rtProgDesc.addShaderModules(shaderModules);
    rtProgDesc.addShaderLibrary("Samples/HelloDXR/HelloDXR.rt.slang");
    rtProgDesc.addTypeConformances(typeConformances);
    rtProgDesc.setMaxTraceRecursionDepth(3); // 1 for calling TraceRay from RayGen, 1 for calling it from the
                                             // primary-ray ClosestHit shader for reflections, 1 for reflection ray
                                             // tracing a shadow ray
    rtProgDesc.setMaxPayloadSize(24);        // The largest ray payload struct (PrimaryRayData) is 24 bytes. The payload size
                                             // should be set as small as possible for maximum performance.

    ref<RtBindingTable> sbt = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
    sbt->setRayGen(rtProgDesc.addRayGen("rayGen"));
    sbt->setMiss(0, rtProgDesc.addMiss("primaryMiss"));
    sbt->setMiss(1, rtProgDesc.addMiss("shadowMiss"));
    auto primary = rtProgDesc.addHitGroup("primaryClosestHit", "primaryAnyHit");
    auto shadow = rtProgDesc.addHitGroup("", "shadowAnyHit");
    sbt->setHitGroup(0, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), primary);
    sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), shadow);

    mpRaytraceProgram = Program::create(getDevice(), rtProgDesc, defines);
    mpRtVars = RtProgramVars::create(getDevice(), mpRaytraceProgram, sbt);

    ProgramDesc ssrProgDesc;
    ssrProgDesc.addShaderModules(shaderModules);
    ssrProgDesc.addTypeConformances(typeConformances);
    ssrProgDesc.addShaderLibrary("Samples/HelloDXR/SSR.cs.slang").csEntry("csMain");
    mpSSRPass = ComputePass::create(getDevice(), ssrProgDesc, defines);

    ProgramDesc postProcessProgDesc;
    postProcessProgDesc.addShaderModules(shaderModules);
    postProcessProgDesc.addTypeConformances(typeConformances);
    postProcessProgDesc.addShaderLibrary("Samples/HelloDXR/PostProcess.cs.slang").csEntry("csMain");
    mpPostProcessPass = ComputePass::create(getDevice(), postProcessProgDesc, defines);
}

void HelloDXR::setPerFrameVars(const Fbo* pTargetFbo)
{
    auto var = mpRtVars->getRootVar();
    var["PerFrameCB"]["invView"] = inverse(mpCamera->getViewMatrix());
    var["PerFrameCB"]["viewportDims"] = float2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
    float fovY = focalLengthToFovY(mpCamera->getFocalLength(), Camera::kDefaultFrameHeight);
    var["PerFrameCB"]["tanHalfFovY"] = std::tan(fovY * 0.5f);
    var["PerFrameCB"]["sampleIndex"] = mSampleIndex++;
    var["PerFrameCB"]["useDOF"] = mUseDOF;
    var["gOutput"] = mpRtOut;
}

void HelloDXR::renderRaster(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRaster");
    pRenderContext->clearUAV(mpReference->getUAV().get(), kClearColor);
    auto var = mpRasterPass->getVars()->getRootVar();
    var["PerFrameCB"]["gBillboardRenderID"] = mBillboardRenderID;
    var["PerFrameCB"]["gViewTheta"] = mViewTheta * float(M_PI);
    var["PerFrameCB"]["gViewPhi"] = mViewPhi * float(M_2PI);
    var["PerFrameCB"]["gCreateBillboards"] = mCreateBillboards;
    var["PerFrameCB"]["gCameraPos"] = spherical_fibonacci_hemi(mBillboardRenderID, TOTAL_VIEWS);
    var["PerFrameCB"]["gShowColor"] = mShowColor;

    // float4x4 viewMat = math::matrixFromLookAt(
    //     spherical_to_cartesian_radXZY(float2(mViewTheta * float(M_PI), mViewPhi * float(M_2PI))),
    //     float3(0, 0, 0),
    //     float3(0, 1, 0),
    //     math::Handedness::RightHanded
    // );
    float4x4 viewMat = math::matrixFromLookAt(spherical_fibonacci_hemi(mBillboardRenderID, TOTAL_VIEWS), float3(0, 0, 0), float3(0, 1, 0));
    float4x4 projMat = math::ortho(-1.f, 1.f, -1.f, 1.f, 0.0f, 2.f);
    var["PerFrameCB"]["viewMat"] = viewMat;
    var["PerFrameCB"]["projMat"] = projMat;

    var["gBillboards"] = mpBillboards;
    var["gBillboardColors"] = mpBillboardColors;
    var["gBillboardPosWs"] = mpBillboardPosWs;
    var["gBillboardNormalWs"] = mpBillboardNormalWs;
    var["gViewProjBuffer"] = mpViewProjBuffer;
    var["gReference"] = mpReference;

    if (mCreateBillboards)
        mpRasterPass->getState()->setFbo(mpBillboardFbo);
    else
        mpRasterPass->getState()->setFbo(pTargetFbo);

    mpScene->rasterize(pRenderContext, mpRasterPass->getState().get(), mpRasterPass->getVars().get(), RasterizerState::CullMode::None);

    if (mShowBillboard)
    {
        if (mShowColor)
            // pRenderContext->blit(mpBillboardColors->getSRV(0, 1, mShowBillboardID, 1), pTargetFbo->getRenderTargetView(0));
            pRenderContext->blit(mpBillboardNormalWs->getSRV(0, 1, mShowBillboardID, 1), pTargetFbo->getRenderTargetView(0));
        else
            pRenderContext->blit(mpBillboards->getSRV(0, 1, mShowBillboardID, 1), pTargetFbo->getRenderTargetView(0));
    }
}

void HelloDXR::renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "renderRT");

    // setPerFrameVars(pTargetFbo.get());

    // pRenderContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);

    // mpScene->raytrace(pRenderContext, mpRaytraceProgram.get(), mpRtVars, uint3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1));
    // pRenderContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));

    pRenderContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
    auto var = mpSSRPass->getRootVar();
    // mpSSRPass->getProgram()->addDefine("TOTAL_VIEWS", std::to_string(TOTAL_VIEWS));

    var["PerFrameCB"]["gFrameDim"] = uint2(mpRtOut->getWidth(), mpRtOut->getHeight());
    var["PerFrameCB"]["gShowBillboardID"] = mShowBillboardID;
    var["PerFrameCB"]["gBillboardTraceCount"] = mBillboardTraceCount;
    var["PerFrameCB"]["gDebugMode"] = mDebugMode;
    var["PerFrameCB"]["gInvalidThreshold"] = mInvalidThreshold;
    var["PerFrameCB"]["gMaxInvalidThreshold"] = mMaxInvalidThreshold;
    var["PerFrameCB"]["gThickness"] = mThickness;
    var["PerFrameCB"]["gViewRenderMask"] = mViewRenderMask;
    var["PerFrameCB"]["gTraceStepCount"] = mTraceStepCount;
    var["PerFrameCB"]["gUseSortedBillboards"] = mUseSortedBillboards;
    var["PerFrameCB"]["gShowColor"] = mShowColor;
    var["PerFrameCB"]["gFwdBillboardCount"] = mFwdBillboardCount;
    var["PerFrameCB"]["gFwdBillboardRadius"] = mFwdBillboardRadius;
    var["PerFrameCB"]["gFwdScreenRadius"] = mFwdScreenRadius;
    var["PerFrameCB"]["gViewTheta"] = mViewTheta;




    var["gMaxSampler"] = mpMaxSampler;
    var["gBillboards"] = mpBillboards;
    var["gBillboardColors"] = mpBillboardColors;
    var["gBillboardPosWs"] = mpBillboardPosWs;
    var["gBillboardNormalWs"] = mpBillboardNormalWs;
    var["gViewProjBuffer"] = mpViewProjBuffer;
    var["gViewProjInvBuffer"] = mpViewProjInvBuffer;
    var["gBillboardNormalBuffer"] = mpBillboardNormalBuffer;
    var["gOutput"] = mpRtOut;
    var["gReference"] = mpReference;
    mpScene->bindShaderData(var["gScene"]);

    mpPixelDebug->beginFrame(pRenderContext, uint2(mpRtOut->getWidth(), mpRtOut->getHeight()));
    mpPixelDebug->prepareProgram(mpSSRPass->getProgram(), mpSSRPass->getRootVar());
    mpSSRPass->execute(pRenderContext, mpRtOut->getWidth(), mpRtOut->getHeight());
    pRenderContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
    mpPixelDebug->endFrame(pRenderContext);
}

void HelloDXR::postprocess(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    FALCOR_ASSERT(mpScene);
    FALCOR_PROFILE(pRenderContext, "postprocess");

    auto var = mpPostProcessPass->getRootVar();
    var["PerFrameCB"]["gFrameDim"] = uint2(pTargetFbo->getWidth(), pTargetFbo->getHeight());
    var["PerFrameCB"]["gShowBillboardID"] = mShowBillboardID;
    var["PerFrameCB"]["gBillboardCount"] = TOTAL_VIEWS;
    var["PerFrameCB"]["gDebugMode"] = mDebugMode;
    var["PerFrameCB"]["gInvalidThreshold"] = mInvalidThreshold;
    var["PerFrameCB"]["gShowDiff"] = mShowDiff;
    var["gBillboards"] = mpBillboards;
    // var["gBillboardPosWs"] = mpBillboardPosWs;
    var["gViewProjBuffer"] = mpViewProjBuffer;
    var["gViewProjInvBuffer"] = mpViewProjInvBuffer;
    var["gBillboardNormalBuffer"] = mpBillboardNormalBuffer;
    var["gRtOut"] = mpRtOut;
    var["gOutput"] = mpPostOut;
    var["gReference"] = mpReference;
    mpScene->bindShaderData(var["gScene"]);

    // mpPixelDebug->beginFrame(pRenderContext, uint2(pTargetFbo->getWidth(), pTargetFbo->getHeight()));
    // mpPixelDebug->prepareProgram(mpPostProcessPass->getProgram(), mpPostProcessPass->getRootVar());
    mpPostProcessPass->execute(pRenderContext, pTargetFbo->getWidth(), pTargetFbo->getHeight());
    pRenderContext->blit(mpPostOut->getSRV(), pTargetFbo->getRenderTargetView(0));
    // mpPixelDebug->endFrame(pRenderContext);
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "HelloDXR";
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = 1800;
    config.windowDesc.height = 1800;
    HelloDXR helloDXR(config);
    return helloDXR.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
