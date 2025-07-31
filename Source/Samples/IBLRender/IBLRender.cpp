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
#include "IBLRender.h"
#include <fstream>
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/UI/TextRenderer.h"
FALCOR_EXPORT_D3D12_AGILITY_SDK
static const std::string kDefaultScene = "dummy_scene.pyscene";
static const float4 kClearColor(0.38f, 0.52f, 0.10f, 1);
uint32_t mSampleGuiWidth = 250;
uint32_t mSampleGuiHeight = 200;
uint32_t mSampleGuiPositionX = 20;
uint32_t mSampleGuiPositionY = 40;
float3 spherical_to_cartesian_rad(float2 sph)
{
    float3 p;
    p.x = -cos(sph.y - M_PI) * sin(sph.x);
    p.y = -sin(sph.y - M_PI) * sin(sph.x);
    p.z = cos(sph.x);
    return p;
}
void createTex(ref<Texture>& tex, ref<Device> device, Falcor::uint2 targetDim, bool buildCuda = false, bool isUint = false)
{
    ResourceBindFlags flag = ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess;
    if (buildCuda)
        flag |= ResourceBindFlags::Shared;

    if (tex.get() == nullptr)
    {
        if (isUint)
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, flag);
        else
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);
    }
    else
    {
        if (tex.get()->getWidth() != targetDim.x || tex.get()->getHeight() != targetDim.y)
        {
            if (isUint)
                tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, flag);
            else
                tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);
        }
    }
}
void createBuffer(ref<Buffer>& buf, ref<Device> device, Falcor::uint2 targetDim, uint itemSize = 4)
{
    if (buf.get() == nullptr)
    {
        buf = device->createBuffer(
            targetDim.x * targetDim.y * itemSize * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr
        );
    }
    else
    {
        if (buf.get()->getElementCount() != targetDim.x * targetDim.y * itemSize * sizeof(float))
        {
            logInfo("Recreating buffer");
            buf = device->createBuffer(
                targetDim.x * targetDim.y * itemSize * sizeof(float),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr
            );
        }
    }
}
IBLRender::IBLRender(const SampleAppConfig& config) : SampleApp(config)
{
    //
}

IBLRender::~IBLRender()
{
    //
}

void IBLRender::loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo)
{
    mpScene = Scene::create(getDevice(), path);
    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = mpScene->getSceneBounds().radius();
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    // Get shader modules and type conformances for types used by the scene.
    // These need to be set on the program in order to use Falcor's material system.
    auto shaderModules = mpScene->getShaderModules();
    auto typeConformances = mpScene->getTypeConformances();

    // Get scene defines. These need to be set on any program using the scene.
    auto defines = mpScene->getSceneDefines();

    defines.add(mpSampleGenerator->getDefines());

    mpDebugPass = ComputePass::create(getDevice(), "Samples/IBLRender/render.cs.slang", "csMain", defines);
}

void IBLRender::onLoad(RenderContext* pRenderContext)
{
    // Load shaders

    mpPixelDebug = std::make_unique<PixelDebug>(getDevice());
    mpDisplayPass = FullScreenPass::create(getDevice(), "Samples/IBLRender/display.ps.slang");
    // mpNNMat = std::make_shared<NNMat>(getDevice(), "leather11_XYZ_BTFNetXYZHU72x2", 0, 1);
    mpNNMat = std::make_shared<NNMat>(getDevice(), mNNMatName, 0, 1);
    mpNNMatIBL = std::make_shared<NNMat>(getDevice(), mNNIBLName, 0, 0);
    mpEnvMap = EnvMap::createFromFile(getDevice(), fmt::format("{}/media/neural_materials/scene/envmap/{}", mProjectPath, mEnvmapName));
    mpEnvMapSampler = std::make_unique<EnvMapSampler>(getDevice(), mpEnvMap);
    mpSampleGenerator = SampleGenerator::create(getDevice(), SAMPLE_GENERATOR_UNIFORM);

    loadScene(kDefaultScene, getTargetFbo().get());
}

void IBLRender::onShutdown()
{
    //
}

void IBLRender::onResize(uint32_t width, uint32_t height) {}

void IBLRender::display(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    auto var = mpDisplayPass->getRootVar()["CB"];
    var["iResolution"] = Falcor::float2(width, height);

    mpDisplayPass->getRootVar()["ouputColor"] = mpOutColor;
    mpDisplayPass->getRootVar()["ouputColorRef"] = mpOutColorRef;
    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpDisplayPass->getProgram(), mpDisplayPass->getRootVar());
    mpDisplayPass->execute(pRenderContext, pTargetFbo);
    mpPixelDebug->endFrame(pRenderContext);
}
void IBLRender::render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    auto var = mpDebugPass->getRootVar();
    var["CB"]["iResolution"] = float2(400);
    var["CB"]["gWo"] = mWo;
    var["CB"]["gWi"] = mWi;
    var["CB"]["gSampleNum"] = mSampleNum;
    var["CB"]["gRotAngles"] = mEnvRotAngle;
    var["CB"]["gShowIBL"] = mShowIBL;

    var["ouputColor"] = mpOutColor;
    var["ouputColorRef"] = mpOutColorRef;

    mpScene->bindShaderData(var["scene"]);

    mpNNMat->bindShaderData(var["CB"]["nnmat"]);
    mpNNMatIBL->bindShaderData(var["CB"]["nnmatIBL"]);
    mpEnvMap->bindShaderData(var["CB"]["envMap"]);
    mpEnvMapSampler->bindShaderData(var["CB"]["envMapSampler"]);
    mpSampleGenerator->bindShaderData(var);

    // mpPixelDebug->beginFrame(pRenderContext, targetDim);
    // mpPixelDebug->prepareProgram(mpDebugPass->getProgram(), mpDebugPass->getRootVar());
    if (mDirty || mOutputing)
        mpDebugPass->execute(pRenderContext, 400, 400);
}

void IBLRender::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    uint2 targetDim = uint2(width, height);

    uint2 outputDim = uint2(400, 400);
    createTex(mpOutColor, pRenderContext->getDevice(), outputDim);
    createTex(mpOutColorRef, pRenderContext->getDevice(), outputDim);

    render(pRenderContext, pTargetFbo);
    display(pRenderContext, pTargetFbo);
    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});

    mFrames++;
}

void IBLRender::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    mDirty = false;
    mpEnvMap->setRotation(mEnvRotAngle);
    mDirty |= w.checkbox("show IBL", mShowIBL);

    mDirty |= w.slider("wo.t", mWo.x, 0.0f, (float)M_PI / 2.0f);
    mDirty |= w.slider("wo.p", mWo.y, 0.0f, 2.0f * (float)M_PI);

    mDirty |= w.slider("wi.t", mWi.x, 0.0f, (float)M_PI / 2.0f);
    mDirty |= w.slider("wi.p", mWi.y, 0.0f, 2.0f * (float)M_PI);
    mDirty |= w.slider("sampleNum", mSampleNum, 1, 4096);

    mDirty |= w.slider("Env rot X", mEnvRotAngle.x, 0.0f, 360.0f);
    if (w.button("X -", true))
    {
        mEnvRotAngle.x -= 5;
    }
    if (w.button("X +", true))
    {
        mEnvRotAngle.x += 5;
    }
    mDirty |= w.slider("Env rot Y", mEnvRotAngle.y, 0.0f, 360.0f);
    if (w.button("Y -", true))
    {
        mEnvRotAngle.y -= 5;
    }
    if (w.button("Y +", true))
    {
        mEnvRotAngle.y += 5;
    }
    mDirty |= w.slider("Env rot Z", mEnvRotAngle.z, 0.0f, 360.0f);
    if (w.button("Z -", true))
    {
        mEnvRotAngle.z -= 5;
    }
    if (w.button("Z +", true))
    {
        mEnvRotAngle.z += 5;
    }
    if (w.button("capture"))
    {
        float3 vWo = spherical_to_cartesian_rad(mWo);
        mpOutColor->captureToFile(
            0,
            0,
            fmt::format(
                "C:/Projects/NNMat/data/IBL/{:05}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.exr",
                // mOutputDir,
                outputCount++,
                vWo.x,
                vWo.y,
                vWo.z,
                mEnvRotAngle.x / 360.0f,
                mEnvRotAngle.y / 360.0f,
                mEnvRotAngle.z / 360.0f
            ),
            Bitmap::FileFormat::ExrFile,
            Bitmap::ExportFlags::Uncompressed
        );
    }
    if (mOutputing)
    {
        float3 vWo = spherical_to_cartesian_rad(mWo);
        mpOutColor->captureToFile(
            0,
            0,
            fmt::format(
                "C:/Projects/NNMat/data/IBL/{}/{:05}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.exr",
                mOutputDir,
                outputCount++,
                vWo.x,
                vWo.y,
                vWo.z,
                mEnvRotAngle.x / 360.0f,
                mEnvRotAngle.y / 360.0f,
                mEnvRotAngle.z / 360.0f
            ),
            Bitmap::FileFormat::ExrFile,
            Bitmap::ExportFlags::Lossy
        );

        mOutputStep = mOutputStep % 5;

        if (mOutputStep == 0)
        {
            mWo.y += 1 / mOutputInterval.y * 2.0f * (float)M_PI;
            if (mWo.y >= 2.0f * (float)M_PI)
            {
                mOutputStep += 1;
                // mWo.y = 1 / (mOutputInterval.y * 2) * 2.0f * (float)M_PI;
                mWo.y = 0;
            }
        }

        else if (mOutputStep == 1)
        {
            // mWo.x += 1 / mOutputInterval.x * (float)M_PI / 2.0f;
            mCosTheta -= 1 / mOutputInterval.x;
            mWo.x = acos(mCosTheta);
            if (mCosTheta <= 0)
            {
                mOutputStep += 1;
                // mCosTheta = 1 - 1 / (mOutputInterval.x * 2);
                mCosTheta = 1;
                mWo.x = acos(mCosTheta);
                // mWo.x = 1 / (mOutputInterval.x * 2) * (float)M_PI / 2.0f;
            }
            else
            {
                mOutputStep -= 1;
            }
        }
        else if (mOutputStep == 3)
        {
            mEnvRotAngle.x += 1 / mOutputInterval.z * 360;
            if (mEnvRotAngle.x >= 360)
            {
                mOutputStep += 1;
                // mEnvRotAngle.x = 1 / (mOutputInterval.z * 2) * 360;
                mEnvRotAngle.x = 0;
            }
            else
            {
                mOutputStep -= 1;
            }
        }
        else if (mOutputStep == 2)
        {
            mEnvRotAngle.y += 1 / mOutputInterval.z * 360;
            if (mEnvRotAngle.y >= 360)
            {
                // mOutputing = false;
                // mEnvRotAngle = float3(1 / (mOutputInterval.z * 2) * 360);
                mEnvRotAngle.y = 0;
                mOutputStep += 1;
                // mEnvRotAngle.y = 1 / (mOutputInterval.z * 2) * 360;
            }
            else
            {
                mOutputStep -= 1;
            }
        }
        else if (mOutputStep == 4)
        {
            mEnvRotAngle.z += 1 / mOutputInterval.z * 360;
            if (mEnvRotAngle.z >= 360)
            {
                mOutputing = false;
                // mEnvRotAngle = float3(1 / (mOutputInterval.z * 2) * 360);
                mEnvRotAngle = float3(0);
            }
            else
            {
                mOutputStep -= 1;
            }
        }
    }
    w.var("Output Interval", mOutputInterval);

    if (w.button("Start Output"))
    {
        mOutputing = true;
        // mWo.x = 1 / (mOutputInterval.x * 2) * (float)M_PI / 2.0f;
        // mCosTheta = 1 - 1 / (mOutputInterval.x * 2);
        mCosTheta = 1;
        mWo.x = acos(mCosTheta);
        // mWo.y = 1 / (mOutputInterval.y * 2) * 2.0f * (float)M_PI;
        mWo.y = 0;
        // mEnvRotAngle = float3(1 / (mOutputInterval.z * 2) * 360);
        mEnvRotAngle = float3(0);
    }
    if (w.button("Stop"))
    {
        mOutputing = false;
    }
    // if (mOutputing)
    // {

    // }

    w.text(fmt::format("Current output index {} / {}", outputCount, mOutputInterval.x * mOutputInterval.y * mOutputInterval.z));
}

bool IBLRender::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return false;
}

void IBLRender::onHotReload(HotReloadFlags reloaded)
{
    //
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Falcor Project Template";
    config.windowDesc.resizableWindow = true;

    config.windowDesc.width = 3000;
    config.windowDesc.height = 1500;
    // config.windowDesc.resizableWindow = true;
    config.windowDesc.enableVSync = false;
    // config.windowDesc.title = "Falcor Shader Toy";

    IBLRender project(config);
    return project.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
