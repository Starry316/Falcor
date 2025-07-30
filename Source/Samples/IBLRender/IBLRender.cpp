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

void IBLRender::onLoad(RenderContext* pRenderContext)
{
    // Load shaders
    mpDebugPass = ComputePass::create(getDevice(), "Samples/IBLRender/render.cs.slang", "csMain");
    mpPixelDebug = std::make_unique<PixelDebug>(getDevice());
    mpDisplayPass = FullScreenPass::create(getDevice(), "Samples/IBLRender/display.ps.slang");
    mpNNMat = std::make_shared<NNMat>(getDevice(), "leather11_XYZ_BTFNetXYZHU72x2", 0, 1);
    mpNNMatIBL = std::make_shared<NNMat>(getDevice(), "leather11_45_IBL_BTFNetIBL2x2", 0, 0);
    mpEnvMap = EnvMap::createFromFile(
        getDevice(), fmt::format("{}/media/neural_materials/scene/envmap/{}", mProjectPath, "45_1k_downsampled.exr")
    );
    mpEnvMapSampler = std::make_unique<EnvMapSampler>(getDevice(), mpEnvMap);
    mpSampleGenerator = SampleGenerator::create(getDevice(), SAMPLE_GENERATOR_UNIFORM);
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

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpDisplayPass->getProgram(), mpDisplayPass->getRootVar());
    mpDisplayPass->execute(pRenderContext, pTargetFbo);
    mpPixelDebug->endFrame(pRenderContext);
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

    // mpDebugPass->getProgram()->addDefines(mpSampleGenerator->getDefines());

    auto var = mpDebugPass->getRootVar();
    var["CB"]["iResolution"] = float2(outputDim);
    var["CB"]["gWo"] = mWo;
    var["CB"]["gWi"] = mWi;
    var["CB"]["gSampleNum"] = mSampleNum;
    var["CB"]["gRotAngles"] = mEnvRotAngle;
    var["CB"]["gShowIBL"] = mShowIBL;

    var["ouputColor"] = mpOutColor;
    mpNNMat->bindShaderData(var["CB"]["nnmat"]);
    mpNNMatIBL->bindShaderData(var["CB"]["nnmatIBL"]);
    mpEnvMap->bindShaderData(var["CB"]["envMap"]);
    // mpEnvMapSampler->bindShaderData(var["CB"]["envMapSampler"]);
    mpSampleGenerator->bindShaderData(var);

    // mpPixelDebug->beginFrame(pRenderContext, targetDim);
    // mpPixelDebug->prepareProgram(mpDebugPass->getProgram(), mpDebugPass->getRootVar());
    if(mDirty||mOutputing)
        mpDebugPass->execute(pRenderContext, outputDim.x, outputDim.y);
    // mpPixelDebug->endFrame(pRenderContext);

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
    mDirty |= w.slider("sampleNum", mSampleNum, 1, 64);

    mDirty |= w.slider("Env rot X", mEnvRotAngle.x, 0.0f, 360.0f);
    if (w.button("X -", true))
    {
        mEnvRotAngle.x -= 5;
    }
    if (w.button("X +", true))
    {
        mEnvRotAngle.x += 5;
    }
    w.slider("Env rot Y", mEnvRotAngle.y, 0.0f, 360.0f);
    if (w.button("Y -", true))
    {
        mEnvRotAngle.y -= 5;
    }
    if (w.button("Y +", true))
    {
        mEnvRotAngle.y += 5;
    }
    w.slider("Env rot Z", mEnvRotAngle.z, 0.0f, 360.0f);
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
            mWo.x += 0.05f * (float)M_PI / 2.0f;
            if (mWo.x > (float)M_PI / 2.0f)
            {
                mOutputStep += 1;
                mWo.x = 0.025f * (float)M_PI / 2.0f;
            }
        }
        else if (mOutputStep == 1)
        {
            mWo.y += 0.02f * 2.0f * (float)M_PI;
            if (mWo.y > 2.0f * (float)M_PI)
            {
                mOutputStep += 1;
                mWo.y = 0.01f * 2.0f * (float)M_PI;
            }
            else
            {
                mOutputStep -= 1;
            }
        }
        else if (mOutputStep == 2)
        {
            mEnvRotAngle.x += 0.02f * 360;
            if (mEnvRotAngle.x > 360)
            {
                mOutputing = false;
                mEnvRotAngle = float3(0.01f * 360);
                // mOutputStep += 1;
                // mEnvRotAngle.x = 0.05f * 360;
            }
            else
            {
                mOutputStep -= 1;
            }
        }
        else if (mOutputStep == 3)
        {
            mEnvRotAngle.y += 0.1f * 360;
            if (mEnvRotAngle.y > 360)
            {
                mOutputStep += 1;
                mEnvRotAngle.y = 0.05f * 360;
            }
            else
            {
                mOutputStep -= 1;
            }
        }
        else if (mOutputStep == 4)
        {
            mEnvRotAngle.z += 0.1f * 360;
            if (mEnvRotAngle.z > 360)
            {
                mOutputing = false;
                mEnvRotAngle = float3(0.05f * 360);
            }
            else
            {
                mOutputStep -= 1;
            }
        }
    }

    if (w.button("Start Output"))
    {
        mOutputing = true;
        mWo.x = 0.025f * (float)M_PI / 2.0f;
        mWo.y = 0.01f * 2.0f * (float)M_PI;
        mEnvRotAngle = float3(0.01f * 360);
    }
    if (w.button("Stop"))
    {
        mOutputing = false;
    }
    // if (mOutputing)
    // {

    // }

    w.text(fmt::format("Current output index {}", outputCount));
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

    config.windowDesc.width = 1500;
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
