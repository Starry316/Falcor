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
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    createTex(mpOutColor, pRenderContext->getDevice(), targetDim);

    auto var = mpDebugPass->getRootVar()["CB"];
    var["iResolution"] = Falcor::float2(width, height);

    mpDebugPass->getRootVar()["ouputColor"] = mpOutColor;
    // mpDebugPass->getRootVar()["ouputColor"] = mpOutColor;

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpDebugPass->getProgram(), mpDebugPass->getRootVar());
    mpDebugPass->execute(pRenderContext, targetDim.x, targetDim.y);
    mpPixelDebug->endFrame(pRenderContext);
    
    
    display(pRenderContext, pTargetFbo);
    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
    
    mFrames++;
}

void IBLRender::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    w.text("Hello from SampleAppTemplate");
    if (w.button("Click Here"))
    {
        msgBox("Info", "Now why would you do that?");
    }
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

    config.windowDesc.width = 1920;
    config.windowDesc.height = 1080;
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
