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
#include "ShaderToy.h"
#include <fstream>
FALCOR_EXPORT_D3D12_AGILITY_SDK

ShaderToy::ShaderToy(const SampleAppConfig& config) : SampleApp(config) {}

ShaderToy::~ShaderToy() {}

void ShaderToy::onLoad(RenderContext* pRenderContext)
{
    // create rasterizer state
    RasterizerState::Desc rsDesc;
    mpNoCullRastState = RasterizerState::create(rsDesc);

    // Depth test
    DepthStencilState::Desc dsDesc;
    dsDesc.setDepthEnabled(false);
    mpNoDepthDS = DepthStencilState::create(dsDesc);

    // Blend state
    BlendState::Desc blendDesc;
    mpOpaqueBS = BlendState::create(blendDesc);

    // Texture sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear).setMaxAnisotropy(8);
    mpLinearSampler = getDevice()->createSampler(samplerDesc);

    // Load shaders
    mpMainPass = FullScreenPass::create(getDevice(), "Samples/ShaderToy/Toy.ps.slang");

    mpTextureSynthesis = std::make_unique<TextureSynthesis>();

    // mpTextureSynthesis->readHFData("D:/textures/ubo/leather11.png", getDevice());
    mpTextureSynthesis->readHFData("D:/textures/synthetic/ganges_river_pebbles_disp_4k.png", getDevice());
    mpNBTF = std::make_unique<NBTF>(getDevice(), mNetName, true);




}

void ShaderToy::onResize(uint32_t width, uint32_t height)
{
    mAspectRatio = (float(width) / float(height));
}

void ShaderToy::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    auto var = mpMainPass->getRootVar()["ToyCB"];
    var["iResolution"] = float2(width, height);
    var["iGlobalTime"] = (float)getGlobalClock().getTime();
    var["gUVScaling"] = mUVScale;
    var["gSynthesis"] = mSynthesis;
    mpTextureSynthesis->bindHFData(mpMainPass->getRootVar()["ToyCB"]["hfData"]);
    mpNBTF->bindShaderData(mpMainPass->getRootVar()["ToyCB"]["nbtf"]);



    // run final pass
    mpMainPass->execute(pRenderContext, pTargetFbo);
}
void ShaderToy::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    w.text("Hello from SampleAppTemplate");
    w.slider("UV Scale", mUVScale, 0.0f, 10.0f);
    w.checkbox("Enable Synthesis", mSynthesis);
    if (w.button("Click Here"))
    {
        msgBox("Info", "Now why would you do that?");
    }
}
int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.width = 1280;
    config.windowDesc.height = 720;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.enableVSync = true;
    config.windowDesc.title = "Falcor Shader Toy";

    ShaderToy shaderToy(config);
    return shaderToy.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
