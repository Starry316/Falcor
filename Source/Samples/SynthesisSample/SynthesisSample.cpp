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
#include "SynthesisSample.h"

FALCOR_EXPORT_D3D12_AGILITY_SDK

uint32_t mSampleGuiWidth = 250;
uint32_t mSampleGuiHeight = 200;
uint32_t mSampleGuiPositionX = 20;
uint32_t mSampleGuiPositionY = 40;

uint32_t mOutputImgSize = 128;
uint32_t mSampleSize = 4000;

SynthesisSample::SynthesisSample(const SampleAppConfig& config) : SampleApp(config) {}

SynthesisSample::~SynthesisSample() {}

void SynthesisSample::onLoad(RenderContext* pRenderContext)
{
    mpSampleGenerator = SampleGenerator::create(getDevice(), SAMPLE_GENERATOR_UNIFORM);

    DefineList defines;
    defines.add(mpSampleGenerator->getDefines());
    mpGenNDFPass = ComputePass::create(getDevice(), "Samples/SynthesisSample/GenNDF.cs.slang", "csMain", defines);
    mpSampleGenerator->bindShaderData(mpGenNDFPass->getRootVar());


    mpDisplayPass = FullScreenPass::create(getDevice(), "Samples/SynthesisSample/Display.ps.slang");
    mpWriteTexPass = ComputePass::create(getDevice(), "Samples/SynthesisSample/WriteTex.cs.slang", "csMain");
    ResourceBindFlags flag = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mpHF = Texture::createFromFile(getDevice(), "D:/Scene/textures/isotropic_positive.hdr", false, false, flag);

    mpSamplesBuffer = getDevice()->createBuffer(
        mSampleSize * mSampleSize * 1 * sizeof(int),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr
    );
    mpImgBuffer = getDevice()->createBuffer(
        mOutputImgSize * mOutputImgSize * 1 * sizeof(float),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        nullptr
    );
    mImgBuffer = std::vector<float>(mOutputImgSize * mOutputImgSize * 1, 0.0f);
    mOutputBuffer = std::vector<int>(mSampleSize * mSampleSize * 1, 0.0f);
    mpOutput = getDevice()->createTexture2D(mOutputImgSize, mOutputImgSize, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);







}

void SynthesisSample::onShutdown()
{
    //
}

void SynthesisSample::onResize(uint32_t width, uint32_t height)
{
    //
}

void SynthesisSample::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    const float4 clearColor(0.0f, 0.0f, 0.0f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);

    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    // ================== GenNDF ==================
    mpGenNDFPass->getRootVar()["CB"]["iResolution"] = float2(1000, 1000);
    mpGenNDFPass->getRootVar()["gHF"] = mpHF;
    mpGenNDFPass->getRootVar()["outputSamples"] = mpSamplesBuffer;
    mpGenNDFPass->getRootVar()["ouputColor"] = mpOutput;

    mpGenNDFPass->getRootVar()["CB"]["mu_p"] = mMu_p;
    mpGenNDFPass->getRootVar()["CB"]["sigma_p"] = mSigma_p;
    mpGenNDFPass->getRootVar()["CB"]["texelWidth"] = 1.0f;
    // mpGenNDFPass->getRootVar()["CB"]["gSampleSize"] = mSampleSize;
    mpGenNDFPass->getRootVar()["CB"]["gImgSize"] = (float)mOutputImgSize;

    mpGenNDFPass->execute(pRenderContext, mSampleSize, mSampleSize);

    // ================== Splat ==================
    mpSamplesBuffer->getBlob(mOutputBuffer.data(), 0, mOutputBuffer.size() * sizeof(float));

    for (size_t i = 0; i < mImgBuffer.size(); i++)
    {
        mImgBuffer[i] = 0;
    }

    for (size_t i = 0; i < mSampleSize * mSampleSize; i++)
    {
        // logInfo("{}", mOutputBuffer[i]);
        mImgBuffer[mOutputBuffer[i]] += 1.0f / mFactor;
    }

    // for (size_t i = 0; i < 128 * 128; i++)
    // {

    //     mImgBuffer[i] = 1.0f ;
    // }

    mpImgBuffer->setBlob(mImgBuffer.data(), 0, mImgBuffer.size() * sizeof(float));



    // ================== WriteTex ==================
    mpWriteTexPass->getRootVar()["CB"]["iResolution"] = float2(mOutputImgSize, mOutputImgSize);
    mpWriteTexPass->getRootVar()["ouputColor"] = mpOutput;
    mpWriteTexPass->getRootVar()["imgBuffer"] = mpImgBuffer;
    mpWriteTexPass->execute(pRenderContext, mOutputImgSize, mOutputImgSize);



    // ================== Display ==================
    mpDisplayPass->getRootVar()["ouputColor"] = mpOutput;
    mpDisplayPass->execute(pRenderContext, pTargetFbo);
}

void SynthesisSample::onGuiRender(Gui* pGui)
{

    if(mGenNDF){
        mpOutput->captureToFile(0, 0, fmt::format("D:/NDF/{}_{}_{}.exr", mMu_p.x, mMu_p.y, mSigma_p),Bitmap::FileFormat::ExrFile);

        mSigma_p += 1.0f;
        if(mSigma_p > 16.0f){
            mSigma_p = 2.0f;
            mMu_p.x += 1.0f;
            if(mMu_p.x > 20.0f){
                mMu_p.x = 0.0f;
                mMu_p.y += 1.0f;
                if(mMu_p.y > 20.0f){
                    mGenNDF = false;
                }
            }
        }

    }

    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    w.text("Hello from SampleAppTemplate");
    if (w.button("Click Here"))
    {
        mpOutput->captureToFile(0, 0, fmt::format("D:/{}_{}_{}.exr",mMu_p.x,mMu_p.y,mSigma_p ),Bitmap::FileFormat::ExrFile);
    }

    if (w.button("Start GenNDF"))
    {
        mGenNDF = true;
    }
    if (w.button("Stop"))
    {
        mGenNDF = false;
    }


    w.slider("factor", mFactor, 1.0f, 10000.0f);
    w.slider("mu_p.x", mMu_p.x, 0.0f, 100.0f);
    w.slider("mu_p.y", mMu_p.y, 0.0f, 100.0f);
    w.slider("sigma_p", mSigma_p, 1.0f, 64.0f);



}

bool SynthesisSample::onKeyEvent(const KeyboardEvent& keyEvent)
{
    return false;
}

bool SynthesisSample::onMouseEvent(const MouseEvent& mouseEvent)
{
    return false;
}

void SynthesisSample::onHotReload(HotReloadFlags reloaded)
{
    //
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.title = "Falcor Project Template";
    config.windowDesc.resizableWindow = true;
    config.windowDesc.width = 1000;
    config.windowDesc.height = 1000;
    SynthesisSample project(config);
    return project.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
