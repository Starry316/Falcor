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
#include "Utils/Texture/Synthesis.h"

FALCOR_EXPORT_D3D12_AGILITY_SDK

uint32_t mSampleGuiWidth = 250;
uint32_t mSampleGuiHeight = 200;
uint32_t mSampleGuiPositionX = 20;
uint32_t mSampleGuiPositionY = 40;

SynthesisSample::SynthesisSample(const SampleAppConfig& config) : SampleApp(config)
{
    // auto textureSynthesis = std::make_unique<TextureSynthesis>(getDevice());
    auto textureSynthesis = std::make_unique<TextureSynthesis>();
}

SynthesisSample::~SynthesisSample()
{
    //
}

void SynthesisSample::onLoad(RenderContext* pRenderContext)
{
    //
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
    const float4 clearColor(0.38f, 0.52f, 0.10f, 1);
    pRenderContext->clearFbo(pTargetFbo.get(), clearColor, 1.0f, 0, FboAttachmentType::All);
}

void SynthesisSample::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200});
    renderGlobalUI(pGui);
    w.text("Hello from SampleAppTemplate");
    if (w.button("Click Here"))
    {
        msgBox("Info", "Now why would you do that?");
    }
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

    SynthesisSample project(config);
    return project.run();
}
// using namespace std;
// double getMaxExpectedProfit(int N, std::vector<int> V, int C, double S) {
//   // Write your code here
//   double expectedProfit = 0.0;
//   double packageValSum = 0.0;
//   double nonStolenProb = 1.0;
//   double lastExpectationVal = 0.0;
//   double curExpectationVal = 0.0;
//   for(int i = 0; i < V.size(); i++){
//     lastExpectationVal = curExpectationVal;
//     packageValSum += V[i];
//     nonStolenProb *= 1.0 - S;
//     curExpectationVal= nonStolenProb * packageValSum;
//     if(curExpectationVal < lastExpectationVal  && lastExpectationVal > C){
//       expectedProfit += lastExpectationVal - C;
//       lastExpectationVal = 0.0;
//       curExpectationVal = 0.0;
//       packageValSum = 0.0;
//       nonStolenProb = 1.0;
//       i--;
//     }
//   }
//   if( curExpectationVal > C) expectedProfit += lastExpectationVal - C;

//   return expectedProfit;
// }
int main(int argc, char** argv)
{
// std::vector<int> v = {10, 2, 8, 6, 4};
//  printf("%f",getMaxExpectedProfit(5, v, 5, 0));


// Write any include statements here






    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
