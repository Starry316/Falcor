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
#pragma once
#include "Falcor.h"
#include "Core/SampleApp.h"
#include "Core/Pass/FullScreenPass.h"
#include "Core/Pass/ComputePass.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Utils/Texture/Synthesis.h"

#include "Utils/Debug/PixelDebug.h"
#include "Utils/Neural/NNMat.h"
using namespace Falcor;

class IBLRender : public SampleApp
{
public:
    IBLRender(const SampleAppConfig& config);
    ~IBLRender();

    void onLoad(RenderContext* pRenderContext) override;
    void onShutdown() override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return mpPixelDebug->onMouseEvent(mouseEvent); }
    void onHotReload(HotReloadFlags reloaded) override;
    void display(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void render(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

private:
    void loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo);

    std::string mOutputDir = "leather11_45_3D";

    std::string mNNMatName = "fabric12_XYZ_BTFNetXYZHU72x2";
    std::string mNNIBLName = "fabric12_45_IBL_BTFNetIBL2x2";

    // std::string mNNMatName = "leather11_XYZ_BTFNetXYZHU72x2";
    // std::string mNNIBLName = "leather11_45_IBL_BTFNetIBL2x2";

    std::string mEnvmapName =  "45_1k_downsampled.exr";

    std::unique_ptr<PixelDebug> mpPixelDebug;
    ref<FullScreenPass> mpDisplayPass;
    ref<ComputePass> mpDebugPass;
    uint mFrames = 1;
    uint outputCount = 0;
    int mSampleNum = 4;
    bool mOutputing = false;
    bool mShowIBL = false;
    bool mDirty = true;

    uint mOutputStep = 0;
    std::string mProjectPath = getProjectDirectory().string();
    ref<Texture> mpOutColor;
    ref<Texture> mpOutColorRef;
    float2 mWo = {0.0f, 0.0f};
    float2 mWi = {0.0f, 0.0f};
    float3 mEnvRotAngle = float3(0.0f, 0.0f, 0.0f);
    std::shared_ptr<NNMat> mpNNMat;
    std::shared_ptr<NNMat> mpNNMatIBL;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
    ref<SampleGenerator> mpSampleGenerator;
    ref<EnvMap> mpEnvMap;

    ref<Scene> mpScene;
    ref<Camera> mpCamera;
    float3 mOutputInterval = float3(10, 15, 20);
    float mCosTheta = 0;

    // ref<Program> mpRaytraceProgram;
    // ref<RtProgramVars> mpRtVars;
    // ref<Texture> mpRtOut;
    // bool mRayTrace = true;


};
