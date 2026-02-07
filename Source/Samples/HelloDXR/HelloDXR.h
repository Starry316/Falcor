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
#include "Core/Pass/RasterPass.h"
#include "Utils/Debug/PixelDebug.h"
#include "SSRDefs.slangh"
using namespace Falcor;

class HelloDXR : public SampleApp
{
public:
    HelloDXR(const SampleAppConfig& config);
    ~HelloDXR();

    void onLoad(RenderContext* pRenderContext) override;
    void onResize(uint32_t width, uint32_t height) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;
    bool onKeyEvent(const KeyboardEvent& keyEvent) override;
    bool onMouseEvent(const MouseEvent& mouseEvent) override;

private:
    void loadScene(const std::filesystem::path& path, const Fbo* pTargetFbo);
    void setPerFrameVars(const Fbo* pTargetFbo);
    void renderRaster(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void renderRT(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);
    void postprocess(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo);

    ref<Scene> mpScene;
    ref<Camera> mpCamera;

    ref<RasterPass> mpRasterPass;
    ref<ComputePass> mpSSRPass;
    ref<ComputePass> mpPostProcessPass;
    ref<Program> mpRaytraceProgram;
    ref<RtProgramVars> mpRtVars;
    ref<Texture> mpRtOut;
    ref<Texture> mpPostOut;
    ref<Texture> mpBillboards;
    ref<Texture> mpBillboardPosWs;
    ref<Texture> mpBillboardNormalWs;
    ref<Texture> mpBillboardColors;
    ref<Texture> mpReference;

    ref<Fbo> mpBillboardFbo;

    ref<Buffer> mpViewProjBuffer;
    ref<Buffer> mpViewProjInvBuffer;
    ref<Buffer> mpBillboardNormalBuffer;

    std::unique_ptr<PixelDebug> mpPixelDebug;

    bool mRayTrace = false;
    bool mUseDOF = false;
    bool mShowBillboard = false;
    bool mCreateBillboards = true;
    bool mShowDiff = false;
    bool mDebugMode = false;
    bool mUseSortedBillboards = true;
    bool mShowColor = false;

    // normalized theta/phi view angles for billboard creation
    float mViewTheta = 0.3f;
    float mViewPhi = 0.0f;
    float mInvalidThreshold = 0.005f;
    float mMaxInvalidThreshold = 0.8f;
    // float mThickness = 0.01f;
    float mThickness = 0.00001f;


    bool3 mViewRenderMask = bool3(true, true, true);

    uint mShowBillboardID = 0;
    uint mBillboardRenderID = 0;
    uint mFrameCount = 0;
    uint mTraceStepCount = 1000;

    uint mFwdBillboardCount = 3;
    uint mFwdBillboardRadius = 2;
    float mFwdScreenRadius = 2;

    uint32_t mSampleIndex = 0xdeadbeef;

    ref<Sampler> mpMaxSampler;

    uint mBillboardTraceCount = TOTAL_VIEWS;
};
