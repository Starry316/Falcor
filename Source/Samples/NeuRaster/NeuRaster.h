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
#include "Core/Program/DefineList.h"
#include "Core/API/GpuTimer.h"

using namespace Falcor;

/**
 * NeuRaster: a full-screen NeuLobes neural-inference visualizer / performance test.
 *
 * Draws a full-screen quad whose four corners are each assigned a probe (UI-selectable). The vertex
 * shader reads each corner probe's MLP weights and computes its neural feature, and the rasterizer
 * hardware-interpolates both across the triangle - the same barycentric blend the PTTest render pass
 * does manually in the pixel shader. The pixel shader then reconstructs the blended weights and runs
 * the exact 3-layer NeuLobes MLP, so the screen shows a smooth blend of the four probes' predictions.
 *
 * Vertex-shader interpolant arrays cannot be sized dynamically, so the MLP dimensions are baked into
 * the shader as NEU_* defines. Those defines are derived from the loaded model's manifest at program
 * creation time (see onLoad), so the shader is always sized to fit the model with no hand-edited
 * constants. loadModel() logs the dimensions, and onLoad warns if they exceed the interpolant budget.
 */
class NeuRaster : public SampleApp
{
public:
    NeuRaster(const SampleAppConfig& config);
    ~NeuRaster();

    void onLoad(RenderContext* pRenderContext) override;
    void onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo) override;
    void onGuiRender(Gui* pGui) override;

private:
    /// Loads the neural model (manifest.json + weight binaries) the same way the render pass does.
    void loadModel(const std::string& dir);
    /// Builds the NEU_* shader defines from the loaded model's dimensions (and validates the budget).
    DefineList buildShaderDefines();
    /// (Re)creates the raster pass for the current blend mode (mVsInterp) by recompiling with defines.
    void createRasterPass();
    /// Builds the full-screen quad (position + per-vertex corner index).
    void buildGeometry();
    /// Binds the model buffers and query parameters to the raster pass root var.
    void bindModel();
    /// Query direction (unit vector) built from the azimuth/elevation UI sliders.
    float3 computeInputDir() const;

    ref<RasterPass> mpRasterPass;
    ref<Vao> mpVao;
    ref<GpuTimer> mpTimer;

    // Perf measurement. GPU time is tracked per blend mode so the two can be compared side by side.
    double mLastGpuMs = 0.0;
    double mGpuMsVsInterp = 0.0; ///< Last GPU draw time in vertex-shader-interpolation mode.
    double mGpuMsPsBlend = 0.0;  ///< Last GPU draw time in fragment-shader-blend mode.
    bool mTimedOnce = false;
    uint32_t mIterations = 1; ///< Inference repeats per pixel (scales throughput work).

    // Blend strategy: true = interpolate weights/feature in the vertex shader (NEU_VS_INTERP=1),
    // false = fetch + blend the corner probes in the fragment shader (NEU_VS_INTERP=0).
    bool mVsInterp = true;
    bool mVsInterpFeasible = true; ///< False if the model's weights exceed the VS interpolant budget.
    uint32_t mVsInterpVec4 = 0;    ///< float4 VS-output interpolants the VS-interp mode would need.

    // Query parameters (UI-controlled).
    uint32_t mCornerProbe[4] = {0, 0, 0, 0}; ///< Probe pool index for each of the quad's 4 corners.
    float mDirAzimuth = 0.0f;                ///< Azimuth in degrees [0,360).
    float mDirElevation = 0.0f;              ///< Elevation in degrees [-90,90].

    // Dimensions the shader was compiled for (for the UI + logs).
    std::string mShaderDimsStatus = "Model not loaded.";

    // Loaded neural model (kept resident; same tensors/layout as the render pass).
    std::string mModelDir = "C:/projects/neulobes/outputs/neulobes_multi_bin";
    bool mModelLoaded = false;

    // Hyperparameters (read from manifest.json, fall back to the reference config on missing keys).
    uint32_t mPeBands = 2;
    uint32_t mRes = 6;
    uint32_t mRank = 2;
    uint32_t mPlaneDim = 4;
    uint32_t mHiddenDim = 4;
    uint32_t mNumMLPs = 1;
    uint32_t mInputDim = 16;
    uint32_t mOutputDim = 3;
    uint32_t mInBlk[3] = {};
    uint32_t mOutBlk[3] = {};

    ref<Buffer> mpVecX;
    ref<Buffer> mpMatYZ;
    ref<Buffer> mpW[3];
    ref<Buffer> mpB[3];
};
