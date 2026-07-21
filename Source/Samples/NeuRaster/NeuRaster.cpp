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
#include "NeuRaster.h"
#include "Utils/Math/Matrix.h"
#include "Core/Program/DefineList.h"
#include <nlohmann/json.hpp>
#include <fstream>

FALCOR_EXPORT_D3D12_AGILITY_SDK

namespace
{
const char kShaderFile[] = "Samples/NeuRaster/NeuRaster.3d.slang";

// Full-screen quad = 2 triangles = 6 vertices.
const uint32_t kQuadVerts = 6;

/// Vertex format for the full-screen quad: clip-space position, corner index (0..3), and the
/// barycentric basis of this vertex within its triangle (used by the fragment-shader-blend path).
struct QuadVertex
{
    float2 pos;
    uint32_t corner;
    float3 bary;
};

/// Reads a little-endian float32 binary file into a vector (empty on failure).
std::vector<float> readFloatBinary(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        logWarning("[NeuRaster] Unable to open file {}", path.string());
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (size > 0 && !file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        logWarning("[NeuRaster] Error reading file {}", path.string());
        return {};
    }
    return buffer;
}
} // namespace

NeuRaster::NeuRaster(const SampleAppConfig& config) : SampleApp(config) {}

NeuRaster::~NeuRaster() {}

void NeuRaster::loadModel(const std::string& dir)
{
    const std::filesystem::path root(dir);

    // Parse manifest.json for hyperparameters; fall back to the reference config on any missing key.
    bool haveLayerBlocks = false;
    std::ifstream ifs(root / "manifest.json");
    if (ifs)
    {
        try
        {
            nlohmann::json j = nlohmann::json::parse(ifs, nullptr, true, true);

            // Pool size is top-level.
            mNumMLPs = j.value("numMLPs", mNumMLPs);

            // Hyperparameters are nested under "hyperparams" (fall back to the root object).
            const nlohmann::json& hp = j.contains("hyperparams") ? j["hyperparams"] : j;
            mPeBands = hp.value("peBands", mPeBands);
            mRank = hp.value("rank", mRank);
            mPlaneDim = hp.value("planeDim", mPlaneDim);
            mHiddenDim = hp.value("hiddenDim", mHiddenDim);
            mInputDim = hp.value("inputDim", mInputDim);
            // Feature-plane resolution is called "planeRes" (also accept "res").
            if (hp.contains("res"))
                mRes = hp["res"].get<uint32_t>();
            else
                mRes = hp.value("planeRes", mRes);

            // Per-layer 4x4 block counts, read directly from manifest.layers when available.
            if (j.contains("layers") && j["layers"].is_array() && j["layers"].size() == 3)
            {
                for (int l = 0; l < 3; ++l)
                {
                    mInBlk[l] = j["layers"][l].value("nin_blocks", 0u);
                    mOutBlk[l] = j["layers"][l].value("nout_blocks", 0u);
                }
                haveLayerBlocks = true;
            }
        }
        catch (const std::exception& e)
        {
            logWarning("[NeuRaster] Failed to parse manifest.json ({}). Using reference config.", e.what());
        }
    }
    else
    {
        logWarning("[NeuRaster] manifest.json not found in {}. Using reference config.", dir);
    }

    // Fall back to deriving per-layer block counts (MLP dims: inputDim -> hiddenDim -> hiddenDim -> outputDim).
    if (!haveLayerBlocks)
    {
        const uint32_t dims[4] = {mInputDim, mHiddenDim, mHiddenDim, mOutputDim};
        auto blk = [](uint32_t n) { return (n + 3u) / 4u; };
        for (int l = 0; l < 3; ++l)
        {
            mInBlk[l] = blk(dims[l]);
            mOutBlk[l] = blk(dims[l + 1]);
        }
    }

    auto readTensor = [&](const std::string& name) { return readFloatBinary(root / name); };

    // Feature tensors (raw float lists uploaded as-is; indexed manually in the shader).
    std::vector<float> vecx = readTensor("fp_vecx.bin");
    std::vector<float> matyz = readTensor("fp_matyz.bin");
    if (vecx.empty() || matyz.empty())
    {
        logWarning("[NeuRaster] Missing feature tensors in {}. Model not loaded.", dir);
        return;
    }

    const ResourceBindFlags flags = ResourceBindFlags::ShaderResource;
    mpVecX = getDevice()->createBuffer(vecx.size() * sizeof(float), flags, MemoryType::DeviceLocal, vecx.data());
    mpMatYZ = getDevice()->createBuffer(matyz.size() * sizeof(float), flags, MemoryType::DeviceLocal, matyz.data());

    // Texture-backed feature planes: width=res, height=numMLPs*rank (one row per (probe,rank) pair).
    // Requires planeDim==4 (channels packed into RGBA) and dims within the HW 2D limit. The raw
    // tensors are already row-major in this exact layout, so they upload with no repacking.
    mpVecXTex = nullptr;
    mpMatYZTex = nullptr;
    mpFeatSampler = nullptr;
    const uint32_t texRows = mNumMLPs * mRank;
    mTexturesFeasible = (mPlaneDim == 4) && (texRows > 0) && (mRes > 0) && (texRows <= 16384) && (mRes <= 16384) &&
                        (vecx.size() == (size_t)texRows * mRes) && (matyz.size() == (size_t)texRows * mRes * 4);
    if (mTexturesFeasible)
    {
        mpVecXTex = getDevice()->createTexture2D(mRes, texRows, ResourceFormat::R32Float, 1, 1, vecx.data(), flags);
        mpMatYZTex = getDevice()->createTexture2D(mRes, texRows, ResourceFormat::RGBA32Float, 1, 1, matyz.data(), flags);

        Sampler::Desc sd;
        sd.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Point);
        sd.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
        mpFeatSampler = getDevice()->createSampler(sd);
    }
    else
    {
        mUseTextures = false; // Force the buffer path if textures are not usable for this model.
    }

    // Weights/biases: repack each layer's raw floats into mul-ready float4x4 / float4 blocks (as in the pass).
    for (int l = 0; l < 3; ++l)
    {
        std::vector<float> w = readTensor(fmt::format("w{}.bin", l));
        std::vector<float> b = readTensor(fmt::format("b{}.bin", l));
        if (w.empty() || b.empty())
        {
            logWarning("[NeuRaster] Missing weights/bias for layer {} in {}. Model not loaded.", l, dir);
            return;
        }

        const size_t numMat = w.size() / 16;
        std::vector<float4x4> mats(numMat);
        for (size_t i = 0; i < numMat; ++i)
            mats[i] = math::matrixFromCoefficients<float, 4, 4>(w.data() + i * 16);

        mpW[l] = getDevice()->createBuffer(numMat * sizeof(float4x4), flags, MemoryType::DeviceLocal, mats.data());
        mpB[l] = getDevice()->createBuffer(b.size() * sizeof(float), flags, MemoryType::DeviceLocal, b.data());
    }

    if (mNumMLPs > 0)
    {
        const uint32_t maxV = mNumMLPs - 1;
        for (uint32_t& c : mCornerProbe)
            c = std::min(c, maxV);
    }
    mModelLoaded = true;
    logInfo(
        "[NeuRaster] Loaded model from {} (peBands={}, res={}, rank={}, planeDim={}, hiddenDim={}, inputDim={}, numMLPs={}).",
        dir, mPeBands, mRes, mRank, mPlaneDim, mHiddenDim, mInputDim, mNumMLPs
    );

    // VS-interpolation interpolant budget (worst case: both weights and feature interpolated in the VS).
    // The hardware allows ~32 float4 outputs, so large models can only run the pixel-shader-blend mode.
    const uint32_t weightsVec4 = 4 * (mInBlk[0] * mOutBlk[0] + mInBlk[1] * mOutBlk[1] + mInBlk[2] * mOutBlk[2]);
    const uint32_t biasVec4 = mOutBlk[0] + mOutBlk[1] + mOutBlk[2];
    const uint32_t featVec4 = (mPlaneDim + 3) / 4;
    mVsInterpVec4 = weightsVec4 + biasVec4 + featVec4 + 1 /*bary*/;
    mVsInterpFeasible = (mVsInterpVec4 <= 30); // reserve registers for SV_Position + margin
    if (!mVsInterpFeasible)
    {
        logWarning(
            "[NeuRaster] VS-interpolation would need {} float4 interpolants (hardware limit ~32); only the "
            "fragment-shader-blend mode is available for this model.",
            mVsInterpVec4
        );
    }
}

DefineList NeuRaster::buildShaderDefines()
{
    // The shader's interpolant array sizes come from these NEU_* defines. Deriving them from the
    // model's manifest dimensions means the shader is always sized to fit whatever model is loaded.
    DefineList defines;
    defines.add("NEU_PE_BANDS", std::to_string(mPeBands));
    defines.add("NEU_PLANE_DIM", std::to_string(mPlaneDim));
    defines.add("NEU_L0_IN_BLK", std::to_string(std::max(mInBlk[0], 1u)));
    defines.add("NEU_L0_OUT_BLK", std::to_string(std::max(mOutBlk[0], 1u)));
    defines.add("NEU_L1_IN_BLK", std::to_string(std::max(mInBlk[1], 1u)));
    defines.add("NEU_L1_OUT_BLK", std::to_string(std::max(mOutBlk[1], 1u)));
    defines.add("NEU_L2_IN_BLK", std::to_string(std::max(mInBlk[2], 1u)));
    defines.add("NEU_L2_OUT_BLK", std::to_string(std::max(mOutBlk[2], 1u)));
    defines.add("NEU_VS_INTERP", mVsInterp ? "1" : "0");
    defines.add("NEU_FEAT_VS_INTERP", mFeatVsInterp ? "1" : "0");
    defines.add("NEU_USE_TEXTURES", mUseTextures ? "1" : "0");

    const std::string dims = fmt::format(
        "peBands={} planeDim={} inBlk=[{},{},{}] outBlk=[{},{},{}]",
        mPeBands, mPlaneDim, mInBlk[0], mInBlk[1], mInBlk[2], mOutBlk[0], mOutBlk[1], mOutBlk[2]
    );
    mShaderDimsStatus = fmt::format(
        "Model: {}. VS-interp needs {} float4 interpolants ({}).",
        dims, mVsInterpVec4, mVsInterpFeasible ? "fits" : "> ~32, PS-blend only"
    );
    return defines;
}

void NeuRaster::buildGeometry()
{
    // Full-screen quad corners in clip space; two triangles (BL,BR,TR) and (BL,TR,TL). Each vertex
    // carries its corner index (0..3) and its barycentric basis (e0/e1/e2 by position within the
    // triangle), so both blend paths have what they need.
    const float2 cornerPos[4] = {float2(-1, -1), float2(1, -1), float2(1, 1), float2(-1, 1)};
    const uint32_t triCorner[kQuadVerts] = {0, 1, 2, 0, 2, 3};
    const float3 basis[3] = {float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1)};

    QuadVertex verts[kQuadVerts];
    for (uint32_t v = 0; v < kQuadVerts; ++v)
    {
        verts[v].pos = cornerPos[triCorner[v]];
        verts[v].corner = triCorner[v];
        verts[v].bary = basis[v % 3];
    }

    auto vb = getDevice()->createBuffer(sizeof(verts), ResourceBindFlags::Vertex, MemoryType::DeviceLocal, verts);

    // Interleaved layout: POSITION (float2) + CORNER (uint) + BARY (float3). Digit-free semantic names.
    auto bufferLayout = VertexBufferLayout::create();
    bufferLayout->addElement("POSITION", offsetof(QuadVertex, pos), ResourceFormat::RG32Float, 1, 0);
    bufferLayout->addElement("CORNER", offsetof(QuadVertex, corner), ResourceFormat::R32Uint, 1, 1);
    bufferLayout->addElement("BARY", offsetof(QuadVertex, bary), ResourceFormat::RGB32Float, 1, 2);
    auto layout = VertexLayout::create();
    layout->addBufferLayout(0, bufferLayout);

    mpVao = Vao::create(Vao::Topology::TriangleList, layout, {vb});
}

float3 NeuRaster::computeInputDir() const
{
    const float kDeg2Rad = 3.14159265358979323846f / 180.f;
    const float az = mDirAzimuth * kDeg2Rad;
    const float el = mDirElevation * kDeg2Rad;
    const float ce = std::cos(el);
    return normalize(float3(ce * std::cos(az), std::sin(el), ce * std::sin(az)));
}

void NeuRaster::bindModel()
{
    auto var = mpRasterPass->getRootVar();

    var["NeuCB"]["gInputDir"] = computeInputDir();
    var["NeuCB"]["gIterations"] = mIterations;
    var["NeuCB"]["gCornerProbe"] = uint4(mCornerProbe[0], mCornerProbe[1], mCornerProbe[2], mCornerProbe[3]);
    var["NeuCB"]["gRes"] = mRes;
    var["NeuCB"]["gRank"] = mRank;

    if (mUseTextures)
    {
        var["gNeuVecXTex"] = mpVecXTex;
        var["gNeuMatYZTex"] = mpMatYZTex;
        var["gNeuFeatSampler"] = mpFeatSampler;
    }
    else
    {
        var["gNeuVecX"] = mpVecX;
        var["gNeuMatYZ"] = mpMatYZ;
    }
    var["gNeuW0"] = mpW[0];
    var["gNeuW1"] = mpW[1];
    var["gNeuW2"] = mpW[2];
    var["gNeuB0"] = mpB[0];
    var["gNeuB1"] = mpB[1];
    var["gNeuB2"] = mpB[2];
}

void NeuRaster::createRasterPass()
{
    DefineList defines = buildShaderDefines();
    mpRasterPass = RasterPass::create(getDevice(), kShaderFile, "vsMain", "psMain", defines);
}

void NeuRaster::onLoad(RenderContext* pRenderContext)
{
    loadModel(mModelDir);
    buildGeometry();

    // The full weight set can't be interpolated for large models; fall back to fragment-shader blend.
    if (!mVsInterpFeasible)
        mVsInterp = false;

    // Bake the loaded model's dimensions + current blend mode into the shader as NEU_* defines.
    createRasterPass();
    mpTimer = GpuTimer::create(getDevice());
}

void NeuRaster::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // Read the previous frame's resolved GPU time first (avoids a same-frame stall).
    if (mpTimer && mTimedOnce)
    {
        mLastGpuMs = mpTimer->getElapsedTime();
        // Attribute the time to the current mode (mode changes are infrequent, so this is accurate).
        (mVsInterp ? mGpuMsVsInterp : mGpuMsPsBlend) = mLastGpuMs;
    }

    pRenderContext->clearFbo(pTargetFbo.get(), float4(0.f), 1.f, 0);

    if (mModelLoaded && mpRasterPass && mpVao)
    {
        bindModel();
        mpRasterPass->getState()->setVao(mpVao);
        mpRasterPass->getState()->setFbo(pTargetFbo);

        // Full-screen inference; the active blend mode (VS interpolation vs PS blend) is baked into
        // the compiled shader. GPU time of the draw is measured with GpuTimer.
        mpTimer->begin();
        mpRasterPass->draw(pRenderContext, kQuadVerts, 0);
        mpTimer->end();
        mpTimer->resolve();
        mTimedOnce = true;
    }
}

void NeuRaster::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "NeuRaster", {380, 380}, {20, 60});
    renderGlobalUI(pGui);

    w.text(mModelLoaded ? fmt::format("Model loaded (numMLPs={})", mNumMLPs) : "Model NOT loaded");
    w.text(fmt::format("peBands={} res={} rank={} planeDim={} inputDim={}", mPeBands, mRes, mRank, mPlaneDim, mInputDim));
    // Dimensions the shader was compiled for (derived from the model) + interpolant-budget note.
    w.text(mShaderDimsStatus);
    w.separator();

    // Corner probe selectors: each of the 4 quad corners evaluates a probe; the rasterizer blends them.
    if (mModelLoaded && mNumMLPs > 0)
    {
        const uint32_t maxV = mNumMLPs - 1;
        for (uint32_t c = 0; c < 4; ++c)
        {
            if (w.var(fmt::format("Corner {} probe", c).c_str(), mCornerProbe[c], 0u, maxV))
                mCornerProbe[c] = std::min(mCornerProbe[c], maxV);
        }
    }
    w.separator();

    // Input-direction control (azimuth/elevation -> unit vector), used by every corner's feature.
    w.var("Azimuth (deg)", mDirAzimuth, 0.0f, 360.0f);
    w.var("Elevation (deg)", mDirElevation, -90.0f, 90.0f);
    const float3 dir = computeInputDir();
    w.text(fmt::format("Input dir: ({:.3f}, {:.3f}, {:.3f})", dir.x, dir.y, dir.z));
    w.separator();

    w.var("MLP iterations / pixel", mIterations, 1u, 4096u);

    // Blend-mode toggle (recompiles the shader) + A/B comparison of the two modes' GPU times.
    w.separator();
    if (mVsInterpFeasible)
    {
        if (w.checkbox("VS weight interpolation (off = fragment-shader blend)", mVsInterp))
            createRasterPass();
    }
    else
    {
        w.text(fmt::format(
            "VS interpolation unavailable: needs {} float4 VS outputs (limit ~32).", mVsInterpVec4
        ));
        w.text("Only fragment-shader blend is available for this model.");
    }
    w.text(fmt::format("Mode: {}", mVsInterp ? "vertex-shader interpolation" : "fragment-shader blend"));

    // Feature-fetch location toggle (recompiles the shader), independent of the weight-blend mode.
    if (w.checkbox("Fetch feature in VS (off = fragment shader)", mFeatVsInterp))
        createRasterPass();
    w.text(fmt::format("Feature fetch: {}", mFeatVsInterp ? "vertex shader" : "fragment shader"));

    // Feature-plane backend toggle (recompiles the shader). Only offered when textures are usable.
    if (mTexturesFeasible)
    {
        if (w.checkbox("Texture feature planes (off = StructuredBuffer)", mUseTextures))
            createRasterPass();
    }
    else if (mModelLoaded)
    {
        w.text("Texture feature planes unavailable (needs planeDim==4 and dims <= 16384).");
    }
    w.text(fmt::format("Feature backend: {}", mUseTextures ? "Texture2D + HW filtering" : "StructuredBuffer"));
    w.text(fmt::format("  VS-interp draw: {:.4f} ms", mGpuMsVsInterp));
    w.text(fmt::format("  PS-blend  draw: {:.4f} ms", mGpuMsPsBlend));
    if (mGpuMsVsInterp > 0.0 && mGpuMsPsBlend > 0.0)
    {
        const double faster = std::min(mGpuMsVsInterp, mGpuMsPsBlend);
        const double slower = std::max(mGpuMsVsInterp, mGpuMsPsBlend);
        w.text(fmt::format(
            "  {} faster by {:.2f}x", mGpuMsVsInterp < mGpuMsPsBlend ? "VS-interp" : "PS-blend", slower / faster
        ));
    }
    w.separator();

    const double frameMs = getFrameRate().getLastFrameTime() * 1000.0;
    const double fps = frameMs > 0.0 ? 1000.0 / frameMs : 0.0;
    w.text(fmt::format("Inference draw (GPU): {:.4f} ms", mLastGpuMs));
    w.text(fmt::format("Frame: {:.3f} ms  ({:.1f} FPS)", frameMs, fps));
}

int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.width = 1920;
    config.windowDesc.height = 1080;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.enableVSync = false; // Disable vsync so the frame rate reflects render cost.
    config.windowDesc.title = "Falcor NeuRaster inference perf test";

    NeuRaster app(config);
    return app.run();
}

int main(int argc, char** argv)
{
    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
