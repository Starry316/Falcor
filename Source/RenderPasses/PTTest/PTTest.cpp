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
#include "PTTest.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Tools/Utils.h"
#include "Utils/Math/Matrix.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <unordered_set>
#include <map>
#include <tuple>
#include <cmath>
#define pX mXYUV.x
#define pY mXYUV.y
#define pU mXYUV.z
#define pV mXYUV.w
extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, PTTest>();
}
float3 spherical_to_cartesian_radXZY(float2 sph)
{
    float3 p;
    p.x = -cos(sph.y - M_PI) * sin(sph.x);
    p.z = -sin(sph.y - M_PI) * sin(sph.x);
    p.y = cos(sph.x);
    return p;
}

float4x4 MakeViewFromCameraPos(float3 cameraPos)
{
    // Forward: from camera to origin
    float3 fwd = normalize(-cameraPos);

    // Stable Y-up basis
    float3 worldUp = float3(0.0f, 1.0f, 0.0f);
    if (abs(dot(fwd, worldUp)) > 0.99999999f)
        worldUp = float3(0.0f, 0.0f, 1.0f);

    float3 right = normalize(cross(worldUp, fwd));
    float3 up = cross(fwd, right);

    // World -> Camera:
    // x = dot(right, P) - dot(right, C)
    // y = dot(up,    P) - dot(up,    C)
    // z = dot(fwd,   P) - dot(fwd,   C)
    return float4x4{
        right.x,
        right.y,
        right.z,
        -dot(right, cameraPos),
        up.x,
        up.y,
        up.z,
        -dot(up, cameraPos),
        fwd.x,
        fwd.y,
        fwd.z,
        -dot(fwd, cameraPos),
        0.f,
        0.f,
        0.f,
        1.f};
}
float4x4 makeOrthoProjection(float halfSize)
{
    float inv = (halfSize != 0.f) ? (1.f / halfSize) : 0.f;

    return float4x4{inv, 0.f, 0.f, 0.f, 0.f, inv, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 1.f};
}
namespace
{
const char kShaderFile[] = "RenderPasses/PTTest/MinimalPathTracer.rt.slang";
const char kExportTriFramesShaderFile[] = "RenderPasses/PTTest/ExportTriFrames.cs.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 72u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" , true},
    { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
    // clang-format on
};

const ChannelList kOutputChannels = {
    // clang-format off
    { "color",          "gOutputColor", "Output color (sum of direct and indirect)", false, ResourceFormat::RGBA32Float },
    // clang-format on
};

const char kMaxBounces[] = "maxBounces";
const char kComputeDirect[] = "computeDirect";
const char kUseImportanceSampling[] = "useImportanceSampling";

// Output phase selection.
const uint32_t kPhaseVerticesOnly = 0;
const uint32_t kPhaseTrianglesOnly = 1;
const uint32_t kPhaseBoth = 2;

// Positions within this distance are treated as the same vertex.
const float kWeldEpsilon = 1e-5f;

/// Maps each mesh vertex to a canonical ID by welding positionally-coincident vertices. Unique
/// positions are assigned contiguous IDs in [0, uniqueCount) in order of first appearance, so the
/// output IDs are dense and regular. Returns remap[origIndex] -> canonicalID.
std::vector<uint32_t> buildVertexWeldRemap(const float3* positions, uint32_t vertexCount)
{
    const double q = 1.0 / kWeldEpsilon;
    auto keyOf = [&](const float3& p)
    {
        return std::make_tuple(
            (int64_t)std::llround(p.x * q), (int64_t)std::llround(p.y * q), (int64_t)std::llround(p.z * q)
        );
    };

    std::map<std::tuple<int64_t, int64_t, int64_t>, uint32_t> posToId;
    std::vector<uint32_t> remap(vertexCount);
    uint32_t nextId = 0;
    for (uint32_t i = 0; i < vertexCount; ++i)
    {
        const auto key = keyOf(positions[i]);
        auto it = posToId.find(key);
        if (it == posToId.end())
        {
            posToId.emplace(key, nextId);
            remap[i] = nextId;
            ++nextId;
        }
        else
        {
            remap[i] = it->second;
        }
    }
    return remap;
}

/// Reads a little-endian float32 binary file into a vector (empty on failure).
std::vector<float> readFloatBinary(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
    {
        logWarning("[NeuLobes] Unable to open file {}", path.string());
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (size > 0 && !file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        logWarning("[NeuLobes] Error reading file {}", path.string());
        return {};
    }
    return buffer;
}
} // namespace

PTTest::PTTest(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    FALCOR_ASSERT(mpSampleGenerator);

    // Load the NeuLobes neural light-probe model (files are optional; a warning is logged if missing).
    loadNeuLobesModel(mNeuLobesDir);
    loadSHModel(mSHDir);
    loadSGModel(mSGDir);
    loadSVModel(mSVDir);
}

void PTTest::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == kMaxBounces)
            mMaxBounces = value;
        else if (key == kComputeDirect)
            mComputeDirect = value;
        else if (key == kUseImportanceSampling)
            mUseImportanceSampling = value;
        else
            logWarning("Unknown property '{}' in PTTest properties.", key);
    }
}

Properties PTTest::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

RenderPassReflection PTTest::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void PTTest::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }

    // If we have no scene, just clear the outputs and return.
    if (!mpScene)
    {
        for (auto it : kOutputChannels)
        {
            Texture* pDst = renderData.getTexture(it.name).get();
            if (pDst)
                pRenderContext->clearTexture(pDst);
        }
        return;
    }

    if (is_set(mpScene->getUpdates(), IScene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), IScene::UpdateFlags::GeometryChanged))
    {
        FALCOR_THROW("This render pass does not support scene changes that require shader recompilation.");
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }
    // Get dimensions of ray dispatch.
    const uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    createTex(mpBarycentric, mpDevice, targetDim);
    createTex(mpWi, mpDevice, targetDim);
    createTex(mpRadiance, mpDevice, targetDim);

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    // mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");

    // For optional I/O resources, set 'is_valid_<name>' defines to inform the program of which ones it can access.
    // TODO: This should be moved to a more general mechanism using Slang.
    mTracer.pProgram->addDefines(getValidResourceDefines(kInputChannels, renderData));
    mTracer.pProgram->addDefines(getValidResourceDefines(kOutputChannels, renderData));

    // Prepare program vars. This may trigger shader compilation.
    // The program should have all necessary defines set at this point.
    if (!mTracer.pVars)
        prepareVars();
    FALCOR_ASSERT(mTracer.pVars);

    // Set constants.
    auto var = mTracer.pVars->getRootVar();
    var["CB"]["gFrameCount"] = mFrameCount;
    var["CB"]["gPRNGDimension"] = dict.keyExists(kRenderPassPRNGDimension) ? dict[kRenderPassPRNGDimension] : 0u;
    var["CB"]["gViewTheta"] = mViewTheta;
    var["CB"]["gViewPhi"] = mViewPhi;
    var["CB"]["gViewSize"] = mViewSize;
    var["CB"]["gBTFViewMode"] = mBTFViewMode;
    var["CB"]["gViewHeight"] = mViewHeight;
    var["CB"]["gViewHeightBot"] = mViewHeightBot;
    var["CB"]["gMaxBounces"] = mMaxBounces;
    var["CB"]["gXYUV"] = mXYUV;
    var["CB"]["gPluckerMode"] = mPluckerMode;
    var["CB"]["gShowSelectedTri"] = mShowSelectedTri;
    var["CB"]["gSelectedInstanceID"] = mSelectedInstanceID;
    var["CB"]["gSelectedTriangleID"] = mSelectedTriangleID;
    var["CB"]["gTriSampleUV"] = float2(mTriSampleU, mTriSampleV);
    // Probe mode: vertex probes sample the full sphere; triangle probes sample the hemisphere. During
    // export this follows the output phase (0 = vertices); interactively it follows the phase dropdown.
    var["CB"]["gVertexMode"] = mIsOutputing ? (mOutputPhase == 0) : (mPhaseSelection != kPhaseTrianglesOnly);
    var["CB"]["gProbePos"] = mProbePos;

    var["gBarycentric"] = mpBarycentric;
    var["gWi"] = mpWi;
    var["gRadiance"] = mpRadiance;
    if (mpTriFrame)
        var["gTriFrame"] = mpTriFrame;

    if (mpEnvMapSampler)
        mpEnvMapSampler->bindShaderData(var["CB"]["gEnvMapSampler"]);

    // Bind NeuLobes neural light-probe model. The runtime flag gates evaluation in the shader,
    // so the resources are always referenced (present in reflection) even when disabled.
    // Ensure the vertex-weld remap exists for the current instance (needed for the pool gather).
    // Cache the selected instance's triangle data (also creates the per-triangle frame buffer used by
    // the hemisphere path). Needed regardless of NeuLobes, since the hemisphere render always runs.
    if (!mpVertexRemap || mVertexRemapInstanceID != mSelectedInstanceID)
        cacheInstanceTriangles();
    bindNeuLobesData(var);
    bindProbeReprData(var);

    // Bind the position-weld remap (maps a triangle's local vertex indices to canonical vertex IDs).
    var["CB"]["gHasVertexRemap"] = (mpVertexRemap != nullptr);
    if (mpVertexRemap)
        var["gVertexRemap"] = mpVertexRemap;
    // Bind I/O buffers. These needs to be done per-frame as the buffers may change anytime.
    auto bind = [&](const ChannelDesc& desc)
    {
        if (!desc.texname.empty())
        {
            var[desc.texname] = renderData.getTexture(desc.name);
        }
    };
    for (auto channel : kInputChannels)
        bind(channel);
    for (auto channel : kOutputChannels)
        bind(channel);



    // Spawn the rays.
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, uint3(targetDim, 1));

    mFrameCount++;
}

uint32_t PTTest::triangleStrataDim() const
{
    const uint32_t target = std::max<uint32_t>(1u, mTriangleSamples);
    return std::max<uint32_t>(1u, (uint32_t)std::lround(std::sqrt((double)target)));
}

uint32_t PTTest::interiorSampleCount() const
{
    if (mSamplingMode == 1) // uniform: exactly the requested count
        return std::max<uint32_t>(1u, mTriangleSamples);
    const uint32_t n = triangleStrataDim(); // stratified: nearest N*N grid
    return n * n;
}

void PTTest::updateInteriorSample()
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (mSamplingMode == 1)
    {
        // Uniform random over the triangle (unit square -> sample_triangle on the shader side).
        mTriSampleU = dist(mSampleRng);
        mTriSampleV = dist(mSampleRng);
    }
    else
    {
        // Stratified: jittered sample inside cell (i, j) of an NxN grid over the unit square.
        const uint32_t n = triangleStrataDim();
        const uint32_t i = mSampleIndex % n;
        const uint32_t j = mSampleIndex / n;
        mTriSampleU = (i + dist(mSampleRng)) / (float)n;
        mTriSampleV = (j + dist(mSampleRng)) / (float)n;
    }
}

void PTTest::updateEdgeSample()
{
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    // Parameter t in (0,1) along the edge (jittered stratum for stratified mode, random for uniform).
    const float t = (mSamplingMode == 1)
        ? dist(mSampleRng)
        : (mEdgeStratum + dist(mSampleRng)) / (float)std::max<uint32_t>(1u, mEdgeSamples);
    // Pick (u, v) so sample_triangle(u, v) lands on the chosen edge (one barycentric == 0):
    //   edge 0 (v0-v1): bary = (1-t, t, 0)  -> u = (1-t)^2, v = 0
    //   edge 1 (v1-v2): bary = (0, 1-t, t)  -> u = t^2,     v = 1
    //   edge 2 (v2-v0): bary = (1-t, 0, t)  -> u = 1,       v = t
    if (mEdgeIndex == 0)
    {
        mTriSampleU = (1.0f - t) * (1.0f - t);
        mTriSampleV = 0.0f;
    }
    else if (mEdgeIndex == 1)
    {
        mTriSampleU = t * t;
        mTriSampleV = 1.0f;
    }
    else
    {
        mTriSampleU = 1.0f;
        mTriSampleV = t;
    }
}

void PTTest::stopOutput()
{
    mOutputStep = 0;
    mOutputIndx = 0;
    mVertexIndex = 0;
    mOutputPhase = 0;
    mSelectedTriangleID = 0;
    mSampleIndex = 0;
    mEdgeIndex = 0;
    mEdgeStratum = 0;
    mIsOutputing = false;
    if (mpScene)
    {
        auto camera = mpScene->getCamera();
        camera->setResetFlag(true);
        camera->setNextStep(false);
        camera->setAccumulating(false);
    }
    mTriSampleU = 0.5f;
    mTriSampleV = 0.5f;
}

void PTTest::handleOutput()
{
        auto camera = mpScene->getCamera();

        const bool doTriangles = (mPhaseSelection != kPhaseVerticesOnly);

        // Phase 0: per-vertex precompute. Each unique vertex is rendered once (probe origin placed at
        // the vertex via a representative triangle + corner), named by its scene vertex ID.
        if (mOutputPhase == 0)
        {
            if (mVertexIndex >= mInstanceVertices.size())
            {
                // No (more) vertices; either move on to triangles or stop.
                if (doTriangles)
                {
                    mOutputPhase = 1;
                    mSelectedTriangleID = 0;
                    mSampleIndex = 0;
                    updateInteriorSample();
                    return;
                }
                stopOutput();
                return;
            }

            const VertexProbe& vp = mInstanceVertices[mVertexIndex];
            mSelectedTriangleID = vp.triangleID;
            mTriSampleU = vertexUV[vp.uvIndex].x;
            mTriSampleV = vertexUV[vp.uvIndex].y;

            camera->setOutputPath(fmt::format(mVertexOutputPath, mSelectedInstanceID, mVertexGlobalID, vp.vertexID));
            if (!camera->isNextStep())
            {
                return;
            }
            camera->setNextStep(false);
            camera->setAccumulating(mIsOutputing);
            camera->setOutputFrameCount(mOutputSPP);

            // This vertex file has been committed; advance the vertex global running index.
            mVertexGlobalID++;
            mVertexIndex++;
            if (mVertexIndex >= mInstanceVertices.size())
            {
                if (doTriangles)
                {
                    // All vertices done; switch to triangle interior sampling.
                    mOutputPhase = 1;
                    mSelectedTriangleID = 0;
                    mSampleIndex = 0;
                    updateInteriorSample();
                }
                else
                {
                    stopOutput();
                }
            }
            return;
        }

        // Phase 1: per-triangle interior sampling (stratified grid or uniform random, mTriangleSamples).
        if (mOutputPhase == 1)
        {
            const uint3 vids = (mSelectedTriangleID < mInstanceTriIndices.size())
                ? mInstanceTriIndices[mSelectedTriangleID]
                : uint3(0);
            camera->setOutputPath(
                fmt::format(mOutputPath, mSelectedInstanceID, mGlobalOutputID, mSelectedTriangleID, vids.x, vids.y, vids.z, mTriSampleU, mTriSampleV)
            );
            if (!camera->isNextStep())
            {
                return;
            }
            camera->setNextStep(false);
            camera->setAccumulating(mIsOutputing);
            camera->setOutputFrameCount(mOutputSPP);

            // This triangle sample file has been committed; advance the global running index.
            mGlobalOutputID++;

            // Advance to the next interior sample.
            mSampleIndex++;
            if (mSampleIndex >= interiorSampleCount())
            {
                // Finished this triangle's interior; advance to the next triangle in the instance.
                mSelectedTriangleID++;
                if (mSelectedTriangleID < mInstanceTriangleCount)
                {
                    mSampleIndex = 0;
                    updateInteriorSample();
                    return;
                }

                // All interiors done; move to edge sampling (if enabled) or stop.
                if (mSampleTriangleEdges)
                {
                    mOutputPhase = 2;
                    mSelectedTriangleID = 0;
                    mEdgeIndex = 0;
                    mEdgeStratum = 0;
                    updateEdgeSample();
                    return;
                }
                stopOutput();
                return;
            }

            // Compute the next interior sample.
            updateInteriorSample();
            return;
        }

        // Phase 2: per-triangle EDGE sampling (data augmentation). mEdgeSamples samples along each
        // of the 3 edges (one barycentric coordinate == 0), emitted via the same triangle output path.
        {
            const uint3 vids = (mSelectedTriangleID < mInstanceTriIndices.size())
                ? mInstanceTriIndices[mSelectedTriangleID]
                : uint3(0);
            camera->setOutputPath(
                fmt::format(mOutputPath, mSelectedInstanceID, mGlobalOutputID, mSelectedTriangleID, vids.x, vids.y, vids.z, mTriSampleU, mTriSampleV)
            );
            if (!camera->isNextStep())
            {
                return;
            }
            camera->setNextStep(false);
            camera->setAccumulating(mIsOutputing);
            camera->setOutputFrameCount(mOutputSPP);

            mGlobalOutputID++;

            // Advance along the current edge, then to the next edge, then the next triangle.
            mEdgeStratum++;
            if (mEdgeStratum >= mEdgeSamples)
            {
                mEdgeStratum = 0;
                mEdgeIndex++;
                if (mEdgeIndex >= 3)
                {
                    mEdgeIndex = 0;
                    mSelectedTriangleID++;
                    if (mSelectedTriangleID >= mInstanceTriangleCount)
                    {
                        stopOutput();
                        return;
                    }
                }
            }

            updateEdgeSample();
            return;
        }
}
void PTTest::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    widget.text(fmt::format("Meshes: {}", mMeshCount));
    widget.text(fmt::format("Instances: {}", mInstanceCount));
    widget.separator();

    // Triangle picking. An instance maps to exactly one mesh (gi.geometryID), so instance ID alone
    // determines the geometry; we resolve and show the mesh ID and its triangle count below.
    dirty |= widget.var("Pick instance ID", mSelectedInstanceID, 0u, mInstanceCount > 0 ? mInstanceCount - 1 : 0u);

    uint32_t selectedTriangleCount = 0;
    if (mpScene && mSelectedInstanceID < mpScene->getGeometryInstanceCount())
    {
        const auto& gi = mpScene->getGeometryInstance(mSelectedInstanceID);
        if (gi.getType() == GeometryType::TriangleMesh)
        {
            selectedTriangleCount = mpScene->getMesh(MeshID{gi.geometryID}).getTriangleCount();
            widget.text(fmt::format("  -> mesh ID: {}, triangles: {}", gi.geometryID, selectedTriangleCount));
        }
        else
        {
            widget.text("  -> instance is not a triangle mesh");
        }
    }

    dirty |= widget.var("Pick triangle ID", mSelectedTriangleID, 0u, selectedTriangleCount > 0 ? selectedTriangleCount - 1 : 0u);
    if (widget.button("Compute triangle world positions"))
        computeSelectedTriangleWorldPositions();
    if (mSelectedTriangleValid)
    {
        widget.text(fmt::format("Vertex IDs: {}, {}, {}", mSelectedTriVertexIDs[0], mSelectedTriVertexIDs[1], mSelectedTriVertexIDs[2]));
        for (int i = 0; i < 3; ++i)
        {
            const float3& p = mSelectedTrianglePosW[i];
            widget.text(fmt::format("v{} (id {}): ({:.4f}, {:.4f}, {:.4f})", i, mSelectedTriVertexIDs[i], p.x, p.y, p.z));
        }
    }
    dirty |= widget.checkbox("Show select Tri", mShowSelectedTri);

    // Uniform triangle sampling coordinates for the primary ray origin.
    dirty |= widget.var("Tri sample u", mTriSampleU, 0.f, 1.f);
    dirty |= widget.var("Tri sample v", mTriSampleV, 0.f, 1.f);

    widget.separator();

    dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.slider("light theta", mLightTheta, 0.0f, 2.0f);
    dirty |= widget.slider("light phi", mLightPhi, 0.0f, 1.0f);

    dirty |= widget.slider("view theta", mViewTheta, 0.0f, 1.0f);
    dirty |= widget.slider("view phi", mViewPhi, 0.0f, 1.0f);
    dirty |= widget.slider("view size", mViewSize, 0.0f, 10.0f);
    dirty |= widget.slider("view height", mViewHeight, 0.0f, 10.0f);
    dirty |= widget.var("view height_", mViewHeight);
    dirty |= widget.slider("view height bot", mViewHeightBot, 0.0f, mViewHeight);

    dirty |= widget.slider("probe pos", mProbePos, 0.0f, 1.0f);
    // dirty |= widget.slider("x", pX, 0.0f, 1.0f);
    // dirty |= widget.slider("y", pY, 0.0f, 1.0f);
    // dirty |= widget.slider("u", pU, 0.0f, 1.0f);
    // dirty |= widget.slider("v", pV, 0.0f, 1.0f);

    dirty |= widget.checkbox("btf mode", mBTFViewMode);
    dirty |= widget.checkbox("Plucker mode", mPluckerMode);

    dirty |= widget.checkbox("Change Light", mChangeLight);

    dirty |= widget.checkbox("Use NeuLobes probe", mUseNeuLobes);
    widget.tooltip("Evaluate a per-vertex light probe (neural / SH / SG / SV) and write it to the output color.", true);
    if (mUseNeuLobes)
    {
        Gui::DropdownList reprList = {
            {0u, "Neural (mu-law)"},
            {1u, "Spherical Harmonics"},
            {2u, "Spherical Gaussians"},
            {3u, "Spherical Voronoi"},
        };
        dirty |= widget.dropdown("Probe representation", reprList, mProbeRepr);
        if (mProbeRepr == 0)
            widget.text(mNeuLobes.loaded ? "Neural: loaded" : "Neural: NOT loaded");
        else if (mProbeRepr == 1)
            widget.text(mSH.loaded ? fmt::format("SH: loaded (deg {}, pool {})", mSH.degree, mSH.poolSize) : "SH: NOT loaded");
        else if (mProbeRepr == 2)
            widget.text(mSG.loaded ? fmt::format("SG: loaded ({} lobes, pool {})", mSG.numSGs, mSG.poolSize) : "SG: NOT loaded");
        else
            widget.text(mSV.loaded ? fmt::format("SV: loaded ({} sites, pool {})", mSV.numSites, mSV.poolSize) : "SV: NOT loaded");

        if (mProbeRepr == 0)
        {
            Gui::DropdownList featInterpList = {
                {0u, "Bilinear"},
                {1u, "Nearest"},
            };
            dirty |= widget.dropdown("Feature interpolation", featInterpList, mFeatInterp);
        }

        dirty |= widget.var("NeuLobes bary", mNeuBary, 0.0f, 1.0f);
        dirty |= widget.var("NeuLobes vertexID (demo)", mNeuVertexID);
    }

    if (widget.button("output frame"))
    {
        mpBarycentric->captureToFile(
                    0, 0, "C:/Data/Render/bary.exr", Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Lossy
                );
        mpWi->captureToFile(
                    0, 0, "C:/Data/Render/wi.exr", Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Lossy
                );
        mpRadiance->captureToFile(
                    0, 0, "C:/Data/Render/rad.exr", Bitmap::FileFormat::ExrFile, Bitmap::ExportFlags::Lossy
                );

    }

    if (widget.button("export tri frames"))
        exportTriFrames(mTriFrameOutputPath);
    widget.textbox("Tri-frame Output Path", mTriFrameOutputPath);

    if (!mChangeLight)
    {
        widget.textbox("Vertex Output Path", mVertexOutputPath);
        widget.textbox("Triangle Output Path", mOutputPath);
    }
    else
    {
        widget.textbox("Output Path", mOutputBTFPath);
    }

    Gui::DropdownList phaseList = {
        {kPhaseVerticesOnly, "Vertices only"},
        {kPhaseTrianglesOnly, "Triangles only"},
        {kPhaseBoth, "Vertices + Triangles"},
    };
    dirty |= widget.dropdown("Output phase", phaseList, mPhaseSelection);

    Gui::DropdownList samplingList = {
        {0u, "Stratified"},
        {1u, "Uniform"},
    };
    dirty |= widget.dropdown("Interior sampling", samplingList, mSamplingMode);
    dirty |= widget.var("Samples per triangle", mTriangleSamples, 1u, 1u << 16);
    if (mSamplingMode == 0)
        widget.text(fmt::format("  (stratified {0}x{0} = {1} samples)", triangleStrataDim(), interiorSampleCount()));

    dirty |= widget.checkbox("Sample triangle edges", mSampleTriangleEdges);
    widget.tooltip("Adds extra samples along the 3 triangle edges as data augmentation.", true);
    if (mSampleTriangleEdges)
        dirty |= widget.var("Samples per edge", mEdgeSamples, 1u, 1u << 16);

    widget.var("OutputSPP", mOutputSPP);
    if (mIsOutputing)
    {
        const char* phaseName = (mOutputPhase == 0) ? "vertices" : ((mOutputPhase == 1) ? "triangle interior" : "triangle edges");
        widget.text(fmt::format(
            "Phase: {} | vertex {}/{} | triangle {}/{} | interior {}/{} | edge {}",
            phaseName,
            mVertexIndex, (uint32_t)mInstanceVertices.size(),
            mSelectedTriangleID, mInstanceTriangleCount,
            mSampleIndex, interiorSampleCount(), mEdgeIndex
        ));
        handleOutput();
        if (widget.button("Stop", true))
        {
            auto camera = mpScene->getCamera();
            camera->setOutputFrameCount(mOutputSPP);
            camera->setAccumulating(false);
            mIsOutputing = false;
            dirty = true;
        }
    }
    else
    {
        if (widget.button("Start Output"))
        {
            // Cache the whole instance's triangles + unique vertices, then start at the selected phase.
            cacheInstanceTriangles();
            mVertexIndex = 0;
            mSelectedTriangleID = 0;
            mSampleIndex = 0;
            mEdgeIndex = 0;
            mEdgeStratum = 0;
            mGlobalOutputID = 0;
            mVertexGlobalID = 0;
            mOutputIndx = 0;

            mIsOutputing = true;
            dirty = true;

            if (mPhaseSelection == kPhaseTrianglesOnly)
            {
                // Triangles only: skip the vertex phase and seed the first interior sample.
                mOutputPhase = 1;
                updateInteriorSample();
            }
            else
            {
                // Vertices only or both: start with the vertex phase.
                mOutputPhase = 0;
                mTriSampleU = vertexUV[0].x;
                mTriSampleV = vertexUV[0].y;
            }

            auto camera = mpScene->getCamera();
            camera->setOutputFrameCount(mOutputSPP);
            camera->setAccumulating(true);
        }

        if (widget.button("Rest"))
        {
            mIsOutputing = false;
            dirty = true;

            auto camera = mpScene->getCamera();
            camera->setOutputFrameCount(mOutputSPP);
            camera->setAccumulating(false);
            mIsOutputing = false;
            dirty = true;
            mOutputStep = 0;
            mOutputOffsetIndx = 0;
            mOutputIndx = 0;
            mVertexIndex = 0;
            mOutputPhase = 0;
            mSelectedTriangleID = 0;
            mSampleIndex = 0;
            mEdgeIndex = 0;
            mEdgeStratum = 0;
            mGlobalOutputID = 0;
            mViewTheta = 0;
            mViewPhi = 0;
            mLightTheta = 0;
            mLightPhi = 0;
        }
    }

    //

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mOptionsChanged = true;
    }
}

void PTTest::computeSelectedTriangleWorldPositions()
{
    mSelectedTriangleValid = false;

    if (!mpScene)
    {
        logWarning("PTTest: no scene loaded.");
        return;
    }
    if (mSelectedInstanceID >= mpScene->getGeometryInstanceCount())
    {
        logWarning("PTTest: instance ID {} out of range ({} instances).", mSelectedInstanceID, mpScene->getGeometryInstanceCount());
        return;
    }

    const GeometryInstanceData& gi = mpScene->getGeometryInstance(mSelectedInstanceID);
    if (gi.getType() != GeometryType::TriangleMesh)
    {
        logWarning("PTTest: instance {} is not a triangle mesh.", mSelectedInstanceID);
        return;
    }

    const MeshID meshID{gi.geometryID};
    const MeshDesc& desc = mpScene->getMesh(meshID);
    const uint32_t vertexCount = desc.vertexCount;
    const uint32_t triangleCount = desc.getTriangleCount();
    if (mSelectedTriangleID >= triangleCount)
    {
        logWarning("PTTest: triangle ID {} out of range ({} triangles in mesh {}).", mSelectedTriangleID, triangleCount, meshID.get());
        return;
    }

    // GPU output buffers filled by getMeshVerticesAndIndices (positions/texcrds are float3, indices are uint3).
    // Positions are in object/local space; indices are local vertex indices for the mesh.
    const ResourceBindFlags uavFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto pPositions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);
    auto pTexcrds = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);
    auto pIndices = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);

    mpScene->getMeshVerticesAndIndices(
        meshID, {{"positions", pPositions}, {"texcrds", pTexcrds}, {"triangleIndices", pIndices}}
    );

    // Copy the data we need to CPU-readable staging buffers.
    auto pIdxStaging = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
    auto pPosStaging = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);

    RenderContext* pRenderContext = mpDevice->getRenderContext();
    pRenderContext->copyBufferRegion(pIdxStaging.get(), 0, pIndices.get(), 0, sizeof(uint3) * triangleCount);
    pRenderContext->copyBufferRegion(pPosStaging.get(), 0, pPositions.get(), 0, sizeof(float3) * vertexCount);
    pRenderContext->submit(true); // Wait for GPU work to complete.

    const uint3* pIdxData = reinterpret_cast<const uint3*>(pIdxStaging->map());
    const float3* pPosData = reinterpret_cast<const float3*>(pPosStaging->map());

    const uint3 tri = pIdxData[mSelectedTriangleID];

    // Weld coincident positions so the displayed vertex IDs match the (welded) output filenames.
    const std::vector<uint32_t> remap = buildVertexWeldRemap(pPosData, vertexCount);

    // Object-to-world transform for this instance.
    const float4x4 objectToWorld = mpScene->getAnimationController()->getGlobalMatrices()[gi.globalMatrixID];

    for (int i = 0; i < 3; ++i)
    {
        mSelectedTriVertexIDs[i] = remap[tri[i]];
        const float3 posO = pPosData[tri[i]];
        const float4 posW = mul(objectToWorld, float4(posO, 1.f));
        mSelectedTrianglePosW[i] = float3(posW.x, posW.y, posW.z) / posW.w;
    }

    pIdxStaging->unmap();
    pPosStaging->unmap();

    mSelectedTriangleValid = true;

    logInfo(
        "PTTest: instance {} (mesh {}) triangle {} world positions: v0=({}, {}, {}), v1=({}, {}, {}), v2=({}, {}, {})",
        mSelectedInstanceID, meshID.get(), mSelectedTriangleID,
        mSelectedTrianglePosW[0].x, mSelectedTrianglePosW[0].y, mSelectedTrianglePosW[0].z,
        mSelectedTrianglePosW[1].x, mSelectedTrianglePosW[1].y, mSelectedTrianglePosW[1].z,
        mSelectedTrianglePosW[2].x, mSelectedTrianglePosW[2].y, mSelectedTrianglePosW[2].z
    );
}

void PTTest::cacheInstanceTriangles()
{
    mInstanceTriIndices.clear();
    mInstanceVertices.clear();
    mInstanceTriangleCount = 0;

    if (!mpScene)
    {
        logWarning("PTTest: no scene loaded.");
        return;
    }
    if (mSelectedInstanceID >= mpScene->getGeometryInstanceCount())
    {
        logWarning("PTTest: instance ID {} out of range ({} instances).", mSelectedInstanceID, mpScene->getGeometryInstanceCount());
        return;
    }

    const GeometryInstanceData& gi = mpScene->getGeometryInstance(mSelectedInstanceID);
    if (gi.getType() != GeometryType::TriangleMesh)
    {
        logWarning("PTTest: instance {} is not a triangle mesh.", mSelectedInstanceID);
        return;
    }

    const MeshID meshID{gi.geometryID};
    const MeshDesc& desc = mpScene->getMesh(meshID);
    const uint32_t vertexCount = desc.vertexCount;
    const uint32_t triangleCount = desc.getTriangleCount();
    if (triangleCount == 0)
    {
        logWarning("PTTest: instance {} (mesh {}) has no triangles.", mSelectedInstanceID, meshID.get());
        return;
    }

    // Fetch the mesh index/position data and read the triangle indices back to the CPU once.
    const ResourceBindFlags uavFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    auto pPositions = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);
    auto pTexcrds = mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);
    auto pIndices = mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, uavFlags, MemoryType::DeviceLocal, nullptr, false);

    mpScene->getMeshVerticesAndIndices(
        meshID, {{"positions", pPositions}, {"texcrds", pTexcrds}, {"triangleIndices", pIndices}}
    );

    auto pIdxStaging =
        mpDevice->createStructuredBuffer(sizeof(uint3), triangleCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);
    auto pPosStaging =
        mpDevice->createStructuredBuffer(sizeof(float3), vertexCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false);

    RenderContext* pRenderContext = mpDevice->getRenderContext();
    pRenderContext->copyBufferRegion(pIdxStaging.get(), 0, pIndices.get(), 0, sizeof(uint3) * triangleCount);
    pRenderContext->copyBufferRegion(pPosStaging.get(), 0, pPositions.get(), 0, sizeof(float3) * vertexCount);
    pRenderContext->submit(true); // Wait for GPU work to complete.

    const uint3* pIdxData = reinterpret_cast<const uint3*>(pIdxStaging->map());
    mInstanceTriIndices.assign(pIdxData, pIdxData + triangleCount);
    pIdxStaging->unmap();

    mInstanceTriangleCount = triangleCount;

    // Per-triangle local->world frame export buffer (ProbeTriFrame = 4 x float4 = 64 bytes/triangle).
    // Zero-initialized so triangles that were never rendered read back as zeros rather than garbage.
    std::vector<float4> triFrameInit((size_t)triangleCount * 4, float4(0.f));
    mpTriFrame = mpDevice->createStructuredBuffer(
        sizeof(float4) * 4, triangleCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal, triFrameInit.data(), false
    );

    // Weld positionally-coincident vertices to a single canonical ID. Mesh formats split a vertex
    // whenever adjacent faces need different normals (flat shading) or UVs (seams), so the same
    // position can appear under several indices. We map each position to the smallest original index
    // sharing it, then remap all triangle corners to those canonical IDs.
    const float3* pPosData = reinterpret_cast<const float3*>(pPosStaging->map());
    std::vector<uint32_t> remap = buildVertexWeldRemap(pPosData, vertexCount);

    // Count unique welded positions for logging.
    std::unordered_set<uint32_t> uniquePositions(remap.begin(), remap.end());
    const size_t uniquePositionCount = uniquePositions.size();
    pPosStaging->unmap();

    // Upload the remap so the shader can map a triangle's (mesh-local) vertex indices to canonical IDs.
    mpVertexRemap = mpDevice->createStructuredBuffer(
        sizeof(uint32_t), vertexCount, ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, remap.data(), false
    );
    mVertexRemapInstanceID = mSelectedInstanceID;

    // Remap triangle corners to canonical vertex IDs (used for the triangle output filenames).
    for (auto& tri : mInstanceTriIndices)
    {
        tri.x = remap[tri.x];
        tri.y = remap[tri.y];
        tri.z = remap[tri.z];
    }

    // Build the unique-vertex list from the canonical IDs. Each triangle corner k selects triangle
    // vertex tri[k]; the barycentric produced by sample_triangle(vertexUV[uv]) picks a specific
    // corner, so we map corner k -> vertexUV index via kCornerToUV (derived from sample_triangle).
    const uint32_t kCornerToUV[3] = {1, 0, 2};
    std::unordered_set<uint32_t> seen;
    for (uint32_t t = 0; t < triangleCount; ++t)
    {
        const uint3 tri = mInstanceTriIndices[t];
        for (uint32_t k = 0; k < 3; ++k)
        {
            const uint32_t vid = tri[k];
            if (seen.insert(vid).second)
                mInstanceVertices.push_back({vid, t, kCornerToUV[k]});
        }
    }

    logInfo(
        "PTTest: cached {} triangles, {} raw vertices welded to {} unique positions for instance {} (mesh {}).",
        triangleCount, vertexCount, uniquePositionCount, mSelectedInstanceID, meshID.get()
    );

    // Validate against the loaded model's pool: canonical IDs must index within [0, numMLPs).
    if (mNeuLobes.loaded)
    {
        uint32_t maxCanonical = 0;
        for (uint32_t v : remap)
            maxCanonical = std::max(maxCanonical, v);
        if (uniquePositionCount != mNeuLobes.numMLPs || maxCanonical >= mNeuLobes.numMLPs)
        {
            logWarning(
                "[NeuLobes] Mesh/pool mismatch: instance {} has {} unique vertices (max canonical ID {}), "
                "but the model pool is numMLPs={}. The vertexID gather will be out of range - the model was "
                "likely trained on a different mesh or with different welding.",
                mSelectedInstanceID, uniquePositionCount, maxCanonical, mNeuLobes.numMLPs
            );
        }
    }
}

void PTTest::loadNeuLobesModel(const std::string& dir)
{
    const std::filesystem::path root(dir);

    // Read manifest.json for hyperparameters; fall back to the reference config on any missing key or failure.
    bool haveLayerBlocks = false;
    std::ifstream ifs(root / "manifest.json");
    if (ifs)
    {
        try
        {
            nlohmann::json j = nlohmann::json::parse(ifs, nullptr, true /*allow exceptions*/, true /*ignore comments*/);

            // Pool size is top-level.
            mNeuLobes.numMLPs = j.value("numMLPs", mNeuLobes.numMLPs);

            // Hyperparameters are nested under "hyperparams".
            const nlohmann::json& hp = j.contains("hyperparams") ? j["hyperparams"] : j;
            mNeuLobes.peBands = hp.value("peBands", mNeuLobes.peBands);
            mNeuLobes.rank = hp.value("rank", mNeuLobes.rank);
            mNeuLobes.planeDim = hp.value("planeDim", mNeuLobes.planeDim);
            mNeuLobes.hiddenDim = hp.value("hiddenDim", mNeuLobes.hiddenDim);
            mNeuLobes.inputDim = hp.value("inputDim", mNeuLobes.inputDim);
            // Feature-plane resolution is called "planeRes" (also accept "res").
            if (hp.contains("res"))
                mNeuLobes.res = hp["res"].get<uint32_t>();
            else
                mNeuLobes.res = hp.value("planeRes", mNeuLobes.res);

            // Per-layer 4x4 block counts, read directly from manifest.layers when available.
            if (j.contains("layers") && j["layers"].is_array() && j["layers"].size() == 3)
            {
                for (int l = 0; l < 3; ++l)
                {
                    mNeuLobes.inBlk[l] = j["layers"][l].value("nin_blocks", 0u);
                    mNeuLobes.outBlk[l] = j["layers"][l].value("nout_blocks", 0u);
                }
                haveLayerBlocks = true;
            }
        }
        catch (const std::exception& e)
        {
            logWarning("[NeuLobes] Failed to parse manifest.json ({}). Using reference config.", e.what());
        }
    }
    else
    {
        logWarning("[NeuLobes] manifest.json not found in {}. Using reference config.", dir);
    }

    // Fall back to deriving per-layer block counts if the manifest didn't provide them.
    // MLP layer dims: inputDim -> hiddenDim -> hiddenDim -> outputDim.
    if (!haveLayerBlocks)
    {
        const uint32_t dims[4] = {mNeuLobes.inputDim, mNeuLobes.hiddenDim, mNeuLobes.hiddenDim, mNeuLobes.outputDim};
        auto blk = [](uint32_t n) { return (n + 3u) / 4u; };
        for (int l = 0; l < 3; ++l)
        {
            mNeuLobes.inBlk[l] = blk(dims[l]);
            mNeuLobes.outBlk[l] = blk(dims[l + 1]);
        }
    }

    auto readTensor = [&](const std::string& name)
    { return readFloatBinary(root / name); };

    // Feature tensors (raw float lists uploaded as-is; indexed manually in the shader).
    std::vector<float> vecx = readTensor("fp_vecx.bin");
    std::vector<float> matyz = readTensor("fp_matyz.bin");
    if (vecx.empty() || matyz.empty())
    {
        logWarning("[NeuLobes] Missing feature tensors in {}. Model not loaded.", dir);
        return;
    }

    const ResourceBindFlags flags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;

    // CPU -> GPU: upload feature tensors to device-local structured buffers.
    mNeuLobes.pVecX = mpDevice->createBuffer(vecx.size() * sizeof(float), flags, MemoryType::DeviceLocal, vecx.data());
    mNeuLobes.pMatYZ = mpDevice->createBuffer(matyz.size() * sizeof(float), flags, MemoryType::DeviceLocal, matyz.data());

    // Weights/biases: repack each layer's raw floats into mul-ready float4x4 / float4 blocks, then upload.
    for (int l = 0; l < 3; ++l)
    {
        std::vector<float> w = readTensor(fmt::format("w{}.bin", l));
        std::vector<float> b = readTensor(fmt::format("b{}.bin", l));
        if (w.empty() || b.empty())
        {
            logWarning("[NeuLobes] Missing weights/bias for layer {} in {}. Model not loaded.", l, dir);
            return;
        }

        const size_t numMat = w.size() / 16;
        std::vector<float4x4> mats(numMat);
        for (size_t i = 0; i < numMat; ++i)
            mats[i] = math::matrixFromCoefficients<float, 4, 4>(w.data() + i * 16);

        mNeuLobes.pW[l] = mpDevice->createBuffer(numMat * sizeof(float4x4), flags, MemoryType::DeviceLocal, mats.data());
        mNeuLobes.pB[l] = mpDevice->createBuffer(b.size() * sizeof(float), flags, MemoryType::DeviceLocal, b.data());
    }

    mNeuLobes.loaded = true;
    logInfo(
        "[NeuLobes] Loaded model from {} (peBands={}, res={}, rank={}, planeDim={}, hiddenDim={}, inputDim={}, numMLPs={}).",
        dir, mNeuLobes.peBands, mNeuLobes.res, mNeuLobes.rank, mNeuLobes.planeDim, mNeuLobes.hiddenDim, mNeuLobes.inputDim, mNeuLobes.numMLPs
    );

    // These must stay within the shader's fixed-array caps (kNeuMaxBlk/kNeuMaxIn/kNeuMaxFeat in
    // MinimalPathTracer.rt.slang). Exceeding them causes out-of-bounds writes -> garbage/bright output.
    const uint32_t kShaderMaxBlk = 16, kShaderMaxIn = 64, kShaderMaxFeat = 16;
    const uint32_t maxInBlk = std::max({mNeuLobes.inBlk[0], mNeuLobes.inBlk[1], mNeuLobes.inBlk[2]});
    const uint32_t maxOutBlk = std::max({mNeuLobes.outBlk[0], mNeuLobes.outBlk[1], mNeuLobes.outBlk[2]});
    if (mNeuLobes.inputDim > kShaderMaxIn || maxInBlk > kShaderMaxBlk || maxOutBlk > kShaderMaxBlk ||
        mNeuLobes.planeDim > kShaderMaxFeat)
    {
        logWarning(
            "[NeuLobes] Model exceeds shader caps (inputDim={} > {}, maxBlk={} > {}, planeDim={} > {}). "
            "Raise kNeuMaxIn/kNeuMaxBlk/kNeuMaxFeat in MinimalPathTracer.rt.slang.",
            mNeuLobes.inputDim, kShaderMaxIn, std::max(maxInBlk, maxOutBlk), kShaderMaxBlk, mNeuLobes.planeDim, kShaderMaxFeat
        );
    }
}

void PTTest::bindNeuLobesData(const ShaderVar& var)
{
    // Runtime toggle gating evaluation in the shader. Only enabled when the model is actually loaded.
    const bool enable = mUseNeuLobes && mNeuLobes.loaded;
    var["CB"]["gUseNeuLobes"] = enable;
    var["CB"]["gNeuBary"] = mNeuBary;
    var["CB"]["gNeuVertexID"] = mNeuVertexID;

    if (!mNeuLobes.loaded)
        return;

    auto cfg = var["CB"]["gNeuLobes"];
    cfg["peBands"] = mNeuLobes.peBands;
    cfg["res"] = mNeuLobes.res;
    cfg["rank"] = mNeuLobes.rank;
    cfg["planeDim"] = mNeuLobes.planeDim;
    cfg["numMLPs"] = mNeuLobes.numMLPs;
    cfg["inputDim"] = mNeuLobes.inputDim;
    cfg["inBlk"] = uint4(mNeuLobes.inBlk[0], mNeuLobes.inBlk[1], mNeuLobes.inBlk[2], 0);
    cfg["outBlk"] = uint4(mNeuLobes.outBlk[0], mNeuLobes.outBlk[1], mNeuLobes.outBlk[2], 0);

    var["gNeuVecX"] = mNeuLobes.pVecX;
    var["gNeuMatYZ"] = mNeuLobes.pMatYZ;
    var["gNeuW0"] = mNeuLobes.pW[0];
    var["gNeuW1"] = mNeuLobes.pW[1];
    var["gNeuW2"] = mNeuLobes.pW[2];
    var["gNeuB0"] = mNeuLobes.pB[0];
    var["gNeuB1"] = mNeuLobes.pB[1];
    var["gNeuB2"] = mNeuLobes.pB[2];
}

void PTTest::loadSHModel(const std::string& dir)
{
    const std::filesystem::path root(dir);
    std::ifstream ifs(root / "manifest.json");
    if (ifs)
    {
        try
        {
            nlohmann::json j = nlohmann::json::parse(ifs, nullptr, true, true);
            mSH.degree = j.value("degree", mSH.degree);
            mSH.numCoeffs = j.value("numBasisFuncs", (mSH.degree + 1) * (mSH.degree + 1));
        }
        catch (const std::exception& e)
        {
            logWarning("[SH] Failed to parse manifest.json ({}).", e.what());
        }
    }
    else
    {
        logWarning("[SH] manifest.json not found in {}. SH model not loaded.", dir);
        return;
    }

    std::vector<float> coeffs = readFloatBinary(root / "coeffs.bin");
    if (coeffs.empty() || mSH.numCoeffs == 0)
    {
        logWarning("[SH] Missing coeffs.bin or invalid degree in {}. SH model not loaded.", dir);
        return;
    }

    mSH.poolSize = (uint32_t)(coeffs.size() / (mSH.numCoeffs * 3));
    mSH.pCoeffs = mpDevice->createStructuredBuffer(
        sizeof(float), (uint32_t)coeffs.size(), ResourceBindFlags::ShaderResource, MemoryType::DeviceLocal, coeffs.data(), false
    );
    mSH.loaded = true;
    logInfo("[SH] Loaded from {} (degree={}, numCoeffs={}, pool={}).", dir, mSH.degree, mSH.numCoeffs, mSH.poolSize);
}

void PTTest::loadSGModel(const std::string& dir)
{
    const std::filesystem::path root(dir);
    std::ifstream ifs(root / "manifest.json");
    if (ifs)
    {
        try
        {
            nlohmann::json j = nlohmann::json::parse(ifs, nullptr, true, true);
            mSG.numSGs = j.value("numSGs", mSG.numSGs);
        }
        catch (const std::exception& e)
        {
            logWarning("[SG] Failed to parse manifest.json ({}).", e.what());
        }
    }
    else
    {
        logWarning("[SG] manifest.json not found in {}. SG model not loaded.", dir);
        return;
    }

    std::vector<float> axes = readFloatBinary(root / "axes.bin");
    std::vector<float> lambdas = readFloatBinary(root / "lambdas.bin");
    std::vector<float> amps = readFloatBinary(root / "amplitudes.bin");
    if (axes.empty() || lambdas.empty() || amps.empty() || mSG.numSGs == 0)
    {
        logWarning("[SG] Missing binaries or invalid numSGs in {}. SG model not loaded.", dir);
        return;
    }

    mSG.poolSize = (uint32_t)(lambdas.size() / mSG.numSGs);
    const ResourceBindFlags flags = ResourceBindFlags::ShaderResource;
    mSG.pAxes = mpDevice->createStructuredBuffer(sizeof(float), (uint32_t)axes.size(), flags, MemoryType::DeviceLocal, axes.data(), false);
    mSG.pLambdas = mpDevice->createStructuredBuffer(sizeof(float), (uint32_t)lambdas.size(), flags, MemoryType::DeviceLocal, lambdas.data(), false);
    mSG.pAmps = mpDevice->createStructuredBuffer(sizeof(float), (uint32_t)amps.size(), flags, MemoryType::DeviceLocal, amps.data(), false);
    mSG.loaded = true;
    logInfo("[SG] Loaded from {} (numSGs={}, pool={}).", dir, mSG.numSGs, mSG.poolSize);
}

void PTTest::exportTriFrames(const std::string& path)
{
    // Lazily build the per-triangle buffer if it hasn't been created yet (e.g. NeuLobes disabled, so
    // cacheInstanceTriangles() was never triggered by the render loop).
    if (!mpTriFrame || mVertexRemapInstanceID != mSelectedInstanceID)
        cacheInstanceTriangles();

    if (!mpTriFrame || mInstanceTriangleCount == 0)
    {
        logWarning("PTTest: no triangle-frame buffer to export (need a loaded scene and a valid triangle-mesh instance).");
        return;
    }

    RenderContext* pRenderContext = mpDevice->getRenderContext();

    // Fill the frame for ALL triangles of the selected instance via the compute pass. This matches the
    // render path's per-triangle frame math, so the whole table is populated (not just the rendered tri).
    if (mpExportTriFramesPass && mpScene)
    {
        auto var = mpExportTriFramesPass->getRootVar();
        mpScene->bindShaderData(var["gScene"]);
        var["gTriFrame"] = mpTriFrame;
        var["CB"]["gInstanceID"] = mSelectedInstanceID;
        var["CB"]["gTriangleCount"] = mInstanceTriangleCount;
        var["CB"]["gTriSampleUV"] = float2(mTriSampleU, mTriSampleV);
        mpExportTriFramesPass->execute(pRenderContext, mInstanceTriangleCount, 1, 1);
    }

    const uint32_t triCount = mInstanceTriangleCount;
    const size_t stride = sizeof(float4) * 4; // ProbeTriFrame (N, T, B, origin)
    auto pStaging = mpDevice->createStructuredBuffer(
        (uint32_t)stride, triCount, ResourceBindFlags::None, MemoryType::ReadBack, nullptr, false
    );
    pRenderContext->copyBufferRegion(pStaging.get(), 0, mpTriFrame.get(), 0, stride * triCount);
    pRenderContext->submit(true); // flush + wait so the readback data is valid.

    const float4* pData = reinterpret_cast<const float4*>(pStaging->map());
    std::ofstream ofs(path);
    if (!ofs)
    {
        logWarning("PTTest: failed to open '{}' for triangle-frame export.", path);
        pStaging->unmap();
        return;
    }
    ofs << "# triIndex Nx Ny Nz Tx Ty Tz Bx By Bz Ox Oy Oz\n";
    for (uint32_t t = 0; t < triCount; ++t)
    {
        const float4& N = pData[t * 4 + 0];
        const float4& T = pData[t * 4 + 1];
        const float4& B = pData[t * 4 + 2];
        const float4& O = pData[t * 4 + 3];
        ofs << t << " " << N.x << " " << N.y << " " << N.z << " " << T.x << " " << T.y << " " << T.z << " " << B.x << " "
            << B.y << " " << B.z << " " << O.x << " " << O.y << " " << O.z << "\n";
    }
    pStaging->unmap();
    logInfo("PTTest: exported {} triangle frames to '{}'.", triCount, path);
}

void PTTest::bindProbeReprData(const ShaderVar& var)
{
    var["CB"]["gProbeRepr"] = mProbeRepr;
    var["CB"]["gFeatInterp"] = mFeatInterp;

    var["CB"]["gSHDegree"] = mSH.degree;
    var["CB"]["gSHNumCoeffs"] = mSH.numCoeffs;
    if (mSH.loaded)
        var["gSHCoeffs"] = mSH.pCoeffs;

    var["CB"]["gSGCount"] = mSG.numSGs;
    if (mSG.loaded)
    {
        var["gSGAxes"] = mSG.pAxes;
        var["gSGLambdas"] = mSG.pLambdas;
        var["gSGAmps"] = mSG.pAmps;
    }

    var["CB"]["gSVCount"] = mSV.numSites;
    if (mSV.loaded)
    {
        var["gSVSites"] = mSV.pSites;
        var["gSVColors"] = mSV.pColors;
        var["gSVBeta"] = mSV.pBeta;
    }
}

void PTTest::loadSVModel(const std::string& dir)
{
    const std::filesystem::path root(dir);
    std::ifstream ifs(root / "manifest.json");
    if (ifs)
    {
        try
        {
            nlohmann::json j = nlohmann::json::parse(ifs, nullptr, true, true);
            mSV.numSites = j.value("numSites", mSV.numSites);
            mSV.fixedSites = j.value("fixedSites", mSV.fixedSites);
        }
        catch (const std::exception& e)
        {
            logWarning("[SV] Failed to parse manifest.json ({}).", e.what());
        }
    }
    else
    {
        logWarning("[SV] manifest.json not found in {}. SV model not loaded.", dir);
        return;
    }

    std::vector<float> sites = readFloatBinary(root / "sites.bin");
    std::vector<float> colors = readFloatBinary(root / "colors.bin");
    std::vector<float> beta = readFloatBinary(root / "beta.bin");
    if (sites.empty() || colors.empty() || beta.empty() || mSV.numSites == 0)
    {
        logWarning("[SV] Missing binaries or invalid numSites in {}. SV model not loaded.", dir);
        return;
    }
    if (!mSV.fixedSites)
    {
        // With learned (free) sites there is no site correspondence across vertices, so the
        // barycentric blend used by the shader is invalid. Blending would need an OT/matching step.
        logWarning("[SV] Model uses learned (free) sites; barycentric triangle blending is not valid. "
                   "Refit with fixed sites for correct interpolation.");
    }

    mSV.poolSize = (uint32_t)beta.size();
    const ResourceBindFlags flags = ResourceBindFlags::ShaderResource;
    mSV.pSites = mpDevice->createStructuredBuffer(sizeof(float), (uint32_t)sites.size(), flags, MemoryType::DeviceLocal, sites.data(), false);
    mSV.pColors = mpDevice->createStructuredBuffer(sizeof(float), (uint32_t)colors.size(), flags, MemoryType::DeviceLocal, colors.data(), false);
    mSV.pBeta = mpDevice->createStructuredBuffer(sizeof(float), (uint32_t)beta.size(), flags, MemoryType::DeviceLocal, beta.data(), false);
    mSV.loaded = true;
    logInfo("[SV] Loaded from {} (numSites={}, pool={}, fixedSites={}).", dir, mSV.numSites, mSV.poolSize, mSV.fixedSites);
}

void PTTest::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mpExportTriFramesPass = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    // Reset scene stats; repopulated below if we have a scene.
    mMeshCount = 0;
    mInstanceCount = 0;
    mSelectedTriangleValid = false;

    if (mpScene)
    {
        mMeshCount = mpScene->getMeshCount();
        mInstanceCount = mpScene->getGeometryInstanceCount();
        logInfo("PTTest: scene has {} meshes and {} geometry instances.", mMeshCount, mInstanceCount);

        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("PTTest: This render pass does not support custom primitives.");
        }

        // Create ray tracing program.
        ProgramDesc desc;
        desc.addShaderModules(mpScene->getShaderModules());
        desc.addShaderLibrary(kShaderFile);
        desc.setMaxPayloadSize(kMaxPayloadSizeBytes);
        desc.setMaxAttributeSize(mpScene->getRaytracingMaxAttributeSize());
        desc.setMaxTraceRecursionDepth(kMaxRecursionDepth);

        mTracer.pBindingTable = RtBindingTable::create(2, 2, mpScene->getGeometryCount());
        auto& sbt = mTracer.pBindingTable;
        sbt->setRayGen(desc.addRayGen("rayGen"));
        sbt->setMiss(0, desc.addMiss("scatterMiss"));
        sbt->setMiss(1, desc.addMiss("shadowMiss"));

        if (mpScene->hasGeometryType(Scene::GeometryType::TriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh),
                desc.addHitGroup("scatterTriangleMeshClosestHit", "scatterTriangleMeshAnyHit")
            );
            sbt->setHitGroup(
                1, mpScene->getGeometryIDs(Scene::GeometryType::TriangleMesh), desc.addHitGroup("", "shadowTriangleMeshAnyHit")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::DisplacedTriangleMesh))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection")
            );
            sbt->setHitGroup(
                1,
                mpScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh),
                desc.addHitGroup("", "", "displacedTriangleMeshIntersection")
            );
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::Curve))
        {
            sbt->setHitGroup(
                0, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::Curve), desc.addHitGroup("", "", "curveIntersection"));
        }

        if (mpScene->hasGeometryType(Scene::GeometryType::SDFGrid))
        {
            sbt->setHitGroup(
                0,
                mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid),
                desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection")
            );
            sbt->setHitGroup(1, mpScene->getGeometryIDs(Scene::GeometryType::SDFGrid), desc.addHitGroup("", "", "sdfGridIntersection"));
        }

        if (mpScene->useEnvLight())
        {
            if (!mpEnvMapSampler)
            {
                mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            }
        }

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());

        // Compute pass that fills the per-triangle frame buffer for ALL triangles of the selected
        // instance (used by the "export tri frames" button). Uses the same gScene geometry queries as
        // the render path so the exported frames match exactly.
        {
            ProgramDesc cdesc;
            cdesc.addShaderModules(mpScene->getShaderModules());
            cdesc.addShaderLibrary(kExportTriFramesShaderFile).csEntry("main");
            cdesc.addTypeConformances(mpScene->getTypeConformances());
            mpExportTriFramesPass = ComputePass::create(mpDevice, cdesc, mpScene->getSceneDefines());
        }
    }




}

void PTTest::prepareVars()
{
    FALCOR_ASSERT(mpScene);
    FALCOR_ASSERT(mTracer.pProgram);

    // Configure program.
    mTracer.pProgram->addDefines(mpSampleGenerator->getDefines());
    mTracer.pProgram->setTypeConformances(mpScene->getTypeConformances());

    // Create program variables for the current program.
    // This may trigger shader compilation. If it fails, throw an exception to abort rendering.
    mTracer.pVars = RtProgramVars::create(mpDevice, mTracer.pProgram, mTracer.pBindingTable);

    // Bind utility classes into shared data.
    auto var = mTracer.pVars->getRootVar();
    mpSampleGenerator->bindShaderData(var);
}
