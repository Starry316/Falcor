/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include <fstream>
#include "HFTracing.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Utils/CudaUtils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, HFTracing>();
}

namespace
{
std::vector<float> readBinaryFile(const char* filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        logError("[MLP] Unable to open file {}", filename);
        return std::vector<float>();
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        logError("[MLP] Error reading file {}", filename);
        return std::vector<float>();
    }
    file.close();
    return buffer;
}
const char kShaderFile[] = "RenderPasses/HFTracing/MinimalPathTracer.rt.slang";

// Ray tracing settings that affect the traversal stack size.
// These should be set as small as possible.
const uint32_t kMaxPayloadSizeBytes = 76u;
const uint32_t kMaxRecursionDepth = 2u;

const char kInputViewDir[] = "viewW";

const ChannelList kInputChannels = {
    // clang-format off
    // { "vbuffer",        "gVBuffer",     "Visibility buffer in packed format" },
    // { kInputViewDir,    "gViewW",       "World-space view direction (xyz float format)", true /* optional */ },
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
} // namespace
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
            logInfo("Recreating texture");
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
            logInfo("Recreating Buffer");
            buf = device->createBuffer(
                targetDim.x * targetDim.y * itemSize * sizeof(float),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr
            );
        }
    }
}
int packInt8x4(int x, int y, int z, int w)
{
    return (x & 0x000000ff) | ((y << 8) & 0x0000ff00) | ((z << 16) & 0x00ff0000) | ((w << 24) & 0xff000000);
}

HFTracing::HFTracing(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);

    // Create a sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_UNIFORM);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
    FALCOR_ASSERT(mpSampleGenerator);
}

void HFTracing::parseProperties(const Properties& props)
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
            logWarning("Unknown property '{}' in HFTracing properties.", key);
    }
}

Properties HFTracing::getProperties() const
{
    Properties props;
    props[kMaxBounces] = mMaxBounces;
    props[kComputeDirect] = mComputeDirect;
    props[kUseImportanceSampling] = mUseImportanceSampling;
    return props;
}

RenderPassReflection HFTracing::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;

    // Define our input/output channels.
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);

    return reflector;
}

void generateMaxMip(RenderContext* pRenderContext, ref<Texture> pTex)
{
    for (uint32_t m = 0; m < pTex->getMipCount() - 1; m++)
    {
        auto srv = pTex->getSRV(m, 1, 0, 1);
        auto rtv = pTex->getRTV(m + 1, 0, 1);
        // only the first channel is used
        const TextureReductionMode redModes[] = {
            TextureReductionMode::Max,
            TextureReductionMode::Min,
            TextureReductionMode::Max,
            TextureReductionMode::Standard,
        };
        const Falcor::float4 componentsTransform[] = {
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
            Falcor::float4(1.0f, 0.0f, 0.0f, 0.0f),
        };
        pRenderContext->blit(
            srv, rtv, RenderContext::kMaxRect, RenderContext::kMaxRect, TextureFilteringMode::Linear, redModes, componentsTransform
        );
    }
}

void HFTracing::createMaxMip(RenderContext* pRenderContext, const RenderData& renderData)
{
    // If we have no scene, just clear the outputs and return.
    if (mpHF.get() == nullptr)
    {
        return;
    }
    if (mpHFMaxMip.get() != nullptr)
    {
        return;
    }
    // int windowSize = pow(2, 5);
    int windowSize = 1;
    int mipHeight = mpHF->getHeight() / windowSize;
    int mipWidth = mpHF->getWidth() / windowSize;
    mpHFMaxMip = mpDevice->createTexture2D(
        mipWidth,
        mipHeight,
        ResourceFormat::R32Float,
        1,
        1,
        nullptr,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess
    );
    // Get dimensions of ray dispatch.
    Falcor::uint2 targetDim = Falcor::uint2(mipWidth, mipHeight);
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    auto var = mpCreateMaxMipPass->getRootVar();
    var["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    var["PerFrameCB"]["gLod"] = 5;
    var["gInputHeightMap"].setSrv(mpHF->getSRV());
    // var["gOutputHeightMap"].setUav(mpHFMaxMip->getUAV());
    var["gOutputHeightMap"].setUav(mpShellHF->getUAV());

    mpCreateMaxMipPass->execute(pRenderContext, targetDim.x, targetDim.y);
}

void HFTracing::nnInferPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    auto var = mpInferPass->getRootVar();
    var["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    var["PerFrameCB"]["gApplySyn"] = mApplySyn;
    var["PerFrameCB"]["gCurvatureParas"] = mCurvatureParas;
    var["PerFrameCB"]["gDebugMLP"] = mMLPDebug;
    var["cudaVaildBuffer"] = mpVaildBuffer;
    var["gOutputColor"] = renderData.getTexture("color");
    var["btfInput"] = mpPackedInputBuffer;
    mpNBTFInt8->bindShaderData(var["PerFrameCB"]["nbtf"]);
    mpNBTFInt8->mpMLP->bindDebugData(var["PerFrameCB"]["nbtf"]["mlp"], mpNBTFInt8->mpMLPCuda->mpFp32Buffer);

    mpPixelDebug->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    mpPixelDebug->prepareProgram(mpInferPass->getProgram(), mpInferPass->getRootVar());

    mpInferPass->execute(pRenderContext, targetDim.x, targetDim.y);
    mpPixelDebug->endFrame(pRenderContext);
    pRenderContext->submit(false);
    pRenderContext->signal(mpFence1.get());
    mpFence1->wait();
}
void HFTracing::cudaInferPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    createBuffer(mpOutputBuffer, mpDevice, targetDim, 4);
    cudaEventRecord(mCudaStart, NULL);
    for (int i = 0; i < cudaInferTimes; i++)
    {
        if (mInferType == InferType::CUDAINT8)
            mpNBTFInt8->mpMLPCuda->inferInt8(
                (int*)mpPackedInputBuffer->getGpuAddress(),
                (float*)mpOutputBuffer->getGpuAddress(),
                targetDim.x,
                targetDim.y,
                (int*)mpVaildBuffer->getGpuAddress(),
                mCurvatureParas.z
            );

        else if (mInferType == InferType::CUDAFP16)

            mpNBTFInt8->mpMLPCuda->inferFp16(
                (int*)mpPackedInputBuffer->getGpuAddress(),
                (float*)mpOutputBuffer->getGpuAddress(),
                targetDim.x,
                targetDim.y,
                (int*)mpVaildBuffer->getGpuAddress(),
                mCurvatureParas.z
            );
        else
            mpNBTFInt8->mpMLPCuda->inferFp32(
                (int*)mpPackedInputBuffer->getGpuAddress(),
                (float*)mpOutputBuffer->getGpuAddress(),
                targetDim.x,
                targetDim.y,
                (int*)mpVaildBuffer->getGpuAddress(),
                mCurvatureParas.z
            );
    }

    // timer end
    cudaDeviceSynchronize();
    cudaEventRecord(mCudaStop, NULL);
    cudaEventSynchronize(mCudaStop);
    cudaEventElapsedTime(&mCudaTime, mCudaStart, mCudaStop);
    mCudaAvgTime += mCudaTime;
    mCudaAccumulatedFrames++;
}

void HFTracing::displayPass(RenderContext* pRenderContext, const RenderData& renderData)
{
    Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);
    auto var = mpDisplayPass->getRootVar();
    var["PerFrameCB"]["gRenderTargetDim"] = targetDim;
    var["gOutputColor"] = renderData.getTexture("color");
    var["gInputColor"] = mpOutputBuffer;
    var["cudaVaildBuffer"] = mpVaildBuffer;
    // mpPixelDebug->beginFrame(pRenderContext, renderData.getDefaultTextureDims());
    // mpPixelDebug->prepareProgram(mpDisplayPass->getProgram(), mpDisplayPass->getRootVar());
    mpDisplayPass->execute(pRenderContext, targetDim.x, targetDim.y);
    // mpPixelDebug->endFrame(pRenderContext);
    // pRenderContext->submit(false);
    // pRenderContext->signal(mpFence2.get());
    // mpFence2->wait();
}
void HFTracing::renderHF(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto& dict = renderData.getDictionary();
    // Get dimensions of ray dispatch.
    const Falcor::uint2 targetDim = renderData.getDefaultTextureDims();
    FALCOR_ASSERT(targetDim.x > 0 && targetDim.y > 0);

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            // lightingChanged = true;
            // mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            // lightingChanged = true;
            // mRecompile = true;
        }
    }

    createBuffer(mpVaildBuffer, mpDevice, targetDim, 1);
    createBuffer(mpPackedInputBuffer, mpDevice, targetDim, 4);

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    // Configure depth-of-field.
    const bool useDOF = mpScene->getCamera()->getApertureRadius() > 0.f;
    if (useDOF && renderData[kInputViewDir] == nullptr)
    {
        logWarning("Depth-of-field requires the '{}' input. Expect incorrect shading.", kInputViewDir);
    }

    // Specialize program.
    // These defines should not modify the program vars. Do not trigger program vars re-creation.
    mTracer.pProgram->addDefine("MAX_BOUNCES", std::to_string(mMaxBounces));
    mTracer.pProgram->addDefine("COMPUTE_DIRECT", mComputeDirect ? "1" : "0");
    mTracer.pProgram->addDefine("USE_IMPORTANCE_SAMPLING", mUseImportanceSampling ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
    mTracer.pProgram->addDefine("USE_ENV_BACKGROUND", mpScene->useEnvBackground() ? "1" : "0");
    if (mRenderType == RenderType::RT)
        mTracer.pProgram->addDefine("RT");
    else
        mTracer.pProgram->removeDefine("RT");
    if (mRenderType == RenderType::WAVEFRONT_SHADER_NN)
        mTracer.pProgram->addDefine("WAVEFRONT_SHADER_NN");
    else
        mTracer.pProgram->removeDefine("WAVEFRONT_SHADER_NN");


    if (mContactRefinement)
        mTracer.pProgram->addDefine("CONTACT_REFINEMENT");
    else
        mTracer.pProgram->removeDefine("CONTACT_REFINEMENT");
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
    var["CB"]["gControlParas"] = mControlParas;
    var["CB"]["gCurvatureParas"] = mCurvatureParas;
    var["CB"]["gApplySyn"] = mApplySyn;
    var["CB"]["gDebugPrism"] = mDebugPrism;
    var["CB"]["gShowTracedHF"] = mShowTracedHF;
    var["CB"]["gTracedShadowRay"] = mTracedShadowRay;
    var["CB"]["gRenderTargetDim"] = targetDim;

    mpTextureSynthesis->bindHFData(var["CB"]["hfData"]);
    if (mpEnvMapSampler)
        mpEnvMapSampler->bindShaderData(var["CB"]["envMapSampler"]);

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

    // Bind textures
    var["gHF"].setSrv(mpHF->getSRV());
    var["gShellHF"].setSrv(mpShellHF->getSRV());
    var["cudaVaildBuffer"] = mpVaildBuffer;
    var["packedInput"] = mpPackedInputBuffer;
    var["gMaxSampler"] = mpMaxSampler;
    mpScene->raytrace(pRenderContext, mTracer.pProgram.get(), mTracer.pVars, Falcor::uint3(targetDim, 1));
    pRenderContext->submit(false);
    pRenderContext->signal(mpFence2.get());
    mpFence2->wait();
}

void HFTracing::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
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
    // Update refresh flag if options that affect the output have changed.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, RenderPassRefreshFlags::None);
        dict[Falcor::kRenderPassRefreshFlags] = flags | Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        mOptionsChanged = false;
    }
    mFrameCount++;

    // ======================================================================================================
    // Step 1: Trace HF to get BTF input, the input data is packed into mpPackedInputBuffer.
    // A valid hit mask mpVaildBuffer stores the screen space valid hits (1: valid, 0: invalid)
    renderHF(pRenderContext, renderData);

    // ======================================================================================================
    // Step 2: Inference the BTF input to get the output color
    if (mRenderType == RenderType::WAVEFRONT_SHADER_NN && mInferType == InferType::SHADER)
    {
        nnInferPass(pRenderContext, renderData);
    }

    if (mRenderType == RenderType::WAVEFRONT_SHADER_NN && mInferType != InferType::SHADER)
    {
        cudaInferPass(pRenderContext, renderData);
        displayPass(pRenderContext, renderData);
    }
}

void HFTracing::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.dropdown("Render Type", mRenderType);
    dirty |= widget.dropdown("Infer Type", mInferType);

    // dirty |= widget.var("Max bounces", mMaxBounces, 0u, 1u << 16);
    // widget.tooltip("Maximum path length for indirect illumination.\n0 = direct only\n1 = one indirect bounce etc.", true);

    // dirty |= widget.checkbox("Evaluate direct illumination", mComputeDirect);
    // widget.tooltip("Compute direct illumination.\nIf disabled only indirect is computed (when max bounces > 0).", true);

    // dirty |= widget.checkbox("Use importance sampling", mUseImportanceSampling);
    // widget.tooltip("Use importance sampling for materials", true);

    dirty |= widget.slider("HF Footprint Scale", mControlParas.x, 0.1f, 100.0f);
    widget.tooltip("Increse = less marching steps", true);
    dirty |= widget.slider("LoD Scale", mControlParas.y, 1.0f, 100.0f);
    widget.tooltip("Scale the LoD", true);
    dirty |= widget.slider("HF Offset", mControlParas.z, 0.0f, 1.0f);
    widget.tooltip("height = Scale * h + Offset", true);
    dirty |= widget.slider("HF Scale", mControlParas.w, 0.0f, 1.0f);
    widget.tooltip("height = Scale * h + Offset", true);

    dirty |= widget.slider("D", mCurvatureParas.x, 0.0f, 1.0f);
    widget.tooltip("Max height to mesh surface, i.e., the HF tracing starting height", true);
    dirty |= widget.slider("UV Scale", mCurvatureParas.z, 0.0f, 50.0f);
    widget.tooltip("Scale the uv coords", true);

    dirty |= widget.checkbox("Traced Shadow Ray", mTracedShadowRay);
    dirty |= widget.slider("Shadow ray offset", mCurvatureParas.y, 0.0f, 1.0f);
    widget.tooltip("Position offset along with the normal dir. To avoid self-occlusion", true);
    // dirty |= widget.slider("mip scale", mCurvatureParas.w, 0.0f, 11.0f);

    dirty |= widget.checkbox("Contact Refinement", mContactRefinement);
    widget.tooltip("use contact refinement tracing", true);
    dirty |= widget.checkbox("Apply Synthesis", mApplySyn);

    dirty |= widget.checkbox("Show Traced HF", mShowTracedHF);
    dirty |= widget.checkbox("Use float4", mMLPDebug);
    widget.tooltip("Use float4 in shader inference (debug only)", true);

    dirty |= widget.slider("CUDA infer times", cudaInferTimes, 1, 20);
    widget.tooltip("For speed test, run cuda infer multiple times to get the avg running time.", true);
    widget.text("CUDA time: " + std::to_string(mCudaTime) + " ms");
    widget.text("CUDA avg time: " + std::to_string(mCudaAvgTime / mCudaAccumulatedFrames) + " ms");
    widget.text("CUDA real avg time: " + std::to_string(mCudaAvgTime / mCudaAccumulatedFrames / cudaInferTimes) + " ms");
    widget.tooltip("This is the real cuda running time", true);

    if (widget.button("Reset Timer"))
    {
        mCudaAvgTime = mCudaTime;
        mCudaAccumulatedFrames = 1;
    }

    mpPixelDebug->renderUI(widget);

    // If rendering options that modify the output have changed, set flag to indicate that.
    // In execute() we will pass the flag to other passes for reset of temporal data etc.
    if (dirty)
    {
        mCudaAvgTime = mCudaTime;
        mCudaAccumulatedFrames = 1;
        mOptionsChanged = true;
    }
}

void HFTracing::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    // Clear data for previous scene.
    // After changing scene, the raytracing program should to be recreated.
    mTracer.pProgram = nullptr;
    mTracer.pBindingTable = nullptr;
    mTracer.pVars = nullptr;
    mFrameCount = 0;

    // Set new scene.
    mpScene = pScene;

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("HFTracing: This render pass does not support custom primitives.");
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

        // logInfo("[HF Tracing] Total mesh num: {} ", mpScene->getMeshCount());
        // for (uint i = 0; i < mpScene->getMeshCount(); i++)
        // {
        //     logInfo("[HF Tracing] Mesh: #{} Triangle: {}", i, mpScene->getMesh(MeshID{i}).getTriangleCount());
        // }

        // mpScene->getMeshVerticesAndIndices(MeshID meshID, const std::map<std::string, ref<Buffer>>& buffers);

        mTracer.pProgram = Program::create(mpDevice, desc, mpScene->getSceneDefines());
    }
    // Read in textures, we use a constant texture now
    mpHF = Texture::createFromFile(
        mpDevice,
        fmt::format("{}/media/BTF/scene/textures/{}", mMediaPath, mHFFileName).c_str(),
        true,
        false,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
    );
    // Read in textures, we use a constant texture now
    mpShellHF = Texture::createFromFile(
        mpDevice,
        fmt::format("{}/media/BTF/scene/textures/{}", mMediaPath, mShellHFFileName).c_str(),
        true,
        false,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
    );
    // mpColor = Texture::createFromFile(
    //     mpDevice,
    //     fmt::format("{}/media/BTF/scene/textures/{}", mMediaPath, mShellHFFileName).c_str(),
    //     true,
    //     true,
    //     ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget
    // );
    generateMaxMip(pRenderContext, mpShellHF);
    generateMaxMip(pRenderContext, mpHF);

    // Create max sampler for texel fetch.
    Sampler::Desc samplerDesc = Sampler::Desc();
    // Max reductions.
    samplerDesc.setReductionMode(TextureReductionMode::Max);
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    mpMaxSampler = mpDevice->createSampler(samplerDesc);
    // mpMaxSampler->breakStrongReferenceToDevice();

    mpTextureSynthesis = std::make_unique<TextureSynthesis>();
    mpTextureSynthesis->readHFData(fmt::format("{}/media/BTF/scene/textures/{}", mMediaPath, mShellHFFileName).c_str(), mpDevice);
    generateMaxMip(pRenderContext, mpTextureSynthesis->mpHFT);

    mpNBTFInt8 = std::make_unique<NBTF>(mpDevice, mNetInt8Name, true);

    DefineList defines = mpScene->getSceneDefines();
    mpVisualizeMapsPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/VisualizeMaps.cs.slang", "csMain", defines);
    mpCreateMaxMipPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/CreateMaxMip.cs.slang", "csMain", defines);
    mpInferPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/Inference.cs.slang", "csMain", defines);
    mpDisplayPass = ComputePass::create(mpDevice, "RenderPasses/HFTracing/Display.cs.slang", "csMain", defines);

    cudaEventCreate(&mCudaStart);
    cudaEventCreate(&mCudaStop);
    // auto kernel = createNVRTCProgram();

    mpFence = mpDevice->createFence();
    mpFence->breakStrongReferenceToDevice();

    mpFence1 = mpDevice->createFence();
    mpFence1->breakStrongReferenceToDevice();

    mpFence2 = mpDevice->createFence();
    mpFence2->breakStrongReferenceToDevice();
}

void HFTracing::prepareVars()
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
