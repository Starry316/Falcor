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
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/UI/TextRenderer.h"
#include "Utils/CudaUtils.h"
FALCOR_EXPORT_D3D12_AGILITY_SDK

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

    mpPixelDebug = std::make_unique<PixelDebug>(getDevice());

    // Texture sampler
    Sampler::Desc samplerDesc;
    samplerDesc.setFilterMode(TextureFilteringMode::Linear, TextureFilteringMode::Linear, TextureFilteringMode::Linear).setMaxAnisotropy(8);
    mpLinearSampler = getDevice()->createSampler(samplerDesc);

    // Load shaders
    mpMainPass = FullScreenPass::create(getDevice(), "Samples/ShaderToy/Toy.ps.slang");
    mpDebugPass = ComputePass::create(getDevice(), "Samples/ShaderToy/InferDebug.cs.slang", "csMain");
    mpDisplayPass = FullScreenPass::create(getDevice(), "Samples/ShaderToy/display.ps.slang");
    mpBindInputPass = ComputePass::create(getDevice(), "Samples/ShaderToy/bindInput.cs.slang", "csMain");

    // mpTextureSynthesis = std::make_unique<TextureSynthesis>();

    // mpTextureSynthesis->readHFData("D:/textures/synthetic/ganges_river_pebbles_disp_4k.png", getDevice());


    mpNBTFInt8 = std::make_unique<NBTF>(getDevice(), mNetInt8Name, true);

    cudaEventCreate(&mCudaStart);
    cudaEventCreate(&mCudaStop);
}

void ShaderToy::onResize(uint32_t width, uint32_t height)
{
    mAspectRatio = (float(width) / float(height));
}

void ShaderToy::shaderInfer(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    createTex(mpOutColor, getDevice(), targetDim);

    auto var = mpDebugPass->getRootVar()["ToyCB"];
    var["iResolution"] = Falcor::float2(width, height);
    var["iGlobalTime"] = (float)getGlobalClock().getTime();
    var["gUVScaling"] = mUVScale;
    var["gSynthesis"] = mSynthesis;
    var["gDebugMLP"] = mDebugMLP;
    var["gWo"] = mWo;
    var["gWi"] = mWi;
    mpDebugPass->getRootVar()["ouputColor"] = mpOutColor;
    // mpTextureSynthesis->bindHFData(mpDebugPass->getRootVar()["ToyCB"]["hfData"]);
    mpNBTFInt8->bindShaderData(mpDebugPass->getRootVar()["ToyCB"]["nbtf"]);
    mpNBTFInt8->mpMLP->bindDebugData(mpDebugPass->getRootVar()["ToyCB"]["nbtf"]["mlp"], mpNBTFInt8->mpMLPCuda->mpFp32Buffer);

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpDebugPass->getProgram(), mpDebugPass->getRootVar());

    // run final pass
    mpDebugPass->execute(pRenderContext, targetDim.x, targetDim.y);
    mpPixelDebug->endFrame(pRenderContext);
}
void ShaderToy::cudaInfer(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    auto output = (float*)mpOutputBuffer->getGpuAddress();
    // timer start
    cudaEventRecord(mCudaStart, NULL);
    for (size_t i = 0; i < mCudaInferTimes; i++)
    {
        if (mRenderType == RenderType::CUDAINT8)
            mpNBTFInt8->mpMLPCuda->inferInt8Test((float*)mpTestInput->getGpuAddress(), output, targetDim.x, targetDim.y, mUVScale);
        else if (mRenderType == RenderType::CUDAFP16)
            mpNBTFInt8->mpMLPCuda->inferFp16Test((float*)mpTestInput->getGpuAddress(), output, targetDim.x, targetDim.y, mUVScale);
        else
            mpNBTFInt8->mpMLPCuda->inferFp32Test((float*)mpTestInput->getGpuAddress(), output, targetDim.x, targetDim.y, mUVScale);
    }

    // timer end
    // cudaDeviceSynchronize();
    cudaEventRecord(mCudaStop, NULL);
    cudaEventSynchronize(mCudaStop);
    cudaEventElapsedTime(&mCudaTime, mCudaStart, mCudaStop);
    mCudaAvgTime += mCudaTime;
}
void ShaderToy::bindInput(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    createBuffer(mpTestInput, getDevice(), targetDim);

    auto var = mpBindInputPass->getRootVar()["CB"];
    var["iResolution"] = Falcor::float2(width, height);
    var["gUVScale"] = mUVScale;
    var["gSynthesis"] = mSynthesis;
    var["gWo"] = mWo;
    var["gWi"] = mWi;

    mpBindInputPass->getRootVar()["testInput"] = mpTestInput;

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpBindInputPass->getProgram(), mpBindInputPass->getRootVar());

    // run final pass
    mpBindInputPass->execute(pRenderContext, targetDim.x, targetDim.y);
    mpPixelDebug->endFrame(pRenderContext);
}

// fill the screen with the shader/cuda output
void ShaderToy::display(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    auto var = mpDisplayPass->getRootVar()["CB"];
    var["iResolution"] = Falcor::float2(width, height);
    var["gSynthesis"] = mSynthesis;
    var["gShowShader"] = mShowShader || mRenderType == RenderType::SHADER_NN;
    mpDisplayPass->getRootVar()["cudaColor"] = mpOutputBuffer;
    mpDisplayPass->getRootVar()["ouputColor"] = mpOutColor;

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpDisplayPass->getProgram(), mpDisplayPass->getRootVar());
    mpDisplayPass->execute(pRenderContext, pTargetFbo);
    mpPixelDebug->endFrame(pRenderContext);
}

void ShaderToy::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);
    createBuffer(mpOutputBuffer, getDevice(), targetDim);

    if (mRenderType == RenderType::SHADER_NN)
    {
        shaderInfer(pRenderContext, pTargetFbo);
    }
    else
    {
        if (mFrames < 2)
            bindInput(pRenderContext, pTargetFbo);
        cudaInfer(pRenderContext, pTargetFbo);
    }
    display(pRenderContext, pTargetFbo);
    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
    mFrames++;
}
void ShaderToy::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200}, {20, 100});
    renderGlobalUI(pGui);
    bool dirty = false;
    dirty |= w.dropdown("Render Type", mRenderType);
    dirty |= w.slider("Wi", mWi, 0.0f, 1.0f);
    dirty |= w.slider("Wo", mWo, 0.0f, 1.0f);

    dirty |= w.slider("UV Scale", mUVScale, 0.0f, 10.0f);
    dirty |= w.checkbox("Enable Synthesis", mSynthesis);
    dirty |= w.checkbox("Show shader", mShowShader);
    dirty |= w.checkbox("debug mlp", mDebugMLP);
    dirty |= w.button("Reset Timer");
    dirty |= w.slider("CUDA infer times", mCudaInferTimes, 1, 20);
    if (dirty)
    {
        mFrames = 1;
        mCudaAvgTime = mCudaTime;
    }

    w.text("CUDA time: " + std::to_string(mCudaTime) + " ms");
    w.text("CUDA avg time: " + std::to_string(mCudaAvgTime / mFrames) + " ms");
    w.text("CUDA avg time (real): " + std::to_string(mCudaAvgTime / mFrames / mCudaInferTimes) + " ms");
    mpPixelDebug->renderUI(w);
}
int runMain(int argc, char** argv)
{
    SampleAppConfig config;
    config.windowDesc.width = 1920;
    config.windowDesc.height = 1080;
    config.windowDesc.resizableWindow = true;
    config.windowDesc.enableVSync = false;
    config.windowDesc.title = "Falcor Shader Toy";

    ShaderToy shaderToy(config);
    return shaderToy.run();
}
void printCudaDeviceProperties(int device)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    std::cout << "Device " << device << ": " << deviceProp.name << std::endl;
    std::cout << "  Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "  Total global memory: " << deviceProp.totalGlobalMem << " bytes" << std::endl;
    std::cout << "  Shared memory per block: " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
    std::cout << "  Registers per block: " << deviceProp.regsPerBlock << std::endl;
    std::cout << "  Warp size: " << deviceProp.warpSize << std::endl;
    std::cout << "  Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "  Max threads per multiprocessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max blocks per multiprocessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "  Max grid size: (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2]
              << ")" << std::endl;
    std::cout << "  Max block dimensions: (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", "
              << deviceProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Memory clock rate: " << deviceProp.memoryClockRate << " kHz" << std::endl;
    std::cout << "  Memory bus width: " << deviceProp.memoryBusWidth << " bits" << std::endl;
    std::cout << "  L2 cache size: " << deviceProp.l2CacheSize << " bytes" << std::endl;
    std::cout << "  Number of multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "  Clock rate: " << deviceProp.clockRate << " kHz" << std::endl;
    std::cout << "  Concurrent kernels: " << (deviceProp.concurrentKernels ? "Yes" : "No") << std::endl;
    std::cout << "  ECC enabled: " << (deviceProp.ECCEnabled ? "Yes" : "No") << std::endl;
}

int main(int argc, char** argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device)
    {
        printCudaDeviceProperties(device);
    }

    return catchAndReportAllExceptions([&]() { return runMain(argc, argv); });
}
