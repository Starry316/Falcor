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
int packInt8x4(int x, int y, int z, int w)
{
    return (x & 0x000000ff) | ((y << 8) & 0x0000ff00) | ((z << 16) & 0x00ff0000) | ((w << 24) & 0xff000000);
}
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
    mpDisplayPass = FullScreenPass::create(getDevice(), "Samples/ShaderToy/display.ps.slang");
    mpBindInputPass = FullScreenPass::create(getDevice(), "Samples/ShaderToy/bindInput.ps.slang");

    mpTextureSynthesis = std::make_unique<TextureSynthesis>();

    // mpTextureSynthesis->readHFData("D:/textures/ubo/leather11.png", getDevice());
    mpTextureSynthesis->readHFData("D:/textures/synthetic/ganges_river_pebbles_disp_4k.png", getDevice());
    mpNBTF = std::make_unique<NBTF>(getDevice(), mNetName, true);
    mpNBTFInt8 = std::make_unique<NBTF>(getDevice(), mNetInt8Name, true);

    std::vector<float> cudaWeight =
        readBinaryFile(fmt::format("{}/media/BTF/networks/Weights_flatten_{}.bin", getProjectDirectory(), mNetName).c_str());
    std::vector<float> cudaBias =
        readBinaryFile(fmt::format("{}/media/BTF/networks/Bias_flatten_{}.bin", getProjectDirectory(), mNetName).c_str());

    mpWeightBuffer =
        getDevice()->createBuffer(cudaWeight.size() * sizeof(float), ResourceBindFlags::Shared, MemoryType::DeviceLocal, cudaWeight.data());

    mpBiasBuffer =
        getDevice()->createBuffer(cudaBias.size() * sizeof(float), ResourceBindFlags::Shared, MemoryType::DeviceLocal, cudaBias.data());

    std::vector<__half> cudaWeightFP16(cudaWeight.size());
    for (size_t i = 0; i < cudaWeight.size(); i++)
    {
        cudaWeightFP16[i] = __float2half(cudaWeight[i]);
    }
    std::vector<__half> cudaBiasFP16(cudaBias.size());
    for (size_t i = 0; i < cudaBias.size(); i++)
    {
        cudaBiasFP16[i] = __float2half(cudaBias[i]);
    }

    mpWeightFP16Buffer = getDevice()->createBuffer(
        cudaWeightFP16.size() * sizeof(__half), ResourceBindFlags::Shared, MemoryType::DeviceLocal, cudaWeightFP16.data()
    );

    mpBiasFP16Buffer = getDevice()->createBuffer(
        cudaBiasFP16.size() * sizeof(__half), ResourceBindFlags::Shared, MemoryType::DeviceLocal, cudaBiasFP16.data()
    );

    std::vector<float>().swap(cudaWeight);
    std::vector<float>().swap(cudaBias);

    std::vector<__half>().swap(cudaWeightFP16);
    std::vector<__half>().swap(cudaBiasFP16);

    cudaEventCreate(&mCudaStart);
    cudaEventCreate(&mCudaStop);

    cudaExtent extent = make_cudaExtent(400, 400, 16);
    cudaArray* d_array;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&d_array, &channelDesc, extent);

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    std::vector<float> int8Weight =
        readBinaryFile(fmt::format("{}/media/BTF/networks/Weight_int8_{}.bin", getProjectDirectory(), mNetInt8Name).c_str());

    std::vector<int> int8WeightInt(int8Weight.size() / 4);
    for (size_t i = 0; i < int8WeightInt.size(); i++)
    {
        int8WeightInt[i] =
            packInt8x4((int)int8Weight[i * 4], (int)int8Weight[i * 4 + 1], (int)int8Weight[i * 4 + 2], (int)int8Weight[i * 4 + 3]);
    }

    mpQInt8Buffer = getDevice()->createBuffer(
        int8WeightInt.size() * sizeof(int), ResourceBindFlags::Shared, MemoryType::DeviceLocal, int8WeightInt.data()
    );
    logInfo("QINT8 buffer size: " + std::to_string(int8Weight.size()));
    logInfo("QINT8 buffer  {} {} {} {}", int8Weight[0], int8Weight[1], int8Weight[2], int8Weight[3]);
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

    auto var = mpMainPass->getRootVar()["ToyCB"];
    var["iResolution"] = Falcor::float2(width, height);
    var["iGlobalTime"] = (float)getGlobalClock().getTime();
    var["gUVScaling"] = mUVScale;
    var["gSynthesis"] = mSynthesis;
    var["gDisplay"] = true;
    mpMainPass->getRootVar()["gInputColor"] = mpOutputBuffer;
    mpMainPass->getRootVar()["cudaInputBuffer"] = mpInputBuffer;
    mpTextureSynthesis->bindHFData(mpMainPass->getRootVar()["ToyCB"]["hfData"]);
    mpNBTF->bindShaderData(mpMainPass->getRootVar()["ToyCB"]["nbtf"]);

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpMainPass->getProgram(), mpMainPass->getRootVar());

    // run final pass
    mpMainPass->execute(pRenderContext, pTargetFbo);
    mpPixelDebug->endFrame(pRenderContext);
}
void ShaderToy::cudaInfer(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    auto input = (float*)mpInputBuffer->getGpuAddress();
    auto output = (float*)mpOutputBuffer->getGpuAddress();
    // timer start
    cudaEventRecord(mCudaStart, NULL);
    for (size_t i = 0; i < 10; i++)
    {
        if (mRenderType == RenderType::CUDAINT8)
            launchInferInt8(
                (int*)mpQInt8Buffer->getGpuAddress(), (int*)mpInputBuffer->getGpuAddress(), output, targetDim.x, targetDim.y, mDebugOffset
            );

        else if (mRenderType == RenderType::CUDAFP16)
        {
            launchInferFP16(
                (__half*)mpWeightFP16Buffer->getGpuAddress(),
                (__half*)mpBiasFP16Buffer->getGpuAddress(),
                (int*)mpInputBuffer->getGpuAddress(),
                output,
                targetDim.x,
                targetDim.y
            );
        }
        else
            launchInferFP32(
                (float*)mpWeightBuffer->getGpuAddress(),
                (float*)mpBiasBuffer->getGpuAddress(),
                (float*)mpInputBuffer->getGpuAddress(),
                output,
                targetDim.x,
                targetDim.y
            );
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

    auto var = mpBindInputPass->getRootVar()["CB"];
    var["iResolution"] = Falcor::float2(width, height);
    var["gUVScaling"] = mUVScale;
    var["gSynthesis"] = mSynthesis;
    var["gRenderType"] = (int)mRenderType;
    if (mRenderType == RenderType::CUDAINT8 || mRenderType == RenderType::CUDAFP16)
        mpBindInputPass->getRootVar()["cudaInputUIntBuffer"].setUav(mpInputBuffer->getUAV());
    else
        mpBindInputPass->getRootVar()["cudaInputBuffer"] = mpInputBuffer;
    mpNBTF->bindShaderData(mpBindInputPass->getRootVar()["CB"]["nbtf"]);
    mpNBTFInt8->bindShaderData(mpBindInputPass->getRootVar()["CB"]["nbtfInt8"]);

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpBindInputPass->getProgram(), mpBindInputPass->getRootVar());

    // run final pass
    mpBindInputPass->execute(pRenderContext, pTargetFbo);
    mpPixelDebug->endFrame(pRenderContext);
}

void ShaderToy::display(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    // iResolution
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);

    auto var = mpDisplayPass->getRootVar()["CB"];
    var["iResolution"] = Falcor::float2(width, height);
    var["gSynthesis"] = mSynthesis;
    mpDisplayPass->getRootVar()["gInputColor"] = mpOutputBuffer;
    mpDisplayPass->getRootVar()["cudaInputBuffer"] = mpInputBuffer;

    mpPixelDebug->beginFrame(pRenderContext, targetDim);
    mpPixelDebug->prepareProgram(mpDisplayPass->getProgram(), mpDisplayPass->getRootVar());
    // run final pass
    mpDisplayPass->execute(pRenderContext, pTargetFbo);
    mpPixelDebug->endFrame(pRenderContext);
}

void ShaderToy::onFrameRender(RenderContext* pRenderContext, const ref<Fbo>& pTargetFbo)
{
    float width = (float)pTargetFbo->getWidth();
    float height = (float)pTargetFbo->getHeight();
    Falcor::uint2 targetDim = Falcor::uint2(width, height);
    createBuffer(mpOutputBuffer, getDevice(), targetDim);
    createBuffer(mpInputBuffer, getDevice(), targetDim, 33);
    // if(mFrames<2)

    if (mRenderType == RenderType::SHADER_NN)
    {
        shaderInfer(pRenderContext, pTargetFbo);
    }
    else
    {
        if(mFrames<2)
            bindInput(pRenderContext, pTargetFbo);
        cudaInfer(pRenderContext, pTargetFbo);
        display(pRenderContext, pTargetFbo);
    }
    getTextRenderer().render(pRenderContext, getFrameRate().getMsg(), pTargetFbo, {20, 20});
    mFrames++;
}
void ShaderToy::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Falcor", {250, 200}, {20, 100});
    renderGlobalUI(pGui);
    bool dirty = false;
    dirty |= w.dropdown("Render Type", mRenderType);


    dirty |= w.slider("UV Scale", mUVScale, 0.0f, 10.0f);
    // w.slider("debugOffset", mDebugOffset, 0, 32);
    dirty |= w.checkbox("Enable Synthesis", mSynthesis);
    dirty |= w.button("Reset Timer");
    if (dirty)
    {
        mFrames = 1;
        mCudaAvgTime = mCudaTime;
    }
    w.text("CUDA time: " + std::to_string(mCudaTime) + " ms");
    w.text("CUDA avg time: " + std::to_string(mCudaAvgTime / mFrames) + " ms");
    w.text("CUDA avg time (real): " + std::to_string(mCudaAvgTime / mFrames / 10) + " ms");
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
