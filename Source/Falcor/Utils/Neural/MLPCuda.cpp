#include "MLPCuda.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "IOHelper.h"
#include "cuda/Inference.h"
#include <fstream>
namespace Falcor
{
int packInt8x4(int x, int y, int z, int w)
{
    return (x & 0x000000ff) | ((y << 8) & 0x0000ff00) | ((z << 16) & 0x00ff0000) | ((w << 24) & 0xff000000);
}

MLPCuda::MLPCuda() {}
void MLPCuda::loadFP32(ref<Device> pDevice, std::string networkPath)
{
    std::vector<float> cudaWeight = readBinaryFile(networkPath.c_str());
    // std::vector<float> cudaBias = readBinaryFile(fmt::format("{}/media/BTF/networks/Bias_flatten_{}.bin", mMediaPath, mNetName).c_str());

    mpFp32Buffer = pDevice->createBuffer(
        cudaWeight.size() * sizeof(float),
        ResourceBindFlags::Shared | ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        cudaWeight.data()
    );

    logInfo("Weight buffer size: " + std::to_string(cudaWeight.size()));

    std::vector<__half> cudaWeightFP16(cudaWeight.size());
    for (size_t i = 0; i < cudaWeight.size(); i++)
    {
        cudaWeightFP16[i] = __float2half(cudaWeight[i]);
    }

    mpFp16Buffer = pDevice->createBuffer(
        cudaWeightFP16.size() * sizeof(__half), ResourceBindFlags::Shared, MemoryType::DeviceLocal, cudaWeightFP16.data()
    );
}
void MLPCuda::loadInt8(ref<Device> pDevice, std::string networkPath)
{
    std::vector<float> int8Weight = readBinaryFile(networkPath.c_str());

    std::vector<int> int8WeightInt(int8Weight.size() / 4);
    for (size_t i = 0; i < int8WeightInt.size(); i++)
    {
        int8WeightInt[i] =
            packInt8x4((int)int8Weight[i * 4], (int)int8Weight[i * 4 + 1], (int)int8Weight[i * 4 + 2], (int)int8Weight[i * 4 + 3]);
    }

    mpInt8Buffer =
        pDevice->createBuffer(int8WeightInt.size() * sizeof(int), ResourceBindFlags::Shared, MemoryType::DeviceLocal, int8WeightInt.data());
    logInfo("QINT8 buffer size: " + std::to_string(int8Weight.size()));
    logInfo("QINT8 buffer  {} {} {} {}", int8Weight[0], int8Weight[1], int8Weight[2], int8Weight[3]);
}

void MLPCuda::inferInt8Histo(int* packedInput, float* output, int width, int height, int* valid, float scale)
{
    launchInferInt8TexHisto(
        (int*)mpInt8Buffer->getGpuAddress(),
        packedInput,
        mHTexObj,
        mDTexObj,
        mUTexObj,
        mTTexObj,
        mInvTexObj,
        output,
        width,
        height,
        valid,
        scale
    );
}

void MLPCuda::inferInt8Autocov(int* packedInput, float* output, int width, int height, int* valid, float scale)
{
    launchInferInt8TexAutocov(
        (int*)mpInt8Buffer->getGpuAddress(),
        packedInput,
        mHTexObj,
        mDTexObj,
        mUTexObj,
        mTTexObj,
        mInvTexObj,
        (float*)mpSampleBuffer->getGpuAddress(),
        output,
        width,
        height,
        valid,
        scale
    );
}

void MLPCuda::inferInt8Hashed(
    int* packedInput,
    float* output,
    int width,
    int height,
    int* valid,
    float scale,
    float scale_patch,
    int matId
)
{
    launchInferInt8TexHashed(
        (int*)mpInt8Buffer->getGpuAddress(),
        packedInput,
        mHTexObj,
        mDTexObj,
        mUTexObj,
        mTTexObj,
        mInvTexObj,
        (float*)mpSampleBuffer->getGpuAddress(),
        output,
        width,
        height,
        valid,
        scale,
        scale_patch,
        matId
    );
}

void MLPCuda::inferInt8(int* packedInput, float* output, int width, int height, int* valid, float scale)
{
    launchInferInt8Tex((int*)mpInt8Buffer->getGpuAddress(), packedInput, mHTexObj, mDTexObj, mUTexObj, output, width, height, valid, scale);
}

void MLPCuda::inferFp32(int* packedInput, float* output, int width, int height, int* valid, float scale)
{
    launchInferFP32Tex((float*)mpFp32Buffer->getGpuAddress(), packedInput, mHTexObj, mDTexObj, mUTexObj, output, width, height, valid, scale);
}

void MLPCuda::inferFp16(int* packedInput, float* output, int width, int height, int* valid, float scale)
{
    launchInferFP16Tex((__half*)mpFp16Buffer->getGpuAddress(), packedInput, mHTexObj, mDTexObj, mUTexObj, output, width, height, valid, scale);
}

void MLPCuda::inferInt8Test(float* testInput, float* output, int width, int height, float scale)
{
    launchInferInt8TexTest((int*)mpInt8Buffer->getGpuAddress(), testInput, mHTexObj, mDTexObj, mUTexObj, output, width, height,  scale);
}

void MLPCuda::inferFp32Test(float* testInput, float* output, int width, int height, float scale)
{
    launchInferFp32TexTest((float*)mpFp32Buffer->getGpuAddress(), testInput, mHTexObj, mDTexObj, mUTexObj, output, width, height,  scale);
}

void MLPCuda::inferFp16Test(float* testInput, float* output, int width, int height, float scale)
{
    launchInferFp16TexTest((__half*)mpFp16Buffer->getGpuAddress(), testInput, mHTexObj, mDTexObj, mUTexObj, output, width, height,  scale);
}


}
