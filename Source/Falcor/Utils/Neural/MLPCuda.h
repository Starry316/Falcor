#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace Falcor
{

class FALCOR_API MLPCuda
{
public:
    MLPCuda();

    void loadInt8(ref<Device> pDevice, std::string networkPath);
    void loadFP32(ref<Device> pDevice, std::string networkPath);
    // void loadTex(ref<Device> pDevice, std::string featurePath);

    void inferInt8(int* packedInput, float* output, int width, int height, int* valid, float scale);
    void inferInt8Histo(int* packedInput, float* output, int width, int height, int* valid, float scale);
    void inferInt8Autocov(int* packedInput, float* output, int width, int height, int* valid, float scale);
    void inferInt8Hashed(int* packedInput, float* hashedUV, float* output, int width, int height, int* valid, float scale);
    void inferFp32(int* packedInput, float* output, int width, int height, int* valid, float scale);
    void inferFp16(int* packedInput, float* output, int width, int height, int* valid, float scale);

    // for infer speed test
    void inferInt8Test(float* testInput, float* output, int width, int height, float scale);
    void inferFp32Test(float* testInput, float* output, int width, int height, float scale);
    void inferFp16Test(float* testInput, float* output, int width, int height, float scale);
    void inferInt8ACFTest(float* testInput, float* output, int width, int height, float scale);

    ref<Buffer> mpInt8Buffer;
    ref<Buffer> mpSampleBuffer;
    ref<Buffer> mpFp32Buffer;
    ref<Buffer> mpFp16Buffer;

    cudaTextureObject_t mHTexObj;
    cudaTextureObject_t mUTexObj;
    cudaTextureObject_t mDTexObj;

    cudaTextureObject_t mTTexObj;
    cudaTextureObject_t mInvTexObj;
};

} // namespace Falcor
