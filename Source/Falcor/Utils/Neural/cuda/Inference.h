#include <cuda_fp16.h>
#include <cuda_runtime.h>
// inference without synthesis
void launchInferInt8(
    int* weight,
    int* packedInput,
    float* quantizationScales,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale
);

// inference with synthesis
void launchInferSyn(
    int* weight,
    int* packedInput,
    float* quantizationScales,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale
);



void launchInferInt8Test(
    const int* weight,
    const int* packedInput,
    const float* quantizationScales,
    const cudaTextureObject_t HP,
    const cudaTextureObject_t DP,
    const cudaTextureObject_t UP,
    float* output,
    const unsigned int width,
    const unsigned int height,
    const float uvScale
);
