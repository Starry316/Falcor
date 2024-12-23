#include <cuda_fp16.h>
#include <cuda_runtime.h>

extern void launchInferFP32(
    float* weight,
    float* bias,
    float* input,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask
);

extern void launchInferInt8TexTest(
    int* weight,
    float* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
     float uvScale
);
void launchInferInt8Tex(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale
);
void launchInferFP32Tex(
    float* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale
);

void launchInferFP16Tex(
    __half* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale
);

// extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);
//
