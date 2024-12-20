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

extern void launchInferInt8(int* weight, int* input, float* output, unsigned int width, unsigned int height, int* validMask);
extern void launchInferFP16(
    __half* weighth,
    __half* biash,
    int* input,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask
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
    int* validMask
);
void launchInferFP32Tex(
    float* weight,
    float* bias,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask
);
// extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);
//
