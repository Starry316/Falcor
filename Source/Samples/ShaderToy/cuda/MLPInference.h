#include <cuda_fp16.h>
#include <cuda_runtime.h>


extern void launchInferFP32(float* weight, float* bias,  float* input, float* output, unsigned int width, unsigned int height);

extern void launchInferInt8(int* weight, int* input, float* output, unsigned int width, unsigned int height, int debugOffset);
extern void launchInferFP16(__half* weighth, __half* biash, int* input, float* output, unsigned int width, unsigned int height);

 void testTexture(int* weight, float* testInput,cudaTextureObject_t HP, cudaTextureObject_t DP,cudaTextureObject_t UP, float* output, unsigned int width, unsigned int height);

 void testTextureFP32(float* weight, float* bias, cudaTextureObject_t HP, cudaTextureObject_t DP,cudaTextureObject_t UP, float* output, unsigned int width, unsigned int height);

// extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);
//
