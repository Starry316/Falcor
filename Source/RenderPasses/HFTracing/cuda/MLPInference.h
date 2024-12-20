#include <cuda_fp16.h>
#include <cuda_runtime.h>


extern void launchInferFP32(float* weight, float* bias,  float* input, float* output, unsigned int width, unsigned int height, int* validMask);

extern void launchInferInt8(int* weight, int* input, float* output, unsigned int width, unsigned int height, int* validMask);
extern void launchInferFP16(__half* weighth, __half* biash, int* input, float* output, unsigned int width, unsigned int height, int* validMask);

// extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);
//
