#include <cuda_fp16.h>
#include <cuda_runtime.h>


extern void launchNNInference(float* weight, float* bias,  float* input, float* output, unsigned int width, unsigned int height);
extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);

extern void launchInt8Test(float* weight, int* input, float* output, unsigned int width, unsigned int height, int debugOffset);
extern void launchValidation(float* weight, float* bias, __half* weighth, __half* biash, unsigned int* input, float* output, unsigned int width, unsigned int height);
