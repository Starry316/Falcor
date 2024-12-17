#include <cuda_fp16.h>
#include <cuda_runtime.h>


extern void launchNNInference(float* weight, float* bias,  float* input, float* output, unsigned int width, unsigned int height);
extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);

extern void launchFP16Test(float* weight, float* bias, cudaTextureObject_t texObj, unsigned int* input, float* output, unsigned int width, unsigned int height);
extern void launchValidation(float* weight, float* bias, __half* weighth, __half* biash, unsigned int* input, float* output, unsigned int width, unsigned int height);
