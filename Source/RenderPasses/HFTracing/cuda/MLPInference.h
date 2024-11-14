#include <cuda_fp16.h>
#include <cuda_runtime.h>
extern void launchCopySurface(
    cudaSurfaceObject_t input,
    cudaSurfaceObject_t output,
    unsigned int width,
    unsigned int height,
    unsigned int format
);

extern void launchNNInference(float* weight, float* bias,  float* input,float* output, unsigned int width, unsigned int height);
extern void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height);
