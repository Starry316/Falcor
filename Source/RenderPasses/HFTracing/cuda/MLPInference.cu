#include "MLPInference.h"

// A wrapper function that launches the kernel.
void launchCopySurface(cudaSurfaceObject_t input, cudaSurfaceObject_t output, unsigned int width, unsigned int height, unsigned int format)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

}
