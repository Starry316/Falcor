// #include <cublasdx.hpp>
// using namespace cublasdx;
#include "cublasdxTest.h"



// The CUDA kernel. This sample simply copies the input surface.
__global__ void copyInput(float* input, float* output, unsigned int width, unsigned int height)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    output[y * width + x] = input[y * width + x] + 0.1;



}
// A wrapper function that launches the kernel.
void cublasTest(float* input, float* output, unsigned int width, unsigned int height){
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    copyInput<<<dimGrid, dimBlock>>>(input, output, width, height);
}
