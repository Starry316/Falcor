#include "MLPInference.h"
#include <cuda_fp16.h>

#define IN_NUM 32
#define HIDDEN_NUM 32
#define OUT_NUM 3

float __device__ __forceinline__ relu(float x)
{
    return max(x, 0.0f);
}
__half __device__ __forceinline__ relu(__half x)
{
    return max(x, 0.0f);
}
int __device__ __forceinline__ relu(int x)
{
    return max(x, 0);
}

float __device__ __forceinline__ leakyrelu(float x)
{
    return max(x, 0.0f) + min(x, 0.0f) * 0.01f;
}
__half __device__ __forceinline__ leakyrelu(__half x)
{
    return max(x, 0.0) + min(x, 0.0) * 0.01;
}



float __device__ __forceinline__ fracf(float x)
{
    return x - floorf(x);
}

// Function to unpack two float16 values from one float32 value
__device__ void unpackFloat32ToFloat16(float packed, float &a, float &b) {
    // Get the bit representation of the packed float32 value
    unsigned int packed_bits = *reinterpret_cast<unsigned int*>(&packed);

    // Extract the two float16 values
    unsigned short ha_bits = packed_bits >> 16;
    unsigned short hb_bits = packed_bits & 0xFFFF;

    // Convert the bit representation back to float16
    __half ha = *reinterpret_cast<__half*>(&ha_bits);
    __half hb = *reinterpret_cast<__half*>(&hb_bits);

    // Convert float16 back to float32
    a = __half2float(ha);
    b = __half2float(hb);
}


// The CUDA kernel. This sample simply copies the input surface.
__global__ void prepareInput(float* weight, float* bias, float* input, float* output, unsigned int width, unsigned int height)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (input[y*width + x] == 0) return;

    int offset = 0;
    int biasOffset = 0;
    int inNum = 32;
    int outNum = 32;
    float* inputVal = input + 32 * (y * width + x) + width * height;
    float val1[32];

    float val2[32];
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[outNum * j + k + offset] * inputVal[j];

        }
        val2[k] = leakyrelu(sum + bias[k+biasOffset]);
    }
    offset += 32 * 32;
    biasOffset += 32;
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[outNum * j + k + offset] * val2[j];

        }
        val1[k] = leakyrelu(sum+ bias[k+biasOffset]);
    }
   biasOffset += 32;
    offset += 32 * 32;

    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[outNum * j + k + offset] * val1[j];
        }
        val2[k] = leakyrelu(sum+ bias[k+biasOffset]);
    }
    offset += 32 * 32;
    biasOffset += 32;
    for (int k = 0; k < 3; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[4 * j + k + offset] * val2[j];
        }
        val1[k] = relu(sum+ bias[k+biasOffset]);
    }



    output[4 * (y * width + x) + 0] = val1[0];
    output[4 * (y * width + x) + 1] = val1[1];
    output[4 * (y * width + x) + 2] = val1[2];

}
// A wrapper function that launches the kernel.
void launchNNInference(float* weight, float* bias, float* input, float* output, unsigned int width, unsigned int height){
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    prepareInput<<<dimGrid, dimBlock>>>(weight,bias, input, output, width, height);
}

