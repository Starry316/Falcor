#include "MLPInference.h"


#define IN_NUM 24
#define IN_1ST_NUM 24
#define HIDDEN_NUM 24
#define OUT_NUM 24

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
__device__ void unpackFloat32ToFloat16(float packed, __half &a, __half &b) {
    // Get the bit representation of the packed float32 value
    unsigned int packed_bits = *reinterpret_cast<unsigned int*>(&packed);

    // Extract the two float16 values
    unsigned short ha_bits = packed_bits >> 16;
    unsigned short hb_bits = packed_bits & 0xFFFF;

    // Convert the bit representation back to float16
    a = *reinterpret_cast<__half*>(&ha_bits);
    b = *reinterpret_cast<__half*>(&hb_bits);
}


// The CUDA kernel. This sample simply copies the input surface.
__global__ void inference(float* weight, float* bias, float* input, float* output, unsigned int width, unsigned int height)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (input[y*width + x + 32 * width*height] == 0) return;
    // float* inputVal = input + 24 * (y * width + x);

    // output[4 * (y * width + x) + 0] = inputVal[8];
    // output[4 * (y * width + x) + 1] = inputVal[9];
    // output[4 * (y * width + x) + 2] = inputVal[10];
    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_1ST_NUM;
    int inNum = IN_NUM;
    int outNum = OUT_NUM;
    // float* inputVal = input + 32 * (y * width + x);
    float* inputVal = input + inNumFirst * (y * width + x);
    float val1[IN_NUM];

    float val2[IN_NUM];
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNumFirst; ++j)
        {
            sum += weight[inNumFirst * j + k + offset] * inputVal[j];

        }
        val2[k] = leakyrelu(sum + bias[k+biasOffset]);
    }
    offset += outNum * inNumFirst;
    biasOffset += outNum;
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[inNum * j + k + offset] * val2[j];

        }
        val1[k] = leakyrelu(sum+ bias[k+biasOffset]);
    }
    offset += outNum * inNum;
    biasOffset += outNum;

    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[outNum * j + k + offset] * val1[j];
        }
        val2[k] = leakyrelu(sum+ bias[k+biasOffset]);
    }
    offset += outNum * inNum;
    biasOffset += outNum;
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
    inference<<<dimGrid, dimBlock>>>(weight,bias, input, output, width, height);
}





__global__ void inferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height)
{

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    if (input[y*width + x + 32 * width*height] == 0) return;

     int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_1ST_NUM;
    int inNum = IN_NUM;
    int outNum = OUT_NUM;
    float* inputVal = input + 12 * (y * width + x);
    __half val1[32];

    __half val2[32];

    for(int i = 0; i < 16; i++){
        unpackFloat32ToFloat16(inputVal[i], val1[2 * i], val1[2 * i+1]);
    }


    output[4 * (y * width + x) + 0] = val1[8];
    output[4 * (y * width + x) + 1] = val1[9];
    output[4 * (y * width + x) + 2] = val1[10];

    return;
    for (int k = 0; k < outNum; ++k)
    {
        __half sum = 0;
        for (int j = 0; j < inNum; ++j)
        {


            sum = __hadd(sum, __hmul(weight[outNum * j + k + offset], val1[j]));
            // sum += weight[outNum * j + k + offset] * val1[j];

        }
        val2[k] = leakyrelu(__hadd(sum, bias[k+biasOffset]));
    }
     offset += outNum * inNum;
    biasOffset += outNum;
    for (int k = 0; k < outNum; ++k)
    {
        __half sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            // sum += weight[outNum * j + k + offset] * val2[j];
            sum = __hadd(sum, __hmul(weight[outNum * j + k + offset], val2[j]));
        }
        val1[k] = leakyrelu(__hadd(sum, bias[k+biasOffset]));
    }
    offset += outNum * inNum;
    biasOffset += outNum;
    for (int k = 0; k < outNum; ++k)
    {
        __half sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum = __hadd(sum, __hmul(weight[outNum * j + k + offset], val1[j]));
            // sum += weight[outNum * j + k + offset] * val1[j];
        }
        val2[k] = leakyrelu(__hadd(sum, bias[k+biasOffset]));
    }
    offset += outNum * inNum;
    biasOffset += outNum;
    for (int k = 0; k < 3; ++k)
    {
        __half sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            // sum += weight[4 * j + k + offset] * val2[j];
            sum = __hadd(sum, __hmul(weight[outNum * j + k + offset], val2[j]));
        }
        val1[k] = relu(__hadd(sum, bias[k+biasOffset]));
    }


    output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    output[4 * (y * width + x) + 1] = __half2float(val1[1]);
    output[4 * (y * width + x) + 2] = __half2float(val1[2]);

}
// A wrapper function that launches the kernel.
void launchNNInferenceFP16(__half* weight, __half* bias, float* input, float* output, unsigned int width, unsigned int height){
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferenceFP16<<<dimGrid, dimBlock>>>(weight,bias, input, output, width, height);
}
