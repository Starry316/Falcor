#include "MLPInference.h"

#define IN_NUM 32
#define IN_1ST_NUM 32
#define HIDDEN_NUM 32
#define OUT_NUM 32
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)
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
    // return max(x, 0.0f) + min(x, 0.0f) * 0.01f;
    return max(x, 0.0f);
}
__half __device__ __forceinline__ leakyrelu(__half x)
{
    return max(x, 0.0) + min(x, 0.0) * 0.01;
}

// The CUDA kernel. This sample simply copies the input surface.
__global__ void inference(float* weight2, float* bias2, float* input, float* output, unsigned int width, unsigned int height)
{
    __shared__ float weight[3328];
    __shared__ float bias[100];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        if (localIdx < 100)
        {
            bias[localIdx] = bias2[localIdx];
        }
        for (int i = 0; i < 13; i++)
        {
            weight[localIdx * 13 + i] = weight2[localIdx * 13 + i];
        }
    }
    __syncthreads();

    if (x >= width || y >= height)
        return;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_1ST_NUM;
    int inNum = IN_NUM;
    int outNum = OUT_NUM;
    float* inputVal = input + 32 * (y * width + x);
    // float* inputVal = input + inNumFirst * (y * width + x);
    float val1[IN_NUM];

    float val2[IN_NUM];
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNumFirst; ++j)
        {
            sum += weight[inNumFirst * j + k + offset] * inputVal[j];
        }
        val2[k] = leakyrelu(sum + bias[k + biasOffset]);
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
        val1[k] = leakyrelu(sum + bias[k + biasOffset]);
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
        val2[k] = leakyrelu(sum + bias[k + biasOffset]);
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
        val1[k] = relu(sum + bias[k + biasOffset]);
    }
    __syncthreads();
    output[4 * (y * width + x) + 0] = val1[0];
    output[4 * (y * width + x) + 1] = val1[1];
    output[4 * (y * width + x) + 2] = val1[2];
}
// A wrapper function that launches the kernel.
void launchNNInference(float* weight, float* bias, float* input, float* output, unsigned int width, unsigned int height)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inference<<<dimGrid, dimBlock>>>(weight, bias, input, output, width, height);
}

inline __device__ void unpackSnorm2x16(unsigned int packed, float& a, float& b)
{
    a = __int2float_rd((int)(packed << 16) >> 16) / 32767.f;
    b = __int2float_rd((int)packed >> 16) / 32767.f;
}

inline __device__ void unpackSnorm2x16(unsigned int packed, __half& a, __half& b)
{
    a = __hdiv(__int2half_rd((int)(packed << 16) >> 16), 32767);
    b = __hdiv(__int2half_rd((int)packed >> 16), 32767);
}

inline __device__ __half hfa_relu(const __half a, const __half b)
{
    return __hmax(__hadd(a, b), CUDART_ZERO_FP16);
}

// The CUDA kernel. This sample simply copies the input surface.
__global__ void fp16test(float* weight, float* bias,cudaTextureObject_t texObj,  unsigned int* input, float* output, unsigned int width, unsigned int height)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_1ST_NUM;
    int inNum = IN_NUM;
    int outNum = OUT_NUM;
    unsigned int* inputVal = input + 16 * (y * width + x);

    __half val1[IN_NUM];
    __half val2[IN_NUM];

    for (int i = 0; i < 16; i++)
    {
        unpackSnorm2x16(inputVal[i], val1[2 * i], val1[2 * i + 1]);
    }
    val2[0] = CUDART_ZERO_FP16;
    val2[1] = CUDART_ZERO_FP16;
    val2[2] = CUDART_ZERO_FP16;
    val2[0] = __hfma(__float2half_rd(2.0f), val1[0], val2[0]);
    val2[1] = hfa_relu(val1[0], val2[0]);
    val2[2] = hfa_relu(val1[1], val2[0]);
    output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    output[4 * (y * width + x) + 1] = __half2float(val2[0]);
    output[4 * (y * width + x) + 2] = __half2float(val2[1]);
    output[4 * (y * width + x) + 3] = __half2float(val2[2]);

    // for (int k = 0; k < outNum; ++k)
    // {
    //     val2[k] = CUDART_ZERO_FP16;
    //     for (int j = 0; j < inNumFirst; ++j)
    //     {
    //         val2[k] = __hfma(__float2half_rd(weight[inNumFirst * j + k + offset]), val1[j], val2[k]);
    //     }
    //     val2[k] = __hmax(__hadd(val2[k], __float2half_rd(bias[k + biasOffset])), CUDART_ZERO_FP16);
    // }
    // offset += outNum * inNumFirst;
    // biasOffset += outNum;
    // for (int k = 0; k < outNum; ++k)
    // {
    //     val1[k] = CUDART_ZERO_FP16;
    //     for (int j = 0; j < inNumFirst; ++j)
    //     {
    //         val1[k] = __hfma(__float2half_rd(weight[inNumFirst * j + k + offset]), val2[j], val1[k]);
    //     }
    //     val1[k] = __hmax(__hadd(val1[k], __float2half_rd(bias[k + biasOffset])), CUDART_ZERO_FP16);
    // }

    // offset += outNum * inNumFirst;
    // biasOffset += outNum;
    // for (int k = 0; k < outNum; ++k)
    // {
    //     val2[k] = CUDART_ZERO_FP16;
    //     for (int j = 0; j < inNumFirst; ++j)
    //     {
    //         val2[k] = __hfma(__float2half_rd(weight[inNumFirst * j + k + offset]), val1[j], val2[k]);
    //     }
    //     val2[k] = __hmax(__hadd(val2[k], __float2half_rd(bias[k + biasOffset])), CUDART_ZERO_FP16);
    // }
    // offset += outNum * inNumFirst;
    // biasOffset += outNum;

    // for (int k = 0; k < 3; ++k)
    // {
    //     val1[k] = CUDART_ZERO_FP16;
    //     for (int j = 0; j < inNumFirst; ++j)
    //     {
    //         val1[k] = __hfma(__float2half_rd(weight[inNumFirst * j + k + offset]), val2[j], val1[k]);
    //     }
    //     // val1[k] = __hfma_relu(val1[k], bias[k+biasOffset], val1[k]);
    //     val1[k] = __hmax(__hadd(val1[k], __float2half_rd(bias[k + biasOffset])), CUDART_ZERO_FP16);
    // }

    // output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    // output[4 * (y * width + x) + 1] = __half2float(val1[1]);
    // output[4 * (y * width + x) + 2] = __half2float(val1[2]);

    // output[4 * (y * width + x) + 2] = abs(inputVal[10]);

    // output[4 * (y * width + x) + 0] = val1[0];
    // output[4 * (y * width + x) + 1] = val1[1];
    // output[4 * (y * width + x) + 2] = val1[2];
}
// A wrapper function that launches the kernel.
void launchFP16Test(float* weight, float* bias, cudaTextureObject_t texObj, unsigned int* input, float* output, unsigned int width, unsigned int height)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    fp16test<<<dimGrid, dimBlock>>>(weight, bias, texObj, input, output, width, height);
}

// The CUDA kernel. This sample simply copies the input surface.
__global__ void validation(
    float* weight,
    float* bias,
    __half* weighth2,
    __half* biash2,
    unsigned int* input,
    float* output,
    unsigned int width,
    unsigned int height
)
{
    __shared__ __half weighth[3328];
    __shared__ __half biash[100];
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 100)
    {
        biash[localIdx] = biash2[localIdx];
        for (int i = 0; i < 13; i++)
        {
            weighth[localIdx * 13 + i] = weighth2[localIdx * 13 + i];
        }
    }
    else
        for (int i = 0; i < 13; i++)
            weighth[localIdx * 13 + i] = weighth2[localIdx * 13 + i];
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_1ST_NUM;
    int inNum = IN_NUM;
    int outNum = OUT_NUM;
    unsigned int* inputVal = input + 16 * (y * width + x);

    __half val1[IN_NUM];
    __half val2[IN_NUM];

    // for (int i = 0; i < 16; i++)
    // {
    //     unpackSnorm2x16(inputVal[i], val1[2 * i], val1[2 * i + 1]);
    // }
    val1[0] = __float2half(u);
    val1[1] = __float2half(v);
    for (int i = 2; i < 32; i++)
    {
         val1[i] = __float2half((float)i / 32);
    }

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNumFirst; ++j)
        {
            val2[k] = __hfma(__float2half_rd(weighth[inNumFirst * j + k + offset]), val1[j], val2[k]);
        }
        val2[k] = __hmax(__hadd(val2[k], __float2half_rd(biash[k + biasOffset])), CUDART_ZERO_FP16);
    }

    offset += outNum * inNumFirst;
    biasOffset += outNum;

    for (int k = 0; k < outNum; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(__float2half_rd(weighth[inNum * j + k + offset]), val2[j], val1[k]);
        }
        val1[k] = __hmax(__hadd(val1[k], __float2half_rd(biash[k + biasOffset])), CUDART_ZERO_FP16);
    }

    offset += outNum * inNum;
    biasOffset += outNum;

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val2[k] = __hfma(__float2half_rd(weighth[inNum * j + k + offset]), val1[j], val2[k]);
        }
        val2[k] = __hmax(__hadd(val2[k], __float2half_rd(biash[k + biasOffset])), CUDART_ZERO_FP16);
    }

    offset += outNum * inNum;
    biasOffset += outNum;

    for (int k = 0; k < 3; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(__float2half_rd(weighth[4 * j + k + offset]), val2[j], val1[k]);
        }
        val1[k] = __hmax(__hadd(val1[k], __float2half_rd(biash[k + biasOffset])), CUDART_ZERO_FP16);
    }
    __syncthreads();
    output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    output[4 * (y * width + x) + 1] = __half2float(val1[1]);
    output[4 * (y * width + x) + 2] = __half2float(val1[2]);
}

// A wrapper function that launches the kernel.
void launchValidation(
    float* weight,
    float* bias,
    __half* weighth,
    __half* biash,
    unsigned int* input,
    float* output,
    unsigned int width,
    unsigned int height
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    validation<<<dimGrid, dimBlock>>>(weight, bias, weighth, biash, input, output, width, height);
}
