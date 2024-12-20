#include "MLPInference.h"
#include "cudaUtils.h"
#define IN_NUM 24
#define HIDDEN_NUM 24
#define OUT_NUM 24
#define HALF_ACC 1

__global__ void inferInt8(int* weight, int* input, float* output, unsigned int width, unsigned int height, int debugOffset)
{
    __shared__ int W[450];
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 225)
    {
        W[2 * localIdx] = weight[2 * localIdx];
        W[2 * localIdx + 1] = weight[2 * localIdx + 1];
    }
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int offset = 0;
    int* inputVal = input + 6 * (y * width + x);

    int hiddenNum = HIDDEN_NUM;
    int hiddenPackedNum = hiddenNum / 4;
    int inNum = IN_NUM / 4;
    int outNum = 3;

    int val1[24];
    int val2[6];

    // val2[0] = quantizeInt8x4_safe(0.0050, 1.0000, 0.0100, 0.9999, scaleIn1);
    // val2[1] = quantizeInt8x4_safe( 0.0200, 0.9998, 0.0025, 0.0025, scaleIn1);

    // layer 1
    for (int k = 0; k < hiddenNum; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < inNum; j++)
        {
            val1[k] = __dp4a(inputVal[j], W[offset + k * inNum + j], val1[k]);
        }
    }
    offset += hiddenNum * inNum;

    for (int k = 0; k < hiddenPackedNum; k++)
    {
#if HALF_ACC
        val2[k] = quantizeInt8x4h_safe(
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale1),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale1),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale1),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale1),
            scaleIn2
        );

#else
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale1),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale1),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale1),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale1),
            scaleIn2
        );
#endif
    }

    // layer 2
    for (int k = 0; k < hiddenNum; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < hiddenPackedNum; j++)
        {
            val1[k] = __dp4a(val2[j], W[offset + k * hiddenPackedNum + j], val1[k]);
        }
    }
    offset += hiddenNum * hiddenPackedNum;
    for (int k = 0; k < hiddenPackedNum; k++)
    {
#if HALF_ACC
        val2[k] = quantizeInt8x4h_safe(
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale2),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale2),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale2),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale2),
            scaleIn3
        );
#else
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale2),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale2),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale2),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale2),
            scaleIn3
        );
#endif
    }

    // layer 3
    for (int k = 0; k < hiddenNum; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < hiddenPackedNum; j++)
        {
            val1[k] = __dp4a(val2[j], W[offset + k * hiddenPackedNum + j], val1[k]);
        }
    }
    offset += hiddenNum * hiddenPackedNum;
    for (int k = 0; k < hiddenPackedNum; k++)
    {
#if HALF_ACC
        val2[k] = quantizeInt8x4h_safe(
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale3),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale3),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale3),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale3),
            scaleIn4
        );
#else
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale3),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale3),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale3),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale3),
            scaleIn4
        );
#endif
    }

    // layer final
    for (int k = 0; k < outNum; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < hiddenPackedNum; j++)
        {
            val1[k] = __dp4a(val2[j], W[offset + k * hiddenPackedNum + j], val1[k]);
        }
    }
    __syncthreads();
#if HALF_ACC
    output[4 * (y * width + x) + 0] = dequantizeInt8h_relu(val1[0], dequantizeScale4);
    output[4 * (y * width + x) + 1] = dequantizeInt8h_relu(val1[1], dequantizeScale4);
    output[4 * (y * width + x) + 2] = dequantizeInt8h_relu(val1[2], dequantizeScale4);
#else
    output[4 * (y * width + x) + 0] = dequantizeInt8f_relu(val1[0], dequantizeScale4);
    output[4 * (y * width + x) + 1] = dequantizeInt8f_relu(val1[1], dequantizeScale4);
    output[4 * (y * width + x) + 2] = dequantizeInt8f_relu(val1[2], dequantizeScale4);
#endif
}

__global__ void inferFP16(__half* weighth2, __half* biash2, int* input, float* output, unsigned int width, unsigned int height)
{
    __shared__ __half weighth[2048];
    __shared__ __half biash[76];
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 76)
    {
        biash[localIdx] = biash2[localIdx];
        for (int i = 0; i < 8; i++)
        {
            weighth[localIdx * 8 + i] = weighth2[localIdx * 8 + i];
        }
    }
    else
        for (int i = 0; i < 8; i++)
            weighth[localIdx * 8 + i] = weighth2[localIdx * 8 + i];
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    float u = (x + 0.5f) / width;
    float v = (y + 0.5f) / height;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_NUM;
    int inNum = IN_NUM;
    int outNum = HIDDEN_NUM;
    int* inputVal = input + 16 * (y * width + x);

    __half val1[IN_NUM];
    __half val2[IN_NUM];

    for (int i = 0; i < 12; i++)
    {
        unpackSnorm2x16(inputVal[i], val1[2 * i], val1[2 * i + 1]);
    }

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNumFirst; ++j)
        {
            val2[k] = __hfma(weighth[inNumFirst * k + j + offset], val1[j], val2[k]);
        }
        val2[k] = __hmax(__hadd(val2[k], biash[k + biasOffset]), CUDART_ZERO_FP16);
    }

    offset += outNum * inNumFirst;
    biasOffset += outNum;

    for (int k = 0; k < outNum; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(weighth[inNum * k + j + offset], val2[j], val1[k]);
        }
        val1[k] = __hmax(__hadd(val1[k], biash[k + biasOffset]), CUDART_ZERO_FP16);
    }

    offset += outNum * inNum;
    biasOffset += outNum;

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val2[k] = __hfma(weighth[inNum * k + j + offset], val1[j], val2[k]);
        }
        val2[k] = __hmax(__hadd(val2[k], biash[k + biasOffset]), CUDART_ZERO_FP16);
    }

    offset += outNum * inNum;
    biasOffset += outNum;

    for (int k = 0; k < 3; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(weighth[inNum * k + j + offset], val2[j], val1[k]);
        }
        val1[k] = __hmax(__hadd(val1[k], biash[k + biasOffset]), CUDART_ZERO_FP16);
    }
    __syncthreads();
    output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    output[4 * (y * width + x) + 1] = __half2float(val1[1]);
    output[4 * (y * width + x) + 2] = __half2float(val1[2]);
}

__global__ void inferFP32(float* weight2, float* bias2, float* input, float* output, unsigned int width, unsigned int height)
{
    __shared__ float weight[2048];
    __shared__ float bias[75];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        if (localIdx < 75)
        {
            bias[localIdx] = bias2[localIdx];
        }
        for (int i = 0; i < 8; i++)
        {
            weight[localIdx * 8 + i] = weight2[localIdx * 8 + i];
        }
    }
    __syncthreads();

    if (x >= width || y >= height)
        return;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_NUM;
    int inNum = IN_NUM;
    int outNum = HIDDEN_NUM;
    float* inputVal = input + 24 * (y * width + x);
    // float* inputVal = input + inNumFirst * (y * width + x);
    float val1[IN_NUM];

    float val2[IN_NUM];
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNumFirst; ++j)
        {
            sum += weight[inNumFirst * k + j + offset] * inputVal[j];
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
            sum += weight[inNum * k + j + offset] * val2[j];
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
            sum += weight[outNum * k + j + offset] * val1[j];
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
            sum += weight[inNum * k + j + offset] * val2[j];
        }
        val1[k] = relu(sum + bias[k + biasOffset]);
    }

    __syncthreads();
    output[4 * (y * width + x) + 0] = val1[0];
    output[4 * (y * width + x) + 1] = val1[1];
    output[4 * (y * width + x) + 2] = val1[2];
}

void launchInferFP32(float* weight, float* bias, float* input, float* output, unsigned int width, unsigned int height)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFP32<<<dimGrid, dimBlock>>>(weight, bias, input, output, width, height);
}
void launchInferFP16(__half* weighth, __half* biash, int* input, float* output, unsigned int width, unsigned int height)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFP16<<<dimGrid, dimBlock>>>(weighth, biash, input, output, width, height);
}
void launchInferInt8(int* weight, int* input, float* output, unsigned int width, unsigned int height, int debugOffset)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8<<<dimGrid, dimBlock>>>(weight, input, output, width, height, debugOffset);
}
