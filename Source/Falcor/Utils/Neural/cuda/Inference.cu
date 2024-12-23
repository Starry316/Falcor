#include "Inference.h"
#include "Utils.h"

#define IN_NUM 24
#define IN_PACKED_NUM IN_NUM / 4
#define HIDDEN_NUM 32
#define HIDDEN_PACKED_NUM HIDDEN_NUM / 4
#define HALF_ACC 1

__global__ void inferInt8Tex(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask, float uvScale
)
{
    __shared__ int W[768];
    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        W[3 * localIdx] = weight[3 * localIdx];
        W[3 * localIdx + 1] = weight[3 * localIdx + 1];
        W[3 * localIdx + 2] = weight[3 * localIdx + 2];
    }
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    if (validMask[y * width + x] == 0)
        return;

    int offset = 0;
    int hiddenNum = HIDDEN_NUM;
    int hiddenPackedNum = HIDDEN_PACKED_NUM;
    int inPackedNum = IN_PACKED_NUM;
    int outNum = 3;

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 2], u, v);

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 0);
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(UP, v *uvScale, u * uvScale, 1);
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val2[4] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val2[5] = quantizeInt8x4f_safe(val, scaleIn1);



    // layer 1
    for (int k = 0; k < hiddenNum; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < inPackedNum; j++)
        {
            val1[k] = __dp4a(val2[j], W[offset + k * inPackedNum + j], val1[k]);
        }
    }
    offset += hiddenNum * inPackedNum;

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

void launchInferInt8Tex(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8Tex<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, validMask, uvScale);
}

__global__ void inferFP32Tex(
    float* weight2,
    float* bias2,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask
)
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
    if (validMask[y * width + x] == 0)
        return;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_NUM;
    int inNum = IN_NUM;
    int outNum = HIDDEN_NUM;
    float val1[IN_NUM];
    float val2[IN_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 2], u, v);
    int inputOffset = 0;
    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val1[0] = val.x;
    val1[1] = val.y;
    val1[2] = val.z;
    val1[3] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, u * 6.5, v * 6.5, 0);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, u * 6.5, v * 6.5, 1);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;

   for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNumFirst; ++j)
        {
            sum += weight[inNumFirst * k + j + offset] * val1[j];
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

void launchInferFP32Tex(
    float* weight,
    float* bias,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask

)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFP32Tex<<<dimGrid, dimBlock>>>(weight, bias, packedInput, HP, DP, UP, output, width, height, validMask);
}
