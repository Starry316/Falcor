#include "Inference.h"
#include "Utils.h"

#define IN_NUM 24
#define IN_PACKED_NUM IN_NUM / 4
#define HIDDEN_NUM 32
#define HIDDEN_PACKED_NUM HIDDEN_NUM / 4
#define HALF_ACC 1
__global__ void inferInt8TexTest(
    int* weight,
    float* testInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    float uvScale
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

    int offset = 0;
    int hiddenNum = HIDDEN_NUM;
    int hiddenPackedNum = HIDDEN_PACKED_NUM;
    int inPackedNum = IN_PACKED_NUM;
    int outNum = 3;

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float u = (float)x / (float)height;
    float v = (float)y / (float)height;
    float h1 = testInput[4 * (y * width + x)];
    float h2 = testInput[4 * (y * width + x) + 1];
    float d1 = testInput[4 * (y * width + x) + 2];
    float d2 = testInput[4 * (y * width + x) + 3];

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 0);
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 1);
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

void launchInferInt8TexTest(
    int* weight,
    float* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8TexTest<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, uvScale);
}
__global__ void inferInt8Tex(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
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

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 1);
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
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8Tex<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, validMask, uvScale);
}

__global__ void inferFP32Tex(
    float* weight2,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    __shared__ float weight[3072];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        for (int i = 0; i < 12; i++)
        {
            weight[localIdx * 12 + i] = weight2[localIdx * 12 + i];
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
    int inNum = HIDDEN_NUM;
    int outNum = HIDDEN_NUM;
    float val1[HIDDEN_NUM];
    float val2[HIDDEN_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 2], u, v);
    u *= uvScale;
    v *= uvScale;
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

    val = tex2DLayered<float4>(UP, v, u, 0);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, v, u, 1);
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
        val2[k] = relu(sum);
    }
    offset += outNum * inNumFirst;
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[inNum * k + j + offset] * val2[j];
        }
        val1[k] = relu(sum);
    }
    offset += outNum * inNum;

    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[outNum * k + j + offset] * val1[j];
        }
        val2[k] = relu(sum);
    }
    offset += outNum * inNum;

    for (int k = 0; k < 3; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[inNum * k + j + offset] * val2[j];
        }
        val1[k] = relu(sum);
    }

    __syncthreads();
    output[4 * (y * width + x) + 0] = val1[0];
    output[4 * (y * width + x) + 1] = val1[1];
    output[4 * (y * width + x) + 2] = val1[2];
}

__global__ void inferFp32TexTest(
    float* weight2,
    float* testInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    float uvScale
)
{
    __shared__ float weight[3072];

    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        for (int i = 0; i < 12; i++)
        {
            weight[localIdx * 12 + i] = weight2[localIdx * 12 + i];
        }
    }
    __syncthreads();

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_NUM;
    int inNum = HIDDEN_NUM;
    int outNum = HIDDEN_NUM;
    float val1[HIDDEN_NUM];
    float val2[HIDDEN_NUM];

    float u = (float)x / (float)height;
    float v = (float)y / (float)height;
    float h1 = testInput[4 * (y * width + x)];
    float h2 = testInput[4 * (y * width + x) + 1];
    float d1 = testInput[4 * (y * width + x) + 2];
    float d2 = testInput[4 * (y * width + x) + 3];

    u *= uvScale;
    v *= uvScale;
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

    val = tex2DLayered<float4>(UP, v, u, 0);
    val1[0 + inputOffset] = val.x;
    val1[1 + inputOffset] = val.y;
    val1[2 + inputOffset] = val.z;
    val1[3 + inputOffset] = val.w;
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, v, u, 1);
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
        val2[k] = relu(sum);
    }
    offset += outNum * inNumFirst;
    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[inNum * k + j + offset] * val2[j];
        }
        val1[k] = relu(sum);
    }
    offset += outNum * inNum;

    for (int k = 0; k < outNum; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[outNum * k + j + offset] * val1[j];
        }
        val2[k] = relu(sum);
    }
    offset += outNum * inNum;

    for (int k = 0; k < 3; ++k)
    {
        float sum = 0;
        for (int j = 0; j < inNum; ++j)
        {
            sum += weight[inNum * k + j + offset] * val2[j];
        }
        val1[k] = relu(sum);
    }

    __syncthreads();
    output[4 * (y * width + x) + 0] = val1[0];
    output[4 * (y * width + x) + 1] = val1[1];
    output[4 * (y * width + x) + 2] = val1[2];
}

void launchInferFp32TexTest(
    float* weight,
    float* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFp32TexTest<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, uvScale);
}

void launchInferFP32Tex(
    float* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFP32Tex<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, validMask, uvScale);
}

__global__ void inferFp16TexTest(
    __half* weight2,
    float* testInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    float uvScale
)
{
    __shared__ __half weight[3072];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        for (int i = 0; i < 12; i++)
        {
            weight[localIdx * 12 + i] = weight2[localIdx * 12 + i];
        }
    }
    __syncthreads();

    if (x >= width || y >= height)
        return;

    int offset = 0;
    int biasOffset = 0;
    int inNumFirst = IN_NUM;
    int inNum = HIDDEN_NUM;
    int outNum = HIDDEN_NUM;
    __half val1[HIDDEN_NUM];
    __half val2[HIDDEN_NUM];

    float u = (float)x / (float)height;
    float v = (float)y / (float)height;
    float h1 = testInput[4 * (y * width + x)];
    float h2 = testInput[4 * (y * width + x) + 1];
    float d1 = testInput[4 * (y * width + x) + 2];
    float d2 = testInput[4 * (y * width + x) + 3];

    u *= uvScale;
    v *= uvScale;
    int inputOffset = 0;
    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val1[0] = __float2half_rn(val.x);
    val1[1] = __float2half_rn(val.y);
    val1[2] = __float2half_rn(val.z);
    val1[3] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, v, u, 0);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, v, u, 1);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNumFirst; ++j)
        {
            val2[k] = __hfma(weight[inNumFirst * k + j + offset], val1[j], val2[k]);
        }
        val2[k] = relu(val2[k]);
    }
    offset += outNum * inNumFirst;
    for (int k = 0; k < outNum; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(weight[inNum * k + j + offset], val2[j], val1[k]);
        }
        val1[k] = __hmax(val1[k], CUDART_ZERO_FP16);
    }
    offset += outNum * inNum;

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val2[k] = __hfma(weight[inNum * k + j + offset], val1[j], val2[k]);
        }
        val2[k] = relu(val2[k]);
    }
    offset += outNum * inNum;

    for (int k = 0; k < 3; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(weight[inNum * k + j + offset], val2[j], val1[k]);
        }
        val1[k] = __hmax(val1[k], CUDART_ZERO_FP16);
    }

    __syncthreads();
    output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    output[4 * (y * width + x) + 1] = __half2float(val1[1]);
    output[4 * (y * width + x) + 2] = __half2float(val1[2]);
}

void launchInferFp16TexTest(
    __half* weight,
    float* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFp16TexTest<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, uvScale);
}

__global__ void inferFP16Tex(
    __half* weight2,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    __shared__ __half weight[3072];

    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int localIdx = threadIdx.y * blockDim.x + threadIdx.x;
    if (localIdx < 256)
    {
        for (int i = 0; i < 12; i++)
        {
            weight[localIdx * 12 + i] = weight2[localIdx * 12 + i];
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
    int inNum = HIDDEN_NUM;
    int outNum = HIDDEN_NUM;
    __half val1[HIDDEN_NUM];
    __half val2[HIDDEN_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 2], u, v);
    u *= uvScale;
    v *= uvScale;
    int inputOffset = 0;

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val1[0] = __float2half_rn(val.x);
    val1[1] = __float2half_rn(val.y);
    val1[2] = __float2half_rn(val.z);
    val1[3] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, v, u, 0);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(UP, v, u, 1);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);
    inputOffset += 4;

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val1[0 + inputOffset] = __float2half_rn(val.x);
    val1[1 + inputOffset] = __float2half_rn(val.y);
    val1[2 + inputOffset] = __float2half_rn(val.z);
    val1[3 + inputOffset] = __float2half_rn(val.w);

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNumFirst; ++j)
        {
            val2[k] = __hfma(weight[inNumFirst * k + j + offset], val1[j], val2[k]);
        }
        val2[k] = relu(val2[k]);
    }
    offset += outNum * inNumFirst;
    for (int k = 0; k < outNum; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(weight[inNum * k + j + offset], val2[j], val1[k]);
        }
        val1[k] = __hmax(val1[k], CUDART_ZERO_FP16);
    }
    offset += outNum * inNum;

    for (int k = 0; k < outNum; ++k)
    {
        val2[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val2[k] = __hfma(weight[inNum * k + j + offset], val1[j], val2[k]);
        }
        val2[k] = relu(val2[k]);
    }
    offset += outNum * inNum;

    for (int k = 0; k < 3; ++k)
    {
        val1[k] = CUDART_ZERO_FP16;
        for (int j = 0; j < inNum; ++j)
        {
            val1[k] = __hfma(weight[inNum * k + j + offset], val2[j], val1[k]);
        }
        val1[k] = __hmax(val1[k], CUDART_ZERO_FP16);
    }

    __syncthreads();
    output[4 * (y * width + x) + 0] = __half2float(val1[0]);
    output[4 * (y * width + x) + 1] = __half2float(val1[1]);
    output[4 * (y * width + x) + 2] = __half2float(val1[2]);
}

void launchInferFP16Tex(
    __half* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferFP16Tex<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, output, width, height, validMask, uvScale);
}
