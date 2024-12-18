#include "MLPInference.h"

#define IN_NUM 24
#define IN_1ST_NUM 24
#define HIDDEN_NUM 24
#define OUT_NUM 24
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)

#define TEST_H 0

__constant__ float scaleIn1 = 0.0035758037120103836;
__constant__ float scaleOut1 = 0.003539581084623933;
__constant__ float dequantizeScale1 = 1.2656847502512392e-05;
__constant__ float scaleIn2 = 0.0021900988649576902;
__constant__ float scaleOut2 = 0.00467855716124177;
__constant__ float dequantizeScale2 = 1.0246502824884374e-05;
__constant__ float scaleIn3 = 0.0011810491560027003;
__constant__ float scaleOut3 = 0.011032089591026306;
__constant__ float dequantizeScale3 = 1.3029440196987707e-05;
__constant__ float scaleIn4 = 0.0019884631037712097;
// __constantfloat scaleOut4 = 0.014394178986549377;
__constant__ float dequantizeScale4 = 2.8622293029911816e-05;

float __device__ __forceinline__ relu(float x)
{
    return max(x, 0.0f);
}
__half __device__ __forceinline__ relu(__half x)
{
    return __hmax(x, CUDART_ZERO_FP16);
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

inline __device__ short2 packInt2x16(int a)
{
    return make_short2((short)((a << 16) >> 16), (short)(a >> 16));
}

inline __device__ int clampInt8(int a)
{
    return min(127, max(-127, a));
}
inline __device__ int quantizeInt8x4f_safe(float a, float b, float c, float d, const float scale)
{
    return (clampInt8(__float2int_rn((a / scale))) & 0x000000ff) | (clampInt8(__float2int_rn(b / scale)) << 8) & 0x0000ff00 |
           (clampInt8(__float2int_rn(c / scale)) << 16) & 0x00ff0000 | (clampInt8(__float2int_rn(d / scale)) << 24) & 0xff000000;
}
inline __device__ int quantizeInt8x4h_safe(__half a, __half b, __half c, __half d, const __half scale)
{
    return (clampInt8(__half2int_rn(__hdiv(a, scale))) & 0x000000ff) | (clampInt8(__half2int_rn(__hdiv(b, scale))) << 8) & 0x0000ff00 |
           (clampInt8(__half2int_rn(__hdiv(c, scale))) << 16) & 0x00ff0000 |
           (clampInt8(__half2int_rn(__hdiv(d, scale))) << 24) & 0xff000000;
}
// inline __device__ int quantizeInt8x4(float a, float b, float c, float d, const float scale)
// {
//     return (__float2int_rn((a / scale)) & 0x000000ff) | (__float2int_rn(b / scale) << 8) & 0x0000ff00 |
//            (__float2int_rn(c / scale) << 16) & 0x00ff0000 | (__float2int_rn(d / scale) << 24) & 0xff000000;
// }
// inline __device__ int quantizeInt8x4(__half a, __half b, __half c, __half d, const __half scale)
// {
//     return (__half2int_rn(__hdiv(a, scale)) & 0x000000ff) | (__half2int_rn(__hdiv(b, scale)) << 8) & 0x0000ff00 |
//            (__half2int_rn(__hdiv(c, scale)) << 16) & 0x00ff0000 | (__half2int_rn(__hdiv(d, scale)) << 24) & 0xff000000;
// }
inline __device__ float dequantizeInt8(const int packedData, const float scale)
{
    return __int2float_rn(packedData) * scale;
}

inline __device__ float dequantizeInt8f_relu(const int packedData, const float scale)
{
    return relu(__int2float_rn(packedData) * scale);
}


inline __device__ __half dequantizeInt8h_relu(const int packedData, const __half scale)
{
    return relu(__hmul(__int2half_rn(packedData), scale));
}




inline __device__ void dequantizeInt8x4(const int packedData, __half& a, __half& b, __half& c, __half& d, const __half scale)
{
    a = __hmul(__int2half_rn((int)packedData << 24 >> 24), scale);
    b = __hmul(__int2half_rn((int)packedData << 16 >> 24), scale);
    c = __hmul(__int2half_rn((int)packedData << 8 >> 24), scale);
    d = __hmul(__int2half_rn((int)packedData >> 24), scale);
}
inline __device__ void dequantizeInt8x4(const int packedData, float& a, float& b, float& c, float& d, const float scale)
{
    a = __int2float_rn((int)packedData << 24 >> 24) * scale;
    b = __int2float_rn((int)packedData << 16 >> 24) * scale;
    c = __int2float_rn((int)packedData << 8 >> 24) * scale;
    d = __int2float_rn((int)packedData >> 24) * scale;
}
inline __device__ void unpackInt8x4(const int packedData, int& a, int& b, int& c, int& d)
{
    a = (int)packedData << 24 >> 24;
    b = (int)packedData << 16 >> 24;
    c = (int)packedData << 8 >> 24;
    d = (int)packedData >> 24;
}

inline __device__ int packInt16x2(int a, int b)
{
    return (a & 0x0000ffff) | ((b << 16) & 0xffff0000);
}

// The CUDA kernel. This sample simply copies the input surface.
__global__ void int8test(int* weight, int* input, float* output, unsigned int width, unsigned int height, int debugOffset)
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

    int hiddenNum = 24;
    int hiddenPackedNum = hiddenNum / 4;
    int inNum = 6;
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
#if TEST_H
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale1),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale1),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale1),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale1),
            scaleIn2
        );
#else
       val2[k] = quantizeInt8x4h_safe(
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale1),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale1),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale1),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale1),
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
#if TEST_H
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale2),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale2),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale2),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale2),
            scaleIn3
        );
#else
        val2[k] = quantizeInt8x4h_safe(
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale2),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale2),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale2),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale2),
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
#if TEST_H
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale3),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale3),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale3),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale3),
            scaleIn4
        );
#else
        val2[k] = quantizeInt8x4h_safe(
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale3),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale3),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale3),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale3),
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
#if TEST_H
    output[4 * (y * width + x) + 0] = dequantizeInt8f_relu(val1[0], dequantizeScale4);
    output[4 * (y * width + x) + 1] = dequantizeInt8f_relu(val1[1], dequantizeScale4);
    output[4 * (y * width + x) + 2] = dequantizeInt8f_relu(val1[2], dequantizeScale4);
#else
    output[4 * (y * width + x) + 0] = dequantizeInt8h_relu(val1[0], dequantizeScale4);
    output[4 * (y * width + x) + 1] = dequantizeInt8h_relu(val1[1], dequantizeScale4);
    output[4 * (y * width + x) + 2] = dequantizeInt8h_relu(val1[2], dequantizeScale4);
#endif
}
// A wrapper function that launches the kernel.
void launchInt8Test(int* weight, int* input, float* output, unsigned int width, unsigned int height, int debugOffset)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    int8test<<<dimGrid, dimBlock>>>(weight, input, output, width, height, debugOffset);
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
    int inNumFirst = IN_1ST_NUM;
    int inNum = IN_NUM;
    int outNum = OUT_NUM;
    unsigned int* inputVal = input + 16 * (y * width + x);

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
