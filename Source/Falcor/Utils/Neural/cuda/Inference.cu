#include "Inference.h"
#include "Utils.h"

#define IN_NUM 24
#define IN_PACKED_NUM IN_NUM / 4
#define HIDDEN_NUM 32
#define HIDDEN_PACKED_NUM HIDDEN_NUM / 4
#define HALF_ACC 1

#ifndef TEST_MULTI
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

    // ======================================
    // edit here!!
    // val is float4
    // val2 is int
    // synthesize on val
    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 0);
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(UP, v * uvScale, u * uvScale, 1);
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1);
    // =======================================


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

__device__ void TriangleGrid(float& w1, float& w2, float& w3, int2& vertex1, int2& vertex2, int2& vertex3, float2 st)
{
    st = float2{3.4641016f * st.x, 3.4641016f * st.y};
    float2 skewedCoord = float2{st.x - 0.57735027f * st.y, 1.15470054f * st.y};
    int2 baseId = int2{(int)floor(skewedCoord.x), (int)floor(skewedCoord.y)};
    float2 fracPart = float2{abs(skewedCoord.x - trunc(skewedCoord.x)), abs(skewedCoord.y - trunc(skewedCoord.y))};
    float3 temp = float3{fracPart.x, fracPart.y, 1.0f - fracPart.x - fracPart.y};
    if (temp.z > 0)
    {
        w1 = temp.z;
        w2 = temp.y;
        w3 = temp.x;
        vertex1 = int2{baseId.x, baseId.y};
        vertex2 = int2{baseId.x, baseId.y + 1};
        vertex2 = int2{baseId.x + 1, baseId.y};
    }
    else
    {
        w1 = -temp.z;
        w2 = 1.0f - temp.y;
        w3 = 1.0f - temp.x;
        vertex1 = int2{baseId.x + 1, baseId.y + 1};
        vertex2 = int2{baseId.x + 1, baseId.y};
        vertex2 = int2{baseId.x, baseId.y + 1};
    }
}

inline __device__ float2 hash22(float2 p)
{
    float2 r = float2{127.1f * p.x + 311.7f * p.y, 269.5f * p.x + 183.3f * p.y};
    //float2 temp = float2{sinf(r.x) * 43758.5453f, sinf(r.y) * 43758.5453f};
    float2 temp = float2{sinf(r.x), sinf(r.y)};
    //return float2{temp.x - floor(temp.x), temp.y - floor(temp.y)};
    return temp;
}

__global__ void inferInt8TexHisto(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
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

    // ======================================
    // edit here!!
    // val is float4
    // val2 is int
    // synthesize on val
    float w1, w2, w3;
    int2 vertex1, vertex2, vertex3;
    float2 uv{u * uvScale, v * uvScale};
    TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);
    float norm = sqrt(w1 * w1 + w2 * w2 + w3 * w3);

    float2 st1 = hash22(float2{(float)vertex1.x, (float)vertex1.y});
    float2 st2 = hash22(float2{(float)vertex2.x, (float)vertex2.y});
    float2 st3 = hash22(float2{(float)vertex3.x, (float)vertex3.y});
    st1 = float2{uv.x + st1.x, uv.y + st1.y};
    st2 = float2{uv.x + st2.x, uv.y + st2.y};
    st3 = float2{uv.x + st3.x, uv.y + st3.y};
    st1 = float2{abs(st1.x - trunc(st1.x)), abs(st1.y - trunc(st1.y))};
    st2 = float2{abs(st2.x - trunc(st2.x)), abs(st2.y - trunc(st2.y))};
    st3 = float2{abs(st3.x - trunc(st3.x)), abs(st3.y - trunc(st3.y))};

    float4 g1 = tex2DLayered<float4>(TP, st1.x, st1.y, 0);
    float4 g2 = tex2DLayered<float4>(TP, st2.x, st2.y, 0);
    float4 g3 = tex2DLayered<float4>(TP, st3.x, st3.y, 0);

    float4 G = float4{
        w1 * g1.x + w2 * g2.x + w3 * g3.x - 0.5f,
        w1 * g1.y + w2 * g2.y + w3 * g3.y - 0.5f,
        w1 * g1.z + w2 * g2.z + w3 * g3.z - 0.5f,
        w1 * g1.w + w2 * g2.w + w3 * g3.w - 0.5f
    };
    G = float4{G.x / norm, G.y / norm, G.z / norm, G.w / norm};
    G = float4{__saturatef(abs(G.x + 0.5f)), __saturatef(abs(G.y + 0.5f)), __saturatef(abs(G.z + 0.5f)), __saturatef(abs(G.w + 0.5f))};

    val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 0).x;
    val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 0).y;
    val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 0).z;
    val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 0).w;
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    g1 = tex2DLayered<float4>(TP, st1.x, st1.y, 1);
    g2 = tex2DLayered<float4>(TP, st2.x, st2.y, 1);
    g3 = tex2DLayered<float4>(TP, st3.x, st3.y, 1);

    G = float4{
        w1 * g1.x + w2 * g2.x + w3 * g3.x - 0.5f,
        w1 * g1.y + w2 * g2.y + w3 * g3.y - 0.5f,
        w1 * g1.z + w2 * g2.z + w3 * g3.z - 0.5f,
        w1 * g1.w + w2 * g2.w + w3 * g3.w - 0.5f
    };
    G = float4{G.x / norm, G.y / norm, G.z / norm, G.w / norm};
    G = float4{__saturatef(abs(G.x + 0.5f)), __saturatef(abs(G.y + 0.5f)), __saturatef(abs(G.z + 0.5f)), __saturatef(abs(G.w + 0.5f))};

    val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 1).x;
    val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 1).y;
    val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 1).z;
    val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 1).w;
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1);
    // =======================================


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

inline __device__ float rnd21(float2 p)
{
    float temp = sinf(12.9898f * p.x + 78.233f * p.y) * 43758.5453f;
    return (temp - floor(temp));
}

#endif

inline __device__ float B0(float2 uv, float scale = 1.0f)
{
    float u = uv.x * scale;
    float v = uv.y * scale;
    return powf(min(min(u - floor(u), ceil(u) - u), min(v - floor(v), ceil(v) - v)), 1.0);
}

inline __device__ float B1(float2 uv, float scale = 1.0f)
{
    float u = uv.x * scale + 0.5f;
    float v = uv.y * scale + 0.5f;
    return powf(min(min(u - floor(u), ceil(u) - u), min(v - floor(v), ceil(v) - v)), 1.0);
}

inline __device__ float B0cos(float2 uv, float scale = 1.0f)
{
    float cosu = sinf(uv.x * scale * 3.14159265f);
    float cosv = sinf(uv.y * scale * 3.14159265f);
    return powf(cosu * cosv * cosu * cosv, 0.5f);
}

inline __device__ float B1cos(float2 uv, float scale = 1.0f)
{
    uv = float2{uv.x * scale + 0.5f, uv.y * scale + 0.5f};
    float cosu = sinf(uv.x * 3.14159265f);
    float cosv = sinf(uv.y * 3.14159265f);
    return powf(cosu * cosv * cosu * cosv, 0.5f);
}

inline __device__ float BSingularity(float2 uv, float scale = 1.0f)
{
    uv = float2{(uv.x * scale - 0.5f) * 1.41421356237f, (uv.y * scale - 0.5f) * 1.41421356237f};
    const float a = 0.78539816f; // Pi / 4
    float cosA = cosf(a);
    float sinA = sinf(a);
    float2 V = float2{cosA * uv.x + sinA * uv.y, -sinA * uv.x + cosA * uv.y};
    float cosu = sinf(V.x * 3.14159265f);
    float cosv = sinf(V.y * 3.14159265f);
    return 0.02f * cosu * cosv * cosu * cosv;
}

#ifndef TEST_MULTI
__global__ void inferInt8TexAutocov(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
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

    // ======================================
    // edit here!!
    // val is float4
    // val2 is int
    // synthesize on val

    // float w1, w2, w3;
    // int2 vertex1, vertex2, vertex3;
    // float2 uv{u * uvScale, v * uvScale};
    // TriangleGrid(w1, w2, w3, vertex1, vertex2, vertex3, uv);
    // float norm = sqrt(w1 * w1 + w2 * w2 + w3 * w3);
    //
    // int id1 = (int)floor(rnd21(float2{(float)vertex1.x, (float)vertex1.y}) * 2048);
    // int id2 = (int)floor(rnd21(float2{(float)vertex2.x, (float)vertex2.y}) * 2048);
    // int id3 = (int)floor(rnd21(float2{(float)vertex3.x, (float)vertex3.y}) * 2048);
    // float2 st1 = float2{sampleList[2 * id1], sampleList[2 * id1 + 1]};
    // float2 st2 = float2{sampleList[2 * id2], sampleList[2 * id2 + 1]};
    // float2 st3 = float2{sampleList[2 * id3], sampleList[2 * id3 + 1]};
    // st1 = float2{uv.x - st1.x, uv.y - st1.y};
    // st2 = float2{uv.x - st2.x, uv.y - st2.y};
    // st3 = float2{uv.x - st3.x, uv.y - st3.y};
    // st1 = float2{abs(st1.x - trunc(st1.x)), abs(st1.y - trunc(st1.y))};
    // st2 = float2{abs(st2.x - trunc(st2.x)), abs(st2.y - trunc(st2.y))};
    // st3 = float2{abs(st3.x - trunc(st3.x)), abs(st3.y - trunc(st3.y))};
    //
    // float4 g1 = tex2DLayered<float4>(TP, st1.y, st1.x, 0);
    // float4 g2 = tex2DLayered<float4>(TP, st2.y, st2.x, 0);
    // float4 g3 = tex2DLayered<float4>(TP, st3.y, st3.x, 0);
    //
    // float4 G = float4{
    //     w1 * g1.x + w2 * g2.x + w3 * g3.x - 0.5f,
    //     w1 * g1.y + w2 * g2.y + w3 * g3.y - 0.5f,
    //     w1 * g1.z + w2 * g2.z + w3 * g3.z - 0.5f,
    //     w1 * g1.w + w2 * g2.w + w3 * g3.w - 0.5f
    // };
    // G = float4{G.x / norm, G.y / norm, G.z / norm, G.w / norm};
    // G = float4{__saturatef(abs(G.x + 0.5f)), __saturatef(abs(G.y + 0.5f)), __saturatef(abs(G.z + 0.5f)), __saturatef(abs(G.w + 0.5f))};
    //
    // val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 0).x;
    // val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 0).y;
    // val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 0).z;
    // val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 0).w;
    // val2[2] = quantizeInt8x4f_safe(val, scaleIn1);
    //
    // g1 = tex2DLayered<float4>(TP, st1.y, st1.x, 1);
    // g2 = tex2DLayered<float4>(TP, st2.y, st2.x, 1);
    // g3 = tex2DLayered<float4>(TP, st3.y, st3.x, 1);
    //
    // G = float4{
    //     w1 * g1.x + w2 * g2.x + w3 * g3.x - 0.5f,
    //     w1 * g1.y + w2 * g2.y + w3 * g3.y - 0.5f,
    //     w1 * g1.z + w2 * g2.z + w3 * g3.z - 0.5f,
    //     w1 * g1.w + w2 * g2.w + w3 * g3.w - 0.5f
    // };
    // G = float4{G.x / norm, G.y / norm, G.z / norm, G.w / norm};
    // G = float4{__saturatef(abs(G.x + 0.5f)), __saturatef(abs(G.y + 0.5f)), __saturatef(abs(G.z + 0.5f)), __saturatef(abs(G.w + 0.5f))};
    //
    // val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 1).x;
    // val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 1).y;
    // val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 1).z;
    // val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 1).w;
    // val2[3] = quantizeInt8x4f_safe(val, scaleIn1);

    float2 uv{u * uvScale, v * uvScale};
    //float2 uv{v * uvScale, u * uvScale};
    //uv = float2{uv.x - floor(uv.x), uv.y - floor(uv.y)};
    float3 b;
    float bSum;

    b = float3{B0cos(uv), B1cos(uv), BSingularity(uv)};
    bSum = b.x + b.y + b.z;
    b = float3{b.x / bSum, b.y / bSum, b.z / bSum};
    float norm = sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
    float2 t0 = float2{floor(uv.x) * 2.0f, floor(uv.y) * 2.0f};
    float2 t1 = float2{floor(uv.x + 0.5f) * 2.0f + 1.0f, floor(uv.y + 0.5f) * 2.0f + 1.0f};
    //float2 t1 = float2{floor(uv.x + 0.5f), floor(uv.y + 0.5f)};
    //float2 t0 = t1;

    int id0 = (int)floor(rnd21(t0) * 2048);
    int id1 = (int)floor(rnd21(t1) * 2048);
    //int id0 = 0;
    //int id1 = 1;
    float2 st0 = float2{sampleList[2 * id0], sampleList[2 * id0 + 1]};
    float2 st1 = float2{sampleList[2 * id1], sampleList[2 * id1 + 1]};
    //float2 st0 = hash22(t0);
    //float2 st1 = hash22(t1);
    //st0 = float2{uv.x - b.x, uv.y - b.y};
    //st1 = float2{uv.x - b.x, uv.y - b.y};
    st0 = float2{uv.x - st0.x, uv.y - st0.y};
    st1 = float2{uv.x - st1.x, uv.y - st1.y};
    //st0 = float2{uv.x - 4.3f, uv.y - 5.7f};
    //st1 = float2{uv.x + 3.6f, uv.y + 6.4f};
    //st0 = float2{st0.x - floor(st0.x), st0.y - floor(st0.y)};
    //st1 = float2{st1.x - floor(st1.x), st1.y - floor(st1.y)};

    float4 g0 = tex2DLayered<float4>(TP, st0.y, st0.x, 0);
    float4 g1 = tex2DLayered<float4>(TP, st1.y, st1.x, 0);
    //float4 g0 = tex2DLayered<float4>(TP, st0.x, st0.y, 0);
    //float4 g1 = tex2DLayered<float4>(TP, st1.x, st1.y, 0);

    float4 G = float4{
        (g0.x - 0.5f) * b.x + (g1.x - 0.5f) * b.y,
        (g0.y - 0.5f) * b.x + (g1.y - 0.5f) * b.y,
        (g0.z - 0.5f) * b.x + (g1.z - 0.5f) * b.y,
        (g0.w - 0.5f) * b.x + (g1.w - 0.5f) * b.y
    };
    //G = float4{
    //    __saturatef(G.x / norm + 0.5f), __saturatef(G.y / norm + 0.5f), __saturatef(G.z / norm + 0.5f), __saturatef(G.w / norm + 0.5f)
    //};
    G = float4{G.x / norm + 0.5f, G.y / norm + 0.5f, G.z / norm + 0.5f, G.w / norm + 0.5f};
    if (G.x < 0.0001)
    {
        G.x = 0.0001;
    }
    if (G.x > 0.999)
    {
        G.x = 0.999;
    }
    if (G.y < 0.0001)
    {
        G.y = 0.0001;
    }
    if (G.y > 0.999)
    {
        G.y = 0.999;
    }
    if (G.z < 0.0001)
    {
        G.z = 0.0001;
    }
    if (G.z > 0.999)
    {
        G.z = 0.999;
    }
    if (G.w < 0.0001)
    {
        G.w = 0.0001;
    }
    if (G.w > 0.999)
    {
        G.w = 0.999;
    }

    val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 0).x;
    val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 0).y;
    val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 0).z;
    val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 0).w;
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    g0 = tex2DLayered<float4>(TP, st0.y, st0.x, 1);
    g1 = tex2DLayered<float4>(TP, st1.y, st1.x, 1);
    //g0 = tex2DLayered<float4>(TP, st0.x, st0.y, 1);
    //g1 = tex2DLayered<float4>(TP, st1.x, st1.y, 1);

    G = float4{
        (g0.x - 0.5f) * b.x + (g1.x - 0.5f) * b.y,
        (g0.y - 0.5f) * b.x + (g1.y - 0.5f) * b.y,
        (g0.z - 0.5f) * b.x + (g1.z - 0.5f) * b.y,
        (g0.w - 0.5f) * b.x + (g1.w - 0.5f) * b.y
    };
    //G = float4{
    //    __saturatef(G.x / norm + 0.5f), __saturatef(G.y / norm + 0.5f), __saturatef(G.z / norm + 0.5f), __saturatef(G.w / norm + 0.5f)
    //};
    G = float4{G.x / norm + 0.5f, G.y / norm + 0.5f, G.z / norm + 0.5f, G.w / norm + 0.5f};
    if (G.x < 0.0001)
    {
        G.x = 0.0001;
    }
    if (G.x > 0.999)
    {
        G.x = 0.999;
    }
    if (G.y < 0.0001)
    {
        G.y = 0.0001;
    }
    if (G.y > 0.999)
    {
        G.y = 0.999;
    }
    if (G.z < 0.0001)
    {
        G.z = 0.0001;
    }
    if (G.z > 0.999)
    {
        G.z = 0.999;
    }
    if (G.w < 0.0001)
    {
        G.w = 0.0001;
    }
    if (G.w > 0.999)
    {
        G.w = 0.999;
    }

    val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 1).x;
    val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 1).y;
    val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 1).z;
    val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 1).w;
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1);
    // =======================================


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
    //output[4 * (y * width + x) + 0] = uv.x;
    //output[4 * (y * width + x) + 1] = uv.y;
    //output[4 * (y * width + x) + 2] = 12.9898f * t1.x + 78.233f * t1.y;
    //output[4 * (y * width + x) + 3] = sinf(12.9898f * t1.x + 78.233f * t1.y) * 43758.5453f;
#else
    output[4 * (y * width + x) + 0] = dequantizeInt8f_relu(val1[0], dequantizeScale4);
    output[4 * (y * width + x) + 1] = dequantizeInt8f_relu(val1[1], dequantizeScale4);
    output[4 * (y * width + x) + 2] = dequantizeInt8f_relu(val1[2], dequantizeScale4);
#endif
}

#endif

#ifdef TEST_MULTI

__global__ void inferInt8TexHashed(
    int* weight,
    int* packedInput,
    float* hashedUV,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale,
    float patchScale,
    int matId
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
    //if (validMask[y * width + x] == 0)
    if (validMask[y * width + x] != matId)
        return;

    int offset = 0;
    int hiddenNum = HIDDEN_NUM;
    int hiddenPackedNum = HIDDEN_PACKED_NUM;
    int inPackedNum = IN_PACKED_NUM;
    int outNum = 3;

    int trueMatId = matId - 1;

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[4 * (y * width + x) + 2], u, v);

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, scaleIn1[trueMatId]);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, scaleIn1[trueMatId]);

    // ======================================
    // edit here!!
    // val is float4
    // val2 is int
    // synthesize on val

    float2 uv{u * uvScale, v * uvScale};
    float3 b;
    float bSum;

    b = float3{B0cos(uv, patchScale), B1cos(uv, patchScale), BSingularity(uv, patchScale)};
    bSum = b.x + b.y + b.z;
    b = float3{b.x / bSum, b.y / bSum, b.z / bSum};
    float norm = sqrt(b.x * b.x + b.y * b.y + b.z * b.z);

    float2 st0 = float2{hashedUV[4 * (y * width + x)], hashedUV[4 * (y * width + x) + 1]};
    float2 st1 = float2{hashedUV[4 * (y * width + x) + 2], hashedUV[4 * (y * width + x) + 3]};

    float4 g0 = tex2DLayered<float4>(TP, st0.y, st0.x, 0);
    float4 g1 = tex2DLayered<float4>(TP, st1.y, st1.x, 0);

    float4 G = float4{
        (g0.x - 0.5f) * b.x + (g1.x - 0.5f) * b.y,
        (g0.y - 0.5f) * b.x + (g1.y - 0.5f) * b.y,
        (g0.z - 0.5f) * b.x + (g1.z - 0.5f) * b.y,
        (g0.w - 0.5f) * b.x + (g1.w - 0.5f) * b.y
    };
    G = float4{G.x / norm + 0.5f, G.y / norm + 0.5f, G.z / norm + 0.5f, G.w / norm + 0.5f};
    if (G.x < 0.0001)
    {
        G.x = 0.0001;
    }
    if (G.x > 0.999)
    {
        G.x = 0.999;
    }
    if (G.y < 0.0001)
    {
        G.y = 0.0001;
    }
    if (G.y > 0.999)
    {
        G.y = 0.999;
    }
    if (G.z < 0.0001)
    {
        G.z = 0.0001;
    }
    if (G.z > 0.999)
    {
        G.z = 0.999;
    }
    if (G.w < 0.0001)
    {
        G.w = 0.0001;
    }
    if (G.w > 0.999)
    {
        G.w = 0.999;
    }

    val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 0).x;
    val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 0).y;
    val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 0).z;
    val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 0).w;
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1[trueMatId]);

    g0 = tex2DLayered<float4>(TP, st0.y, st0.x, 1);
    g1 = tex2DLayered<float4>(TP, st1.y, st1.x, 1);

    G = float4{
        (g0.x - 0.5f) * b.x + (g1.x - 0.5f) * b.y,
        (g0.y - 0.5f) * b.x + (g1.y - 0.5f) * b.y,
        (g0.z - 0.5f) * b.x + (g1.z - 0.5f) * b.y,
        (g0.w - 0.5f) * b.x + (g1.w - 0.5f) * b.y
    };
    G = float4{G.x / norm + 0.5f, G.y / norm + 0.5f, G.z / norm + 0.5f, G.w / norm + 0.5f};
    if (G.x < 0.0001)
    {
        G.x = 0.0001;
    }
    if (G.x > 0.999)
    {
        G.x = 0.999;
    }
    if (G.y < 0.0001)
    {
        G.y = 0.0001;
    }
    if (G.y > 0.999)
    {
        G.y = 0.999;
    }
    if (G.z < 0.0001)
    {
        G.z = 0.0001;
    }
    if (G.z > 0.999)
    {
        G.z = 0.999;
    }
    if (G.w < 0.0001)
    {
        G.w = 0.0001;
    }
    if (G.w > 0.999)
    {
        G.w = 0.999;
    }

    val.x = tex2DLayered<float4>(InvP, G.x, 0.0f, 1).x;
    val.y = tex2DLayered<float4>(InvP, G.y, 0.0f, 1).y;
    val.z = tex2DLayered<float4>(InvP, G.z, 0.0f, 1).z;
    val.w = tex2DLayered<float4>(InvP, G.w, 0.0f, 1).w;
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1[trueMatId]);
    // =======================================


    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val2[4] = quantizeInt8x4f_safe(val, scaleIn1[trueMatId]);

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val2[5] = quantizeInt8x4f_safe(val, scaleIn1[trueMatId]);

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
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale1[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale1[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale1[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale1[trueMatId]),
            scaleIn2[trueMatId]
        );

#else
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale1[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale1[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale1[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale1[trueMatId]),
            scaleIn2[trueMatId]
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
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale2[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale2[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale2[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale2[trueMatId]),
            scaleIn3[trueMatId]
        );
#else
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale2[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale2[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale2[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale2[trueMatId]),
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
            dequantizeInt8h_relu(val1[4 * k], dequantizeScale3[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 1], dequantizeScale3[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 2], dequantizeScale3[trueMatId]),
            dequantizeInt8h_relu(val1[4 * k + 3], dequantizeScale3[trueMatId]),
            scaleIn4[trueMatId]
        );
#else
        val2[k] = quantizeInt8x4f_safe(
            dequantizeInt8f_relu(val1[4 * k], dequantizeScale3[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 1], dequantizeScale3[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 2], dequantizeScale3[trueMatId]),
            dequantizeInt8f_relu(val1[4 * k + 3], dequantizeScale3[trueMatId]),
            scaleIn4[trueMatId]
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
    output[4 * (y * width + x) + 0] = dequantizeInt8h_relu(val1[0], dequantizeScale4[trueMatId]);
    output[4 * (y * width + x) + 1] = dequantizeInt8h_relu(val1[1], dequantizeScale4[trueMatId]);
    output[4 * (y * width + x) + 2] = dequantizeInt8h_relu(val1[2], dequantizeScale4[trueMatId]);
#else
    output[4 * (y * width + x) + 0] = dequantizeInt8f_relu(val1[0], dequantizeScale4[trueMatId]);
    output[4 * (y * width + x) + 1] = dequantizeInt8f_relu(val1[1], dequantizeScale4[trueMatId]);
    output[4 * (y * width + x) + 2] = dequantizeInt8f_relu(val1[2], dequantizeScale4[trueMatId]);
#endif
}

void launchInferInt8TexHashed(
    int* weight,
    int* packedInput,
    float* hashedUV,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale,
    float patchScale,
    int matId
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8TexHashed<<<dimGrid, dimBlock>>>(
        weight, packedInput, hashedUV, HP, DP, UP, TP, InvP, sampleList, output, width, height, validMask, uvScale, patchScale, matId
    );
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
}

void launchInferInt8TexHisto(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
}

void launchInferInt8TexAutocov(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
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
}
#endif 

#ifndef TEST_MULTI

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

void launchInferInt8TexHisto(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8TexHisto<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, TP, InvP, output, width, height, validMask, uvScale);
}

void launchInferInt8TexAutocov(
    int* weight,
    int* packedInput,
    cudaTextureObject_t HP,
    cudaTextureObject_t DP,
    cudaTextureObject_t UP,
    cudaTextureObject_t TP,
    cudaTextureObject_t InvP,
    float* sampleList,
    float* output,
    unsigned int width,
    unsigned int height,
    int* validMask,
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8TexAutocov<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, TP, InvP, sampleList, output, width, height, validMask, uvScale);
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
#endif
