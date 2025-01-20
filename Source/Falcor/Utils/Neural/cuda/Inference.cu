#include "Inference.h"
#include "Utils.h"

#define IN_NUM 24
#define IN_PACKED_NUM 6
#define HIDDEN_NUM 32
#define HIDDEN_PACKED_NUM 8
#define HALF_ACC 0
__global__ void inferInt8TexTest(
    int* weight,
    int* packedInput,
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

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float u = (float)x / (float)height;
    float v = (float)y / (float)height;

    // float u ;
    // float v ;
    float h1 ;
    float h2 ;
    float d1 ;
    float d2 ;

    unpackUnorm2x16(packedInput[2 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[2 * (y * width + x) + 1], d1, d2);


    // unpackUnorm2x16(packedInput[3 * (y * width + x) + 0], h1, h2);
    // unpackUnorm2x16(packedInput[3 * (y * width + x) + 1], d1, d2);
    // unpackUnorm2x16(packedInput[3 * (y * width + x) + 2], u, v);

    // float h1 = testInput[4 * (y * width + x)];
    // float h2 = testInput[4 * (y * width + x) + 1];
    // float d1 = testInput[4 * (y * width + x) + 2];
    // float d2 = testInput[4 * (y * width + x) + 3];




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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < IN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[k * IN_PACKED_NUM + j], val1[k]);
        }
    }

    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[192 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[448 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < 3; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[704 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }

        // val1[0] = 0;
        // for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        // {
        //     val1[0] = __dp4a(val2[j], W[704 +  j], val1[0]);
        // }

        // val1[1] = 0;
        // for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        // {
        //     val1[1] = __dp4a(val2[j], W[704 + 1 * HIDDEN_PACKED_NUM + j], val1[1]);
        // }

        // val1[2] = 0;
        // for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        // {
        //     val1[2] = __dp4a(val2[j], W[704 + 2 * HIDDEN_PACKED_NUM + j], val1[2]);
        // }


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
    int* packedInput,
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < IN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[k * IN_PACKED_NUM + j], val1[k]);
        }
    }

    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[192 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[448 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < 3; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[704 + k * HIDDEN_PACKED_NUM + j], val1[k]);
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
    float u1, v1, u2, v2;
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

    float2 uv{u * uvScale, v * uvScale};
    float3 b;
    float bSum;

    b = float3{B0cos(uv), B1cos(uv), BSingularity(uv)};
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
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

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




__global__ void inferInt8TexHashedOptimized(
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


    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float h1, h2;
    float d1, d2;
    float u, v;
    float u1, v1, u2, v2;
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 1], d1, d2);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 2], u, v);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 3], u1, v1);
    unpackUnorm2x16(packedInput[5 * (y * width + x) + 4], u2, v2);

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, scaleIn1);

    // ======================================
    // edit here!!
    // val is float4
    // val2 is int
    // synthesize on val

    u *= uvScale;
    v *= uvScale;
    float norm;
    float b0, b1, bs;

    b0 = B0cos(u, v);
    b1 = B1cos(u, v);
    bs = BSingularity(u, v);
    norm = b0 + b1 + bs;

    b0 /= norm;
    b1 /= norm;
    bs /= norm;

    norm = sqrt(b0 * b0 + b1 * b1 + bs * bs);


    float4 g0 = tex2DLayered<float4>(TP, v1, u1, 0);
    float4 g1 = tex2DLayered<float4>(TP, v2, u2, 0);
    float Gx, Gy, Gz, Gw;

    Gx = (g0.x - 0.5f) * b0 + (g1.x - 0.5f) *b1;
    Gy = (g0.y - 0.5f) * b0 + (g1.y - 0.5f) *b1;
    Gz = (g0.z - 0.5f) * b0 + (g1.z - 0.5f) *b1;
    Gw = (g0.w - 0.5f) * b0 + (g1.w - 0.5f) *b1;




    Gx = clampG(Gx / norm + 0.5f);
    Gy = clampG(Gy / norm + 0.5f);
    Gz = clampG(Gz / norm + 0.5f);
    Gw = clampG(Gw / norm + 0.5f);



    val.x = tex2DLayered<float4>(InvP, Gx, 0.0f, 0).x;
    val.y = tex2DLayered<float4>(InvP, Gy, 0.0f, 0).y;
    val.z = tex2DLayered<float4>(InvP, Gz, 0.0f, 0).z;
    val.w = tex2DLayered<float4>(InvP, Gw, 0.0f, 0).w;
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    g0 = tex2DLayered<float4>(TP, v1, u1, 1);
    g1 = tex2DLayered<float4>(TP, v2, u2, 1);

    Gx = (g0.x - 0.5f) * b0 + (g1.x - 0.5f) *b1;
    Gy = (g0.y - 0.5f) * b0 + (g1.y - 0.5f) *b1;
    Gz = (g0.z - 0.5f) * b0 + (g1.z - 0.5f) *b1;
    Gw = (g0.w - 0.5f) * b0 + (g1.w - 0.5f) *b1;
     Gx = clampG(Gx / norm + 0.5f);
    Gy = clampG(Gy / norm + 0.5f);
    Gz = clampG(Gz / norm + 0.5f);
    Gw = clampG(Gw / norm + 0.5f);

    val.x = tex2DLayered<float4>(InvP, Gx, 0.0f, 1).x;
    val.y = tex2DLayered<float4>(InvP, Gy, 0.0f, 1).y;
    val.z = tex2DLayered<float4>(InvP, Gz, 0.0f, 1).z;
    val.w = tex2DLayered<float4>(InvP, Gw, 0.0f, 1).w;
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1);
    // =======================================


    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val2[4] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val2[5] = quantizeInt8x4f_safe(val, scaleIn1);


   // layer 1
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < IN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[k * IN_PACKED_NUM + j], val1[k]);
        }
    }

    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[192 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[448 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < 3; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[704 + k * HIDDEN_PACKED_NUM + j], val1[k]);
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

// void launchInferInt8TexHisto(
//     int* weight,
//     int* packedInput,
//     cudaTextureObject_t HP,
//     cudaTextureObject_t DP,
//     cudaTextureObject_t UP,
//     cudaTextureObject_t TP,
//     cudaTextureObject_t InvP,
//     float* output,
//     unsigned int width,
//     unsigned int height,
//     int* validMask,
//     float uvScale
// )
// {
//     dim3 dimBlock(16, 16);
//     dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
//     inferInt8TexHisto<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, TP, InvP, output, width, height, validMask, uvScale);
// }

// void launchInferInt8TexAutocov(
//     int* weight,
//     int* packedInput,
//     cudaTextureObject_t HP,
//     cudaTextureObject_t DP,
//     cudaTextureObject_t UP,
//     cudaTextureObject_t TP,
//     cudaTextureObject_t InvP,
//     float* sampleList,
//     float* output,
//     unsigned int width,
//     unsigned int height,
//     int* validMask,
//     float uvScale
// )
// {
//     dim3 dimBlock(16, 16);
//     dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
//     inferInt8TexAutocov<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, TP, InvP, sampleList, output, width, height, validMask, uvScale);
// }

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
    float uvScale
)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    // inferInt8TexHashed<<<dimGrid, dimBlock>>>(weight, packedInput, hashedUV, HP, DP, UP, TP, InvP, sampleList, output, width, height, validMask, uvScale);
    inferInt8TexHashedOptimized<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP, TP, InvP, sampleList, output, width, height, validMask, uvScale);
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
    int* packedInput,
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
    float h1 = 0.0;
    float h2 = 0.0;
    float d1 = 0.0;
    float d2 = 0.0;

    unpackUnorm2x16(packedInput[2 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[2 * (y * width + x) + 1], d1, d2);

    // float h1 = testInput[4 * (y * width + x)];
    // float h2 = testInput[4 * (y * width + x) + 1];
    // float d1 = testInput[4 * (y * width + x) + 2];
    // float d2 = testInput[4 * (y * width + x) + 3];

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
    int* packedInput,
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
    int* packedInput,
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
    float h1 = 0.0;
    float h2 = 0.0;
    float d1 = 0.0;
    float d2 = 0.0;

    unpackUnorm2x16(packedInput[2 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[2 * (y * width + x) + 1], d1, d2);

    // float h1 = testInput[4 * (y * width + x)];
    // float h2 = testInput[4 * (y * width + x) + 1];
    // float d1 = testInput[4 * (y * width + x) + 2];
    // float d2 = testInput[4 * (y * width + x) + 3];

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
    int* packedInput,
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



__global__ void inferInt8TexACFTest(
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

    int val1[HIDDEN_NUM];
    int val2[HIDDEN_PACKED_NUM];

    float u = (float)x / (float)height;
    float v = (float)y / (float)height;
    float h1;
    float h2;
    float d1;
    float d2;

    unpackUnorm2x16(packedInput[2 * (y * width + x) + 0], h1, h2);
    unpackUnorm2x16(packedInput[2 * (y * width + x) + 1], d1, d2);

    // float h1 = testInput[4 * (y * width + x)];
    // float h2 = testInput[4 * (y * width + x) + 1];
    // float d1 = testInput[4 * (y * width + x) + 2];
    // float d2 = testInput[4 * (y * width + x) + 3];

    float4 val = tex2DLayered<float4>(HP, h1, h2, 0);
    val2[0] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(HP, h1, h2, 1);
    val2[1] = quantizeInt8x4f_safe(val, scaleIn1);


    val = tex2DLayered<float4>(DP, d1, d2, 0);
    val2[4] = quantizeInt8x4f_safe(val, scaleIn1);

    val = tex2DLayered<float4>(DP, d1, d2, 1);
    val2[5] = quantizeInt8x4f_safe(val, scaleIn1);


    u *= uvScale;
    v *= uvScale;
    float norm;
    float b0, b1, bs;

    b0 = B0cos(u, v);
    b1 = B1cos(u, v);
    bs = BSingularity(u, v);
    norm = b0 + b1 + bs;

    b0 /= norm;
    b1 /= norm;
    bs /= norm;

    norm = sqrt(b0 * b0 + b1 * b1 + bs * bs);

    // float2 t0 = float2{floor(u) * 2.0f, floor(v) * 2.0f};
    // float2 t1 = float2{floor(u + 0.5f) * 2.0f + 1.0f, floor(v + 0.5f) * 2.0f + 1.0f};

    // =======================================================================
    int id0 = (int)floor(rnd21(floor(u) * 2.0f, floor(v) * 2.0f) * 2048);
    int id1 = (int)floor(rnd21(floor(u + 0.5f) * 2.0f + 1.0f, floor(v + 0.5f) * 2.0f + 1.0f) * 2048);

    float u1 = sampleList[2 * id0];
    float v1 = sampleList[2 * id0 + 1];

    float u2 = sampleList[2 * id1];
    float v2 = sampleList[2 * id1 + 1];
    // =======================================================================


    float4 g0 = tex2DLayered<float4>(TP, v - v1, u - u1, 0);
    float4 g1 = tex2DLayered<float4>(TP, v -v2, u - u2, 0);
    float Gx, Gy, Gz, Gw;
    Gx = (g0.x - 0.5f) * b0 + (g1.x - 0.5f) *b1;
    Gy = (g0.y - 0.5f) * b0 + (g1.y - 0.5f) *b1;
    Gz = (g0.z - 0.5f) * b0 + (g1.z - 0.5f) *b1;
    Gw = (g0.w - 0.5f) * b0 + (g1.w - 0.5f) *b1;




    Gx = clampG(Gx / norm + 0.5f);
    Gy = clampG(Gy / norm + 0.5f);
    Gz = clampG(Gz / norm + 0.5f);
    Gw = clampG(Gw / norm + 0.5f);



    val.x = tex2DLayered<float4>(InvP, Gx, 0.0f, 0).x;
    val.y = tex2DLayered<float4>(InvP, Gy, 0.0f, 0).y;
    val.z = tex2DLayered<float4>(InvP, Gz, 0.0f, 0).z;
    val.w = tex2DLayered<float4>(InvP, Gw, 0.0f, 0).w;
    val2[2] = quantizeInt8x4f_safe(val, scaleIn1);

    g0 = tex2DLayered<float4>(TP, v1, u1, 1);
    g1 = tex2DLayered<float4>(TP, v2, u2, 1);

    Gx = (g0.x - 0.5f) * b0 + (g1.x - 0.5f) *b1;
    Gy = (g0.y - 0.5f) * b0 + (g1.y - 0.5f) *b1;
    Gz = (g0.z - 0.5f) * b0 + (g1.z - 0.5f) *b1;
    Gw = (g0.w - 0.5f) * b0 + (g1.w - 0.5f) *b1;
     Gx = clampG(Gx / norm + 0.5f);
    Gy = clampG(Gy / norm + 0.5f);
    Gz = clampG(Gz / norm + 0.5f);
    Gw = clampG(Gw / norm + 0.5f);

    val.x = tex2DLayered<float4>(InvP, Gx, 0.0f, 1).x;
    val.y = tex2DLayered<float4>(InvP, Gy, 0.0f, 1).y;
    val.z = tex2DLayered<float4>(InvP, Gz, 0.0f, 1).z;
    val.w = tex2DLayered<float4>(InvP, Gw, 0.0f, 1).w;
    val2[3] = quantizeInt8x4f_safe(val, scaleIn1);
    // =======================================================================

     // layer 1
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < IN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[k * IN_PACKED_NUM + j], val1[k]);
        }
    }

    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[192 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < HIDDEN_NUM; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[448 + k * HIDDEN_PACKED_NUM + j], val1[k]);
        }
    }
    for (int k = 0; k < HIDDEN_PACKED_NUM; k++)
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
    for (int k = 0; k < 3; k++)
    {
        val1[k] = 0;
        for (int j = 0; j < HIDDEN_PACKED_NUM; j++)
        {
            val1[k] = __dp4a(val2[j], W[704 + k * HIDDEN_PACKED_NUM + j], val1[k]);
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


void launchInferInt8TexACFTest(
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
     float uvScale
){

    dim3 dimBlock(16, 16);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
    inferInt8TexACFTest<<<dimGrid, dimBlock>>>(weight, packedInput, HP, DP, UP,TP,InvP,sampleList, output, width, height, uvScale);

}
