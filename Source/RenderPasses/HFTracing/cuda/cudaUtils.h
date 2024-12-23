#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)


__constant__ float scaleIn1 = 0.0020032862666994333;
__constant__ float scaleOut1 = 0.0035241753794252872;
__constant__ float dequantizeScale1 = 7.0599321588815656e-06;
__constant__ float scaleIn2 = 0.0016174662159755826;
__constant__ float scaleOut2 = 0.006601687055081129;
__constant__ float dequantizeScale2 = 1.0678006219677627e-05;
__constant__ float scaleIn3 = 0.0008834964828565717;
__constant__ float scaleOut3 = 0.011556388810276985;
__constant__ float dequantizeScale3 = 1.0210028449364472e-05;
__constant__ float scaleIn4 = 0.0011115598026663065;
__constant__ float scaleOut4 = 0.012618034146726131;
__constant__ float dequantizeScale4 = 1.4025699783815071e-05;



// __constant__ float scaleIn1 = 0.0035758037120103836;
// __constant__ float scaleOut1 = 0.003539581084623933;
// __constant__ float dequantizeScale1 = 1.2656847502512392e-05;
// __constant__ float scaleIn2 = 0.0021900988649576902;
// __constant__ float scaleOut2 = 0.00467855716124177;
// __constant__ float dequantizeScale2 = 1.0246502824884374e-05;
// __constant__ float scaleIn3 = 0.0011810491560027003;
// __constant__ float scaleOut3 = 0.011032089591026306;
// __constant__ float dequantizeScale3 = 1.3029440196987707e-05;
// __constant__ float scaleIn4 = 0.0019884631037712097;
// // __constantfloat scaleOut4 = 0.014394178986549377;
// __constant__ float dequantizeScale4 = 2.8622293029911816e-05;

// =====================================================================================================================
// Activation Functions
// =====================================================================================================================
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
    return max(x, 0.0f) + min(x, 0.0f) * 0.01f;
}
__half __device__ __forceinline__ leakyrelu(__half x)
{
    return __hmax(x, CUDART_ZERO_FP16) + __hmul(__hmin(x, CUDART_ZERO_FP16), __float2half_rn(0.01f));
}




// =====================================================================================================================
// Packing and Unpacking Functions
// =====================================================================================================================
inline __device__ void unpackSnorm2x16(unsigned int packed, float& a, float& b)
{
    a = __int2float_rd((int)(packed << 16) >> 16) / 32767.f;
    b = __int2float_rd((int)packed >> 16) / 32767.f;
}

inline __device__ void unpackUnorm2x16(unsigned int packed, float& a, float& b)
{
    a = __uint2float_rd((unsigned int)(packed << 16) >> 16) / 65535.f;
    b = __uint2float_rd((unsigned int)packed >> 16) / 65535.f;
}


inline __device__ void unpackSnorm2x16(unsigned int packed, __half& a, __half& b)
{
    a = __hdiv(__int2half_rd((int)(packed << 16) >> 16), 32767);
    b = __hdiv(__int2half_rd((int)packed >> 16), 32767);
}

inline __device__ void unpackSnorm2x16(int packed, __half& a, __half& b)
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
inline __device__ int quantizeInt8x4f_safe(float4 v, const float scale)
{
    return (clampInt8(__float2int_rn((v.x / scale))) & 0x000000ff) | (clampInt8(__float2int_rn(v.y / scale)) << 8) & 0x0000ff00 |
           (clampInt8(__float2int_rn(v.z / scale)) << 16) & 0x00ff0000 | (clampInt8(__float2int_rn(v.w / scale)) << 24) & 0xff000000;
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
