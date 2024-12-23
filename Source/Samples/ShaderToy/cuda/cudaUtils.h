#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)

__constant__ float scaleIn1 = 0.0010155849158763885;
__constant__ float scaleOut1 = 0.002965300576761365;
__constant__ float dequantizeScale1 = 3.0115145364106866e-06;
__constant__ float scaleIn2 = 0.0007379016024060547;
__constant__ float scaleOut2 = 0.0030863999854773283;
__constant__ float dequantizeScale2 = 2.2774595436203526e-06;
__constant__ float scaleIn3 = 0.0007874887087382376;
__constant__ float scaleOut3 = 0.0036407688166946173;
__constant__ float dequantizeScale3 = 2.8670642677752767e-06;
__constant__ float scaleIn4 = 0.0010824678465723991;
// __constantfloat scaleOut4 = 0.0035632268991321325;
__constant__ float dequantizeScale4 = 3.857078354485566e-06;

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
    // return relu(x);
    // return max(x, 0.0f) + min(x, 0.0f) * 0.01f;
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
