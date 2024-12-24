#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "CUDADefines.h"
#define CUDART_ZERO_FP16 __ushort_as_half((unsigned short)0x0000U)
#define CUDART_ONE_FP16 __ushort_as_half((unsigned short)0x3C00U)




#ifdef PEBBLE
__constant__ float scaleIn1 = 0.0024755310732871294;
__constant__ float scaleOut1 = 0.002964676357805729;
__constant__ float dequantizeScale1 = 7.339148396567907e-06;
__constant__ float scaleIn2 = 0.002071766648441553;
__constant__ float scaleOut2 = 0.005431175231933594;
__constant__ float dequantizeScale2 = 1.1252127478655893e-05;
__constant__ float scaleIn3 = 0.0010126761626452208;
__constant__ float scaleOut3 = 0.010284638032317162;
__constant__ float dequantizeScale3 = 1.041500763676595e-05;
__constant__ float scaleIn4 = 0.0009620034252293408;
__constant__ float scaleOut4 = 0.015066449530422688;
__constant__ float dequantizeScale4 = 1.4493975868390407e-05;
#endif

#ifdef LEATHER
__constant__ float scaleIn1 = 0.0029792573768645525;
__constant__ float scaleOut1 = 0.003814279567450285;
__constant__ float dequantizeScale1 = 1.1363720659574028e-05;
__constant__ float scaleIn2 = 0.002033429918810725;
__constant__ float scaleOut2 = 0.004576520062983036;
__constant__ float dequantizeScale2 = 9.306032552558463e-06;
__constant__ float scaleIn3 = 0.0020841280929744244;
__constant__ float scaleOut3 = 0.007710449397563934;
__constant__ float dequantizeScale3 = 1.6069563571363688e-05;
__constant__ float scaleIn4 = 0.0025302129797637463;
__constant__ float scaleOut4 = 0.009046046994626522;
__constant__ float dequantizeScale4 = 2.288842551934067e-05;
#endif
