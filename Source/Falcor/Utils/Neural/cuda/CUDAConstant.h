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


