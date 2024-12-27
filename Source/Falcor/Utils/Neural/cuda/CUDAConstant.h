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
__constant__ float scaleIn1 = 0.003400295041501522;
__constant__ float scaleOut1 = 0.0033392827026546;
__constant__ float dequantizeScale1 = 1.1354546586517245e-05;
__constant__ float scaleIn2 = 0.0024283595848828554;
__constant__ float scaleOut2 = 0.004313669633120298;
__constant__ float dequantizeScale2 = 1.047514069796307e-05;
__constant__ float scaleIn3 = 0.0021721271332353354;
__constant__ float scaleOut3 = 0.009137849323451519;
__constant__ float dequantizeScale3 = 1.9848570445901714e-05;
__constant__ float scaleIn4 = 0.0016346105840057135;
__constant__ float scaleOut4 = 0.009822787716984749;
__constant__ float dequantizeScale4 = 1.605643228685949e-05;
#endif


#ifdef LEATHER_04R
__constant__ float scaleIn1 = 0.003255440853536129;
__constant__ float scaleOut1 = 0.003255606861785054;
__constant__ float dequantizeScale1 = 1.0598435437714215e-05;
__constant__ float scaleIn2 = 0.001738795661367476;
__constant__ float scaleOut2 = 0.003979842644184828;
__constant__ float dequantizeScale2 = 6.920133273524698e-06;
__constant__ float scaleIn3 = 0.0014758751494809985;
__constant__ float scaleOut3 = 0.0071127330884337425;
__constant__ float dequantizeScale3 = 1.0497506082174368e-05;
__constant__ float scaleIn4 = 0.0036747485864907503;
__constant__ float scaleOut4 = 0.008306708186864853;
__constant__ float dequantizeScale4 = 3.0525065085384995e-05;
#endif


#ifdef BRICK
__constant__ float scaleIn1 = 0.0015997408190742135;
__constant__ float scaleOut1 = 0.0030100212898105383;
__constant__ float dequantizeScale1 = 4.81525376017089e-06;
__constant__ float scaleIn2 = 0.0012171040289103985;
__constant__ float scaleOut2 = 0.002856734674423933;
__constant__ float dequantizeScale2 = 3.4769432204484474e-06;
__constant__ float scaleIn3 = 0.000973439309746027;
__constant__ float scaleOut3 = 0.004938448779284954;
__constant__ float dequantizeScale3 = 4.807280220120447e-06;
__constant__ float scaleIn4 = 0.00084302929462865;
__constant__ float scaleOut4 = 0.006031387019902468;
__constant__ float dequantizeScale4 = 5.084636086394312e-06;
#endif

#ifdef METAL
__constant__ float scaleIn1 = 0.001956135965883732;
__constant__ float scaleOut1 = 0.004767794162034988;
__constant__ float dequantizeScale1 = 9.32645343709737e-06;
__constant__ float scaleIn2 = 0.0014222803292796016;
__constant__ float scaleOut2 = 0.005467925686389208;
__constant__ float dequantizeScale2 = 7.776922757329885e-06;
__constant__ float scaleIn3 = 0.0015256208134815097;
__constant__ float scaleOut3 = 0.0069251591339707375;
__constant__ float dequantizeScale3 = 1.0565167031018063e-05;
__constant__ float scaleIn4 = 0.002577820559963584;
__constant__ float scaleOut4 = 0.019315816462039948;
__constant__ float dequantizeScale4 = 4.979271034244448e-05;
#endif
#ifdef TILE
__constant__ float scaleIn1 = 0.001563933095894754;
__constant__ float scaleOut1 = 0.002727729035541415;
__constant__ float dequantizeScale1 = 4.265985808160622e-06;
__constant__ float scaleIn2 = 0.0005605574697256088;
__constant__ float scaleOut2 = 0.0033667355310171843;
__constant__ float dequantizeScale2 = 1.8872487999033183e-06;
__constant__ float scaleIn3 = 0.001244415994733572;
__constant__ float scaleOut3 = 0.003601019037887454;
__constant__ float dequantizeScale3 = 4.481165888137184e-06;
__constant__ float scaleIn4 = 0.0020106337033212185;
__constant__ float scaleOut4 = 0.0022378810681402683;
__constant__ float dequantizeScale4 = 4.499559054238489e-06;
#endif
