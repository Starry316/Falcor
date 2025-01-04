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


#ifdef METAL2
__constant__ float scaleIn1 = 0.0021415925584733486;
__constant__ float scaleOut1 = 0.0034566777758300304;
__constant__ float dequantizeScale1 = 7.402795290545328e-06;
__constant__ float scaleIn2 = 0.0014761027414351702;
__constant__ float scaleOut2 = 0.004053246695548296;
__constant__ float dequantizeScale2 = 5.983008577459259e-06;
__constant__ float scaleIn3 = 0.0013351700035855174;
__constant__ float scaleOut3 = 0.004780988208949566;
__constant__ float dequantizeScale3 = 6.383432264556177e-06;
__constant__ float scaleIn4 = 0.0031514717265963554;
__constant__ float scaleOut4 = 0.004843319766223431;
__constant__ float dequantizeScale4 = 1.526358573755715e-05;
#endif

#ifdef METAL3
__constant__ float scaleIn1 = 0.002941413316875696;
__constant__ float scaleOut1 = 0.0039719571359455585;
__constant__ float dequantizeScale1 = 1.1683167940645944e-05;
__constant__ float scaleIn2 = 0.0016188048757612705;
__constant__ float scaleOut2 = 0.007177545223385096;
__constant__ float dequantizeScale2 = 1.1619044926192146e-05;
__constant__ float scaleIn3 = 0.0015138275921344757;
__constant__ float scaleOut3 = 0.010399818420410156;
__constant__ float dequantizeScale3 = 1.5743531548650935e-05;
__constant__ float scaleIn4 = 0.004079051315784454;
__constant__ float scaleOut4 = 0.005417930893599987;
__constant__ float dequantizeScale4 = 2.2100017304182984e-05;
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

#ifdef TILE2
__constant__ float scaleIn1 = 0.0021086351480334997;
__constant__ float scaleOut1 = 0.0029193938244134188;
__constant__ float dequantizeScale1 = 6.1559362620755564e-06;
__constant__ float scaleIn2 = 0.0015336197102442384;
__constant__ float scaleOut2 = 0.003363907802850008;
__constant__ float dequantizeScale2 = 5.158955445949687e-06;
__constant__ float scaleIn3 = 0.0009424724266864359;
__constant__ float scaleOut3 = 0.0051096538081765175;
__constant__ float dequantizeScale3 = 4.815707598027075e-06;
__constant__ float scaleIn4 = 0.0011210687225684524;
__constant__ float scaleOut4 = 0.005765164736658335;
__constant__ float dequantizeScale4 = 6.463145837187767e-06;
#endif

#ifdef TILE3
__constant__ float scaleIn1 = 0.0014810031279921532;
__constant__ float scaleOut1 = 0.005262565799057484;
__constant__ float dequantizeScale1 = 7.793876648065634e-06;
__constant__ float scaleIn2 = 0.0008918925886973739;
__constant__ float scaleOut2 = 0.004719282500445843;
__constant__ float dequantizeScale2 = 4.209093276585918e-06;
__constant__ float scaleIn3 = 0.0016177105717360973;
__constant__ float scaleOut3 = 0.00512092188000679;
__constant__ float dequantizeScale3 = 8.28416978038149e-06;
__constant__ float scaleIn4 = 0.004170930944383144;
__constant__ float scaleOut4 = 0.0026971197221428156;
__constant__ float dequantizeScale4 = 1.1249499948462471e-05;
#endif

#ifdef FABRIC
__constant__ float scaleIn1 = 0.0024292529560625553;
__constant__ float scaleOut1 = 0.002898578764870763;
__constant__ float dequantizeScale1 = 7.041381195449503e-06;
__constant__ float scaleIn2 = 0.0018462417647242546;
__constant__ float scaleOut2 = 0.004594826605170965;
__constant__ float dequantizeScale2 = 8.483160854666494e-06;
__constant__ float scaleIn3 = 0.0009650049614720047;
__constant__ float scaleOut3 = 0.0065849218517541885;
__constant__ float dequantizeScale3 = 6.354482138704043e-06;
__constant__ float scaleIn4 = 0.001465745852328837;
__constant__ float scaleOut4 = 0.005693068262189627;
__constant__ float dequantizeScale4 = 8.344591151399072e-06;
#endif
#ifdef FABRIC09
__constant__ float scaleIn1 = 0.003008828032761812;
__constant__ float scaleOut1 = 0.003582818666473031;
__constant__ float dequantizeScale1 = 1.0780085176520515e-05;
__constant__ float scaleIn2 = 0.0019265773007646203;
__constant__ float scaleOut2 = 0.0033055702224373817;
__constant__ float dequantizeScale2 = 6.368436515913345e-06;
__constant__ float scaleIn3 = 0.0011477346997708082;
__constant__ float scaleOut3 = 0.005390819162130356;
__constant__ float dequantizeScale3 = 6.187230155774159e-06;
__constant__ float scaleIn4 = 0.0010917092440649867;
__constant__ float scaleOut4 = 0.00801730714738369;
__constant__ float dequantizeScale4 = 8.752568646741565e-06;
#endif
#ifdef WEAVE
__constant__ float scaleIn1 = 0.002025123918429017;
__constant__ float scaleOut1 = 0.0036470419727265835;
__constant__ float dequantizeScale1 = 7.385711796814576e-06;
__constant__ float scaleIn2 = 0.0017646728083491325;
__constant__ float scaleOut2 = 0.007439339999109507;
__constant__ float dequantizeScale2 = 1.3128001228324138e-05;
__constant__ float scaleIn3 = 0.0012104109628126025;
__constant__ float scaleOut3 = 0.009341366589069366;
__constant__ float dequantizeScale3 = 1.130689270212315e-05;
__constant__ float scaleIn4 = 0.001690503559075296;
__constant__ float scaleOut4 = 0.012903540395200253;
__constant__ float dequantizeScale4 = 2.1813480998389423e-05;
#endif

#ifdef DUMMY
__constant__ float scaleIn1 = 0.0009549204260110855;
__constant__ float scaleOut1 = 0.0024820209946483374;
__constant__ float dequantizeScale1 = 2.370132506257505e-06;
__constant__ float scaleIn2 = 0.0030207473319023848;
__constant__ float scaleOut2 = 0.0020915379282087088;
__constant__ float dequantizeScale2 = 6.318007763184141e-06;
__constant__ float scaleIn3 = 0.004318909253925085;
__constant__ float scaleOut3 = 0.002192113781347871;
__constant__ float dequantizeScale3 = 9.4675406216993e-06;
__constant__ float scaleIn4 = 0.006152340676635504;
__constant__ float scaleOut4 = 0.002306189388036728;
__constant__ float dequantizeScale4 = 1.4188462955644354e-05;
#endif
