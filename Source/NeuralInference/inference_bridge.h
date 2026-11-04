#pragma once
// GPU-resident, FP16 cooperative-vector inference. No geometry traversal or GL.
// All functions return 1 on success, 0 on error (except error/destroy).
#include <stdint.h>
#if defined(_WIN32) && defined(MPMAT_INFERENCE_EXPORTS)
#define MPMAT_INFER_API __declspec(dllexport)
#else
#define MPMAT_INFER_API
#endif
#define MPMAT_INFER_ABI_VERSION 1u
#define MPMAT_INFER_MAX_MIPS 11u

// Byte layout is shared with Slang StructuredBuffer (48 bytes, four-byte fields).
// Material-space origin/direction preserve FP32 precision; view points toward
// the camera in the orthonormal local shading frame. LOD is already resolved
// at reference Z, including bias/repeat/crop once. valid==0 writes zero lobes.
typedef struct MpmatInferenceRay {
    float origin[3];
    float direction[3];
    float view[3];
    float lod;
    uint32_t valid;
    uint32_t reserved;
} MpmatInferenceRay;

enum MpmatInferenceFlags {
    MPMAT_INFER_PERIODIC = 1u,
    MPMAT_INFER_NEAREST_FEATURE = 2u,
    // Also prepare source appearance codes during lobe inference. The latent
    // mip path reuses its ray encoder; raw heads run a separate source encoder.
    MPMAT_INFER_CACHE_APPEARANCE = 4u
};
typedef struct MpmatInferenceOptions {
    uint32_t struct_size; // sizeof(MpmatInferenceOptions)
    uint32_t flags;       // MpmatInferenceFlags
    float source_crop;    // [0,.5); applies within each repeated source tile
    float fixed_lod;      // -1 uses Ray.lod; otherwise clamped fractional LOD
} MpmatInferenceOptions;

typedef struct MpmatInferenceInfo {
    uint32_t struct_size; // caller sets sizeof(MpmatInferenceInfo)
    uint32_t abi_version;
    uint32_t ray_bytes;
    uint32_t output_stride_floats; // 7*lobes+1
    uint32_t model_kind;
    uint32_t planes;
    uint32_t channels;
    uint32_t base_resolution;
    uint32_t mip_levels;  // non-mip assets expose one real mip level
    uint32_t mip_resolutions[MPMAT_INFER_MAX_MIPS];
    uint32_t lobes;
    uint32_t head_input_kind; // 0 raw features/valid/view, 1 latent(+view)
    uint32_t coverage_kind;   // 0 independent head sigmoid, 1 constant one
    uint32_t head_parameterization;
    float bounds[6]; // xmin,ymin,zmin,xmax,ymax,zmax
    float minimum_roughness;
    float parallel_epsilon;
    float decode_mu;
    uint32_t capabilities; // bit0=coop lobes; bit1=FP16 appearance; bit2=synthesis; bit3=coop appearance/cache flag; bit4=packed FP16 lobes
} MpmatInferenceInfo;

#ifdef __cplusplus
extern "C" {
#endif
// Reads the model/head sections of an exported .mpmat asset. Geometry/BVH
// sections are skipped and never uploaded. Creation converts matrices once.
MPMAT_INFER_API int mpmat_infer_create(const char* asset_path_utf8, int cuda_device, void** context);
MPMAT_INFER_API int mpmat_infer_get_info(void* context, MpmatInferenceInfo* info);
// Call at setup/resize. May allocate and wait for previous work of this context.
MPMAT_INFER_API int mpmat_infer_reserve(void* context, uint32_t max_rays);
// Rendering-only legacy plane synthesis, same settings as the native viewer.
// mode 0=off, 1=hex, 2=histogram. Nonzero modes reject learned-mip architectures.
// May preprocess/upload once; call only when controls change, outside frame timing.
MPMAT_INFER_API int mpmat_infer_set_synthesis(void* context, int mode, int seed,
    float scale, float power, float rotation, float border_contrast);
// Device pointers must belong to cuda_device and remain alive until the stream
// completes. This enqueues work with no frame upload/readback or device-wide
// synchronization. Reusing a full launch-parameter ring may wait for that
// slot's prior asynchronous host-to-device parameter copy to finish.
// Caller serializes API calls on a context; GPU streams are internally ordered.
// Separate contexts may run independently.
// Output is local axis XYZ, roughness, RGB energy per lobe, then coverage alpha.
// No cosine, F/G, light color, tone mapping or output quantization is applied.
MPMAT_INFER_API int mpmat_infer_lobes(void* context, const MpmatInferenceRay* device_rays,
    uint32_t count, const MpmatInferenceOptions* options, float* device_lobes, void* cuda_stream);
// Same decoded values as infer_lobes, stored as IEEE FP16 bits. The current
// cooperative decoder already rounds each component to FP16 before widening
// to float, so this transport does not add numerical quantization. Stride is
// (output_stride_floats+7)&~7 half elements; trailing padding and invalid rays
// are all zero. device_lobes must be 16-byte aligned (capability bit4).
MPMAT_INFER_API int mpmat_infer_lobes_fp16(void* context, const MpmatInferenceRay* device_rays,
    uint32_t count, const MpmatInferenceOptions* options, uint16_t* device_lobes, void* cuda_stream);
// Optional original-network reference. Prepare ray codes once after camera or
// material changes, then shade any number of direct-light direction buffers.
// Both infer_lobes variants invalidate codes unless CACHE_APPEARANCE was requested. Explicit
// prepare always re-encodes its inputs. Supported compact shapes use cooperative
// vectors (capability bit3); other compatible source shapes retain the shared
// ordinary FP16 CUDA oracle. No FP32 inference path is selected.
MPMAT_INFER_API int mpmat_infer_prepare_appearance(void* context, const MpmatInferenceRay* device_rays,
    uint32_t count, const MpmatInferenceOptions* options, void* cuda_stream);
// Directions are N local float4s (xyz used, w ignored), surface-to-light.
// RGBA output is N float4s: decoded scene-linear RGB * source coverage, alpha.
// No Li, extra cosine/G, visibility, hemisphere clipping or display conversion.
MPMAT_INFER_API int mpmat_infer_appearance(void* context, uint32_t count,
    const float* device_local_directions_xyzw, float* device_rgba, void* cuda_stream);
MPMAT_INFER_API void mpmat_infer_destroy(void* context);
MPMAT_INFER_API const char* mpmat_infer_error(void);
#ifdef __cplusplus
}
static_assert(sizeof(MpmatInferenceRay)==48, "Inference ray ABI");
static_assert(sizeof(MpmatInferenceOptions)==16, "Inference options ABI");
#endif
#undef MPMAT_INFER_API
