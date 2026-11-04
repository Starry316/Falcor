#pragma once
#include "model_types.h"

// Shared with the OptiX raygen; all large buffers remain on the GPU.
struct CoopLaunch {
    Model model;
    LobeNetwork lobe;
    LocalCache local;
    float* outputs;
    const unsigned char* weights;
    unsigned matrix[6],bias[6];
    int count,periodic;
    float crop;
    __half* appearance_codes=nullptr;
    const float4* directions=nullptr;
    bool packed_lobes=false;
};
// mode: 0 lobe inference, 1 source ray encoder, 2 source appearance decoder.
void* coop_backend_create(const Model& model,const LobeNetwork& lobe,cudaStream_t stream,int mode=0);
void coop_backend_destroy(void* backend);
void coop_backend_launch(void* backend,const Model& model,const LobeNetwork& lobe,
    LocalCache local,int count,bool periodic,float crop,float* outputs,cudaStream_t stream,
    __half* appearance_codes=nullptr,const float4* directions=nullptr,bool packed_lobes=false);
