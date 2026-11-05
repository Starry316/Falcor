#include "coop_backend.h"
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <type_traits>
#include "coop_ptx.h"

namespace {
void check(OptixResult r){if(r!=OPTIX_SUCCESS)throw std::runtime_error(std::string("Cooperative vector: ")+optixGetErrorName(r));}
void cucheck(cudaError_t r){if(r!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(r));}
size_t align64(size_t n){return (n+63)/64*64;}
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {char header[OPTIX_SBT_RECORD_HEADER_SIZE];};
struct CoopBackend {
    // A reference frame may enqueue several decodes. Two slots throttled the
    // CPU to copies from the previous frame even though GPU ordering was valid.
    // Keep a bounded pinned ring; wait only before reusing its own copied slot.
    static constexpr unsigned launch_slots=32;
    OptixDeviceContext context=nullptr;OptixModule module=nullptr;OptixPipeline pipeline=nullptr;OptixProgramGroup raygen=nullptr;
    Record* record=nullptr;unsigned char* weights=nullptr;CoopLaunch* launch=nullptr;CoopLaunch* staging=nullptr;
    cudaEvent_t copied[launch_slots]={};unsigned slot=0,matrix[6]={},bias[6]={};OptixShaderBindingTable sbt={};
    // Own bound-value storage for at least as long as the compiled module.
    // Bind only immutable layout scalars, never runtime pointers or options.
    CoopLaunch specialization={};std::vector<OptixModuleCompileBoundValueEntry> bound_values;
    ~CoopBackend(){
        if(pipeline)optixPipelineDestroy(pipeline);if(raygen)optixProgramGroupDestroy(raygen);if(module)optixModuleDestroy(module);
        cudaFree(record);cudaFree(weights);cudaFree(launch);for(auto e:copied)if(e)cudaEventDestroy(e);if(staging)cudaFreeHost(staging);
        if(context)optixDeviceContextDestroy(context);
    }
};
void specialize_layout(CoopBackend& b,const Model& model,const LobeNetwork& lobe){
    static_assert(std::is_standard_layout_v<CoopLaunch>&&std::is_standard_layout_v<Model>&&
        std::is_standard_layout_v<LobeNetwork>,"Cooperative launch offsets require standard-layout types");
    // Load-time diagnostic switch for matched specialized/unspecialized runs.
    const char* option=std::getenv("MPMAT_COOP_SPECIALIZE");if(option&&std::strcmp(option,"0")==0)return;
#ifndef MPMAT_INFERENCE_EXPORTS
    // Enable by default for the measured external-renderer SDK. Keep the
    // standalone renderer's broad experimental shape matrix opt-in: every
    // distinct specialized layout has its own initial OptiX compilation cost.
    if(!option||std::strcmp(option,"1")!=0)return;
#endif
    auto& p=b.specialization;p.model=model;p.lobe=lobe;
    std::copy(b.matrix,b.matrix+6,p.matrix);std::copy(b.bias,b.bias+6,p.bias);
    auto bind=[&](size_t offset,size_t size,const char* name){
        b.bound_values.push_back({offset,size,reinterpret_cast<const unsigned char*>(&p)+offset,name});
    };
    // Per-field entries exclude padding and all sampling controls that may
    // change after creating the pipeline (LOD, filtering and synthesis).
#define BIND_MODEL(field) bind(offsetof(CoopLaunch,model)+offsetof(Model,field),sizeof(p.model.field),"model." #field)
    BIND_MODEL(k);BIND_MODEL(r);BIND_MODEL(c);BIND_MODEL(w);BIND_MODEL(aw);BIND_MODEL(z);
    BIND_MODEL(bands);BIND_MODEL(angular);BIND_MODEL(stride);BIND_MODEL(kind);
    BIND_MODEL(angle_mode);BIND_MODEL(angle_r);BIND_MODEL(angle_c);BIND_MODEL(frames);
    BIND_MODEL(appearance_inputs);BIND_MODEL(depth);BIND_MODEL(mip_levels);
#undef BIND_MODEL
#define BIND_LOBE(field) bind(offsetof(CoopLaunch,lobe)+offsetof(LobeNetwork,field),sizeof(p.lobe.field),"lobe." #field)
    BIND_LOBE(latent);BIND_LOBE(learned_alpha);BIND_LOBE(inputs);BIND_LOBE(lobes);
    BIND_LOBE(stride);BIND_LOBE(parameterization);
#undef BIND_LOBE
    const size_t network=offsetof(CoopLaunch,lobe)+offsetof(LobeNetwork,network);
    bind(network+offsetof(Model,w),sizeof(p.lobe.network.w),"lobe.network.w");
    bind(network+offsetof(Model,depth),sizeof(p.lobe.network.depth),"lobe.network.depth");
    bind(offsetof(CoopLaunch,matrix),sizeof(p.matrix),"matrix");
    bind(offsetof(CoopLaunch,bias),sizeof(p.bias),"bias");
}
}
void* coop_backend_create(const Model& m,const LobeNetwork& l,cudaStream_t stream,int mode){
    if(mode==0&&!l.latent&&((m.c!=2&&m.c!=4)||m.angular!=3||m.frames||m.angle_mode||l.inputs>64||
        (l.network.w!=16&&l.network.w!=32&&l.network.w!=64)||l.stride>32||!l.inputs)
        )
        throw std::runtime_error("Cooperative vector requires C2/C4, <=64 inputs, width 16/32/64 and <=4 lobes");
    if(mode==0&&l.latent&&(m.kind!=8||m.k!=10||m.c!=2||m.z!=16||(m.w!=16&&m.w!=64)||
        (l.inputs!=16&&l.inputs!=19)||(l.network.w!=21&&l.network.w!=32&&l.network.w!=64)||l.lobes!=3||l.parameterization))
        throw std::runtime_error("Cooperative latent IBL requires K10 C2 latent16, ray width 16/64, IBL width 21/32/64 and 3 canonical lobes");
    const int ray_inputs=m.k*(m.c+(m.kind==8?0:1))+3;
    if(mode&&(m.z>31||ray_inputs>64||m.appearance_inputs>64||m.w>128||m.aw>128||m.frames||m.angle_mode))
        throw std::runtime_error("Cooperative appearance requires <=31 latent, <=64 inputs, <=128 hidden and raw angular features");
    auto* b=new CoopBackend;
    try{
        check(optixInit());OptixDeviceContextOptions options={};check(optixDeviceContextCreate(nullptr,&options,&b->context));
        unsigned support=0;check(optixDeviceContextGetProperty(b->context,OPTIX_DEVICE_PROPERTY_COOP_VEC,&support,sizeof(support)));
        if(!(support&OPTIX_DEVICE_PROPERTY_COOP_VEC_FLAG_STANDARD))throw std::runtime_error("Device/driver does not support OptiX cooperative vectors");
        // Convert once to the hardware's optimal inference layout. Preserve FP16
        // weights/biases exactly; padding contributes only zero matrix columns.
        int width=l.network.w,padded=(l.inputs+15)/16*16,head_width=l.latent?((width+15)/16*16):width;
        struct Layer {int k,n,pk,pn;const __half* matrix;const __half* bias;};std::vector<Layer> network;
        if(mode==1){
            width=m.w;head_width=16;while(head_width<width)head_width*=2;padded=(ray_inputs+15)/16*16;
            for(int i=0;i<=m.depth;++i){int k=i?m.w:ray_inputs,n=i==m.depth?m.z+(m.kind==8?0:1):m.w;
                network.push_back({k,n,i?head_width:padded,i==m.depth?32:head_width,m.half_matrix[i],m.half_bias[i]});}
        }else if(mode==2){
            width=m.aw;head_width=16;while(head_width<width)head_width*=2;padded=(m.appearance_inputs+15)/16*16;
            int depth=((m.kind==4&&m.depth==1)||m.kind==7)?1:2,start=m.depth+1;
            for(int i=0;i<=depth;++i){int k=i?m.aw:m.appearance_inputs,n=i==depth?3:m.aw;
                network.push_back({k,n,i?head_width:padded,i==depth?16:head_width,m.half_matrix[start+i],m.half_bias[start+i]});}
        }else{
        if(l.latent)for(int i=0;i<=m.depth;++i){int k=i?m.w:23,n=i==m.depth?16:m.w;
            network.push_back({k,n,i?m.w:32,n,m.half_matrix[i],m.half_bias[i]});}
        for(int i=0;i<=l.network.depth;++i){int k=i?width:l.inputs,n=i==l.network.depth?((l.latent&&!l.learned_alpha)?21:l.stride):width;
            network.push_back({k,n,i?head_width:padded,i==l.network.depth?32:head_width,l.network.half_matrix[i],l.network.half_bias[i]});}
        }
        int layers=int(network.size());
        std::vector<OptixCoopVecMatrixDescription> src(layers),dst(layers);
        size_t src_bytes=0,dst_bytes=0;
        for(int i=0;i<layers;++i){
            unsigned k=network[i].pk,n=network[i].pn;
            size_t bytes=0;check(optixCoopVecMatrixComputeSize(b->context,n,k,OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,
                OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,0,&bytes));
            src[i]={n,k,unsigned(src_bytes),OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,OPTIX_COOP_VEC_MATRIX_LAYOUT_ROW_MAJOR,k*2,n*k*2};
            dst[i]={n,k,unsigned(dst_bytes),OPTIX_COOP_VEC_ELEM_TYPE_FLOAT16,OPTIX_COOP_VEC_MATRIX_LAYOUT_INFERENCING_OPTIMAL,0,unsigned(bytes)};
            b->matrix[i]=unsigned(dst_bytes);src_bytes=align64(src_bytes+n*k*2);dst_bytes=align64(dst_bytes+bytes);
        }
        for(int i=0;i<layers;++i){b->bias[i]=unsigned(dst_bytes);dst_bytes=align64(dst_bytes+dst[i].N*2);}
        std::vector<unsigned char> raw(src_bytes),converted(dst_bytes);
        for(int i=0;i<layers;++i){
            int k=network[i].k,n=network[i].n;
            std::vector<__half> matrix(size_t(k)*n),bias(n);
            cucheck(cudaMemcpy(matrix.data(),network[i].matrix,matrix.size()*2,cudaMemcpyDeviceToHost));
            cucheck(cudaMemcpy(bias.data(),network[i].bias,bias.size()*2,cudaMemcpyDeviceToHost));
            auto* target=reinterpret_cast<__half*>(raw.data()+src[i].offsetInBytes);
            for(int j=0;j<n;++j)for(int v=0;v<k;++v)target[j*src[i].K+v]=matrix[v*n+j];
            std::memcpy(converted.data()+b->bias[i],bias.data(),n*2);
        }
        unsigned char* input=nullptr;
        cucheck(cudaMalloc(&b->weights,dst_bytes));cucheck(cudaMemcpyAsync(b->weights,converted.data(),dst_bytes,cudaMemcpyHostToDevice,stream));
        cucheck(cudaMalloc(&input,src_bytes));
        try{
            cucheck(cudaMemcpyAsync(input,raw.data(),src_bytes,cudaMemcpyHostToDevice,stream));
            OptixNetworkDescription in={src.data(),unsigned(layers)},out={dst.data(),unsigned(layers)};
            check(optixCoopVecMatrixConvert(b->context,stream,1,&in,reinterpret_cast<CUdeviceptr>(input),0,
                &out,reinterpret_cast<CUdeviceptr>(b->weights),0));cucheck(cudaStreamSynchronize(stream));
        }catch(...){cudaFree(input);throw;}cudaFree(input);
        OptixModuleCompileOptions module_options={};module_options.optLevel=OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        specialize_layout(*b,m,l);module_options.boundValues=b->bound_values.data();
        module_options.numBoundValues=unsigned(b->bound_values.size());
        OptixPipelineCompileOptions compile={};compile.pipelineLaunchParamsVariableName="coop_params";
        std::string entry=l.latent?("__raygen__mip"+std::to_string(m.w)+"_"+std::to_string(head_width)+"_"+std::to_string(padded)):
            ("__raygen__coop"+std::to_string(width)+"_"+std::to_string(padded));
        if(mode)entry=std::string(mode==1?"__raygen__encode":"__raygen__appearance")+std::to_string(head_width)+"_"+std::to_string(padded);
        // Specialize only the kernel used by this pipeline. Compiling every
        // architecture/reference variant here can stall cold loads for minutes.
        const auto ptx=mpmat_coop_ptx_for_entry(entry);
        char log[8192]={};size_t log_size=sizeof(log);
        auto result=optixModuleCreate(b->context,&module_options,&compile,ptx.data(),ptx.size(),log,&log_size,&b->module);
        if(result!=OPTIX_SUCCESS)throw std::runtime_error(std::string("Cooperative vector module: ")+log);
        OptixProgramGroupOptions group_options={};OptixProgramGroupDesc desc={};
        desc.kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;desc.raygen.module=b->module;desc.raygen.entryFunctionName=entry.c_str();
        check(optixProgramGroupCreate(b->context,&desc,1,&group_options,nullptr,nullptr,&b->raygen));
        OptixPipelineLinkOptions link={};link.maxTraceDepth=0;
        check(optixPipelineCreate(b->context,&compile,&link,&b->raygen,1,nullptr,nullptr,&b->pipeline));
        OptixStackSizes stacks={};check(optixUtilAccumulateStackSizes(b->raygen,&stacks,b->pipeline));unsigned traversal=0,state=0,continuation=0;
        check(optixUtilComputeStackSizes(&stacks,0,0,0,&traversal,&state,&continuation));check(optixPipelineSetStackSize(b->pipeline,traversal,state,continuation,1));
        Record record={};check(optixSbtRecordPackHeader(b->raygen,&record));cucheck(cudaMalloc(&b->record,sizeof(record)));
        cucheck(cudaMemcpy(b->record,&record,sizeof(record),cudaMemcpyHostToDevice));b->sbt.raygenRecord=reinterpret_cast<CUdeviceptr>(b->record);
        cucheck(cudaMalloc(&b->launch,sizeof(CoopLaunch)));cucheck(cudaMallocHost(&b->staging,CoopBackend::launch_slots*sizeof(CoopLaunch)));
        for(auto& e:b->copied)cucheck(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));return b;
    }catch(...){delete b;throw;}
}
void coop_backend_destroy(void* ptr){delete static_cast<CoopBackend*>(ptr);}
void coop_backend_launch(void* ptr,const Model& m,const LobeNetwork& l,LocalCache local,int count,bool periodic,float crop,float* outputs,cudaStream_t stream,
    __half* appearance_codes,const float4* directions,bool packed_lobes){
    auto& b=*static_cast<CoopBackend*>(ptr);unsigned slot=b.slot++%CoopBackend::launch_slots;cucheck(cudaEventSynchronize(b.copied[slot]));
    // Ordinary lobe rendering leaves appearance_codes null. Reference modes
    // opt into source-code writes, reusing the mip encoder in the same launch.
    auto& p=b.staging[slot];p={};p.model=m;p.lobe=l;p.local=local;p.outputs=outputs;p.weights=b.weights;
    p.appearance_codes=appearance_codes;p.directions=directions;p.packed_lobes=packed_lobes;
    std::copy(b.matrix,b.matrix+6,p.matrix);std::copy(b.bias,b.bias+6,p.bias);p.count=count;p.periodic=periodic;p.crop=crop;
#ifndef NDEBUG
    // Launch parameters must still contain the actual values: specialization
    // is an optional compiler optimization, not a substitute for valid input.
    for(const auto& value:b.bound_values)if(std::memcmp(
        reinterpret_cast<const unsigned char*>(&p)+value.pipelineParamOffsetInBytes,
        value.boundValuePtr,value.sizeInBytes)!=0)
        throw std::runtime_error(std::string("Cooperative vector immutable layout changed: ")+value.annotation);
#endif
    cucheck(cudaMemcpyAsync(b.launch,&p,sizeof(p),cudaMemcpyHostToDevice,stream));cucheck(cudaEventRecord(b.copied[slot],stream));
    check(optixLaunch(b.pipeline,stream,reinterpret_cast<CUdeviceptr>(b.launch),sizeof(p),&b.sbt,unsigned(count),1,1));
}
