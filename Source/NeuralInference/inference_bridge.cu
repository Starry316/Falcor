#include "inference_bridge.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "model_types.h"
#include "coop_backend.h"
#include "synthesis.h"
#define CU(call) do {auto status=(call);if(status!=cudaSuccess)throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(status));}while(0)
#include "feature_sampling.h"
#include "rgba16f.h"
constexpr int WARPS=4, THREADS=128, MAX_INPUT=640;
__device__ float fdot(float3 a,float3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
__device__ float3 fnorm(float3 a){float s=1.f/fmaxf(sqrtf(fdot(a,a)),1e-8f);return make_float3(a.x*s,a.y*s,a.z*s);}
#include "inference_bridge_networks.h"

namespace {
thread_local std::string error_message;
template<class T> void upload(T*& device,const std::vector<T>& values){
    CU(cudaMalloc(&device,values.size()*sizeof(T)));
    CU(cudaMemcpy(device,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
}
std::vector<__half> half_values(const std::vector<float>& values){
    std::vector<__half> result(values.size());
    for(size_t i=0;i<values.size();++i){
        if(!std::isfinite(values[i])||fabsf(values[i])>65504.f)throw std::runtime_error("Asset contains a value outside FP16 range");
        result[i]=__float2half_rn(values[i]);
    }
    return result;
}
struct Reader {
    std::ifstream file;
    uint64_t remaining;
    explicit Reader(const char* path):file(std::filesystem::u8path(path),std::ios::binary|std::ios::ate){
        if(!file)throw std::runtime_error("Cannot open inference asset");
        remaining=uint64_t(file.tellg());file.seekg(0);
    }
    void skip(uint64_t bytes){
        if(bytes>remaining)throw std::runtime_error("Truncated inference asset");
        file.seekg(std::streamoff(bytes),std::ios::cur);remaining-=bytes;
        if(!file)throw std::runtime_error("Failed seeking inference asset");
    }
    template<class T> std::vector<T> read(size_t count){
        if(count>remaining/sizeof(T))throw std::runtime_error("Truncated inference asset");
        std::vector<T> result(count);file.read(reinterpret_cast<char*>(result.data()),count*sizeof(T));
        if(!file)throw std::runtime_error("Failed reading inference asset");remaining-=count*sizeof(T);return result;
    }
    std::string magic(){auto value=read<char>(16);return std::string(value.data(),std::find(value.begin(),value.end(),'\0')-value.begin());}
};
struct Context {
    int device=0;Model model={};LobeNetwork head={};void* coop=nullptr;
    uint32_t capacity=0;cudaEvent_t completed=nullptr;bool submitted=false;
    void* appearance_encoder=nullptr;void* appearance_decoder=nullptr;
    __half* appearance_codes=nullptr;uint32_t appearance_count=0;
    std::vector<float> source_features; // exact FP32 source for optional histogram preprocessing
    MpmatInferenceInfo info={};
    ~Context(){
        cudaSetDevice(device);if(submitted&&completed)cudaEventSynchronize(completed);
        coop_backend_destroy(coop);coop_backend_destroy(appearance_encoder);coop_backend_destroy(appearance_decoder);
        cudaFree(appearance_codes);if(completed)cudaEventDestroy(completed);
        release_mip_texture(model.mip_texture);cudaFree(model.heights);cudaFree(model.half_weights);
        release_feature_texture(model.gaussian_texture);cudaFree(model.half_inverse_histogram);
        cudaFree(head.network.half_weights);cudaFree(head.half_normalization);
    }
};
void affine_layout(Model& model,const std::vector<float>& weights,const std::vector<int>& inputs,const std::vector<int>& outputs){
    size_t total=0;for(size_t i=0;i<inputs.size();++i)total+=size_t(inputs[i]+1)*outputs[i];
    if(total!=weights.size())throw std::runtime_error("Inference network layout/weight count mismatch");
    upload(model.half_weights,half_values(weights));size_t offset=0;
    for(size_t i=0;i<inputs.size();++i){model.half_matrix[i]=model.half_weights+offset;offset+=size_t(inputs[i])*outputs[i];
        model.half_bias[i]=model.half_weights+offset;offset+=outputs[i];}
    model.fp16=true;
}
void load(Context& context,const char* path){
    Reader reader(path);auto magic=reader.magic();Model& m=context.model;LobeNetwork& h=context.head;
    const bool coverage_mip=magic=="MPMATCUDA11";
    if(!coverage_mip&&(magic.size()!=10||magic.substr(0,9)!="MPMATCUDA"||magic[9]<'2'||magic[9]>'9'))
        throw std::runtime_error("Expected an MPMat native model with a matched lobe head");
    m.kind=reader.read<int>(1)[0];
    const int asset_kind=m.kind;
    if(m.kind!=2&&m.kind!=4&&m.kind!=5&&m.kind!=7&&m.kind!=8&&!(coverage_mip&&m.kind==10))
        throw std::runtime_error("Inference bridge supports raw-direction angular/compact and learned-mip assets");
    if(coverage_mip?m.kind!=10:magic[9]-'1'!=m.kind)throw std::runtime_error("Native model magic/kind mismatch");
    auto dims=reader.read<int>(coverage_mip?10:m.kind==8?12:m.kind==2?13:m.kind==4?10:11);
    // Coverage mips have exactly the ordinary angular-D network layout.
    // Keep that internal layout while retaining native kind10 in public metadata.
    if(coverage_mip)m.kind=2;
    m.k=dims[0];m.r=dims[1];m.c=dims[2];m.w=dims[3];m.z=dims[4];m.bands=dims[5];m.angular=3;m.stride=m.z+4;
    m.aw=(m.kind==5||m.kind==7||m.kind==8)?dims[10]:m.w;
    m.depth=(m.kind==4||m.kind==8)?dims[9]:2;m.appearance_inputs=m.z+12;
    if(m.k<1||m.k>32||m.r<2||m.r>16384||(m.c!=2&&m.c!=4)||m.w<1||m.w>128||m.aw<1||m.aw>128||m.z<1||m.z>64||m.bands||
       (m.depth!=1&&m.depth!=2)||dims[6]<1||dims[7]<1||dims[8]!=dims[6])throw std::runtime_error("Invalid inference model dimensions");
    if(m.kind==2&&!coverage_mip&&(dims[9]!=0||dims[12]!=0))throw std::runtime_error("Inference bridge requires angular D without learned frames");
    m.mip_levels=coverage_mip?dims[9]:m.kind==8?dims[11]:1;
    if(m.mip_levels<1||m.mip_levels>11||m.r%(1<<(m.mip_levels-1))||(m.r>>(m.mip_levels-1))<2)
        throw std::runtime_error("Invalid learned mip dimensions");
    auto scalar=reader.read<float>(2),bounds=reader.read<float>(6),heights=reader.read<float>(m.k);
    m.mu=scalar[0];m.epsilon=scalar[1];std::copy(bounds.begin(),bounds.end(),m.bounds);
    if(!std::isfinite(m.mu)||m.mu<0||!std::isfinite(m.epsilon)||m.epsilon<=0)throw std::runtime_error("Invalid model scalar metadata");
    for(int i=0;i<3;++i)if(!std::isfinite(bounds[i])||!std::isfinite(bounds[i+3])||bounds[i+3]<=bounds[i])throw std::runtime_error("Invalid model bounds");
    for(int i=0;i<m.k;++i)if(!std::isfinite(heights[i])||heights[i]<bounds[2]||heights[i]>bounds[5]||(i&&heights[i]<=heights[i-1]))throw std::runtime_error("Invalid plane heights");
    size_t features_count=0;for(int level=0;level<m.mip_levels;++level){m.mip_offsets[level]=features_count;int r=m.r>>level;features_count+=size_t(m.k)*m.c*r*r;}
    auto features=reader.read<float>(features_count);half_values(features); // reject overflow before hardware upload
    upload(m.heights,heights);upload_mip_texture(m.mip_texture,features.data(),m.k,m.r,m.c,m.mip_levels,m.mip_offsets);
    if(m.kind!=8&&!coverage_mip)context.source_features=std::move(features);
    // Preserve raw and latent architecture metadata; one-level texture storage
    // does not remove validity inputs or turn legacy networks into mip networks.
    std::vector<int> inputs={m.kind==8?m.k*m.c+3:m.k*(m.c+1)+3},outputs={m.w};
    if(m.depth==2){inputs.push_back(m.w);outputs.push_back(m.w);}
    inputs.push_back(m.w);outputs.push_back(m.z+(m.kind==8?0:1));
    inputs.push_back(m.appearance_inputs);outputs.push_back(m.aw);
    if(!(m.kind==4&&m.depth==1)&&m.kind!=7){inputs.push_back(m.aw);outputs.push_back(m.aw);}
    inputs.push_back(m.aw);outputs.push_back(3);
    size_t weight_count=0;for(size_t i=0;i<inputs.size();++i)weight_count+=size_t(inputs[i]+1)*outputs[i];
    affine_layout(m,reader.read<float>(weight_count),inputs,outputs);
    // Geometry, acceleration, and camera framing belong exclusively to Falcor.
    reader.skip(uint64_t(dims[6])*27*8+uint64_t(dims[7])*6*8+uint64_t(dims[7])*4*4+uint64_t(dims[8])*4+6*8);
    bool found=false;
    while(reader.remaining){
        auto extension=reader.magic();
        if(extension=="MPMATCURV1"){int vertices=reader.read<int>(1)[0];if(vertices<1)throw std::runtime_error("Invalid curvature extension");reader.skip(uint64_t(vertices)*7*4+uint64_t(dims[6])*3*4);continue;}
        if(extension.size()!=10||extension.substr(0,9)!="MPMATLOBE"||extension[9]<'1'||extension[9]>'5'||found)throw std::runtime_error("Unknown or duplicate native lobe extension");
        int version=extension[9]-'0';auto d=reader.read<int>(version+1);found=true;
        h.inputs=d[0];h.network.w=d[1];h.lobes=version>=2?d[2]:2;h.network.depth=version>=3?d[3]:2;
        h.parameterization=version>=4?d[4]:0;int layout=version==5?d[5]:0;
        h.latent=layout!=0;h.learned_alpha=layout==2;h.stride=7*h.lobes+1;h.network.kind=2;
        if((layout!=0&&layout!=1&&layout!=2)||(h.latent?(m.kind!=8||(h.inputs!=m.z&&h.inputs!=m.z+3)):h.inputs!=m.k*(m.c+1)+3)||
           h.network.w<1||h.network.w>128||h.lobes<1||h.lobes>8||(h.network.depth!=1&&h.network.depth!=2)||(h.parameterization!=0&&h.parameterization!=1))throw std::runtime_error("Invalid lobe architecture");
        auto settings=reader.read<float>(4),normalization=reader.read<float>(h.inputs*2);
        h.min_r=settings[0];h.log_min_r=logf(h.min_r);
        if(!std::isfinite(h.min_r)||h.min_r<=0||h.min_r>=1)throw std::runtime_error("Invalid minimum roughness");
        for(int j=0;j<3;++j){h.scale[j]=settings[j+1];if(!std::isfinite(h.scale[j])||h.scale[j]<=0||h.scale[j]>65504||float(__float2half_rn(h.scale[j]))<=0)throw std::runtime_error("Invalid lobe scale");}
        for(int j=h.inputs;j<h.inputs*2;++j)if(normalization[j]<=0||float(__float2half_rn(normalization[j]))<=0)throw std::runtime_error("Invalid normalization scale");
        upload(h.half_normalization,half_values(normalization));
        std::vector<int> ni={h.inputs},no={h.network.w};if(h.network.depth==2){ni.push_back(h.network.w);no.push_back(h.network.w);}
        ni.push_back(h.network.w);no.push_back(h.stride-((h.latent&&!h.learned_alpha)?1:0));
        size_t total=0;for(size_t i=0;i<ni.size();++i)total+=size_t(ni[i]+1)*no[i];
        affine_layout(h.network,reader.read<float>(total),ni,no);
    }
    if(!found)throw std::runtime_error("Inference asset has no matched lobe head");
    context.coop=coop_backend_create(m,h,nullptr); // no generic fallback
    CU(cudaEventCreateWithFlags(&context.completed,cudaEventDisableTiming));
    auto& info=context.info;info.struct_size=sizeof(info);info.abi_version=MPMAT_INFER_ABI_VERSION;info.ray_bytes=sizeof(MpmatInferenceRay);
    info.output_stride_floats=h.stride;info.model_kind=asset_kind;info.planes=m.k;info.channels=m.c;info.base_resolution=m.r;info.mip_levels=m.mip_levels;
    for(int i=0;i<m.mip_levels;++i)info.mip_resolutions[i]=m.r>>i;
    info.lobes=h.lobes;info.head_input_kind=h.latent?1:0;info.coverage_kind=h.latent&&!h.learned_alpha?1:0;info.head_parameterization=h.parameterization;
    std::copy(bounds.begin(),bounds.end(),info.bounds);info.minimum_roughness=h.min_r;info.parallel_epsilon=m.epsilon;info.decode_mu=m.mu;info.capabilities=((m.kind==8||coverage_mip)?3u:7u)|16u;
    if(m.z<=31&&m.k*(m.c+(m.kind==8?0:1))+3<=64&&m.appearance_inputs<=64&&m.w<=128&&m.aw<=128&&!m.frames&&!m.angle_mode)
        info.capabilities|=8;
}
__global__ void appearance_reference(Model model,const __half* codes,uint32_t count,const float4* directions,float4* output){
    uint32_t id=blockIdx.x*WARPS+threadIdx.x/32;int warp=threadIdx.x/32,lane=threadIdx.x%32;if(id>=count)return;
    const __half* code=codes+size_t(id)*model.stride;float4 direction=directions[id];float3 wi=make_float3(direction.x,direction.y,direction.z);
    if(float(code[model.z])==0.f){if(!lane)output[id]=make_float4(0,0,0,0);return;}
    if(!isfinite(wi.x)||!isfinite(wi.y)||!isfinite(wi.z)){if(!lane)output[id]=make_float4(0,0,0,float(code[model.z]));return;}
    __shared__ __half aa[WARPS][MAX_INPUT],bb[WARPS][MAX_INPUT];__half* a=aa[warp];__half* b=bb[warp];
    prepare_appearance(model,code,fnorm(wi),a);decode_material(model,a,b);
    if(!lane){float rgb[3],alpha=float(code[model.z]);for(int j=0;j<3;++j){float encoded=float(softplus(b[j]));
        rgb[j]=(model.mu>0?expm1f(fminf(encoded*log1pf(model.mu),80.f))/model.mu:encoded)*alpha;}
        output[id]=make_float4(rgb[0],rgb[1],rgb[2],alpha);}
}
template<class F> int guarded(F&& function){try{function();error_message.clear();return 1;}catch(const std::exception& error){error_message=error.what();return 0;}}
Context& checked(void* value){if(!value)throw std::runtime_error("Null inference context");auto& c=*static_cast<Context*>(value);CU(cudaSetDevice(c.device));return c;}
Model sampling_model(Context& c,const MpmatInferenceOptions* options){
    if(!options||options->struct_size!=sizeof(*options))throw std::runtime_error("Inference options ABI size mismatch");
    if(options->flags&~7u||!std::isfinite(options->source_crop)||options->source_crop<0||options->source_crop>=.5f||
        !std::isfinite(options->fixed_lod)||(options->fixed_lod<0&&options->fixed_lod!=-1.f))throw std::runtime_error("Invalid inference sampling options");
    Model model=c.model;model.feature_lod=options->fixed_lod;model.nearest_fetch=(options->flags&MPMAT_INFER_NEAREST_FEATURE)!=0;return model;
}
void ensure_appearance(Context& c,cudaStream_t stream){
    if(c.info.capabilities&8u){
        if(!c.appearance_decoder)c.appearance_decoder=coop_backend_create(c.model,c.head,stream,2);
    }
    if(!c.appearance_codes)CU(cudaMalloc(&c.appearance_codes,size_t(c.capacity)*c.model.stride*sizeof(__half)));
}
void encode_appearance(Context& c,const Model& model,LocalCache rays,uint32_t count,const MpmatInferenceOptions* options,cudaStream_t stream){
    // A latent lobe launch already writes the exact source ray code. Compile
    // the separate encoder only for raw heads or an explicit prepare request.
    if((c.info.capabilities&8u)&&!c.appearance_encoder)c.appearance_encoder=coop_backend_create(c.model,c.head,stream,1);
    if(c.appearance_encoder)coop_backend_launch(c.appearance_encoder,model,c.head,rays,int(count),
        (options->flags&MPMAT_INFER_PERIODIC)!=0,options->source_crop,nullptr,stream,c.appearance_codes);
    else encode_kernel<<<(count+WARPS-1)/WARPS,THREADS,0,stream>>>(model,int(count),rays,
        (options->flags&MPMAT_INFER_PERIODIC)!=0,options->source_crop,c.appearance_codes,nullptr);
}
void enqueue_lobes(void* value,const MpmatInferenceRay* rays,uint32_t count,const MpmatInferenceOptions* options,
    void* output,void* cuda_stream,bool packed){
    auto& c=checked(value);Model model=sampling_model(c,options);
    if(!rays||!output||!count||count>c.capacity)throw std::runtime_error("Invalid device buffers/count; reserve capacity before inference");
    if(packed&&(reinterpret_cast<uintptr_t>(output)&15u))throw std::runtime_error("Packed FP16 lobe output must be 16-byte aligned");
    auto stream=static_cast<cudaStream_t>(cuda_stream);if(c.submitted)CU(cudaStreamWaitEvent(stream,c.completed,0));
    c.appearance_count=0;
    bool cache=(options->flags&MPMAT_INFER_CACHE_APPEARANCE)!=0;
    if(cache)ensure_appearance(c,stream);
    LocalCache local={nullptr,nullptr,nullptr,reinterpret_cast<const ExternalMaterialRay*>(rays)};
    coop_backend_launch(c.coop,model,c.head,local,int(count),(options->flags&MPMAT_INFER_PERIODIC)!=0,
        options->source_crop,static_cast<float*>(output),stream,cache&&c.head.latent?c.appearance_codes:nullptr,nullptr,packed);
    if(cache){if(!c.head.latent)encode_appearance(c,model,local,count,options,stream);c.appearance_count=count;}
    CU(cudaGetLastError());CU(cudaEventRecord(c.completed,stream));c.submitted=true;
}
}

extern "C" int mpmat_infer_create(const char* path,int device,void** output){
    if(output)*output=nullptr;return guarded([&]{if(!path||!*path||!output)throw std::runtime_error("Missing inference asset/context output");
        CU(cudaSetDevice(device));auto context=std::make_unique<Context>();context->device=device;load(*context,path);*output=context.release();});
}
extern "C" int mpmat_infer_get_info(void* value,MpmatInferenceInfo* info){return guarded([&]{auto& c=checked(value);
    if(!info||info->struct_size!=sizeof(*info))throw std::runtime_error("Inference info ABI size mismatch");*info=c.info;});}
extern "C" int mpmat_infer_reserve(void* value,uint32_t count){return guarded([&]{auto& c=checked(value);
    if(!count||count>uint32_t(std::numeric_limits<int>::max()))throw std::runtime_error("Invalid inference capacity");
    if(count<=c.capacity)return;if(c.submitted)CU(cudaEventSynchronize(c.completed));
    cudaFree(c.appearance_codes);c.appearance_codes=nullptr;c.capacity=count;c.appearance_count=0;});}
extern "C" int mpmat_infer_set_synthesis(void* value,int mode,int seed,float scale,float power,float rotation,float border){return guarded([&]{
    auto& c=checked(value);auto& m=c.model;
    if(mode<0||mode>2||seed<0||!std::isfinite(scale)||scale<.01f||scale>100||!std::isfinite(power)||power<1||power>32||
        !std::isfinite(rotation)||rotation<0||rotation>25||!std::isfinite(border)||border<.01f||border>.99f)throw std::runtime_error("Invalid synthesis settings");
    if(mode&&!(c.info.capabilities&4u))throw std::runtime_error("Synthesis is unavailable for independently trained mip architectures");
    if(m.synthesis_mode==mode&&m.synthesis_seed==uint32_t(seed)&&m.synthesis_scale==scale&&m.synthesis_power==power&&m.synthesis_rotation==rotation&&m.synthesis_border==border)return;
    if(c.submitted)CU(cudaEventSynchronize(c.completed));
    if(mode==2&&!m.gaussian_texture.array){
        std::vector<float> gaussian,lut;mpmat_synthesis::histogram(c.source_features,size_t(m.r)*m.r,gaussian,lut);
        auto half_lut=half_values(lut);FeatureTexture texture;__half* inverse=nullptr;
        try{ // Preprocess from the exact FP32 source before FP16 texture upload.
            upload_feature_texture(texture,gaussian.data(),m.k,m.r,m.c);upload(inverse,half_lut);
        }catch(...){release_feature_texture(texture);cudaFree(inverse);throw;}
        m.gaussian_texture=texture;m.half_inverse_histogram=inverse;
    }
    m.synthesis_mode=mode;m.synthesis_seed=uint32_t(seed);m.synthesis_scale=scale;m.synthesis_power=power;m.synthesis_rotation=rotation;m.synthesis_border=border;
    c.appearance_count=0;
});}
extern "C" int mpmat_infer_lobes(void* value,const MpmatInferenceRay* rays,uint32_t count,const MpmatInferenceOptions* options,float* output,void* cuda_stream){return guarded([&]{
    enqueue_lobes(value,rays,count,options,output,cuda_stream,false);
});}
extern "C" int mpmat_infer_lobes_fp16(void* value,const MpmatInferenceRay* rays,uint32_t count,const MpmatInferenceOptions* options,uint16_t* output,void* cuda_stream){return guarded([&]{
    enqueue_lobes(value,rays,count,options,output,cuda_stream,true);
});}
extern "C" int mpmat_infer_prepare_appearance(void* value,const MpmatInferenceRay* rays,uint32_t count,const MpmatInferenceOptions* options,void* cuda_stream){return guarded([&]{
    auto& c=checked(value);Model model=sampling_model(c,options);
    if(!rays||!count||count>c.capacity)throw std::runtime_error("Invalid device buffers/count; reserve capacity before inference");
    auto stream=static_cast<cudaStream_t>(cuda_stream);if(c.submitted)CU(cudaStreamWaitEvent(stream,c.completed,0));
    ensure_appearance(c,stream);
    encode_appearance(c,model,LocalCache{nullptr,nullptr,nullptr,reinterpret_cast<const ExternalMaterialRay*>(rays)},count,options,stream);
    CU(cudaGetLastError());CU(cudaEventRecord(c.completed,stream));c.submitted=true;c.appearance_count=count;
});}
extern "C" int mpmat_infer_appearance(void* value,uint32_t count,const float* directions,float* output,void* cuda_stream){return guarded([&]{
    auto& c=checked(value);if(!directions||!output||!count||count!=c.appearance_count)throw std::runtime_error("Prepare matching appearance ray codes before shading");
    auto stream=static_cast<cudaStream_t>(cuda_stream);CU(cudaStreamWaitEvent(stream,c.completed,0));
    if(c.appearance_decoder)coop_backend_launch(c.appearance_decoder,c.model,c.head,LocalCache{},int(count),false,0,output,stream,c.appearance_codes,reinterpret_cast<const float4*>(directions));
    else appearance_reference<<<(count+WARPS-1)/WARPS,THREADS,0,stream>>>(c.model,c.appearance_codes,count,reinterpret_cast<const float4*>(directions),reinterpret_cast<float4*>(output));
    CU(cudaGetLastError());CU(cudaEventRecord(c.completed,stream));c.submitted=true;
});}
extern "C" void mpmat_infer_destroy(void* context){delete static_cast<Context*>(context);}
extern "C" const char* mpmat_infer_error(void){return error_message.c_str();}
