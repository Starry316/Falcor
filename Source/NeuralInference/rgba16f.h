#pragma once
// C2/C4 spatial feature textures: native RG16F/RGBA16F hardware filtering.
// Only the filtered result is rounded back to half; C2 is never padded in storage.
struct Half4 {__half v[4];};

#ifndef MPMAT_COOP_DEVICE
static void release_feature_texture(FeatureTexture& t){
    if(t.clamp)cudaDestroyTextureObject(t.clamp);if(t.wrap)cudaDestroyTextureObject(t.wrap);
    if(t.array)cudaFreeArray(t.array);t={};
}

static void release_mip_texture(FeatureMipTexture& t){
    for(auto object:{t.clamp,t.wrap,t.clamp_point,t.wrap_point})if(object)cudaDestroyTextureObject(object);
    if(t.array)cudaFreeMipmappedArray(t.array);t={};
}

static void upload_mip_texture(FeatureMipTexture& texture,const float* source,int planes,int resolution,int channels,int levels,const size_t* offsets){
    FeatureMipTexture replacement;
    try{
        auto desc=cudaCreateChannelDesc(16,16,channels==4?16:0,channels==4?16:0,cudaChannelFormatKindFloat);
        CU(cudaMallocMipmappedArray(&replacement.array,&desc,make_cudaExtent(resolution,resolution,planes),levels,cudaArrayLayered));
        for(int level=0;level<levels;++level){int r=resolution>>level;size_t pixels=size_t(r)*r;
            std::vector<__half> packed(size_t(planes)*pixels*channels);const float* data=source+offsets[level];
            for(int p=0;p<planes;++p)for(size_t i=0;i<pixels;++i)for(int c=0;c<channels;++c)
                packed[(size_t(p)*pixels+i)*channels+c]=__float2half_rn(data[(size_t(p)*channels+c)*pixels+i]);
            cudaArray_t array=nullptr;CU(cudaGetMipmappedArrayLevel(&array,replacement.array,level));
            cudaMemcpy3DParms copy={};copy.srcPtr=make_cudaPitchedPtr(packed.data(),r*channels*sizeof(__half),r,r);
            copy.dstArray=array;copy.extent=make_cudaExtent(r,r,planes);copy.kind=cudaMemcpyHostToDevice;CU(cudaMemcpy3D(&copy));
        }
        cudaResourceDesc resource={};resource.resType=cudaResourceTypeMipmappedArray;resource.res.mipmap.mipmap=replacement.array;
        for(int point=0;point<2;++point)for(int wrap=0;wrap<2;++wrap){
            cudaTextureDesc sampler={};sampler.normalizedCoords=1;sampler.readMode=cudaReadModeElementType;
            sampler.filterMode=point?cudaFilterModePoint:cudaFilterModeLinear;sampler.mipmapFilterMode=cudaFilterModeLinear;
            sampler.addressMode[0]=sampler.addressMode[1]=wrap?cudaAddressModeWrap:cudaAddressModeClamp;
            sampler.minMipmapLevelClamp=0;sampler.maxMipmapLevelClamp=float(levels-1);sampler.maxAnisotropy=1;
            auto* object=point?(wrap?&replacement.wrap_point:&replacement.clamp_point):(wrap?&replacement.wrap:&replacement.clamp);
            CU(cudaCreateTextureObject(object,&resource,&sampler,nullptr));
        }
        release_mip_texture(texture);texture=replacement;
    }catch(...){release_mip_texture(replacement);throw;}
}

static void upload_feature_texture(FeatureTexture& t,const float* source,int planes,int resolution,int channels){
    if(channels!=2&&channels!=4)throw std::runtime_error("Feature texture requires 2 or 4 channels");
    FeatureTexture replacement;
    try{
        size_t pixels=size_t(resolution)*resolution;
        std::vector<__half> packed(size_t(planes)*pixels*channels);
        for(int p=0;p<planes;++p)for(size_t i=0;i<pixels;++i)for(int c=0;c<channels;++c)
            packed[(size_t(p)*pixels+i)*channels+c]=__float2half_rn(source[(size_t(p)*channels+c)*pixels+i]);
        auto desc=cudaCreateChannelDesc(16,16,channels==4?16:0,channels==4?16:0,cudaChannelFormatKindFloat);
        auto extent=make_cudaExtent(resolution,resolution,planes);
        CU(cudaMalloc3DArray(&replacement.array,&desc,extent,cudaArrayLayered));
        cudaMemcpy3DParms copy={};
        copy.srcPtr=make_cudaPitchedPtr(packed.data(),resolution*channels*sizeof(__half),resolution,resolution);
        copy.dstArray=replacement.array;copy.extent=extent;copy.kind=cudaMemcpyHostToDevice;
        CU(cudaMemcpy3D(&copy));
        cudaResourceDesc resource={};resource.resType=cudaResourceTypeArray;resource.res.array.array=replacement.array;
        cudaTextureDesc sampler={};sampler.normalizedCoords=1;sampler.filterMode=cudaFilterModeLinear;sampler.readMode=cudaReadModeElementType;
        sampler.addressMode[0]=sampler.addressMode[1]=cudaAddressModeClamp;
        CU(cudaCreateTextureObject(&replacement.clamp,&resource,&sampler,nullptr));
        sampler.addressMode[0]=sampler.addressMode[1]=cudaAddressModeWrap;
        CU(cudaCreateTextureObject(&replacement.wrap,&resource,&sampler,nullptr));
        release_feature_texture(t);t=replacement;
    }catch(...){release_feature_texture(replacement);throw;}
}

#endif

// One layered texture instruction performs spatial and inter-level filtering.
// Every level contains its independently trained data; no levels are generated.
__device__ __forceinline__ Half4 sample_mip_texture(const Model& m,const Local& ray,int plane,bool periodic,float crop,float lod){
    float u,v;if(!material_uv(m,ray,plane,periodic,crop,u,v))return {};
    bool wrap=periodic&&crop==0;
    cudaTextureObject_t object=m.nearest_fetch?(wrap?m.mip_texture.wrap_point:m.mip_texture.clamp_point):
        (wrap?m.mip_texture.wrap:m.mip_texture.clamp);
    if(m.c==2){float2 value=tex2DLayeredLod<float2>(object,u,v,plane,lod);
        return {{__half(value.x),__half(value.y),__half(0.f),__half(0.f)}};}
    float4 value=tex2DLayeredLod<float4>(object,u,v,plane,lod);
    return {{__half(value.x),__half(value.y),__half(value.z),__half(value.w)}};
}

__device__ Half4 rgba16f_lookup(Model m,int p,float x,float y,bool circular,bool gaussian=false){
    FeatureTexture texture=gaussian?m.gaussian_texture:m.feature_texture;
    // sample_uv includes a virtual circular border. Removing its one-texel
    // offset and adding CUDA's half-texel center preserves that convention.
    if(m.nearest_fetch){x=floorf(x+.5f);y=floorf(y+.5f);}
    float offset=circular?-.5f:.5f;
    if(!gaussian&&m.mip_texture.clamp){
        cudaTextureObject_t object=m.nearest_fetch?(circular?m.mip_texture.wrap_point:m.mip_texture.clamp_point):
            (circular?m.mip_texture.wrap:m.mip_texture.clamp);
        if(m.c==2){float2 f=tex2DLayeredLod<float2>(object,(x+offset)/m.r,(y+offset)/m.r,p,0.f);
            return {{__half(f.x),__half(f.y),__half(0.f),__half(0.f)}};}
        float4 f=tex2DLayeredLod<float4>(object,(x+offset)/m.r,(y+offset)/m.r,p,0.f);
        return {{__half(f.x),__half(f.y),__half(f.z),__half(f.w)}};
    }
    if(m.c==2){
        float2 f=tex2DLayered<float2>(circular?texture.wrap:texture.clamp,(x+offset)/m.r,(y+offset)/m.r,p);
        return {{__float2half_rn(f.x),__float2half_rn(f.y),__half(0.f),__half(0.f)}};
    }
    float4 f=tex2DLayered<float4>(circular?texture.wrap:texture.clamp,(x+offset)/m.r,(y+offset)/m.r,p);
    return {{__float2half_rn(f.x),__float2half_rn(f.y),__float2half_rn(f.z),__float2half_rn(f.w)}};
}

__device__ Half4 synthesis_rgba16f(Model m,const Local& ray,int plane,float crop){
    Half4 result={};
    if(fabsf(ray.d.z)<=m.epsilon)return result;
    float t=(m.heights[plane]-ray.o.z)/ray.d.z;
    float u=(ray.o.x+t*ray.d.x-m.bounds[0])/(m.bounds[3]-m.bounds[0]);
    float v=(ray.o.y+t*ray.d.y-m.bounds[1])/(m.bounds[4]-m.bounds[1]);
    if(t<0||!isfinite(u)||!isfinite(v)||fabsf(u)>=1e7f||fabsf(v)>=1e7f)return result;
    float sx=(u-.57735027f*v)*m.synthesis_scale,sy=1.15470054f*v*m.synthesis_scale;
    int bx=int(floorf(sx)),by=int(floorf(sy));float fx=sx-bx,fy=sy-by;bool upper=fx+fy>1;
    int side=upper?1:0;
    int vx[3]={bx+side,bx+side,bx+1-side},vy[3]={by+side,by+1-side,by+side};
    float w[3]={upper?fx+fy-1:1-fx-fy,upper?1-fy:fy,upper?1-fx:fx},sum=0,square=0;
    for(int i=0;i<3;++i){w[i]=powf(fmaxf(w[i],0),m.synthesis_power);sum+=w[i];}
    for(int i=0;i<3;++i)w[i]/=fmaxf(sum,1e-30f);
    if(m.synthesis_border!=.5f){
        float exponent=logf(1-m.synthesis_border)/logf(.5f);sum=0;
        for(int i=0;i<3;++i){w[i]=w[i]<.5f?.5f*powf(fmaxf(2*w[i],0.f),exponent):1-.5f*powf(fmaxf(2-2*w[i],0.f),exponent);sum+=w[i];}
        for(int i=0;i<3;++i)w[i]/=fmaxf(sum,1e-30f);
    }
    for(int i=0;i<3;++i){
        square+=w[i]*w[i];
        uint32_t h=synthesis_hash(uint32_t(vx[i])*0x8da6b343u^uint32_t(vy[i])*0xd8163841u^m.synthesis_seed);
        uint32_t h2=synthesis_hash(h^0x9e3779b9u);
        float x=u+float(h&0xffffffu)/16777216.f,y=v+float(h2&0xffffffu)/16777216.f;
        if(m.synthesis_rotation!=0){
            uint32_t angle_hash=synthesis_hash(h^0xa511e9b3u);
            float angle=(float(angle_hash&0xffffffu)/16777216.f*2*CUDART_PI_F-CUDART_PI_F)*m.synthesis_rotation;
            float cx=(vx[i]+.5f*vy[i])/m.synthesis_scale,cy=vy[i]/(1.15470054f*m.synthesis_scale);
            float dx=u-cx,dy=v-cy,cs=cosf(angle),sn=sinf(angle);
            x=cs*dx-sn*dy+cx+float(h&0xffffffu)/16777216.f;
            y=sn*dx+cs*dy+cy+float(h2&0xffffffu)/16777216.f;
        }
        x=crop+(1-2*crop)*(x-floorf(x));y=crop+(1-2*crop)*(y-floorf(y));
        float gx=2*x-1,gy=2*y-1;int r=m.r;bool circular=crop==0;
        if(circular){gx=gx*m.r/(m.r+2);gy=gy*m.r/(m.r+2);r+=2;}
        x=fminf(fmaxf(((gx+1)*r-1)*.5f,0),float(r-1));y=fminf(fmaxf(((gy+1)*r-1)*.5f,0),float(r-1));
        Half4 value=rgba16f_lookup(m,plane,x,y,circular,m.synthesis_mode==2);
        for(int c=0;c<m.c;++c)result.v[c]+=__half(w[i])*value.v[c];
    }
    if(m.synthesis_mode==2){
        for(int c=0;c<m.c;++c){
            float position=float(__hmin(__hmax((result.v[c]/hsqrt(__half(square))+__half(5.f))*__half(409.5f),__half(0.f)),__half(4095.f)));
            int lo=int(position),hi=min(lo+1,4095);float fraction=position-lo;
            const __half* lut=m.half_inverse_histogram+(plane*m.c+c)*4096;
            result.v[c]=lut[lo]*__half(1-fraction)+lut[hi]*__half(fraction);
        }
    }
    return result;
}

__device__ Half4 sample_rgba16f(Model m,const Local& ray,int plane,bool periodic,float crop,bool& valid){
    if(m.mip_levels>1){
        float u,v;valid=material_uv(m,ray,plane,periodic,crop,u,v);
        float lod=fminf(fmaxf(m.feature_lod<0?ray.lod:m.feature_lod,0.f),float(m.mip_levels-1));
        return valid?sample_mip_texture(m,ray,plane,periodic,crop,lod):Half4{};
    }
    // Legacy one-level mip assets retain the original UV arithmetic exactly.
    // rgba16f_lookup selects their mip texture without changing coordinates.
    float x,y;valid=sample_uv(m,ray,plane,periodic,crop,x,y);
    if(m.synthesis_mode&&periodic)return synthesis_rgba16f(m,ray,plane,crop);
    return valid?rgba16f_lookup(m,plane,x,y,periodic&&crop==0):Half4{};
}

template<class T> __device__ void load_spatial_base(Model m,const Local& ray,bool periodic,float crop,T* a,float* samples=nullptr){
    int lane=threadIdx.x%32;
    if constexpr(std::is_same_v<T,__half>){
        if(m.c==2||m.c==4){
            for(int p=lane;p<m.k;p+=32){bool valid;Half4 value=sample_rgba16f(m,ray,p,periodic,crop,valid);
                for(int c=0;c<m.c;++c){a[p*m.c+c]=value.v[c];if(samples)samples[p*m.c+c]=float(value.v[c]);}
                a[m.k*m.c+p]=__half(valid?1.f:0.f);
            }
            return;
        }
    }
    // FP32 remains the training/reference oracle; other channel counts retain
    // the compatible manual sampler.
    for(int j=lane;j<m.k*m.c;j+=32){
        int p=j/m.c,c=j%m.c;float x,y;T value=T(0.f);
        if(m.synthesis_mode&&periodic)value=synthesis_feature<T>(m,ray,p,c,crop);
        else if(sample_uv(m,ray,p,periodic,crop,x,y)){
            if(m.nearest_fetch){a[j]=texel<T>(m,p,c,int(x),int(y),periodic&&crop==0);if(samples)samples[j]=float(a[j]);continue;}
            int ix=int(floorf(x)),iy=int(floorf(y));T fx=T(x-ix),fy=T(y-iy),one=T(1.f);bool circular=periodic&&crop==0;
            value=texel<T>(m,p,c,ix,iy,circular)*((one-fx)*(one-fy));
            value+=texel<T>(m,p,c,ix+1,iy,circular)*(fx*(one-fy));
            value+=texel<T>(m,p,c,ix,iy+1,circular)*((one-fx)*fy);
            value+=texel<T>(m,p,c,ix+1,iy+1,circular)*(fx*fy);
        }
        a[j]=value;if(samples)samples[j]=float(value);
    }
    for(int p=lane;p<m.k;p+=32){float x,y;a[m.k*m.c+p]=sample_uv(m,ray,p,periodic,crop,x,y)?1.f:0.f;}
}

template<class T> __device__ void load_spatial_inputs(Model m,const Local& ray,bool periodic,float crop,T* a,float* samples=nullptr,T* scratch=nullptr){
    if(m.kind!=8){load_spatial_base(m,ray,periodic,crop,a,samples);return;}
    float lod=fminf(fmaxf(m.feature_lod<0?ray.lod:m.feature_lod,0.f),float(m.mip_levels-1));
    if constexpr(std::is_same_v<T,__half>){if(m.c==2||m.c==4){
        for(int p=threadIdx.x%32;p<m.k;p+=32){Half4 value=sample_mip_texture(m,ray,p,periodic,crop,lod);
            for(int c=0;c<m.c;++c){a[p*m.c+c]=value.v[c];if(samples)samples[p*m.c+c]=float(value.v[c]);}}
        return;
    }}
    int lo=int(floorf(lod)),hi=int(ceilf(lod));Model level=m;
    level.r=m.r>>lo;level.features=m.features+m.mip_offsets[lo];
    if(m.half_features)level.half_features=m.half_features+m.mip_offsets[lo];
    load_spatial_base(level,ray,periodic,crop,a,nullptr);__syncwarp();
    if(hi!=lo&&scratch){
        level.r=m.r>>hi;level.features=m.features+m.mip_offsets[hi];
        if(m.half_features)level.half_features=m.half_features+m.mip_offsets[hi];
        load_spatial_base(level,ray,periodic,crop,scratch,nullptr);__syncwarp();
        T fraction=T(lod-lo);
        for(int j=threadIdx.x%32;j<m.k*m.c;j+=32)a[j]=a[j]+fraction*(scratch[j]-a[j]);
        __syncwarp();
    }
    if(samples)for(int j=threadIdx.x%32;j<m.k*m.c;j+=32)samples[j]=float(a[j]);
}
