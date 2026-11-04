#pragma once
// Shared material-space sampler for CUDA and cooperative-vector shaders.
__device__ __forceinline__ bool material_uv(const Model& m,const Local& ray,int plane,bool periodic,float crop,float& ux,float& uy){
    bool parallel=fabsf(ray.d.z)<=m.epsilon;
    float distance=(m.heights[plane]-ray.o.z)/(parallel?1.f:ray.d.z);
    ux=(ray.o.x+distance*ray.d.x-m.bounds[0])/(m.bounds[3]-m.bounds[0]);
    uy=(ray.o.y+distance*ray.d.y-m.bounds[1])/(m.bounds[4]-m.bounds[1]);
    bool valid=!parallel&&distance>=0;
    if(periodic){ux=ux-floorf(ux);uy=uy-floorf(uy);ux=crop+(1-2*crop)*ux;uy=crop+(1-2*crop)*uy;}
    else valid=valid&&ux>=0&&ux<=1&&uy>=0&&uy<=1;
    valid=valid&&isfinite(ux)&&isfinite(uy);
    if(!valid){ux=uy=0;return false;}return true;
}

__device__ bool sample_uv(Model m,const Local& ray,int plane,bool periodic,float crop,float& x,float& y){
    float ux,uy;if(!material_uv(m,ray,plane,periodic,crop,ux,uy)){x=y=0;return false;}
    // Reproduce grid_sample's align_corners=False unnormalization, including
    // the circular one-texel pad used by the reference when crop is zero.
    float gx=2*ux-1,gy=2*uy-1;int r=m.r;
    if(periodic&&crop==0){gx=gx*m.r/(m.r+2);gy=gy*m.r/(m.r+2);r+=2;}
    x=fminf(fmaxf(((gx+1)*r-1)*.5f,0),float(r-1));
    y=fminf(fmaxf(((gy+1)*r-1)*.5f,0),float(r-1));
    if(m.nearest_fetch){x=floorf(x+.5f);y=floorf(y+.5f);}
    return true;
}

template<class T> __device__ T feature_at(Model m,size_t index);
template<> __device__ float feature_at<float>(Model m,size_t index){return m.features[index];}
template<> __device__ __half feature_at<__half>(Model m,size_t index){return m.half_features[index];}
template<class T=float> __device__ T texel(Model m,int plane,int channel,int x,int y,bool circular){
    if(circular){x=(x-1+m.r)%m.r;y=(y-1+m.r)%m.r;}
    else{x=min(max(x,0),m.r-1);y=min(max(y,0),m.r-1);}
    return feature_at<T>(m,((size_t(plane)*m.c+channel)*m.r+y)*m.r+x);
}

__device__ uint32_t synthesis_hash(uint32_t x){
    x=(x^(x>>16))*0x7feb352du;x=(x^(x>>15))*0x846ca68bu;return x^(x>>16);
}
template<class T=float> __device__ T synthesis_feature(Model m,const Local& ray,int plane,int channel,float crop){
    if(fabsf(ray.d.z)<=m.epsilon)return 0;
    float t=(m.heights[plane]-ray.o.z)/ray.d.z;
    float u=(ray.o.x+t*ray.d.x-m.bounds[0])/(m.bounds[3]-m.bounds[0]);
    float v=(ray.o.y+t*ray.d.y-m.bounds[1])/(m.bounds[4]-m.bounds[1]);
    if(t<0||!isfinite(u)||!isfinite(v)||fabsf(u)>=1e7f||fabsf(v)>=1e7f)return 0;
    float sx=(u-.57735027f*v)*m.synthesis_scale,sy=1.15470054f*v*m.synthesis_scale;
    int bx=int(floorf(sx)),by=int(floorf(sy));float fx=sx-bx,fy=sy-by;bool upper=fx+fy>1;
    int side=upper?1:0;
    int vx[3]={bx+side,bx+side,bx+1-side},vy[3]={by+side,by+1-side,by+side};
    float w[3]={upper?fx+fy-1:1-fx-fy,upper?1-fy:fy,upper?1-fx:fx},sum=0,square=0;T result=T(0.f);
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
        if(m.nearest_fetch){x=floorf(x+.5f);y=floorf(y+.5f);}
        int ix=int(floorf(x)),iy=int(floorf(y));T tx=T(x-ix),ty=T(y-iy),value=T(0.f),one=T(1.f);
        for(int j=0;j<(m.nearest_fetch?1:4);++j){int xx=ix+(j&1),yy=iy+(j>>1);
            if(circular){xx=(xx-1+m.r)%m.r;yy=(yy-1+m.r)%m.r;}
            else{xx=min(max(xx,0),m.r-1);yy=min(max(yy,0),m.r-1);}
            T sample;
            size_t index=(size_t(plane)*m.c+channel)*m.r*m.r+size_t(yy)*m.r+xx;
            if constexpr(std::is_same_v<T,float>)sample=(m.synthesis_mode==2?m.gaussian:m.features)[index];
            else sample=(m.synthesis_mode==2?m.half_gaussian:m.half_features)[index];
            value+=sample*((j&1)?tx:one-tx)*((j>>1)?ty:one-ty);}
        result+=T(w[i])*value;
    }
    if(m.synthesis_mode==2){
        // Lattice/lookup coordinates retain spatial precision; feature blending is T.
        float position;
        if constexpr(std::is_same_v<T,float>)position=fminf(fmaxf((result/sqrtf(square)+5)*4095.f/10.f,0),4095.f);
        else position=float(__hmin(__hmax((result/hsqrt(T(square))+T(5.f))*T(409.5f),T(0.f)),T(4095.f)));
        int lo=int(position),hi=min(lo+1,4095);float fraction=position-lo;
        if constexpr(std::is_same_v<T,float>){const float* lut=m.inverse_histogram+(plane*m.c+channel)*4096;result=lut[lo]*(1-fraction)+lut[hi]*fraction;}
        else{const __half* h=m.half_inverse_histogram+(plane*m.c+channel)*4096;result=h[lo]*T(1-fraction)+h[hi]*T(fraction);}
    }
    return result;
}

