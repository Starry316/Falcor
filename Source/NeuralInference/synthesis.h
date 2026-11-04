#pragma once
// Rendering-only histogram preprocessing. No Python/PyTorch runtime dependency.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace mpmat_synthesis {
constexpr int lut_size=4096;
inline float normal_quantile(double p){
    double lo=-9,hi=9;
    for(int j=0;j<40;++j){double mid=(lo+hi)*.5;
        if(.5*(1+std::erf(mid/std::sqrt(2.)))<p)lo=mid;else hi=mid;}
    return float((lo+hi)*.5);
}
inline void histogram(const std::vector<float>& source,size_t pixels,
                      std::vector<float>& gaussian,std::vector<float>& inverse){
    size_t channels=source.size()/pixels;
    gaussian.resize(source.size());inverse.resize(channels*lut_size);
    std::vector<float> ranks(pixels);
    for(size_t i=0;i<pixels;++i)ranks[i]=normal_quantile((double(i)+.5)/pixels);
    std::vector<size_t> order(pixels);
    for(size_t c=0;c<channels;++c){
        const float* values=source.data()+c*pixels;
        std::iota(order.begin(),order.end(),size_t(0));
        std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return values[a]<values[b];});
        for(size_t i=0;i<pixels;){size_t stop=i+1;
            while(stop<pixels&&values[order[stop]]==values[order[i]])++stop;
            float z=stop==i+1?ranks[i]:normal_quantile((double(i)+double(stop))*.5/pixels);
            for(size_t j=i;j<stop;++j)gaussian[c*pixels+order[j]]=z;i=stop;}
        for(int i=0;i<lut_size;++i){
            // Match the FP32 inverse-CDF LUT used by the Python oracle.
            float z=-5.f+10.f*i/(lut_size-1),p=.5f*(1+std::erf(z/std::sqrt(2.f)));
            float index=std::clamp(p*float(pixels)-.5f,0.f,float(pixels-1));
            size_t lo=size_t(index),hi=std::min(lo+1,pixels-1);float t=index-float(lo);
            inverse[c*lut_size+i]=values[order[lo]]*(1-t)+values[order[hi]]*t;
        }
    }
}
}
