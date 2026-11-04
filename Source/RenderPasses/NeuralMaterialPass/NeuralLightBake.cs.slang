#include "NeuralLighting.slangh"

Texture2D<float4> gBakeSource;
SamplerState gBakeSampler;
RWTexture3D<float4> gBakeOutput;
cbuffer gLightBakeCB
{
    uint3 gBakeDimensions;
    uint gBakeCells;
    uint gBakeTextureKind;
    uint gBakeVertexCount;
    float gBakeLogMin;
    float gBakeLogMax;
    uint gBakeSlice;
    float2 gBakeVertices[16];
};

// Normal-tail approximation uses erfc directly, avoiding subtraction of numbers
// close to one in exterior texels. Maximum error is about 1.5e-7.
float neuralErfc(float x)
{
    float t=1.f/(1.f+.5f*abs(x));
    float r=t*exp(-x*x-1.26551223f+t*(1.00002368f+t*(.37409196f+t*(.09678418f+t*(-.18628806f+t*(.27886807f+t*(-1.13520398f+t*(1.48851587f+t*(-.82215223f+t*.17087277f)))))))));
    return x>=0?r:2.f-r;
}

float gaussianMass(float lo,float hi,float center,float sigma)
{
    float a=(lo-center)/(1.4142135623730951f*sigma),b=(hi-center)/(1.4142135623730951f*sigma);
    return max(0.f,a>=0?.5f*(neuralErfc(a)-neuralErfc(b)):.5f*(neuralErfc(-b)-neuralErfc(-a)));
}

float3 bakeEmission(float2 uv)
{
    if(gBakeTextureKind==3) return max(gBakeSource.SampleLevel(gBakeSampler,saturate(uv),0).rgb,0.f);
    if(gBakeTextureKind==1||gBakeTextureKind==2)
        uv=(min(floor(saturate(uv)*12.f),11.f)+.5f)/12.f;
    return neuralEmission(uv,gBakeTextureKind);
}

[numthreads(8,8,1)]
void main(uint3 tid:SV_DispatchThreadID)
{
    tid.z += gBakeSlice;
    if(any(tid>=gBakeDimensions)) return;
    float2 uv=-1.f+3.f*float2(tid.xy)/float2(gBakeDimensions.xy-1);
    float logSigma=lerp(gBakeLogMin,gBakeLogMax,float(tid.z)/float(gBakeDimensions.z-1));
    float sigma=exp2(logSigma);
    float distance2=0; float2 nearest=uv;
    if(!neuralInside(uv,gBakeVertexCount,gBakeVertices))
    {
        distance2=1e30f;
        for(uint edge=0;edge<gBakeVertexCount;++edge)
        {
            float2 a=gBakeVertices[edge],d=gBakeVertices[(edge+1)%gBakeVertexCount]-a;
            float2 p=a+d*saturate(dot(uv-a,d)/max(dot(d,d),1e-20f));
            float candidate=dot(uv-p,uv-p);
            if(candidate<distance2) { distance2=candidate; nearest=p; }
        }
    }
    sigma=sqrt(sigma*sigma+distance2/9.f);
    float3 sum=0; float mass=0;
    uint cells=clamp(gBakeCells,1u,64u);
    // This finite emission quadrature is a load-time prefilter only. The live
    // lighting shader integrates the entire light and samples this volume once.
    for(uint y=0;y<cells;++y)
    {
        float wy=gaussianMass(float(y)/cells,float(y+1)/cells,uv.y,sigma);
        if(wy<1e-24f) continue;
        for(uint x=0;x<cells;++x)
        {
            float2 sampleUv=(float2(x,y)+.5f)/cells;
            if(!neuralInside(sampleUv,gBakeVertexCount,gBakeVertices)) continue;
            float wx=gaussianMass(float(x)/cells,float(x+1)/cells,uv.x,sigma);
            float weight=wx*wy;
            float3 color=bakeEmission(sampleUv);
            sum+=weight*max(color,0.f); mass+=weight;
        }
    }
    // Sub-texel queries on a polygon boundary can miss all mask-cell centers.
    // Use the limiting boundary color; constant emission still stays constant.
    float3 fallback=bakeEmission(saturate(nearest));
    gBakeOutput[tid]=float4(mass>1e-25f?sum/mass:max(fallback,0.f),1);
}
