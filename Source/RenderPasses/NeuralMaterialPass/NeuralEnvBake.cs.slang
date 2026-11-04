import Scene.Scene;
import Rendering.Lights.EnvMapSampler;
#include "NeuralLighting.slangh"

// Keep sampler scalar metadata in an explicit reflected parameter block. A
// bare global struct produces an implicit CBV with newer Slang compilers.
ParameterBlock<EnvMapSampler> gEnvSampler;
RWTexture2DArray<float4> gEnvOutput;
cbuffer gEnvBakeCB
{
    uint gBakeSize;
    uint gBakeSamples;
    float gBakeRoughness;
    uint gUseEnvImportance;
    uint gBakeFace;
};

float3 rawEnvironment(float3 localDirection)
{
    float2 uv=gScene.envMap.localToUv(localDirection);
    return gScene.envMap.envMap.SampleLevel(gScene.envMap.envSampler,uv,0).rgb;
}

[numthreads(8,8,1)]
void main(uint3 tid:SV_DispatchThreadID)
{
    tid.z += gBakeFace;
    if(any(tid.xy>=gBakeSize)||tid.z>=6) return;
    float3 axis=neuralCubeDirection(tid.z,2.f*(float2(tid.xy)+.5f)/gBakeSize-1.f);
    float3 t,b; neuralBasis(axis,t,b);
    float3 sum=0; float mass=0;
    uint count=max(1u,gBakeSamples/2);
    float s=gBakeRoughness*gBakeRoughness; s*=s;
    float importanceMean=gUseEnvImportance!=0?gEnvSampler.importanceMap.Load(int3(0,0,gEnvSampler.importanceBaseMip)):0;
    bool importance=gUseEnvImportance!=0&&importanceMean>0;
    for(uint i=0;i<count;++i)
    {
        float u=(float(i)+.5f)/float(count);
        float phi=2.f*kNeuralPi*neuralRadical(i,0x72db6821u);
        float cosH2=(1.f-u)/(1.f+(s-1.f)*u);
        float z=2.f*cosH2-1.f,r=sqrt(max(0.f,1.f-z*z));
        float3 d=normalize(axis*z+t*(r*cos(phi))+b*(r*sin(phi)));
        float q=neuralProposal(z,gBakeRoughness);
        float p=importance?gEnvSampler.evalPdf(gScene.envMap.toWorld(d)):q;
        float weight=q*max(z,0.f)/max(.5f*(q+p),1e-30f);
        sum+=weight*rawEnvironment(d); mass+=weight;
        if(importance)
        {
            EnvMapSample ls;
            if(gEnvSampler.sample(float2(u,neuralRadical(i,0x6a09e667u)),ls))
            {
                d=normalize(gScene.envMap.toLocal(ls.dir)); z=clamp(dot(axis,d),-1.f,1.f);
                q=neuralProposal(z,gBakeRoughness);
                weight=q*max(z,0.f)/max(.5f*(q+ls.pdf),1e-30f);
                sum+=weight*rawEnvironment(d); mass+=weight;
            }
        }
    }
    gEnvOutput[tid]=float4(sum/max(mass,1e-30f),1);
}
