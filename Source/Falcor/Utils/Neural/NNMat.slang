#include "Utils/Math/MathConstants.slangh"

import MLP;
import Utils.Math.MathHelpers;

#ifndef MAX_NN_DIM
#define MAX_NN_DIM 4
#endif

struct NNMat
{
    Texture2DArray<float4> uP;
    Texture2DArray<float4> uFP;
    Texture2DArray<float4> hP;
    Texture2DArray<float4> dP;

    int2 uDims;
    int2 hDims;
    int2 dDims;

    int inputSize;
    MLP mlp;

    SamplerState gSampler;
    SamplerState gPointSampler;
    float3 evalDebug(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        float4 x[MAX_NN_DIM];
        float4 uFine[2];
        int count = 0;
        uv.y = 1 - uv.y;
        uv = uv.yx;
        for (uint i = 0; i < uDims.y; i++)
        {
            // x[count] = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            x[count] = uP.SampleLevel(gPointSampler, float3(uv, i), 0.f);
            count += 1;
        }

        for (uint i = 0; i < hDims.y; i++)
        {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            count += 1;
        }

        for (uint i = 0; i < uDims.y; i++)
        {
            uFine[i] = uFP.SampleLevel(gSampler, float3(uv, i), 0.f);
        
        }

        // return uP.SampleLevel(gSampler, float3(uv, 0), 0.f).xyz;
        // return uFP.SampleLevel(gSampler, float3(uv, 0), 0.f).xyz;
        return mlp.evalBTF(x, uFine).xyz;
    }

    // float3 eval(float3 wi, float3 wo, float2 uv)
    // {
    //     float2 h, d;
    //     dirToRusinkiewicz(wi, wo, h, d);
    //     float3 vh = spherical_to_cartesian_rad(h);
    //     float3 vd = spherical_to_cartesian_rad(d);

    //     if (vh.z > 0.7f)
    //     {
    //         vh.z = 0.7;
    //         vh = normalize(vh);
    //     }
    //     // if (vd.z > 0.9f) vd = normalize(float3(2, 2, 4));

    //     h = hemisphereDirToDiskSquare(vh);
    //     d = hemisphereDirToDiskSquare(vd);
    //     d = float2(0.6, 0.6);

    //     // h.x /= M_PI * 0.5f;
    //     // h.y *= M_1_2PI;
    //     // d.x /= M_PI * 0.5f;
    //     // d.y *= M_1_2PI;

    //     float4 x[MAX_NN_DIM];
    //     int count = 0;
    //     for (int i = 0; i < hDims.y; i++)
    //     {
    //         x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
    //         count += 1;
    //     }
    //     for (int i = 0; i < uDims.y; i++)
    //     {
    //         x[count] = uP.SampleLevel(gSampler, float3(uv, i), 0.f);
    //         count += 1;
    //     }

    //     for (int i = 0; i < dDims.y; i++)
    //     {
    //         x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
    //         count += 1;
    //     }
    //     return mlp.eval(x).xyz;
    // }
}