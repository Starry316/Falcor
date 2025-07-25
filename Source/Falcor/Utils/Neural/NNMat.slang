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

    Texture2D<float4> LUTs;

    bool isHisto;
    bool isWi;

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

        float inputVal[24];
        int offset = 0;
        int inNumFirst = 0;
        for (uint i = 0; i < uDims.y; i++)
        {
            x[count] = uP.SampleLevel(gPointSampler, float3(uv, i), 0.f);

            inputVal[offset] = x[count].x;
            inputVal[offset + 1] = x[count].y;
            inputVal[offset + 2] = x[count].z;
            inputVal[offset + 3] = x[count].w;
            offset += 4;
            count += 1;
        }

        for (uint i = 0; i < hDims.y; i++)
        {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);

            inputVal[offset] = x[count].x;
            inputVal[offset + 1] = x[count].y;
            inputVal[offset + 2] = x[count].z;
            inputVal[offset + 3] = x[count].w;
            offset += 4;
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);

            inputVal[offset] = x[count].x;
            inputVal[offset + 1] = x[count].y;
            inputVal[offset + 2] = x[count].z;
            inputVal[offset + 3] = x[count].w;
            offset += 4;
            count += 1;
        }

        inNumFirst = offset;

        for (uint i = 0; i < uDims.y; i++)
        {
            uFine[i] = uFP.SampleLevel(gSampler, float3(uv, i), 0.f);
        }


        if (isWi) {
            inputVal[16] = wi.z;
            inNumFirst += 1;
        }
        // float3 ret = mlp.evalBTF(x, uFine).xyz;
        float3 ret = mlp.evalBTF(inputVal, uFine, inNumFirst).xyz;
        if (isHisto)
        {
            // float lutIdx = 1 - ((floor(100 * (acos(wi.z) / M_PI * 2) / 10)) / 10.0f);
            float lutIdx = ((floor(100 * (acos(wi.z) / M_PI * 2) / 10)) / 10.0f);
            ret.x = LUTs.SampleLevel(gPointSampler, float2(ret.x, lutIdx), 0.f).x;
            ret.y = LUTs.SampleLevel(gPointSampler, float2(ret.y, lutIdx), 0.f).y;
            ret.z = LUTs.SampleLevel(gPointSampler, float2(ret.z, lutIdx), 0.f).z;
        }

        return ret;
    }
}
