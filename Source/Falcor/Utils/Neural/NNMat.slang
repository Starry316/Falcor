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

        // float2 wiRad = cartesian_to_spherical_rad(wi);
        // float2 woRad = cartesian_to_spherical_rad(wo);
        if (isWi) {
            // inputVal[16] = wi.z;
            // inNumFirst += 1;

            // inputVal[16] = wi.z;
            // inNumFirst += 1;
            float2 wiRad = cartesian_to_spherical_rad(wi);
            float2 woRad = cartesian_to_spherical_rad(wo);
            // inputVal[8] = wi.x;
            // inputVal[9] = wi.y;
            // inputVal[10] = wi.z;
            // inputVal[11] = wo.x;
            // inputVal[12] = wo.y;
            // inputVal[13] = wo.z;
            // inNumFirst = 14;

            // inputVal[12] = wi.x;
            // inputVal[13] = wi.y;
            // inputVal[14] = wi.z;
            // inputVal[15] = wo.x;
            // inputVal[16] = wo.y;
            // inputVal[17] = wo.z;
            // inNumFirst = 18;

            inputVal[7] = wi.x;
            inputVal[8] = wi.y;
            inputVal[9] = wi.z;
            inputVal[10] = wo.x;
            inputVal[11] = wo.y;
            inputVal[12] = wo.z;
            inputVal[13] = vh.x;
            inputVal[14] = vh.y;
            inputVal[15] = vh.z;
            // inputVal[16] = vh.z;


            inNumFirst = 16;


            // inputVal[8] = cos(wiRad.x);
            // inputVal[9] = sin(wiRad.x);
            // inputVal[10] = cos(wiRad.y);
            // inputVal[11] = sin(wiRad.y);
            // inputVal[12] = cos(woRad.x);
            // inputVal[13] = sin(woRad.x);
            // inputVal[14] = cos(woRad.y);
            // inputVal[15] = sin(woRad.y);

            // inputVal[16] = cos(2*wiRad.x);
            // inputVal[17] = sin(2*wiRad.x);
            // inputVal[18] = cos(2*wiRad.y);
            // inputVal[19] = sin(2*wiRad.y);
            // inputVal[20] = cos(2*woRad.x);
            // inputVal[21] = sin(2*woRad.x);
            // inputVal[22] = cos(2*woRad.y);
            // inputVal[23] = sin(2*woRad.y);


            // inNumFirst += 8;


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
