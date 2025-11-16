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

    Texture2DArray<float4> u1P;
    // Texture2DArray<float4> u2P;
    // Texture2DArray<float4> u3P;
    // Texture2DArray<float4> u4P;
    // Texture2DArray<float4> u5P;


    int2 uDims;
    int2 hDims;
    int2 dDims;

    int inputSize;
    MLP mlp;

    SamplerState gSampler;
    SamplerState gPointSampler;

    float3 HDRRevert(float3 I) {
        return (exp(abs(I) * log(17.0)) - 1) / 16;
    }

    float3 evalIBL(float3 wo, float2 uv, float3 rotAngles)
    {
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

        inNumFirst = offset;

        for (uint i = 0; i < uDims.y; i++)
        {
            uFine[i] = uFP.SampleLevel(gSampler, float3(uv, i), 0.f);
        }

        inputVal[7] = wo.x;
        inputVal[8] = wo.y;
        inputVal[9] = wo.z;

        inputVal[10] = cos(rotAngles.y);
        inputVal[11] = sin(rotAngles.y);
        inputVal[12] = cos(2 * rotAngles.y);
        inputVal[13] = sin(2 * rotAngles.y);

        inNumFirst = 14;

        return HDRRevert(mlp.evalBTF(inputVal, uFine, inNumFirst).xyz);
    }

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


            float2 wiRad = cartesian_to_spherical_rad(wi);
            float2 woRad = cartesian_to_spherical_rad(wo);

            // inputVal[8] = wi.x;
            // inputVal[9] = wi.y;
            // inputVal[10] = wi.z;
            // inputVal[11] = wo.x;
            // inputVal[12] = wo.y;
            // inputVal[13] = wo.z;
            // inputVal[14] = vh.x;
            // inputVal[15] = vh.y;
            // inputVal[16] = vh.z;
            // inNumFirst = 17;
            inputVal[7] = wi.x;
            inputVal[8] = wi.y;
            inputVal[9] = wi.z;
            inputVal[10] = wo.x;
            inputVal[11] = wo.y;
            inputVal[12] = wo.z;
            inputVal[13] = vh.x;
            inputVal[14] = vh.y;
            inputVal[15] = vh.z;
            inNumFirst = 16;
        float3 ret = mlp.evalBTF(inputVal, uFine, inNumFirst).xyz;


        return ret;
    }

    void evalT(float3 wo, float2 uv, inout float uv_remapped[10])
    {
        // wo.z = max(wo.z, 0.1);
        wo = normalizeSafe(wo);

        // uv = 1 - uv;

        // uv = uv.yx;
        float inputVal[5];
        inputVal[0] = uv.x;
        inputVal[1] = uv.y;
        inputVal[2] = wo.x;
        inputVal[3] = wo.y;
        inputVal[4] = wo.z;

        mlp.evalT(inputVal, uv_remapped);
        // mlp.eval8T(inputVal, uv_remapped);
        for (uint i = 0; i < 10; i++)
        {
            uv_remapped[i] = 0.5 * uv_remapped[i] + 0.5;
        }
    }

    float3 evalBTF(float3 wi, float3 wo, inout float uv_remapped[10])
    // float3 evalBTF(float3 wi, float3 wo, inout float2 uv)
    {
        // wo.z = max(wo.z, 0.1);
        // wo.z = max(wo.z, 0.9);
        wo = normalizeSafe(wo);

        // wi.z = max(wi.z, 0.9);
        wi = normalizeSafe(wi);
        float4 tex;
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);


        h = hemisphereDirToDiskSquare(vh).yx;
        d = hemisphereDirToDiskSquare(vd).yx;
        int offset = 0;
        int count = 0;
        float inputVal[56];
        // inputVal[0] = vh.x;
        // inputVal[1] = vh.y;
        // inputVal[2] = vh.z;
        // inputVal[3] = vd.x;
        // inputVal[4] = vd.y;
        // inputVal[5] = vd.z;

        inputVal[0] = wo.x;
        inputVal[1] = wo.y;
        inputVal[2] = wo.z;
        inputVal[3] = wi.x;
        inputVal[4] = wi.y;
        inputVal[5] = wi.z;

        // offset = 3;
        offset = 6;

        // tex = hP.SampleLevel(gSampler, float3(h, 0), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        // tex = hP.SampleLevel(gSampler, float3(h, 1), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        // tex = dP.SampleLevel(gSampler, float3(d, 0), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        // tex = dP.SampleLevel(gSampler, float3(d, 1), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;


        // tex = u1P.SampleLevel(gSampler, float3(uv_remapped[0 + 0], uv_remapped[1 + 0], 0), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        // tex = u1P.SampleLevel(gSampler, float3(uv_remapped[0 + 2], uv_remapped[1 + 2], 1), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        // tex = u1P.SampleLevel(gSampler, float3(uv_remapped[0 + 4], uv_remapped[1 + 4], 2), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        // tex = u1P.SampleLevel(gSampler, float3(uv_remapped[0 + 6], uv_remapped[1 + 6], 3), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        // tex = u1P.SampleLevel(gSampler, float3(uv_remapped[0 + 8], uv_remapped[1 + 8], 4), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        int layer = 0;
        float2 uvSample;

        // reuse uvSample, load each uv twice while advancing texture-array layer index
        uvSample = float2(uv_remapped[0 + 0], uv_remapped[1 + 0]);
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        uvSample = float2(uv_remapped[0 + 2], uv_remapped[1 + 2]);
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        uvSample = float2(uv_remapped[0 + 4], uv_remapped[1 + 4]);
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        uvSample = float2(uv_remapped[0 + 6], uv_remapped[1 + 6]);
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        uvSample = float2(uv_remapped[0 + 8], uv_remapped[1 + 8]);
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f); layer++;
        inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;



        // int disOffset = 14;
        //return abs(float3(inputVal[disOffset], inputVal[disOffset + 1], inputVal[disOffset+2]));

        return mlp.evalBTFMLP(inputVal);


    }

}
