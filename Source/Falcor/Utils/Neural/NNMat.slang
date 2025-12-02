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

    // Local Shirley square->disk mapping (contained in NNMat to avoid polluting global namespace)
    float2 squareToDiskShirleyLocal(float2 uv, float eps)
    {
        uv = 2.f * uv - 1.f;
        float u = uv.x;
        float v = uv.y;

        float absu = abs(u);
        float absv = abs(v);
        bool mask = absu > absv;

        float r = mask ? u : v;
        float phi;
        if (mask)
        {
            phi = M_PI_4 * (v / (u + eps));
        }
        else
        {
            phi = M_PI_2 - M_PI_4 * (u / (v + eps));
        }

        if (absu + absv == 0.0f)
        {
            phi = 0.0f;
            r = 0.0f;
        }

        float dx = r * cos(phi);
        float dy = r * sin(phi);
        return float2(dx, dy);
    }

    float2 squareToDiskShirleyLocal(float2 uv)
    {
        return squareToDiskShirleyLocal(uv, 1e-12f);
    }

    float3 squareToDiskHemisphereShirleyLocal(float2 uv, float eps)
    {
        float2 d = squareToDiskShirleyLocal(uv, eps);
        float sq = min(1.0f, dot(d, d));
        float z = sqrt(max(0.0f, 1.0f - sq));
        return float3(d, z);
    }

    float3 squareToDiskHemisphereShirleyLocal(float2 uv)
    {
        return squareToDiskHemisphereShirleyLocal(uv, 1e-12f);
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

    void evalTTri(float3 wo, float2 uv, inout float uv_remapped[10])
    {
        wo = normalizeSafe(wo);
        float3 pos = squareToDiskHemisphereShirley(uv);
        pos = normalizeSafe(pos);

        float inputVal[6];
        inputVal[0] = -wo.x;
        inputVal[1] = -wo.y;
        inputVal[2] = -wo.z;
        inputVal[3] = pos.x;
        inputVal[4] = pos.y;
        inputVal[5] = pos.z;
        float2 t = mlp.evalTTriMLP(inputVal);
        float3 pos1 = 0.5 * tanh(-t.x * wo + pos) + 0.5;
        float3 pos2 = 0.5 * tanh(-t.y * wo + pos) + 0.5;

        uv_remapped[0] = pos1.x;
        uv_remapped[1] = pos1.y;
        uv_remapped[2] = pos1.z;
        uv_remapped[3] = pos2.x;
        uv_remapped[4] = pos2.y;
        uv_remapped[5] = pos2.z;

    }
    float3 evalLFTri(float3 wi, float3 wo, inout float uv_remapped[10])
    {
        wo = normalizeSafe(wo);
        wi = normalizeSafe(wi);
        float4 tex;

        int offset = 0;
        int count = 0;
        float inputVal[78];


        inputVal[0] = wo.x;
        inputVal[1] = wo.y;
        inputVal[2] = wo.z;

        inputVal[3] = wi.x;
        inputVal[4] = wi.y;
        inputVal[5] = wi.z;

        offset = 3;
        // offset = 6;


        float3 pos1 = float3(uv_remapped[0], uv_remapped[1], uv_remapped[2]);
        float3 pos2 = float3(uv_remapped[3], uv_remapped[4], uv_remapped[5]);

        uv_remapped[0] = pos1.x;
        uv_remapped[1] = pos1.y;
        uv_remapped[2] = pos1.y;
        uv_remapped[3] = pos1.z;
        uv_remapped[4] = pos1.x;
        uv_remapped[5] = pos1.z;

        uv_remapped[6] = pos2.x;
        uv_remapped[7] = pos2.y;
        uv_remapped[8] = pos2.y;
        uv_remapped[9] = pos2.z;
        // uv_remapped[4] = pos2.x;
        // uv_remapped[5] = pos2.z;


        int layerNum = 3;
        int layer = 0;
        int uvOffset = 0;
        float2 uvSample;
        [unroll]
        for (int i = 0; i < 5; i++) {
            uvSample = float2(uv_remapped[0 + uvOffset], uv_remapped[1 + uvOffset]);
            uvOffset += 2;
            for (int j = 0; j < layerNum; j++) {
                tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f);
                layer++;
                inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
            }
        }

        for (int j = 0; j < layerNum; j++) {
            tex = hP.SampleLevel(gSampler, float3(pos2.x, pos2.z, j), 0.f);
            inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        }


        // int disOffset = 14;
        // return abs(float3(inputVal[disOffset], inputVal[disOffset + 1], inputVal[disOffset+2]));
        return mlp.evalBTFMLP(inputVal, offset);
    }


    void evalT(float3 wo, float2 uv, inout float uv_remapped[10])
    {
        // uv = uv.yx;
        // wo.z = max(wo.z, 0.1);
        wo = normalizeSafe(wo);
        float3 pos = squareToDiskHemisphereShirley(uv);
        pos = normalizeSafe(pos);


        float inputVal[8];
        // inputVal[0] = uv.x;
        // inputVal[1] = uv.y;
        // inputVal[2] = wo.x;
        // inputVal[3] = wo.y;
        // inputVal[4] = wo.z;
        // inputVal[5] = pos.x;
        // inputVal[6] = pos.y;
        // inputVal[7] = pos.z;


        inputVal[0] = wo.x;
        inputVal[1] = wo.y;
        inputVal[2] = wo.z;
        inputVal[3] = pos.x;
        inputVal[4] = pos.y;
        inputVal[5] = pos.z;


        // mlp.evalT(inputVal, uv_remapped);
        mlp.eval8T(inputVal, uv_remapped);
        for (uint i = 0; i < 10; i++)
        {
            uv_remapped[i] = 0.5 * uv_remapped[i] + 0.5;
        }
    }

    float3 evalBTF(float3 wi, float3 wo, inout float uv_remapped[10])
    // float3 evalBTF(float3 wi, float3 wo, inout float2 uv)
    {
        wo = normalizeSafe(wo);
        wi = normalizeSafe(wi);
        float4 tex;

        int offset = 0;
        int count = 0;
        float inputVal[75];
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

        offset = 3;
        // offset = 6;

        // tex = hP.SampleLevel(gSampler, float3(h, 0), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        // tex = hP.SampleLevel(gSampler, float3(h, 1), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        // tex = dP.SampleLevel(gSampler, float3(d, 0), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
        // tex = dP.SampleLevel(gSampler, float3(d, 1), 0.f);
        // inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;

        int layerNum = 3;
        int layer = 0;
        int uvOffset = 0;
        float2 uvSample;
        [unroll]
        for (int i = 0; i < 5; i++) {
            uvSample = float2(uv_remapped[0 + uvOffset], uv_remapped[1 + uvOffset]);
            uvOffset += 2;
            for (int j = 0; j < layerNum; j++) {
                tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f);
                layer++;
                inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
            }
        }

        // int disOffset = 14;
        // return abs(float3(inputVal[disOffset], inputVal[disOffset + 1], inputVal[disOffset+2]));
        return mlp.evalBTFMLP(inputVal, offset);
    }

    float3 evalFeatureMap(float3 wi, float3 wo, inout float uv_remapped[10], int level)
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
        float inputVal[64];


        inputVal[0] = wo.x;
        inputVal[1] = wo.y;
        inputVal[2] = wo.z;
        inputVal[3] = wi.x;
        inputVal[4] = wi.y;
        inputVal[5] = wi.z;

        offset = 6;
        // offset = 6;


        int layerNum = 3;
        int layer = 0;
        int uvOffset = 0;
        float2 uvSample;
        [unroll]
        for (int i = 0; i < 5; i++) {
            uvSample = float2(uv_remapped[0 + uvOffset], uv_remapped[1 + uvOffset]);
            uvOffset += 2;
            for (int j = 0; j < layerNum; j++) {
                tex = u1P.SampleLevel(gSampler, float3(uvSample.x, uvSample.y, layer), 0.f);
                layer++;
                inputVal[offset] = tex.x; inputVal[offset + 1] = tex.y; inputVal[offset + 2] = tex.z; inputVal[offset + 3] = tex.w; offset += 4;
            }
        }
        int disOffset = 6 + level * layerNum * 4;
        return abs(float3(inputVal[disOffset], inputVal[disOffset + 1], inputVal[disOffset + 2]));


    }



}
