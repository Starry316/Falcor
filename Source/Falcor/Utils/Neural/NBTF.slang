#include "Utils/Math/MathConstants.slangh"

import MLP;
import Utils.Math.MathHelpers;
import Utils.Texture.HexTiling;
import Utils.Texture.HistogramBlending;
import Utils.Math.FormatConversion;
#ifndef MAX_NN_DIM
#define MAX_NN_DIM 8
#endif

struct NBTF
{
    Texture2DArray<float4> uP;
    Texture2DArray<float4> hP;
    Texture2DArray<float4> dP;

    HistogramBlendingFeatureData histoFeatureData;

    int2 uDims;
    int2 hDims;
    int2 dDims;

    int inputSize;
    MLP mlp;

    SamplerState gSampler;



    // float3 evalDebug(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy, bool fp16 = false, bool synthesize = false)
    // {
    //     float2 h, d;
    //     dirToRusinkiewicz(wi, wo, h, d);
    //     float3 vh = spherical_to_cartesian_rad(h);
    //     float3 vd = spherical_to_cartesian_rad(d);

    //     h = hemisphereDirToDiskSquare(vh);
    //     d = hemisphereDirToDiskSquare(vd);

    //     float4 x[MAX_NN_DIM];
    //     int count = 0;
    //     for (uint i = 0; i < hDims.y; i++)
    //     {
    //         x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
    //         count += 1;
    //     }
    //     for (uint i = 0; i < uDims.y; i++)
    //     {
    //         if (synthesize)
    //             // return histoFeatureData.acf.SampleLevel(gSampler, uv, 0);
    //             x[count] = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
    //         // x[count] = HexTiling(uP, gSampler, ddx, ddy, uv, i);
    //         else
    //         {
    //             x[count] = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
    //         }

    //         count += 1;
    //     }

    //     for (uint i = 0; i < dDims.y; i++)
    //     {
    //         x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
    //         count += 1;
    //     }

    //     if (fp16) {
    //         for (int i = 0; i < MAX_NN_DIM; i++)
    //         {
    //             // x[i] = unpackUnorm4x8(packUnorm4x8(x[i]));
    //             x[i].xy = unpackSnorm2x16(packSnorm2x16(x[i].xy));
    //             x[i].zw = unpackSnorm2x16(packSnorm2x16(x[i].zw));
    //         }
    //         return mlp.evalfp16(x).xyz;
    //     }


    //     return mlp.eval(x).xyz;
    // }



    float3 evalDebug(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        float4 x[MAX_NN_DIM];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            if (synthesize)
                // return histoFeatureData.acf.SampleLevel(gSampler, uv, 0);
                x[count] = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            // x[count] = HexTiling(uP, gSampler, ddx, ddy, uv, i);
            else
            {
                x[count] = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            }

            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            count += 1;
        }
        return mlp.eval(x).xyz;
    }

    float3 eval(float3 wi, float3 wo, float2 uv)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        if (vh.z > 0.7f)
        {
            vh.z = 0.7;
            vh = normalize(vh);
        }
        // if (vd.z > 0.9f) vd = normalize(float3(2, 2, 4));

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);
        d = float2(0.6, 0.6);

        // h.x /= M_PI * 0.5f;
        // h.y *= M_1_2PI;
        // d.x /= M_PI * 0.5f;
        // d.y *= M_1_2PI;

        float4 x[MAX_NN_DIM];
        int count = 0;
        for (int i = 0; i < hDims.y; i++)
        {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }
        for (int i = 0; i < uDims.y; i++)
        {
            x[count] = uP.SampleLevel(gSampler, float3(uv, i), 0.f);
            count += 1;
        }

        for (int i = 0; i < dDims.y; i++)
        {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            count += 1;
        }
        return mlp.eval(x).xyz;
    }

    void bindInput(
        RWBuffer<float> inputBuffer,
        uint offset,
        float3 wi,
        float3 wo,
        float2 uv,

        float2 ddx,
        float2 ddy,
        bool synthesize = false
    )
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            // inputBuffer[count + offset] = packFloat32ToFloat16(val.x, val.y);
            // inputBuffer[count + offset + 1] = packFloat32ToFloat16(val.x, val.y);
            // count += 2;
            inputBuffer[count + offset] = val.x;
            inputBuffer[count + offset + 1] = val.y;
            inputBuffer[count + offset + 2] = val.z;
            inputBuffer[count + offset + 3] = val.w;
            count += 4;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            // inputBuffer[count + offset] = packFloat32ToFloat16(val.x, val.y);
            // inputBuffer[count + offset + 1] = packFloat32ToFloat16(val.x, val.y);
            // count += 2;
            inputBuffer[count + offset] = val.x;
            inputBuffer[count + offset + 1] = val.y;
            inputBuffer[count + offset + 2] = val.z;
            inputBuffer[count + offset + 3] = val.w;
            count += 4;
        }
        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            // inputBuffer[count + offset] = packFloat32ToFloat16(val.x, val.y);
            // inputBuffer[count + offset + 1] = packFloat32ToFloat16(val.x, val.y);
            // count += 2;
            inputBuffer[count + offset] = val.x;
            inputBuffer[count + offset + 1] = val.y;
            inputBuffer[count + offset + 2] = val.z;
            inputBuffer[count + offset + 3] = val.w;
            count += 4;
        }
    }

    void bindInputFP16(
        RWBuffer<uint> inputBuffer,
        uint offset,
        float3 wi,
        float3 wo,
        float2 uv,
        float2 ddx,
        float2 ddy,
        bool synthesize = false
    )
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            // float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            float4 val = float4(0.2, 0.1, 0.3, 0.4);
            inputBuffer[count + offset] = packSnorm2x16(val.xy);
            inputBuffer[count + offset + 1] = packSnorm2x16(val.zw);
            count += 2;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            inputBuffer[count + offset] = packSnorm2x16(val.xy);
            inputBuffer[count + offset + 1] = packSnorm2x16(val.zw);
            count += 2;
        }
        for (uint i = 0; i < dDims.y; i++)
        {
            // float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            float4 val = float4(0.2, 0.1, 0.3, 0.4);
            inputBuffer[count + offset] = packSnorm2x16(val.xy);
            inputBuffer[count + offset + 1] = packSnorm2x16(val.zw);
            count += 2;
        }
    }
}
