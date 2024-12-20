#include "Utils/Math/MathConstants.slangh"

import MLP;
import Utils.Math.MathHelpers;
import Utils.Texture.HexTiling;
import Utils.Texture.HistogramBlending;
import Utils.Math.FormatConversion;
#ifndef MAX_NN_DIM
#define MAX_NN_DIM 8
#endif
static float scaleIn1 = 0.0035758037120103836;

inline int clampInt8(int a)
{
    return min(127, max(-127, a));
}
inline int quantizeInt8x4_safe(float4 v, const float scale)
{
    return (clampInt8(int((v.x / scale))) & 0x000000ff) | (clampInt8(int(v.y / scale)) << 8) & 0x0000ff00 |
           (clampInt8(int(v.z / scale)) << 16) & 0x00ff0000 | (clampInt8(int(v.w / scale)) << 24) & 0xff000000;
}

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

    float4 evalDebug_realDebug(float3 wi, float3 wo, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        float x[24];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        return mlp.eval(x);
    }

    float4 evalDebug_realDebug(float2 h, float2 d, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {

        float x[24];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            x[count * 4] = val.x;
            x[count * 4 + 1] = val.y;
            x[count * 4 + 2] = val.z;
            x[count * 4 + 3] = val.w;
            count += 1;
        }
        return mlp.eval(x);
    }
    float3 evalDebug(float2 h, float2 d, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {

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
        return mlp.eval_relu(x).xyz;
    }
    float3 evalDebug_relu(float2 h, float2 d, float2 uv, float2 ddx, float2 ddy, bool synthesize = false)
    {
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
        return mlp.eval_relu(x).xyz;
    }


    void bindInput(
        RWBuffer<float> inputBuffer,
        uint offset,
        float2 h,
        float2 d,
        float2 uv,

        float2 ddx,
        float2 ddy,
        bool synthesize = false
    )
    {

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
                // val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
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
                // val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
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
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            // float4 val = float4(0.2, 0.1, 0.3, 0.4);
            inputBuffer[count + offset] = packSnorm2x16(val.xy);
            inputBuffer[count + offset + 1] = packSnorm2x16(val.zw);
            count += 2;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                // val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            inputBuffer[count + offset] = packSnorm2x16(val.xy);
            inputBuffer[count + offset + 1] = packSnorm2x16(val.zw);
            count += 2;
        }
        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            // float4 val = float4(0.2, 0.1, 0.3, 0.4);
            inputBuffer[count + offset] = packSnorm2x16(val.xy);
            inputBuffer[count + offset + 1] = packSnorm2x16(val.zw);
            count += 2;
        }
    }

    void bindInputInt8(
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
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            inputBuffer[count + offset] = quantizeInt8x4_safe(val, scaleIn1);
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                // val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            inputBuffer[count + offset] = quantizeInt8x4_safe(val, scaleIn1);
            count += 1;
        }
        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            inputBuffer[count + offset] = quantizeInt8x4_safe(val, scaleIn1);
            count += 1;
        }
    }

    void bindInputInt8(
        RWBuffer<uint> inputBuffer,
        uint offset,
        float2 h,
        float2 d,
        float2 uv,
        float2 ddx,
        float2 ddy,
        bool synthesize = false
    )
    {

        int count = 0;
        for (uint i = 0; i < hDims.y; i++)
        {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            inputBuffer[count + offset] = quantizeInt8x4_safe(val, scaleIn1);
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++)
        {
            float4 val;
            if (synthesize)
                // val = HexTiling(uP, gSampler, ddx, ddy, uv, i);
                val = HistorgramBlending(histoFeatureData, gSampler, uv, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            inputBuffer[count + offset] = quantizeInt8x4_safe(val, scaleIn1);
            count += 1;
        }
        for (uint i = 0; i < dDims.y; i++)
        {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            inputBuffer[count + offset] = quantizeInt8x4_safe(val, scaleIn1);
            count += 1;
        }
    }
}
