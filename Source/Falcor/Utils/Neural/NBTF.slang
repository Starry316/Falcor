#include "Utils/Math/MathConstants.slangh"

import MLP;
import Utils.Math.MathHelpers;
import Utils.Texture.HexTiling;

#ifndef MAX_NN_DIM
#define MAX_NN_DIM 8
#endif
// Function to convert a float32 value to float16
uint float32ToFloat16(float value) {
    uint f32 = asuint(value);
    uint sign = (f32 >> 16) & 0x8000;
    uint mantissa = (f32 & 0x007FFFFF) >> 13;
    uint exponent = ((f32 >> 23) & 0xFF) - 112;

    if (exponent > 0x1F) {
        exponent = 0x1F;
        mantissa = 0;
    } else if (exponent <= 0) {
        if (exponent < -10) {
            mantissa = 0;
        } else {
            mantissa = (mantissa | 0x0800) >> (1 - exponent);
        }
        exponent = 0;
    }

    return sign | (exponent << 10) | mantissa;
}

// Function to pack two float32 values into one float32 value using float16 precision
float packFloat32ToFloat16(float a, float b) {
    uint ha = float32ToFloat16(a);
    uint hb = float32ToFloat16(b);

    uint packed = (ha << 16) | hb;
    return asfloat(packed);
}
struct NBTF
{
    Texture2DArray<float4> uP;
    Texture2DArray<float4> hP;
    Texture2DArray<float4> dP;

    int2 uDims;
    int2 hDims;
    int2 dDims;

    int inputSize;
    MLP mlp;

    SamplerState gSampler;
    float3 evalDebug(float3 wi, float3 wo, float2 uv, float zLimit, float zMinLimit, float2 ddx, float2 ddy, bool synthesize=false)
    {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        if (vh.z > zLimit) {
            float sinTheta = sqrt(1 - vh.z * vh.z);
            float newSinTheta = sqrt(1 - zLimit * zLimit);
            vh.z = zLimit;
            vh.xy *= newSinTheta / sinTheta;
            vh = normalize(vh);
        }

        if (vh.z < zMinLimit) {
            float sinTheta = sqrt(1 - vh.z * vh.z);
            float newSinTheta = sqrt(1 - zMinLimit * zMinLimit);
            vh.z = zMinLimit;
            vh.xy *= newSinTheta / sinTheta;
            vh = normalize(vh);
        }



        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);


        float4 x[MAX_NN_DIM];
        int count = 0;
        for (uint i = 0; i < hDims.y; i++) {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }
        for (uint i = 0; i < uDims.y; i++) {
            if (synthesize)
                x[count] = HexTiling(uP, gSampler, uv.yx, i, ddx, ddy);
            else
                x[count] = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            count += 1;
        }

        for (uint i = 0; i < dDims.y; i++) {
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

        if (vh.z > 0.7f) {
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
        for (int i = 0; i < hDims.y; i++) {
            x[count] = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            count += 1;
        }
        for (int i = 0; i < uDims.y; i++) {
            x[count] = uP.SampleLevel(gSampler, float3(uv, i), 0.f);
            count += 1;
        }

        for (int i = 0; i < dDims.y; i++) {
            x[count] = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            count += 1;
        }
        return mlp.eval(x).xyz;

    }

    void bindInput( RWBuffer<float> inputBuffer, uint offset, float3 wi, float3 wo, float2 uv, float zLimit, float zMinLimit, float2 ddx, float2 ddy, bool synthesize = false ) {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        if (vh.z > zLimit) {
            float sinTheta = sqrt(1 - vh.z * vh.z);
            float newSinTheta = sqrt(1 - zLimit * zLimit);
            vh.z = zLimit;
            vh.xy *= newSinTheta / sinTheta;
            vh = normalize(vh);
        }

        if (vh.z < zMinLimit) {
            float sinTheta = sqrt(1 - vh.z * vh.z);
            float newSinTheta = sqrt(1 - zMinLimit * zMinLimit);
            vh.z = zMinLimit;
            vh.xy *= newSinTheta / sinTheta;
            vh = normalize(vh);
        }

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        int count = 0;
        for (uint i = 0; i < hDims.y; i++) {
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
        for (uint i = 0; i < uDims.y; i++) {
            float4 val;
            if (synthesize)
                val = HexTiling(uP, gSampler, uv.yx, i, ddx, ddy);
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
        for (uint i = 0; i < dDims.y; i++) {
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

    void bindInputFP16( RWBuffer<float> inputBuffer, uint offset, float3 wi, float3 wo, float2 uv, float zLimit, float zMinLimit, float2 ddx, float2 ddy, bool synthesize = false ) {
        float2 h, d;
        dirToRusinkiewicz(wi, wo, h, d);
        float3 vh = spherical_to_cartesian_rad(h);
        float3 vd = spherical_to_cartesian_rad(d);

        if (vh.z > zLimit) {
            float sinTheta = sqrt(1 - vh.z * vh.z);
            float newSinTheta = sqrt(1 - zLimit * zLimit);
            vh.z = zLimit;
            vh.xy *= newSinTheta / sinTheta;
            vh = normalize(vh);
        }

        if (vh.z < zMinLimit) {
            float sinTheta = sqrt(1 - vh.z * vh.z);
            float newSinTheta = sqrt(1 - zMinLimit * zMinLimit);
            vh.z = zMinLimit;
            vh.xy *= newSinTheta / sinTheta;
            vh = normalize(vh);
        }

        h = hemisphereDirToDiskSquare(vh);
        d = hemisphereDirToDiskSquare(vd);

        int count = 0;
        for (uint i = 0; i < hDims.y; i++) {
            float4 val = hP.SampleLevel(gSampler, float3(h, i), 0.f);
            inputBuffer[count + offset] = packFloat32ToFloat16(val.x, val.y);
            inputBuffer[count + offset + 1] = packFloat32ToFloat16(val.x, val.y);
            count += 2;
            // inputBuffer[count + offset] = val.x;
            // inputBuffer[count + offset + 1] = val.y;
            // inputBuffer[count + offset + 2] = val.z;
            // inputBuffer[count + offset + 3] = val.w;
            // count += 4;
        }
        for (uint i = 0; i < uDims.y; i++) {
            float4 val;
            if (synthesize)
                val = HexTiling(uP, gSampler, uv.yx, i, ddx, ddy);
            else
                val = uP.SampleGrad(gSampler, float3(uv, i), ddx, ddy);
            inputBuffer[count + offset] = packFloat32ToFloat16(val.x, val.y);
            inputBuffer[count + offset + 1] = packFloat32ToFloat16(val.x, val.y);
            count += 2;
            // inputBuffer[count + offset] = val.x;
            // inputBuffer[count + offset + 1] = val.y;
            // inputBuffer[count + offset + 2] = val.z;
            // inputBuffer[count + offset + 3] = val.w;
            // count += 4;
        }
        for (uint i = 0; i < dDims.y; i++) {
            float4 val = dP.SampleLevel(gSampler, float3(d, i), 0.f);
            inputBuffer[count + offset] = packFloat32ToFloat16(val.x, val.y);
            inputBuffer[count + offset + 1] = packFloat32ToFloat16(val.x, val.y);
            count += 2;
            // inputBuffer[count + offset] = val.x;
            // inputBuffer[count + offset + 1] = val.y;
            // inputBuffer[count + offset + 2] = val.z;
            // inputBuffer[count + offset + 3] = val.w;
            // count += 4;
        }
    }
}
