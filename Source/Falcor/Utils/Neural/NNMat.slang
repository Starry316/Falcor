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

    // Euler angles to rotation matrix (XYZ order)
    float3x3 EulerToRotationMatrix(float3 euler)
    {
        // Extract individual angles
        float x = euler.x; // Pitch
        float y = euler.y; // Yaw
        float z = euler.z; // Roll

        // Precompute sines and cosines
        float cx = cos(x);
        float sx = sin(x);
        float cy = cos(y);
        float sy = sin(y);
        float cz = cos(z);
        float sz = sin(z);

        // Construct rotation matrix (XYZ order)
        float3x3 rotMatrix;

        // First row
        rotMatrix[0][0] = cy * cz;
        rotMatrix[0][1] = cy * sz;
        rotMatrix[0][2] = -sy;

        // Second row
        rotMatrix[1][0] = sx * sy * cz - cx * sz;
        rotMatrix[1][1] = sx * sy * sz + cx * cz;
        rotMatrix[1][2] = sx * cy;

        // Third row
        rotMatrix[2][0] = cx * sy * cz + sx * sz;
        rotMatrix[2][1] = cx * sy * sz - sx * cz;
        rotMatrix[2][2] = cx * cy;

        return rotMatrix;
    }

    // Rotation matrix to Euler angles (XYZ order)
    float3 RotationMatrixToEuler(float3x3 R)
    {
        float3 euler;

        // Check for gimbal lock (when cos(y) is close to 0)
        if (abs(R[0][2]) > 0.99999f)
        {
            // Gimbal lock case
            euler.x = 0.0f;           // Can be set to any value
            euler.y = -asin(R[0][2]); // -PI/2 or PI/2
            euler.z = atan2(-R[1][0], R[1][1]);
        }
        else
        {
            // Regular case
            euler.y = -asin(R[0][2]);
            euler.x = atan2(R[1][2], R[2][2]);
            euler.z = atan2(R[0][1], R[0][0]);
        }

        return euler;
    }

    // Find rotation matrix that transforms basis1 to basis2
    float3x3 FindRotationBetweenBases(float3x3 basis1, float3x3 basis2)
    {
        // Assuming both bases are orthonormal
        // basis1 and basis2 are 3x3 matrices where columns are the basis vectors

        // The rotation matrix is: R = basis2 * transpose(basis1)
        float3x3 basis1Transpose;
        basis1Transpose[0] = float3(basis1[0][0], basis1[1][0], basis1[2][0]);
        basis1Transpose[1] = float3(basis1[0][1], basis1[1][1], basis1[2][1]);
        basis1Transpose[2] = float3(basis1[0][2], basis1[1][2], basis1[2][2]);

        // Multiply matrices
        float3x3 rotationMatrix;
        rotationMatrix = mul(basis2, basis1Transpose);
        // Perform matrix multiplication manually

        return rotationMatrix;
    }

    float3 rotationDebug(float3 T, float3 B, float3 N, float3 v)
    {
        float3x3 rot = float3x3(T, B, N);

        float3 eulerangles = RotationMatrixToEuler(rot);
        float3x3 rotRef = EulerToRotationMatrix(eulerangles);
        // return mul(rotRef, v);
        return float3(rotRef[1][0], rotRef[1][1], rotRef[1][2]);
    }

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

        if (isWi)
        {
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
        }
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
