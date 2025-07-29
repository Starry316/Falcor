import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    bool gNeedHDRRecon;
}
// final render image
RWTexture2D<float4> gOutputColor;
// reflectance
Buffer<float> gInputColor;
Buffer<int> cudaValidBuffer;

float3 extractEulerAngleXYZ(const float3x3 m)
{
    float t1 = atan2(m[1][2], m[2][2]);
    float c2 = sqrt(m[0][0] * m[0][0] + m[0][1] * m[0][1]);
    float t2 = atan2(-m[0][2], c2);
    float s1 = sin(t1);
    float c1 = cos(t1);
    float t3 = atan2(s1 * m[2][0] - c1 * m[1][0], c1 * m[1][1] - s1 * m[2][1]);
    return float3(-t2, -t3, -t1);
}


float3x3 eulerToMat3(const float3 euler)
{
    float cx = cos(euler.x);
    float sx = sin(euler.x);
    float cy = cos(euler.y);
    float sy = sin(euler.y);
    float cz = cos(euler.z);
    float sz = sin(euler.z);
    // Rotation matrices for each axis
    float3x3 Rx = float3x3(
            1.0, 0.0, 0.0,
            0.0, cx, -sx,
            0.0, sx, cx
        );

    float3x3 Ry = float3x3(
            cy, 0.0, sy, 
            0.0, 1.0, 0.0, 
            -sy, 0.0, cy 
        );

    float3x3 Rz = float3x3(
        cz, -sz, 0.0, // First column
        sz, cz, 0.0, // Second column
        0.0, 0.0, 1.0 // Third column
    );

    return mul(mul(Rz, Ry), Rx);
    // return mul(Rz, mul(Ry, Rx));
}

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

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);

    print("rot Ref1", gOutputColor[pixel]);

    if (cudaValidBuffer[pixel.y * gRenderTargetDim.x + pixel.x] == 0)
        return;
    float4 btfVal = float4(
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]
    );


    if (any(isnan(btfVal)) || any(isinf(btfVal)))
        btfVal = float4(0);
    if (gNeedHDRRecon) {
        btfVal = exp(btfVal) - 1.0f;
        btfVal = pow(btfVal, 1.0f / 1.4f);
    }

    gOutputColor[pixel] *= btfVal;
}

