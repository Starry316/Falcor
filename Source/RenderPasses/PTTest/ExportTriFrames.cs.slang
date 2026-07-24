/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/

// Computes the per-triangle local->world frame for EVERY triangle of a selected instance and writes
// it to gTriFrame. The frame math must match traceLightProbeHemiPrimaryRay in MinimalPathTracer.rt.slang
// exactly: N = geometric face normal, T = perp_stark(N), B = cross(N, T), origin = sampled point on the
// triangle using the same gTriSampleUV. One thread per triangle.

import Scene.Scene;
import Utils.Math.MathHelpers;

// Must match ProbeTriFrame in MinimalPathTracer.rt.slang (4 x float4 = 64 bytes).
struct ProbeTriFrame
{
    float4 N;      ///< xyz = world-space face normal (local +Z axis).
    float4 T;      ///< xyz = world-space tangent (local +X axis).
    float4 B;      ///< xyz = world-space bitangent (local +Y axis).
    float4 origin; ///< xyz = world-space sample origin on the triangle.
};

RWStructuredBuffer<ProbeTriFrame> gTriFrame;

cbuffer CB
{
    uint gInstanceID;     ///< Geometry instance whose triangles are processed.
    uint gTriangleCount;  ///< Number of triangles in the instance's mesh.
    float2 gTriSampleUV;  ///< Barycentric sample coordinates (same seed as the render path).
}

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint tri = dispatchThreadId.x;
    if (tri >= gTriangleCount)
        return;

    GeometryInstanceID instance;
    instance.index = gInstanceID;

    float3 pos[3];
    gScene.getVertexPositionsW(instance, tri, pos);
    float3 faceN = gScene.getFaceNormalW(instance, tri);

    // Sample point on the triangle + tangent frame from the face normal (matches the render path).
    float3 bary = sample_triangle(gTriSampleUV);
    float3 sampledPos = bary.x * pos[0] + bary.y * pos[1] + bary.z * pos[2];

    float3 N = faceN;
    float3 T = normalize(perp_stark(N));
    float3 B = cross(N, T);

    ProbeTriFrame tf;
    tf.N = float4(N, 0.0f);
    tf.T = float4(T, 0.0f);
    tf.B = float4(B, 0.0f);
    tf.origin = float4(sampledPos, 1.0f);
    gTriFrame[tri] = tf;
}
