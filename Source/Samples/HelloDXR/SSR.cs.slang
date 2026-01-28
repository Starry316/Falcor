#include "Utils/Math/MathConstants.slangh"
#include "utils.slangh"
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import Scene.Intersection;
import Utils.Geometry.GeometryHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGenerator;

cbuffer PerFrameCB
{
    uint2 gFrameDim;
    uint gShowBillboardID;
    uint gBillboardCount;
    // uint gDebugViewID;
    bool gDebugMode;
    float gInvalidThreshold;
    float gMaxInvalidThreshold;
    float gThickness;
    bool3 gViewRenderMask;

};
RWTexture2D<float4> gOutput;
Texture2DArray<float4> gBillboards;
Texture2D<float4> gReference;
StructuredBuffer<float4x4> gViewProjBuffer;
StructuredBuffer<float4x4> gViewProjInvBuffer;
StructuredBuffer<float3> gBillboardNormalBuffer;

SamplerState gSampler;
uint3 findTop3Normals(float3 rayDir)
{
    float3 rd = rayDir;
    float rdLen2 = dot(rd, rd);
    if (rdLen2 <= 1e-12f)
        return uint3(0, 1, 2);
    rd = normalize(rd);

    float bestDot0 = -1e9f;
    uint idx0 = 0xffffffffu;
    float bestDot1 = -1e9f;
    uint idx1 = 0xffffffffu;
    float bestDot2 = -1e9f;
    uint idx2 = 0xffffffffu;

    for (uint i = 0; i < gBillboardCount; ++i)
    {
        float3 n = gBillboardNormalBuffer[i];
        float nlen2 = dot(n, n);
        if (nlen2 <= 1e-12f)
            continue;



        n = normalize(n);

        float d = dot(n, rd);

        if (d > bestDot0)
        {
            bestDot2 = bestDot1;
            idx2 = idx1;
            bestDot1 = bestDot0;
            idx1 = idx0;
            bestDot0 = d;
            idx0 = i;
        }
        else if (d > bestDot1)
        {
            bestDot2 = bestDot1;
            idx2 = idx1;
            bestDot1 = d;
            idx1 = i;
        }
        else if (d > bestDot2)
        {
            bestDot2 = d;
            idx2 = i;
        }
    }

    if (idx0 == 0xffffffffu)
        idx0 = 0;
    if (idx1 == 0xffffffffu)
        idx1 = (gBillboardCount > 1) ? 1 : idx0;
    if (idx2 == 0xffffffffu)
        idx2 = (gBillboardCount > 2) ? 2 : idx1;

    return uint3(idx0, idx1, idx2);
}

float maskedDiff(float3 a, float3 b, float3 c, bool validA, bool validB, bool validC)
{
    if (!validA && !validB && !validC)
        return 0.0f;
    float3 ab = a - b;
    float3 ac = a - c;
    float3 bc = b - c;
    float lenAB = validA && validB ? length(ab) : 0.0f;
    float lenAC = validA && validC ? length(ac) : 0.0f;
    float lenBC = validB && validC ? length(bc) : 0.0f;
    float count = (validA ? 1.0f : 0.0f) + (validB ? 1.0f : 0.0f) + (validC ? 1.0f : 0.0f);
    return (lenAB + lenAC + lenBC) / count;
}
float maskedMinDiff(float3 a, float3 b, float3 c, bool validA, bool validB, bool validC)
{
    if (!validA && !validB && !validC)
        return 0.0f;
    float3 ab = a - b;
    float3 ac = a - c;
    float3 bc = b - c;
    float minLen = 1e9f;
    float lenAB = validA && validB ? length(ab) : 1e9f;
    float lenAC = validA && validC ? length(ac) : 1e9f;
    float lenBC = validB && validC ? length(bc) : 1e9f;
    minLen = min(minLen, lenAB);
    minLen = min(minLen, lenAC);
    minLen = min(minLen, lenBC);
    return minLen;
}
float maskedMaxDiff(float3 a, float3 b, float3 c, bool validA, bool validB, bool validC)
{
    if (!validA && !validB && !validC)
        return 0.0f;
    float3 ab = a - b;
    float3 ac = a - c;
    float3 bc = b - c;
    float maxLen = -1;
    float lenAB = validA && validB ? length(ab) : -1;
    float lenAC = validA && validC ? length(ac) : -1;
    float lenBC = validB && validC ? length(bc) : -1;
    maxLen = max(maxLen, lenAB);
    maxLen = max(maxLen, lenAC);
    maxLen = max(maxLen, lenBC);
    return maxLen;
}

float3 maskedDiffEach(float3 a, float3 b, float3 c, bool validA, bool validB, bool validC)
{
    if (!validA && !validB && !validC)
        return float3(0.0f, 0.0f, 0.0f);
    float3 ab = a - b;
    float3 ac = a - c;
    float3 bc = b - c;
    float lenAB = validA && validB ? length(ab) : 100.0f;
    float lenAC = validA && validC ? length(ac) : 100.0f;
    float lenBC = validB && validC ? length(bc) : 100.0f;
    return float3(lenAB, lenAC, lenBC);
}
bool hasOutlier(float a, float b, float c)
{
    float maxv = max(a, max(b, c));
    float minv = min(a, min(b, c));
    // float eps = maxv
    return maxv > 3 * minv;
    // return (maxv - minv) > eps;
}
bool traceBillboard(float3 posW, float3 posWOut, uint billboardID, out float3 posUV)
{
    float3 pos = mul(gViewProjBuffer[billboardID], float4(posW, 1.f)).xyz;
    float3 posOut = mul(gViewProjBuffer[billboardID], float4(posWOut, 1.f)).xyz;

    pos.xy = 0.5 * pos.xy + 0.5;
    posOut.xy = 0.5 * posOut.xy + 0.5;

    pos.y = 1.0 - pos.y;
    posOut.y = 1.0 - posOut.y;

    float3 dir = posOut - pos;

    float stepSize = 0.001;

    for (uint i = 0; i < 1000; ++i)
    {
        pos += dir * stepSize;

        if (any(pos.xy <= 0) || any(pos.xy >= 1))
        {
            posUV = float3(0, 0, 0);
            return false;
        }
        float4 val = gBillboards.SampleLevel(gSampler, float3(pos.xy, billboardID), 0);
        // if (pos.z >= val.z && abs(pos.z - val.z) < gThickness)
        if (pos.z >= val.z)
        {
            pos.y = 1 - pos.y;
            pos.xy = pos.xy * 2.0 - 1.0;
            posUV = mul(gViewProjInvBuffer[billboardID], float4(pos, 1)).xyz;
            return true;
        }
    }
    posUV = float3(0, 0, 0);
    return false;
}





[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gFrameDim))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    Ray ray = gScene.camera.computeRayPinhole(pixel, gFrameDim, false);

    float t;
    float2 nearFar;
    // bool hitBounding = intersectRayAABB(ray.origin, ray.dir, float3(-1), float3(1), nearFar);
    bool hitBounding = intersectRaySphere(ray.origin, ray.dir, float3(0.0f), 1, nearFar);

    // gOutput[pixel].xyz = float3(0, 0, 0);
    gOutput[pixel].xyz = checkerboard(pixel);

    // miss bounding box
    if (!hitBounding)
        return;


    Ray rayCenter = gScene.camera.computeRayPinhole(gFrameDim/2, gFrameDim, false);

    // uint3 nearestBillboardID = findTop3Normals(rayCenter.dir);
    uint3 nearestBillboardID = findTop3Normals(ray.dir);
    float3 pos1, pos2, pos3;
    bool hit1, hit2, hit3;
    float3 startPosW = ray.origin + nearFar.x * ray.dir;
    float3 endPosW = ray.origin + nearFar.y * ray.dir;

    // hit1 = traceBillboard(startPosW, endPosW, nearestBillboardID.x, pos1);
    // hit2 = traceBillboard(startPosW, endPosW, nearestBillboardID.y, pos2);
    // hit3 = traceBillboard(startPosW, endPosW, nearestBillboardID.z, pos3);

    // hit1 = traceBillboard(startPosW, endPosW, nearestBillboardID.x, pos1);
    // if (hit1)
    //     startPosW = pos1;
    // hit2 = traceBillboard(startPosW, endPosW, nearestBillboardID.y, pos2);
    // if (hit2)
    //     startPosW = pos2;
    // hit3 = traceBillboard(startPosW, endPosW, nearestBillboardID.z, pos3);

    // hit1 = hit3 && hit2 && hit1;
    // hit2 = hit1;
    // hit3 = hit1;


    float3 finalHitPosW, posOutW;
    bool hit = true;
    for (int i = 0; i < gBillboardCount; ++i)
    {
        // if (!traceBillboard(startPosW, endPosW, nearestBillboardID[i], posOutW)){
        if (!traceBillboard(startPosW, endPosW, i, posOutW)){
            hit = false;
            break;
        }

        finalHitPosW = posOutW;
        startPosW = posOutW;
    }

    if (hit) {
        gOutput[pixel].xyz = 0.5f * finalHitPosW + 0.5f;

    }
    else {
        gOutput[pixel].xyz = float3(0, 0, 0);
    }
    return;

    float3 blendedPos;

    float intersectionMinDiff = maskedMinDiff(pos1, pos2, pos3, hit1, hit2, hit3);
    float intersectionMaxDiff = maskedMaxDiff(pos1, pos2, pos3, hit1, hit2, hit3);
    float intersectionAvgDiff = maskedDiff(pos1, pos2, pos3, hit1, hit2, hit3);

    float3 intersectionDiffEach = maskedDiffEach(pos1, pos2, pos3, hit1, hit2, hit3);

    print("nearestBillboardID", nearestBillboardID);
    print("hit", bool3(hit1, hit2, hit3));
    // print("w", w);
    print("pos1", pos1);
    print("pos2", pos2);
    print("pos3", pos3);
    print("intersection Min Diff", intersectionMinDiff);
    print("intersection Max Diff", intersectionMaxDiff);
    print("intersection Avg Diff", intersectionAvgDiff);
    print("intersection Diff AB, AC, BC", intersectionDiffEach);


    if (gDebugMode)
    {
        bool hit = traceBillboard(ray.origin + nearFar.x * ray.dir, ray.origin + nearFar.y * ray.dir, nearestBillboardID[gShowBillboardID], pos1);
        if (hit)
            gOutput[pixel].xyz = 0.5f * pos1 + 0.5f;
        return;
    }

    if (intersectionMinDiff > gInvalidThreshold || intersectionMaxDiff > gMaxInvalidThreshold )
    {
        gOutput[pixel].xyz = float3(0, 0, 0);
        return;
    }

    // if (hasOutlier(intersectionDiffEach.x, intersectionDiffEach.y, intersectionDiffEach.z) && intersectionMinDiff > gInvalidThreshold)
    // {
    //     gOutput[pixel].xyz = float3(0, 0, 0);
    //     return;
    // }



    float3 w = float3(0.333f, 0.3333f, 0.3333f);
    if (hit1 && hit2 && hit3)
    {
        blendedPos = w.x * pos1 + w.y * pos2 + w.z * pos3;
    }
    else
    {
        gOutput[pixel].xyz = float3(0, 0, 0);
        return;
    }
    if (!all(blendedPos == float3(0,0,0)))
        gOutput[pixel].xyz = 0.5f * blendedPos + 0.5f;


}
