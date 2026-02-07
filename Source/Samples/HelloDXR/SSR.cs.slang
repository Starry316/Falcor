#include "Utils/Math/MathConstants.slangh"
#include "utils.slangh"
#include "SSRDefs.slangh"
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
    uint gBillboardTraceCount;
    uint gTraceStepCount;
    uint gFwdBillboardCount;
    uint gFwdBillboardRadius;
    float gFwdScreenRadius;
    // uint gDebugViewID;
    bool gDebugMode;
    float gInvalidThreshold;
    float gMaxInvalidThreshold;
    float gThickness;
    float gViewTheta;
    bool3 gViewRenderMask;
    bool gUseSortedBillboards;
    bool gShowColor;

};

#define USE_NORMAL gViewRenderMask.x
#define USE_NORMAL_TRACE gViewRenderMask.y
RWTexture2D<float4> gOutput;
Texture2DArray<float4> gBillboards;
Texture2DArray<float4> gBillboardColors;
Texture2DArray<float4> gBillboardPosWs;
Texture2DArray<float4> gBillboardNormalWs;
Texture2D<float4> gReference;
StructuredBuffer<float4x4> gViewProjBuffer;
StructuredBuffer<float4x4> gViewProjInvBuffer;
StructuredBuffer<float3> gBillboardNormalBuffer;

SamplerState gSampler;
SamplerState gMaxSampler;
uint3 findTop3Normals(float3 rayDir)
{
    // float3 rd = rayDir;
    // float rdLen2 = dot(rd, rd);
    // if (rdLen2 <= 1e-12f)
    //     return uint3(0, 1, 2);
    // rd = normalize(rd);

    // float bestDot0 = -1e9f;
    // uint idx0 = 0xffffffffu;
    // float bestDot1 = -1e9f;
    // uint idx1 = 0xffffffffu;
    // float bestDot2 = -1e9f;
    // uint idx2 = 0xffffffffu;

    // for (uint i = 0; i < gBillboardCount; ++i)
    // {
    //     float3 n = gBillboardNormalBuffer[i];
    //     float nlen2 = dot(n, n);
    //     if (nlen2 <= 1e-12f)
    //         continue;

    //     n = normalize(n);

    //     float d = dot(n, rd);

    //     if (d > bestDot0)
    //     {
    //         bestDot2 = bestDot1;
    //         idx2 = idx1;
    //         bestDot1 = bestDot0;
    //         idx1 = idx0;
    //         bestDot0 = d;
    //         idx0 = i;
    //     }
    //     else if (d > bestDot1)
    //     {
    //         bestDot2 = bestDot1;
    //         idx2 = idx1;
    //         bestDot1 = d;
    //         idx1 = i;
    //     }
    //     else if (d > bestDot2)
    //     {
    //         bestDot2 = d;
    //         idx2 = i;
    //     }
    // }

    // if (idx0 == 0xffffffffu)
    //     idx0 = 0;
    // if (idx1 == 0xffffffffu)
    //     idx1 = (gBillboardCount > 1) ? 1 : idx0;
    // if (idx2 == 0xffffffffu)
    //     idx2 = (gBillboardCount > 2) ? 2 : idx1;

    return uint3(0, 1, 2);
}

void sortBillboardsByDot(float3 rayDir, out uint sortedIdx[TOTAL_VIEWS])
{
    // float3 rd = rayDir;
    float dots[TOTAL_VIEWS];

    [unroll(TOTAL_VIEWS)]
    for (uint i = 0; i < TOTAL_VIEWS; ++i)
    {
        float3 n = gBillboardNormalBuffer[i];
        n = normalize(n);
        // dots[i] = abs(dot(n, rayDir));
        dots[i] = dot(n, rayDir);
        sortedIdx[i] = i;
    }

    [loop]
    for (uint i = 0; i < TOTAL_VIEWS; ++i)
    {
        uint best = i;
        float bestDot = dots[sortedIdx[i]];

        [loop]
        for (uint j = i + 1; j < TOTAL_VIEWS; ++j)
        {
            float d = dots[sortedIdx[j]];
            if (d > bestDot)
            {
                bestDot = d;
                best = j;
            }
        }

        if (best != i)
        {
            uint tmp = sortedIdx[i];
            sortedIdx[i] = sortedIdx[best];
            sortedIdx[best] = tmp;
        }
    }
}

// bool traceBillboard(float3 posWIn, float3 posWOut, uint billboardID, out float3 posWHit)
// {
//     float h, w, c;
//     gBillboards.GetDimensions(h, w, c);

//     float3 pos = mul(gViewProjBuffer[billboardID], float4(posWIn, 1.f)).xyz;
//     float3 posOut = mul(gViewProjBuffer[billboardID], float4(posWOut, 1.f)).xyz;

//     pos.xy = 0.5 * pos.xy + 0.5;
//     posOut.xy = 0.5 * posOut.xy + 0.5;

//     pos.y = 1.0 - pos.y;
//     posOut.y = 1.0 - posOut.y;

//     // detect if the starting point is already inside the HF
//     float depthIn = gBillboards.SampleLevel(gMaxSampler, float3(pos.xy, billboardID), 0).z;
//     if (pos.z >= depthIn || abs(pos.z - depthIn) < gThickness)
//     {
//         return true;
//     }

//     float3 dir = posOut - pos;

//     uint stepCount = max(gTraceStepCount, 1u);
//     float stepSize = 1.0f / stepCount;
//     for (uint i = 0; i < stepCount; ++i)
//     {
//         pos += dir * stepSize;

//         if (any(pos.xy <= 0) || any(pos.xy >= 1))
//         {
//             posWHit = float3(0, 0, 0);
//             return false;
//         }
//         float4 val = gBillboards.SampleLevel(gMaxSampler, float3(pos.xy, billboardID), 0);
//         if (pos.z >= val.z || abs(pos.z - val.z) < gThickness)
//         {
//             pos.y = 1 - pos.y;
//             pos.xy = pos.xy * 2.0 - 1.0;
//             posWHit = mul(gViewProjInvBuffer[billboardID], float4(pos, 1)).xyz;
//             return true;
//         }
//     }
//     posWHit = float3(0, 0, 0);
//     return false;
// }

// bool traceBillboard(float3 posWIn, float3 posWOut, uint billboardID, out float3 posWHit)
// {
//     uint texW = 0, texH = 0, texLayers = 0;
//     gBillboards.GetDimensions(texW, texH, texLayers);

//     float3 pos = mul(gViewProjBuffer[billboardID], float4(posWIn, 1.f)).xyz;
//     float3 posOut = mul(gViewProjBuffer[billboardID], float4(posWOut, 1.f)).xyz;

//     pos.xy = 0.5 * pos.xy + 0.5;
//     posOut.xy = 0.5 * posOut.xy + 0.5;

//     pos.y = 1.0 - pos.y;
//     posOut.y = 1.0 - posOut.y;

//     // detect if the starting point is already inside the HF
//     float depthIn = gBillboards.SampleLevel(gMaxSampler, float3(pos.xy, billboardID), 0).z;
//     if (pos.z >= depthIn || abs(pos.z - depthIn) < gThickness)
//     {
//         return true;
//     }

//     float3 dir = posOut - pos;

//     uint stepCount = max(gTraceStepCount, 1u);
//     float stepSize = 1.0f / stepCount;

//     float2 deltaUV = dir.xy / (float)stepCount;
//     float deltaZ = dir.z / (float)stepCount;

//     float3 cur = pos;

//     for (uint i = 0; i < stepCount; ++i)
//     {
//         cur.xy += deltaUV;
//         cur.z += deltaZ;

//         if (cur.x <= 0.0 || cur.x >= 1.0 || cur.y <= 0.0 || cur.y >= 1.0)
//         {
//             posWHit = float3(0, 0, 0);
//             return false;
//         }

//         // UV -> texel integer coords（floor）
//         int2 texCoord = int2(floor(cur.xy * float2((float)texW, (float)texH)));

//         texCoord = clamp(texCoord, int2(0, 0), int2((int)texW - 1, (int)texH - 1));

//         float4 val = gBillboards.Load(int4(texCoord.xy, (int)billboardID, 0));
//         // float4 val = gBillboards.SampleLevel(gMaxSampler, float3(texCoord.xy, billboardID), 0);
//         float depthTex = val.z;

//         if (cur.z >= depthTex || abs(cur.z - depthTex) < gThickness)
//         {
//             float3 prev = cur - float3(deltaUV, deltaZ);
//             float3 low = prev;
//             float3 high = cur;
//             const int refineSteps = 4;

//             for (int r = 0; r < refineSteps; ++r)
//             {
//                 float3 mid = (low + high) * 0.5;
//                 int2 midTex = int2(floor(mid.xy * float2((float)texW, (float)texH)));
//                 midTex = clamp(midTex, int2(0, 0), int2((int)texW - 1, (int)texH - 1));
//                 float4 midVal = gBillboards.Load(int4(midTex.xy, (int)billboardID, 0));
//                 // float4 midVal = gBillboards.SampleLevel(gMaxSampler, float3(midTex.xy, billboardID), 0);
//                 float midDepth = midVal.z;
//                 if (mid.z >= midDepth || abs(mid.z - midDepth) < gThickness)
//                 {
//                     high = mid;
//                 }
//                 else
//                 {
//                     low = mid;
//                 }
//             }

//             float3 hitUV = (low + high) * 0.5;

//             hitUV.y = 1.0 - hitUV.y;
//             hitUV.xy = hitUV.xy * 2.0 - 1.0;

//             posWHit = mul(gViewProjInvBuffer[billboardID], float4(hitUV, 1)).xyz;
//             return true;
//         }
//     }
//     posWHit = float3(0, 0, 0);
//     return false;
// }
bool getBillboardColor(float3 posW, uint billboardID, out float4 outColor)
{
    float3 pos = mul(gViewProjBuffer[billboardID], float4(posW, 1.f)).xyz;
    pos.xy = 0.5 * pos.xy + 0.5;
    pos.y = 1.0 - pos.y;

    float depth = gBillboards.SampleLevel(gMaxSampler, float3(pos.xy, billboardID), 0).z;
    outColor = gBillboardColors.SampleLevel(gSampler, float3(pos.xy, billboardID), 0);
    return abs(depth - pos.z) < gInvalidThreshold;
}

float getBillboardColorDepth(float3 posW, uint billboardID, out float4 outColor)
{
    float3 pos = mul(gViewProjBuffer[billboardID], float4(posW, 1.f)).xyz;
    pos.xy = 0.5 * pos.xy + 0.5;
    pos.y = 1.0 - pos.y;

    float depth = gBillboards.SampleLevel(gMaxSampler, float3(pos.xy, billboardID), 0).z;
    outColor = gBillboardColors.SampleLevel(gSampler, float3(pos.xy, billboardID), 0);
    return pos.z < depth ? 1e9f : abs(depth - pos.z);
}

bool traceBillboard(float3 posWIn, float3 posWOut, uint billboardID, out float3 posWHit)
{
    uint texW = 0, texH = 0, texLayers = 0;
    gBillboards.GetDimensions(texW, texH, texLayers);

    float3 pos = mul(gViewProjBuffer[billboardID], float4(posWIn, 1.f)).xyz;
    float3 posOut = mul(gViewProjBuffer[billboardID], float4(posWOut, 1.f)).xyz;

    pos.xy = 0.5 * pos.xy + 0.5;
    posOut.xy = 0.5 * posOut.xy + 0.5;
    pos.y = 1.0 - pos.y;
    posOut.y = 1.0 - posOut.y;

    float3 dir = posOut - pos;

    // detect if the starting point is already inside the HF
    float depthIn = gBillboards.SampleLevel(gMaxSampler, float3(pos.xy, billboardID), 0).z;
    if (pos.z >= depthIn || abs(pos.z - depthIn) < gThickness)
    {
        return true;
    }
    //
    // float absUx = abs(dir.x) * (float)texW;
    // float absUy = abs(dir.y) * (float)texH;

    // "near perpendicular" case. cover within one texel
    float absUx = abs(dir.x) * (float)texW;
    float absUy = abs(dir.y) * (float)texH;
    const float perpEps = 0.5f; //
    if (absUx < perpEps && absUy < perpEps)
    {
        int2 texCoord = int2(floor(pos.xy * float2((float)texW, (float)texH)));
        texCoord = clamp(texCoord, int2(0, 0), int2((int)texW - 1, (int)texH - 1));
        float4 val = gBillboards.Load(int4(texCoord.x, texCoord.y, billboardID, 0));
        float depthTex = val.z;

        float zNear = min(pos.z, posOut.z);
        float zFar = max(pos.z, posOut.z);

        // if (depthTex >= zNear - gThickness && depthTex <= zFar + gThickness)
        if (zNear >= depthTex || zFar >= depthTex)
        {
            float3 hitUV = pos;
            hitUV.y = 1.0 - hitUV.y;
            hitUV.xy = hitUV.xy * 2.0 - 1.0;
            posWHit = mul(gViewProjInvBuffer[billboardID], float4(hitUV, 1)).xyz;
            return true;
        }
        else
        {
            posWHit = float3(0, 0, 0);
            return false;
        }
    }

    uint stepCount = max(1u, (uint)ceil(max(absUx, absUy)));

    // stepCount = min(stepCount, 4096u);

    float2 deltaUV = dir.xy / (float)stepCount;
    float deltaZ = dir.z / (float)stepCount;

    float3 cur = pos;

    for (uint i = 0; i < stepCount; ++i)
    {
        cur.xy += deltaUV;
        cur.z += deltaZ;

        if (cur.x <= 0.0 || cur.x >= 1.0 || cur.y <= 0.0 || cur.y >= 1.0)
        {
            posWHit = float3(0, 0, 0);
            return false;
        }

        // UV -> texel integer coords（floor）
        int2 texCoord = int2(floor(cur.xy * float2((float)texW, (float)texH)));

        texCoord = clamp(texCoord, int2(0, 0), int2((int)texW - 1, (int)texH - 1));

        float4 val = gBillboards.Load(int4(texCoord.xy, (int)billboardID, 0));
        float depthTex = val.z;

        if (cur.z >= depthTex || abs(cur.z - depthTex) < gThickness)
        {
            float3 prev = cur - float3(deltaUV, deltaZ);
            float3 low = prev;
            float3 high = cur;

            // if (gViewRenderMask.x) {
            const int refineSteps = 4;

            for (int r = 0; r < refineSteps; ++r)
            {
                float3 mid = (low + high) * 0.5;
                int2 midTex = int2(floor(mid.xy * float2((float)texW, (float)texH)));
                midTex = clamp(midTex, int2(0, 0), int2((int)texW - 1, (int)texH - 1));
                float4 midVal = gBillboards.Load(int4(midTex.xy, (int)billboardID, 0));
                float midDepth = midVal.z;
                if (mid.z >= midDepth || abs(mid.z - midDepth) < gThickness)
                {
                    high = mid;
                }
                else
                {
                    low = mid;
                }
            }
            // }

            float3 hitUV = (low + high) * 0.5;

            hitUV.y = 1.0 - hitUV.y;
            hitUV.xy = hitUV.xy * 2.0 - 1.0;

            posWHit = mul(gViewProjInvBuffer[billboardID], float4(hitUV, 1)).xyz;
            return true;
        }
    }

    posWHit = float3(0, 0, 0);
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
    bool hitBounding = intersectRaySphere(ray.origin, ray.dir, float3(0.0f), 1, nearFar);

    // gOutput[pixel].xyz = float3(0.38f, 0.38f, 0.38f);
    gOutput[pixel].xyz = checkerboard(pixel/2);

    // miss bounding box
    if (!hitBounding)
        return;

    Ray rayCenter = gScene.camera.computeRayPinhole(gFrameDim / 2, gFrameDim, false);

    uint sortedIdx[TOTAL_VIEWS];
    sortBillboardsByDot(ray.dir, sortedIdx);

    if (gDebugMode)
    {
        float3 pos1;
        uint realID = gShowBillboardID;
        if (gUseSortedBillboards)
            realID = sortedIdx[gShowBillboardID];

        bool hit = traceBillboard(ray.origin + nearFar.x * ray.dir, ray.origin + nearFar.y * ray.dir, realID, pos1);
        if (hit)
            gOutput[pixel].xyz = 0.5f * pos1 + 0.5f;
        return;
    }

    // float h, wd, c;
    // gBillboards.GetDimensions(h, wd, c);
    // print("h,w,c", float3(h, wd, c));

    float3 startPosW = ray.origin + nearFar.x * ray.dir;
    float3 endPosW = ray.origin + nearFar.y * ray.dir;

    float3 finalHitPosW, posOutW;
    float3 lastHitPosW = startPosW;
    uint lastHitBillboardID = 0;
    bool hit = true;
    int curBillboardID = 0;
    for (curBillboardID = 0; curBillboardID < gBillboardTraceCount; ++curBillboardID)
    {
        uint realID = curBillboardID;
        if (gUseSortedBillboards)
            realID = sortedIdx[curBillboardID];
        if (USE_NORMAL_TRACE && abs(dot(gBillboardNormalBuffer[realID], ray.dir)) < gViewTheta)
            continue;
        if (!traceBillboard(startPosW, endPosW, realID, posOutW))
        {
            hit = false;
            break;
        }

        finalHitPosW = posOutW;
        startPosW = posOutW;

        // if(length(posOutW - lastHitPosW) > 1e-6f )
        // {
        //     lastHitBillboardID = realID;
        //     lastHitPosW = posOutW;
        // }
    }

    if (hit)
    {
        float cloestDist = 1e9f;
        float3 cloestPosW = finalHitPosW;
        float3 cloestUVID = float3(0, 0, -1);

        if (!gShowColor)
        {
            gOutput[pixel].xyz = 0.5f * finalHitPosW + 0.5f;
            return;
        }
        float3 avgColor = float3(0, 0, 0);
        uint validCount = 0;
        uint totalCount = 0;

        for (curBillboardID = 0; curBillboardID < gFwdBillboardCount; ++curBillboardID)
        {
            uint realID = curBillboardID;
            if (gUseSortedBillboards)
                realID = sortedIdx[curBillboardID];

            float3 pos = mul(gViewProjBuffer[realID], float4(finalHitPosW, 1.f)).xyz;
            pos.xy = 0.5 * pos.xy + 0.5;
            pos.y = 1.0 - pos.y;

            // uint2 texelPos = uint2((pos.xy * float2(wd, h))- 0.5);
            for (int dy = -int(gFwdBillboardRadius); dy <= int(gFwdBillboardRadius); ++dy)
            {
                for (int dx = -int(gFwdBillboardRadius); dx <= int(gFwdBillboardRadius); ++dx)
                {
                    float2 curTexel = pos.xy + float2(dx, dy) / float2(BILLBOARD_RESOLUTION);
                    if (curTexel.x >= 1.0 || curTexel.y >= 1.0 || curTexel.x < 0 || curTexel.y < 0)
                        continue;

                    if (USE_NORMAL && dot(normalizeSafe(gBillboardNormalWs.SampleLevel(gSampler, float3(curTexel.xy, realID), 0).xyz), -ray.dir) <= 0.0f)
                        continue;

                    totalCount += 1;
                    // float3 curPosW = mul(gViewProjInvBuffer[realID], float4(curTexel, 1.0f)).xyz;


                    float3 curPosW = gBillboardPosWs.SampleLevel(gSampler, float3(curTexel.xy, realID), 0).xyz;
                    float4 curPosProj = mul(gScene.camera.data.viewProjMat, float4(curPosW, 1.0f));
                    float3 curPosNDC = curPosProj.xyz / curPosProj.w;
                    curPosNDC.xy = 0.5 * curPosNDC.xy + 0.5;
                    curPosNDC.y = 1.0 - curPosNDC.y;
                    float2 screen = curPosNDC.xy * gFrameDim - 0.5f;
                    float2 pcenter = float2(pixel) + 0.5f;

                    if (abs(screen.x - pcenter.x) > gFwdScreenRadius || abs(screen.y - pcenter.y) > gFwdScreenRadius)
                        continue;

                    avgColor += gBillboardColors.SampleLevel(gSampler, float3(curTexel.xy, realID), 0).xyz;
                    validCount += 1;

                }
            }
        }
        if (validCount > 0)
            gOutput[pixel].xyz = avgColor / (float)validCount;

        // gOutput[pixel].xyz = gBillboardColors.SampleLevel(gSampler, cloestUVID, 0).xyz;
        print("validCount", validCount);
        print("totalCount", totalCount);
    }

    // if (hit)
    // {
    //     gOutput[pixel].xyz = 0.5f * finalHitPosW + 0.5f;
    //     if (!gShowColor)
    //     {
    //         return;
    //     }

    //     float minDiff = 1e9f;
    //     float3 avgColor = float3(0, 0, 0);
    //     uint validCount = 0;

    //     for (curBillboardID = 0; curBillboardID < gBillboardTraceCount; ++curBillboardID)
    //     {
    //         uint realID = curBillboardID;

    //         if (gUseSortedBillboards)
    //             realID = sortedIdx[curBillboardID];

    //         if (dot(normalize(gBillboardNormalBuffer[realID]), ray.dir) <= 0.0f)
    //         {
    //             continue;
    //         }
    //         float4 outColor = float4(0, 0, 0, 0);
    //         if (getBillboardColor(finalHitPosW, realID, outColor))
    //         {
    //             gOutput[pixel] = outColor;

    //             avgColor += outColor.xyz;
    //             validCount += 1;
    //             break;
    //         }

    //         // float diff = getBillboardColorDepth(finalHitPosW, realID, outColor);
    //         // if (diff < minDiff)
    //         // {
    //         //     minDiff = diff;
    //         //     gOutput[pixel] = outColor;
    //         //     lastHitBillboardID = realID;
    //         // }
    //     }
    //     if (validCount > 0)
    //     {
    //         avgColor /= (float)validCount;
    //         gOutput[pixel].xyz = avgColor;
    //     }
    //     print("minDiff", minDiff);
    //     print("lastHitBillboardID", lastHitBillboardID);
    // }

    // else
    // {
    //     gOutput[pixel].xyz = float3(0, 0, 0);
    // }

    print("final hit billboard count", curBillboardID);
    print("is hit", hit);

    return;

    // float3 blendedPos;

    // float intersectionMinDiff = maskedMinDiff(pos1, pos2, pos3, hit1, hit2, hit3);
    // float intersectionMaxDiff = maskedMaxDiff(pos1, pos2, pos3, hit1, hit2, hit3);
    // float intersectionAvgDiff = maskedDiff(pos1, pos2, pos3, hit1, hit2, hit3);

    // float3 intersectionDiffEach = maskedDiffEach(pos1, pos2, pos3, hit1, hit2, hit3);

    // print("nearestBillboardID", nearestBillboardID);
    // print("hit", bool3(hit1, hit2, hit3));
    // // print("w", w);
    // print("pos1", pos1);
    // print("pos2", pos2);
    // print("pos3", pos3);
    // print("intersection Min Diff", intersectionMinDiff);
    // print("intersection Max Diff", intersectionMaxDiff);
    // print("intersection Avg Diff", intersectionAvgDiff);
    // print("intersection Diff AB, AC, BC", intersectionDiffEach);

    // if (intersectionMinDiff > gInvalidThreshold || intersectionMaxDiff > gMaxInvalidThreshold)
    // {
    //     gOutput[pixel].xyz = float3(0, 0, 0);
    //     return;
    // }

    // if (hasOutlier(intersectionDiffEach.x, intersectionDiffEach.y, intersectionDiffEach.z) && intersectionMinDiff > gInvalidThreshold)
    // {
    //     gOutput[pixel].xyz = float3(0, 0, 0);
    //     return;
    // }

    // float3 w = float3(0.333f, 0.3333f, 0.3333f);
    // if (hit1 && hit2 && hit3)
    // {
    //     blendedPos = w.x * pos1 + w.y * pos2 + w.z * pos3;
    // }
    // else
    // {
    //     gOutput[pixel].xyz = float3(0, 0, 0);
    //     return;
    // }
    // if (!all(blendedPos == float3(0, 0, 0)))
    //     gOutput[pixel].xyz = 0.5f * blendedPos + 0.5f;
}
