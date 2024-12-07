#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Scene.Displacement.DisplacementMapping;
import Utils.Geometry.IntersectionHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MatrixUtils;
import Utils.Texture.HexTiling;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
SamplerState mySampler;
bool traceSynthesizedHeightMapEstimated( const DisplacementData displacementData, float3 startPoint, float3 endPoint, out float intersectedT, out float intersectedHeight )
{
    intersectedT = {};
    intersectedHeight = {};

    // const float entryHeightData = displacementData.readValue(startPoint.xy);
    const float entryHeightData = displacementData.texture.SampleLevel(mySampler, 1.5f * (startPoint.xy / displacementData.size), 0).x;
    // float entryHeightData = HexTilingHeight( displacementData.texture, displacementData.samplerState, float2(0.0001), float2(0.0001), (startPoint.xy / displacementData.size) * 1 );
    // entryHeightData = displacementData.mapValue(entryHeightData);
#if DISPLACEMENT_TWO_SIDED == 0
    if (startPoint.z < entryHeightData)
    {
        intersectedT = 0.f; // This seems to make the intersection ignored. TODO: experiment with eg. 0.000001
        intersectedHeight = startPoint.z;
        return true;
    }
#endif
    float3 tracingPath = endPoint - startPoint;
    float finestStepLength = 0.0001;
    float maxFinestStep = 500;
    float3 deltaPos = tracingPath / maxFinestStep;

    float3 curPos = startPoint;
    bool flagFound = false;
    const float tMax = solveT(startPoint, tracingPath, endPoint);

    for (uint step = 0; step < maxFinestStep; step++)
    {
        // float height = displacementData.readValue(curPos.xy);
        float height = displacementData.texture.SampleLevel(mySampler, 1.5f * (curPos.xy / displacementData.size), 0).x;
        if ((height + FLT_EPSILON >= curPos.z || curPos.z <= 0))
        {
            intersectedT = tMax * (float(step) / maxFinestStep);
            intersectedHeight = curPos.z;
            flagFound = true;
            break;
        }
        curPos += deltaPos;

    }

    return flagFound;






    // float3 viewDirection = endPoint - startPoint;
    // int maxCount = ceil(abs(viewDirection.x)) + ceil(abs(viewDirection.y)) + 2;
    // maxCount = min(maxCount * kRaymarchingSampleCountFactor, kRaymarchingMaxSampleCount); // Prevent freeze due to corner cases. Todo: Investigate why this happens.
    // maxCount = 100;
    // float3 incrementUnit = viewDirection / maxCount / 2;
    // viewDirection = normalize(viewDirection);
    // const float tMax = solveT(startPoint, viewDirection, endPoint);

    // float3 lastTestPoint;
    // float3 currentTestPoint = startPoint;
    // float lastHeightDelta = currentTestPoint.z - entryHeightData;

    // for (int i = 0; i < maxCount; i++) {
    //     lastTestPoint = currentTestPoint;
    //     currentTestPoint += incrementUnit;
    //     const float heightData = displacementData.readValue(currentTestPoint.xy);
    //     // float heightData = HexTilingHeight( displacementData.texture, displacementData.samplerState, 0.0001, 0.0001, (currentTestPoint.xy / displacementData.size) * 1 );
    //     // float heightData = HexTilingHeight( displacementData.texture, displacementData.samplerState, 0.0001, 0.0001, (currentTestPoint.xy / displacementData.size) * 1 );
    //     // heightData = displacementData.mapValue(heightData);
    //     if (currentTestPoint.z < heightData) {
    //         intersectedT = tMax * (float(i) / maxCount);
    //         intersectedHeight = currentTestPoint.z;
    //         return true;
    //     }
    // }
    // return false;


//     for (int i = 0; i < maxCount; i++)
//     {
//         lastTestPoint = currentTestPoint;
//         currentTestPoint += incrementUnit;

//         // const float heightData = displacementData.readValue(currentTestPoint.xy);
//         const float heightData = HexTilingHeight(displacementData.texture, displacementData.samplerState, 0.0001, 0.0001, (currentTestPoint.xy) * 5)*0.5;
//         const float currentHeightDelta = currentTestPoint.z - heightData;

// #if DISPLACEMENT_TWO_SIDED == 0
//         if (currentHeightDelta < 0.f)
// #else
//         if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) || (sign(lastHeightDelta) != sign(currentHeightDelta)))
// #endif
//         {
//             float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
//             float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

//             intersectedT = tMax * (estimatedT / maxCount);
//             intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);

//             return true;
//         }
//         lastHeightDelta = currentHeightDelta;
//     }

//     return false;
}
bool calcSynthesizedDisplacementIntersection( const Ray ray, const StaticVertexData vertices[3], const DisplacementData displacementData, out DisplacementIntersection result )
{
    result = {};

    const float2 shellMinMax = float2(0.f, 1.0f);
    const float minHeight = shellMinMax.x;
    const float maxHeight = shellMinMax.y;

    float3 extrudedP0 = vertices[0].position + vertices[0].normal * maxHeight;
    float3 extrudedP1 = vertices[1].position + vertices[1].normal * maxHeight;
    float3 extrudedP2 = vertices[2].position + vertices[2].normal * maxHeight;
    float3 intrudedP0 = vertices[0].position + vertices[0].normal * minHeight;
    float3 intrudedP1 = vertices[1].position + vertices[1].normal * minHeight;
    float3 intrudedP2 = vertices[2].position + vertices[2].normal * minHeight;

    // 1. ray "prism" intersection to grab valid start/end t for the valid segment along ray direction
    float tStart = ray.tMin;

    // extruded triangle
    IntersectAttribute extrudedTriangleIntersectAttri;
    rayTriangleIntersectionTest(extrudedP0, extrudedP1, extrudedP2, ray.origin, ray.dir, maxHeight,
        extrudedTriangleIntersectAttri);

    // min: entry point, max: exit point
    IntersectAttribute minIntersection = extrudedTriangleIntersectAttri;
    IntersectAttribute maxIntersection = extrudedTriangleIntersectAttri;

    // first slab/fin
    IntersectAttribute slabIntersectAttri0;
    IntersectAttribute slabIntersectAttri1;
    rayBilinearPatchIntersectionTest( intrudedP0, extrudedP0, intrudedP1, extrudedP1, float3(1, 0, 0), float3(0, 1, 0),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1 );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // second slab/fin
    rayBilinearPatchIntersectionTest( intrudedP1, extrudedP1, intrudedP2, extrudedP2, float3(0, 1, 0), float3(0, 0, 1),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1 );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // third slab/fin
    rayBilinearPatchIntersectionTest( intrudedP2, extrudedP2, intrudedP0, extrudedP0, float3(0, 0, 1), float3(1, 0, 0),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1 );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // test base triangle lastly
    IntersectAttribute baseTriangleIntersectAttri;
    rayTriangleIntersectionTest(intrudedP0, intrudedP1, intrudedP2, ray.origin, ray.dir, minHeight,
        baseTriangleIntersectAttri);
    minIntersection.min(baseTriangleIntersectAttri);
    maxIntersection.max(baseTriangleIntersectAttri);

    // Early out when no hit.
    if (!(minIntersection.intersected && maxIntersection.intersected) || (minIntersection.t == maxIntersection.t) || (minIntersection.t > ray.tMax) || (maxIntersection.t < ray.tMin))
    {
        return false;
    }

    // 2. convert intersection data from object space to trace space
    float2 minIntersectionUV = minIntersection.barycentric[0] * vertices[0].texCrd.xy + minIntersection.barycentric[1] * vertices[1].texCrd.xy + minIntersection.barycentric[2] * vertices[2].texCrd.xy;
    float2 maxIntersectionUV = maxIntersection.barycentric[0] * vertices[0].texCrd.xy + maxIntersection.barycentric[1] * vertices[1].texCrd.xy + maxIntersection.barycentric[2] * vertices[2].texCrd.xy;

    // start/end point texture space point
    float3 minIntersectionTexSpaceCoord = float3(minIntersectionUV * displacementData.size, minIntersection.textureSpaceHeight);
    float3 maxIntersectionTexSpaceCoord = float3(maxIntersectionUV * displacementData.size, maxIntersection.textureSpaceHeight);
    float validStartT = max(tStart, minIntersection.t);
    float validEndT = maxIntersection.t;
    minIntersectionTexSpaceCoord = lerp( minIntersectionTexSpaceCoord, maxIntersectionTexSpaceCoord, (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t) );
    float3 startBarycentric = lerp( minIntersection.barycentric, maxIntersection.barycentric, (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t) );
    float3 endBarycentric = maxIntersection.barycentric;

    // 3. trace height map in texture space
    float intersectedT = 0.f;
    float intersectedHeight = 0.f;
    bool ret = true;
    // #if DISPLACEMENT_DEBUG_DISPLAY_SHELL == 0
    ret = traceSynthesizedHeightMapEstimated( displacementData, minIntersectionTexSpaceCoord, maxIntersectionTexSpaceCoord, intersectedT, intersectedHeight );
    if (ret)
    {
        const float3 traceSpaceViewDirection = normalize(maxIntersectionTexSpaceCoord - minIntersectionTexSpaceCoord);
        // get t max in trace space.
        const float tMax = solveT(minIntersectionTexSpaceCoord, traceSpaceViewDirection, maxIntersectionTexSpaceCoord);
        // intersectedT is in trace space, interpolate barycentric in there.
        float3 intersectionProjectedBarycentric = lerp(startBarycentric, endBarycentric, intersectedT / tMax);
        result.barycentrics = float2( intersectionProjectedBarycentric.y, 1.f - intersectionProjectedBarycentric.x - intersectionProjectedBarycentric.y ); // intersectionProjectedBarycentric;
        result.displacement = intersectedHeight;
        result.t = lerp(validStartT, validEndT, intersectedT / tMax);

        return result.t > ray.tMin && result.t < ray.tMax;
        // return ret;
    }
// #else

    // IntersectAttribute debugIntersection = minIntersection;
    // if (debugIntersection.intersected)
    // {
    //     // print("debugIntersection.intersected", debugIntersection.t);
    //     result.t = debugIntersection.t;
    //     result.barycentrics = float2(debugIntersection.barycentric.y, 1.f - debugIntersection.barycentric.x - debugIntersection.barycentric.y);
    //     result.displacement = 0.f;
    //     // return result.t > ray.tMin && result.t < ray.tMax;
    //     return true;
    // }
// #endif
    return false;
}
/** Intersect a displaced triangle.
    \param[in] ray Ray in world space.
    \param[in] vertices Triangle vertices in object space.
    \param[in] worldMat Triangle object to world transform.
    \param[in] displacementData Displacement data.
    \param[out] result Intersection result (only valid if triangle is intersected).
    \return Returns true if displaced triangle is intersected.
*/
bool intersectSynthesizedDisplacedTriangle( const Ray ray, StaticVertexData vertices[3], const float4x4 worldMat, const DisplacementData displacementData, out DisplacementIntersection result )
{
    result = {};

    float3 vertPosTr[3];
    vertPosTr[0] = mul(worldMat, float4(vertices[0].position, 1.f)).xyz;
    vertPosTr[1] = mul(worldMat, float4(vertices[1].position, 1.f)).xyz;
    vertPosTr[2] = mul(worldMat, float4(vertices[2].position, 1.f)).xyz;
    // Compute real-length normals.
    vertices[0].normal = mul(worldMat, float4(vertices[0].position + vertices[0].normal, 1.f)).xyz - vertPosTr[0];
    vertices[1].normal = mul(worldMat, float4(vertices[1].position + vertices[1].normal, 1.f)).xyz - vertPosTr[1];
    vertices[2].normal = mul(worldMat, float4(vertices[2].position + vertices[2].normal, 1.f)).xyz - vertPosTr[2];
    vertices[0].position = vertPosTr[0];
    vertices[1].position = vertPosTr[1];
    vertices[2].position = vertPosTr[2];
    bool valid = calcSynthesizedDisplacementIntersection(ray, vertices, displacementData, result);
    // Transform displacement into WS distance.
    if (!kDisplacementScalingUsePreciseLength)
    {
        result.displacement *= (length(vertices[0].normal) + length(vertices[1].normal) + length(vertices[2].normal)) / 3.f;
    }
    else
    {
        // Proper normal interpolation.
        const float3 barycentrics = result.getBarycentricWeights();
        float3 interpNormal = (vertices[0].normal * barycentrics[0] + vertices[1].normal * barycentrics[1] + vertices[2].normal * barycentrics[2]);
        result.displacement *= length(interpNormal);
    }
    return valid;
}


/** Helper for intersecting rays against displaced triangle meshes.
*/
struct SynthesizedDisplacedTriangleMeshIntersector
{
    /** Intersection attributes.
    */
    struct Attribs
    {
        float2 barycentrics;
        float displacement;
    };

    /** Intersects a ray with a displaced triangle.
        \param[in] ray Ray in world-space.
        \param[in] instanceID Geometry instance ID.
        \param[in] primitiveIndex Primitive index.
        \param[out] attribs Intersection attributes.
        \param[out] t Intersection t.
        \return True if the ray intersects the displaced triangle.
    */
    static bool intersect(const Ray ray, const GeometryInstanceID instanceID, const uint primitiveIndex, out Attribs attribs, out float t)
    {
        attribs = {};
        t = {};

        const uint materialID = gScene.getMaterialID(instanceID);
        const uint3 indices = gScene.getIndices(instanceID, primitiveIndex);
        const StaticVertexData vertices[3] = { gScene.getVertex(indices[0]), gScene.getVertex(indices[1]), gScene.getVertex(indices[2]) };
        const float4x4 worldMat = gScene.getWorldMatrix(instanceID);

        DisplacementData displacementData;
        gScene.materials.loadDisplacementData(materialID, displacementData);

        DisplacementIntersection result;
        if (intersectSynthesizedDisplacedTriangle(ray, vertices, worldMat, displacementData, result))
        {
            attribs.barycentrics = result.barycentrics;
            attribs.displacement = result.displacement;
            t = result.t;
            return true;
        }

        return false;
    }
};
