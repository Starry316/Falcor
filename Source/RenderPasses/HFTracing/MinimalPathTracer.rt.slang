/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,w
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

/**
 * Minimal path tracer.
 *
 * The purpose is to use it for validation of more complex renderers.
 * The implementation here should be kept as simple/naive as possible.
 *
 * At each hit point (including the primary hit loaded from the V-buffer),
 * analytic light sources (point, directional) are sampled uniformly using
 * 1 shadow ray, and 1 scatter ray is traced to sample the hemisphere.
 * At hit/miss the scatter ray includes light from emissive surface and
 * the environment map, respectively. Traversal stops at a fixed path length.
 *
 * Each type of light (analytic, emissive, env map) can be individually
 * enabled/disabled from the host. This clutters the code a bit, but it is
 * important as not all other renderes may support all three light types.
 *
 * The host sets the following defines:
 *
 * MAX_BOUNCES             Maximum number of indirect bounces (0 means no indirect).
 * COMPUTE_DIRECT          Nonzero if direct illumination should be included.
 * USE_IMPORTANCE_SAMPLING Nonzero if importance sampling should be used for sampling materials.
 * USE_ANALYTIC_LIGHTS     Nonzero if Falcor's analytic lights should be used.
 * USE_EMISSIVE_LIGHTS     Nonzero if emissive geometry should be used as lights.
 * USE_ENV_LIGHT           Nonzero if env map is available and should be used as light source.
 * USE_ENV_BACKGROUND      Nonzero if env map is available and should be used as background.
 * is_valid_<name>         1 if optional I/O buffer with this name should be used.
 */

#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
import Scene.Raytracing;
import Scene.Intersection;
import Utils.Math.MathHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Sampling.SampleGenerator;
import Utils.Texture.HexTiling;
import Rendering.Lights.LightHelpers;

import Utils.Debug.PixelDebug;
import Rendering.Lights.EnvMapSampler;
import Utils.Neural.NBTF;
import Scene.Displacement.DisplacementMapping;
// import Tracing.MyIntersection;
cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    float4 gControlParas;
    float4 gCurvatureParas; // d, r
    float2 gInvFrameDim;
    bool gApplySyn;
    bool gUseFP16;
    EnvMapSampler envMapSampler;
    NBTF nbtf;
    uint gDebugPrism;
    bool gTraceShell;
}

// Outputs
RWTexture2D<float4> gOutputColor;
// Texture Inputs
Texture2D<float3> gColor;

Texture2D<float> gHF;
Texture2D<float> gShellHF;
Texture2D<float> gHFMaxMip;

RWTexture2D<float4> wiWox;
RWTexture2D<float4> uvWoyz;
RWTexture2D<float4> ddxy;
RWBuffer<float> cudaInputBuffer;


SamplerState gSampler;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

#define MAX_VIEW_THETA gControlParas.y* M_PI
#define HF_OFFSET gControlParas.z
#define HF_SCALING gControlParas.w

#define UV_SCALING gCurvatureParas.z * 10
#define MAX_HEIGHT gCurvatureParas.x

/**
 * Payload for shadow ray.
 */
struct ShadowRayData
{
    bool visible;
};

/**
 * Payload for scatter ray (up to 72B).
 */
struct ScatterRayData
{
    float3 radiance;  ///< Accumulated outgoing radiance from path.
    bool terminated;  ///< Set to true when path is terminated.
    float3 thp;       ///< Current path throughput. This is updated at each path vertex.
    uint pathLength;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.pathLength = 0;
        this.radiance = float3(0, 0, 0);
        this.thp = float3(1, 1, 1);
        this.origin = float3(0, 0, 0);
        this.direction = float3(0, 0, 0);
        this.sg = sg;
    }
};

/**
 * Setup ShadingData based on loaded vertex/material attributes for a hit point.
 * @param[in] hit Hit information.
 * @param[in] rayOrigin Ray origin.
 * @param[in] rayDir Normalized ray direction.
 * @param[in] lod Method for computing texture level-of-detail.
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, const ITextureSampler lod)
{
    VertexData v = {};
    uint materialID = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
#endif
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera)
{
    return camera.computeRayPinhole(launchIndex, launchDim).dir;
}

/**
 * Traces a shadow ray towards a light source.
 * @param[in] origin Ray origin for the shadow ray.
 * @param[in] dir Direction from shading point towards the light source (normalized).
 * @param[in] distance Distance to the light source.
 * @return True if light is visible, false otherwise.
 */
bool traceShadowRay(float3 origin, float3 dir, float distance)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.f;
    ray.TMax = distance;

    ShadowRayData rayData;
    rayData.visible = false; // Set to true by miss shader if ray is not terminated before
    TraceRay(
        gScene.rtAccel,
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH,
        0xff /* instanceInclusionMask */,
        1 /* hitIdx */,
        rayTypeCount,
        1 /* missIdx */,
        ray,
        rayData
    );

    return rayData.visible;
}

/**
 * Traces a scatter ray based on ray parameters stored in the ray payload.
 * @param[in] rayData Describes the ray parameters. The struct is modified based on the result.
 */
void traceScatterRay(inout ScatterRayData rayData)
{
    RayDesc ray;
    ray.Origin = rayData.origin;
    ray.Direction = rayData.direction;
    ray.TMin = 0.f;
    ray.TMax = kRayTMax;

    uint rayFlags = 0; // TODO: Set cull mode from the app
    TraceRay(gScene.rtAccel, rayFlags, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, rayTypeCount, 0 /* missIdx */, ray, rayData);
}

bool generateEnvMapSample(inout SampleGenerator sg, inout float3 Li, inout float3 dir, inout float pdf)
{
    if (!kUseEnvLight)
        return false;

    EnvMapSample lightSample;
    if (!envMapSampler.sample(sampleNext2D(sg), lightSample))
        return false;

    Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    pdf = lightSample.pdf;
    dir = lightSample.dir;

    return any(Li > 0.f);
}

/**
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectAnalytic(const ShadingData sd, const IMaterialInstance mi, inout SampleGenerator sg)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return float3(0.f);

    // Pick one of the analytic light sources randomly with equal probability.
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);
    float invPdf = lightCount; // Light selection pdf = 1.0 / lightCount.

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(sd.posW, gScene.getLight(lightIndex), sg, ls))
        return float3(0.f);

    // Reject sample if not in the hemisphere of a BSDF lobe.
    const uint lobeTypes = mi.getLobeTypes(sd);
    const bool hasReflection = lobeTypes & uint(LobeType::Reflection);
    const bool hasTransmission = lobeTypes & uint(LobeType::Transmission);
    float NdotL = dot(sd.getOrientedFaceNormal(), ls.dir);
    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, ls.dir, ls.distance);
    if (!V)
        return float3(0.f);

    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
}

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.thp *= bsdfSample.weight;
        return any(rayData.thp > 0.f);
    }

    return false;
}
void computeRayDifferentials(
    const TriangleHit hit,
    float3 rayDir,
    float hitT,
    const Camera camera,
    float2 invFrameDim,
    out float2 ddx,
    out float2 ddy
)
{
    // TODO: Is this code correct for instance transforms that flip the handedness of the coordinate system?

    // Ray differentials
    float3 P[3];
    gScene.getVertexPositionsW(hit.instanceID, hit.primitiveIndex, P);
    float3 e1 = P[1] - P[0];
    float3 e2 = P[2] - P[0];
    float3 d = rayDir;
    float k = dot(cross(e1, e2), d);
    k = abs(k) > 1e-20f ? rcp(k) : 0.0f;
    float3 cu = cross(e2, d);
    float3 cv = cross(d, e1);
    // Assumes a normalized ray direction
    // dDdx in ray gen
    float3 dx = camera.data.cameraU * 2.f * invFrameDim.x / camera.data.focalDistance;
    // dDdy in ray gen
    float3 dy = camera.data.cameraV * 2.f * invFrameDim.y / camera.data.focalDistance;
    // Transfer to primary hit
    float3 q = dx * hitT;
    float3 r = dy * hitT;
    float dudx = k * dot(cu, q);
    float dudy = k * dot(cu, r);
    float dvdx = k * dot(cv, q);
    float dvdy = k * dot(cv, r);
    float2 T[3];
    gScene.getVertexTexCoords(hit.instanceID, hit.primitiveIndex, T);
    float2 g1 = T[1] - T[0];
    float2 g2 = T[2] - T[0];
    float dsdx = (dudx * g1.x + dvdx * g2.x);
    float dsdy = (dudy * g1.x + dvdy * g2.x);
    float dtdx = (dudx * g1.y + dvdx * g2.y);
    float dtdy = (dudy * g1.y + dvdy * g2.y);
    ddx = float2(dsdx, dtdx);
    ddy = float2(dsdy, dtdy);
}
void computeRayDifferentials(
    const GeometryInstanceID instanceID,
    const uint primitiveIndex,
    float3 rayDir,
    float hitT,
    const Camera camera,
    float2 invFrameDim,
    out float2 ddx,
    out float2 ddy
)
{
    // TODO: Is this code correct for instance transforms that flip the handedness of the coordinate system?

    // Ray differentials
    float3 P[3];
    gScene.getVertexPositionsW(instanceID, primitiveIndex, P);
    float3 e1 = P[1] - P[0];
    float3 e2 = P[2] - P[0];
    float3 d = rayDir;
    float k = dot(cross(e1, e2), d);
    k = abs(k) > 1e-20f ? rcp(k) : 0.0f;
    float3 cu = cross(e2, d);
    float3 cv = cross(d, e1);
    // Assumes a normalized ray direction
    // dDdx in ray gen
    float3 dx = camera.data.cameraU * 2.f * invFrameDim.x / camera.data.focalDistance;
    // dDdy in ray gen
    float3 dy = camera.data.cameraV * 2.f * invFrameDim.y / camera.data.focalDistance;
    // Transfer to primary hit
    float3 q = dx * hitT;
    float3 r = dy * hitT;
    float dudx = k * dot(cu, q);
    float dudy = k * dot(cu, r);
    float dvdx = k * dot(cv, q);
    float dvdy = k * dot(cv, r);
    float2 T[3];
    gScene.getVertexTexCoords(instanceID, primitiveIndex, T);
    float2 g1 = T[1] - T[0];
    float2 g2 = T[2] - T[0];
    float dsdx = (dudx * g1.x + dvdx * g2.x);
    float dsdy = (dudy * g1.x + dvdy * g2.x);
    float dtdx = (dudx * g1.y + dvdx * g2.y);
    float dtdy = (dudy * g1.y + dvdy * g2.y);
    ddx = float2(dsdx, dtdx);
    ddy = float2(dsdy, dtdy);
}

float4 traceDispPath(const uint2 pixel, const uint2 frameDim)
{
    float4 outColor = float4(0.f);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);
    ScatterRayData rayData;
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    rayData.terminated = false;
    rayData.radiance = float3(0.0f);
    rayData.thp = float3(1);
    rayData.pathLength = 1;
    traceScatterRay(rayData);
    // for (uint depth = 0; depth < kMaxBounces && !rayData.terminated; depth++)
    // {
    //     // Trace scatter ray. If it hits geometry, the closest hit shader samples
    //     // direct illumination and generates the next scatter ray.

    // }

    return float4(rayData.radiance, 1.0f);
}
//
// Shader entry points for miss shaders.
//
[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.thp * Le;
    }
}

[shader("miss")]
void shadowMiss(inout ShadowRayData rayData)
{
    // The miss shader is executed if the ray misses all geometry. Mark as visible.
    rayData.visible = true;
}

//
// Shader entry points for TriangleMesh hit groups.
//
[shader("anyhit")]
void scatterTriangleMeshAnyHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}

[shader("closesthit")]
void scatterTriangleMeshClosestHit(inout ScatterRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    TriangleHit triangleHit;
    triangleHit.instanceID = getGeometryInstanceID();
    triangleHit.primitiveIndex = PrimitiveIndex();
    triangleHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(triangleHit), rayData);
}

[shader("anyhit")]
void shadowTriangleMeshAnyHit(inout ShadowRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    // Alpha test for non-opaque geometry.
    GeometryInstanceID instanceID = getGeometryInstanceID();
    VertexData v = getVertexData(instanceID, PrimitiveIndex(), attribs);
    const uint materialID = gScene.getMaterialID(instanceID);
    if (gScene.materials.alphaTest(v, materialID, 0.f))
        IgnoreHit();
}
bool traceSynthesizedHeightMapEstimated( const DisplacementData displacementData, float3 startPoint, float3 endPoint, float2 ddx, float2 ddy, out float intersectedT, out float intersectedHeight )
{
    intersectedT = {};
    intersectedHeight = {};
    ddx = 0.0001;
    ddy = 0.0001;
    // const float entryHeightData = displacementData.texture.SampleLevel(displacementData.samplerState, startPoint.xy * UV_SCALING, 0).x * HF_SCALING + HF_OFFSET;
    //  const float
    float entryHeightData;
    if (gApplySyn)
        entryHeightData = HexTiling(gShellHF, displacementData.samplerState, ddx, ddy, startPoint.xy * UV_SCALING).x * HF_SCALING + HF_OFFSET;
    else
        entryHeightData = gShellHF.SampleLevel(displacementData.samplerState, startPoint.xy * UV_SCALING, 0).x * HF_SCALING + HF_OFFSET;

    if (startPoint.z < entryHeightData)
    {
        intersectedT = 0.000001f; // This seems to make the intersection ignored. TODO: experiment with eg. 0.000001
        intersectedHeight = startPoint.z;
        return true;
    }

    float3 viewDirection = endPoint - startPoint;
    int maxCount = ceil(abs(viewDirection.x)) + ceil(abs(viewDirection.y)) + 2;
    maxCount = min(maxCount * kRaymarchingSampleCountFactor, kRaymarchingMaxSampleCount); // Prevent freeze due to corner cases. Todo: Investigate why this happens.
    // maxCount = 200;
    float3 incrementUnit = viewDirection / maxCount;
    viewDirection = normalize(viewDirection);
    const float tMax = solveT(startPoint, viewDirection, endPoint);

    float3 lastTestPoint;
    float3 currentTestPoint = startPoint;
    float lastHeightDelta = currentTestPoint.z - entryHeightData;

    for (int i = 0; i < maxCount; i++)
    {
        lastTestPoint = currentTestPoint;
        currentTestPoint += incrementUnit;

        // const float heightData = displacementData.texture.SampleLevel(displacementData.samplerState, currentTestPoint.xy * UV_SCALING, 0).x * HF_SCALING + HF_OFFSET;
        float heightData;

        if (gApplySyn)
            heightData = HexTiling(gShellHF, displacementData.samplerState, ddx, ddy, currentTestPoint.xy * UV_SCALING).x * HF_SCALING + HF_OFFSET;
        else
            heightData = gShellHF.SampleLevel(displacementData.samplerState, currentTestPoint.xy * UV_SCALING, 0).x * HF_SCALING + HF_OFFSET;
        const float currentHeightDelta = currentTestPoint.z - heightData;

        if (currentHeightDelta < 0.f)
        // if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) || (sign(lastHeightDelta) != sign(currentHeightDelta)))
        {
            float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
            float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

            intersectedT = tMax * (estimatedT / maxCount);
            intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);
            return true;
        }
        lastHeightDelta = currentHeightDelta;
    }

    return false;
}
bool calcSynthesizedDisplacementIntersection( const Ray ray, const StaticVertexData vertices[3], const DisplacementData displacementData, const float2 ddx, const float2 ddy, out DisplacementIntersection result )
{
    result = {};

    const float2 shellMinMax = float2(0.f, MAX_HEIGHT);
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

    // // second slab/fin
    rayBilinearPatchIntersectionTest( intrudedP1, extrudedP1, intrudedP2, extrudedP2, float3(0, 1, 0), float3(0, 0, 1),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1 );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // // third slab/fin
    rayBilinearPatchIntersectionTest( intrudedP2, extrudedP2, intrudedP0, extrudedP0, float3(0, 0, 1), float3(1, 0, 0),
        ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri0, slabIntersectAttri1 );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // IntersectAttribute slabIntersectAttri2;
    // IntersectAttribute slabIntersectAttri3;
    // // // second slab/fin
    // rayBilinearPatchIntersectionTest( intrudedP1, extrudedP1, intrudedP2, extrudedP2, float3(0, 1, 0), float3(0, 0, 1),
    //     ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri2, slabIntersectAttri3 );
    // minIntersection.min(slabIntersectAttri2);
    // maxIntersection.max(slabIntersectAttri2);
    // maxIntersection.max(slabIntersectAttri3);

    // IntersectAttribute slabIntersectAttri4;
    // IntersectAttribute slabIntersectAttri5;
    // // // third slab/fin
    // rayBilinearPatchIntersectionTest( intrudedP2, extrudedP2, intrudedP0, extrudedP0, float3(0, 0, 1), float3(1, 0, 0),
    //     ray.origin, ray.dir, ray.tMax, minHeight, maxHeight, slabIntersectAttri4, slabIntersectAttri5 );
    // minIntersection.min(slabIntersectAttri4);
    // maxIntersection.max(slabIntersectAttri4);
    // maxIntersection.max(slabIntersectAttri5);


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

    // IntersectAttribute debugIntersection = extrudedTriangleIntersectAttri;
    // if (gDebugPrism == 1) debugIntersection = baseTriangleIntersectAttri;
    // else if (gDebugPrism == 2) debugIntersection = slabIntersectAttri0;
    // else if (gDebugPrism == 3) debugIntersection = slabIntersectAttri2;
    // else if (gDebugPrism == 4) debugIntersection = slabIntersectAttri4;
    // else if (gDebugPrism == 5) debugIntersection = minIntersection;
    // else if (gDebugPrism == 6) debugIntersection = maxIntersection;
    // bool borderlineIntersection = false;
    // if (debugIntersection.intersected && gDebugPrism <7)
    // {
    //     // print("debugIntersection.intersected", debugIntersection.t);
    //     result.t = debugIntersection.t;
    //     result.barycentrics = float2(debugIntersection.barycentric.y, 1.f - debugIntersection.barycentric.x - debugIntersection.barycentric.y);

    //     float minBary = min(min(debugIntersection.barycentric.x, debugIntersection.barycentric.y), debugIntersection.barycentric.z);
    //     float intensity = 3 * minBary;

    //     result.displacement = intensity;
    //     // return result.t > ray.tMin && result.t < ray.tMax;
    //     // return true;
    //     borderlineIntersection = true;
    // }



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

    float3 minUV = float3(minIntersectionUV, minIntersection.textureSpaceHeight);
    float3 maxUV = float3(maxIntersectionUV, maxIntersection.textureSpaceHeight);
    minUV = lerp(minUV, maxUV, (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t));
    ret = traceSynthesizedHeightMapEstimated(displacementData, minUV, maxUV, ddx, ddy, intersectedT, intersectedHeight);

    if (ret)
    {
        const float3 traceSpaceViewDirection = normalize(maxIntersectionTexSpaceCoord - minIntersectionTexSpaceCoord);
        // get t max in trace space.
        // const float tMax = solveT(minIntersectionTexSpaceCoord, traceSpaceViewDirection, maxIntersectionTexSpaceCoord);
        const float tMax = solveT(minUV, normalize(maxUV - minUV) , maxUV);
        // intersectedT is in trace space, interpolate barycentric in there.
        float3 intersectionProjectedBarycentric = lerp(startBarycentric, endBarycentric, intersectedT / tMax);
        result.barycentrics = float2( intersectionProjectedBarycentric.y, 1.f - intersectionProjectedBarycentric.x - intersectionProjectedBarycentric.y ); // intersectionProjectedBarycentric;
        result.displacement = intersectedHeight;
        result.t = lerp(validStartT, validEndT, intersectedT / tMax);
        // result.barycentrics = float2(hitUV.y, 1.f - hitUV.x - hitUV.y);;
        return result.t > ray.tMin && result.t < ray.tMax;
        // return ret;
    }

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
bool intersectSynthesizedDisplacedTriangle( const Ray ray, StaticVertexData vertices[3], const float4x4 worldMat, const DisplacementData displacementData, const float2 ddx, const float2 ddy, out DisplacementIntersection result )
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

    // vertices[0].normal = mul(worldMat, float4(vertices[0].normal, 1.f)).xyz;
    // vertices[1].normal = mul(worldMat, float4(vertices[1].normal, 1.f)).xyz;
    // vertices[2].normal = mul(worldMat, float4(vertices[2].normal, 1.f)).xyz;

    vertices[0].position = vertPosTr[0];
    vertices[1].position = vertPosTr[1];
    vertices[2].position = vertPosTr[2];
    bool valid = calcSynthesizedDisplacementIntersection(ray, vertices, displacementData,ddx, ddy, result);
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
    static bool intersect( const Ray ray, const GeometryInstanceID instanceID, const uint primitiveIndex, out Attribs attribs, out float t)
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

        float2 ddx, ddy;
        computeRayDifferentials(instanceID, primitiveIndex, ray.dir, ray.tMax, gScene.camera, gInvFrameDim, ddx, ddy);
        print("ddx", ddx);
        print("ddy", ddy);
        if (intersectSynthesizedDisplacedTriangle(ray, vertices, worldMat, displacementData, ddx, ddy, result))
        {
            attribs.barycentrics = result.barycentrics;
            attribs.displacement = result.displacement;
            t = result.t;
            return true;
        }

        return false;
    }
};

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    float t;

    SynthesizedDisplacedTriangleMeshIntersector::Attribs attribs;
    if (SynthesizedDisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, SynthesizedDisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    printSetPixel(pixel);
    // float3 color = tracePath(pixel, frameDim);
    // gOutputColor[pixel]  = tracePath(pixel, frameDim);
    gOutputColor[pixel] = traceDispPath(pixel, frameDim);
    // gOutputColor[pixel] = float4(color, 1.f);
}
/**
 * Process a hit.
 * Loads the shading data, samples analytic lights and samples a new scatter ray.
 * Terminates the path if maximum number of bounces is reached.
 * @param[in] hit Hit info.
 * @param[in,out] rayData Ray payload.
 *
 */
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    let lod = ExplicitLodTextureSampler(0.f);

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction, lod);
    float3 wo = sd.frame.toLocal(sd.V);
    float3 envLi;
    float3 envDir;
    float envPdf;
    rayData.radiance = pow(hit.getDisplacedTriangleHit().displacement, 1 / 1.4);
    if (gTraceShell) {
        // if (gApplySyn)
        //     rayData.radiance = HexTiling(gColor, gSampler, float2(0.0001), float2(0.0001), UV_SCALING * (sd.uv));
        // else
        //     rayData.radiance = gColor.SampleLevel(gSampler, sd.uv * UV_SCALING, 0);

        if (generateEnvMapSample(rayData.sg, envLi, envDir, envPdf))
        {
            float3 wi = sd.frame.toLocal(envDir);
            float NdotL = wi.z > 0 ? wi.z : 0;
            rayData.radiance = envLi * NdotL *
                nbtf.evalDebug(wi, wo, (sd.uv.yx) * UV_SCALING, gCurvatureParas.y, gCurvatureParas.w, 0.0001, 0.0001, gApplySyn);
        }
    }






        // rayData.radiance = float3(sd.uv, 0) ;
    rayData.terminated = false;

    // // Add emitted light.
    // if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    // {
    //     rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
    // }

    // // Check whether to terminate based on max depth.
    // if (rayData.pathLength >= kMaxBounces)
    // {
    //     rayData.terminated = true;
    //     return;
    // }

    // // Compute ray origin for new rays spawned from the hit.
    // float3 rayOrigin;
    // rayOrigin = sd.computeRayOrigin();
    // // Add contribution of direct light from analytic lights.
    // if (kUseAnalyticLights)
    // {
    //     float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
    //     rayData.radiance += rayData.thp * Lr;
    // }

    // // Generate scatter ray for the next path segment.
    // // The raygen shader will continue the path based on the returned payload.
    // if (!generateScatterRay(sd, mi, rayOrigin, rayData))
    // {
    //     rayData.terminated = true;
    //     return;
    // }

    // rayData.pathLength++;
}

