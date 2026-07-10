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
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
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
import Utils.Geometry.GeometryHelpers;
import Utils.Geometry.IntersectionHelpers;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapSampler;
cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    EnvMapSampler gEnvMapSampler;
    float gViewTheta;
    float gViewPhi;
    float gViewHeight;
    float gViewSize;
    float gViewHeightBot;
    bool gBTFViewMode;
    bool gPluckerMode;
    bool gShowSelectedTri;
    uint gSelectedInstanceID;
    uint gSelectedTriangleID;
    float2 gTriSampleUV;
    uint gMaxBounces;
    float3 gProbePos;
    float3 gSurfaceNormal;
    float4 gXYUV;

}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gBarycentric;
RWTexture2D<float4> gWi;
RWTexture2D<float4> gRadiance;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)

#define pX  gXYUV.x
#define pY  gXYUV.y
#define pU  gXYUV.z
#define pV  gXYUV.w


// static const uint kMaxBounces = MAX_BOUNCES;
static const bool kComputeDirect = COMPUTE_DIRECT;
static const bool kUseImportanceSampling = USE_IMPORTANCE_SAMPLING;
static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const bool kUseEnvBackground = USE_ENV_BACKGROUND;
static const float3 kDefaultBackgroundColor = float3(0, 0, 0);
static const float kRayTMax = FLT_MAX;

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
 * @return ShadingData struct.
 */
ShadingData loadShadingData(const HitInfo hit, const float3 rayOrigin, const float3 rayDir)
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
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
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
    if (is_valid(gViewW))
    {
        // If we have the view vector bound as a buffer, just fetch it. No need to compute anything.
        return -gViewW[launchIndex].xyz;
    }
    else
    {
        // Compute the view vector. This must exactly match what the G-buffer pass is doing (jitter etc.).
        // Note that we do not take depth-of-field into account as it would require exactly matching the
        // sample generator between the passes, which is error prone. The host side will issue a warning instead.
        return camera.computeRayPinhole(launchIndex, launchDim).dir;
    }
}

/**
 * Returns the primary ray's direction.
 */
float3 getPrimaryRayDir(uint2 launchIndex, uint2 launchDim, const Camera camera, float2 jitter)
{
    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (launchIndex + jitter) / launchDim;

    float2 ndc = float2(2, -2) * p + float2(-1, 1);

    // Compute the non-normalized ray direction assuming a pinhole camera.
    return normalize(ndc.x * camera.data.cameraU + ndc.y * camera.data.cameraV + camera.data.cameraW);

    // return camera.computeRayPinhole(launchIndex, launchDim).dir;
}
// Simple orthographic primary ray generator.
// Assumptions:
// - World up is +Y.
// - Camera always looks at the origin (0,0,0).
// - Camera position lies on a sphere centered at the origin.
// - Ortho view half-size (radius) = 1 in world units.
// - jitter is in *pixel units* (e.g., [-0.5,0.5]).

void getPrimaryOrthogonalRayPos(
    uint2 launchIndex,
    uint2 launchDim,
    float3 cameraPos,
    float2 jitter,
    inout float3 rayDir,
    inout float3 rayPos )
{
    // Forward: from camera to origin
    float3 fwd = normalize(-cameraPos);

    // Stable Y-up basis
    float3 worldUp = float3(0.0f, 1.0f, 0.0f);

    // If looking almost straight up/down, switch to a different up to avoid degeneracy
    if (abs(dot(fwd, worldUp)) > 0.99999999f) worldUp = float3(0.0f, 0.0f, 1.0f);

    float3 right = normalize(cross(worldUp, fwd));
    float3 up = cross(fwd, right); // already normalized if fwd/right are normalized

    // Pixel -> NDC in [-1,1], origin top-left
    float2 p = (float2(launchIndex) + jitter) / float2(launchDim); // [0,1]
    float2 ndc = float2(p.x * 2.0f - 1.0f, 1.0f - p.y * 2.0f);

    // Ortho half-size (view radius) = 1
    const float halfSize = 1.0f;

    // Ray: constant direction, origin slides on the image plane
    rayDir = fwd;
    rayPos = cameraPos + right * (ndc.x * halfSize) + up * (ndc.y * halfSize);
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
float3 shiftBackFacingShadingNormal(float3 N, float3 V) {

    // Fix back-facing
    float NdotV = dot(N, V);
    N += V * max(0.0, -NdotV + 1e-5);
    N = normalize(N);
    return N;
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
    // float NdotL = dot(sd.frame.N, ls.dir);
    // NdotL = max(NdotL, 1e-4);

    if ((NdotL <= kMinCosTheta && !hasTransmission) || (NdotL >= -kMinCosTheta && !hasReflection))
        return float3(0.f);

    // Get origin with offset applied in direction of the geometry normal to avoid self-intersection.
    const float3 origin = computeRayOrigin(sd.posW, dot(sd.faceN, ls.dir) >= 0.f ? sd.faceN : -sd.faceN);

    // Test visibility by tracing a shadow ray.
    bool V = traceShadowRay(origin, ls.dir, ls.distance);
    if (!V)
        return float3(0.f);



    // sd.frame.N = shiftBackFacingShadingNormal(sd.frame.N, sd.V);

    // Evaluate contribution.
    return mi.eval(sd, ls.dir, sg) * ls.Li * invPdf;
    // return 0.8 / 3.1415926 * ls.Li * invPdf;
}

/**
 * Generate a new scatter ray or terminate.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in] isCurveHit True if on curve hit.
 * @param[in] rayOrigin Ray origin for the new ray.
 * @param[in,out] rayData Ray payload.
 * @return True if the path continues.
 */
bool generateScatterRay(const ShadingData sd, const IMaterialInstance mi, bool isCurveHit, float3 rayOrigin, inout ScatterRayData rayData)
{
    // Sample material.
    BSDFSample bsdfSample;
    if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    {
        rayData.origin = rayOrigin;
        if (!isCurveHit && bsdfSample.isLobe(LobeType::Transmission))
        {
            rayData.origin = sd.computeRayOrigin(false);
        }
        rayData.direction = bsdfSample.wo;
        rayData.thp *= bsdfSample.weight;
        return any(rayData.thp > 0.f);
    }

    return false;
}

bool generateEnvMapSample(float3 origin, inout SampleGenerator sg, inout float3 Li, inout float3 dir, inout float pdf)
{
    // if (!kUseEnvLight)
    //     return false;

    EnvMapSample lightSample;
    if (!gEnvMapSampler.sample(sampleNext2D(sg), lightSample))
        return false;

    if (!traceShadowRay(origin, lightSample.dir, kRayTMax))
        return false;

    Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    pdf = lightSample.pdf;
    dir = lightSample.dir;

    return any(Li > 0.f);
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
    const bool isCurveHit = hit.getType() == HitType::Curve;

    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod, MaterialInstanceHints::AdjustShadingNormal);

    // Debug: highlight the user-selected triangle on the primary hit.
    if (gShowSelectedTri && rayData.pathLength == 1 && hit.getType() == HitType::Triangle)
    {
        const TriangleHit th = hit.getTriangleHit();
        if (th.instanceID.index == gSelectedInstanceID && th.primitiveIndex == gSelectedTriangleID)
        {
            uint2 pixel = DispatchRaysIndex().xy;
            gBarycentric[pixel] = float4(1 - th.barycentrics.x- th.barycentrics.y, th.barycentrics.x, th.barycentrics.y, 0.f);
            float3 wi = reflect(-sd.V, sd.frame.N);
            gWi[pixel] = float4(wi.xzy, 0.f);
            gRadiance[pixel] = float4(mi.eval(sd, wi, rayData.sg), 1.f);

            rayData.radiance = float3(th.barycentrics.x, th.barycentrics.y, 0.f);
            rayData.terminated = true;
            return;
        }
    }



    // Add emitted light.
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
    }


    // Compute ray origin for new rays spawned from the hit.
    float3 rayOrigin;
    rayOrigin = sd.computeRayOrigin();
    // rayOrigin = sd.posW + sd.faceN * 0.0001f;

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
    }
    // if (kUseEnvLight )
    // {
    //     float3 envLi;
    //     float3 envDir;
    //     float envPdf;
    //     if (generateEnvMapSample(rayOrigin, rayData.sg, envLi, envDir, envPdf)) {
    //         if (dot(sd.getOrientedFaceNormal(), envDir) > kMinCosTheta)
    //             rayData.radiance += mi.eval(sd, envDir, rayData.sg) * rayData.thp * envLi;
    //     }
    // }

    // Generate scatter ray for the next path segment.
    // The raygen shader will continue the path based on the returned payload.
    if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
    {
        rayData.terminated = true;
        return;
    }

    //
    rayData.pathLength++;
}

ScatterRayData traceFullSpherePosPrimaryRay(const uint2 pixel, const uint2 frameDim, SampleGenerator sg) {
    float2 originUV = ((float2)pixel + sampleNext2D(sg)) / (float2)frameDim;

    // ndir_to_oct_equal_area_unorm
    float3 primaryRayOrigin;
    float3 primaryRayDir;
    float3 cameraPos = spherical_to_cartesian_rad(float2(gViewTheta * M_PI, gViewPhi * M_2PI)).xzy * 5;

    // getPrimaryOrthogonalRayPos(pixel, frameDim, cameraPos, sampleNext2D(sg), primaryRayDir, primaryRayOrigin);
    getPrimaryOrthogonalRayPos(pixel, frameDim, cameraPos, 0.5, primaryRayDir, primaryRayOrigin);


    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    rayData.pathLength = 1;
    traceScatterRay(rayData);
    return rayData;
}

ScatterRayData traceLightProbePrimaryRay(const uint2 pixel, const uint2 frameDim, SampleGenerator sg) {
    float2 originUV = ((float2)pixel + sampleNext2D(sg)) / (float2)frameDim;

    // ndir_to_oct_equal_area_unorm
    // Look up the selected triangle's geometry directly from the scene.
    GeometryInstanceID selectedInstance;
    selectedInstance.index = gSelectedInstanceID;
    float3 selectedTriPos[3];
    gScene.getVertexPositionsW(selectedInstance, gSelectedTriangleID, selectedTriPos);
    float3 selectedTriNormal = gScene.getFaceNormalW(selectedInstance, gSelectedTriangleID);
    // Uniformly sample a position on the selected triangle using the UI-provided sample coordinates.
    float3 bary = sample_triangle(gTriSampleUV);
    float3 sampledPos = bary.x * selectedTriPos[0] + bary.y * selectedTriPos[1] + bary.z * selectedTriPos[2];
    float3 primaryRayOrigin = computeRayOrigin(sampledPos, selectedTriNormal);
    // float3 primaryRayOrigin = sampledPos;
    float3 primaryRayDir;
    // float3 cameraPos = gProbePos;
    // spherical_to_cartesian_rad(float2(gViewTheta * M_PI, gViewPhi * M_2PI)).xzy * 5;
    primaryRayDir = oct_to_ndir_equal_area_unorm(originUV).xzy;

    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    rayData.pathLength = 1;

    // // Lower hemisphere
    // if (dot(selectedTriNormal, primaryRayDir) <= kMinCosTheta) {
    //     rayData.terminated = true;
    //     rayData.radiance = float3(0.0, 0.0, 0.0);
    //     return rayData;
    // }

    traceScatterRay(rayData);
    return rayData;
}


float3 tracePath(const uint2 pixel, const uint2 frameDim)
{

    // Create sample generator.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    // Prepare ray payload.
    ScatterRayData rayData;
    // ScatterRayData rayData = tracePrimaryRay(pixel, frameDim, sg);
    if (!gBTFViewMode)
        rayData = traceLightProbePrimaryRay(pixel, frameDim, sg);
        // rayData = traceBTFPrimaryRay(pixel, frameDim, sg);
        // if (gPluckerMode)
        //     rayData = traceFullSpherePosPrimaryRay(pixel, frameDim, sg);
        // else
        //     rayData = traceFullSpherePrimaryRay(pixel, frameDim, sg);


    else
        rayData = tracePrimaryRay(pixel, frameDim, sg);



    for (uint bounce = 0; bounce < gMaxBounces; bounce++) {
        if (rayData.terminated) break;
        traceScatterRay(rayData);
    }

    return rayData.radiance;
}

float3 fromLocal(const float3 v, const float3 T, const float3 B, const float3 N) {
    return T * v.x + B * v.y + N * v.z;
}


ScatterRayData traceFullSpherePrimaryRay(const uint2 pixel, const uint2 frameDim, SampleGenerator sg) {
    float2 originUV = ((float2)pixel + sampleNext2D(sg)) / (float2)frameDim;
    // float2 originUV = ((float2)pixel + 0.5f) / (float2)frameDim;

    // ndir_to_oct_equal_area_unorm
    float3 primaryRayOrigin;

    // float3 sphericalDir = squareToDiskHemisphereShirley(originUV).xzy;
    float3 sphericalDir = oct_to_ndir_equal_area_unorm(originUV).xzy;
    float R = gViewHeight;

    primaryRayOrigin = sphericalDir * R;

    float3 primaryRayDir;
    primaryRayDir = spherical_to_cartesian_rad(float2(gViewTheta * M_PI, gViewPhi * M_2PI));

    primaryRayDir = -primaryRayDir.xzy;

    // float3 T = perp_stark(sphericalDir);
    // float3 B = cross(sphericalDir, T);
    // primaryRayDir = -fromLocal(primaryRayDir, T, B, sphericalDir);


    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    rayData.pathLength = 1;
    traceScatterRay(rayData);
    return rayData;
}
// ScatterRayData traceFullSpherePrimaryRay(const uint2 pixel, const uint2 frameDim, SampleGenerator sg) {
//         float2 originUV = ((float2)pixel + sampleNext2D(sg)) / (float2)frameDim;
//         originUV = 2 * originUV - 1;
//         // ndir_to_oct_equal_area_unorm
//         float3 primaryRayOrigin;

//         primaryRayOrigin = float3(originUV.x * gViewSize, gViewHeight, originUV.y * gViewSize);

//         float3 primaryRayDir;
//         primaryRayDir = spherical_to_cartesian_rad(float2(gViewTheta * M_PI, gViewPhi * M_2PI));

//         primaryRayDir = -primaryRayDir.xzy;

//         // float3 T = perp_stark(sphericalDir);
//         // float3 B = cross(sphericalDir, T);
//         // primaryRayDir = -fromLocal(primaryRayDir, T, B, sphericalDir);

//         // Prepare ray payload.
//         ScatterRayData rayData = ScatterRayData(sg);
//         rayData.origin = primaryRayOrigin;
//         rayData.direction = primaryRayDir;
//         rayData.pathLength = 1;
//         traceScatterRay(rayData);
//         return rayData;
// }

ScatterRayData traceBTFPrimaryRay(const uint2 pixel, const uint2 frameDim, SampleGenerator sg) {
    float2 originUV = ((float2)pixel + sampleNext2D(sg)) / (float2)frameDim;
    // float2 originUV = ((float2)pixel + 0.5) / (float2)frameDim;
    // originUV = 1 - frac(originUV);
    originUV = 2 * originUV - 1;
    originUV = originUV * 0.5;

    float3 primaryRayOrigin;
    float3 primaryRayDir;

    // if (gPluckerMode) {
    //     float3 posXY = float3(originUV.x, gViewHeight, (2 * pY - 1) );
    //     float3 posUV = float3(originUV.y, gViewHeightBot, (2 * pV - 1));
    //     primaryRayOrigin = posXY;
    //     primaryRayDir = normalizeSafe(posUV - posXY);
    // }
    if (gPluckerMode) {
        // float3 posUV = float3((2 * pV - 1), gViewHeightBot, (2 * pY - 1));
        // float3 posXY = float3(originUV.y, gViewHeight, originUV.x);
        float2 targetXY = float2(gViewTheta, gViewPhi);
        // TO -0.5, 0.5
        targetXY = 2 * targetXY - 1;

        float3 posUV = float3(gViewSize * originUV.x, gViewHeight, gViewSize * originUV.y);
        float3 posXY = float3(gViewSize * targetXY.x, gViewHeightBot, gViewSize * targetXY.y);
        primaryRayOrigin = posUV;
        primaryRayDir = normalizeSafe(posXY - posUV);
    }

    else {
        primaryRayOrigin = float3(gViewSize * originUV.x, gViewHeight, gViewSize * originUV.y);
        primaryRayDir = spherical_to_cartesian_rad(float2(gViewTheta * M_PI_2, gViewPhi * M_2PI));
        primaryRayDir = primaryRayDir.xzy;
        // primaryRayDir.y = -primaryRayDir.y;
        primaryRayDir = -primaryRayDir;
    }




    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    rayData.pathLength = 1;
    traceScatterRay(rayData);
    return rayData;
}


ScatterRayData tracePrimaryRay(const uint2 pixel, const uint2 frameDim, SampleGenerator sg) {
    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera, sampleNext2D(sg));

    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);
    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;
    rayData.pathLength = 1;
    traceScatterRay(rayData);
    return rayData;
}


//
// Shader entry points for miss shaders.
//

[shader("miss")]
void scatterMiss(inout ScatterRayData rayData)
{
    // Ray missed the scene. Mark the ray as terminated.
    rayData.terminated = true;
    // if (rayData.pathLength == 1) {
    //     rayData.radiance = 0;
    //     if (gShowOffset) {

    //         // float t;
    //         // intersectRaySphere(rayData.origin, rayData.direction, float3(0), 1.1, t, 0.f, kRayTMax, false);
    //         rayData.radiance = 0;
    //     }
    //     return;
    // }

    // Add contribution from distant light (env map) in this direction.
    if (kUseEnvLight) {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += rayData.thp * Le;
    }
    // if (kUseEnvLight && rayData.pathLength == 1)
    // if (kUseEnvLight && (kComputeDirect || rayData.pathLength > 0))
    // {
    //     float3 Le = gScene.envMap.eval(WorldRayDirection());
    //     rayData.radiance += rayData.thp * Le;
    // }




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

//
// Shader entry points for DisplacedTriangleMesh hit groups.
//

[shader("intersection")]
void displacedTriangleMeshIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    DisplacedTriangleMeshIntersector::Attribs attribs;
    float t;
    if (DisplacedTriangleMeshIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterDisplacedTriangleMeshClosestHit(inout ScatterRayData rayData, DisplacedTriangleMeshIntersector::Attribs attribs)
{
    DisplacedTriangleHit displacedTriangleHit;
    displacedTriangleHit.instanceID = getGeometryInstanceID();
    displacedTriangleHit.primitiveIndex = PrimitiveIndex();
    displacedTriangleHit.barycentrics = attribs.barycentrics;
    displacedTriangleHit.displacement = attribs.displacement;
    handleHit(HitInfo(displacedTriangleHit), rayData);
}

//
// Shader entry points for Curve hit groups.
//

[shader("intersection")]
void curveIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    CurveIntersector::Attribs attribs;
    float t;
    if (CurveIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), attribs, t))
    {
        ReportHit(t, 0, attribs);
    }
}

[shader("closesthit")]
void scatterCurveClosestHit(inout ScatterRayData rayData, CurveIntersector::Attribs attribs)
{
    CurveHit curveHit;
    curveHit.instanceID = getGeometryInstanceID();
    curveHit.primitiveIndex = PrimitiveIndex();
    curveHit.barycentrics = attribs.barycentrics;
    handleHit(HitInfo(curveHit), rayData);
}

//
// Shader entry points for SDFGrid hit groups.
//

[shader("intersection")]
void sdfGridIntersection()
{
    const Ray ray = Ray(WorldRayOrigin(), WorldRayDirection(), RayTMin(), RayTCurrent());
    SDFGridHitData sdfGridHitData;
    float t;
    if (SDFGridIntersector::intersect(ray, getGeometryInstanceID(), PrimitiveIndex(), sdfGridHitData, t))
    {
        ReportHit(t, 0, sdfGridHitData);
    }
}

[shader("closesthit")]
void scatterSdfGridClosestHit(inout ScatterRayData rayData, SDFGridHitData sdfGridHitData)
{
    SDFGridHit sdfGridHit;
    sdfGridHit.instanceID = getGeometryInstanceID();
    sdfGridHit.hitData = sdfGridHitData;
    handleHit(HitInfo(sdfGridHit), rayData);
}

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;

    float3 color = tracePath(pixel, frameDim);

    gOutputColor[pixel] = float4(color, 1.f);
}
