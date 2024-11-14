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
import BTFSynthesis;
import Utils.Debug.PixelDebug;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EnvMapSampler;
import Utils.Neural.NBTF;
cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    float4 gControlParas;
    float4 gCurvatureParas; // d, r
    bool gApplySyn;
    bool gLocalFrame;
    bool gUseFP16;
    EnvMapSampler envMapSampler;
    NBTF nbtf;
}

// Inputs
Texture2D<PackedHitInfo> gVBuffer;
Texture2D<float4> gViewW; // Optional

// Outputs
RWTexture2D<float4> gOutputColor;
// Texture Inputs
Texture2D<float3> gColor;
Texture2D<float4> gNormalMap;
Texture2D<float4> gTangentMap;
Texture2D<float4> gPosMap;
Texture2D<float> gHF;
Texture2D<float> gHFMaxMip;

RWTexture2D<float4> wiWox;
RWTexture2D<float4> uvWoyz;
RWTexture2D<float4> dfDxy;
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
    rayData.radiance = 0.1;

    // let lod = ExplicitLodTextureSampler(0.f);

    // // Load shading data.
    // ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction, lod);

    // float3 exitPosW;
    // float3 HFColor = 0.0;
    // bool hitHF = true;
    // // bool hitHF = raySurfaceMarching(sd, hit, 1.f / float2(1920, 1080), exitPosW, HFColor);
    // if (hitHF) {
    //     rayData.radiance = HFColor;
    //     rayData.terminated = true;
    // }
    // rayData.radiance = 0;
    // rayData.direction = sd.V;

    // // Create material instance.
    // let mi = gScene.materials.getMaterialInstance(sd, lod);

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

float3 rayDirToLocal(float3 v, float2 uv)
{
    float4 tangentW = gTangentMap.SampleLevel(gSampler, uv, 0);
    float3 N = gNormalMap.SampleLevel(gSampler, uv, 0).xyz;
    float3 T = normalize(tangentW.xyz - N * dot(tangentW.xyz, N));
    float3 B = cross(N, T) * tangentW.w;
    return float3(dot(v, T), dot(v, B), dot(v, N));
}

float3 rayDirToLocal(float3 v, float2 uv, float3 N)
{
    float4 tangentW = gTangentMap.SampleLevel(gSampler, uv, 0);
    float3 T = normalize(tangentW.xyz - N * dot(tangentW.xyz, N));
    float3 B = cross(N, T) * tangentW.w;
    return float3(dot(v, T), dot(v, B), dot(v, N));
}



bool raySurfaceMarching(ShadingData sd, HitInfo hit, inout SampleGenerator sg, uint2 frameDim, uint2 pixel, out float3 exitPosW, out float3 outColor)
{
    // ==========================================
    // HF Tracing prism free
    // ==========================================
    float2 curUV = sd.uv;
    float3 deltaPosW = -sd.V;
    float3 wo = sd.frame.toLocal(sd.V);
    exitPosW = sd.posW;
    // Compute LoD
    float2 dfdx, dfdy;
    float hitT = length(sd.posW - gScene.camera.getPosition());
    computeRayDifferentials(hit.getTriangleHit(), -sd.V, hitT, gScene.camera, 1.0f / frameDim, dfdx, dfdy);
    dfdx *= gControlParas.y;
    dfdy *= gControlParas.y;
    float footprint = max(length(dfdx), length(dfdy)) * gControlParas.x;

    float finestStepLength = footprint;
    float coarseStepLength = 32 * finestStepLength;
    float maxFinestStep = 1.0f / finestStepLength;
    float maxCoarseStep = maxFinestStep / 32;

    // d is the max height from the surface
    float d = gCurvatureParas.x;
    float3 curPos = float3(curUV, d);

    bool flagFound = false;
#ifdef CONTACT_REFINEMENT
    float3 previousPos = curPos;
    for (uint coarseStep = 0; coarseStep < maxCoarseStep; coarseStep++)
    {
        float coarseHeight;
        if (gApplySyn)
            coarseHeight = HexTiling(gHFMaxMip, gSampler, dfdx, dfdy, (curPos.xy) * UV_SCALING) * HF_SCALING + HF_OFFSET;
        else
            coarseHeight = gHFMaxMip.SampleGrad(gSampler, (curPos.xy) * UV_SCALING, dfdx, dfdy) * HF_SCALING + HF_OFFSET;
        if ((coarseHeight + FLT_EPSILON >= curPos.z || curPos.z <= 0))
        {
            // find intersection in coarse level
            float3 curFinePos = previousPos;
            for (uint step = 0; step < 32; step++)
            {
                // fine level ray marching
                float height;
                if (gApplySyn)
                    height = HexTiling(gHF, gSampler, dfdx, dfdy, (curFinePos.xy) * UV_SCALING) * HF_SCALING + HF_OFFSET;
                else
                    height = gHF.SampleGrad(gSampler, (curFinePos.xy) * UV_SCALING, dfdx, dfdy) * HF_SCALING + HF_OFFSET;
                if ((height + FLT_EPSILON >= curFinePos.z || curFinePos.z <= 0))
                {
                    // find intersection
                    flagFound = true;
                    break;
                }
#ifdef HF_BOUND
                // else if (curFinePos.z >= d + FLT_EPSILON || any(curFinePos.xy < 0) || any(curFinePos.xy > 1))
                else if (curFinePos.z >= d + FLT_EPSILON || any(curFinePos.y < 0) || any(curFinePos.y > 1))
                {
                    // out of bounds
                    break;
                }
#endif
                else
                {
                    // keep marching
                    bool dummyB;
                    float4 tangentW = gTangentMap.SampleLevel(gSampler, curFinePos.xy, 0);
                    float3 N = gNormalMap.SampleLevel(gSampler, curFinePos.xy, 0).xyz;
                    ShadingFrame sf = ShadingFrame.createSafe(N, tangentW, dummyB);
                    float3 deltaPos = sf.toLocal(deltaPosW) * finestStepLength;
                    curFinePos += deltaPos;
                }
                curPos = curFinePos;
                previousPos = curPos;
            }
            if (flagFound)
                break;
        }
#ifdef HF_BOUND
        else if (curPos.z >= d + FLT_EPSILON || any(curPos.y < 0) || any(curPos.y > 1))
        {
            // out of bounds
            break;
        }
#endif
        else
        {
            // keep marching
            bool dummyB;
            float4 tangentW = gTangentMap.SampleLevel(gSampler, curPos.xy, 0);
            float3 N = gNormalMap.SampleLevel(gSampler, curPos.xy, 0).xyz;
            ShadingFrame sf = ShadingFrame.createSafe(N, tangentW, dummyB);
            float3 deltaPos = sf.toLocal(deltaPosW) * coarseStepLength;
            previousPos = curPos;
            curPos += deltaPos;
        }
    }
#else
    // raymarching
    for (uint step = 0; step < maxFinestStep; step++)
    {
        float height;
        if (gApplySyn)
            height = HexTiling(gHF, gSampler, dfdx, dfdy, (curPos.xy) * UV_SCALING) * HF_SCALING + HF_OFFSET;
        else
            height = gHF.SampleGrad(gSampler, (curPos.xy) * UV_SCALING, dfdx, dfdy) * HF_SCALING + HF_OFFSET;
        if ((height + FLT_EPSILON >= curPos.z || curPos.z <= 0))
        {
            // find intersection
            flagFound = true;
            break;
        }
#ifdef HF_BOUND
        else if (curPos.z >= d + FLT_EPSILON || any(curPos.y < 0) || any(curPos.y > 1))
        {
            // out of bounds
            break;
        }
#endif
        else
        {
            // keep marching
            bool dummyB;
            float4 tangentW = gTangentMap.SampleLevel(gSampler, curPos.xy, 0);
            float3 N = gNormalMap.SampleLevel(gSampler, curPos.xy, 0).xyz;
            ShadingFrame sf = ShadingFrame.createSafe(N, tangentW, dummyB);
            float3 deltaPos = sf.toLocal(deltaPosW) * finestStepLength;
            curPos += deltaPos;
        }
    }
#endif
    if (flagFound)
    {
#ifdef CUDA_INFER
        float3 envLi;
        float3 envDir;
        float envPdf;

        if (generateEnvMapSample(sg, envLi, envDir, envPdf))
        {
            bool dummyB;
            float4 tangentW = gTangentMap.SampleLevel(gSampler, curPos.xy, 0);
            float3 N = gNormalMap.SampleLevel(gSampler, curPos.xy, 0).xyz;
            ShadingFrame sf = ShadingFrame.createSafe(N, tangentW, dummyB);

            float3 wi = sf.toLocal(envDir);
            // if (!gLocalFrame) sf = sd.frame;
            float NdotL = wi.z > 0 ? 1 : 0;
            // outColor = envLi * NdotL *
            //            nbtf.evalDebug(wi, wo, (curPos.yx) * UV_SCALING, gCurvatureParas.y, gCurvatureParas.w, dfdy, dfdx, gApplySyn);
            outColor = envLi * NdotL;
            uint cudaInputIndex = 32 * (pixel.y * frameDim.x + pixel.x);
            cudaInputBuffer[pixel.y * frameDim.x + pixel.x + 32 * frameDim.x * frameDim.y] = 1;
            // if (gUseFP16)
                // nbtf.bindInputFP16( cudaInputBuffer, cudaInputIndex, wi, wo, (curPos.yx) * UV_SCALING, gCurvatureParas.y, gCurvatureParas.w, dfdy, dfdx, gApplySyn );
            // else
            nbtf.bindInput( cudaInputBuffer, cudaInputIndex, wi, wo, (curPos.yx) * UV_SCALING, gCurvatureParas.y, gCurvatureParas.w, dfdy, dfdx, gApplySyn );
        }
#elif defined(NN_INFER)
        float3 envLi;
        float3 envDir;
        float envPdf;

        if (generateEnvMapSample(sg, envLi, envDir, envPdf))
        {
            bool dummyB;
            float4 tangentW = gTangentMap.SampleLevel(gSampler, curPos.xy, 0);
            float3 N = gNormalMap.SampleLevel(gSampler, curPos.xy, 0).xyz;
            ShadingFrame sf = ShadingFrame.createSafe(N, tangentW, dummyB);
            float3 wi = sf.toLocal(envDir);
            float NdotL = wi.z > 0 ? 1 : 0;
            outColor = envLi * NdotL;
            wiWox[pixel] = float4(wi, wo.x);
            uvWoyz[pixel] = float4((curPos.yx) * UV_SCALING, wo.yz);
            dfDxy[pixel] = float4(dfdy, dfdx);
        }
#else
        if (gApplySyn)
            outColor = HexTiling(gColor, gSampler, dfdx, dfdy, (curPos.xy) * UV_SCALING);
        else
        {
            ShadingData curSD = sd;
            bool dummyB;
            float4 tangentW = gTangentMap.SampleLevel(gSampler, curPos.xy, 0);
            float3 N = gNormalMap.SampleLevel(gSampler, curPos.xy, 0).xyz;
            ShadingFrame sf = ShadingFrame.createSafe(N, tangentW, dummyB);
            curSD.frame = sf;
            curSD.posW = gPosMap.Sample(gSampler, curPos.xy).xyz;
            curSD.uv = curPos.xy * UV_SCALING;
            float3 envLi;
            float3 envDir;
            float envPdf;
            if (generateEnvMapSample(sg, envLi, envDir, envPdf))
            {
                let lod = ExplicitGradientTextureSampler(dfdx, dfdy);
                outColor = envLi * gScene.materials.getMaterialInstance(curSD, lod).eval(curSD, envDir, sg);
            }
            else
            {
                outColor = float3(0.0);
            }
        }

#endif
    }
#ifdef CUDA_INFER
    else
    {
        cudaInputBuffer[pixel.y * frameDim.x + pixel.x] = 0;
    }
#endif
    return flagFound;
}
/**
 * This is the main entry point for the minimal path tracer.
 *
 * One path per pixel is generated, which is traced into the scene.
 * The path tracer is written as a for-loop over path segments.
 *
 * Built-in light sources (point, directional) are sampled explicitly at each
 * path vertex. The contributions from area lights (env map and mesh lights)
 * are explicitly added by the scatter ray hit/miss shaders.
 *
 * @param[in] pixel Pixel to trace a path for.
 * @param[in] frameDim Dimension of the frame in pixels.
 * @return Returns the estimated color (radiance).
 */
float4 tracePath(const uint2 pixel, const uint2 frameDim)
{
    float4 outColor = float4(0.f);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);

    // let light = gScene.getLight(0);
    // float dummyT;

    // float3 lightPos = mul(light.transMat, float4(light.posW, 1.f)).xyz;
    // if (intersectRaySphere(primaryRayOrigin, primaryRayDir, lightPos, light.transMat._11, dummyT)) return float3(1, 1, 1);

    const HitInfo hit = HitInfo(gVBuffer[pixel]);
    if (hit.isValid())
    {
        // Pixel represents a valid primary hit. Compute its contribution.
        let lod = ExplicitLodTextureSampler(0.f);
        // Load shading data.
        ShadingData sd = loadShadingData(hit, primaryRayOrigin, primaryRayDir, lod);

        // Create material instance at shading point.
        let mi = gScene.materials.getMaterialInstance(sd, lod);

        // Create sample generator.
        SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

        // Advance the generator to the first available dimension.
        // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
        for (uint i = 0; i < gPRNGDimension; i++)
            sampleNext1D(sg);

        // Compute ray origin for new rays spawned from the G-buffer.

        float3 exitPosW;
        float3 HFColor;
        outColor = 0.1;
        bool hitHF = raySurfaceMarching(sd, hit, sg, frameDim, pixel, exitPosW, HFColor);
        if (hitHF)
        {
            outColor.xyz = HFColor;
            outColor.w = 1;
        }
        else
        {
            outColor.xyz = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
        }
        // ScatterRayData rayData = ScatterRayData(sg);
        // rayData.origin = exitPosW;
        // rayData.direction = sd.V;
        // rayData.terminated = hitHF;
        // for (uint depth = 0; depth < 1 && !rayData.terminated; depth++)
        // {
        //     // Trace scatter ray. If it hits geometry, the closest hit shader samples
        //     // direct illumination and generates the next scatter ray.
        //     traceScatterRay(rayData);
        // }
        // outColor += 0.8 * rayData.radiance;
        // if (kComputeDirect)
        // {
        //     // Always output directly emitted light, independent of whether emissive materials are treated as light sources or not.
        //     outColor += mi.getProperties(sd).emission;

        //     // Add contribution of direct light from analytic lights.
        //     // Light probe and mesh lights are handled by the scatter ray hit/miss shaders.
        //     outColor += kUseAnalyticLights ? evalDirectAnalytic(sd, mi, sg) : float3(0.f);
        // }

        // // Prepare ray payload.
        // ScatterRayData rayData = ScatterRayData(sg);

        // // Generate scatter ray.
        // if (!generateScatterRay(sd, mi, isCurveHit, rayOrigin, rayData))
        //     rayData.terminated = true;

        // // Follow path into the scene and compute its total contribution.
        // for (uint depth = 0; depth <= kMaxBounces && !rayData.terminated; depth++)
        // {
        //     // Trace scatter ray. If it hits geometry, the closest hit shader samples
        //     // direct illumination and generates the next scatter ray.
        //     traceScatterRay(rayData);
        // }

        // Store contribution from scatter ray.
        // outColor += rayData.radiance;
    }
    else
    {
        // Background pixel.
        outColor.xyz = kUseEnvBackground ? gScene.envMap.eval(primaryRayDir) : kDefaultBackgroundColor;
    }

    return outColor;
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
    gOutputColor[pixel]  = tracePath(pixel, frameDim);

    // gOutputColor[pixel] = float4(color, 1.f);
}
