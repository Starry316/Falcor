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
import Utils.Math.FormatConversion;
import Utils.Debug.PixelDebug;
import Rendering.Lights.EnvMapSampler;
import Utils.Neural.NBTF;
import Scene.Displacement.DisplacementMapping;
import Utils.Texture.HistogramBlending;
import BTFUtils;
cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    float4 gControlParas;
    float4 gCurvatureParas; // d, r
    uint2 gRenderTargetDim;
    bool gApplySyn;
    bool gUseFP16;
    EnvMapSampler envMapSampler;
    bool gShowTracedHF;
    bool gTracedShadowRay;
    HistogramBlendingHFData hfData[2];
    float gPatchScale;
    uint2 gSelectedPixel;
    uint gMatId;
    bool gUseEditMap;
}
#define M_1_32 0.03125 // 1/32
#define M_1_16 0.0625  // 1/16


// Outputs
RWTexture2D<float4> gOutputColor;

Texture2D<float> gHF;
Texture2D<float4> gShellHF[2];
Texture2D<float4> gNormalMap;
Texture2D<float4> gTangentMap;
Texture2D<float4> gPosMap;
Texture2D<float4> gEditMap;

RWBuffer<uint4> packedInput;
RWBuffer<int> cudaVaildBuffer;
RWBuffer<float4> hashedUV;
RWBuffer<int> gSelectBuffer;

SamplerState gSampler;
SamplerState gMaxSampler;

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

#define FOOTPRINT_SCALING gControlParas.x
#define MAX_VIEW_THETA gControlParas.y* M_PI
#define HF_OFFSET gControlParas.z
#define HF_SCALING gControlParas.w

#define UV_SCALE gCurvatureParas.z
#define MAX_HEIGHT gCurvatureParas.x
float MISWeight(float p0, float p1)
{
    // return 0.5f;
    float q0 = p0 * p0;
    float q1 = p1 * p1;
    return q0 / max((q0 + q1), 0.0001f);
}
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
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.

    SampleGenerator sg; ///< Per-ray state for the sample generator (up to 16B).

    /**
     * Initializes ray payload with default parameters.
     */
    __init(SampleGenerator sg)
    {
        this.terminated = false;
        this.radiance = float3(0, 0, 0);
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
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
}

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

ShadingData loadShadingData(
    const HitInfo hit,
    const float3 rayOrigin,
    const float3 rayDir,
    out GeometryInstanceID instanceID,
    uint primitiveIndex
)
{
    VertexData v = {};
    uint materialID = {};

    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        instanceID = displacedTriangleHit.instanceID;
        primitiveIndex = displacedTriangleHit.primitiveIndex;
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
    }
    ShadingData sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);

    return sd;
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

bool generateEnvMapSample(float3 origin, inout SampleGenerator sg, inout float3 Li, inout float3 dir, inout float pdf)
{
    if (!kUseEnvLight)
        return false;

    EnvMapSample lightSample;
    if (!envMapSampler.sample(sampleNext2D(sg), lightSample))
        return false;
    if (gTracedShadowRay)
    {
        bool V = traceShadowRay(origin, lightSample.dir, kRayTMax);
        if (!V)
            return false;
    }

    Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
    pdf = lightSample.pdf;
    dir = lightSample.dir;

    return any(Li > 0.f);
}

bool generateEnvMapSamplePrism(inout SampleGenerator sg, inout float3 Li, inout float3 dir, inout float pdf)
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
    // BSDFSample bsdfSample;
    // if (mi.sample(sd, rayData.sg, bsdfSample, kUseImportanceSampling))
    // {
    //     rayData.origin = rayOrigin;
    //     if (bsdfSample.isLobe(LobeType::Transmission))
    //     {
    //         rayData.origin = sd.computeRayOrigin(false);
    //     }
    //     rayData.direction = bsdfSample.wo;
    //     rayData.thp *= bsdfSample.weight;
    //     rayData.BRDFPdf = bsdfSample.pdf;
    //     return any(rayData.thp > 0.f);
    // }

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

float4 tracePath(const uint2 pixel, const uint2 frameDim)
{
    // Create sample generator.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    const float3 primaryRayOrigin = gScene.camera.getPosition();

    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera, sampleNext2D(sg));
    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);

    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;


    traceScatterRay(rayData);
    // =================================================================================================
    // I used rayData.terminated to determine if the ray is terminated.
    // There can be a better way to do this.
    //cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] = rayData.terminated ? 0 : 1;
    cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] = rayData.terminated ? 0 : (gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]] + 1);

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

    uint2 pixel = DispatchRaysIndex().xy;

    // Add contribution from distant light (env map) in this direction.
    // if (kUseEnvLight && (kComputeDirect || rayData.pathLength == 0))
    if (kUseEnvLight)
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        // rayData.radiance += rayData.thp * Le * MISWeight(rayData.BRDFPdf, envMapSampler.evalPdf(WorldRayDirection()));
        rayData.radiance += Le;
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

bool traceSynthesizedHeightMapEstimated(
    float3 startPoint,
    float3 endPoint,
    float2 ddx,
    float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    uint2 pixel = DispatchRaysIndex().xy;

    uint coarseLength = 32;
    intersectedT = {};
    intersectedHeight = {};

    float3 viewDirection = endPoint - startPoint;

    uint width, height;
    gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].GetDimensions(width, height);

    // we assume we only have square a height field texture
    float pixelSizeInTexel = max(length(ddx) * width, length(ddy) * width) * UV_SCALE;
    pixelSizeInTexel = max(min(pixelSizeInTexel, width / float(coarseLength * 2)), 5.f);


    float shellSize = max(abs(viewDirection.x) * width, abs(viewDirection.y) * width) * UV_SCALE;

    int fineSteps = max(ceil(shellSize / pixelSizeInTexel * 2), 1);
    fineSteps = min(fineSteps, kRaymarchingMaxSampleCount);
    // fineSteps = 4;
    int coarseSteps = max(ceil(float(fineSteps) / coarseLength), 1);
    float3 incrementUnitCoarse, incrementUnitFine;
    if (fineSteps > coarseLength) {
        incrementUnitCoarse = viewDirection / coarseSteps;
        incrementUnitFine = incrementUnitCoarse / coarseLength;
        // incrementUnitFine * 1.0001;
    }
    else {
        incrementUnitFine = viewDirection / fineSteps;
    }


    viewDirection = normalize(viewDirection);
    const float tMax = solveT(startPoint, viewDirection, endPoint);
    float3 lastTestPoint;
    float3 currentTestPoint = startPoint;

    float lastHeightDelta = 0.0001;

    if (fineSteps <= coarseLength)
    {
        for (int i = 0; i < fineSteps; i++)
        {
            lastTestPoint = currentTestPoint;
            currentTestPoint += incrementUnitFine;
            float heightData;
            if (gApplySyn)
                heightData =
                    AutocovarianceBlendingDual(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], gMaxSampler, ddx, ddy, frac(abs(currentTestPoint.xy)) * UV_SCALE, gPatchScale) * HF_SCALING +
                    HF_OFFSET;
            else
                heightData = gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].SampleGrad(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;
            const float currentHeightDelta = currentTestPoint.z - heightData;
            if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) ||
                (sign(lastHeightDelta) != sign(currentHeightDelta)))
            {
                float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
                float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);
                intersectedT = tMax * (estimatedT / fineSteps);
                intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);
                return true;
            }
            lastHeightDelta = currentHeightDelta;
        }
        return false;
    }

    for (int i = 0; i < coarseSteps; i++)
    {
        lastTestPoint = currentTestPoint;
        currentTestPoint += incrementUnitCoarse;
        float heightData;
        if (gApplySyn)
            heightData = AutocovarianceBlendingDual(
                             hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], gMaxSampler, ddx * coarseLength, ddy * coarseLength, frac(abs(currentTestPoint.xy)) * UV_SCALE, gPatchScale
                         ) * HF_SCALING +
                         HF_OFFSET;
        else
            heightData =
                gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].SampleGrad(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, ddx * coarseLength, ddy * coarseLength).x *HF_SCALING +HF_OFFSET;
        const float currentHeightDelta = currentTestPoint.z - heightData;
        if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) ||
            (sign(lastHeightDelta) != sign(currentHeightDelta)))
        {
            // float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
            // float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);
            // intersectedT = tMax * (estimatedT / coarseSteps );
            // intersectedHeight = currentTestPoint.z;
            // return true;
            float3 lastTestPointFine;
            float3 currentTestPointFine = lastTestPoint;
            float lastHeightDeltaFine = 0.0001;
            // return true;
            for (int j = 0; j < coarseLength; j++)
            {
                lastTestPointFine = currentTestPointFine;
                currentTestPointFine += incrementUnitFine;
                float heightDataFine;
                if (gApplySyn)
                    heightDataFine =
                        AutocovarianceBlendingDual(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], gSampler, ddx / 2, ddy / 2, frac(abs(currentTestPointFine.xy)) * UV_SCALE, gPatchScale) * HF_SCALING +
                        HF_OFFSET;
                else
                    heightDataFine =
                        gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].SampleGrad(gMaxSampler, frac(abs(currentTestPointFine.xy)) * UV_SCALE, ddx / 2, ddy / 2).x * HF_SCALING + HF_OFFSET;

                const float currentHeightDeltaFine = currentTestPointFine.z - heightDataFine;
                if (((currentHeightDeltaFine <= 0.f) && (currentHeightDeltaFine > -kSurfaceThickness)) ||
                    (sign(lastHeightDeltaFine) != sign(currentHeightDeltaFine)))
                {
                    float heightDeltaRatio = abs(currentHeightDeltaFine) / (abs(currentHeightDeltaFine) + abs(lastHeightDeltaFine));
                    float estimatedTFine = lerp(float(j) + 1.f, float(j), heightDeltaRatio);
                    intersectedT = tMax * ((float(i)  + estimatedTFine / coarseLength) / coarseSteps);
                    intersectedHeight = lerp(currentTestPointFine.z, lastTestPointFine.z, heightDeltaRatio);
                    return true;
                }
                lastHeightDeltaFine = currentHeightDeltaFine;
            }
        }
        lastHeightDelta = currentHeightDelta;
    }
    return false;
}

bool traceSynthesizedHeightMapEstimatedNoMip(
    float3 startPoint,
    float3 endPoint,
    const float2 ddx,
    const float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    uint2 pixel = DispatchRaysIndex().xy;

    intersectedT = {};
    intersectedHeight = {};

    float3 viewDirection = endPoint - startPoint;
    uint width, height;
    gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].GetDimensions(width, height);

    // ray segement inside the shell
    float shellSize = max(abs(viewDirection.x) * width, abs(viewDirection.y) * width) * UV_SCALE;
    // float lod = log2(shellSize / 2);

    // tracing steps
    int maxCount = max(ceil(shellSize * 2), 1);
    maxCount = min(maxCount, kRaymarchingMaxSampleCount);


    float3 incrementUnit = viewDirection / maxCount;
    viewDirection = normalize(viewDirection);
    const float tMax = solveT(startPoint, viewDirection, endPoint);

    float3 lastTestPoint;
    float3 currentTestPoint = startPoint;
    float lastHeightDelta = 0.0001;

    for (int i = 0; i < maxCount; i++)
    {
        lastTestPoint = currentTestPoint;
        currentTestPoint += incrementUnit;
        float heightData;
        if (gApplySyn)
            heightData =
                AutocovarianceBlendingDual(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], gMaxSampler, 0, frac(abs(currentTestPoint.xy)) * UV_SCALE, gPatchScale) * HF_SCALING + HF_OFFSET;
        else
            heightData = gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].SampleLevel(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, 0).x * HF_SCALING + HF_OFFSET;
        const float currentHeightDelta = currentTestPoint.z - heightData;
        //
        if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) ||
            (sign(lastHeightDelta) != sign(currentHeightDelta)))
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

bool traceSynthesizedHeightMapEstimatedSimpleMip(
    float3 startPoint,
    float3 endPoint,
    const float2 ddx,
    const float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    uint2 pixel = DispatchRaysIndex().xy;

    intersectedT = {};
    intersectedHeight = {};

    float3 viewDirection = endPoint - startPoint;
    uint width, height;
    gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].GetDimensions(width, height);

    // ray segement inside the shell
    float lengthView = length(float2(abs(viewDirection.x) * width, abs(viewDirection.y) * height)) * UV_SCALE;
    float lod = log2(lengthView / 2);

    // camera ddx and ddy
    float lengthDdx = length(ddx) * width;
    float lengthDdy = length(ddy) * height;
    float maxGradient = max(lengthDdx, lengthDdy) * UV_SCALE;
    maxGradient = min(maxGradient, width / 4.0f);
    float minLod = min(log2(maxGradient), 7);

    // tracing steps
    int maxCount = ceil(lengthView / maxGradient);
    maxCount = min(maxCount * kRaymarchingSampleCountFactor, kRaymarchingMaxSampleCount);
    maxCount = max(maxCount, 1);

    float3 incrementUnit = viewDirection / maxCount;
    viewDirection = normalize(viewDirection);
    const float tMax = solveT(startPoint, viewDirection, endPoint);

    float3 lastTestPoint;
    float3 currentTestPoint = startPoint;
    float lastHeightDelta = 0.0001;

    for (int i = 0; i < maxCount; i++)
    {
        lastTestPoint = currentTestPoint;
        currentTestPoint += incrementUnit;
        float heightData;
        if (gApplySyn)

            heightData =
                AutocovarianceBlendingDual(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], gMaxSampler, minLod, frac(abs(currentTestPoint.xy)) * UV_SCALE, gPatchScale) * HF_SCALING + HF_OFFSET;
        else
            // heightData = gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].SampleGrad(gMaxSampler, frac(abs(currentTestPoint.xy))* UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;
            heightData = gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].SampleLevel(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, minLod).x * HF_SCALING + HF_OFFSET;
        const float currentHeightDelta = currentTestPoint.z - heightData;
        //
        // if (currentHeightDelta >= 0.f)
        if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) ||
            (sign(lastHeightDelta) != sign(currentHeightDelta)))
        {
            float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
            float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

            intersectedT = tMax * (estimatedT / maxCount);
            intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);
            // intersectedHeight = maxGradient;
            return true;
        }
        lastHeightDelta = currentHeightDelta;
    }

    return false;
}


bool calcSynthesizedDisplacementIntersection(
    const Ray ray,
    const StaticVertexData vertices[3],
    out DisplacementIntersection result
)
{
    uint2 pixel = DispatchRaysIndex().xy;
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
    rayTriangleIntersectionTest(extrudedP0, extrudedP1, extrudedP2, ray.origin, ray.dir, maxHeight, extrudedTriangleIntersectAttri);

    // min: entry point, max: exit point
    IntersectAttribute minIntersection = extrudedTriangleIntersectAttri;
    IntersectAttribute maxIntersection = extrudedTriangleIntersectAttri;

    // first slab/fin
    IntersectAttribute slabIntersectAttri0;
    IntersectAttribute slabIntersectAttri1;
    rayBilinearPatchIntersectionTest(
        intrudedP0,
        extrudedP0,
        intrudedP1,
        extrudedP1,
        float3(1, 0, 0),
        float3(0, 1, 0),
        ray.origin,
        ray.dir,
        ray.tMax,
        minHeight,
        maxHeight,
        slabIntersectAttri0,
        slabIntersectAttri1
    );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // // second slab/fin
    rayBilinearPatchIntersectionTest(
        intrudedP1,
        extrudedP1,
        intrudedP2,
        extrudedP2,
        float3(0, 1, 0),
        float3(0, 0, 1),
        ray.origin,
        ray.dir,
        ray.tMax,
        minHeight,
        maxHeight,
        slabIntersectAttri0,
        slabIntersectAttri1
    );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // // third slab/fin
    rayBilinearPatchIntersectionTest(
        intrudedP2,
        extrudedP2,
        intrudedP0,
        extrudedP0,
        float3(0, 0, 1),
        float3(1, 0, 0),
        ray.origin,
        ray.dir,
        ray.tMax,
        minHeight,
        maxHeight,
        slabIntersectAttri0,
        slabIntersectAttri1
    );
    minIntersection.min(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri0);
    maxIntersection.max(slabIntersectAttri1);

    // test base triangle lastly
    IntersectAttribute baseTriangleIntersectAttri;
    rayTriangleIntersectionTest(intrudedP0, intrudedP1, intrudedP2, ray.origin, ray.dir, minHeight, baseTriangleIntersectAttri);
    minIntersection.min(baseTriangleIntersectAttri);
    maxIntersection.max(baseTriangleIntersectAttri);

    // Early out when no hit.
    if (!(minIntersection.intersected && maxIntersection.intersected) || (minIntersection.t == maxIntersection.t) ||
        (minIntersection.t > ray.tMax) || (maxIntersection.t < ray.tMin))
    {
        return false;
    }

    // 2. convert intersection data from object space to trace space
    float2 minIntersectionUV = minIntersection.barycentric[0] * vertices[0].texCrd.xy +
                               minIntersection.barycentric[1] * vertices[1].texCrd.xy +
                               minIntersection.barycentric[2] * vertices[2].texCrd.xy;
    float2 maxIntersectionUV = maxIntersection.barycentric[0] * vertices[0].texCrd.xy +
                               maxIntersection.barycentric[1] * vertices[1].texCrd.xy +
                               maxIntersection.barycentric[2] * vertices[2].texCrd.xy;

    // start/end point texture space point
    uint width, height;
    gShellHF[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]].GetDimensions(width, height);
    float2 displacementSize = float2(width, height);
    float3 minIntersectionTexSpaceCoord = float3(minIntersectionUV * displacementSize, minIntersection.textureSpaceHeight);
    float3 maxIntersectionTexSpaceCoord = float3(maxIntersectionUV * displacementSize, maxIntersection.textureSpaceHeight);
    float validStartT = max(tStart, minIntersection.t);
    float validEndT = maxIntersection.t;
    minIntersectionTexSpaceCoord = lerp(
        minIntersectionTexSpaceCoord,
        maxIntersectionTexSpaceCoord,
        (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t)
    );
    float3 startBarycentric = lerp(
        minIntersection.barycentric,
        maxIntersection.barycentric,
        (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t)
    );
    float3 endBarycentric = maxIntersection.barycentric;

    // 3. trace height map in texture space

    // calculate ray differentials from the extruded triangle. Would it better to use the base triangle?
    float2 ddx, ddy;
    computeRayDifferentials(
        getGeometryInstanceID(),
        PrimitiveIndex(),
        ray.dir,
        extrudedTriangleIntersectAttri.t,
        gScene.camera,
        1.0f / float2(gRenderTargetDim),
        ddx,
        ddy
    );
    ddx *= FOOTPRINT_SCALING;
    ddy *= FOOTPRINT_SCALING;
    float intersectedT = 0.f;
    float intersectedHeight = 0.f;
    bool ret = true;
    float3 minUV = float3(minIntersectionUV, minIntersection.textureSpaceHeight);
    float3 maxUV = float3(maxIntersectionUV, maxIntersection.textureSpaceHeight);
    minUV = lerp(minUV, maxUV, (validStartT - minIntersection.t) / (maxIntersection.t - minIntersection.t));
#ifdef CONTACT_REFINEMENT
    ret = traceSynthesizedHeightMapEstimatedSimpleMip( minUV, maxUV, ddx, ddy, intersectedT, intersectedHeight);
#else
    ret = traceSynthesizedHeightMapEstimated(minUV, maxUV, ddx, ddy, intersectedT, intersectedHeight);
#endif
    if (ret)
    {
        const float3 traceSpaceViewDirection = normalize(maxIntersectionTexSpaceCoord - minIntersectionTexSpaceCoord);
        // get t max in trace space.
        // const float tMax = solveT(minIntersectionTexSpaceCoord, traceSpaceViewDirection, maxIntersectionTexSpaceCoord);
        const float tMax = solveT(minUV, normalize(maxUV - minUV), maxUV);
        // intersectedT is in trace space, interpolate barycentric in there.
        float3 intersectionProjectedBarycentric = lerp(startBarycentric, endBarycentric, intersectedT / tMax);
        result.barycentrics = float2(
            intersectionProjectedBarycentric.y, 1.f - intersectionProjectedBarycentric.x - intersectionProjectedBarycentric.y
        ); // intersectionProjectedBarycentric;
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
    \param[out] result Intersection result (only valid if triangle is intersected).
    \return Returns true if displaced triangle is intersected.
*/
bool intersectSynthesizedDisplacedTriangle(
    const Ray ray,
    StaticVertexData vertices[3],
    const float4x4 worldMat,

    out DisplacementIntersection result
)
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
    if (dot(ray.dir, vertices[0].normal) > 0 || dot(ray.dir, vertices[1].normal) > 0 || dot(ray.dir, vertices[2].normal) > 0)
    {
        return false;
    }


    vertices[0].position = vertPosTr[0];
    vertices[1].position = vertPosTr[1];
    vertices[2].position = vertPosTr[2];
    bool valid = calcSynthesizedDisplacementIntersection(ray, vertices,  result);
    // Transform displacement into WS distance.
    if (!kDisplacementScalingUsePreciseLength)
    {
        result.displacement *= (length(vertices[0].normal) + length(vertices[1].normal) + length(vertices[2].normal)) / 3.f;
    }
    else
    {
        // Proper normal interpolation.
        const float3 barycentrics = result.getBarycentricWeights();
        float3 interpNormal =
            (vertices[0].normal * barycentrics[0] + vertices[1].normal * barycentrics[1] + vertices[2].normal * barycentrics[2]);
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



        DisplacementIntersection result;

        if (intersectSynthesizedDisplacedTriangle(ray, vertices, worldMat, result))
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
    handleDisplacementHit(HitInfo(displacedTriangleHit), rayData);
}
[shader("intersection")]
void displacedTriangleMeshIntersectionShadow()
{
    float t;
    GeometryInstanceID instanceID = getGeometryInstanceID();
    const uint3 indices = gScene.getIndices(instanceID, PrimitiveIndex());
    const StaticVertexData vertices[3] = { gScene.getVertex(indices[0]), gScene.getVertex(indices[1]), gScene.getVertex(indices[2]) };
    const float4x4 worldMat = gScene.getWorldMatrix(instanceID);

    float3 vertPosTr[3];
    vertPosTr[0] = mul(worldMat, float4(vertices[0].position, 1.f)).xyz;
    vertPosTr[1] = mul(worldMat, float4(vertices[1].position, 1.f)).xyz;
    vertPosTr[2] = mul(worldMat, float4(vertices[2].position, 1.f)).xyz;
    float3 barycentrics;
    BuiltInTriangleIntersectionAttributes attribs;
    if (intersectRayTriangle(WorldRayOrigin(), WorldRayDirection(), vertPosTr, barycentrics, t))
    {
        ReportHit(t, 1, attribs);
    }
}

//
// Shader entry point for ray generation shader.
//
[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    // printSetPixel(pixel);


    float4 outputColor = tracePath(pixel, frameDim);
    // float4 outputColor = tracePathPrism(pixel, frameDim);
    gOutputColor[pixel] = outputColor;
}
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    // Add emitted light.
    // if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    if (kUseEmissiveLights )
    {
        // rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
        rayData.radiance += mi.getProperties(sd).emission;
    }

    // always true for common surface
    rayData.terminated = true;
    
    uint2 pixel = DispatchRaysIndex().xy;
    int id = 4 * (floor(frac(abs(sd.uv)).y * gRenderTargetDim.y) * gRenderTargetDim.x + floor(frac(abs(sd.uv)).x * gRenderTargetDim.x));
    gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = id;
    if(gUseEditMap)
    {
        uint width, height;
        gEditMap.GetDimensions(width, height);
        uint2 texel_map = uint2(floor(frac(abs(sd.uv)).x * width), floor(frac(abs(sd.uv)).y * height));

        gSelectBuffer[id] = gEditMap[texel_map].x;
    }
    else if(distance(pixel, gSelectedPixel) <= 100)
    {
        gSelectBuffer[id] = gMatId;
    }


    float3 wo = sd.frame.toLocal(sd.V);
    float3 envLi;
    float3 envDir;
    float envPdf;
    if (generateEnvMapSample(sd.computeRayOrigin(), rayData.sg, envLi, envDir, envPdf))
    {
        if (sd.materialID == 1)
        {
            float2 ddx, ddy;
            float hitT = length(sd.posW - gScene.camera.getPosition());
            computeRayDifferentials(hit.getTriangleHit(), -sd.V, hitT, gScene.camera, 1.0f / float2(gRenderTargetDim), ddx, ddy);
            ddx *= FOOTPRINT_SCALING;
            ddy *= FOOTPRINT_SCALING;


            float3 wi = sd.frame.toLocal(envDir);
            float NdotL = wi.z > 0 ? wi.z : 0;
            uint4 packedData;
            if (gApplySyn) {
                // float2 uv_raw = frac(abs(sd.uv)) * UV_SCALE;
                // float4 uv_hashed = getHashUV(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], uv_raw);
                // packedData = packInputSyn(wi, wo, frac(abs(uv_hashed.xy)), frac(abs(uv_hashed.wz)));
                packedData = packInput(wi, wo, frac(abs(sd.uv)), ddx, ddy);
                float2 uv_raw = frac(abs(sd.uv)) * UV_SCALE;
                float4 uv_hashed = getHashUV(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], uv_raw, gPatchScale);
                hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = uv_hashed.x;
                hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = uv_hashed.y;
                hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = uv_hashed.z;
                hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = uv_hashed.w;
            }
            else
                packedData = packInput(wi, wo, frac(abs(sd.uv)), ddx, ddy);

            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = packedData.x;
            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = packedData.y;
            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = packedData.z;
            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = packedData.w;


            rayData.radiance = envLi * NdotL;
            rayData.terminated = false;
            return;
        }
        else
            rayData.radiance += envLi * mi.eval(sd, envDir, rayData.sg);
    }


}

void handleDisplacementHit(const HitInfo hit, inout ScatterRayData rayData)
{
    
    // we use terminated to indicate that if the cuda inference is needed
    rayData.terminated = false;
    // Load shading data.
    GeometryInstanceID instanceID;
    uint primitiveIndex;

    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction, instanceID, primitiveIndex);
    float3 wo = sd.frame.toLocal(sd.V);
    float3 envLi;
    float3 envDir;
    float envPdf;
    float2 ddx, ddy;
    float hitT = length(sd.posW - gScene.camera.getPosition());
    computeRayDifferentials(instanceID, primitiveIndex, -sd.V, hitT, gScene.camera, 1.0f / float2(gRenderTargetDim), ddx, ddy);
    ddx *= FOOTPRINT_SCALING;
    ddy *= FOOTPRINT_SCALING;
    
    uint2 pixel = DispatchRaysIndex().xy;
    int id = 4 * (floor(frac(abs(sd.uv)).y * gRenderTargetDim.y) * gRenderTargetDim.x + floor(frac(abs(sd.uv)).x * gRenderTargetDim.x));
    gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = id;
    if(gUseEditMap)
    {
        uint width, height;
        gEditMap.GetDimensions(width, height);
        uint2 texel_map = uint2(floor(frac(abs(sd.uv)).x * width), floor(frac(abs(sd.uv)).y * height));

        gSelectBuffer[id] = gEditMap[texel_map].x;
    }
    else if(distance(pixel, gSelectedPixel) <= 100)
    {
        gSelectBuffer[id] = gMatId;
    }

    if (gShowTracedHF)
    {
        rayData.radiance = pow(hit.getDisplacedTriangleHit().displacement, 1 / 1.4);
        // rayData.radiance = hit.getDisplacedTriangleHit().displacement;
        // rayData.terminated = false;
        rayData.terminated = true;
        return;
    }

    rayData.radiance = float3(0, 0, 0);
    float3 newPos = sd.posW + sd.frame.N * gCurvatureParas.y;
    if (generateEnvMapSample(newPos, rayData.sg, envLi, envDir, envPdf))
    {
        float3 wi = sd.frame.toLocal(envDir);
        float NdotL = wi.z > 0 ? wi.z : 0;

        uint4 packedData;
        if (gApplySyn) {
            // float2 uv_raw = frac(abs(sd.uv)) * UV_SCALE;
            // float4 uv_hashed = getHashUV(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], uv_raw);
            // packedData = packInputSyn(wi, wo, frac(abs(uv_hashed.xy)), frac(abs(uv_hashed.wz)));
            packedData = packInput(wi, wo, frac(abs(sd.uv)), ddx, ddy);
            float2 uv_raw = frac(abs(sd.uv)) * UV_SCALE;
            float4 uv_hashed = getHashUV(hfData[gSelectBuffer[gSelectBuffer[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1]]], uv_raw, gPatchScale);
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = uv_hashed.x;
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = uv_hashed.y;
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = uv_hashed.z;
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = uv_hashed.w;

        }
        else
            packedData = packInput(wi, wo, frac(abs(sd.uv)), ddx, ddy);

        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = packedData.x;
        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = packedData.y;
        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = packedData.z;
        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = packedData.w;
        

        rayData.origin = newPos;
        rayData.radiance = envLi * NdotL;
    }

}

