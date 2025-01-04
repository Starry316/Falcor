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
    uint gDebugPrism;
    bool gShowTracedHF;
    bool gTracedShadowRay;
    HistogramBlendingHFData hfData;
}
#define M_1_32 0.03125 // 1/32
#define M_1_16 0.0625  // 1/16


// Outputs
RWTexture2D<float4> gOutputColor;

Texture2D<float> gHF;
Texture2D<float4> gShellHF;
Texture2D<float> gHFMaxMip;
Texture2D<float4> gNormalMap;
Texture2D<float4> gTangentMap;
Texture2D<float4> gPosMap;

RWBuffer<uint4> packedInput;
RWBuffer<int> cudaVaildBuffer;
RWBuffer<float4> hashedUV;

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
    float3 thp;       ///< Current path throughput. and pixel position.
    uint pathLength;  ///< Path length in number of path segments (0 at origin, 1 at first secondary hit, etc.). Max 2^31.
    float3 origin;    ///< Next path segment origin.
    float3 direction; ///< Next path segment direction.
    float BRDFPdf;    ///< PDF of the last BSDF sample. Used for MIS.

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
        this.BRDFPdf = 1.0f;
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

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        instanceID = triangleHit.instanceID;
        primitiveIndex = triangleHit.primitiveIndex;
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        instanceID = displacedTriangleHit.instanceID;
        primitiveIndex = displacedTriangleHit.primitiveIndex;
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
        rayData.BRDFPdf = bsdfSample.pdf;
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

float4 tracePath(const uint2 pixel, const uint2 frameDim)
{
    // Create sample generator.
    SampleGenerator sg = SampleGenerator(pixel, gFrameCount);

    // Advance the generator to the first available dimension.
    // TODO: This is potentially expensive. We may want to store/restore the state from memory if it becomes a problem.
    for (uint i = 0; i < gPRNGDimension; i++)
        sampleNext1D(sg);

    const float3 primaryRayOrigin = gScene.camera.getPosition();
    const float3 primaryRayDir = getPrimaryRayDir(pixel, frameDim, gScene.camera);
    // Prepare ray payload.
    ScatterRayData rayData = ScatterRayData(sg);

    rayData.origin = primaryRayOrigin;
    rayData.direction = primaryRayDir;

    for (uint depth = 0; depth < 1 && !rayData.terminated; depth++)
    {
        traceScatterRay(rayData);
    }
    // =================================================================================================
    // I used rayData.terminated to determine if the ray is terminated.
    // There can be a better way to do this.
    cudaVaildBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] = rayData.terminated ? 0 : 1;

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
    if (kUseEnvLight && (kComputeDirect || rayData.pathLength == 0))
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        // rayData.radiance += rayData.thp * Le * MISWeight(rayData.BRDFPdf, envMapSampler.evalPdf(WorldRayDirection()));
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

bool traceSynthesizedHeightMapEstimated(
    const DisplacementData displacementData,
    float3 startPoint,
    float3 endPoint,
    const float2 ddx,
    const float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    uint coarseLength =16;
    intersectedT = {};
    intersectedHeight = {};

    float3 viewDirection = endPoint - startPoint;

    uint width, height;
    gShellHF.GetDimensions(width, height);

    // camera ddx and ddy
    float maxGradient = max(length(ddx), length(ddx));
    float lod = log2(maxGradient * UV_SCALE);


    float shellSize = max(abs(viewDirection.x) * width, abs(viewDirection.y) * width) * UV_SCALE;
    int fineSteps = max(ceil(shellSize * 2), 1);
    fineSteps = min(fineSteps, kRaymarchingMaxSampleCount);
    int coarseSteps = max(ceil(float(fineSteps) / coarseLength), 1);
    float3 incrementUnitCoarse, incrementUnitFine;
    if (fineSteps > coarseLength) {
        incrementUnitCoarse = viewDirection / coarseSteps;
        incrementUnitFine = incrementUnitCoarse / coarseLength;
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
                    AutocovarianceBlendingDual(hfData, gMaxSampler, ddx, ddy, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING +
                    HF_OFFSET;
            else
                heightData = gShellHF.SampleGrad(gSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;
            const float currentHeightDelta = currentTestPoint.z - heightData;
            //
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
                             hfData, gMaxSampler, ddx * coarseLength / 2, ddy * coarseLength / 2, frac(abs(currentTestPoint.xy)) * UV_SCALE
                         ) * HF_SCALING +
                         HF_OFFSET;
        else
            heightData =
                gShellHF.SampleGrad(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, ddx * coarseLength / 2, ddy * coarseLength / 2).x *
                    HF_SCALING +
                HF_OFFSET;
        const float currentHeightDelta = currentTestPoint.z - heightData;
        if (currentHeightDelta <= 0.f)
        {
            // trace fine level
            float3 lastTestPointFine;
            float3 currentTestPointFine = lastTestPoint;
            lastHeightDelta = 0.0001;

            for (int j = 0; j < fineSteps; j++)
            {
                lastTestPointFine = currentTestPointFine;
                currentTestPointFine += incrementUnitFine;
                float heightDataFine;
                if (gApplySyn)
                    heightDataFine =
                        AutocovarianceBlendingDual(hfData, gSampler, ddx / 2, ddy / 2, frac(abs(currentTestPointFine.xy)) * UV_SCALE) * HF_SCALING +
                        HF_OFFSET;
                else
                    heightDataFine =
                        gShellHF.SampleGrad(gMaxSampler, frac(abs(currentTestPointFine.xy)) * UV_SCALE, ddx / 2, ddy / 2).x * HF_SCALING + HF_OFFSET;

                const float currentHeightDeltaFine = currentTestPointFine.z - heightDataFine;
                if (((currentHeightDeltaFine <= 0.f) && (currentHeightDeltaFine > -kSurfaceThickness)) ||
                    (sign(lastHeightDelta) != sign(currentHeightDeltaFine)))
                {
                    float heightDeltaRatio = abs(currentHeightDeltaFine) / (abs(currentHeightDeltaFine) + abs(lastHeightDelta));
                    float estimatedT = lerp(float(j) + 1.f, float(j), heightDeltaRatio);
                    // intersectedT = tMax * (float(i) / coarseSteps + (float(estimatedT) / fineSteps) / coarseLength);

                    intersectedT = tMax * (float(i) / coarseSteps + (float(j) / fineSteps) / coarseLength);

                    // intersectedHeight = lerp(currentTestPointFine.z, lastTestPointFine.z, heightDeltaRatio);
                    intersectedHeight = currentTestPointFine.z;
                    return true;
                }
                lastHeightDelta = currentHeightDeltaFine;
            }
        }

    }
    // for (int i = 0; i < coarseSteps; i++)
    // {
    //     lastTestPoint = currentTestPoint;
    //     currentTestPoint += incrementUnitCoarse;
    //     float heightData;
    //     if (gApplySyn)
    //         heightData =
    //             AutocovarianceBlendingDual(hfData, gMaxSampler, ddx, ddy, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING +
    //             HF_OFFSET;
    //     else
    //         heightData = gShellHF.SampleGrad(gSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;
    //     const float currentHeightDelta = currentTestPoint.z - heightData;
    //     //
    //     if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) ||
    //         (sign(lastHeightDelta) != sign(currentHeightDelta)))
    //     {
    //         float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
    //         float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);
    //         intersectedT = tMax * (estimatedT / fineSteps);
    //         intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);
    //         return true;
    //     }
    //     lastHeightDelta = currentHeightDelta;
    // }

    // for (int i = 0; i < fineSteps; i++)
    // {
    //     lastTestPoint = currentTestPoint;
    //     currentTestPoint += incrementUnitFine;
    //     float heightData;
    //     if (gApplySyn)
    //         heightData =
    //             AutocovarianceBlendingDual(hfData, gMaxSampler, ddx, ddy, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING +
    //             HF_OFFSET;
    //     else
    //         heightData = gShellHF.SampleGrad(gSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;
    //     const float currentHeightDelta = currentTestPoint.z - heightData;
    //     //
    //     if (((currentHeightDelta <= 0.f) && (currentHeightDelta > -kSurfaceThickness)) ||
    //         (sign(lastHeightDelta) != sign(currentHeightDelta)))
    //     {
    //         float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
    //         float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);
    //         intersectedT = tMax * (estimatedT / fineSteps);
    //         intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);
    //         return true;
    //     }
    //     lastHeightDelta = currentHeightDelta;
    // }

    // float fineFootprintPixel = ceil(min(fineStepTS * UV_SCALE, width * 0.25f / coarseLength));
    // int fineSteps = coarseLength;

    // float coarseFootprintPixel = fineFootprintPixel * coarseLength;
    // int coarseSteps = ceil(min(width / coarseFootprintPixel, kRaymarchingMaxSampleCount));

    // float coarseLod = log2(coarseFootprintPixel);
    // float fineLod = log2(fineFootprintPixel);

    // float3 incrementUnitFine = fineStepTS * viewDirection;
    // float3 incrementUnitCoarse = incrementUnitFine * coarseLength;

    // float lastHeightDelta = 0.0001;
    // for (int i = 0; i < coarseSteps && i * (fineStepTS * coarseLength) <= tMax; i++)
    // {
    //     lastTestPoint = currentTestPoint;
    //     currentTestPoint += incrementUnitCoarse;
    //     float heightData;
    //     if (gApplySyn)
    //         heightData =
    //             AutocovarianceBlendingDual(hfData, gMaxSampler, coarseLod, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING +
    //             HF_OFFSET;
    //     else
    //         heightData = gShellHF.SampleLevel(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, coarseLod).x * HF_SCALING +
    //         HF_OFFSET;
    //     const float currentHeightDelta = currentTestPoint.z - heightData;
    //     if ((currentHeightDelta <= 0.f))
    //     {
    //         // trace fine level
    //         float3 lastTestPointFine;
    //         float3 currentTestPointFine = lastTestPoint;
    //         float lastHeightDelta = 0.0001;

    //         float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
    //         float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

    //         intersectedT = tMax * (estimatedT / coarseSteps);
    //         intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);

    //         return true;

    // for (int j = 0; j < fineSteps; j++) {
    //     lastTestPointFine = currentTestPointFine;
    //     currentTestPointFine += incrementUnitFine;
    //     float heightDataFine;
    //     if (gApplySyn)
    //         heightDataFine = AutocovarianceBlendingDual(hfData, gMaxSampler, minLod, frac(abs(currentTestPoint.xy)) * UV_SCALE) *
    //         HF_SCALING + HF_OFFSET;
    //     else
    //         heightDataFine = gShellHF.SampleLevel(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, minLod).x * HF_SCALING
    //         + HF_OFFSET;
    //     const float currentHeightDeltaFine = currentTestPointFine.z - heightDataFine;
    //     // if (((currentHeightDeltaFine <= 0.f)))
    //     if (((currentHeightDeltaFine <= 0.f) && (currentHeightDeltaFine > -kSurfaceThickness)) ||
    //         (sign(lastHeightDelta) != sign(currentHeightDeltaFine)))
    //     {
    //         float heightDeltaRatio = abs(currentHeightDeltaFine) / (abs(currentHeightDeltaFine) + abs(lastHeightDelta));
    //         float estimatedT = lerp(float(j) + 1.f, float(j), heightDeltaRatio);
    //         intersectedT = tMax * (float(i) / coarseSteps + (float(estimatedT) / fineSteps) / coarseLength);
    //         // intersectedT = tMax * (estimatedT / fineSteps);
    //         intersectedHeight = lerp(currentTestPointFine.z, lastTestPointFine.z, heightDeltaRatio);
    //         return true;
    //     }
    //     lastHeightDelta = currentHeightDeltaFine;
    // }
    // }
    // lastHeightDelta = currentHeightDelta;
    // }
    return false;
}

bool traceSynthesizedHeightMapEstimatedNoMip(
    const DisplacementData displacementData,
    float3 startPoint,
    float3 endPoint,
    const float2 ddx,
    const float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    intersectedT = {};
    intersectedHeight = {};

    float3 viewDirection = endPoint - startPoint;
    uint width, height;
    gShellHF.GetDimensions(width, height);

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
                AutocovarianceBlendingDual(hfData, gMaxSampler, 0, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING + HF_OFFSET;
        else
            heightData = gShellHF.SampleLevel(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, 0).x * HF_SCALING + HF_OFFSET;
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
    const DisplacementData displacementData,
    float3 startPoint,
    float3 endPoint,
    const float2 ddx,
    const float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    intersectedT = {};
    intersectedHeight = {};

    float3 viewDirection = endPoint - startPoint;
    uint width, height;
    gShellHF.GetDimensions(width, height);

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
                AutocovarianceBlendingDual(hfData, gMaxSampler, minLod, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING + HF_OFFSET;
        else
            // heightData = gShellHF.SampleGrad(gMaxSampler, frac(abs(currentTestPoint.xy))* UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;
            heightData = gShellHF.SampleLevel(gMaxSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, minLod).x * HF_SCALING + HF_OFFSET;
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

// contact refinement hf tracing
bool traceSynthesizedHeightMapEstimated_CR(
    const DisplacementData displacementData,
    float3 startPoint,
    float3 endPoint,
    const float2 ddx,
    const float2 ddy,
    out float intersectedT,
    out float intersectedHeight
)
{
    intersectedT = {};
    intersectedHeight = {};
    float2 coarseDdx = ddx * 16;
    float2 coarseDdy = ddy * 16;

    float entryHeightData;
    if (gApplySyn)
        // entryHeightData = HexTilingHeight(gShellHF, gSampler, ddx, ddy, startPoint.xy * UV_SCALE) * HF_SCALING + HF_OFFSET;
        // entryHeightData = HistorgramBlending(hfData, gSampler, ddx, ddy, frac(abs(startPoint.xy)) * UV_SCALE) * HF_SCALING + HF_OFFSET;
        entryHeightData =
            AutocovarianceBlendingDual(hfData, gSampler, ddx, ddy, frac(abs(startPoint.xy)) * UV_SCALE) * HF_SCALING + HF_OFFSET;
    else
        entryHeightData = gShellHF.SampleGrad(gSampler, frac(abs(startPoint.xy)) * UV_SCALE, ddx, ddy).x * HF_SCALING + HF_OFFSET;

    // if (startPoint.z < entryHeightData)
    // {
    //     intersectedT = 0.000001f; // This seems to make the intersection ignored. TODO: experiment with eg. 0.000001
    //     intersectedHeight = startPoint.z;
    //     return true;
    // }

    float3 viewDirection = endPoint - startPoint;

    float footprint = max(max(length(ddx), length(ddy)), 0.0001f);
    float maxDistance = length(viewDirection.xy);
    int maxCount = ceil(maxDistance / footprint) + 1;
    bool useCoarse = true;
    if (maxCount < 16)
    {
        coarseDdx = ddx;
        coarseDdy = ddy;
        useCoarse = false;
    }
    else
    {
        maxCount = ceil(maxCount / 16);
    }

    // int maxCount = ceil(abs(viewDirection.x)) + ceil(abs(viewDirection.y)) + 2;
    // maxCount = min(maxCount * kRaymarchingSampleCountFactor, kRaymarchingMaxSampleCount); // Prevent freeze due to corner cases. Todo:
    //                                                                                       // Investigate why this happens.
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
        float heightData;
        if (gApplySyn)
            // heightData = HexTilingHeight(gShellHF, gSampler, coarseDdx, coarseDdy, currentTestPoint.xy * UV_SCALE) * HF_SCALING +
            // HF_OFFSET;
            // heightData = HistorgramBlending(hfData, gSampler, coarseDdx, coarseDdy, frac(abs(currentTestPoint.xy)) * UV_SCALE) *
            // HF_SCALING + HF_OFFSET;
            heightData =
                AutocovarianceBlendingDual(hfData, gSampler, coarseDdx, coarseDdy, frac(abs(currentTestPoint.xy)) * UV_SCALE) * HF_SCALING +
                HF_OFFSET;
        else
            heightData =
                gShellHF.SampleGrad(gSampler, frac(abs(currentTestPoint.xy)) * UV_SCALE, coarseDdx, coarseDdy).x * HF_SCALING + HF_OFFSET;
        const float currentHeightDelta = currentTestPoint.z - heightData;
        if (currentHeightDelta <= 0.f)
        {
            if (useCoarse)
            {
                float heightDeltaRatio = abs(currentHeightDelta) / (abs(currentHeightDelta) + abs(lastHeightDelta));
                float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

                intersectedT = tMax * (estimatedT / maxCount);
                intersectedHeight = lerp(currentTestPoint.z, lastTestPoint.z, heightDeltaRatio);
                return true;
            }
            else
            {
                float3 lastTestPointFine = lastTestPoint;
                float3 currentTestPointFine = lastTestPointFine;
                float lastHeightDeltaFine = currentHeightDelta;
                for (int j = 0; j < 16; j++)
                {
                    lastTestPointFine = currentTestPointFine;
                    currentTestPointFine += incrementUnit * M_1_16;
                    float heightFine;
                    if (gApplySyn)
                        // heightFine = HexTilingHeight(gShellHF, gSampler, ddx, ddy, currentTestPointFine.xy * UV_SCALE) * HF_SCALING +
                        // HF_OFFSET;
                        // heightFine = HistorgramBlending(hfData, gSampler, ddx, ddy, frac(abs(currentTestPointFine.xy)) * UV_SCALE) *
                        // HF_SCALING + HF_OFFSET;
                        heightFine = AutocovarianceBlendingDual(hfData, gSampler, ddx, ddy, frac(abs(currentTestPointFine.xy)) * UV_SCALE) *
                                         HF_SCALING +
                                     HF_OFFSET;
                    else
                        heightFine = gShellHF.SampleGrad(gSampler, frac(abs(currentTestPointFine.xy)) * UV_SCALE, ddx, ddy).x * HF_SCALING +
                                     HF_OFFSET;
                    const float currentFineHeightDelta = currentTestPointFine.z - heightFine;
                    if (currentFineHeightDelta <= 0.f)
                    {
                        float heightDeltaRatio = abs(currentFineHeightDelta) / (abs(currentFineHeightDelta) + abs(lastHeightDeltaFine));
                        float estimatedT = lerp(float(i) + 1.f, float(i), heightDeltaRatio);

                        intersectedT = tMax * ((float(j) / 16) * M_1_16 + estimatedT / maxCount);
                        intersectedHeight = lerp(currentTestPointFine.z, lastTestPointFine.z, heightDeltaRatio);
                        return true;
                    }
                    lastHeightDeltaFine = currentFineHeightDelta;
                }
            }
        }

        lastHeightDelta = currentHeightDelta;
    }

    return false;
}

bool calcSynthesizedDisplacementIntersection(
    const Ray ray,
    const StaticVertexData vertices[3],
    const DisplacementData displacementData,
    out DisplacementIntersection result
)
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
    float3 minIntersectionTexSpaceCoord = float3(minIntersectionUV * displacementData.size, minIntersection.textureSpaceHeight);
    float3 maxIntersectionTexSpaceCoord = float3(maxIntersectionUV * displacementData.size, maxIntersection.textureSpaceHeight);
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
    ret = traceSynthesizedHeightMapEstimatedSimpleMip(displacementData, minUV, maxUV, ddx, ddy, intersectedT, intersectedHeight);
#else
    ret = traceSynthesizedHeightMapEstimated(displacementData, minUV, maxUV, ddx, ddy, intersectedT, intersectedHeight);
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
    \param[in] displacementData Displacement data.
    \param[out] result Intersection result (only valid if triangle is intersected).
    \return Returns true if displaced triangle is intersected.
*/
bool intersectSynthesizedDisplacedTriangle(
    const Ray ray,
    StaticVertexData vertices[3],
    const float4x4 worldMat,
    const DisplacementData displacementData,

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
    if (kUseEmissiveLights && (kComputeDirect || rayData.pathLength > 0))
    {
        rayData.radiance += rayData.thp * mi.getProperties(sd).emission;
    }

    // always true for common surface
    rayData.terminated = true;

    // Add contribution of direct light from analytic lights.
    if (kUseAnalyticLights)
    {
        float3 Lr = evalDirectAnalytic(sd, mi, rayData.sg);
        rayData.radiance += rayData.thp * Lr;
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

            uint2 pixel = DispatchRaysIndex().xy;

            float3 wi = sd.frame.toLocal(envDir);
            float NdotL = wi.z > 0 ? wi.z : 0;
            uint4 packedData = packInput(wi, wo, frac(abs(sd.uv)), ddx, ddy);

            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = packedData.x;
            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = packedData.y;
            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = packedData.z;
            packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = packedData.w;
            float2 uv_raw = frac(abs(sd.uv)) * UV_SCALE;
            float4 uv_hashed = getHashUV(hfData, uv_raw);
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = uv_hashed.x;
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = uv_hashed.y;
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = uv_hashed.z;
            hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = uv_hashed.w;

            rayData.radiance = rayData.thp * envLi * NdotL;
            rayData.terminated = false;
            return;
        }
        // rayData.radiance += rayData.thp * envLi * mi.eval(sd, envDir, rayData.sg) * MISWeight(envPdf, mi.evalPdf(sd, envDir));
        else
            rayData.radiance += rayData.thp * envLi * mi.eval(sd, envDir, rayData.sg);
    }

    // // Generate scatter ray for the next path segment.
    // // The raygen shader will continue the path based on the returned payload.
    // if (!generateScatterRay(sd, mi, sd.computeRayOrigin(), rayData))
    // {
    //     rayData.terminated = true;
    //     return;
    // }

    rayData.pathLength++;
}



void handleDisplacementHit(const HitInfo hit, inout ScatterRayData rayData)
{
    if (gShowTracedHF)
    {
        rayData.radiance = pow(hit.getDisplacedTriangleHit().displacement, 1 / 1.4);
        // rayData.radiance = hit.getDisplacedTriangleHit().displacement;
        // rayData.terminated = false;
        rayData.terminated = true;
        return;
    }
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

    rayData.radiance = float3(0, 0, 0);
    float3 newPos = sd.posW + sd.frame.N * gCurvatureParas.y;
    if (generateEnvMapSample(newPos, rayData.sg, envLi, envDir, envPdf))
    {
        uint2 pixel = DispatchRaysIndex().xy;
        float3 wi = sd.frame.toLocal(envDir);
        float NdotL = wi.z > 0 ? wi.z : 0;
        uint4 packedData = packInput(wi, wo, frac(abs(sd.uv)), ddx, ddy);

        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = packedData.x;
        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = packedData.y;
        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = packedData.z;
        packedInput[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = packedData.w;

        float2 uv_raw = frac(abs(sd.uv)) * UV_SCALE;
        float4 uv_hashed = getHashUV(hfData, uv_raw);
        hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x)] = uv_hashed.x;
        hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1] = uv_hashed.y;
        hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2] = uv_hashed.z;
        hashedUV[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3] = uv_hashed.w;

        rayData.origin = newPos;
        rayData.radiance = rayData.thp * envLi * NdotL;
    }
    // #endif

    // rayData.pathLength++;
    // if (rayData.pathLength == 2)
    // {
    //     rayData.terminated = true;
    //     return;
    // }
    // float pdf;
    // float3 scatterDir = sample_cosine_hemisphere_concentric(sampleNext2D(rayData.sg), pdf);
    // rayData.origin = sd.computeRayOrigin();
    // rayData.direction = sd.frame.fromLocal(scatterDir);
    // rayData.thp *= nbtf.evalDebug(scatterDir, wo, sd.uv * UV_SCALE, ddx, ddy, gApplySyn) / pdf;
    // rayData.BRDFPdf = pdf;
    // if (all(rayData.thp == 0.f))
    // {
    //     rayData.terminated = true;
    //     return;
    // }
}

