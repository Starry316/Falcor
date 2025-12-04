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
import Utils.Math.FormatConversion;
import Utils.Debug.PixelDebug;
import Utils.Neural.NNMat;
import Utils.Neural.MLP;
import Rendering.Lights.LightHelpers;
import Rendering.Lights.EnvMapSampler;
import Scene.Displacement.DisplacementMapping;

import Tools.BTFUtils;
import Utils.Texture.HistogramBlending;

cbuffer CB
{
    uint gFrameCount;    // Frame count since scene was loaded.
    uint gPRNGDimension; // First available PRNG dimension.
    uint2 gRenderTargetDim;
    float4 gControlParas;

    bool gShowTracedHF;
    bool gTracedShadowRay;
    bool gApplySyn;
    bool gShowGT;

    bool gShowFeatureMap;
    int  gFeatureLevel;
    EnvMapSampler envMapSampler;
    NNMat nnmatT;
    NNMat nnmatBTF;
}

// Outputs
RWTexture2D<float4> gOutputColor;

SamplerState gSampler;
SamplerState gPointSampler;

// Static configuration based on defines set from the host.
#define is_valid(name) (is_valid_##name != 0)
static const bool kUseEnvLight = USE_ENV_LIGHT;
static const float kRayTMax = FLT_MAX;

#define MAX_HEIGHT gControlParas.x
#define UV_SCALE gControlParas.y
#define HF_OFFSET gControlParas.z
#define HF_SCALE gControlParas.w
#define SHADOW_OFFSET 0.1

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

    // #if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        v = gScene.getVertexData(triangleHit);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
    }
    // #endif
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

// Envmap sampling
bool generateEnvMapSample(const float3 origin, inout SampleGenerator sg, inout float3 Li, inout float3 dir, inout float pdf)
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
/**
 * Evaluates the direct illumination from analytic lights.
 * This function samples Falcor's light list uniformly with one shadow ray.
 * @param[in] sd Shading data.
 * @param[in] mi Material instance.
 * @param[in,out] sg SampleGenerator object.
 * @return Outgoing radiance in view direction.
 */
float3 evalDirectAnalytic(const float3 origin, inout SampleGenerator sg, inout float3 dir)
{
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0)
        return float3(0.f);

    // Sample local light source.
    AnalyticLightSample ls;
    if (!sampleLight(origin, gScene.getLight(0), sg, ls))
        return float3(0.f);
    dir = ls.dir;

    if (gTracedShadowRay)
    {
        bool V = traceShadowRay(origin, ls.dir, ls.distance);
        if (!V)
            return float3(0.f);
    }

    // Evaluate contribution.
    return ls.Li;
}
// Returns right-handed frame: T x B = N.
// Replace previous buildLocalFrameFromNormal with Mitsuba-like stable construction.
void buildLocalFrameFromNormal(in float3 N_in, out float3 B, out float3 T)
{
    float3 n = normalize(N_in);

    // Choose construction depending on dominant component to avoid numerical instability.
    // if (abs(n.x) > abs(n.y))
    // {
    // Use x/z plane to build tangent
    // float invLen = 1.0f / sqrt(n.x * n.x + n.z * n.z);
    // T = float3(-n.z * invLen, 0.0f, n.x * invLen);
    // }
    // else
    // {
    //     // Use y/z plane to build tangent
    float invLen = 1.0f / sqrt(n.y * n.y + n.z * n.z);
    T = float3(0.0f, n.z * invLen, -n.y * invLen);
    // }

    // Ensure orthonormal right-handed frame: T x B = N
    B = cross(n, T);
    // (T and B should already be normalized by construction, but normalize to be safe)
    T = normalize(T);
    B = normalize(B);
}

// Convert world vector to local coordinates (T, B, N basis) using only world normal.
float3 toLocal(in float3 worldVec, in float3 worldNormal)
{
    float3 T, B;
    buildLocalFrameFromNormal(worldNormal, T, B);
    return float3(dot(worldVec, T), dot(worldVec, B), dot(worldVec, worldNormal));
}

// Rotate vector v around normalized axis by angle (radians) using Rodrigues' formula
float3 rotateAroundAxis(float3 v, float3 axis, float angle)
{
    float ca = cos(angle);
    float sa = sin(angle);
    return v * ca + cross(axis, v) * sa + axis * dot(axis, v) * (1.0 - ca);
}
// compute alpha so that the spherical patch mapped from a unit square is (approximately) square
// Solve alpha = (2 * asin( A / (2 * R^2 * alpha * sin(thetaCenter)) )) / sin(thetaCenter)
// using fixed-point iteration with damping.
// ...existing code...
float computeAlphaForSquare(float R, float thetaCenter, float A)
{
    const float EPS = 1e-8f;
    const int MAX_IT = 50;
    const float TWO_PI = 6.283185307179586f;

    float s = sin(thetaCenter);
    // near-pole fallback (degenerate)
    if (abs(s) < 1e-6f)
    {
        float alpha_approx = 2.0f * sqrt(max(0.0f, A)) / max(R, EPS);
        return min(alpha_approx, TWO_PI);
    }

    // initial guess (small-angle approximation)
    float alpha = sqrt(max(0.0f, A)) / max(R * s, EPS);
    alpha = clamp(alpha, 1e-6f, TWO_PI);

    for (int it = 0; it < MAX_IT; ++it)
    {
        // f(alpha) = 2 * R^2 * alpha * s * sin( (s * alpha)/2 ) - A
        float half_sa = 0.5f * s * alpha;
        float sin_half_sa = sin(half_sa);
        float cos_half_sa = cos(half_sa);

        float f = 2.0f * R * R * alpha * s * sin_half_sa - A;

        if (abs(f) < 1e-8f)
            break;

        // derivative: df/dalpha = 2 * R^2 * s * [ sin( s*alpha/2 ) + alpha * (s/2) * cos( s*alpha/2 ) ]
        float df = 2.0f * R * R * s * (sin_half_sa + alpha * (s * 0.5f) * cos_half_sa);

        // safe Newton step with damping
        float step = f / max(abs(df), EPS);
        // damp aggressive steps
        step = clamp(step, -0.5f * alpha, 0.5f * alpha);
        alpha -= step;

        // keep in valid range
        alpha = clamp(alpha, 1e-6f, TWO_PI);
    }

    return clamp(alpha, 1e-6f, TWO_PI);
}
// ...existing code...
// ...existing code...
// Replace usages: mapSquareToSpherePatch previously took alpha; compute it when alpha <= 0.
// Modify mapSquareToSpherePatch signature or call-site accordingly. Example below shows call-agnostic version
void mapSquareToSpherePatch(
    float2 uv,
    float3 sphereCenter,
    float R,
    float phiCenter,
    float thetaCenter,
    out float3 outPos,
    out float3 outNormal
)
{
    const float A = 1.0f; // desired mapped area on sphere surface

    // compute alpha so the patch is square on the sphere (depends on R and thetaCenter)
    float alpha = computeAlphaForSquare(R, thetaCenter, A);

    // compute z-range (z = cos(theta)) span that yields area A for given alpha and R:
    float deltaZ = A / (R * R * alpha); // z1 - z0
    deltaZ = clamp(deltaZ, 0.0f, 2.0f);

    float zc = cos(thetaCenter);
    float z0 = zc - 0.5f * deltaZ;
    float z1 = zc + 0.5f * deltaZ;

    // clamp into [-1, 1]
    z0 = clamp(z0, -1.0f, 1.0f);
    z1 = clamp(z1, -1.0f, 1.0f);

    // map v -> z linearly in [z0, z1]
    float z = lerp(z0, z1, uv.y);

    // map u -> phi within [phiCenter - alpha/2, phiCenter + alpha/2]
    float phi = phiCenter - 0.5f * alpha + uv.x * alpha;

    // local (north-pole aligned) direction on unit sphere
    float sinTheta = sqrt(max(0.0f, 1.0f - z * z));
    float3 localDir = float3(sinTheta * cos(phi), sinTheta * sin(phi), z);

    // rotate localDir so its patch center aligns with (thetaCenter, phiCenter)
    float3 centerDir = float3(sin(thetaCenter) * cos(phiCenter), sin(thetaCenter) * sin(phiCenter), cos(thetaCenter));
    float3 pole = float3(0.0f, 0.0f, 1.0f);
    float cosAngle = dot(pole, centerDir);
    float3 axis = cross(pole, centerDir);
    float3 worldDir;
    if (length(axis) < 1e-6f)
    {
        if (cosAngle > 0.0f)
            worldDir = localDir;
        else
            worldDir = float3(-localDir.x, -localDir.y, -localDir.z);
    }
    else
    {
        axis = normalize(axis);
        float angle = acos(clamp(cosAngle, -1.0f, 1.0f));
        worldDir = rotateAroundAxis(localDir, axis, angle);
    }

    outNormal = worldDir;
    outPos = sphereCenter + R * worldDir;
}

// // inverse mapping that computes alpha internally as well
float2 mapSpherePointToSquareUV(float3 point, float3 sphereCenter, float R, float phiCenter, float thetaCenter, out bool inside)
{
    const float TWO_PI = 6.283185307179586f;
    const float A = 1.0f;

    float3 worldDir = normalize(point - sphereCenter);

    float3 centerDir = float3(sin(thetaCenter) * cos(phiCenter), sin(thetaCenter) * sin(phiCenter), cos(thetaCenter));
    float3 pole = float3(0.0f, 1.0f, 0.0f);
    float cosAngle = dot(pole, centerDir);
    float3 axis = cross(pole, centerDir);

    float3 localDir;
    if (length(axis) < 1e-6f)
    {
        if (cosAngle > 0.0f)
            localDir = worldDir;
        else
            localDir = float3(-worldDir.x, -worldDir.y, -worldDir.z);
    }
    else
    {
        axis = normalize(axis);
        float angle = acos(clamp(cosAngle, -1.0f, 1.0f));
        localDir = rotateAroundAxis(worldDir, axis, -angle);
    }

    // compute alpha
    float alpha = computeAlphaForSquare(R, thetaCenter, A);

    float deltaZ = A / (R * R * alpha);
    deltaZ = clamp(deltaZ, 0.0f, 2.0f);
    float zc = cos(thetaCenter);
    float z0 = clamp(zc - 0.5f * deltaZ, -1.0f, 1.0f);
    float z1 = clamp(zc + 0.5f * deltaZ, -1.0f, 1.0f);

    float denom = z1 - z0;
    float v;
    if (abs(denom) < 1e-8f)
    {
        v = 0.0f;
    }
    else
    {
        v = (localDir.z - z0) / denom;
    }

    float phiLocal = atan2(localDir.y, localDir.x);
    float phiStart = phiCenter - 0.5f * alpha;

    float dPhi = phiLocal - phiStart;
    dPhi = fmod(dPhi, TWO_PI);
    if (dPhi < 0.0f)
        dPhi += TWO_PI;

    float u = dPhi / alpha;

    inside = (u >= 0.0f && u <= 1.0f && v >= 0.0f && v <= 1.0f);
    return float2(u, 1 - v);
}

// inverse mapping that computes alpha internally as well
// float2 mapSpherePointToSquareUV(float3 point, float3 sphereCenter, float R, float phiCenter, float thetaCenter, out bool inside)
// {
//     // Cube-face radial map (map a single cube face centered at (phiCenter,thetaCenter)
//     // onto the unit square). We rotate the world direction so that the face center
//     // aligns with the +Z face normal, then perform the typical cube-face projection
//     // x' = x/z, y' = y/z and check |x'|,|y'| <= 1 (inside the face).

//     float3 worldDir = normalize(point - sphereCenter);

//     // face center direction
//     float3 centerDir = float3(sin(thetaCenter) * cos(phiCenter), sin(thetaCenter) * sin(phiCenter), cos(thetaCenter));
//     const float3 pole = float3(0.0f, 0.0f, 1.0f);

//     // Rotate worldDir so that centerDir -> pole (i.e. face normal becomes +Z)
//     float cosAngle = clamp(dot(centerDir, pole), -1.0f, 1.0f);
//     float3 axis = cross(centerDir, pole);

//     float3 localDir;
//     float axisLen = length(axis);
//     if (axisLen < 1e-6f) {
//         // center is already near pole or opposite
//         if (cosAngle > 0.0f) localDir = worldDir; // aligned
//         else localDir = float3(-worldDir.x, -worldDir.y, -worldDir.z); // 180 deg flip
//     }
//     else {
//         axis = axis / axisLen;
//         float angle = acos(cosAngle);
//         // rotate worldDir by 'angle' around 'axis' to bring face center to +Z
//         localDir = rotateAroundAxis(worldDir, axis, angle);
//     }

//     // For cube-face projection we require localDir.z > 0 (in front of the face)
//     if (localDir.z <= 1e-8f) {
//         inside = false;
//         return float2(0.0f, 0.0f);
//     }

//     // project to plane z = 1: x' = x/z, y' = y/z
//     float xp = localDir.x / localDir.z;
//     float yp = localDir.y / localDir.z;

//     // inside if within face bounds [-1,1]
//     inside = (abs(xp) <= 1.0f + 1e-6f && abs(yp) <= 1.0f + 1e-6f);

//     // map from [-1,1] to [0,1]
//     float u = 0.5f * (xp + 1.0f);
//     float v = 0.5f * (yp + 1.0f);

//     // match earlier convention (returns v flipped)
//     return float2(u, 1.0f - v);
// }
float3 customSphereIntersectionTest(ScatterRayData rayData, float3 sphereCenter, float sphereRadius)
{
    float t;
    float3 backgroundColor = float3(0.0f);

    if (intersectRaySphere(rayData.origin, rayData.direction, sphereCenter, sphereRadius, t, 0.f, kRayTMax, false))
    {
        float3 hitPos = rayData.origin + rayData.direction * t;
        float3 normal = normalize(hitPos - sphereCenter);
        float h = HF_OFFSET;
        float3 subSphereCenter = sphereCenter - h * normal;
        // subSphereCenter.y -= h;

        bool inside, insideInternal;

        bool hitInternal =
            intersectRaySphere(rayData.origin, rayData.direction, subSphereCenter, sphereRadius, t, 0.f, kRayTMax, false);
        float2 uv = mapSpherePointToSquareUV(hitPos, sphereCenter, sphereRadius, M_PI_2, M_PI_2, inside);
        float2 xy = mapSpherePointToSquareUV(
            rayData.origin + rayData.direction * t, subSphereCenter, sphereRadius, M_PI_2, M_PI_2, insideInternal
        );
        float3 background = gScene.envMap.eval(rayData.direction);
        if (inside)
        {
            if (gShowTracedHF) {
                if (hitInternal)
                {
                    // if (insideInternal) {
                        float3 wo = toLocal(-rayData.direction, normalize(rayData.origin + rayData.direction * t - sphereCenter));

                        float uv_remapped[10];
                        nnmatT.evalT(wo, frac(uv * UV_SCALE), uv_remapped);
                        float3 ret = float3(nnmatBTF.evalBTF(float3(0), wo, uv_remapped));

                        if (all(ret > HF_SCALE))
                        {
                            if (gApplySyn)
                                background = float3(0.2, 0.1, 0.1) + 0.2 + 0.8 * background;
                            return background;
                        }

                        else
                            return ret;
                }
                else {
                    if (gApplySyn)
                    {
                        return 0.2 + 0.8 * background;
                    }
                }
            }
            else {
                float3 wo = toLocal(-rayData.direction, normal);
                // if (gShowTracedHF && insideInternal && hitInternal)
                // {
                //     // xy = uv - wo.xy * h;
                //     // if (any(xy < 0.0f) || any(xy > 1.0f))
                //     //     return 0.0;
                //     // xy = clamp(xy, float2(0.0), float2(1.0));
                //     wo = normalize(float3(uv - xy, h));
                // }

                float uv_remapped[10];
                nnmatT.evalT(wo, frac(uv * UV_SCALE), uv_remapped);
                float3 ret = float3(nnmatBTF.evalBTF(float3(0), wo, uv_remapped));

                if (all(ret > HF_SCALE))
                {
                    if (gApplySyn)
                    {
                        if (hitInternal)
                                background = float3(0.2, 0.1, 0.1) + 0.2 + 0.8 * background;
                        else background = 0.2 + 0.8 * background;
                    }
                    return background;
                }

                else
                    return ret;

            }



        }
        // if (hitInternal)
        // return float3(0.2, 0.1, 0.1) + float3(0.2f) + 0.8 * gScene.envMap.eval(rayData.direction);
        if (gApplySyn)
        {
            if (hitInternal)
                return float3(0.2, 0.1, 0.1) + float3(0.2f) + 0.8 * gScene.envMap.eval(rayData.direction);

            return float3(0.2f) + 0.8 * gScene.envMap.eval(rayData.direction);
        }
        return gScene.envMap.eval(rayData.direction);
    }
    return gScene.envMap.eval(rayData.direction);
}
/**
 * Inverse of sample_sphere: maps a (possibly non-normalized) direction on the
 * unit sphere back to the uniform sample `u` that would produce it.
 * @param[in] v Direction (will be normalized).
 * @return `u` in [0,1)^2 such that sample_sphere(u) ~= normalize(v).
 */
float2 sample_sphere_inverse(float3 v)
{
    // Ensure we operate on a unit vector
    v = normalize(v);

    // Recover u.x from cosTheta: cosTheta = 1 - 2*u.x => u.x = (1 - cosTheta)/2
    float cosTheta = v.z;
    float ux = (1.0f - cosTheta) * 0.5f;
    ux = saturate(ux);

    // Recover u.y from phi. Use atan2(y,x) in [-pi,pi], remap to [0,2pi).
    float uy;
    float2 xy = v.xy;
    if (length(xy) < 1e-8f)
    {
        // At the poles phi is undefined; choose 0.
        uy = 0.0f;
    }
    else
    {
        float phi = atan2(v.y, v.x);
        if (phi < 0.0f)
            phi += M_2PI;
        uy = phi * M_1_2PI; // divide by 2*pi
        // Numerical safety: keep in [0,1)
        uy = clamp(uy, 0.0f, nextafter(1.0f, 0.0f));
    }

    return float2(ux, uy);
}

float2 cartesian_to_spherical_rad_my(float3 p)
{
    p = normalize(p);
    float2 sph;
    sph.x = cos(p.z);
    sph.y = cos(-p.y) + M_PI;
    return sph;
}

float3 customSphereIntersectionTestLF(ScatterRayData rayData, float3 sphereCenter, float sphereRadius)
{
    float t;
    float3 background = 0;
    if (kUseEnvLight)
        background = gScene.envMap.eval(rayData.direction);
    else background = handleMiss(DispatchRaysIndex().xy);
    float3 wo = -rayData.direction.xzy;
    sphereRadius = 3.25;
    sphereCenter.y = 15.3 - sphereRadius;

    if (intersectRaySphere(rayData.origin, rayData.direction, sphereCenter, sphereRadius, t, 0.f, kRayTMax, false))
    {
        float3 hitPos = rayData.origin + rayData.direction * t;
        float3 ret = float3(0.0f);
        hitPos.y -= sphereCenter.y;
        hitPos /= sphereRadius;
        hitPos = hitPos.xzy;

        // hemisphere detection

        if (hitPos.z <= 0)
            return background;
        float2 uv = hemisphereDirToDiskSquare(normalizeSafe(hitPos));
        // float2 uv = cartesian_to_spherical_rad(hitPos);
        // float uv_remapped[6];


        float3 pos1, pos2;
        // nnmatT.evalTTriDoubleSide(wo, uv, pos1, pos2);
        nnmatT.evalTTriDoubleSideSoft(wo, uv, pos1, pos2);


        ret = nnmatBTF.evalLFTriDoubleSideFull(wo, wo, pos1, pos2);
        // ret = nnmatBTF.evalLFTriDoubleSide(wo, wo, pos1, pos2);


        if (gShowFeatureMap)
            ret = nnmatBTF.evalFeatureMap(wo, wo, pos1, pos2, gFeatureLevel);
        if (all(ret < HF_SCALE) && !gApplySyn)
        {
            return background;
        }
        // return float3(uv_remapped[0], uv_remapped[1], uv_remapped[2]);
        return ret;
    }
    return background;
}


float3 customSphereIntersectionTestBTF(ScatterRayData rayData, float3 sphereCenter, float sphereRadius)
{
    float t;
    float3 background;
//     if (kUseEnvLight)
//         background = gScene.envMap.eval(rayData.direction);
//     else background = handleMiss(DispatchRaysIndex().xy);
//     float3 wo = -rayData.direction.xzy;
//     sphereRadius = 1;
//     // sphereRadius = 3.25;
//     // sphereCenter.y = 15.3 - sphereRadius;

//     if (intersectRaySphere(rayData.origin, rayData.direction, sphereCenter, sphereRadius, t, 0.f, kRayTMax, false))
//     {
//         float3 hitPos = rayData.origin + rayData.direction * t;
//         float3 envLi = float3(0.0f);
//         float3 wiEnv = float3(0.0f); // world space incident direction
//         float3 wiLight = float3(0.0f);
//         float3 Li = evalDirectAnalytic(hitPos, rayData.sg, wiLight);
//         // float3 Li = float3(1.0f);
//         float3 ret = float3(0.0f);
//         if (kUseEnvLight)
//         {
//             float dummyPdf;
//             generateEnvMapSample(hitPos, rayData.sg, envLi, wiEnv, dummyPdf);
//         }

//         hitPos.y -= sphereCenter.y;
//         hitPos /= sphereRadius;
//         // hemisphere detection
//         if (hitPos.y <= 0)
//             return background;

//         float2 uv = hemisphereDirToDiskSquare(normalizeSafe(hitPos.xzy));
//         float uv_remapped[10];
//         // nnmatT.evalT(wo, frac(uv * UV_SCALE), uv_remapped);

//         nnmatT.evalTTri(wo, frac(uv * UV_SCALE), uv_remapped);

//         ret = envLi * nnmatBTF.evalLFTri(wiEnv.xzy, wo, uv_remapped);

//         // if (gShowFeatureMap)
//         //     return nnmatBTF.evalFeatureMap(float3(0), wo, uv_remapped, gFeatureLevel);

//         // #ifdef POINTLIGHT
//         // ret = Li * (exp(nnmatBTF.evalBTF(wiLight.xzy, wo, uv_remapped)) - 1);
//         // ret = envLi * (exp(nnmatBTF.evalBTF(wiEnv.xzy, wo, uv_remapped)) - 1);

// // #endif

//         return ret ;
//     }
    return background;
}


// #define LFRENDERING
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

// #ifdef LFRENDERING

    return float4(customSphereIntersectionTestLF(rayData, float3(0.0,0, 0.0), 1), 1.0f);
    // return float4(customSphereIntersectionTest(rayData, float3(0.0, -MAX_HEIGHT * 20, 0.0), MAX_HEIGHT * 20), 1.0f);
// #endif

//     return float4(customSphereIntersectionTestBTF(rayData, float3(0.0, 0, 0.0), 1), 1.0f);




    // ===========================  NeuMat  ===========================
    // We only trace the primary ray and mark the valid pixels in cudaValidBuffer.
    // Only the valid pixels are needed for cuda inference.
    // traceScatterRay(rayData);
    // // cudaValidBuffer[(pixel.y * gRenderTargetDim.x + pixel.x)] = rayData.terminated ? 0 : 1;
    // return float4(rayData.radiance, 1.0f);
}
float3 handleMiss(const uint2 pixel)
{
    // Draw a checkerboard pattern.
    return ((pixel.x ^ pixel.y) & 0x8) != 0 ? float3(0.5f) : float3(0.1f);
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
    if (kUseEnvLight)
    {
        float3 Le = gScene.envMap.eval(WorldRayDirection());
        rayData.radiance += Le;
    }
    else {
        rayData.radiance = handleMiss(pixel);
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
    gOutputColor[pixel] = tracePath(pixel, frameDim);
}

// hit non-displaced triangle
void handleHit(const HitInfo hit, inout ScatterRayData rayData)
{
    // Load shading data.
    ShadingData sd = loadShadingData(hit, rayData.origin, rayData.direction);

    // Create material instance.
    let lod = ExplicitLodTextureSampler(0.f);
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    float3 wo = sd.frame.toLocal(sd.V);
    float3 wiWS; // world space incident direction

    float3 Li = evalDirectAnalytic(sd.computeRayOrigin(), rayData.sg, wiWS);

    float uv_remapped[10];

#ifdef LFRENDERING
    rayData.terminated = true;
    // float2 uv_raw = frac(abs(sd.uv) * UV_SCALE);

    // nnmatT.evalT(wo, frac(uv_raw), uv_remapped);
    // rayData.radiance = nnmatBTF.evalBTF(float3(0), wo, uv_remapped);
    rayData.radiance = (sd.frame.N + 1) * 0.5 ;
    return;
#endif

#ifdef POINTLIGHT
    if (sd.materialID == 1)
    {
        float3 wi = sd.frame.toLocal(wiWS);
        rayData.terminated = true;
        float NdotL = wi.z > 0 ? wi.z : 0;
        float2 uv_raw = frac(abs(sd.uv) * UV_SCALE);
        // float3 fr = nnmat.evalDebug(wi, wo, uv_raw, 0, 0);
        //  rayData.radiance = Li * fr * NdotL;

        // float2 xy = uv_raw - wo.xy * 0.55;
        // xy = clamp(xy, float2(0.0), float2(1.0));
        // wo = normalize(float3(uv_raw - xy, 0.55));
        // if (any(xy >= float2(1.0)) || any(xy <= float2(0.0)))
        // {

        //     return;
        // }

        rayData.terminated = true;

        nnmatT.evalT(wo, frac(uv_raw), uv_remapped);
        rayData.radiance = Li * nnmatBTF.evalBTF(wi, wo, uv_remapped) * NdotL;
        // rayData.radiance = nnmatBTF.evalBTF(wi, wo, uv_raw);
    }
    else
    {
        rayData.terminated = true;
        rayData.radiance = Li * mi.eval(sd, wiWS, rayData.sg);
    }
#endif
    if (kUseEnvLight)
    {
        float3 envLi;
        float envPdf;
        if (generateEnvMapSample(sd.computeRayOrigin(), rayData.sg, envLi, wiWS, envPdf))
        {
            if (sd.materialID == 1)
            {
                float3 wi = sd.frame.toLocal(wiWS);
                // float3 wi = wiWS.xzy;
                float NdotL = wi.z > 0 ? wi.z : 0;
                float2 uv_raw = frac(abs(sd.uv) * UV_SCALE);
                float2 xy = uv_raw - wo.xy * 0.55;
                // if (any(xy >= float2(1.0)) || any(xy <= float2(0.0)))
                // {
                //     rayData.terminated = false;
                //     return;
                // }
                xy = clamp(xy, float2(0.0), float2(1.0));
                // wo = normalize(float3(uv_raw - xy, 0.55));

                // float3 fr = nnmat.evalDebug(wi, wo, uv_raw, 0, 0);
                nnmatT.evalT(wo, frac(uv_raw), uv_remapped);
                float3 fr = nnmatBTF.evalBTF(wi, wo, uv_remapped);

                // float3 fr = float3(0.0f);
                rayData.terminated = true;
                rayData.radiance += envLi * fr * NdotL;
            }
            else
            {
                rayData.terminated = true;
                rayData.radiance += envLi * mi.eval(sd, wiWS, rayData.sg);
            }
        }
    }
}

