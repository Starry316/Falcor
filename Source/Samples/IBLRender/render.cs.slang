/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"
import Scene.Scene;
import Scene.SceneTypes;
import Utils.Texture.HistogramBlending;
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;
import Utils.Sampling.SampleGenerator;
import Rendering.Lights.EnvMapSampler;
import Utils.Math.MathHelpers;
import Utils.Neural.NNMat;
import Scene.Lights.EnvMap;
cbuffer CB
{
    float2 iResolution;
    float2 gWo;
    float2 gWi;
    NNMat nnmat;
    NNMat nnmatIBL;
    EnvMap envMap;
    uint gSampleNum;
    float3 gRotAngles;
    bool gShowIBL;
    EnvMapSampler envMapSampler;
};

RWTexture2D<float4> ouputColor;
RWTexture2D<float4> ouputColorRef;

ParameterBlock<Scene> scene;

bool sampleEvnmap(const float2 rnd, out EnvMapSample result, float lod = 0.f)
{
    float2 p = rnd; // Random sample in [0,1)^2.
    uint2 pos = 0;  // Top-left texel pos of current 2x2 region.

    // Iterate over mips of 2x2...NxN resolution.
    for (int mip = envMapSampler.importanceBaseMip - 1; mip >= 0; mip--)
    {
        // Scale position to current mip.
        pos *= 2;

        // Load the four texels at the current position.
        float w[4];
        w[0] = envMapSampler.importanceMap.Load(int3(pos, mip));
        w[1] = envMapSampler.importanceMap.Load(int3(pos + uint2(1, 0), mip));
        w[2] = envMapSampler.importanceMap.Load(int3(pos + uint2(0, 1), mip));
        w[3] = envMapSampler.importanceMap.Load(int3(pos + uint2(1, 1), mip));

        float q[2];
        q[0] = w[0] + w[2];
        q[1] = w[1] + w[3];

        uint2 off;

        // Horizontal warp.
        float d = q[0] / (q[0] + q[1]); // TODO: Do we need to guard against div-by-zero. We should ensure we never go down a path that has
                                        // p=0.

        if (p.x < d) // left
        {
            off.x = 0;
            p.x = p.x / d;
        }
        else // right
        {
            off.x = 1;
            p.x = (p.x - d) / (1.f - d);
        }

        // Vertical warp.
        // Avoid stack allocation by not using dynamic indexing.
        // float e = w[off.x] / q[off.x];
        float e = off.x == 0 ? (w[0] / q[0]) : (w[1] / q[1]);

        if (p.y < e) // bottom
        {
            off.y = 0;
            p.y = p.y / e;
        }
        else // top
        {
            off.y = 1;
            p.y = (p.y - e) / (1.f - e);
        }

        pos += off;
    }

    // At this point, we have chosen a texel 'pos' in the range [0,dimension) for each component.
    // The 2D sample point 'p' has been warped along the way, and is in the range [0,1) representing sub-texel location.

    // Compute final sample position and map to direction.
    float2 uv = ((float2)pos + p) * envMapSampler.importanceInvDim; // Final sample in [0,1)^2.
    float3 dir = oct_to_ndir_equal_area_unorm(uv);

    // Compute final pdf.
    // We sample exactly according to the intensity of where the final samples lies in the octahedral map, normalized to its average
    // intensity.
    float avg_w = envMapSampler.importanceMap.Load(int3(0, 0, envMapSampler.importanceBaseMip)); // 1x1 mip holds integral over importance
                                                                                                 // map. TODO: Replace by constant or
                                                                                                 // rescale in setup so that the integral
                                                                                                 // is 1.0
    float pdf = envMapSampler.importanceMap[pos] / avg_w;

    result.dir = envMap.toWorld(dir);
    result.pdf = pdf * M_1_4PI;
    result.Le = envMap.eval(result.dir, lod);

    return true;
}

inline float3 toLocal(float3 wi)
{
    return wi.xzy;
}
[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= iResolution))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    float2 uv = (float2(pixel) + float2(0.5)) / iResolution;

    float3 vWo = spherical_to_cartesian_rad(gWo);
    uv.y = 1 - uv.y;
    if (gShowIBL)
    {
        ouputColorRef[pixel].xyz = nnmatIBL.evalIBL(vWo, uv, M_PI * gRotAngles / 180);
      
    }

    SampleGenerator sg = SampleGenerator(pixel, 1);


    float3 radiance = float3(0);
    uint sampleNum = gSampleNum;
    for (uint i = 0; i < sampleNum; i++)
    {
        float2 sample = sampleNext2D(sg);
        EnvMapSample envSample;
        sampleEvnmap(sample, envSample, 0);
        float3 fr = 0;
        float3 wi = toLocal(envSample.dir);

        if (wi.z > 0)
            fr = nnmat.evalDebug(wi, vWo, uv, 0, 0);

        radiance += fr * envSample.Le;
        // for (uint j = 0; j < sampleNum; j++) {
        //     // float2 sample = float2(((float)i + 0.5f) / (float)sampleNum, ((float)j + 0.5f) / (float)sampleNum);

        //         // float3 wi = sample_hemisphere(sample);

        //     // float3 Le = envMapSampler.eval(wi);

        // }
    }
    ouputColor[pixel].xyz = radiance / ((float)sampleNum);
}
