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

import Utils.Texture.HistogramBlending;
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;

// import Rendering.Lights.EnvMapSampler;
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
    // EnvMapSampler envMapSampler;
};

RWTexture2D<float4> ouputColor;




[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= iResolution))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    float2 uv = (float2(pixel) + float2(0.5)) / iResolution;
    // float2 h, d;
    // dirToRusinkiewicz(sample_hemisphere(gWi), sample_hemisphere(gWo), h, d);
    // float3 vh = spherical_to_cartesian_rad(h);
    // float3 vd = spherical_to_cartesian_rad(d);

    // h = hemisphereDirToDiskSquare(vh);
    // d = hemisphereDirToDiskSquare(vd);

    float3 vWo = spherical_to_cartesian_rad(gWo);

    if (gShowIBL) {
        ouputColor[pixel].xyz = nnmatIBL.evalIBL(vWo, uv, M_PI * gRotAngles / 180);
    }

    else {
        float3 radiance = float3(0);
        uint sampleNum = gSampleNum;
        for (uint i = 0; i < sampleNum; i++) {
            for (uint j = 0; j < sampleNum; j++) {
                float2 sample = float2(((float)i + 0.5f) / (float)sampleNum, ((float)j + 0.5f) / (float)sampleNum);
                float3 wi = sample_hemisphere(sample);
                float3 fr = nnmat.evalDebug(wi, vWo, uv, 0, 0);
                float3 Le = envMap.eval(wi);
                radiance += fr * Le;
            }
        }
        ouputColor[pixel].xyz = radiance * M_2PI / ((float)sampleNum * (float)sampleNum);
    }

}