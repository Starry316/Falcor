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
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import Utils.Sampling.SampleGenerator;
#include "Utils/Math/MathConstants.slangh"
cbuffer CB
{
    float2 iResolution;
    float2 mu_p;
    float sigma_p ;
    float texelWidth;
    float gSampleSize;
    float gImgSize;
};

Texture2D<float4> gHF;
RWTexture2D<float4> ouputColor;
RWBuffer<int> outputSamples;
SamplerState gSampler;
float2 sampleGaussian2D(float2 s)
{
    float r = sqrt(-2.0 * log(s.x));
    float theta = M_2PI * s.y;
    return float2(r * cos(theta), r * sin(theta));
}
float3 calNormal(float2 pos, float2 dim)
{
    // float offset = -5.344140625;
    float offset = 0;
    float3 normal = float3(
            gHF.SampleLevel(gSampler, float2((pos.x - 0.5) / dim.x, pos.y / dim.y), 0).x + offset -
            gHF.SampleLevel(gSampler, float2((pos.x + 0.5) / dim.x, pos.y / dim.y), 0).x + offset   ,
            gHF.SampleLevel(gSampler, float2(pos.x / dim.x, (pos.y - 0.5) / dim.y), 0).x + offset -
            gHF.SampleLevel(gSampler, float2(pos.x / dim.x, (pos.y + 0.5) / dim.y), 0).x + offset   ,
            1.0
    );
    return normalize(normal);
}

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= iResolution))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);

    float intrinsicRoughness = 1.0 / sqrt(iResolution.x * iResolution.y);

    uint2 texDim;
    gHF.GetDimensions(texDim.x, texDim.y);

    SampleGenerator sg = SampleGenerator(pixel, 1);

    float2 sampledPos = (float2(pixel) + sampleNext2D(sg)) / iResolution;
    float2 g = sampleGaussian2D(sampledPos);

    float2 x = g * sigma_p / texelWidth;
    x += mu_p;
    float2 normal = calNormal(x, texDim).xy;
    normal += intrinsicRoughness * sampleGaussian2D(sampleNext2D(sg));
    int ix = int((1 + normal.x) / 2.0f * gImgSize);
    int iy = int((1 - normal.y) / 2.0f * gImgSize);

    int indx = ix + gImgSize * iy;

    outputSamples[pixel.y * (int)iResolution.x + pixel.x] = indx;
    // ouputColor[pixel] = float4(normal, 0.0, 1.0);
    // ouputColor[pixel] = float4((1 + normal) / 2.0f, 0.0, 1.0);

}
