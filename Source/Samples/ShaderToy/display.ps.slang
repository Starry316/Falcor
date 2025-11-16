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
#include "Utils/Math/MathConstants.slangh"
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;
import Utils.Neural.NNMat;
import Utils.Math.MathHelpers;
cbuffer CB
{
    float2 iResolution;
    NNMat nnmatT;
    NNMat nnmatBTF;
    float2 gWi;
    float2 gWo;
    bool gShowShader;
    bool gIsIBL;
    float gRealRes;

};
// cuda infer output
// Buffer<float> cudaColor;
// Texture2D<float4> ouputColor;

float4 mainImage(float2 texC, uint2 pixel)
{
    texC = (floor(texC * gRealRes) + 0.5)/ gRealRes;

    float uv_remapped[10];
    float3 wo = spherical_to_cartesian_rad(float2(gWo.x * M_PI_2, gWo.y * M_2PI));
    float3 wi = spherical_to_cartesian_rad(float2(gWi.x * M_PI_2, gWi.y * M_2PI));
    float2 uv = texC;
    print("texC", texC);
    print("pixel", pixel);
    print("wi", wi);
    print("wo", wo);

    if (gIsIBL)
    {
        uv = gWi;

        wi = squareToDiskHemisphereShirley(texC);
    }

    nnmatT.evalT(wo, uv, uv_remapped);
    float3 fr = nnmatBTF.evalBTF(wi, wo, uv_remapped);

    return float4(fr, 1.0);
}

float4 main(float2 texC: TEXCOORD) : SV_TARGET
{
    uint2 pixel = uint2(floor(texC * iResolution));
    printSetPixel(pixel);
    return mainImage(texC, pixel);
}
