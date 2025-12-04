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
    float3 wo = spherical_to_cartesian_rad(float2(gWo.x * M_PI, gWo.y * M_2PI));
    float3 wi = spherical_to_cartesian_rad(float2(gWi.x * M_PI, gWi.y * M_2PI));
    float2 uv = texC;
    // uv = float2(0.5, 0.5);


    float3 pos = squareToDiskHemisphereShirley(uv);
    pos = normalizeSafe(pos);
    wo = normalizeSafe(wo);


    print("uv", uv);
    print("pixel", pixel);
    print("wi", wi);
    print("wo", wo);
    print("pos", pos);

    float3 pos1, pos2;
    nnmatT.evalTTriDoubleSidePos(wo, uv, pos1, pos2);


    // nnmatT.evalTTriDoubleSide(wo, frac(uv), pos1, pos2);
    // ret = float3(nnmatBTF.evalLFTriDoubleSide(float3(0), wo, pos1, pos2));

    // float inputVal[6];
    // inputVal[0] = -wo.x;
    // inputVal[1] = -wo.y;
    // inputVal[2] = -wo.z;
    // inputVal[3] = pos.x;
    // inputVal[4] = pos.y;
    // inputVal[5] = pos.z;
    // float2 t = nnmatT.mlp.evalTTriMLP(inputVal);
    // pos1 = 0.5 * tanh(-t.x * wo + pos) + 0.5;
    // pos2 = 0.5 * tanh(-t.y * wo + pos) + 0.5;

    float3 fr = nnmatBTF.evalLFTriDoubleSideFull(wi, wo, pos1, pos2);

    // print("pos1", pos1);
    // print("pos2", pos2);
    // print("t", t);

    // print("tw0", nnmatBTF.mlp.tBias[0]);
    // print("tw1", nnmatBTF.mlp.tBias[1]);
    // print("tw2", nnmatBTF.mlp.tBias[2]);
    // print("tw3", nnmatBTF.mlp.tBias[3]);
    // print("tw4", nnmatBTF.mlp.tBias[4]);

    // print("uv1", float2(uv_remapped[2], uv_remapped[3]));
    // print("uv2", float2(uv_remapped[4], uv_remapped[5]));
    // print("uv3", float2(uv_remapped[6], uv_remapped[7]));
    // print("uv4", float2(uv_remapped[8], uv_remapped[9]));

    // // float3 fr = nnmatBTF.evalFeatureMap(wi, wo, uv_remapped, 4);


    // float3 fr = nnmatBTF.evalBTF(wi, wo, uv_remapped);

    // print("fr", fr);

    return float4(fr, 1.0);

}

float4 main(float2 texC: TEXCOORD) : SV_TARGET
{
    uint2 pixel = uint2(floor(texC * iResolution));
    printSetPixel(pixel);
    return mainImage(texC, pixel);
}
