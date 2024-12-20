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
import Utils.Texture.HistogramBlending;
import Utils.Neural.NBTF;
import Utils.Debug.PixelDebug;
import Utils.Math.FormatConversion;
import Utils.Math.MathHelpers;
cbuffer ToyCB
{
    float2 iResolution;
    float iGlobalTime;
    HistogramBlendingHFData hfData;
    float gUVScaling;
    NBTF nbtf;
    bool gSynthesis;
    bool gDisplay;
    float2 gWo;
    float2 gWi;
};

Buffer<float> gInputColor;
RWBuffer<uint> cudaInputUIntBuffer;
RWBuffer<float> cudaInputBuffer;
SamplerState gSampler;


float4 mainImage(float2 fragCoord, uint2 pixel)
{
    float2 uv = fragCoord / iResolution.y;

    float2 ddx = 0.0000001;
    float2 ddy = 0.0000001;

    float3 wi = sample_hemisphere(gWi);
    float3 wo = sample_hemisphere(gWo);
    float2 h = float2(0.951858, 0.01944);
    float2 d = float2(0.875318, 0.58938);
    return float4(nbtf.evalDebug(wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis), 1.0);
}



float4 main(float2 texC: TEXCOORD) : SV_TARGET
{
    uint2 pixel = uint2(floor(texC * iResolution));
    printSetPixel(pixel);
    return mainImage(texC * iResolution, pixel);

}
