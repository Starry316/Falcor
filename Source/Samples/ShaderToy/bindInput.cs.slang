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
cbuffer CB
{
    float2 iResolution;
    float gUVScaling;
    NBTF nbtf;
    NBTF nbtfInt8;
    bool gSynthesis;
    bool gFP16;
    int gRenderType;
    float2 gWo;
    float2 gWi;
};

Buffer<float> gQInt8Buffer;
RWBuffer<uint> cudaInputUIntBuffer;
RWBuffer<int> cudaInputIntBuffer;
RWBuffer<float> cudaInputBuffer;

uint packInt8x4(int4 v)
{
    return (v.x & 0x000000ff) | ((v.y << 8) & 0x0000ff00) | ((v.z << 16) & 0x00ff0000) | ((v.w << 24) & 0xff000000);
}
uint packInt16x2(int2 v)
{
    return (v.x & 0x0000ffff) | ((v.y << 16) & 0xffff0000);
}

void bindInputInt8(float2 uv, uint2 pixel)
{
    uint offset = 6 * (pixel.y * iResolution.x + pixel.x);
    float2 ddx = 0.0000001;
    float2 ddy = 0.0000001;
    float3 wi = sample_hemisphere(gWi);
    float3 wo = sample_hemisphere(gWo);
    nbtfInt8.bindInputInt8(cudaInputUIntBuffer, offset, wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);
}

void bindInputFP16(float2 uv, uint2 pixel)
{
    uint offset = 16 * (pixel.y * iResolution.x + pixel.x);
    float2 ddx = 0.0001;
    float2 ddy = 0.0001;
    float3 wi = sample_hemisphere(gWi);
    float3 wo = sample_hemisphere(gWo);

    nbtf.bindInputFP16(cudaInputUIntBuffer, offset, wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);
}

void bindInput(float2 uv, uint2 pixel)
{
    float2 ddx = 0.0001;
    float2 ddy = 0.0001;
    float3 wi = sample_hemisphere(gWi);
    float3 wo = sample_hemisphere(gWo);
    uint offset = 24 * (pixel.y * iResolution.x + pixel.x);
    nbtf.bindInput(cudaInputBuffer, offset, wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);
}
[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= iResolution))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    float2 uv = (float2(pixel)) / iResolution.y;

    if (gRenderType == 1)
        bindInput(uv, pixel);
    else if (gRenderType == 2)
        bindInputFP16(uv, pixel);
    else
        bindInputInt8(uv, pixel);

}
