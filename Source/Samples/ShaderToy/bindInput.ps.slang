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
cbuffer CB
{
    float2 iResolution;
    float gUVScaling;
    NBTF nbtf;
    NBTF nbtfInt8;
    bool gSynthesis;
    bool gFP16;
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

void bindInputUV(float2 fragCoord, uint2 pixel)
{
    uint offset = 6 * (pixel.y * iResolution.x + pixel.x);
    float2 uv = fragCoord / iResolution.y;
    float2 ddx = 0.0001;
    float2 ddy = 0.0001;
    float3 wi = normalize(float3(0.2, 0.2, 1.0));
    float3 wo = normalize(float3(0.3, 0.2, 0.8));
    nbtfInt8.bindInputInt8(cudaInputUIntBuffer, offset, wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);
}


// void bindInputUV(float2 fragCoord, uint2 pixel)
// {
//     uint offset = 16 * (pixel.y * iResolution.x + pixel.x);

//     int4 w1 = int4(12, 2, 32, 46);
//     int4 w2 = int4(3, 21, 6, -2);
//     int4 w3 = int4(-3, -21, 6, -2);
//     int2 inputData = int2(5, -16);
//     cudaInputUIntBuffer[offset + 1] = packInt8x4(w1);
//     cudaInputUIntBuffer[offset + 2] = packInt8x4(w2);
//     cudaInputUIntBuffer[offset + 3] = packInt8x4(w3);
//     cudaInputUIntBuffer[offset] = packInt16x2(inputData);

// }


void bindInputFP16(float2 fragCoord, uint2 pixel)
{

    uint offset = 16 * (pixel.y * iResolution.x + pixel.x);
    float2 uv = fragCoord / iResolution.y;
    float2 ddx = 0.0001;
    float2 ddy = 0.0001;
    float3 wi = normalize(float3(0.2, 0.2, 1.0));
    float3 wo = normalize(float3(0.3, 0.2, 0.8));

    nbtf.bindInputFP16(cudaInputUIntBuffer, offset, wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);
}

void bindInput(float2 fragCoord, uint2 pixel)
{


    float2 uv = fragCoord / iResolution.y;
    float2 ddx = 0.0001;
    float2 ddy = 0.0001;
    float3 wi = normalize(float3(0.2, 0.2, 1.0));
    float3 wo = normalize(float3(0.3, 0.2, 0.8));
    uint offset = 32 * (pixel.y * iResolution.x + pixel.x);
    nbtf.bindInput(cudaInputBuffer, offset, wi, wo, uv * gUVScaling, ddx, ddy, gSynthesis);
}


float4 main(float2 texC: TEXCOORD) : SV_TARGET
{
    uint2 pixel = uint2(floor(texC * iResolution));
    printSetPixel(pixel);
    if (gFP16)
        bindInputUV(texC * iResolution, pixel);

    else
        bindInputFP16(texC * iResolution, pixel);
        // bindInput(texC * iResolution, pixel);
    return float4(0.0, 0.0, 0.0, 1.0);
}
