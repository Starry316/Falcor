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
    float gUVScale;
    bool gSynthesis;
    float2 gWo;
    float2 gWi;
};

RWBuffer<uint> testInput;

uint packInt8x4(int4 v)
{
    return (v.x & 0x000000ff) | ((v.y << 8) & 0x0000ff00) | ((v.z << 16) & 0x00ff0000) | ((v.w << 24) & 0xff000000);
}
uint packInt16x2(int2 v)
{
    return (v.x & 0x0000ffff) | ((v.y << 16) & 0xffff0000);
}


[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= iResolution))
        return;
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    float2 uv = (float2(pixel)) / iResolution.y;


    float2 h, d;
    dirToRusinkiewicz(sample_hemisphere(gWi), sample_hemisphere(gWo), h, d);
    float3 vh = spherical_to_cartesian_rad(h);
    float3 vd = spherical_to_cartesian_rad(d);

    h = hemisphereDirToDiskSquare(vh);
    d = hemisphereDirToDiskSquare(vd);

    // testInput[3 * (pixel.y * iResolution.x + pixel.x) + 2] = packUnorm2x16(uv);
    // testInput[3 * (pixel.y * iResolution.x + pixel.x)] = packUnorm2x16(h);
    // testInput[3 * (pixel.y * iResolution.x + pixel.x) + 1] = packUnorm2x16(d);
    testInput[2 * (pixel.y * iResolution.x + pixel.x)] = packUnorm2x16(h);
    testInput[2 * (pixel.y * iResolution.x + pixel.x) + 1] = packUnorm2x16(d);



}
