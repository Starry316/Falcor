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
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
import Scene.Raster;
import Utils.Sampling.TinyUniformSampleGenerator;
import Rendering.Lights.LightHelpers;
import Utils.Math.MathHelpers;
cbuffer PerFrameCB
{
    uint2 gFrameDim;
    uint gBillboardRenderID;
    float gViewTheta;
    float gViewPhi;
    float3 gCameraPos;
    float4x4 viewMat;
    float4x4 projMat;
    bool gCreateBillboards;
    bool gShowColor;
};

RasterizerOrderedTexture2DArray<float4> gBillboards;
RasterizerOrderedTexture2DArray<float4> gBillboardColors;
RasterizerOrderedTexture2DArray<float4> gBillboardPosWs;
RasterizerOrderedTexture2DArray<float4> gBillboardNormalWs;

RasterizerOrderedTexture2D<float4> gReference;
StructuredBuffer<float4x4> gViewProjBuffer;
#define is_valid(name) (is_valid_##name != 0)


VSOut orthogonalVS(VSIn vIn)
{
    VSOut vOut;
    const GeometryInstanceID instanceID = { vIn.instanceID };

    float4x4 worldMat = gScene.getWorldMatrix(instanceID);
    float3 posW = mul(worldMat, float4(vIn.pos, 1.f)).xyz;
    vOut.posW = posW;

    vOut.posH = mul(gViewProjBuffer[gBillboardRenderID], float4(posW, 1.f));

    vOut.instanceID = instanceID;
    vOut.materialID = gScene.getMaterialID(instanceID);

    vOut.texC = vIn.texC;
    vOut.normalW = mul(gScene.getInverseTransposeWorldMatrix(instanceID), vIn.unpack().normal);
    float4 tangent = vIn.unpack().tangent;
    vOut.tangentW = float4(mul((float3x3)gScene.getWorldMatrix(instanceID), tangent.xyz), tangent.w);

    // Compute the vertex position in the previous frame.
    float3 prevPos = vIn.pos;
    GeometryInstanceData instance = gScene.getGeometryInstance(instanceID);
    if (instance.isDynamic())
    {
        uint prevVertexIndex = gScene.meshes[instance.geometryID].prevVbOffset + vIn.vertexID;
        prevPos = gScene.prevVertices[prevVertexIndex].position;
    }
    float3 prevPosW = mul(gScene.getPrevWorldMatrix(instanceID), float4(prevPos, 1.f)).xyz;
    vOut.prevPosH = mul(gScene.camera.data.prevViewProjMatNoJitter, float4(prevPosW, 1.f));

    return vOut;
}

VSOut vsMain(VSIn vIn)
{
    if(gCreateBillboards)
        return orthogonalVS(vIn);
    return defaultVS(vIn);
}
[earlydepthstencil]
float4 psMain(VSOut vsOut, uint triangleIndex: SV_PrimitiveID) : SV_TARGET
{
    let lod = ImplicitLodTextureSampler();
    if (alphaTest(vsOut, triangleIndex, lod))
        discard;

    uint2 pixel = uint2(vsOut.posH.xy - 0.5);

    float3 viewDir = normalize(gScene.camera.getPosition() - vsOut.posW);
    if (gCreateBillboards)
        viewDir = normalize(gCameraPos);
    ShadingData sd = prepareShadingData(vsOut, triangleIndex, viewDir);

    // // Create material instance.
    let mi = gScene.materials.getMaterialInstance(sd, lod);

    float3 color = mi.getProperties(sd).emission;

    TinyUniformSampleGenerator sg = TinyUniformSampleGenerator(pixel, 0);

    // // Direct lighting from analytic light sources
    for (int i = 0; i < gScene.getLightCount(); i++)
    {
        AnalyticLightSample ls;
        evalLightApproximate(sd.posW, gScene.getLight(i), ls);
        color += mi.eval(sd, ls.dir, sg) * ls.Li;
    }

    color = mi.getProperties(sd).diffuseReflectionAlbedo;
    // color = 0.5f * vsOut.normalW + 0.5f;

    if (!gShowColor)
        color = 0.5f * vsOut.posW + 0.5f;


    // color = mi.eval(sd, sd.faceN, sg);
    // gBillboards[uint3(pixel.x, pixel.y, gBillboardRenderID)] = float4(0.5f * vsOut.normalW + 0.5f, 1.0f);
    if (gCreateBillboards) {
        gBillboards[uint3(pixel.x, pixel.y, gBillboardRenderID)] = vsOut.posH.z;
        // gBillboardColors[uint3(pixel.x, pixel.y, gBillboardRenderID)] = float4(0.5f * vsOut.normalW + 0.5f, 1.0f);
        gBillboardColors[uint3(pixel.x, pixel.y, gBillboardRenderID)] = float4(color, 1.0f);
        gBillboardPosWs[uint3(pixel.x, pixel.y, gBillboardRenderID)] = float4(vsOut.posW, 1.0f);
        gBillboardNormalWs[uint3(pixel.x, pixel.y, gBillboardRenderID)] = float4(vsOut.normalW, 1.0f);

    }
    gReference[pixel] = float4(color, 1.0f);

    return float4(color, 1.0f);
}
