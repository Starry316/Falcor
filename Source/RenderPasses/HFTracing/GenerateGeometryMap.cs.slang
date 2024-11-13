import Scene.Scene;
import Scene.SceneTypes;
import Utils.Geometry.IntersectionHelpers;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    uint vertexCount;
    uint vbOffset;
    uint triangleCount;
    uint ibOffset;
    bool use16BitIndices;
    float4 gControlParas;
    uint gTriID;

}
SamplerState gSampler;

RWTexture2D<float4> gOutputColor;
RWTexture2D<float4> gOutputNormalMap;
RWTexture2D<float4> gOutputTangentMap;
RWTexture2D<float4> gOutputPosMap;




ParameterBlock<Scene> scene;

uint3 getMeshIndices(uint triangleId)
{
    if (triangleId >= triangleCount) return 0;
    uint3 vtxIndices = scene.getLocalIndices(ibOffset, triangleId, use16BitIndices);
    return vtxIndices;
}

StaticVertexData getMeshVertexData(uint vertexId)
{
    // if (vertexId >= vertexCount) return;
    StaticVertexData vtxData = scene.getVertex(vertexId + vbOffset);
    // positions = vtxData.position;
    // texcrds = float3(vtxData.texCrd, 0.f);
    return vtxData;
}

VertexData computeVD(StaticVertexData vertices[3], float3 barycentrics)
{
    GeometryInstanceID geoID;
    geoID.index = 0;
    const float4x4 worldMat = scene.getWorldMatrix(geoID);
    const float3x3 worldInvTransposeMat = scene.getInverseTransposeWorldMatrix(geoID);
    const bool isFrontFaceCW = scene.isObjectFrontFaceCW(geoID);
    return scene.computeVertexData(vertices, worldMat, worldInvTransposeMat, isFrontFaceCW, barycentrics);

}


[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;

    if (gTriID >= triangleCount) return;

    // const uint2 pixelPos = dispatchThreadId.yx;
    const uint2 pixelPos = dispatchThreadId.yx;
    float2 uv = (float2(pixelPos)) / gRenderTargetDim;
    // uv.x = 1-uv.x;
    float3 rayOrigin = float3(uv, 1);
    float3 rayDir = float3(0, 0, -1);

    float3 barycentrics;
    float dummyT;

    uint triangleId = gTriID;
    uint3 inds = getMeshIndices(triangleId);
    // float3 texc1, texc2, texc3, pos1, pos2, pos3;
    StaticVertexData vd1 = getMeshVertexData(inds.x);
    StaticVertexData vd2 = getMeshVertexData(inds.y);
    StaticVertexData vd3 = getMeshVertexData(inds.z);
    float3 vertices[3];
    vertices[0] = float3(vd1.texCrd, 0.f);
    vertices[1] = float3(vd2.texCrd, 0.f);
    vertices[2] = float3(vd3.texCrd, 0.f);

    if (intersectRayTriangle(rayOrigin, rayDir, vertices, barycentrics, dummyT)) {
        VertexData vd = computeVD( { vd1, vd2, vd3 }, barycentrics);
        if (isnan(vd.tangentW.x) || isnan(vd.tangentW.y) || isnan(vd.tangentW.z) || isnan(vd.tangentW.w))
            vd.tangentW = float4(0);
        gOutputNormalMap[pixelPos.xy] = float4(vd.normalW, 0.0);
        gOutputTangentMap[pixelPos.xy] = vd.tangentW;
        gOutputPosMap[pixelPos.xy].xyz = vd.posW;
    }
}
