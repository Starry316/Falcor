// Translate Falcor 9 material storage into the original renderer's BSDF inputs.
// The actual BSDF evaluation and sampling remain in Legacy/Material.
import Scene.Scene;
import Scene.Material.TextureSampler;
import Utils.Math.MathHelpers;
import Utils.Geometry.GeometryHelpers;
import RenderPasses.NeuralRadianceCache.Legacy.Material.IBxDF;
__exported import RenderPasses.NeuralRadianceCache.Legacy.ShadingData;

LegacyShadingData prepareLegacyShadingData<L : ITextureSampler>(VertexData v, uint materialID, float3 viewDir, L lod)
{
    LegacyShadingData sd = {};
    let ms = gScene.materials;
    let header = ms.getMaterialHeader(materialID);
    let data = ms.getBasicMaterialData(materialID);
    let sampler = ms.getTextureSampler(header.getDefaultTextureSamplerID());
    float4 base = ms.sampleTexture(data.texBaseColor, sampler, v.texC, data.baseColor, lod);
    float4 spec = ms.sampleTexture(data.texSpecular, sampler, v.texC, data.specular, lod);
    sd.posW = v.posW;
    sd.V = viewDir;
    sd.N = v.normalW;
    float nt = dot(v.tangentW.xyz, sd.N);
    bool validTangentSpace = v.tangentW.w != 0.f && dot(v.tangentW.xyz,v.tangentW.xyz) > 0.f && abs(nt) < .9999f;
    sd.T = validTangentSpace ? normalize(v.tangentW.xyz - sd.N * nt) : perp_stark(sd.N);
    sd.B = cross(sd.N, sd.T) * (validTangentSpace ? v.tangentW.w : 1.f);
    sd.faceN = v.faceNormalW;
    sd.frontFacing = dot(viewDir, sd.faceN) >= 0.f;
    sd.uv = v.texC;
    sd.materialID = materialID;
    sd.setMaterialType(header.getMaterialType());
    sd.setDoubleSided(header.isDoubleSided());
    sd.setThinSurface(header.isThinSurface());
    sd.setActiveLobes((uint)LegacyLobeType::All);
    sd.opacity = base.a;
    sd.IoR = header.getIoR();
    sd.eta = sd.frontFacing ? 1.f / sd.IoR : sd.IoR;
    float f = (sd.IoR - 1.f) / (sd.IoR + 1.f);
    if (data.getShadingModel() == ShadingModel::MetalRough)
    {
        sd.diffuse = base.rgb * (1.f - spec.b);
        sd.specular = lerp(float3(f*f), base.rgb, spec.b);
        sd.linearRoughness = spec.g;
        sd.metallic = spec.b;
    }
    else
    {
        sd.diffuse = base.rgb;
        sd.specular = spec.rgb;
        sd.linearRoughness = 1.f - spec.a;
        float d = dot(sd.diffuse, float3(.2126,.7152,.0722));
        float s = dot(sd.specular, float3(.2126,.7152,.0722));
        float b = s + d - .08f;
        sd.metallic = s == 0.f ? 0.f : (sqrt(b*b - .16f*(.04f-s)) - b) * 12.5f;
    }
    sd.ggxAlpha = max(.0064f, sd.linearRoughness * sd.linearRoughness);
    sd.diffuseTransmission = data.diffuseTransmission;
    sd.specularTransmission = data.specularTransmission;
    if (sd.diffuseTransmission > 0.f || sd.specularTransmission > 0.f)
        sd.transmission = ms.sampleTexture(data.texTransmission, sampler, v.texC, float4(data.transmission, 0), lod).rgb;
    if (sd.frontFacing) sd.emissive = ms.sampleTexture(data.texEmissive, sampler, v.texC, float4(data.emissive, 1), lod).rgb * data.emissiveFactor;
    if (validTangentSpace && data.getNormalMapType() != NormalMapType::None)
    {
        float3 n = ms.sampleTexture(data.texNormalMap, sampler, v.texC, float4(0), lod).rgb;
        // Preserve the research renderer's RG normal decoding for alignment.
        float rgZ = sqrt(1.f - saturate(dot(n.xy,n.xy)));
        n = n * 2.f - 1.f;
        if (data.getNormalMapType() == NormalMapType::RG) n.z = rgZ;
        sd.N = normalize(sd.T*n.x + sd.B*n.y + sd.N*n.z);
        sd.T = normalize(v.tangentW.xyz - sd.N*dot(v.tangentW.xyz,sd.N));
        sd.B = cross(sd.N,sd.T) * v.tangentW.w;
    }
    sd.NdotV = dot(sd.N, sd.V);
    if (!sd.frontFacing && sd.isDoubleSided()) { sd.N = -sd.N; sd.NdotV = -sd.NdotV; }
    return sd;
}

LegacyShadingData prepareLegacyShadingData(VertexData v, uint materialID, float3 viewDir, float lod)
{ return prepareLegacyShadingData(v, materialID, viewDir, ExplicitLodTextureSampler(lod)); }

void adjustShadingNormal(inout LegacyShadingData sd, VertexData v)
{
    float cosine = dot(sd.V,sd.N);
    if (cosine <= .1f)
    {
        float3 ng = sd.frontFacing ? v.faceNormalW : -v.faceNormalW;
        sd.N = normalize(lerp(ng, sd.N, saturate(cosine * 10.f)));
        sd.NdotV = dot(sd.N,sd.V);
        float nt = dot(v.tangentW.xyz,sd.N);
        bool valid = v.tangentW.w != 0.f && dot(v.tangentW.xyz,v.tangentW.xyz) > 0.f && abs(nt) < .9999f;
        sd.T = valid ? normalize(v.tangentW.xyz-sd.N*nt) : perp_stark(sd.N);
        sd.B = cross(sd.N,sd.T) * (valid ? v.tangentW.w : 1.f);
    }
}
