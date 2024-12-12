import Utils.Neural.NBTF;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    bool gApplySyn;
    float4 gCurvatureParas;
    NBTF nbtf;
}
SamplerState gSampler;
//
RWTexture2D<float4> gOutputColor;
Texture2D<float4> wiWox;
Texture2D<float4> uvWoyz;
Texture2D<float4> dfDxy;

[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    float4 color = gOutputColor[pixel];
    if (color.w > 0.5f)
    {
        float4 angle1 = wiWox[pixel];
        float4 uvAngle2 = uvWoyz[pixel];
        float4 dfdxy = dfDxy[pixel];
        float3 wi = angle1.xyz;
        float3 wo = float3(angle1.w, uvAngle2.zw);
        float2 uv = uvAngle2.xy;
        gOutputColor[pixel].xyz *= nbtf.evalDebug(wi, wo, uv, dfdxy.xy, dfdxy.zw, gApplySyn);
    }


}
