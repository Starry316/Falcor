import Utils.Debug.PixelDebug;
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
}
RWTexture2D<float4> gOutputColor;

float rnd21(float2 p)
{
    float temp = sin(12.9898f * p.x + 78.233f * p.y) * 43758.5453f;
    return temp - floor(temp);
}

float2 hash(float2 p)
{
    float2 r = float2(127.1f * p.x + 311.7f * p.y, 269.5f * p.x + 183.3f * p.y);
    //float2 temp = sin(r) * 43758.5453f;
    float2 temp = sin(r);
    //return (temp - floor(temp));
    return temp;
}

Buffer<float> gInputColor;
Buffer<int> cudaVaildBuffer;
[numthreads(16, 16, 1)]
void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);

    if (cudaVaildBuffer[pixel.y * gRenderTargetDim.x + pixel.x] == 0)
        return;
    float4 btfVal = float4(
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 0],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 1],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 2],
        gInputColor[4 * (pixel.y * gRenderTargetDim.x + pixel.x) + 3]
        //1.0
    );

    if (any(isnan(btfVal)) || any(isinf(btfVal)))
        btfVal = float4(0);

    //print("Li", uvWoyz[pixel]);
    print("Li", btfVal);

    //float2 t0 = float2(0.3f, 0.5f);
    //float2 t1 = float2(2.1f, 4.7f);
    //float4 test_hash = float4(rnd21(t0), rnd21(t1), 0, 1);
    //float4 test_hash = float4(floor(rnd21(t0) * 2048), floor(rnd21(t1) * 2048), 0, 1);

    float2 uv = btfVal.xy;
    float2 t1 = floor(uv + float2(0.5f, 0.5f)) * 2.0f + 1.0f;
    float4 test_hash = float4(uv.x, uv.y, 12.9898f * t1.x + 78.233f * t1.y, sin(12.9898f * t1.x + 78.233f * t1.y) * 43758.5453f);
    print("hash", test_hash);
    print("test", float4(sin(1211.876f) * 43758.5453f, 0, 0, 0));

    gOutputColor[pixel] *= btfVal;
}
