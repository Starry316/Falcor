
cbuffer PerFrameCB
{
    uint2 gRenderTargetDim;
    int gLod;
}

Texture2D<float> gInputHeightMap;
RWTexture2D<float> gOutputHeightMap;



// [numthreads(16, 16, 1)]
// void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
// {
//     if (any(dispatchThreadId.xy >= gRenderTargetDim))
//         return;

//     int windowSize = pow(2, gLod);
//     const uint2 pixelPos = dispatchThreadId.xy;
//     float2 uv = (float2(pixelPos)) / gRenderTargetDim;
//     gOutputHeightMap[pixelPos] = 0;
//     for (int i = 0; i < windowSize; i++)
//     {
//         for (int j = 0; j < windowSize; j++) {
//             uint2 offset = uint2(i, j);
//             gOutputHeightMap[pixelPos] = max(gOutputHeightMap[pixelPos], gInputHeightMap[pixelPos * windowSize + offset]);
//         }
//     }

// }

void csMain(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    if (any(dispatchThreadId.xy >= gRenderTargetDim))
        return;
    const uint2 pixelPos = dispatchThreadId.xy;
    gOutputHeightMap[pixelPos] = gInputHeightMap[pixelPos];

}
