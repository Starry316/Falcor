#include "cudaTextureHelper.h"
#include <iostream>
cudaTextureObject_t createCudaTextureArray(std::vector<float> data, int width, int height, int layers)
{

    // 创建 CUDA 通道描述符
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();

    // 分配 CUDA 分层数组
    cudaExtent extent = make_cudaExtent(width, height, layers);
    cudaArray_t cuArray;
    cudaMalloc3DArray(&cuArray, &channelDesc, extent, cudaArrayLayered);

    // 将数据复制到 CUDA 分层数组
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)data.data(), width * sizeof(float4), width, height);
    copyParams.dstArray = cuArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copyParams);

    // 创建 CUDA 资源描述符
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 创建 CUDA 纹理描述符
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.addressMode[2] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // 创建 CUDA 纹理对象
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    return texObj;
}


cudaTextureObject_t createCuda2DTexture(std::vector<float> data, int width, int height)
{

    // 创建 CUDA 通道描述符
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();


    std::cout << "width: " << width << " height: " << height <<  std::endl;
    std::cout << "data size: " << data.size() << std::endl;


    // 分配 CUDA 数组
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    // 将数据复制到 CUDA 数组
    cudaMemcpy2DToArray(cuArray, 0, 0, data.data(), width * sizeof(float4), width * sizeof(float4), height, cudaMemcpyHostToDevice);

    // 创建 CUDA 资源描述符
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // 创建 CUDA 纹理描述符
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // 创建 CUDA 纹理对象
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    return texObj;
}

