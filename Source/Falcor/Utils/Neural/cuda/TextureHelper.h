#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>
cudaTextureObject_t createCudaTextureArray(std::vector<float> data, int width, int height, int depth);

cudaTextureObject_t createCuda1DTextureArray(const std::vector<float>& data, int width,  int layers);
