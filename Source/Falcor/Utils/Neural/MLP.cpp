#include "MLP.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include <fstream>
namespace Falcor
{
std::vector<float> readBinaryFile(const char* filename)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file)
    {
        logError("[MLP] Unable to open file {}", filename);
        return std::vector<float>();
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<float> buffer(size / sizeof(float));
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size))
    {
        logError("[MLP] Error reading file {}", filename);
        return std::vector<float>();
    }
    file.close();
    return buffer;
}

MLP::MLP(ref<Device> pDevice, std::string networkName)
{
    mNetworkName = networkName;
    std::filesystem::path projectDir = getProjectDirectory();
    std::vector<float> weightsBuffer =
        readBinaryFile(fmt::format("{}/media/BTF/networks/Weights_{}.bin", projectDir.string(), networkName).c_str());
    std::vector<float> biasBuffer =
        readBinaryFile(fmt::format("{}/media/BTF/networks/Bias_{}.bin", projectDir.string(), networkName).c_str());
    std::vector<float> metaBuffer =
        readBinaryFile(fmt::format("{}/media/BTF/networks/NNMeta_{}.bin", projectDir.string(), networkName).c_str());
    mLayerNum = metaBuffer[0];
    int totalWeightNum = 0;
    int totalBiasNum = 0;
    logInfo("[MLP] Layer num: {}", mLayerNum);
    for (int i = 1; i < mLayerNum * 2 + 1; i += 2)
    {
        logInfo("[MLP] {} -> {}", metaBuffer[i], metaBuffer[i + 1]);
        totalWeightNum += metaBuffer[i] * metaBuffer[i + 1] / 16;
        totalBiasNum += metaBuffer[i + 1] / 4;
        mMaxDim = std::max(mMaxDim, (int)std::max(metaBuffer[i], metaBuffer[i + 1]));
    }
    logInfo("[MLP] Layer num: {}, total weight num: {} total bias num: {}, max dim: {}", mLayerNum, totalWeightNum, totalBiasNum, mMaxDim);

    std::vector<float4x4> dummyWeights(totalWeightNum);
    std::vector<float4> dummyBias(totalBiasNum);

    for (int i = 0; i < totalWeightNum; i++)
    {
        dummyWeights[i] = math::matrixFromCoefficients<float, 4, 4>(weightsBuffer.data() + i * 16);
    }
    for (int i = 0; i < totalBiasNum; i++)
    {
        dummyBias[i] = float4(biasBuffer[i * 4], biasBuffer[i * 4 + 1], biasBuffer[i * 4 + 2], biasBuffer[i * 4 + 3]);
    }

    mpBias = pDevice->createBuffer(
        totalBiasNum * sizeof(float4),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        dummyBias.data()
    );
    mpWeights = pDevice->createBuffer(
        totalWeightNum * sizeof(float4x4),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        dummyWeights.data()
    );

    mpMeta = pDevice->createBuffer(
        (2 * mLayerNum + 1) * sizeof(float),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        metaBuffer.data()
    );
    std::vector<float>().swap(weightsBuffer);
    std::vector<float>().swap(biasBuffer);
    std::vector<float>().swap(metaBuffer);

    // Sampler::Desc samplerDesc;
    // samplerDesc.setAddressingMode(TextureAddressingMode::Wrap, TextureAddressingMode::Wrap, TextureAddressingMode::Wrap);
    // pSampler = pDevice->createSampler(samplerDesc);

    // Sampler::Desc samplerTileDesc;
    // samplerTileDesc.setAddressingMode(TextureAddressingMode::Clamp, TextureAddressingMode::Clamp, TextureAddressingMode::Clamp);
    // pTileSampler = pDevice->createSampler(samplerTileDesc);
}
void MLP::bindShaderData(const ShaderVar& var) const
{
    var["layerNum"] = mLayerNum;
    var["weights"] = mpWeights;
    var["bias"] = mpBias;
    var["meta"] = mpMeta;
}

} // namespace Falcor