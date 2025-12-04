#include "MLP.h"
#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "IOHelper.h"
#include <fstream>
namespace Falcor
{

MLP::MLP(ref<Device> pDevice, std::string networkName, bool isT, bool isBTFMLP)
{
    mNetworkName = networkName;
    std::filesystem::path projectDir = getProjectDirectory();
    if (isT){

        std::vector<float> weightsBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/btf/t_weight_{}.bin", projectDir.string(), networkName).c_str());
        std::vector<float> biasBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/btf/t_bias_{}.bin", projectDir.string(), networkName).c_str());

        logInfo("[MLP] T weight size {} T bias size {}", weightsBuffer.size(), biasBuffer.size());
        logInfo("[MLP] T bias {} {} {} {}  ", biasBuffer[0], biasBuffer[1], biasBuffer[2], biasBuffer[3]);
        mpTWeights = pDevice->createBuffer(
            weightsBuffer.size() * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            weightsBuffer.data()
        );
        mpTBias = pDevice->createBuffer(
            biasBuffer.size() * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            biasBuffer.data()
        );
    }
    else if(isBTFMLP){
        std::vector<float> weightsBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/btf/mlp_weight_{}.bin", projectDir.string(), networkName).c_str());
        std::vector<float> biasBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/btf/mlp_bias_{}.bin", projectDir.string(), networkName).c_str());

        logInfo("[MLP] MLP weight size {} MLP bias size {}", weightsBuffer.size(), biasBuffer.size());
        mpTWeights = pDevice->createBuffer(
            weightsBuffer.size() * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            weightsBuffer.data()
        );
        mpTBias = pDevice->createBuffer(
            biasBuffer.size() * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            biasBuffer.data()
        );


    }
    else{
        // weight_
        // std::vector<float> weightsBuffer =
        //     readBinaryFile(fmt::format("{}/media/BTF/networks/Weights_{}.bin", projectDir.string(), networkName).c_str());
        // std::vector<float> biasBuffer =
        //     readBinaryFile(fmt::format("{}/media/BTF/networks/Bias_{}.bin", projectDir.string(), networkName).c_str());
        // std::vector<float> metaBuffer =
        //     readBinaryFile(fmt::format("{}/media/BTF/networks/NNMeta_{}.bin", projectDir.string(), networkName).c_str());

        std::vector<float> weightsBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/networks/weight_{}.bin", projectDir.string(), networkName).c_str());


        // std::vector<float> biasBuffer =
        //     readBinaryFile(fmt::format("{}/media/nn_mat/networks/Bias_{}.bin", projectDir.string(), networkName).c_str());
        std::vector<float> metaBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/networks/meta_{}.bin", projectDir.string(), networkName).c_str());

        std::vector<float> debugWeightBuffer =
            readBinaryFile(fmt::format("{}/media/nn_mat/networks/weight_linear_{}.bin", projectDir.string(), networkName).c_str());

        // for (int i = 0; i < metaBuffer.size() ; i ++){
        //     logInfo("[MLP] META {} ", metaBuffer[i]);
        // }

        mLayerNum = metaBuffer[0];
        int totalWeightNum = 0;
        int totalBiasNum = 0;
        logInfo("[MLP] {} Layer num: {}", networkName, mLayerNum);
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
        // for (int i = 0; i < totalBiasNum; i++)
        // {
        //     dummyBias[i] = float4(biasBuffer[i * 4], biasBuffer[i * 4 + 1], biasBuffer[i * 4 + 2], biasBuffer[i * 4 + 3]);
        // }

        // mpBias = pDevice->createBuffer(
        //     totalBiasNum * sizeof(float4),
        //     ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        //     MemoryType::DeviceLocal,
        //     dummyBias.data()
        // );
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

        logInfo("[MLP] debugWeights size {}",debugWeightBuffer.size() );
        mpDebugWeights = pDevice->createBuffer(
            debugWeightBuffer.size() * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            debugWeightBuffer.data()
        );

        std::vector<float>().swap(weightsBuffer);
        // std::vector<float>().swap(biasBuffer);
        std::vector<float>().swap(metaBuffer);
        std::vector<float>().swap(debugWeightBuffer);
    }





}
void MLP::bindShaderData(const ShaderVar& var) const
{
    var["layerNum"] = mLayerNum;
    var["weights"] = mpWeights;
    // var["bias"] = mpBias;
    var["meta"] = mpMeta;
    var["debugWeights"] = mpDebugWeights;
}
void MLP::bindTData(const ShaderVar& var) const
{
    var["tWeights"] = mpTWeights;
    var["tBias"] = mpTBias;
}
void MLP::bindDebugData(const ShaderVar& var, ref<Buffer> w) const
{
    var["debugWeights"] = w;
}

} // namespace Falcor
