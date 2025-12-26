#include "NNMat.h"

#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include "Utils/Math/FormatConversion.h"
#include "IOHelper.h"
#include <fstream>
namespace Falcor
{
std::vector<float> readBinaryFile(const char* filename);

void NNMat::loadFeature(ref<Device> pDevice, std::string featurePath)
{
    std::filesystem::path projectDir = getProjectDirectory();

    std::vector<float> PlaneMetaBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/networks/tex_meta_{}.bin", projectDir.string(), featurePath).c_str());

    mUP.texDim = int2(PlaneMetaBuffer[0], PlaneMetaBuffer[1]);
    mUFP.texDim = int2(PlaneMetaBuffer[2], PlaneMetaBuffer[3]);
    mHP.texDim = int2(PlaneMetaBuffer[4], PlaneMetaBuffer[5]);
    mDP.texDim = int2(PlaneMetaBuffer[6], PlaneMetaBuffer[7]);

    logInfo("[NNMat] Plane Dims");
    logInfo("[NNMat] U: {}, U: {}, H: {}, D: {}", mUP.texDim, mUFP.texDim, mHP.texDim, mDP.texDim);

    std::vector<float> DPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/networks/D_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> UPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/networks/U_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> UFPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/networks/UF_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> HPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/networks/H_{}.bin", projectDir.string(), featurePath).c_str());

    ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;


    mUP.featureTex =
        pDevice->createTexture2D(mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y, 1, UPlaneBuffer.data(), bindFlags);
    mUFP.featureTex = pDevice->createTexture2D(
        mUFP.texDim.x, mUFP.texDim.x, ResourceFormat::RGBA32Float, mUFP.texDim.y, 1, UFPlaneBuffer.data(), bindFlags
    );

    mDP.featureTex =
        pDevice->createTexture2D(mDP.texDim.x, mDP.texDim.x, ResourceFormat::RGBA32Float, mDP.texDim.y, 1, DPlaneBuffer.data(), bindFlags);

    mHP.featureTex =
        pDevice->createTexture2D(mHP.texDim.x, mHP.texDim.x, ResourceFormat::RGBA32Float, mHP.texDim.y, 1, HPlaneBuffer.data(), bindFlags);

}

void NNMat::loadBTFFeature(ref<Device> pDevice, std::string featurePath)
{
    std::filesystem::path projectDir = getProjectDirectory();

    std::vector<float> PlaneMetaBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/btf/tex_meta_{}.bin", projectDir.string(), featurePath).c_str());

    mUP.texDim = int2(PlaneMetaBuffer[0], PlaneMetaBuffer[1]);
    mHP.texDim = int2(PlaneMetaBuffer[2], PlaneMetaBuffer[3]);
    mDP.texDim = int2(PlaneMetaBuffer[4], PlaneMetaBuffer[5]);

    int uPLayerNum = 5;
    if (PlaneMetaBuffer.size() > 6)
    {
        uPLayerNum = int(PlaneMetaBuffer[6]);
    }

    logInfo("[NNMatBTF] Plane Dims");
    logInfo("[NNMatBTF] U: {}, H: {}, D: {}", mUP.texDim,  mHP.texDim, mDP.texDim);
    logInfo("[NNMatBTF] U Plane num: {}", uPLayerNum);


    ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;

    std::vector<float> DPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/btf/D_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> HPlaneBuffer =
        readBinaryFile(fmt::format("{}/media/nn_mat/btf/H_{}.bin", projectDir.string(), featurePath).c_str());
    mDP.featureTex =
        pDevice->createTexture2D(mDP.texDim.x, mDP.texDim.x, ResourceFormat::RGBA32Float, mDP.texDim.y, 1, DPlaneBuffer.data(), bindFlags);

    mHP.featureTex =
        pDevice->createTexture2D(mHP.texDim.x, mHP.texDim.x, ResourceFormat::RGBA32Float, mHP.texDim.y, 1, HPlaneBuffer.data(), bindFlags);


    std::vector<float> UPlaneBuffer;
    UPlaneBuffer=  readBinaryFile(fmt::format("{}/media/nn_mat/btf/U_{}.bin", projectDir.string(), featurePath).c_str());
    mUP1.featureTex = pDevice->createTexture2D(mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y * uPLayerNum, 1, UPlaneBuffer.data(), bindFlags);

    // UPlaneBuffer=  readBinaryFile(fmt::format("{}/media/nn_mat/btf/U2_{}.bin", projectDir.string(), featurePath).c_str());
    // mUP2.featureTex = pDevice->createTexture2D(mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y, 1, UPlaneBuffer.data(), bindFlags);

    // UPlaneBuffer=  readBinaryFile(fmt::format("{}/media/nn_mat/btf/U3_{}.bin", projectDir.string(), featurePath).c_str());
    // mUP3.featureTex = pDevice->createTexture2D(mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y, 1, UPlaneBuffer.data(), bindFlags);

    // UPlaneBuffer=  readBinaryFile(fmt::format("{}/media/nn_mat/btf/U4_{}.bin", projectDir.string(), featurePath).c_str());
    // mUP4.featureTex = pDevice->createTexture2D(mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y, 1, UPlaneBuffer.data(), bindFlags);

    // UPlaneBuffer=  readBinaryFile(fmt::format("{}/media/nn_mat/btf/U5_{}.bin", projectDir.string(), featurePath).c_str());
    // mUP5.featureTex = pDevice->createTexture2D(mUP.texDim.x, mUP.texDim.x, ResourceFormat::RGBA32Float, mUP.texDim.y, 1, UPlaneBuffer.data(), bindFlags);

}


NNMat::NNMat(ref<Device> pDevice, std::string networkName, bool isT, bool isWi)
{
    mNetworkName = networkName;
    mIsWi = isWi;
    // mpTextureSynthesis = std::make_unique<TextureSynthesis>();
    if(!isT && !isWi){
        loadFeature(pDevice, networkName);
        mpMLP = std::make_unique<MLP>(pDevice, networkName, false, false);
    }
    if(isWi){
        loadBTFFeature(pDevice, networkName);
        mpMLP = std::make_unique<MLP>(pDevice, networkName, false, true);
    }
    if(isT)
        mpMLP = std::make_unique<MLP>(pDevice, networkName, true, false);

    Sampler::Desc samplerDesc = Sampler::Desc();
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
    samplerDesc.setAddressingMode(
        TextureAddressingMode::Border,
        TextureAddressingMode::Border,
        TextureAddressingMode::Border
    );
    mpPointSampler = pDevice->createSampler(samplerDesc);


}

void NNMat::bindShaderData(const ShaderVar& var) const
{
    mpMLP->bindShaderData(var["mlp"]);

    var["uDims"] = mUP.texDim;
    var["hDims"] = mHP.texDim;
    var["dDims"] = mDP.texDim;

    var["uP"].setSrv(mUP.featureTex->getSRV());
    var["uFP"].setSrv(mUFP.featureTex->getSRV());
    var["hP"].setSrv(mHP.featureTex->getSRV());
    var["dP"].setSrv(mDP.featureTex->getSRV());





    var["inputSize"] = (mDP.texDim.y + mHP.texDim.y + mUP.texDim.y) * 4;


    var["gPointSampler"] = mpPointSampler;
}
void NNMat::bindShaderDataT(const ShaderVar& var) const
{
    mpMLP->bindTData(var["mlp"]);
    if(mIsWi){
        var["uDims"] = mUP1.texDim;
        var["hDims"] = mHP.texDim;
        var["dDims"] = mDP.texDim;

        var["hP"].setSrv(mHP.featureTex->getSRV());
        var["dP"].setSrv(mDP.featureTex->getSRV());

        var["u1P"].setSrv(mUP1.featureTex->getSRV());
        // var["u2P"].setSrv(mUP2.featureTex->getSRV());
        // var["u3P"].setSrv(mUP3.featureTex->getSRV());
        // var["u4P"].setSrv(mUP4.featureTex->getSRV());
        // var["u5P"].setSrv(mUP5.featureTex->getSRV());
    }
    // var["uDims"] = mUP.texDim;
    // var["hDims"] = mHP.texDim;
    // var["dDims"] = mDP.texDim;

    // var["uP"].setSrv(mUP.featureTex->getSRV());
    // var["uFP"].setSrv(mUFP.featureTex->getSRV());






    // var["inputSize"] = (mDP.texDim.y + mHP.texDim.y + mUP.texDim.y) * 4;


    // var["gPointSampler"] = mpPointSampler;
}
} // namespace Falcor
