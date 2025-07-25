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

    // std::vector<float>().swap(DPlaneBuffer);
    // std::vector<float>().swap(HPlaneBuffer);
    // std::vector<float>().swap(UPlaneBuffer);
}

void NNMat::loadLUTs(ref<Device> pDevice, std::string lutPath)
{
    std::filesystem::path projectDir = getProjectDirectory();
    ResourceBindFlags bindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess;
    mLUTs.featureTex = Texture::createFromFile(pDevice, fmt::format("{}/media/nn_mat/networks/LUT_{}.exr", projectDir.string(), lutPath), false, false,bindFlags,Bitmap::ImportFlags::None);
    mLUTs.featureTex->getWidth();
    // logInfo("LUT =============== {}", mLUTs.featureTex->getWidth());
    // logInfo("{}", mLUTs.featureTex->getFormat());


}

NNMat::NNMat(ref<Device> pDevice, std::string networkName, bool isHisto, bool isWi)
{
    mNetworkName = networkName;

    // mpTextureSynthesis = std::make_unique<TextureSynthesis>();
    loadFeature(pDevice, networkName);
    mIsWi = isWi;
    mIsHisto = isHisto;
    if (mIsHisto)
        loadLUTs(pDevice, networkName);
    

    mpMLP = std::make_unique<MLP>(pDevice, networkName);

    Sampler::Desc samplerDesc = Sampler::Desc();
    samplerDesc.setFilterMode(TextureFilteringMode::Point, TextureFilteringMode::Point, TextureFilteringMode::Point);
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

    if (mIsHisto)
        var["LUTs"].setSrv(mLUTs.featureTex->getSRV());
    var["isHisto"] = mIsHisto;
    var["isWi"] = mIsWi;

  

    var["inputSize"] = (mDP.texDim.y + mHP.texDim.y + mUP.texDim.y) * 4;


    var["gPointSampler"] = mpPointSampler;
}

} // namespace Falcor
