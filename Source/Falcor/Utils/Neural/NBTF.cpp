#include "NBTF.h"

#include "Core/Error.h"
#include "Core/Platform/OS.h"
#include "Core/API/Device.h"
#include <fstream>
namespace Falcor
{
std::vector<float> readBinaryFile(const char* filename);

void NBTF::loadFeature(ref<Device> pDevice, std::string featurePath)
{
    std::filesystem::path projectDir = getProjectDirectory();
    std::vector<float> PlaneMetaBuffer = readBinaryFile(fmt::format("{}/media/BTF/networks/PlaneMeta_{}.bin", projectDir.string(),featurePath).c_str());
    mUP.texDim = int2(PlaneMetaBuffer[0], PlaneMetaBuffer[1]);
    mHP.texDim = int2(PlaneMetaBuffer[2], PlaneMetaBuffer[3]);
    mDP.texDim = int2(PlaneMetaBuffer[4], PlaneMetaBuffer[5]);
    logInfo("[NBTF] Plane Dims");
    logInfo("[NBTF] U: {}, H: {}, D: {}", mUP.texDim, mHP.texDim, mDP.texDim);

    std::vector<float> DPlaneBuffer = readBinaryFile(fmt::format("{}/media/BTF/networks/DPlane_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> UPlaneBuffer = readBinaryFile(fmt::format("{}/media/BTF/networks/UPlane_{}.bin", projectDir.string(), featurePath).c_str());
    std::vector<float> HPlaneBuffer = readBinaryFile(fmt::format("{}/media/BTF/networks/HPlane_{}.bin", projectDir.string(), featurePath).c_str());
    mUP.featureTex = pDevice->createTexture2D(
        mUP.texDim.x,
        mUP.texDim.x,
        ResourceFormat::RGBA32Float,
        mUP.texDim.y,
        Resource::kMaxPossible,
        UPlaneBuffer.data(),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mDP.featureTex = pDevice->createTexture2D(
        mDP.texDim.x,
        mDP.texDim.x,
        ResourceFormat::RGBA32Float,
        mDP.texDim.y,
        1,
        DPlaneBuffer.data(),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );

    mHP.featureTex = pDevice->createTexture2D(
        mHP.texDim.x,
        mHP.texDim.x,
        ResourceFormat::RGBA32Float,
        mHP.texDim.y,
        1,
        HPlaneBuffer.data(),
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
    );
    std::vector<float>().swap(DPlaneBuffer);
    std::vector<float>().swap(HPlaneBuffer);
    std::vector<float>().swap(UPlaneBuffer);
}

NBTF::NBTF(ref<Device> pDevice, std::string networkName)
{
    mNetworkName = networkName;
    mpMLP = std::make_unique<MLP>(pDevice, networkName);
    loadFeature(pDevice, networkName);
}
void NBTF::bindShaderData(const ShaderVar& var) const
{
    mpMLP->bindShaderData(var["mlp"]);
    var["uDims"] = mUP.texDim;
    var["hDims"] = mHP.texDim;
    var["dDims"] = mDP.texDim;

    // var["CB"]["gHasT"] = hasT;
    // if (hasT)
    // {
    //     var["gTP"].setSrv(pTPTex->getSRV());
    //     var["gInvP"].setSrv(pTPInvTex->getSRV());
    // }
    var["uP"].setSrv(mUP.featureTex->getSRV());
    var["hP"].setSrv(mHP.featureTex->getSRV());
    var["dP"].setSrv(mDP.featureTex->getSRV());
}

} // namespace Falcor
