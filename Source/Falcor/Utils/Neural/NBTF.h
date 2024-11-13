#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>
#include "MLP.h"

namespace Falcor
{
struct FeatureTex{
    int2 texDim;
    ref<Texture> featureTex;
};



class FALCOR_API NBTF
{
public:

    NBTF(ref<Device> pDevice, std::string networkPath);

    void loadFeature(ref<Device> pDevice, std::string featurePath);

    void bindShaderData(const ShaderVar& var) const;


private:
    FeatureTex mHP;
    FeatureTex mDP;
    FeatureTex mUP;
    FeatureTex mTP;
    FeatureTex mTPInv;

    std::unique_ptr<MLP> mpMLP;
    std::string mNetworkName;
    int mLayerNum;
    int mMaxDim;

    // Synthesis parameters

};

}
