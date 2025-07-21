#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>
#include "MLP.h"
#include "Utils/Texture/Synthesis.h"
namespace Falcor
{
struct NNFeatureTex
{
    int2 texDim;
    ref<Texture> featureTex;
};

class FALCOR_API NNMat
{
public:
    NNMat(ref<Device> pDevice, std::string networkPath);

    void loadFeature(ref<Device> pDevice, std::string featurePath);

    void bindShaderData(const ShaderVar& var) const;

    NNFeatureTex mHP;
    NNFeatureTex mDP;
    NNFeatureTex mUP;
    NNFeatureTex mUFP;

    std::unique_ptr<MLP> mpMLP;
    ref<Sampler> mpPointSampler;

    std::string mNetworkName;

    int mLayerNum;
    int mMaxDim;
};

} // namespace Falcor
