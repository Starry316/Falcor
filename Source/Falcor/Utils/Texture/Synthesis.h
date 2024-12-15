#pragma once
#include "Core/Macros.h"
#include "Core/API/Buffer.h"
#include "Core/Program/ShaderVar.h"
#include <memory>
#include <random>

namespace Falcor
{
/**
 * Implements Texture Synthesis
 */
class FALCOR_API TextureSynthesis
{
public:
    /**
     * Create an alias table.
     * The weights don't need to be normalized to sum up to 1.
     * @param[in] pDevice GPU device.
     * @param[in] weights The weights we'd like to sample each entry proportional to.
     * @param[in] rng The random number generator to use when creating the table.
     */
    TextureSynthesis();

    void bindShaderData(const ShaderVar& var) const;

    void readHFData(std::string hfPath, ref<Device> pDevice) ;
    void bindHFData(const ShaderVar& var);
    void precomputeFeatureData(std::vector<float> data, uint2 dataDim, ref<Device> pDevice);
    void bindFeatureData(const ShaderVar& var);

    ref<Texture> mpHFT;
private:
    float HTRotStength = 0.5f;
    ref<Texture> mpColor;
    ref<Texture> mpHFInvT;
    ref<Texture> mpFeatureT;
    ref<Texture> mpFeatureInvT;
    ref<Texture> mpACF;

};

}
