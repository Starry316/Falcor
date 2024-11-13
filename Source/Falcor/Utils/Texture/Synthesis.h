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
    TextureSynthesis(ref<Device> pDevice);

    void bindShaderData(const ShaderVar& var) const;


private:
    float HTRotStength = 0.5f;
    ref<Texture> mpColor;
};

}
