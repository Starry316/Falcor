#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"

using namespace Falcor;

/** Normalize Falcor's accumulated sample sums, then composite linear HDR. */
class NeuralCompositePass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(NeuralCompositePass, "NeuralCompositePass", "Normalize reference samples and composite neural direct illumination.");
    static ref<NeuralCompositePass> create(ref<Device> device, const Properties& props)
    { return make_ref<NeuralCompositePass>(device, props); }
    NeuralCompositePass(ref<Device> device, const Properties& props);
    Properties getProperties() const override;
    RenderPassReflection reflect(const CompileData& data) override;
    void execute(RenderContext* context, const RenderData& data) override;
private:
    ref<ComputePass> mpPass;
    std::string mControlPrefix;
};
