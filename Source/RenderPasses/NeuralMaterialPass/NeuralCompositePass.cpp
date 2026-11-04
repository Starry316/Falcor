#include "NeuralCompositePass.h"
#include <cstring>
#include <fstream>

NeuralCompositePass::NeuralCompositePass(ref<Device> device, const Properties& props)
    : RenderPass(device), mControlPrefix(props.get<std::string>("referenceControlPrefix", "neuralReference"))
{}

Properties NeuralCompositePass::getProperties() const
{
    Properties props; props["referenceControlPrefix"] = mControlPrefix; return props;
}

RenderPassReflection NeuralCompositePass::reflect(const CompileData& data)
{
    RenderPassReflection r;
    r.addInput("foreground", "Deterministic premultiplied neural HDR and independent coverage");
    r.addInput("display", "Environment background and emitter/debug overlays");
    r.addInput("environment", "AccumulatePass output: average environment sample sum and count");
    r.addInput("area", "AccumulatePass output: average polygon sample sum and count");
    r.addOutput("color", "Final scene-linear HDR, before Falcor tone mapping")
        .format(ResourceFormat::RGBA32Float).bindFlags(ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
    return r;
}

void NeuralCompositePass::execute(RenderContext* context, const RenderData& data)
{
    if (!mpPass) mpPass = ComputePass::create(mpDevice, "RenderPasses/NeuralMaterialPass/NeuralComposite.cs.slang", "main");
    auto& dict = data.getDictionary();
    const bool environment = dict.getValue(mControlPrefix + ".environmentEnabled", false);
    const bool area = dict.getValue(mControlPrefix + ".areaEnabled", false);
    const auto output = data.getTexture("color");
    auto root = mpPass->getRootVar();
    root["gForeground"] = data.getTexture("foreground"); root["gDisplay"] = data.getTexture("display");
    root["gEnvironment"] = data.getTexture("environment"); root["gArea"] = data.getTexture("area");
    root["gColor"] = output;
    auto cb = root["gCompositeCB"];
    cb["gFrameDim"] = uint2(output->getWidth(), output->getHeight());
    cb["gEnvironmentEnabled"] = uint32_t(environment); cb["gAreaEnabled"] = uint32_t(area);
    cb["gMaterialOpacity"] = dict.getValue(mControlPrefix + ".opacity", 1.f);
    cb["gOpaqueSubstrate"] = uint32_t(dict.getValue(mControlPrefix + ".substrate", false));
    mpPass->execute(context, output->getWidth(), output->getHeight());

    const auto dumpPath = dict.getValue(mControlPrefix + ".dumpPath", std::string());
    if (dumpPath.empty()) return;
    dict[mControlPrefix + ".dumpPath"] = std::string();
    // Explicit test/debug readback only. No CPU readback in interactive rendering.
    const std::filesystem::path directory(dumpPath);
    auto write = [&](const std::string& name, const void* bytes, size_t count)
    {
        std::ofstream file(directory / (name + ".bin"), std::ios::binary);
        FALCOR_CHECK(file.good(), "Cannot write composite capture {}", name);
        file.write(static_cast<const char*>(bytes), std::streamsize(count));
        FALCOR_CHECK(file.good(), "Failed writing composite capture {}", name);
    };
    auto color = context->readTextureSubresource(output.get(), 0);
    write("color", color.data(), color.size());
    const size_t pixels = size_t(output->getWidth()) * output->getHeight();
    auto foregroundBytes = context->readTextureSubresource(data.getTexture("foreground").get(), 0);
    std::vector<float4> foreground(pixels);
    std::memcpy(foreground.data(), foregroundBytes.data(), foregroundBytes.size());
    for (const char* channel : {"environment", "area"})
    {
        auto bytes = context->readTextureSubresource(data.getTexture(channel).get(), 0);
        const std::string name = channel == std::string("environment") ? "referenceEnvironment" : "referenceArea";
        write(name + "Accum", bytes.data(), bytes.size());
        std::vector<float4> values(pixels);
        std::memcpy(values.data(), bytes.data(), bytes.size());
        for (size_t i = 0; i < pixels; ++i)
        {
            const float weight = values[i].w;
            values[i] = float4(weight > 0.f ? values[i].xyz() / weight : float3(0.f), foreground[i].w);
        }
        write(name, values.data(), values.size() * sizeof(float4));
    }
}
