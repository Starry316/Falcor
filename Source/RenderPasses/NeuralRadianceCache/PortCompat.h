#pragma once
#include <cuda_runtime.h>
#include "Falcor.h"
#include "Utils/CudaUtils.h"
#include "Core/AssetResolver.h"
#include "Rendering/Utils/PixelStats.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "RenderGraph/RenderPass.h"
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <cassert>

namespace Falcor
{
// The legacy backend creates resources outside the render pass. Keep that
// interface local to this plugin instead of modifying Falcor's core API.
inline Device*& portDeviceSlot() { static Device* device = nullptr; return device; }
inline ref<Device> portDevice() { return ref<Device>(portDeviceSlot()); }
inline void setPortDevice(const ref<Device>& device) { portDeviceSlot() = device.get(); }
inline bool portFindFile(const std::string& name, std::string& result)
{
    auto relative = name;
    const std::string prefix = "RenderPasses/NeuralRadianceCache/";
    if (auto pos = relative.find(prefix); pos != std::string::npos) relative = relative.substr(pos + prefix.size());
    auto path = std::filesystem::path(NIRC_SOURCE_DIR) / relative;
    if (!std::filesystem::exists(path)) path = name;
    if (!std::filesystem::exists(path)) return false;
    result = path.string();
    return true;
}
struct PortBuffer
{
    static ref<Buffer> create(size_t size, ResourceBindFlags flags, MemoryType memory = MemoryType::DeviceLocal, const void* data = nullptr)
    { return portDevice()->createBuffer(size, flags, memory, data); }
    static ref<Buffer> createStructured(uint32_t stride, uint32_t count, ResourceBindFlags flags, MemoryType memory = MemoryType::DeviceLocal, const void* data = nullptr, bool counter = true)
    { return portDevice()->createStructuredBuffer(stride, count, flags, memory, data, counter); }
};
struct PortTexture
{
    static ref<Texture> create2D(uint32_t w, uint32_t h, ResourceFormat format, uint32_t arrays, uint32_t mips, const void* data, ResourceBindFlags flags)
    { return portDevice()->createTexture2D(w, h, format, arrays, mips, data, flags); }
};
using PortBuffers = std::unordered_map<std::string, ref<Resource>>;
inline void portStructuredBuffer(PortBuffers& buffers, const std::string& name, uint32_t count, uint32_t stride, ResourceBindFlags flags)
{
    auto& resource = buffers[name];
    if (!resource || resource->asBuffer()->getElementCount() != count || resource->asBuffer()->getStructSize() != stride)
        resource = PortBuffer::createStructured(stride, std::max(1u, count), flags);
}
// Structured resources retain their original strides and UAV counters, but are
// owned by this pass because Falcor 9 graph reflection only describes raw buffers.
struct PortRenderData
{
    const RenderData& original;
    const PortBuffers& buffers;
    operator const RenderData&() const { return original; }
    const ref<Resource>& getResource(const std::string& name) const
    {
        auto it = buffers.find(name);
        return it == buffers.end() ? original.getResource(name) : it->second;
    }
    const ref<Resource>& operator[](const std::string& name) const { return getResource(name); }
    ref<Texture> getTexture(const std::string& name) const { const auto& r = getResource(name); return r ? r->asTexture() : nullptr; }
    const uint2& getDefaultTextureDims() const { return original.getDefaultTextureDims(); }
};
}
namespace FalcorCUDA
{
inline bool initCUDA() { return Falcor::portDevice()->initCudaDevice(); }
inline cudaSurfaceObject_t mapTextureToSurface(Falcor::ref<Falcor::Texture> texture, uint32_t flags)
{ return Falcor::cuda_utils::mapTextureToSurface(texture, flags); }
}
