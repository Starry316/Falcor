
#include "Utils.h"
namespace Falcor
{
void createTex(ref<Texture>& tex, ref<Device> device, Falcor::uint2 targetDim, bool buildCuda, bool isUint)
{
    ResourceBindFlags flag = ResourceBindFlags::ShaderResource | ResourceBindFlags::RenderTarget | ResourceBindFlags::UnorderedAccess;
    if (buildCuda)
        flag |= ResourceBindFlags::Shared;

    if (tex.get() == nullptr)
    {
        if (isUint)
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, flag);
        else
            tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);
    }
    else
    {
        if (tex.get()->getWidth() != targetDim.x || tex.get()->getHeight() != targetDim.y)
        {
            logInfo("Recreating texture");
            if (isUint)
                tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Uint, 1, 1, nullptr, flag);
            else
                tex = device->createTexture2D(targetDim.x, targetDim.y, ResourceFormat::RGBA32Float, 1, 1, nullptr, flag);
        }
    }
};

void createBuffer(ref<Buffer>& buf, ref<Device> device, Falcor::uint2 targetDim, uint itemSize )
{
    if (buf.get() == nullptr)
    {
        buf = device->createBuffer(
            targetDim.x * targetDim.y * itemSize * sizeof(float),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr
        );
    }
    else
    {
        if (buf.get()->getElementCount() != targetDim.x * targetDim.y * itemSize * sizeof(float))
        {
            logInfo("Recreating Buffer");
            buf = device->createBuffer(
                targetDim.x * targetDim.y * itemSize * sizeof(float),
                ResourceBindFlags::ShaderResource | ResourceBindFlags::Shared | ResourceBindFlags::UnorderedAccess,
                MemoryType::DeviceLocal,
                nullptr
            );
        }
    }
};

}
