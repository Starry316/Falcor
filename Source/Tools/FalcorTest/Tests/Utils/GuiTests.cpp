#include "Testing/UnitTest.h"
#include "Utils/UI/Gui.h"

namespace Falcor
{
GPU_TEST(GuiDropdownScopedValues)
{
    auto device = ctx.getDevice();
    Gui gui(device, 640, 480);
    auto fbo = Fbo::create2D(device, 640, 480, Fbo::Desc().setColorTarget(0, ResourceFormat::RGBA8Unorm));
    uint32_t first = 7, second = 7;
    for (uint32_t frame = 0; frame < 4; ++frame)
    {
        gui.beginFrame();
        {
            Gui::Window window(&gui, "Dropdown regression", {640, 480});
            if (auto group = window.group("First group", true))
                EXPECT_FALSE(group.dropdown("Mode", {{1, "One"}, {7, "Seven"}}, first));
            if (auto group = window.group("Second group", true))
                EXPECT_FALSE(group.dropdown("Mode", {{7, "Seven"}, {1, "One"}}, second));
        }
        gui.render(device->getRenderContext(), fbo, 1.f / 60.f);
        EXPECT_EQ(first, 7u);
        EXPECT_EQ(second, 7u);
        device->endFrame();
    }
}

GPU_TEST(GuiNumericClampNotification)
{
    auto device = ctx.getDevice();
    Gui gui(device, 640, 480);
    auto fbo = Fbo::create2D(device, 640, 480, Fbo::Desc().setColorTarget(0, ResourceFormat::RGBA8Unorm));
    float scalar = -1.f;
    float2 vector(-2.f, 4.f);
    for (uint32_t frame = 0; frame < 3; ++frame)
    {
        gui.beginFrame();
        {
            Gui::Window window(&gui, "Numeric clamp", {640, 480});
            EXPECT_EQ(window.var("Scalar", scalar, 0.f, 1.f), frame == 0);
            EXPECT_EQ(window.var("Vector", vector, 0.f, 1.f), frame == 0);
        }
        gui.render(device->getRenderContext(), fbo, 1.f / 60.f);
        EXPECT_EQ(scalar, 0.f);
        EXPECT_EQ(vector.x, 0.f);
        EXPECT_EQ(vector.y, 1.f);
        device->endFrame();
    }
}

GPU_TEST(GuiDropdownDynamicList)
{
    auto device = ctx.getDevice();
    Gui gui(device, 640, 480);
    auto fbo = Fbo::create2D(device, 640, 480, Fbo::Desc().setColorTarget(0, ResourceFormat::RGBA8Unorm));
    uint32_t selected = 7;
    const std::vector<Gui::DropdownList> lists = {
        {{1, "One"}, {7, "Seven"}},  // Initial selection at index 1.
        {{7, "Seven"}, {1, "One"}},  // Reorder without changing the value.
        {{7, "Seven"}},             // Shrink without changing the value.
        {{1, "One"}},               // Current value temporarily unavailable.
        {},                         // Empty list.
        {{7, "Seven"}},
    };
    for (const auto& list : lists)
    {
        gui.beginFrame();
        {
            Gui::Window window(&gui, "Dynamic dropdown", {640, 480});
            EXPECT_FALSE(window.dropdown("Mode", list, selected));
        }
        gui.render(device->getRenderContext(), fbo, 1.f / 60.f);
        EXPECT_EQ(selected, 7u);
        device->endFrame();
    }
}
} // namespace Falcor
