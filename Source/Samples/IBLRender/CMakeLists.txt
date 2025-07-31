add_falcor_executable(IBLRender)

target_sources(IBLRender PRIVATE
    IBLRender.cpp
    IBLRender.h
    render.cs.slang
    display.ps.slang
    IBL.rt.slang
)

target_copy_shaders(IBLRender Samples/IBLRender)

target_source_group(IBLRender "Samples")
