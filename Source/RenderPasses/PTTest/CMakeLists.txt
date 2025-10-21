add_plugin(PTTest)

target_sources(PTTest PRIVATE
    PTTest.cpp
    PTTest.h
    MinimalPathTracer.rt.slang
)
target_copy_shaders(PTTest RenderPasses/PTTest)
target_source_group(PTTest "RenderPasses")
