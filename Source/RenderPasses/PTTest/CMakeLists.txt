add_plugin(PTTest)

target_sources(PTTest PRIVATE
    PTTest.cpp
    PTTest.h
    MinimalPathTracer.rt.slang
    ShadeLightProbe.rt.slang
    Tools/Utils.cpp
    Tools/Utils.h
)
target_copy_shaders(PTTest RenderPasses/PTTest)
target_source_group(PTTest "RenderPasses")
