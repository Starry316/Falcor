add_plugin(HFTracing)

target_sources(HFTracing PRIVATE
    HFTracing.cpp
    HFTracing.h
    MinimalPathTracer.rt.slang
    BTFSynthesis.slang
    Precompute.cs.slang
    PostProcess.cs.slang
    CreateMaxMip.cs.slang
    CountValidPixels.cs.slang
)

target_copy_shaders(HFTracing RenderPasses/HFTracing)
target_source_group(HFTracing "RenderPasses")
