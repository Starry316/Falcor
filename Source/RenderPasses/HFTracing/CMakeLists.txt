add_plugin(HFTracing)

target_sources(HFTracing PRIVATE
    HFTracing.cpp
    HFTracing.h
    MinimalPathTracer.rt.slang
    BTFSynthesis.slang
    GenerateGeometryMap.cs.slang
    VisualizeMaps.cs.slang
    CreateMaxMip.cs.slang
    Inference.cs.slang
    cuda/MLPInference.cu
    cuda/MLPInference.h
)
set_target_properties(HFTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_copy_shaders(HFTracing RenderPasses/HFTracing)
target_source_group(HFTracing "RenderPasses")
