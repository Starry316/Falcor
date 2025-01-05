add_plugin(AccumulateSTPass)

target_sources(AccumulateSTPass PRIVATE
    AccumulateSTPass.cpp
    AccumulateSTPass.h
    Accumulate.cs.slang
)
target_copy_shaders(AccumulateSTPass RenderPasses/AccumulateSTPass)
target_source_group(AccumulateSTPass "RenderPasses")
