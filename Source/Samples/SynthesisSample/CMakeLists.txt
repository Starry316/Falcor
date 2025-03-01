add_falcor_executable(SynthesisSample)

target_sources(SynthesisSample PRIVATE
    SynthesisSample.cpp
    SynthesisSample.h
    GenNDF.cs.slang
    Display.ps.slang
    WriteTex.cs.slang
)

target_copy_shaders(SynthesisSample Samples/SynthesisSample)

target_source_group(SynthesisSample "SynthesisSample")
