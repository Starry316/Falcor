add_falcor_executable(HelloDXR)

target_sources(HelloDXR PRIVATE
    HelloDXR.cpp
    HelloDXR.h
    HelloDXR.3d.slang
    HelloDXR.rt.slang
    SSR.cs.slang
    utils.slangh
    PostProcess.cs.slang
    SSRDefs.slangh
)

target_copy_shaders(HelloDXR Samples/HelloDXR)

target_source_group(HelloDXR "Samples")
