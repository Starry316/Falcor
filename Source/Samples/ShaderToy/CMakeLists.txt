add_falcor_executable(ShaderToy)

target_include_directories(ShaderToy PUBLIC  ${CUDA_INCLUDE_DIRS})
target_link_libraries(ShaderToy PUBLIC  ${CUDA_LIBRARIES} )

target_sources(ShaderToy PRIVATE
    ShaderToy.cpp
    ShaderToy.h
    Toy.ps.slang
    cuda/MLPInference.cu
    cuda/MLPInference.h
    cuda/cudaUtils.h
    display.ps.slang
    bindInput.cs.slang
    InferDebug.cs.slang
    cuda/cudaTextureHelper.h
    cuda/cudaTextureHelper.cpp
)
set_target_properties(ShaderToy PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_copy_shaders(ShaderToy Samples/ShaderToy)

target_source_group(ShaderToy "Samples")
