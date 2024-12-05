add_plugin(HFTracing)

set(TensorRT_INCLUDE_DIRS D:/TensorRT/include/)
set(TensorRT_LIBRARIES
    D:/TensorRT/lib/nvinfer_10.lib
    D:/TensorRT/lib/nvinfer_plugin_10.lib
    D:/TensorRT/lib/nvonnxparser_10.lib
)

find_package(mathdx REQUIRED COMPONENTS cublasdx CONFIG PATHS "D:/Projects/nvidia/mathdx/24.08/")

# message(STATUS ${mathdx_cublasdx_FOUND})


target_include_directories(HFTracing PUBLIC ${TensorRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS})
target_link_libraries(HFTracing PUBLIC ${TensorRT_LIBRARIES} mathdx::cublasdx ${CUDA_LIBRARIES} nvrtc)

target_sources(HFTracing PRIVATE
    HFTracing.cpp
    HFTracing.h
    MinimalPathTracer.rt.slang
    BTFSynthesis.slang
    GenerateGeometryMap.cs.slang
    VisualizeMaps.cs.slang
    CreateMaxMip.cs.slang
    Inference.cs.slang
    Display.cs.slang
    cuda/MLPInference.cu
    cuda/MLPInference.h
    Tools/Utils.cpp
    Tools/Utils.h
    Tools/CommonNvrtc.hpp
    # cuda/cublasdxTest.cu
    # cuda/cublasdxTest.h

)
set_target_properties(HFTracing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_copy_shaders(HFTracing RenderPasses/HFTracing)
target_source_group(HFTracing "RenderPasses")
