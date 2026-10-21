add_plugin(NeuralRadianceCache)
set(NIRC_RESEARCH_ROOT "D:/Projects/radcache" CACHE PATH "Original NRC/NIRC research checkout")
set(TCNN_ROOT "${NIRC_RESEARCH_ROOT}/tiny-cuda-nn")

# Preserve the research fork's encodings, loss, optimizer and fused kernels.
add_library(NircTinyCudaNN STATIC
    ${TCNN_ROOT}/src/common.cu
    ${TCNN_ROOT}/src/common_device.cu
    ${TCNN_ROOT}/src/encoding.cu
    ${TCNN_ROOT}/src/fully_fused_mlp.cu
    ${TCNN_ROOT}/src/loss.cu
    ${TCNN_ROOT}/src/network.cu
    ${TCNN_ROOT}/src/object.cu
    ${TCNN_ROOT}/src/optimizer.cu
    ${TCNN_ROOT}/src/reduce_sum.cu
    ${TCNN_ROOT}/src/cnpy.cpp
)
target_include_directories(NircTinyCudaNN PUBLIC ${TCNN_ROOT}/include ${TCNN_ROOT}/dependencies)
target_include_directories(NircTinyCudaNN PRIVATE ${CMAKE_SOURCE_DIR}/external/packman/deps/include)
# Falcor includes JSON before the neural headers. Compile the neural library
# against that same JSON ABI instead of TCNN's bundled older copy.
target_compile_options(NircTinyCudaNN PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-include=${CMAKE_SOURCE_DIR}/external/include/nlohmann/json.hpp>
    $<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_SOURCE_DIR}/external/include/nlohmann/json.hpp>
)
target_link_directories(NircTinyCudaNN PUBLIC ${CMAKE_SOURCE_DIR}/external/packman/deps/lib)
target_link_libraries(NircTinyCudaNN PUBLIC zlib)
target_compile_definitions(NircTinyCudaNN PUBLIC TCNN_MIN_GPU_ARCH=120 CUDA_INC_DIR="${CUDAToolkit_INCLUDE_DIRS}")
target_compile_options(NircTinyCudaNN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=/bigobj;--diag-suppress=20012>)
set_target_properties(NircTinyCudaNN PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17 CXX_STANDARD 17 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
target_link_libraries(NircTinyCudaNN PUBLIC CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::curand CUDA::cusparse CUDA::nvrtc)

file(GLOB_RECURSE NIRC_SHADERS CONFIGURE_DEPENDS *.slang *.slangh)
target_sources(NeuralRadianceCache PRIVATE
    DummyNeuralNetwork.cpp DummyNeuralNetwork.h NNLaplacian.cpp NNLaplacian.h
    NNLaplacianKernels.h NNLaplacianDataTypes.h PortCompat.h BootstrapUtils2.cpp
    Legacy/PathTracer/PathTracer.cpp Legacy/PathTracer/PathTracer.h
    ${NIRC_SHADERS}
)
set_source_files_properties(DummyNeuralNetwork.cpp NNLaplacian.cpp PROPERTIES LANGUAGE CUDA)
target_include_directories(NeuralRadianceCache PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(NeuralRadianceCache PRIVATE NIRC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}" CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
target_compile_options(NeuralRadianceCache PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=/bigobj;-Xcompiler=/Zc:preprocessor;--diag-suppress=20012>)
target_compile_options(NeuralRadianceCache PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/WX-;/wd5038;/wd5263>)
set_target_properties(NeuralRadianceCache PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17 CXX_STANDARD 17)
target_link_libraries(NeuralRadianceCache PRIVATE NircTinyCudaNN)
target_copy_shaders(NeuralRadianceCache RenderPasses/NeuralRadianceCache)

add_executable(NircHybridTests ${NIRC_RESEARCH_ROOT}/build-support/test-hybrid-network.cu)
target_include_directories(NircHybridTests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/external/include)
target_link_libraries(NircHybridTests PRIVATE NircTinyCudaNN)
target_compile_options(NircHybridTests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-Xcompiler=/Zc:preprocessor>)
set_target_properties(NircHybridTests PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17 CXX_STANDARD 17 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

add_executable(NircHybridProfile ${NIRC_RESEARCH_ROOT}/build-support/profile-hybrid-inference.cu)
target_include_directories(NircHybridProfile PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/external/include)
target_link_libraries(NircHybridProfile PRIVATE NircTinyCudaNN)
target_compile_options(NircHybridProfile PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-lineinfo;-Xcompiler=/Zc:preprocessor>)
set_target_properties(NircHybridProfile PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17 CXX_STANDARD 17 MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
