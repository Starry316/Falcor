cmake_minimum_required(VERSION 3.24)

# Also allow an isolated build of this source directory, without Falcor or an
# MPMat checkout. This uses exactly the same library target as the renderer.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(NeuralInference LANGUAGES CXX CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

set(FALCOR_NEURAL_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures for neural inference (native, or e.g. 120)")
if(FALCOR_NEURAL_CUDA_ARCHITECTURES STREQUAL "native" AND NOT CMAKE_CUDA_ARCHITECTURES_NATIVE)
    message(FATAL_ERROR "No CUDA GPU detected. For an offline/cross build set FALCOR_NEURAL_CUDA_ARCHITECTURES to the target architecture.")
endif()

# These headers provide cooperative vectors and are independent of Falcor's
# optional OptixDenoiser SDK. Download a fixed revision, never the moving HEAD.
include(FetchContent)
FetchContent_Declare(neural_optix_headers
    GIT_REPOSITORY https://github.com/NVIDIA/optix-dev.git
    GIT_TAG f1f6dd803f3159992d248178f6e09421c6eb8b6d
)
FetchContent_MakeAvailable(neural_optix_headers)
set(neural_optix_include "${neural_optix_headers_SOURCE_DIR}/include")
if(NOT EXISTS "${neural_optix_include}/optix.h")
    message(FATAL_ERROR "Pinned OptiX cooperative-vector headers are missing")
endif()

set(coop_ptx "${CMAKE_CURRENT_BINARY_DIR}/coop_programs.ptx")
set(coop_header "${CMAKE_CURRENT_BINARY_DIR}/coop_ptx.h")
add_executable(neural_embed_ptx EmbedPtx.cpp)
set_target_properties(neural_embed_ptx PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON FOLDER "Neural material")
file(GLOB neural_headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE optix_headers CONFIGURE_DEPENDS "${neural_optix_include}/*.h")
if(MSVC)
    set(neural_ptx_host_options -Xcompiler=/utf-8)
endif()
add_custom_command(OUTPUT "${coop_header}"
    BYPRODUCTS "${coop_ptx}"
    COMMAND "${CMAKE_CUDA_COMPILER}" --ptx --std=c++17 --gpu-architecture=compute_75
        --fmad=false ${neural_ptx_host_options} -I "${neural_optix_include}" -I "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/coop_programs.cu" -o "${coop_ptx}"
    COMMAND neural_embed_ptx "${coop_ptx}" "${coop_header}"
    DEPENDS neural_embed_ptx coop_programs.cu ${neural_headers} ${optix_headers}
    COMMENT "Compiling and embedding neural cooperative-vector device code"
    VERBATIM
)

add_library(NeuralInferenceHeaders INTERFACE)
target_include_directories(NeuralInferenceHeaders INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
add_library(mpmat_inference SHARED inference_bridge.cu inference_bridge.cpp coop_backend.cpp "${coop_header}" ${neural_headers})
set_property(SOURCE coop_backend.cpp APPEND PROPERTY OBJECT_DEPENDS "${coop_header}")
target_link_libraries(mpmat_inference PUBLIC NeuralInferenceHeaders PRIVATE CUDA::cudart_static ${CMAKE_DL_LIBS})
target_include_directories(mpmat_inference PRIVATE "${neural_optix_include}" "${CMAKE_CURRENT_BINARY_DIR}" ${CUDAToolkit_INCLUDE_DIRS})
target_compile_definitions(mpmat_inference PRIVATE NOMINMAX MPMAT_INFERENCE_EXPORTS=1)
target_compile_options(mpmat_inference PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
if(MSVC)
    target_compile_options(mpmat_inference PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/utf-8>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()
set_target_properties(mpmat_inference PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${FALCOR_NEURAL_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Static
    FOLDER "Neural material"
)
