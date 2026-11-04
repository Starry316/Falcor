# This optional plugin consumes a portable inference SDK, never another checkout.
set(MPMAT_SDK_DIR "" CACHE PATH "Exported MPMat SDK containing include/inference_bridge.h and runtime/mpmat_inference.dll")
if(NOT MPMAT_SDK_DIR)
    message(STATUS "NeuralMaterialPass disabled: set MPMAT_SDK_DIR to an exported inference SDK")
    return()
endif()
if(NOT WIN32 OR NOT FALCOR_HAS_CUDA)
    message(FATAL_ERROR "NeuralMaterialPass currently requires Windows D3D12 and CUDA")
endif()
if(NOT EXISTS "${MPMAT_SDK_DIR}/include/inference_bridge.h" OR NOT EXISTS "${MPMAT_SDK_DIR}/runtime/mpmat_inference.dll")
    message(FATAL_ERROR "MPMAT_SDK_DIR must contain include/inference_bridge.h and runtime/mpmat_inference.dll")
endif()

add_plugin(NeuralMaterialPass)
target_sources(NeuralMaterialPass PRIVATE NeuralMaterialPass.cpp NeuralMaterialPass.h NeuralLightMotion.h
    NeuralCompositePass.cpp NeuralCompositePass.h)
file(GLOB neural_shaders CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.slang" "${CMAKE_CURRENT_SOURCE_DIR}/*.slangh")
target_sources(NeuralMaterialPass PRIVATE ${neural_shaders})
target_include_directories(NeuralMaterialPass PRIVATE "${MPMAT_SDK_DIR}/include")
target_link_libraries(NeuralMaterialPass PRIVATE CUDA::cudart_static CUDA::cuda_driver)
target_copy_shaders(NeuralMaterialPass RenderPasses/NeuralMaterialPass)
target_source_group(NeuralMaterialPass "RenderPasses")

add_custom_target(NeuralMaterialRuntime
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${MPMAT_SDK_DIR}/runtime" "${FALCOR_OUTPUT_DIRECTORY}"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/Data" "${FALCOR_OUTPUT_DIRECTORY}/data/NeuralMaterial"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/Data/NeuralMaterial" "${FALCOR_OUTPUT_DIRECTORY}/data/NeuralMaterial"
    COMMENT "Copying portable MPMat inference runtime"
)
add_dependencies(NeuralMaterialPass NeuralMaterialRuntime)
