set(neural_default OFF)
if(WIN32 AND FALCOR_HAS_CUDA)
    set(neural_default ON)
endif()
option(FALCOR_ENABLE_NEURAL_MATERIAL "Build neural materials and their CUDA inference library" ${neural_default})
if(NOT FALCOR_ENABLE_NEURAL_MATERIAL)
    return()
endif()
if(NOT WIN32 OR NOT FALCOR_HAS_CUDA)
    message(FATAL_ERROR "NeuralMaterialPass currently requires Windows D3D12 and CUDA")
endif()
if(DEFINED MPMAT_SDK_DIR)
    message(STATUS "Ignoring obsolete MPMAT_SDK_DIR; neural inference is built from project sources")
    unset(MPMAT_SDK_DIR CACHE)
endif()
add_subdirectory("${PROJECT_SOURCE_DIR}/Source/NeuralInference" "${PROJECT_BINARY_DIR}/Source/NeuralInference")

add_plugin(NeuralMaterialPass)
target_sources(NeuralMaterialPass PRIVATE NeuralMaterialPass.cpp NeuralMaterialPass.h NeuralLightMotion.h
    NeuralCompositePass.cpp NeuralCompositePass.h)
file(GLOB neural_shaders CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.slang" "${CMAKE_CURRENT_SOURCE_DIR}/*.slangh")
target_sources(NeuralMaterialPass PRIVATE ${neural_shaders})
target_link_libraries(NeuralMaterialPass PRIVATE NeuralInferenceHeaders CUDA::cudart_static CUDA::cuda_driver)
target_copy_shaders(NeuralMaterialPass RenderPasses/NeuralMaterialPass)
target_source_group(NeuralMaterialPass "RenderPasses")

add_custom_target(NeuralMaterialRuntime
    COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:mpmat_inference>" "${FALCOR_OUTPUT_DIRECTORY}/mpmat_inference.dll"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/Data" "${FALCOR_OUTPUT_DIRECTORY}/data/NeuralMaterial"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/Data/NeuralMaterial" "${FALCOR_OUTPUT_DIRECTORY}/data/NeuralMaterial"
    DEPENDS mpmat_inference
    COMMENT "Deploying neural inference built from project sources"
    VERBATIM
)
add_dependencies(NeuralMaterialPass NeuralMaterialRuntime)
