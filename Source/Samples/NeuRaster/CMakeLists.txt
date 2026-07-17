add_falcor_executable(NeuRaster)

target_sources(NeuRaster PRIVATE
    NeuRaster.cpp
    NeuRaster.h
    NeuRaster.3d.slang
)

target_copy_shaders(NeuRaster Samples/NeuRaster)

target_source_group(NeuRaster "Samples")
