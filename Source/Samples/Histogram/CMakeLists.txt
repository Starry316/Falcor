add_falcor_executable(Histogram)

target_sources(Histogram PRIVATE
    Histogram.cpp
    Histogram.h
    external/Utils.h
    external/Utils.cpp
    external/BTF.h
)

target_copy_shaders(Histogram Samples/Histogram)

target_source_group(Histogram "Samples")
