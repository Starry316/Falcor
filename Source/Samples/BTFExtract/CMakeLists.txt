add_falcor_executable(BTFExtract)

target_sources(BTFExtract PRIVATE
    BTFExtract.cpp
    BTFExtract.h
    external/BTF.h
)

target_copy_shaders(BTFExtract Samples/BTFExtract)

target_source_group(BTFExtract "Samples")
