set(DSP_SOURCES
    AudioFile.cpp
    Effects.cpp
    SpectralProcessor.cpp
    StemSeparation.cpp
    NoiseReduction.cpp
    ConvolutionReverb.cpp
)

add_library(MolinAntro_DSP STATIC ${DSP_SOURCES})

target_include_directories(MolinAntro_DSP
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_DSP
    PUBLIC
        MolinAntro_Core
)
