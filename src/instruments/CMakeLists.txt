set(INSTRUMENTS_SOURCES
    Synthesizer.cpp
    Sampler.cpp
)

add_library(MolinAntro_Instruments STATIC ${INSTRUMENTS_SOURCES})

target_include_directories(MolinAntro_Instruments
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Instruments
    PUBLIC
        MolinAntro_Core
        MolinAntro_MIDI
        MolinAntro_DSP
)
