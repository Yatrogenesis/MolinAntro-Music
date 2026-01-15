set(SEQUENCER_SOURCES
    SessionView.cpp
    PatternSequencer.cpp
)

add_library(MolinAntro_Sequencer STATIC ${SEQUENCER_SOURCES})

target_include_directories(MolinAntro_Sequencer
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Sequencer
    PUBLIC
        MolinAntro_Core
        MolinAntro_MIDI
)
