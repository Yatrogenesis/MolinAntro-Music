set(MIDI_SOURCES
    MIDIEngine.cpp
)

add_library(MolinAntro_MIDI STATIC ${MIDI_SOURCES})

target_include_directories(MolinAntro_MIDI
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_MIDI
    PUBLIC
        MolinAntro_Core
        Threads::Threads
)
