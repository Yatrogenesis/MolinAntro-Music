# Main executable
add_executable(MolinAntroDaw main.cpp)

target_link_libraries(MolinAntroDaw
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_MIDI
        MolinAntro_Instruments
        MolinAntro_UI
        MolinAntro_Plugins
        MolinAntro_Utils
        Threads::Threads
)

# Complete feature demonstration executable
add_executable(MolinAntroDaw_Demo demo_complete.cpp)

target_link_libraries(MolinAntroDaw_Demo
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_MIDI
        MolinAntro_Instruments
        Threads::Threads
)

# Spectral processing demonstration executable
add_executable(MolinAntroDaw_Spectral demo_spectral.cpp)

target_link_libraries(MolinAntroDaw_Spectral
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_Instruments
        Threads::Threads
)

# Set executable properties
set_target_properties(MolinAntroDaw MolinAntroDaw_Demo MolinAntroDaw_Spectral PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS MolinAntroDaw MolinAntroDaw_Demo MolinAntroDaw_Spectral
    RUNTIME DESTINATION bin
)
