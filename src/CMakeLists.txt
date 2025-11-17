# Main executable
add_executable(MolinAntroDaw main.cpp)

target_link_libraries(MolinAntroDaw
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_UI
        MolinAntro_Plugins
        MolinAntro_Utils
        Threads::Threads
)

# Set executable properties
set_target_properties(MolinAntroDaw PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install
install(TARGETS MolinAntroDaw
    RUNTIME DESTINATION bin
)
