set(CORE_SOURCES
    AudioEngine.cpp
    Transport.cpp
    AudioBuffer.cpp
)

add_library(MolinAntro_Core STATIC ${CORE_SOURCES})

target_include_directories(MolinAntro_Core
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Core
    PUBLIC
        Threads::Threads
)
