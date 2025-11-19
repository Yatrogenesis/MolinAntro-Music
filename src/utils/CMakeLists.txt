# Utils module - cryptography and security utilities

set(UTILS_SOURCES
    Crypto.cpp
)

add_library(MolinAntro_Utils STATIC ${UTILS_SOURCES})

target_include_directories(MolinAntro_Utils
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Utils
    PUBLIC
        MolinAntro_Core
)
