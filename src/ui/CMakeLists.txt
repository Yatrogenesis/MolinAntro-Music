# UI module - Terminal and GUI support
# Includes: Console UI, Advanced Terminal UI, Qt6 framework (TBD)

set(UI_SOURCES
    ConsoleUI.cpp
    TerminalUI.cpp
)

add_library(MolinAntro_UI STATIC ${UI_SOURCES})

target_include_directories(MolinAntro_UI
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_UI
    PUBLIC
        MolinAntro_Core
)
