# UI module - placeholder for future Qt6/JUCE implementation
# For MVP, we use console-based UI

set(UI_SOURCES
    ConsoleUI.cpp
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
