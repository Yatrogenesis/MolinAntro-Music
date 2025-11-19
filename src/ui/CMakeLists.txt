# UI module - Terminal and Professional GUI support
# ACME Edition v3.0.0 - Complete UI/UX System

set(UI_SOURCES
    # Legacy console UI
    ConsoleUI.cpp
    TerminalUI.cpp

    # Professional GUI System (ACME Edition)
    Theme.cpp
    Components.cpp
    Preferences.cpp
    Animation.cpp
)

add_library(MolinAntro_UI STATIC ${UI_SOURCES})

target_include_directories(MolinAntro_UI
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_UI
    PUBLIC
        MolinAntro_Core
        MolinAntro_DSP
)

# Enable C++20 features for UI
target_compile_features(MolinAntro_UI PUBLIC cxx_std_20)
