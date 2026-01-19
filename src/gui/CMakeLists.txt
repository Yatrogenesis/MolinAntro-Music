# Qt6 GUI Module for MolinAntro DAW
# Requires Qt6 Widgets, Gui, Core modules

set(GUI_SOURCES
    Qt6MainWindow.cpp
    Qt6Components.cpp
    Qt6Styles.cpp
)

set(GUI_HEADERS
    ${CMAKE_SOURCE_DIR}/include/gui/Qt6MainWindow.h
    ${CMAKE_SOURCE_DIR}/include/gui/Qt6Components.h
    ${CMAKE_SOURCE_DIR}/include/gui/Qt6Styles.h
)

# Only build if Qt6 and BUILD_QT6_GUI are enabled
if(BUILD_QT6_GUI)
    find_package(Qt6 REQUIRED COMPONENTS Widgets Gui Core)

    add_library(MolinAntro_GUI STATIC ${GUI_SOURCES} ${GUI_HEADERS})

    target_include_directories(MolinAntro_GUI
        PUBLIC
            ${CMAKE_SOURCE_DIR}/include
    )

    target_link_libraries(MolinAntro_GUI
        PUBLIC
            MolinAntro_Core
            MolinAntro_DSP
            Qt6::Widgets
            Qt6::Gui
            Qt6::Core
    )

    target_compile_definitions(MolinAntro_GUI
        PUBLIC
            BUILD_QT6_GUI
    )

    # Enable Qt MOC (Meta-Object Compiler) for Q_OBJECT classes
    set_target_properties(MolinAntro_GUI PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
        AUTORCC ON
    )
endif()
