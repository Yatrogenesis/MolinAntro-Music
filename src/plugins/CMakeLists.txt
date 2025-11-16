# Plugin module - placeholder for future VST3/AU implementation

add_library(MolinAntro_Plugins INTERFACE)

target_include_directories(MolinAntro_Plugins
    INTERFACE
        ${CMAKE_SOURCE_DIR}/include
)
