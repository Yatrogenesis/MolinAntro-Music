# Plugin module - VST3/AU hosting and built-in plugins

set(PLUGINS_SOURCES
    PluginHost.cpp
)

add_library(MolinAntro_Plugins STATIC ${PLUGINS_SOURCES})

target_include_directories(MolinAntro_Plugins
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Plugins
    PUBLIC
        MolinAntro_Core
)
