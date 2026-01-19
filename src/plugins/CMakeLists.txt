# Plugin module - VST3/AU/CLAP/LV2 hosting and built-in plugins
# SOTA plugin hosting with full VST3 SDK integration

set(PLUGINS_SOURCES
    PluginHost.cpp
    VST3Host.cpp
)

add_library(MolinAntro_Plugins STATIC ${PLUGINS_SOURCES})

target_include_directories(MolinAntro_Plugins
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Plugins
    PUBLIC
        MolinAntro_Core
        MolinAntro_MIDI
)

# Optional VST3 SDK integration
option(MOLINANTRO_USE_VST3_SDK "Enable VST3 SDK integration for native plugin loading" OFF)

if(MOLINANTRO_USE_VST3_SDK AND EXISTS "${CMAKE_SOURCE_DIR}/external/vst3sdk")
    message(STATUS "VST3 SDK found - enabling native VST3 plugin loading")

    # Add VST3 SDK subdirectory
    set(SMTG_CREATE_VST2_VERSION OFF CACHE BOOL "Disable VST2" FORCE)
    set(SMTG_BUILD_UNIVERSAL_BINARY OFF CACHE BOOL "Disable universal binary" FORCE)
    add_subdirectory(${CMAKE_SOURCE_DIR}/external/vst3sdk ${CMAKE_BINARY_DIR}/vst3sdk EXCLUDE_FROM_ALL)

    target_compile_definitions(MolinAntro_Plugins
        PRIVATE
            HAVE_VST3_SDK=1
    )

    target_include_directories(MolinAntro_Plugins
        PRIVATE
            ${CMAKE_SOURCE_DIR}/external/vst3sdk
            ${CMAKE_SOURCE_DIR}/external/vst3sdk/pluginterfaces
            ${CMAKE_SOURCE_DIR}/external/vst3sdk/base
            ${CMAKE_SOURCE_DIR}/external/vst3sdk/public.sdk
    )

    target_link_libraries(MolinAntro_Plugins
        PRIVATE
            sdk
            sdk_common
            sdk_hosting
            pluginterfaces
    )
else()
    message(STATUS "VST3 SDK not found or disabled - using stub implementation")
    message(STATUS "  To enable: cmake -DMOLINANTRO_USE_VST3_SDK=ON ...")
    message(STATUS "  Requires: vst3sdk in external/ directory")
endif()

# Platform-specific plugin paths
if(WIN32)
    target_compile_definitions(MolinAntro_Plugins PRIVATE
        VST3_DEFAULT_PATH="C:/Program Files/Common Files/VST3"
    )
elseif(APPLE)
    target_compile_definitions(MolinAntro_Plugins PRIVATE
        VST3_DEFAULT_PATH="/Library/Audio/Plug-Ins/VST3"
        AU_DEFAULT_PATH="/Library/Audio/Plug-Ins/Components"
    )
else()
    target_compile_definitions(MolinAntro_Plugins PRIVATE
        VST3_DEFAULT_PATH="/usr/lib/vst3"
        LV2_DEFAULT_PATH="/usr/lib/lv2"
    )
endif()
