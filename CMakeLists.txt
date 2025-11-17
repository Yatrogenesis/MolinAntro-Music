cmake_minimum_required(VERSION 3.20)

# Version information
include(cmake/Version.cmake)

project(MolinAntroDaw
    VERSION ${MOLINANTRO_VERSION}
    DESCRIPTION "Professional Digital Audio Workstation with SOTA Features"
    LANGUAGES CXX C
)

# C++20 Standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE integration
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_VST3 "Build VST3 plugin support" ON)
option(BUILD_STANDALONE "Build standalone application" ON)
option(ENABLE_GPU_ACCELERATION "Enable GPU acceleration" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Platform-specific optimizations
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    if(MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
endif()

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# AddressSanitizer for debugging
if(ENABLE_ASAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Find required packages
find_package(Threads REQUIRED)

# Core library
add_subdirectory(src/core)
add_subdirectory(src/dsp)
add_subdirectory(src/midi)
add_subdirectory(src/instruments)
add_subdirectory(src/ui)
add_subdirectory(src/plugins)
add_subdirectory(src/utils)

# Standalone application
if(BUILD_STANDALONE)
    add_subdirectory(src)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation
install(TARGETS MolinAntroDaw
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    COMPONENT Runtime
)

# Install additional executables
if(TARGET MolinAntroDaw_Demo)
    install(TARGETS MolinAntroDaw_Demo
        RUNTIME DESTINATION bin
        COMPONENT Runtime
    )
endif()

if(TARGET MolinAntroDaw_Spectral)
    install(TARGETS MolinAntroDaw_Spectral
        RUNTIME DESTINATION bin
        COMPONENT Runtime
    )
endif()

# Install documentation
install(FILES
    ${CMAKE_SOURCE_DIR}/README.md
    ${CMAKE_SOURCE_DIR}/docs/Qt6_GUI_Integration.md
    ${CMAKE_SOURCE_DIR}/docs/VST3_SDK_Integration.md
    DESTINATION share/doc/molinantro
    COMPONENT Documentation
    OPTIONAL
)

# CPack configuration for installers
set(CPACK_PACKAGE_NAME "MolinAntro-DAW")
set(CPACK_PACKAGE_VENDOR "MolinAntro Technologies")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Professional Digital Audio Workstation with SOTA Features")
set(CPACK_PACKAGE_DESCRIPTION "MolinAntro DAW v2.0 - Complete professional audio workstation featuring real-time processing, AI stem separation, spectral editing, military-grade security, and more.")
set(CPACK_PACKAGE_VERSION_MAJOR ${MOLINANTRO_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${MOLINANTRO_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${MOLINANTRO_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${MOLINANTRO_VERSION})
set(CPACK_PACKAGE_INSTALL_DIRECTORY "MolinAntro DAW")
set(CPACK_PACKAGE_CONTACT "contact@molinantro.tech")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/MolinAntro/MolinAntro-Music")

# License and readme
if(EXISTS "${CMAKE_SOURCE_DIR}/LICENSE")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE")
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/README.md")
    set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")
endif()

# Components
set(CPACK_COMPONENTS_ALL Runtime Documentation)
set(CPACK_COMPONENT_RUNTIME_DISPLAY_NAME "MolinAntro DAW Application")
set(CPACK_COMPONENT_RUNTIME_DESCRIPTION "Core application and tools")
set(CPACK_COMPONENT_RUNTIME_REQUIRED TRUE)
set(CPACK_COMPONENT_DOCUMENTATION_DISPLAY_NAME "Documentation")
set(CPACK_COMPONENT_DOCUMENTATION_DESCRIPTION "Integration guides and documentation")

# Platform-specific packaging
if(WIN32)
    # Windows NSIS installer
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "MolinAntro DAW ${MOLINANTRO_VERSION}")
    set(CPACK_NSIS_PACKAGE_NAME "MolinAntro DAW")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_MODIFY_PATH ON)
    set(CPACK_NSIS_MUI_ICON "${CMAKE_SOURCE_DIR}/resources/icon.ico")
    set(CPACK_NSIS_MUI_UNIICON "${CMAKE_SOURCE_DIR}/resources/icon.ico")
    set(CPACK_NSIS_INSTALLED_ICON_NAME "bin\\\\MolinAntroDaw.exe")
    set(CPACK_NSIS_HELP_LINK "https://github.com/MolinAntro/MolinAntro-Music")
    set(CPACK_NSIS_URL_INFO_ABOUT "https://github.com/MolinAntro/MolinAntro-Music")
    set(CPACK_NSIS_CONTACT "contact@molinantro.tech")

    # Start menu shortcuts
    set(CPACK_NSIS_MENU_LINKS
        "bin/MolinAntroDaw.exe" "MolinAntro DAW"
        "share/doc/molinantro/README.md" "Documentation"
    )

elseif(APPLE)
    # macOS DMG
    set(CPACK_GENERATOR "DragNDrop;TGZ")
    set(CPACK_DMG_VOLUME_NAME "MolinAntro DAW ${MOLINANTRO_VERSION}")
    set(CPACK_DMG_FORMAT "UDBZ")
    set(CPACK_DMG_BACKGROUND_IMAGE "${CMAKE_SOURCE_DIR}/resources/dmg_background.png")
    set(CPACK_DMG_DS_STORE "${CMAKE_SOURCE_DIR}/resources/DS_Store")
    set(CPACK_BUNDLE_NAME "MolinAntro DAW")
    set(CPACK_BUNDLE_ICON "${CMAKE_SOURCE_DIR}/resources/icon.icns")
    set(CPACK_BUNDLE_PLIST "${CMAKE_SOURCE_DIR}/resources/Info.plist")

else()
    # Linux packages
    set(CPACK_GENERATOR "DEB;RPM;TGZ")

    # Debian package
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "MolinAntro Technologies <contact@molinantro.tech>")
    set(CPACK_DEBIAN_PACKAGE_SECTION "sound")
    set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.27), libstdc++6 (>= 8), libgcc1")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://github.com/MolinAntro/MolinAntro-Music")
    set(CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT")

    # RPM package
    set(CPACK_RPM_PACKAGE_LICENSE "Proprietary")
    set(CPACK_RPM_PACKAGE_GROUP "Applications/Multimedia")
    set(CPACK_RPM_PACKAGE_URL "https://github.com/MolinAntro/MolinAntro-Music")
    set(CPACK_RPM_PACKAGE_REQUIRES "glibc >= 2.27, libstdc++ >= 8")
    set(CPACK_RPM_FILE_NAME "RPM-DEFAULT")

    # Desktop entry
    set(DESKTOP_FILE_CONTENT "[Desktop Entry]
Name=MolinAntro DAW
Comment=Professional Digital Audio Workstation
Exec=MolinAntroDaw
Icon=molinantro
Terminal=false
Type=Application
Categories=AudioVideo;Audio;Mixer;Recorder;")

    file(WRITE "${CMAKE_BINARY_DIR}/molinantro-daw.desktop" "${DESKTOP_FILE_CONTENT}")
    install(FILES "${CMAKE_BINARY_DIR}/molinantro-daw.desktop"
        DESTINATION share/applications
        COMPONENT Runtime
    )
endif()

# Archive naming
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${MOLINANTRO_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${MOLINANTRO_VERSION}-Source")

# Source package
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
    "\\\\.git/"
    "\\\\.gitignore"
    "build/"
    "\\\\.DS_Store"
    "\\\\.vscode/"
    "\\\\.idea/"
)

include(CPack)
