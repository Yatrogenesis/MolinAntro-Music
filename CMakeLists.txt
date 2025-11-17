cmake_minimum_required(VERSION 3.20)

project(MolinAntroDaw
    VERSION 1.0.0
    DESCRIPTION "Professional Digital Audio Workstation"
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
)

# CPack configuration for installers
set(CPACK_PACKAGE_NAME "MolinAntro DAW")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "MolinAntro Technologies")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Professional Digital Audio Workstation")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop")
else()
    set(CPACK_GENERATOR "DEB;RPM")
endif()

include(CPack)
