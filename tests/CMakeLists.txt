cmake_minimum_required(VERSION 3.20)

# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

# Enable testing
enable_testing()

# Unit tests
add_subdirectory(unit)

# Integration tests
add_subdirectory(integration)

# E2E tests
add_subdirectory(e2e)
