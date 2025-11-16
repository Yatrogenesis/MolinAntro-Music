# End-to-End tests
set(E2E_TEST_SOURCES
    CompleteWorkflowTest.cpp
    PerformanceTest.cpp
)

add_executable(E2ETests ${E2E_TEST_SOURCES})

target_link_libraries(E2ETests
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_UI
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(E2ETests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Discover tests
gtest_discover_tests(E2ETests
    PROPERTIES
        LABELS "e2e"
        TIMEOUT 60
)
