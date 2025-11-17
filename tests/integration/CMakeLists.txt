# Integration tests
set(INTEGRATION_TEST_SOURCES
    EngineTransportIntegrationTest.cpp
    FileIOIntegrationTest.cpp
)

add_executable(IntegrationTests ${INTEGRATION_TEST_SOURCES})

target_link_libraries(IntegrationTests
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_UI
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(IntegrationTests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Discover tests
gtest_discover_tests(IntegrationTests
    PROPERTIES
        LABELS "integration"
)
