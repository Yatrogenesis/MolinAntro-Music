# Unit tests
set(UNIT_TEST_SOURCES
    AudioBufferTest.cpp
    AudioEngineTest.cpp
    TransportTest.cpp
    AudioFileTest.cpp
)

add_executable(UnitTests ${UNIT_TEST_SOURCES})

target_link_libraries(UnitTests
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(UnitTests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

# Discover tests
gtest_discover_tests(UnitTests
    PROPERTIES
        LABELS "unit"
)
