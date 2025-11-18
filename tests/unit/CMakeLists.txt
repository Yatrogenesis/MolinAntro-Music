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

# AI Module Tests (ACME Edition)
add_executable(AITests AITests.cpp)

target_link_libraries(AITests
    PRIVATE
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_MIDI
        MolinAntro_AI
        GTest::gtest_main
        GTest::gmock
)

target_include_directories(AITests
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

gtest_discover_tests(AITests
    PROPERTIES
        LABELS "ai;unit"
)
