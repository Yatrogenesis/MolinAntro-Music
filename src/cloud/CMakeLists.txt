set(CLOUD_SOURCES
    CloudCollaboration.cpp
    WebSocketClient.cpp
)

add_library(MolinAntro_Cloud STATIC ${CLOUD_SOURCES})

target_include_directories(MolinAntro_Cloud
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Cloud
    PUBLIC
        MolinAntro_Core
        MolinAntro_MIDI
        Threads::Threads
)

# =============================================================================
# REAL NETWORKING DEPENDENCIES - WebSocket++ and Asio
# =============================================================================

# Fetch WebSocket++ (header-only, BSD-3 license)
include(FetchContent)
FetchContent_Declare(
    websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
    GIT_TAG 0.8.2
)
FetchContent_MakeAvailable(websocketpp)

# Fetch standalone Asio (header-only, Boost license)
FetchContent_Declare(
    asio
    GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
    GIT_TAG asio-1-30-2
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(asio)

# Add include paths for websocketpp and asio
target_include_directories(MolinAntro_Cloud
    PRIVATE
        ${websocketpp_SOURCE_DIR}
        ${asio_SOURCE_DIR}/asio/include
)

# WebSocket++ compile definitions
target_compile_definitions(MolinAntro_Cloud
    PRIVATE
        ASIO_STANDALONE
        _WEBSOCKETPP_CPP11_THREAD_
)

# Windows-specific
if(WIN32)
    target_compile_definitions(MolinAntro_Cloud PRIVATE _WIN32_WINNT=0x0601)
    target_link_libraries(MolinAntro_Cloud PRIVATE ws2_32 wsock32)
endif()

# =============================================================================
# JSON Library (optional but recommended)
# =============================================================================

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)
target_link_libraries(MolinAntro_Cloud PRIVATE nlohmann_json::nlohmann_json)
target_compile_definitions(MolinAntro_Cloud PRIVATE HAVE_NLOHMANN_JSON=1)

# =============================================================================
# OpenSSL for TLS (REQUIRED for secure WebSocket)
# =============================================================================

find_package(OpenSSL REQUIRED)
target_link_libraries(MolinAntro_Cloud PRIVATE OpenSSL::SSL OpenSSL::Crypto)
target_compile_definitions(MolinAntro_Cloud PRIVATE HAVE_OPENSSL=1)

# =============================================================================
# Optional: libcurl for HTTP requests (REST API calls)
# =============================================================================

find_package(CURL)
if(CURL_FOUND)
    target_link_libraries(MolinAntro_Cloud PRIVATE CURL::libcurl)
    target_compile_definitions(MolinAntro_Cloud PRIVATE HAVE_CURL=1)
endif()
