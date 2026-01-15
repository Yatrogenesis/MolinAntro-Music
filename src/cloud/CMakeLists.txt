set(CLOUD_SOURCES
    CloudCollaboration.cpp
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

# Optional: libcurl for HTTP requests
find_package(CURL)
if(CURL_FOUND)
    target_link_libraries(MolinAntro_Cloud PRIVATE CURL::libcurl)
    target_compile_definitions(MolinAntro_Cloud PRIVATE HAVE_CURL=1)
endif()

# Optional: OpenSSL for encryption
find_package(OpenSSL)
if(OpenSSL_FOUND)
    target_link_libraries(MolinAntro_Cloud PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    target_compile_definitions(MolinAntro_Cloud PRIVATE HAVE_OPENSSL=1)
endif()
