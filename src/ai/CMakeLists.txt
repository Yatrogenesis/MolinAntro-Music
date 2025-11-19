# AI Module - MolinAntro DAW ACME Edition v3.0.0
# Voice Cloning, AI Mastering, Musical Analysis, GPU Acceleration

cmake_minimum_required(VERSION 3.20)

# AI Module Library
add_library(MolinAntro_AI
    VoiceCloning.cpp
    AIMastering.cpp
    MusicalAnalysis.cpp
    GPUAccelerator.cpp
)

target_include_directories(MolinAntro_AI
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/ai
)

# Link dependencies
target_link_libraries(MolinAntro_AI
    PUBLIC
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_MIDI
        Threads::Threads
)

# Platform-specific optimizations
if(UNIX AND NOT APPLE)
    target_link_libraries(MolinAntro_AI PRIVATE m) # Link math library on Linux
endif()

# Optional: ONNX Runtime (for production ML models)
# find_package(onnxruntime)
# if(onnxruntime_FOUND)
#     target_link_libraries(MolinAntro_AI PRIVATE onnxruntime::onnxruntime)
#     target_compile_definitions(MolinAntro_AI PRIVATE HAVE_ONNX=1)
# endif()

# Optional: CUDA support
# find_package(CUDA)
# if(CUDA_FOUND)
#     target_compile_definitions(MolinAntro_AI PRIVATE __CUDA__=1)
#     target_link_libraries(MolinAntro_AI PRIVATE ${CUDA_LIBRARIES})
# endif()

# Compiler settings
target_compile_features(MolinAntro_AI PUBLIC cxx_std_20)

# Export symbols for shared library
set_target_properties(MolinAntro_AI PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# Installation
install(TARGETS MolinAntro_AI
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
