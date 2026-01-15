# AI Module - MolinAntro DAW ACME Edition v3.0.0
# Voice Cloning, AI Mastering, Musical Analysis, GPU Acceleration

cmake_minimum_required(VERSION 3.20)

# AI Module Library
add_library(MolinAntro_AI
    VoiceCloning.cpp
    AIMastering.cpp
    MusicalAnalysis.cpp
    GPUAccelerator.cpp
    SmartInstruments.cpp
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
if(ACME_EDITION OR ENABLE_AI_MODULES)
    target_link_libraries(MolinAntro_AI PRIVATE onnxruntime)
    target_compile_definitions(MolinAntro_AI PRIVATE HAVE_ONNX=1)
    # Ensure includes are found if not propagated by FetchContent
    if(ONNXRUNTIME_INCLUDE_DIR)
        target_include_directories(MolinAntro_AI PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    endif()
endif()

# Platform-specific: Apple Accelerate
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    target_link_libraries(MolinAntro_AI PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(MolinAntro_AI PRIVATE ACCELERATE_NEW_LAPACK)
endif()

# Optional: CUDA support
find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    target_compile_definitions(MolinAntro_AI PRIVATE ENABLE_CUDA=1)
    target_link_libraries(MolinAntro_AI PRIVATE CUDA::cudart CUDA::cufft CUDA::cublas)
endif()

# Optional: OpenCL support
find_package(OpenCL)
if(OpenCL_FOUND)
    target_compile_definitions(MolinAntro_AI PRIVATE ENABLE_OPENCL=1)
    target_link_libraries(MolinAntro_AI PRIVATE OpenCL::OpenCL)
endif()

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
