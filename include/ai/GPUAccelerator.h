#pragma once

#include "core/AudioBuffer.h"
#include <complex>
#include <vector>
#include <memory>
#include <string>

namespace MolinAntro {
namespace AI {

/**
 * @brief GPU acceleration backend
 */
class GPUAccelerator {
public:
    /**
     * @brief Supported GPU backends
     */
    enum class Backend {
        CPU,        ///< CPU fallback (SIMD optimized)
        CUDA,       ///< NVIDIA CUDA
        Metal,      ///< Apple Metal
        OpenCL      ///< Cross-platform OpenCL
    };

    /**
     * @brief Device information
     */
    struct DeviceInfo {
        Backend backend{Backend::CPU};
        std::string name;
        size_t totalMemory{0};     ///< Total GPU memory (bytes)
        size_t freeMemory{0};      ///< Available GPU memory (bytes)
        int computeCapability{0};  ///< CUDA compute capability
        bool supportsFloat16{false};
        bool supportsFloat64{false};
    };

    GPUAccelerator();
    ~GPUAccelerator();

    /**
     * @brief Initialize GPU acceleration
     *
     * @param preferredBackend Preferred backend (auto-detected if CPU)
     * @return true if GPU available
     */
    bool initialize(Backend preferredBackend = Backend::CPU);

    /**
     * @brief Get current backend
     */
    Backend getBackend() const;

    /**
     * @brief Get device information
     */
    DeviceInfo getDeviceInfo() const;

    /**
     * @brief Check if GPU is available
     */
    bool isGPUAvailable() const;

    /**
     * @brief Detect best available backend
     */
    static Backend detectBestBackend();

    // ============================================
    // FFT Operations (GPU-accelerated)
    // ============================================

    /**
     * @brief GPU-accelerated FFT
     *
     * @param input Real-valued input signal
     * @param output Complex frequency domain output
     * @param size FFT size (power of 2)
     */
    void fft(const float* input,
            std::complex<float>* output,
            int size);

    /**
     * @brief GPU-accelerated inverse FFT
     */
    void ifft(const std::complex<float>* input,
             float* output,
             int size);

    /**
     * @brief Batch FFT (multiple channels)
     */
    void batchFFT(const float** inputs,
                 std::complex<float>** outputs,
                 int size,
                 int numChannels);

    // ============================================
    // Convolution (for reverb, filtering)
    // ============================================

    /**
     * @brief GPU-accelerated convolution
     *
     * @param signal Input signal
     * @param kernel Impulse response / filter kernel
     * @param output Output buffer
     * @param signalLen Signal length
     * @param kernelLen Kernel length
     */
    void convolve(const float* signal,
                 const float* kernel,
                 float* output,
                 int signalLen,
                 int kernelLen);

    /**
     * @brief Fast convolution using FFT (overlap-add)
     */
    void fastConvolve(const float* signal,
                     const float* kernel,
                     float* output,
                     int signalLen,
                     int kernelLen);

    // ============================================
    // Neural Network Inference
    // ============================================

    /**
     * @brief Load neural network model
     *
     * @param modelPath Path to ONNX model
     * @return Model ID for inference
     */
    int loadModel(const std::string& modelPath);

    /**
     * @brief Run inference on loaded model
     *
     * @param modelID Model identifier
     * @param input Input tensor
     * @param inputShape Input shape [batch, channels, height, width]
     * @param output Output tensor
     * @param outputShape Output shape
     */
    void runInference(int modelID,
                     const float* input,
                     const int* inputShape,
                     int numInputDims,
                     float* output,
                     int* outputShape,
                     int numOutputDims);

    /**
     * @brief Unload model from GPU memory
     */
    void unloadModel(int modelID);

    // ============================================
    // Matrix Operations
    // ============================================

    /**
     * @brief GPU matrix multiplication (C = A * B)
     */
    void matmul(const float* A,
               const float* B,
               float* C,
               int M, int N, int K);  // A: MxK, B: KxN, C: MxN

    /**
     * @brief Element-wise operations
     */
    void elementwiseAdd(const float* a, const float* b, float* result, int size);
    void elementwiseMul(const float* a, const float* b, float* result, int size);
    void elementwiseDiv(const float* a, const float* b, float* result, int size);

    // ============================================
    // Resampling & Pitch Shifting
    // ============================================

    /**
     * @brief GPU-accelerated resampling
     *
     * @param input Input samples
     * @param inputSize Input length
     * @param output Output buffer
     * @param outputSize Desired output length
     * @param quality Quality (0-10, higher=better)
     */
    void resample(const float* input,
                 int inputSize,
                 float* output,
                 int outputSize,
                 int quality = 5);

    /**
     * @brief Phase vocoder for pitch shifting
     */
    void pitchShift(const float* input,
                   float* output,
                   int size,
                   float pitchRatio,     // 2.0 = +12 semitones
                   bool preserveFormants = true);

    // ============================================
    // Memory Management
    // ============================================

    /**
     * @brief Allocate GPU memory
     */
    void* allocateGPU(size_t bytes);

    /**
     * @brief Free GPU memory
     */
    void freeGPU(void* ptr);

    /**
     * @brief Copy data to GPU
     */
    void copyToGPU(void* gpuPtr, const void* cpuPtr, size_t bytes);

    /**
     * @brief Copy data from GPU
     */
    void copyFromGPU(void* cpuPtr, const void* gpuPtr, size_t bytes);

    /**
     * @brief Synchronize GPU operations
     */
    void synchronize();

    // ============================================
    // Performance Monitoring
    // ============================================

    struct PerformanceStats {
        float lastOperationTime{0.0f};    ///< Last operation time (ms)
        float avgOperationTime{0.0f};     ///< Average operation time (ms)
        size_t gpuMemoryUsed{0};          ///< GPU memory in use (bytes)
        int operationCount{0};
        bool usingGPU{false};             ///< Whether GPU is being used
    };

    PerformanceStats getStats() const;
    void resetStats();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace AI
} // namespace MolinAntro
