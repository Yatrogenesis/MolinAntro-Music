/**
 * MolinAntro DAW - GPU-Accelerated Audio Processor
 * SOTA x5 Implementation - CUDA/Metal/OpenCL Backend
 *
 * Features:
 * - Real-time convolution reverb (up to 60 seconds IR)
 * - Parallel FFT processing (65536 points)
 * - Neural network inference acceleration
 * - Spectral processing at 768 kHz
 * - Zero-latency GPU-CPU synchronization
 *
 * Author: F. Molina-Burgos / MolinAntro Technologies
 * Copyright (C) 2026 - All Rights Reserved
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <queue>

namespace MolinAntro {
namespace DSP {
namespace GPU {

// =============================================================================
// GPU BACKEND ABSTRACTION
// =============================================================================

enum class GPUBackend {
    None,
    CUDA,
    Metal,
    OpenCL,
    Vulkan
};

enum class GPUMemoryType {
    DeviceOnly,     // GPU-only memory
    HostVisible,    // CPU-readable GPU memory
    Managed,        // Unified memory (CUDA managed)
    Pinned          // Pinned host memory for fast transfers
};

struct GPUDeviceInfo {
    std::string name;
    std::string vendor;
    uint64_t totalMemory;
    uint64_t availableMemory;
    int computeUnits;
    int maxWorkGroupSize;
    int warpSize;  // 32 for NVIDIA, 64 for AMD, varies for others
    bool supportsDoublePrecision;
    bool supportsAsyncTransfers;
    GPUBackend backend;
};

// =============================================================================
// GPU BUFFER
// =============================================================================

class GPUBuffer {
public:
    virtual ~GPUBuffer() = default;

    virtual void* getDevicePtr() = 0;
    virtual void* getHostPtr() = 0;
    virtual size_t size() const = 0;

    virtual void copyToDevice(const void* hostData, size_t bytes, size_t offset = 0) = 0;
    virtual void copyToHost(void* hostData, size_t bytes, size_t offset = 0) = 0;
    virtual void copyToDeviceAsync(const void* hostData, size_t bytes, size_t offset = 0) = 0;
    virtual void copyToHostAsync(void* hostData, size_t bytes, size_t offset = 0) = 0;

    virtual void synchronize() = 0;
};

// =============================================================================
// GPU KERNEL INTERFACE
// =============================================================================

class GPUKernel {
public:
    virtual ~GPUKernel() = default;

    virtual void setArgument(int index, const GPUBuffer& buffer) = 0;
    virtual void setArgument(int index, int value) = 0;
    virtual void setArgument(int index, float value) = 0;
    virtual void setArgument(int index, const void* data, size_t size) = 0;

    virtual void launch(int globalWorkSize, int localWorkSize = 0) = 0;
    virtual void launch2D(int globalX, int globalY, int localX = 0, int localY = 0) = 0;
    virtual void launch3D(int globalX, int globalY, int globalZ,
                         int localX = 0, int localY = 0, int localZ = 0) = 0;
};

// =============================================================================
// GPU CONTEXT
// =============================================================================

class GPUContext {
public:
    virtual ~GPUContext() = default;

    virtual bool initialize() = 0;
    virtual void shutdown() = 0;

    virtual GPUBackend getBackend() const = 0;
    virtual GPUDeviceInfo getDeviceInfo() const = 0;

    virtual std::unique_ptr<GPUBuffer> createBuffer(size_t size, GPUMemoryType type) = 0;
    virtual std::unique_ptr<GPUKernel> createKernel(const std::string& source, const std::string& entryPoint) = 0;

    virtual void synchronize() = 0;

    // Factory method
    static std::unique_ptr<GPUContext> create(GPUBackend preferredBackend = GPUBackend::None);
};

// =============================================================================
// GPU FFT PROCESSOR
// =============================================================================

class GPUFFTProcessor {
public:
    GPUFFTProcessor(GPUContext& context, size_t fftSize, size_t batchSize = 1);
    ~GPUFFTProcessor();

    void forward(const float* input, float* outputReal, float* outputImag);
    void inverse(const float* inputReal, const float* inputImag, float* output);

    void forwardBatch(const float* input, float* outputReal, float* outputImag, size_t numBatches);
    void inverseBatch(const float* inputReal, const float* inputImag, float* output, size_t numBatches);

    size_t getFFTSize() const { return fftSize_; }
    size_t getBatchSize() const { return batchSize_; }

private:
    GPUContext& context_;
    size_t fftSize_;
    size_t batchSize_;

    std::unique_ptr<GPUBuffer> inputBuffer_;
    std::unique_ptr<GPUBuffer> outputRealBuffer_;
    std::unique_ptr<GPUBuffer> outputImagBuffer_;
    std::unique_ptr<GPUBuffer> twiddleBuffer_;

    std::unique_ptr<GPUKernel> fftKernel_;
    std::unique_ptr<GPUKernel> ifftKernel_;

    void initializeTwiddles();
};

// =============================================================================
// GPU CONVOLUTION REVERB
// =============================================================================

class GPUConvolutionReverb {
public:
    // Supports impulse responses up to 60 seconds at 96kHz
    static constexpr size_t MAX_IR_SAMPLES = 60 * 96000;
    static constexpr size_t PARTITION_SIZE = 4096;

    GPUConvolutionReverb(GPUContext& context, size_t maxIRLength, size_t blockSize);
    ~GPUConvolutionReverb();

    void setImpulseResponse(const float* ir, size_t length, float sampleRate);
    void process(const float* input, float* output, size_t numSamples);
    void processStereo(const float* inputL, const float* inputR,
                       float* outputL, float* outputR, size_t numSamples);

    void setWetDryMix(float wetLevel, float dryLevel);
    void setPreDelay(float milliseconds);

    size_t getLatency() const { return latency_; }

private:
    GPUContext& context_;
    size_t maxIRLength_;
    size_t blockSize_;
    size_t latency_;

    float wetLevel_ = 1.0f;
    float dryLevel_ = 0.0f;
    size_t preDelaySamples_ = 0;

    // Partitioned convolution buffers
    size_t numPartitions_;
    std::unique_ptr<GPUBuffer> irPartitionsReal_;
    std::unique_ptr<GPUBuffer> irPartitionsImag_;
    std::unique_ptr<GPUBuffer> inputBuffer_;
    std::unique_ptr<GPUBuffer> fdlReal_;  // Frequency-domain delay line
    std::unique_ptr<GPUBuffer> fdlImag_;
    std::unique_ptr<GPUBuffer> accumReal_;
    std::unique_ptr<GPUBuffer> accumImag_;
    std::unique_ptr<GPUBuffer> outputBuffer_;

    std::unique_ptr<GPUFFTProcessor> fft_;
    std::unique_ptr<GPUKernel> complexMultiplyAccumKernel_;

    std::vector<float> overlapBuffer_;
    size_t fdlIndex_ = 0;

    void partitionIR(const float* ir, size_t length);
};

// =============================================================================
// GPU SPECTRAL PROCESSOR
// =============================================================================

class GPUSpectralProcessor {
public:
    enum class WindowType {
        Rectangular,
        Hann,
        Hamming,
        Blackman,
        BlackmanHarris,
        Kaiser,
        FlatTop,
        Gaussian
    };

    GPUSpectralProcessor(GPUContext& context, size_t fftSize = 4096, size_t hopSize = 1024);
    ~GPUSpectralProcessor();

    void setWindowType(WindowType type, float param = 0.0f);

    // Analysis
    void analyze(const float* input, size_t numSamples,
                 float* magnitudes, float* phases);

    // Synthesis
    void synthesize(const float* magnitudes, const float* phases,
                    float* output, size_t numSamples);

    // Processing
    using SpectralCallback = std::function<void(float* magnitudes, float* phases, size_t numBins)>;
    void process(const float* input, float* output, size_t numSamples, SpectralCallback callback);

    // Preset operations
    void stretch(const float* input, float* output, size_t inputSamples, float ratio);
    void pitchShift(const float* input, float* output, size_t numSamples, float semitones);
    void denoise(const float* input, float* output, size_t numSamples, float threshold);
    void harmonicPercussiveSeparation(const float* input, float* harmonic, float* percussive, size_t numSamples);

private:
    GPUContext& context_;
    size_t fftSize_;
    size_t hopSize_;

    std::unique_ptr<GPUBuffer> windowBuffer_;
    std::unique_ptr<GPUBuffer> inputBuffer_;
    std::unique_ptr<GPUBuffer> outputBuffer_;
    std::unique_ptr<GPUBuffer> magnitudeBuffer_;
    std::unique_ptr<GPUBuffer> phaseBuffer_;
    std::unique_ptr<GPUBuffer> prevPhaseBuffer_;
    std::unique_ptr<GPUBuffer> synthPhaseBuffer_;

    std::unique_ptr<GPUFFTProcessor> fft_;
    std::unique_ptr<GPUKernel> windowKernel_;
    std::unique_ptr<GPUKernel> cartesianToPolarKernel_;
    std::unique_ptr<GPUKernel> polarToCartesianKernel_;
    std::unique_ptr<GPUKernel> overlapAddKernel_;

    std::vector<float> analysisWindow_;
    std::vector<float> synthesisWindow_;
    std::vector<float> overlapBuffer_;

    void generateWindow(WindowType type, float param);
};

// =============================================================================
// GPU NEURAL PROCESSOR
// =============================================================================

class GPUNeuralProcessor {
public:
    GPUNeuralProcessor(GPUContext& context);
    ~GPUNeuralProcessor();

    // Load ONNX model
    bool loadModel(const std::string& modelPath);
    bool loadModelFromMemory(const void* data, size_t size);

    // Get model info
    std::vector<std::string> getInputNames() const;
    std::vector<std::string> getOutputNames() const;
    std::vector<int64_t> getInputShape(const std::string& name) const;
    std::vector<int64_t> getOutputShape(const std::string& name) const;

    // Inference
    void setInput(const std::string& name, const float* data, const std::vector<int64_t>& shape);
    void run();
    void getOutput(const std::string& name, float* data);

    // Real-time audio inference
    void processAudioBlock(const float* input, float* output, size_t numSamples, size_t channels);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    GPUContext& context_;
};

// =============================================================================
// GPU AUDIO STREAM
// =============================================================================

class GPUAudioStream {
public:
    using ProcessCallback = std::function<void(const float* input, float* output,
                                               size_t numSamples, size_t numChannels)>;

    GPUAudioStream(GPUContext& context, size_t bufferSize, size_t numChannels);
    ~GPUAudioStream();

    void setProcessCallback(ProcessCallback callback);

    void push(const float* input, size_t numSamples);
    void pop(float* output, size_t numSamples);

    bool processAsync();
    void synchronize();

    size_t getLatency() const { return latency_; }
    float getGPUUtilization() const;

private:
    GPUContext& context_;
    size_t bufferSize_;
    size_t numChannels_;
    size_t latency_;

    ProcessCallback processCallback_;

    std::unique_ptr<GPUBuffer> inputRingBuffer_;
    std::unique_ptr<GPUBuffer> outputRingBuffer_;
    std::unique_ptr<GPUBuffer> processingBuffer_;

    std::atomic<size_t> writeIndex_{0};
    std::atomic<size_t> readIndex_{0};

    std::thread processingThread_;
    std::atomic<bool> running_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
};

// =============================================================================
// GPU STEM SEPARATOR (AI-based)
// =============================================================================

class GPUStemSeparator {
public:
    enum class StemType {
        Vocals,
        Drums,
        Bass,
        Other,
        Piano,
        Guitar
    };

    GPUStemSeparator(GPUContext& context);
    ~GPUStemSeparator();

    bool loadModel(const std::string& modelPath);  // Demucs/HTDemucs ONNX

    // Process entire file
    void separate(const float* input, size_t numSamples, size_t channels,
                  std::vector<std::vector<float>>& stems);

    // Real-time processing (higher latency, ~6 seconds)
    void separateRealtime(const float* input, float* vocals, float* drums,
                          float* bass, float* other, size_t numSamples);

    void setStemGains(float vocals, float drums, float bass, float other);

private:
    GPUContext& context_;
    std::unique_ptr<GPUNeuralProcessor> model_;

    size_t chunkSize_ = 343980;  // ~7.15 seconds at 48kHz (Demucs default)
    size_t overlap_ = 171990;    // 50% overlap
    size_t sampleRate_ = 48000;

    float stemGains_[6] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffers_[6];
};

// =============================================================================
// GPU MASTERING ENGINE
// =============================================================================

class GPUMasteringEngine {
public:
    struct Parameters {
        // EQ
        float lowGain = 0.0f;     // -12 to +12 dB
        float midGain = 0.0f;
        float highGain = 0.0f;

        // Compression
        float compressionThreshold = -10.0f;  // dB
        float compressionRatio = 3.0f;
        float compressionAttack = 10.0f;      // ms
        float compressionRelease = 100.0f;    // ms

        // Limiting
        float limiterThreshold = -1.0f;       // dB
        float limiterRelease = 50.0f;         // ms
        float truePeak = -0.3f;               // dBTP

        // Stereo
        float stereoWidth = 1.0f;             // 0-2
        float midSideBalance = 0.5f;          // 0-1

        // Loudness
        float targetLUFS = -14.0f;            // -24 to -6
        bool loudnessNormalize = true;

        // AI Enhancement
        bool useAI = false;
        float aiStrength = 0.5f;
    };

    GPUMasteringEngine(GPUContext& context);
    ~GPUMasteringEngine();

    bool loadAIModel(const std::string& modelPath);

    void setParameters(const Parameters& params);
    Parameters getParameters() const { return params_; }

    void process(const float* input, float* output, size_t numSamples, size_t channels);

    // Analysis
    float getLoudness() const;           // Integrated LUFS
    float getTruePeakLevel() const;      // dBTP
    float getDynamicRange() const;       // LU
    float getStereoCorrelation() const;  // -1 to 1

private:
    GPUContext& context_;
    Parameters params_;

    std::unique_ptr<GPUBuffer> inputBuffer_;
    std::unique_ptr<GPUBuffer> outputBuffer_;
    std::unique_ptr<GPUBuffer> eqBuffer_;
    std::unique_ptr<GPUBuffer> compBuffer_;
    std::unique_ptr<GPUBuffer> limiterBuffer_;

    std::unique_ptr<GPUKernel> eqKernel_;
    std::unique_ptr<GPUKernel> compressorKernel_;
    std::unique_ptr<GPUKernel> limiterKernel_;
    std::unique_ptr<GPUKernel> stereoKernel_;
    std::unique_ptr<GPUKernel> loudnessKernel_;

    std::unique_ptr<GPUNeuralProcessor> aiModel_;

    std::atomic<float> currentLUFS_{-70.0f};
    std::atomic<float> currentTruePeak_{-96.0f};
    std::atomic<float> dynamicRange_{0.0f};
    std::atomic<float> stereoCorrelation_{1.0f};
};

} // namespace GPU
} // namespace DSP
} // namespace MolinAntro
