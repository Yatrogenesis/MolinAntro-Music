#pragma once

/**
 * @file DemucsONNX.h
 * @brief SOTA Stem Separation using Demucs/HTDemucs ONNX models
 *
 * This implements real deep learning-based source separation compatible with:
 * - Demucs v4 (Hybrid Transformer)
 * - HTDemucs (Hybrid Time-Frequency Domain)
 * - Open-Unmix
 * - Spleeter (legacy)
 *
 * Architecture:
 * - Time-domain U-Net with skip connections
 * - Bi-LSTM temporal modeling
 * - Multi-head self-attention (Transformer variant)
 * - Spectrogram masking refinement
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "core/AudioBuffer.h"
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <functional>
#include <array>

// Forward declarations for ONNX Runtime
namespace Ort {
    class Env;
    class Session;
    class SessionOptions;
    class MemoryInfo;
    class Value;
}

namespace MolinAntro {
namespace DSP {

/**
 * @brief Model architecture types supported
 */
enum class DemucsModelType {
    Demucs_v2,          // Original time-domain U-Net
    Demucs_v3,          // Hybrid time-frequency
    HTDemucs,           // Hybrid Transformer Demucs (SOTA)
    OpenUnmix,          // Open-Unmix spectrogram masking
    Spleeter,           // Deezer Spleeter (legacy)
    Custom              // User-provided ONNX model
};

/**
 * @brief Stem types for separation
 */
enum class StemType {
    Vocals,
    Drums,
    Bass,
    Other,
    Piano,      // 5-stem models
    Guitar      // 6-stem models
};

/**
 * @brief Configuration for ONNX inference
 */
struct DemucsConfig {
    DemucsModelType modelType = DemucsModelType::HTDemucs;

    // Model paths
    std::string modelPath;                      // Main model
    std::string vocalsModelPath;                // Separate vocal model (optional)

    // Audio processing
    int sampleRate = 44100;                     // Model expected sample rate
    int channels = 2;                           // Stereo
    int segmentLength = 44100 * 10;             // 10 seconds default
    int overlap = 44100 * 1;                    // 1 second overlap

    // STFT parameters (for hybrid models)
    int nfft = 4096;
    int hopLength = 1024;
    int winLength = 4096;

    // Inference
    bool useGPU = true;                         // CUDA/DirectML/CoreML
    int gpuDeviceId = 0;
    int numThreads = 4;                         // CPU threads
    bool fp16 = false;                          // Half precision (faster on GPU)

    // Output
    int numStems = 4;                           // 4 or 5 or 6 stems
    float softmaskTemperature = 1.0f;           // Softmax temperature for masking
    bool residualOther = true;                  // Other = Input - (Vocals + Drums + Bass)
};

/**
 * @brief Separated stems output
 */
struct SeparatedStems {
    std::unique_ptr<Core::AudioBuffer> vocals;
    std::unique_ptr<Core::AudioBuffer> drums;
    std::unique_ptr<Core::AudioBuffer> bass;
    std::unique_ptr<Core::AudioBuffer> other;
    std::unique_ptr<Core::AudioBuffer> piano;   // 5-stem only
    std::unique_ptr<Core::AudioBuffer> guitar;  // 6-stem only

    // Confidence scores per stem (0-1)
    std::array<float, 6> confidence = {0};

    // Processing stats
    float processingTimeMs = 0.0f;
    bool usedGPU = false;
};

/**
 * @brief Progress callback for long operations
 */
using ProgressCallback = std::function<void(float progress, const std::string& stage)>;

/**
 * @brief SOTA Stem Separator using ONNX Runtime
 *
 * Implements Demucs/HTDemucs architecture for state-of-the-art source separation.
 * Quality matches or exceeds commercial solutions like iZotope RX, LALAL.AI, etc.
 */
class DemucsONNX {
public:
    DemucsONNX();
    ~DemucsONNX();

    // Prevent copying (ONNX sessions are not copyable)
    DemucsONNX(const DemucsONNX&) = delete;
    DemucsONNX& operator=(const DemucsONNX&) = delete;
    DemucsONNX(DemucsONNX&&) noexcept;
    DemucsONNX& operator=(DemucsONNX&&) noexcept;

    /**
     * @brief Initialize with configuration
     * @param config Model and processing configuration
     * @return true if initialization successful
     */
    bool initialize(const DemucsConfig& config);

    /**
     * @brief Load ONNX model from file
     * @param modelPath Path to .onnx model file
     * @param modelType Type of model architecture
     * @return true if model loaded successfully
     */
    bool loadModel(const std::string& modelPath, DemucsModelType modelType = DemucsModelType::HTDemucs);

    /**
     * @brief Check if model is loaded and ready
     */
    bool isReady() const;

    /**
     * @brief Get model information
     */
    std::string getModelInfo() const;

    /**
     * @brief Separate stems from audio
     * @param input Input audio buffer (stereo)
     * @return Separated stems
     */
    SeparatedStems separate(const Core::AudioBuffer& input);

    /**
     * @brief Separate with progress callback
     */
    SeparatedStems separate(const Core::AudioBuffer& input, ProgressCallback callback);

    /**
     * @brief Extract single stem
     * @param input Input audio
     * @param stem Which stem to extract
     * @return Audio buffer containing only the requested stem
     */
    std::unique_ptr<Core::AudioBuffer> extractStem(const Core::AudioBuffer& input, StemType stem);

    /**
     * @brief Real-time streaming separation (lower quality, lower latency)
     * @param input Input block
     * @param outputs Output buffers for each stem
     * @param blockSize Processing block size
     */
    void processBlock(const Core::AudioBuffer& input,
                      std::array<Core::AudioBuffer*, 4>& outputs,
                      int blockSize);

    /**
     * @brief Prepare for real-time processing
     */
    void prepareRealtime(int sampleRate, int blockSize);

    /**
     * @brief Reset internal state (for streaming)
     */
    void reset();

    // Configuration
    void setConfig(const DemucsConfig& config);
    const DemucsConfig& getConfig() const { return config_; }

    // GPU control
    bool isGPUAvailable() const;
    void setUseGPU(bool useGPU);
    std::string getGPUInfo() const;

private:
    // ONNX Runtime components
    struct OrtComponents;
    std::unique_ptr<OrtComponents> ort_;

    // Configuration
    DemucsConfig config_;
    bool initialized_ = false;

    // Audio processing
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffers_[6];  // Up to 6 stems

    // STFT buffers (for hybrid models)
    std::vector<std::complex<float>> stftBuffer_;
    std::vector<float> window_;

    // Streaming state
    std::vector<float> overlapBuffer_;
    int overlapSamples_ = 0;

    // Internal methods
    void initializeONNX();
    void setupExecutionProvider();

    // Preprocessing
    std::vector<float> preprocessAudio(const Core::AudioBuffer& input);
    void normalizeAudio(std::vector<float>& audio);
    void resampleIfNeeded(std::vector<float>& audio, int sourceSR, int targetSR);

    // STFT operations
    void computeSTFT(const std::vector<float>& audio,
                     std::vector<std::complex<float>>& spectrum);
    void computeISTFT(const std::vector<std::complex<float>>& spectrum,
                      std::vector<float>& audio);
    void createWindow(int size);

    // Inference
    std::vector<std::vector<float>> runInference(const std::vector<float>& input);
    std::vector<std::vector<float>> runHybridInference(const std::vector<float>& timeDomain,
                                                        const std::vector<std::complex<float>>& freqDomain);

    // Mask estimation (for spectrogram models)
    std::vector<std::vector<float>> estimateMasks(const std::vector<std::complex<float>>& mixture);
    void applySoftmask(std::vector<std::complex<float>>& spectrum,
                       const std::vector<float>& mask,
                       float temperature);

    // Postprocessing
    void postprocessStems(std::vector<std::vector<float>>& stems);
    std::unique_ptr<Core::AudioBuffer> vectorToBuffer(const std::vector<float>& data, int channels);

    // Overlap-add for streaming
    void processWithOverlap(const std::vector<float>& input,
                            std::vector<std::vector<float>>& outputs);
    void applyOverlapAdd(std::vector<float>& output,
                         const std::vector<float>& segment,
                         int position);

    // Utility
    float calculateSDR(const std::vector<float>& reference,
                       const std::vector<float>& estimate);
};

/**
 * @brief Factory for creating DemucsONNX instances with preset configurations
 */
class DemucsFactory {
public:
    /**
     * @brief Create instance optimized for quality (HTDemucs)
     */
    static std::unique_ptr<DemucsONNX> createHighQuality(const std::string& modelPath);

    /**
     * @brief Create instance optimized for speed
     */
    static std::unique_ptr<DemucsONNX> createFast(const std::string& modelPath);

    /**
     * @brief Create instance for real-time streaming
     */
    static std::unique_ptr<DemucsONNX> createRealtime(const std::string& modelPath, int sampleRate);

    /**
     * @brief Create instance for 5-stem separation (includes piano)
     */
    static std::unique_ptr<DemucsONNX> create5Stem(const std::string& modelPath);

    /**
     * @brief Download pretrained model from HuggingFace
     * @param modelName Model identifier (e.g., "htdemucs", "htdemucs_ft")
     * @param outputPath Where to save the model
     * @return true if download successful
     */
    static bool downloadModel(const std::string& modelName, const std::string& outputPath);

    /**
     * @brief List available models
     */
    static std::vector<std::string> listAvailableModels();

    /**
     * @brief Get recommended model for use case
     */
    static std::string getRecommendedModel(bool realtime, int numStems);
};

/**
 * @brief Benchmark utility for stem separation
 */
class DemucsBenchmark {
public:
    struct BenchmarkResult {
        float processingTimeMs;
        float realtimeFactor;      // >1 means faster than real-time
        float vocalsSDR;           // Signal-to-Distortion Ratio (dB)
        float drumsSDR;
        float bassSDR;
        float otherSDR;
        float avgSDR;
        float gpuMemoryMB;
        float cpuUsagePercent;
    };

    /**
     * @brief Run benchmark on test audio
     */
    static BenchmarkResult benchmark(DemucsONNX& separator,
                                      const Core::AudioBuffer& testAudio,
                                      const SeparatedStems* groundTruth = nullptr);

    /**
     * @brief Compare quality with reference implementation
     */
    static void compareWithReference(DemucsONNX& separator,
                                      const std::string& testFile,
                                      const std::string& referenceDir);
};

} // namespace DSP
} // namespace MolinAntro
