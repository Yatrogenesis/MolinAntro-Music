#pragma once

#include "core/AudioBuffer.h"
#include "dsp/Effects.h"
#include <vector>
#include <complex>
#include <memory>
#include <string>
#include <mutex>

namespace MolinAntro {
namespace DSP {

/**
 * @brief Professional Convolution Reverb Engine
 *
 * Features:
 * - FFT-based convolution using overlap-add method
 * - Zero-latency mode with partitioned convolution
 * - Stereo true-stereo (4-channel) IR support
 * - Pre-delay control
 * - Wet/dry mix
 * - IR trimming and normalization
 *
 * Supports IR formats: WAV, AIFF (via AudioFile)
 *
 * Author: F. Molina-Burgos, MolinAntro Technologies
 */
class ConvolutionReverb : public AudioEffect {
public:
    // Maximum IR length: 10 seconds at 96kHz
    static constexpr size_t MAX_IR_SAMPLES = 960000;
    static constexpr int MIN_FFT_SIZE = 256;
    static constexpr int MAX_FFT_SIZE = 8192;

    /**
     * @brief IR loading result
     */
    struct IRInfo {
        int sampleRate;
        int numChannels;
        size_t numSamples;
        float lengthSeconds;
        bool isTrueStereo;  // 4-channel IR (LL, LR, RL, RR)
        std::string name;
    };

    ConvolutionReverb();
    ~ConvolutionReverb() override;

    // AudioEffect interface
    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    // ========================================================================
    // IR Management
    // ========================================================================

    /**
     * @brief Load impulse response from file
     * @param filepath Path to WAV/AIFF file
     * @return true if loaded successfully
     */
    bool loadIR(const std::string& filepath);

    /**
     * @brief Load impulse response from buffer
     * @param ir Impulse response buffer (mono or stereo)
     * @param irSampleRate Sample rate of the IR
     */
    void loadIR(const Core::AudioBuffer& ir, int irSampleRate);

    /**
     * @brief Load true-stereo IR (4-channel: LL, LR, RL, RR)
     */
    void loadTrueStereoIR(const Core::AudioBuffer& ir, int irSampleRate);

    /**
     * @brief Get information about loaded IR
     */
    const IRInfo& getIRInfo() const { return irInfo_; }

    /**
     * @brief Check if an IR is loaded
     */
    bool isIRLoaded() const { return irLoaded_; }

    // ========================================================================
    // Parameters
    // ========================================================================

    /**
     * @brief Set wet/dry mix (0.0 = dry, 1.0 = wet)
     */
    void setMix(float mix) { mix_ = std::clamp(mix, 0.0f, 1.0f); }
    float getMix() const { return mix_; }

    /**
     * @brief Set pre-delay in milliseconds
     */
    void setPreDelay(float ms);
    float getPreDelay() const { return preDelayMs_; }

    /**
     * @brief Set output gain in dB
     */
    void setGain(float dB) { gainDB_ = std::clamp(dB, -60.0f, 12.0f); }
    float getGain() const { return gainDB_; }

    /**
     * @brief Enable/disable low CPU mode (higher latency)
     */
    void setLowCPUMode(bool enable);
    bool isLowCPUMode() const { return lowCPUMode_; }

    /**
     * @brief Set IR trim (start/end in seconds)
     */
    void setIRTrim(float startSec, float endSec);

    /**
     * @brief Enable/disable IR normalization
     */
    void setNormalize(bool normalize) { normalize_ = normalize; if (irLoaded_) reprocessIR(); }

    /**
     * @brief Set high-frequency damping (0.0 = none, 1.0 = maximum)
     */
    void setDamping(float damping) { damping_ = std::clamp(damping, 0.0f, 1.0f); }
    float getDamping() const { return damping_; }

    // ========================================================================
    // Presets
    // ========================================================================

    /**
     * @brief Generate synthetic IR (for when no file is available)
     */
    void generateSyntheticIR(float decayTime, float roomSize, float brightness);

private:
    // ========================================================================
    // FFT Convolution Engine
    // ========================================================================

    /**
     * @brief Single partition for partitioned convolution
     */
    struct Partition {
        std::vector<std::complex<float>> irFreqL;
        std::vector<std::complex<float>> irFreqR;
        // For true stereo
        std::vector<std::complex<float>> irFreqLR;
        std::vector<std::complex<float>> irFreqRL;
    };

    /**
     * @brief Convolution state for one channel
     */
    struct ConvolutionState {
        std::vector<float> inputBuffer;
        std::vector<float> outputBuffer;
        std::vector<std::complex<float>> fftBuffer;
        std::vector<std::complex<float>> accumulator;
        int inputWritePos;
        int outputReadPos;
        int partitionIndex;
    };

    // FFT operations
    void fft(std::vector<std::complex<float>>& data, bool inverse = false);
    void fftIterative(std::complex<float>* data, int n, bool inverse);
    int nextPowerOfTwo(int n);

    // IR processing
    void reprocessIR();
    void partitionIR();
    void normalizeIR(std::vector<float>& ir);
    void applyDampingToIR();
    void resampleIR(const std::vector<float>& input, int inputRate,
                    std::vector<float>& output, int outputRate);

    // Convolution processing
    void processPartitionedConvolution(float* left, float* right, int numSamples);
    void processDirectConvolution(float* left, float* right, int numSamples);

    // Pre-delay
    void processPreDelay(float* left, float* right, int numSamples);

    // ========================================================================
    // State
    // ========================================================================

    bool irLoaded_ = false;
    IRInfo irInfo_;

    // Raw IR data (before FFT)
    std::vector<float> irL_;
    std::vector<float> irR_;
    std::vector<float> irLR_;  // True stereo
    std::vector<float> irRL_;  // True stereo
    bool trueStereo_ = false;

    // FFT partitions
    std::vector<Partition> partitions_;
    int fftSize_ = 2048;
    int partitionSize_ = 1024;  // fftSize / 2
    int numPartitions_ = 0;

    // Convolution state
    ConvolutionState stateL_;
    ConvolutionState stateR_;

    // Pre-delay buffer
    std::vector<float> preDelayBufferL_;
    std::vector<float> preDelayBufferR_;
    int preDelayWritePos_ = 0;
    int preDelaySamples_ = 0;
    float preDelayMs_ = 0.0f;

    // Parameters
    float mix_ = 0.5f;
    float gainDB_ = 0.0f;
    float damping_ = 0.0f;
    bool normalize_ = true;
    bool lowCPUMode_ = false;

    // IR trim
    float trimStartSec_ = 0.0f;
    float trimEndSec_ = 0.0f;

    // Thread safety for IR loading
    std::mutex irMutex_;

    // Temp buffers for processing
    std::vector<float> tempL_;
    std::vector<float> tempR_;
    std::vector<float> wetL_;
    std::vector<float> wetR_;
};

/**
 * @brief Factory for common reverb IRs
 */
class IRFactory {
public:
    /**
     * @brief Generate exponentially decaying IR
     */
    static std::vector<float> generateExponentialDecay(
        int sampleRate, float decayTime, float density = 1.0f);

    /**
     * @brief Generate room simulation IR
     */
    static std::vector<float> generateRoom(
        int sampleRate, float roomSizeMeters, float rt60, float brightness);

    /**
     * @brief Generate plate reverb IR
     */
    static std::vector<float> generatePlate(
        int sampleRate, float decayTime, float damping);

    /**
     * @brief Generate spring reverb IR
     */
    static std::vector<float> generateSpring(
        int sampleRate, float decayTime, float tension);

    /**
     * @brief Generate hall reverb IR
     */
    static std::vector<float> generateHall(
        int sampleRate, float rt60, float diffusion);
};

} // namespace DSP
} // namespace MolinAntro
