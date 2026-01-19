#pragma once

/**
 * @file StemSeparation.h
 * @brief SOTA Stem Separation with Hybrid Architecture
 *
 * Provides state-of-the-art audio source separation using:
 * - Layer 1: DemucsONNX (Neural ONNX inference) - Best quality
 * - Layer 2: HPSS + NMF (DSP fallback) - No model required
 * - Layer 3: Frequency Masking (Fast fallback) - Real-time
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "core/AudioBuffer.h"
#include "dsp/SpectralProcessor.h"
#include "dsp/DemucsONNX.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <array>

namespace MolinAntro {
namespace DSP {

/**
 * @brief Separation quality/speed tradeoff
 */
enum class SeparationQuality {
    Realtime,       // Frequency masking (lowest latency)
    Fast,           // HPSS-based (good quality, fast)
    Standard,       // NMF (better quality)
    HighQuality,    // Neural if available, NMF fallback
    Maximum         // Neural only (fails if no model)
};

/**
 * @brief Unified stem separation result
 */
struct StemSeparationResult {
    std::unique_ptr<Core::AudioBuffer> vocals;
    std::unique_ptr<Core::AudioBuffer> drums;
    std::unique_ptr<Core::AudioBuffer> bass;
    std::unique_ptr<Core::AudioBuffer> other;
    std::unique_ptr<Core::AudioBuffer> piano;   // 5-stem models
    std::unique_ptr<Core::AudioBuffer> guitar;  // 6-stem models

    // Quality metrics
    std::array<float, 6> confidence = {0};      // Per-stem confidence 0-1
    float estimatedSDR = 0.0f;                  // Estimated Signal-Distortion Ratio

    // Processing info
    std::string methodUsed;                     // "Neural", "HPSS", "NMF", "FreqMask"
    float processingTimeMs = 0.0f;
    bool usedGPU = false;
    int numStems = 4;
};

/**
 * @brief Progress callback for long operations
 */
using SeparationProgressCallback = std::function<void(float progress, const std::string& stage)>;

/**
 * @brief SOTA Unified Stem Separator
 *
 * Primary interface for stem separation. Automatically selects the best
 * available method based on configuration and hardware.
 *
 * Architecture:
 *   [Input Audio]
 *        │
 *        ▼
 *   ┌─────────────────────────────────────────┐
 *   │  Try Neural (DemucsONNX)                │
 *   │  - Requires .onnx model                 │
 *   │  - Best quality (SDR > 8dB)             │
 *   └────────────────┬────────────────────────┘
 *                    │ Model not available?
 *                    ▼
 *   ┌─────────────────────────────────────────┐
 *   │  Fallback to DSP (HPSS/NMF)             │
 *   │  - No model required                    │
 *   │  - Good quality (SDR ~5-6dB)            │
 *   └────────────────┬────────────────────────┘
 *                    │ Need real-time?
 *                    ▼
 *   ┌─────────────────────────────────────────┐
 *   │  Frequency Masking                      │
 *   │  - Lowest latency                       │
 *   │  - Basic quality (SDR ~2-3dB)           │
 *   └─────────────────────────────────────────┘
 */
class StemSeparator {
public:
    StemSeparator();
    ~StemSeparator();

    // Prevent copying
    StemSeparator(const StemSeparator&) = delete;
    StemSeparator& operator=(const StemSeparator&) = delete;
    StemSeparator(StemSeparator&&) noexcept;
    StemSeparator& operator=(StemSeparator&&) noexcept;

    /**
     * @brief Initialize with sample rate
     */
    void initialize(float sampleRate);

    /**
     * @brief Load neural model for SOTA quality
     * @param modelPath Path to .onnx model file
     * @param modelType Type of model architecture
     * @return true if model loaded successfully
     */
    bool loadNeuralModel(const std::string& modelPath,
                         DemucsModelType modelType = DemucsModelType::HTDemucs);

    /**
     * @brief Check if neural model is loaded
     */
    bool hasNeuralModel() const;

    /**
     * @brief Get information about loaded model
     */
    std::string getModelInfo() const;

    /**
     * @brief Separate audio into stems
     * @param input Input audio buffer (mono or stereo)
     * @param quality Quality/speed tradeoff
     * @return Separated stems
     */
    StemSeparationResult separate(const Core::AudioBuffer& input,
                                   SeparationQuality quality = SeparationQuality::HighQuality);

    /**
     * @brief Separate with progress callback
     */
    StemSeparationResult separate(const Core::AudioBuffer& input,
                                   SeparationProgressCallback callback,
                                   SeparationQuality quality = SeparationQuality::HighQuality);

    /**
     * @brief Extract single stem (more efficient for single stem)
     */
    std::unique_ptr<Core::AudioBuffer> extractStem(const Core::AudioBuffer& input,
                                                    StemType stem,
                                                    SeparationQuality quality = SeparationQuality::HighQuality);

    // Real-time processing
    void prepareRealtime(int blockSize);
    void processBlock(const Core::AudioBuffer& input,
                      std::array<Core::AudioBuffer*, 4>& outputs);
    void reset();

    // Configuration
    void setNumStems(int numStems);  // 4, 5, or 6
    int getNumStems() const { return numStems_; }

    void setUseGPU(bool useGPU);
    bool isGPUAvailable() const;

    /**
     * @brief Get estimated quality for current configuration
     * @return Estimated SDR in dB
     */
    float getEstimatedQuality(SeparationQuality quality) const;

private:
    float sampleRate_ = 48000.0f;
    int numStems_ = 4;
    bool useGPU_ = true;

    // Neural separator (SOTA)
    std::unique_ptr<DemucsONNX> neural_;

    // DSP fallback separators
    class HPSSSeparator;
    std::unique_ptr<HPSSSeparator> hpss_;

    class NMFSeparatorInternal;
    std::unique_ptr<NMFSeparatorInternal> nmf_;

    class FrequencyMaskSeparator;
    std::unique_ptr<FrequencyMaskSeparator> freqMask_;

    // Internal methods
    StemSeparationResult separateNeural(const Core::AudioBuffer& input,
                                         SeparationProgressCallback callback);
    StemSeparationResult separateHPSS(const Core::AudioBuffer& input,
                                       SeparationProgressCallback callback);
    StemSeparationResult separateNMF(const Core::AudioBuffer& input,
                                      SeparationProgressCallback callback);
    StemSeparationResult separateFreqMask(const Core::AudioBuffer& input);

    // Quality estimation
    void estimateConfidence(StemSeparationResult& result);
};

// ============================================================================
// Legacy classes (kept for backwards compatibility)
// ============================================================================

/**
 * @brief NMF-based stem separator
 * @deprecated Use StemSeparator with SeparationQuality::Standard
 */
class NMFStemSeparator {
public:
    struct SeparatedStems {
        std::unique_ptr<Core::AudioBuffer> vocals;
        std::unique_ptr<Core::AudioBuffer> drums;
        std::unique_ptr<Core::AudioBuffer> bass;
        std::unique_ptr<Core::AudioBuffer> other;
    };

    NMFStemSeparator();
    ~NMFStemSeparator();

    void setNumComponents(int numComponents);
    void setNumIterations(int iterations);
    void setSampleRate(float sampleRate);

    SeparatedStems separate(const Core::AudioBuffer& input);

    using ProgressCallback = std::function<void(float progress, const std::string& message)>;
    void setProgressCallback(ProgressCallback callback);

private:
    int numComponents_;
    int numIterations_;
    float sampleRate_;
    ProgressCallback progressCallback_;

    std::unique_ptr<SpectralProcessor> spectralProcessor_;

    struct NMFResult {
        std::vector<std::vector<float>> W;
        std::vector<std::vector<float>> H;
    };

    NMFResult performNMF(const std::vector<std::vector<float>>& V);
    void initializeMatrices(NMFResult& result, int rows, int cols);
    void updateNMF(NMFResult& result, const std::vector<std::vector<float>>& V);

    enum class StemType { Vocals, Drums, Bass, Other };

    std::vector<StemType> classifyComponents(const NMFResult& nmf);
    StemType classifyComponent(const std::vector<float>& basisVector, const std::vector<float>& activation);

    std::unique_ptr<Core::AudioBuffer> reconstructStem(
        const NMFResult& nmf,
        const std::vector<int>& componentIndices,
        int numSamples
    );

    float calculateSpectralCentroid(const std::vector<float>& spectrum);
    float calculateSpectralSpread(const std::vector<float>& spectrum, float centroid);
    float calculateTemporalVariance(const std::vector<float>& activation);
};

/**
 * @brief Frequency masking separator (fast, low quality)
 * @deprecated Use StemSeparator with SeparationQuality::Realtime
 */
class FrequencyMaskingSeparator {
public:
    struct SeparatedStems {
        std::unique_ptr<Core::AudioBuffer> vocals;
        std::unique_ptr<Core::AudioBuffer> drums;
        std::unique_ptr<Core::AudioBuffer> bass;
        std::unique_ptr<Core::AudioBuffer> other;
    };

    FrequencyMaskingSeparator();

    void setSampleRate(float sampleRate);
    SeparatedStems separate(const Core::AudioBuffer& input);

private:
    float sampleRate_;
    std::unique_ptr<SpectralProcessor> spectralProcessor_;

    void applyFrequencyMask(
        std::vector<SpectralProcessor::SpectralFrame>& frames,
        float minFreq,
        float maxFreq
    );
};

/**
 * @brief Real-time stem separator
 * @deprecated Use StemSeparator::prepareRealtime() and processBlock()
 */
class RealtimeStemSeparator {
public:
    RealtimeStemSeparator();
    ~RealtimeStemSeparator();

    void prepare(float sampleRate, int blockSize);
    void process(
        const Core::AudioBuffer& input,
        Core::AudioBuffer& vocalsOut,
        Core::AudioBuffer& drumsOut,
        Core::AudioBuffer& bassOut,
        Core::AudioBuffer& otherOut
    );

private:
    float sampleRate_;
    int blockSize_;
    std::unique_ptr<FrequencyMaskingSeparator> separator_;
    std::vector<Core::AudioBuffer> overlapBuffers_;
};

// ============================================================================
// HPSS (Harmonic-Percussive Source Separation)
// ============================================================================

/**
 * @brief HPSS-based separator using median filtering
 *
 * Separates audio into harmonic (tonal) and percussive (transient) components.
 * Used as intermediate step for full stem separation.
 */
class HPSSProcessor {
public:
    HPSSProcessor();
    ~HPSSProcessor();

    void initialize(float sampleRate, int fftSize = 4096);

    struct HPSSResult {
        std::unique_ptr<Core::AudioBuffer> harmonic;
        std::unique_ptr<Core::AudioBuffer> percussive;
        std::unique_ptr<Core::AudioBuffer> residual;
    };

    /**
     * @brief Separate into harmonic and percussive components
     * @param input Input audio
     * @param harmonicMargin Margin for harmonic mask (higher = more harmonic content)
     * @param percussiveMargin Margin for percussive mask (higher = more percussive content)
     */
    HPSSResult separate(const Core::AudioBuffer& input,
                        float harmonicMargin = 2.0f,
                        float percussiveMargin = 2.0f);

private:
    float sampleRate_;
    int fftSize_;
    int hopSize_;
    std::unique_ptr<SpectralProcessor> spectral_;

    // Median filtering
    std::vector<float> medianFilterTime(const std::vector<std::vector<float>>& spectrogram,
                                         int bin, int kernelSize);
    std::vector<float> medianFilterFreq(const std::vector<std::vector<float>>& spectrogram,
                                         int frame, int kernelSize);
};

} // namespace DSP
} // namespace MolinAntro
