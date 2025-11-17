#pragma once

#include "core/AudioBuffer.h"
#include "dsp/SpectralProcessor.h"
#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace MolinAntro {
namespace DSP {

/**
 * AI-based Stem Separation using NMF (Non-negative Matrix Factorization)
 * Separates audio into: vocals, drums, bass, and other
 *
 * This is a mathematical approach that doesn't require neural networks,
 * making it fast and suitable for real-time processing.
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

    // Configuration
    void setNumComponents(int numComponents); // NMF rank (default: 16)
    void setNumIterations(int iterations);    // NMF iterations (default: 100)
    void setSampleRate(float sampleRate);

    // Separation
    SeparatedStems separate(const Core::AudioBuffer& input);

    // Progress callback
    using ProgressCallback = std::function<void(float progress, const std::string& message)>;
    void setProgressCallback(ProgressCallback callback);

private:
    int numComponents_;
    int numIterations_;
    float sampleRate_;
    ProgressCallback progressCallback_;

    std::unique_ptr<SpectralProcessor> spectralProcessor_;

    // NMF core algorithm
    struct NMFResult {
        std::vector<std::vector<float>> W; // Basis matrix (frequency x components)
        std::vector<std::vector<float>> H; // Activation matrix (components x time)
    };

    NMFResult performNMF(const std::vector<std::vector<float>>& V);
    void initializeMatrices(NMFResult& result, int rows, int cols);
    void updateNMF(NMFResult& result, const std::vector<std::vector<float>>& V);

    // Stem classification based on spectral characteristics
    enum class StemType {
        Vocals,
        Drums,
        Bass,
        Other
    };

    std::vector<StemType> classifyComponents(const NMFResult& nmf);
    StemType classifyComponent(const std::vector<float>& basisVector, const std::vector<float>& activation);

    // Reconstruction
    std::unique_ptr<Core::AudioBuffer> reconstructStem(
        const NMFResult& nmf,
        const std::vector<int>& componentIndices,
        int numSamples
    );

    // Helper functions
    float calculateSpectralCentroid(const std::vector<float>& spectrum);
    float calculateSpectralSpread(const std::vector<float>& spectrum, float centroid);
    float calculateTemporalVariance(const std::vector<float>& activation);
};

/**
 * Simplified Source Separation using Frequency Masking
 * Faster but less accurate than NMF
 */
class FrequencyMaskingSeparator {
public:
    struct SeparatedStems {
        std::unique_ptr<Core::AudioBuffer> vocals;     // 200-4000 Hz (with harmonics)
        std::unique_ptr<Core::AudioBuffer> drums;      // Wideband (transient-focused)
        std::unique_ptr<Core::AudioBuffer> bass;       // 20-250 Hz
        std::unique_ptr<Core::AudioBuffer> other;      // Residual
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
 * Real-time Stem Separator (optimized for low latency)
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

    // Buffering for smooth output
    std::vector<Core::AudioBuffer> overlapBuffers_;
};

} // namespace DSP
} // namespace MolinAntro
