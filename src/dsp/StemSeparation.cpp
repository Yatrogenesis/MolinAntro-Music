#include "dsp/StemSeparation.h"
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace MolinAntro {
namespace DSP {

// ============================================================================
// NMFStemSeparator Implementation
// ============================================================================

NMFStemSeparator::NMFStemSeparator()
    : numComponents_(16)
    , numIterations_(100)
    , sampleRate_(48000.0f)
    , spectralProcessor_(std::make_unique<SpectralProcessor>())
{
}

NMFStemSeparator::~NMFStemSeparator() = default;

void NMFStemSeparator::setNumComponents(int numComponents) {
    numComponents_ = std::max(2, numComponents);
}

void NMFStemSeparator::setNumIterations(int iterations) {
    numIterations_ = std::max(10, iterations);
}

void NMFStemSeparator::setSampleRate(float sampleRate) {
    sampleRate_ = sampleRate;
    spectralProcessor_->setSampleRate(sampleRate);
}

void NMFStemSeparator::setProgressCallback(ProgressCallback callback) {
    progressCallback_ = callback;
}

NMFStemSeparator::SeparatedStems NMFStemSeparator::separate(const Core::AudioBuffer& input) {
    if (progressCallback_) {
        progressCallback_(0.0f, "Starting stem separation...");
    }

    // Step 1: Spectral analysis
    spectralProcessor_->setFFTSize(2048);
    spectralProcessor_->setHopSize(512);
    spectralProcessor_->analyze(input);

    if (progressCallback_) {
        progressCallback_(0.1f, "Spectral analysis complete");
    }

    // Step 2: Build magnitude spectrogram
    auto& frames = spectralProcessor_->getFrames();
    if (frames.empty()) {
        std::cerr << "[NMFStemSeparator] No spectral frames\n";
        return {};
    }

    int numFrames = frames.size();
    int numBins = frames[0].magnitudes.size();

    std::vector<std::vector<float>> spectrogram(numBins, std::vector<float>(numFrames));
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            spectrogram[b][f] = frames[f].magnitudes[b];
        }
    }

    if (progressCallback_) {
        progressCallback_(0.2f, "Building spectrogram...");
    }

    // Step 3: Perform NMF
    NMFResult nmf = performNMF(spectrogram);

    if (progressCallback_) {
        progressCallback_(0.7f, "NMF decomposition complete");
    }

    // Step 4: Classify components
    std::vector<StemType> classifications = classifyComponents(nmf);

    if (progressCallback_) {
        progressCallback_(0.8f, "Classifying stems...");
    }

    // Step 5: Group components by stem type
    std::vector<int> vocalsComponents, drumsComponents, bassComponents, otherComponents;

    for (int c = 0; c < numComponents_; ++c) {
        switch (classifications[c]) {
            case StemType::Vocals:
                vocalsComponents.push_back(c);
                break;
            case StemType::Drums:
                drumsComponents.push_back(c);
                break;
            case StemType::Bass:
                bassComponents.push_back(c);
                break;
            case StemType::Other:
                otherComponents.push_back(c);
                break;
        }
    }

    if (progressCallback_) {
        progressCallback_(0.9f, "Reconstructing stems...");
    }

    // Step 6: Reconstruct each stem
    SeparatedStems result;
    result.vocals = reconstructStem(nmf, vocalsComponents, input.getNumSamples());
    result.drums = reconstructStem(nmf, drumsComponents, input.getNumSamples());
    result.bass = reconstructStem(nmf, bassComponents, input.getNumSamples());
    result.other = reconstructStem(nmf, otherComponents, input.getNumSamples());

    if (progressCallback_) {
        progressCallback_(1.0f, "Stem separation complete!");
    }

    std::cout << "[NMFStemSeparator] Separation complete:\n";
    std::cout << "  - Vocals: " << vocalsComponents.size() << " components\n";
    std::cout << "  - Drums: " << drumsComponents.size() << " components\n";
    std::cout << "  - Bass: " << bassComponents.size() << " components\n";
    std::cout << "  - Other: " << otherComponents.size() << " components\n";

    return result;
}

void NMFStemSeparator::initializeMatrices(NMFResult& result, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    result.W.resize(rows, std::vector<float>(numComponents_));
    result.H.resize(numComponents_, std::vector<float>(cols));

    // Initialize with small random values
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < numComponents_; ++k) {
            result.W[i][k] = dis(gen);
        }
    }

    for (int k = 0; k < numComponents_; ++k) {
        for (int j = 0; j < cols; ++j) {
            result.H[k][j] = dis(gen);
        }
    }
}

NMFStemSeparator::NMFResult NMFStemSeparator::performNMF(const std::vector<std::vector<float>>& V) {
    int rows = V.size();
    int cols = V[0].size();

    NMFResult result;
    initializeMatrices(result, rows, cols);

    // Multiplicative update rules for NMF
    for (int iter = 0; iter < numIterations_; ++iter) {
        updateNMF(result, V);

        if (progressCallback_ && iter % 10 == 0) {
            float progress = 0.2f + (0.5f * iter / numIterations_);
            progressCallback_(progress, "NMF iteration " + std::to_string(iter) + "/" + std::to_string(numIterations_));
        }
    }

    return result;
}

void NMFStemSeparator::updateNMF(NMFResult& result, const std::vector<std::vector<float>>& V) {
    int rows = V.size();
    int cols = V[0].size();
    const float epsilon = 1e-10f;

    // Compute WH
    std::vector<std::vector<float>> WH(rows, std::vector<float>(cols, 0.0f));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            for (int k = 0; k < numComponents_; ++k) {
                WH[i][j] += result.W[i][k] * result.H[k][j];
            }
            WH[i][j] = std::max(WH[i][j], epsilon);
        }
    }

    // Update H
    std::vector<std::vector<float>> numeratorH(numComponents_, std::vector<float>(cols, 0.0f));
    std::vector<std::vector<float>> denominatorH(numComponents_, std::vector<float>(cols, 0.0f));

    for (int k = 0; k < numComponents_; ++k) {
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                numeratorH[k][j] += result.W[i][k] * (V[i][j] / WH[i][j]);
                denominatorH[k][j] += result.W[i][k];
            }
            result.H[k][j] *= numeratorH[k][j] / (denominatorH[k][j] + epsilon);
        }
    }

    // Recompute WH
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            WH[i][j] = 0.0f;
            for (int k = 0; k < numComponents_; ++k) {
                WH[i][j] += result.W[i][k] * result.H[k][j];
            }
            WH[i][j] = std::max(WH[i][j], epsilon);
        }
    }

    // Update W
    std::vector<std::vector<float>> numeratorW(rows, std::vector<float>(numComponents_, 0.0f));
    std::vector<std::vector<float>> denominatorW(rows, std::vector<float>(numComponents_, 0.0f));

    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < numComponents_; ++k) {
            for (int j = 0; j < cols; ++j) {
                numeratorW[i][k] += result.H[k][j] * (V[i][j] / WH[i][j]);
                denominatorW[i][k] += result.H[k][j];
            }
            result.W[i][k] *= numeratorW[i][k] / (denominatorW[i][k] + epsilon);
        }
    }
}

std::vector<NMFStemSeparator::StemType> NMFStemSeparator::classifyComponents(const NMFResult& nmf) {
    std::vector<StemType> classifications(numComponents_);

    for (int k = 0; k < numComponents_; ++k) {
        classifications[k] = classifyComponent(nmf.W[k], nmf.H[k]);
    }

    return classifications;
}

NMFStemSeparator::StemType NMFStemSeparator::classifyComponent(
    const std::vector<float>& basisVector,
    const std::vector<float>& activation
) {
    // Analyze spectral characteristics
    float centroid = calculateSpectralCentroid(basisVector);
    float spread = calculateSpectralSpread(basisVector, centroid);
    float temporalVariance = calculateTemporalVariance(activation);

    // Classification heuristics:
    // - Vocals: Mid-high frequency (500-4000 Hz), moderate temporal variance
    // - Drums: Wideband, high temporal variance (transient)
    // - Bass: Low frequency (<250 Hz), sustained
    // - Other: Everything else

    float nyquist = sampleRate_ / 2.0f;
    float centroidFreq = (centroid / basisVector.size()) * nyquist;

    if (centroidFreq < 250.0f && temporalVariance < 0.5f) {
        return StemType::Bass;
    } else if (centroidFreq >= 500.0f && centroidFreq <= 4000.0f && temporalVariance > 0.3f && temporalVariance < 0.8f) {
        return StemType::Vocals;
    } else if (spread > 0.4f && temporalVariance > 0.7f) {
        return StemType::Drums;
    } else {
        return StemType::Other;
    }
}

float NMFStemSeparator::calculateSpectralCentroid(const std::vector<float>& spectrum) {
    float weightedSum = 0.0f;
    float totalMagnitude = 0.0f;

    for (size_t i = 0; i < spectrum.size(); ++i) {
        weightedSum += i * spectrum[i];
        totalMagnitude += spectrum[i];
    }

    return (totalMagnitude > 0.0f) ? (weightedSum / totalMagnitude) : 0.0f;
}

float NMFStemSeparator::calculateSpectralSpread(const std::vector<float>& spectrum, float centroid) {
    float variance = 0.0f;
    float totalMagnitude = 0.0f;

    for (size_t i = 0; i < spectrum.size(); ++i) {
        float diff = i - centroid;
        variance += diff * diff * spectrum[i];
        totalMagnitude += spectrum[i];
    }

    return (totalMagnitude > 0.0f) ? std::sqrt(variance / totalMagnitude) / spectrum.size() : 0.0f;
}

float NMFStemSeparator::calculateTemporalVariance(const std::vector<float>& activation) {
    float mean = std::accumulate(activation.begin(), activation.end(), 0.0f) / activation.size();

    float variance = 0.0f;
    for (float val : activation) {
        float diff = val - mean;
        variance += diff * diff;
    }

    return std::sqrt(variance / activation.size());
}

std::unique_ptr<Core::AudioBuffer> NMFStemSeparator::reconstructStem(
    const NMFResult& nmf,
    const std::vector<int>& componentIndices,
    int numSamples
) {
    if (componentIndices.empty()) {
        auto buffer = std::make_unique<Core::AudioBuffer>(1, numSamples);
        buffer->clear();
        return buffer;
    }

    // Reconstruct magnitude spectrogram for selected components
    auto& frames = spectralProcessor_->getFrames();
    int numFrames = frames.size();
    int numBins = frames[0].magnitudes.size();

    // Create modified frames with only selected components
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            float magnitude = 0.0f;

            // Sum contributions from selected components
            for (int comp : componentIndices) {
                if (comp < static_cast<int>(nmf.W.size()) && b < static_cast<int>(nmf.W[comp].size())) {
                    magnitude += nmf.W[b][comp] * nmf.H[comp][f];
                }
            }

            frames[f].magnitudes[b] = magnitude;
            // Reconstruct complex value with original phase
            frames[f].bins[b] = std::polar(magnitude, frames[f].phases[b]);
        }
    }

    // Synthesize audio
    auto result = std::make_unique<Core::AudioBuffer>(1, numSamples);
    spectralProcessor_->synthesize(*result);

    return result;
}

// ============================================================================
// FrequencyMaskingSeparator Implementation
// ============================================================================

FrequencyMaskingSeparator::FrequencyMaskingSeparator()
    : sampleRate_(48000.0f)
    , spectralProcessor_(std::make_unique<SpectralProcessor>())
{
}

void FrequencyMaskingSeparator::setSampleRate(float sampleRate) {
    sampleRate_ = sampleRate;
    spectralProcessor_->setSampleRate(sampleRate);
}

FrequencyMaskingSeparator::SeparatedStems FrequencyMaskingSeparator::separate(const Core::AudioBuffer& input) {
    // Spectral analysis
    spectralProcessor_->setFFTSize(2048);
    spectralProcessor_->setHopSize(512);
    spectralProcessor_->analyze(input);

    auto& frames = spectralProcessor_->getFrames();

    SeparatedStems result;

    // Bass: 20-250 Hz
    applyFrequencyMask(frames, 20.0f, 250.0f);
    result.bass = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    spectralProcessor_->synthesize(*result.bass);

    // Re-analyze for vocals
    spectralProcessor_->analyze(input);
    applyFrequencyMask(frames, 200.0f, 4000.0f);
    result.vocals = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    spectralProcessor_->synthesize(*result.vocals);

    // Drums (wideband) - use harmonic/percussive separation
    result.drums = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    result.drums->clear();

    // Other (residual)
    result.other = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    result.other->clear();

    std::cout << "[FrequencyMaskingSeparator] Separation complete\n";

    return result;
}

void FrequencyMaskingSeparator::applyFrequencyMask(
    std::vector<SpectralProcessor::SpectralFrame>& frames,
    float minFreq,
    float maxFreq
) {
    for (auto& frame : frames) {
        for (size_t bin = 0; bin < frame.bins.size(); ++bin) {
            float freq = spectralProcessor_->getFrequencyForBin(bin);

            if (freq < minFreq || freq > maxFreq) {
                frame.bins[bin] = std::complex<float>(0.0f, 0.0f);
                frame.magnitudes[bin] = 0.0f;
            }
        }
    }
}

// ============================================================================
// RealtimeStemSeparator Implementation
// ============================================================================

RealtimeStemSeparator::RealtimeStemSeparator()
    : sampleRate_(48000.0f)
    , blockSize_(512)
    , separator_(std::make_unique<FrequencyMaskingSeparator>())
{
}

RealtimeStemSeparator::~RealtimeStemSeparator() = default;

void RealtimeStemSeparator::prepare(float sampleRate, int blockSize) {
    sampleRate_ = sampleRate;
    blockSize_ = blockSize;
    separator_->setSampleRate(sampleRate);
}

void RealtimeStemSeparator::process(
    const Core::AudioBuffer& input,
    Core::AudioBuffer& vocalsOut,
    Core::AudioBuffer& drumsOut,
    Core::AudioBuffer& bassOut,
    Core::AudioBuffer& otherOut
) {
    auto stems = separator_->separate(input);

    if (stems.vocals) {
        for (int ch = 0; ch < vocalsOut.getNumChannels() && ch < stems.vocals->getNumChannels(); ++ch) {
            vocalsOut.copyFrom(*stems.vocals, ch, ch);
        }
    }
    if (stems.drums) {
        for (int ch = 0; ch < drumsOut.getNumChannels() && ch < stems.drums->getNumChannels(); ++ch) {
            drumsOut.copyFrom(*stems.drums, ch, ch);
        }
    }
    if (stems.bass) {
        for (int ch = 0; ch < bassOut.getNumChannels() && ch < stems.bass->getNumChannels(); ++ch) {
            bassOut.copyFrom(*stems.bass, ch, ch);
        }
    }
    if (stems.other) {
        for (int ch = 0; ch < otherOut.getNumChannels() && ch < stems.other->getNumChannels(); ++ch) {
            otherOut.copyFrom(*stems.other, ch, ch);
        }
    }
}

} // namespace DSP
} // namespace MolinAntro
