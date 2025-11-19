#pragma once

#include "core/AudioBuffer.h"
#include <vector>
#include <complex>
#include <cmath>
#include <memory>

namespace MolinAntro {
namespace DSP {

/**
 * Professional Spectral Processing Engine
 * Features: FFT/IFFT, spectral editing, noise reduction, harmonic analysis
 * SOTA-level implementation matching iZotope RX, Adobe Audition spectral editing
 */
class SpectralProcessor {
public:
    enum class WindowType {
        Hann,
        Hamming,
        Blackman,
        BlackmanHarris,
        Rectangular
    };

    struct SpectralFrame {
        std::vector<std::complex<float>> bins;
        std::vector<float> magnitudes;
        std::vector<float> phases;
        int frameIndex;
        double timePosition;
    };

    SpectralProcessor();
    ~SpectralProcessor();

    // Configuration
    void setFFTSize(int size);
    void setHopSize(int size);
    void setWindowType(WindowType type);
    void setSampleRate(float sampleRate);

    // Analysis
    void analyze(const Core::AudioBuffer& input);
    std::vector<SpectralFrame>& getFrames() { return frames_; }
    const std::vector<SpectralFrame>& getFrames() const { return frames_; }

    // Spectral editing operations
    void applyGain(int frameIndex, float minFreq, float maxFreq, float gainDB);
    void suppressNoise(float thresholdDB);
    void harmonicEnhancement(float fundamentalFreq, int numHarmonics, float gainDB);
    void spectralGate(float thresholdDB);
    void denoiseSpectralSubtraction(const Core::AudioBuffer& noiseProfile);

    // Forensic analysis
    struct ForensicReport {
        std::vector<float> spectralCentroid;
        std::vector<float> spectralRolloff;
        std::vector<float> spectralFlux;
        float averagePitch;
        bool editingDetected;
        std::vector<int> anomalyFrames;
    };
    ForensicReport performForensicAnalysis();

    // Synthesis
    void synthesize(Core::AudioBuffer& output);

    // Utilities
    int getFFTSize() const { return fftSize_; }
    int getHopSize() const { return hopSize_; }
    float getFrequencyForBin(int bin) const;
    int getBinForFrequency(float freq) const;

private:
    // FFT implementation (Cooley-Tukey radix-2)
    void fft(std::vector<std::complex<float>>& data, bool inverse = false);
    void fftRecursive(std::complex<float>* data, int n, bool inverse);

    // Window functions
    void generateWindow();
    float windowFunction(int n, int N, WindowType type);

    // Internal processing
    void computeMagnitudesAndPhases(SpectralFrame& frame);
    void applyWindow(std::vector<float>& frame);

    int fftSize_;
    int hopSize_;
    float sampleRate_;
    WindowType windowType_;

    std::vector<float> window_;
    std::vector<SpectralFrame> frames_;

    // Overlap-add synthesis
    std::vector<float> outputBuffer_;
    int outputPosition_;
};

/**
 * Spectral Noise Reduction (SOTA-level like iZotope RX)
 * Multi-band spectral subtraction with psychoacoustic modeling
 */
class SpectralNoiseReduction {
public:
    SpectralNoiseReduction();

    void learnNoiseProfile(const Core::AudioBuffer& noiseSection);
    void process(Core::AudioBuffer& buffer, float reductionAmount = 1.0f);

    void setThreshold(float thresholdDB);
    void setReduction(float reductionDB);
    void setSmoothingFrames(int frames);

private:
    std::unique_ptr<SpectralProcessor> processor_;
    std::vector<float> noiseProfile_;
    float thresholdDB_;
    float reductionDB_;
    int smoothingFrames_;
    bool profileLearned_;
};

/**
 * Harmonic/Percussive Source Separation
 * Separates audio into harmonic and percussive components
 * Based on median filtering in time/frequency domain
 */
class HarmonicPercussiveSeparation {
public:
    HarmonicPercussiveSeparation();

    struct SeparationResult {
        std::unique_ptr<Core::AudioBuffer> harmonic;
        std::unique_ptr<Core::AudioBuffer> percussive;
        std::unique_ptr<Core::AudioBuffer> residual;
    };

    SeparationResult separate(const Core::AudioBuffer& input);

    void setHarmonicFilterLength(int length);
    void setPercussiveFilterLength(int length);

private:
    std::unique_ptr<SpectralProcessor> processor_;
    int harmonicFilterLength_;
    int percussiveFilterLength_;

    void medianFilter(std::vector<std::vector<float>>& spectrogram, int filterLength, bool horizontal);
};

} // namespace DSP
} // namespace MolinAntro
