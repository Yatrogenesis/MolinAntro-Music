#include "dsp/SpectralProcessor.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MolinAntro {
namespace DSP {

// ============================================================================
// SpectralProcessor Implementation
// ============================================================================

SpectralProcessor::SpectralProcessor()
    : fftSize_(2048)
    , hopSize_(512)
    , sampleRate_(48000.0f)
    , windowType_(WindowType::Hann)
    , outputPosition_(0)
{
    generateWindow();
}

SpectralProcessor::~SpectralProcessor() = default;

void SpectralProcessor::setFFTSize(int size) {
    // Ensure power of 2
    int powerOf2 = 1;
    while (powerOf2 < size) {
        powerOf2 *= 2;
    }
    fftSize_ = powerOf2;
    generateWindow();
    frames_.clear();
}

void SpectralProcessor::setHopSize(int size) {
    hopSize_ = size;
}

void SpectralProcessor::setWindowType(WindowType type) {
    windowType_ = type;
    generateWindow();
}

void SpectralProcessor::setSampleRate(float sampleRate) {
    sampleRate_ = sampleRate;
}

float SpectralProcessor::windowFunction(int n, int N, WindowType type) {
    const float ratio = static_cast<float>(n) / static_cast<float>(N - 1);

    switch (type) {
        case WindowType::Hann:
            return 0.5f * (1.0f - std::cos(2.0f * M_PI * ratio));

        case WindowType::Hamming:
            return 0.54f - 0.46f * std::cos(2.0f * M_PI * ratio);

        case WindowType::Blackman:
            return 0.42f - 0.5f * std::cos(2.0f * M_PI * ratio)
                   + 0.08f * std::cos(4.0f * M_PI * ratio);

        case WindowType::BlackmanHarris:
            return 0.35875f
                   - 0.48829f * std::cos(2.0f * M_PI * ratio)
                   + 0.14128f * std::cos(4.0f * M_PI * ratio)
                   - 0.01168f * std::cos(6.0f * M_PI * ratio);

        case WindowType::Rectangular:
        default:
            return 1.0f;
    }
}

void SpectralProcessor::generateWindow() {
    window_.resize(fftSize_);
    for (int i = 0; i < fftSize_; ++i) {
        window_[i] = windowFunction(i, fftSize_, windowType_);
    }
}

void SpectralProcessor::applyWindow(std::vector<float>& frame) {
    for (size_t i = 0; i < frame.size() && i < window_.size(); ++i) {
        frame[i] *= window_[i];
    }
}

void SpectralProcessor::fft(std::vector<std::complex<float>>& data, bool inverse) {
    int n = data.size();

    // Bit-reversal permutation
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) {
            std::swap(data[i], data[j]);
        }

        int k = n / 2;
        while (k <= j) {
            j -= k;
            k /= 2;
        }
        j += k;
    }

    // Cooley-Tukey FFT
    for (int len = 2; len <= n; len *= 2) {
        float angle = 2.0f * M_PI / len * (inverse ? 1 : -1);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));

        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<float> u = data[i + j];
                std::complex<float> v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        float scale = 1.0f / n;
        for (auto& sample : data) {
            sample *= scale;
        }
    }
}

void SpectralProcessor::computeMagnitudesAndPhases(SpectralFrame& frame) {
    int numBins = frame.bins.size();
    frame.magnitudes.resize(numBins);
    frame.phases.resize(numBins);

    for (int i = 0; i < numBins; ++i) {
        frame.magnitudes[i] = std::abs(frame.bins[i]);
        frame.phases[i] = std::arg(frame.bins[i]);
    }
}

void SpectralProcessor::analyze(const Core::AudioBuffer& input) {
    frames_.clear();

    if (input.getNumChannels() == 0 || input.getNumSamples() == 0) {
        return;
    }

    // Process first channel (mono or left)
    const float* channelData = input.getReadPointer(0);
    int numSamples = input.getNumSamples();

    // Sliding window analysis
    int frameIndex = 0;
    for (int pos = 0; pos + fftSize_ <= numSamples; pos += hopSize_) {
        SpectralFrame frame;
        frame.frameIndex = frameIndex++;
        frame.timePosition = static_cast<double>(pos) / sampleRate_;

        // Copy and window the frame
        std::vector<float> frameData(fftSize_);
        std::copy(channelData + pos, channelData + pos + fftSize_, frameData.begin());
        applyWindow(frameData);

        // Convert to complex
        frame.bins.resize(fftSize_);
        for (int i = 0; i < fftSize_; ++i) {
            frame.bins[i] = std::complex<float>(frameData[i], 0.0f);
        }

        // Perform FFT
        fft(frame.bins, false);

        // Compute magnitude and phase
        computeMagnitudesAndPhases(frame);

        frames_.push_back(std::move(frame));
    }

    std::cout << "[SpectralProcessor] Analyzed " << frames_.size()
              << " frames (FFT size: " << fftSize_ << ", hop: " << hopSize_ << ")\n";
}

float SpectralProcessor::getFrequencyForBin(int bin) const {
    return (bin * sampleRate_) / fftSize_;
}

int SpectralProcessor::getBinForFrequency(float freq) const {
    return static_cast<int>((freq * fftSize_) / sampleRate_ + 0.5f);
}

void SpectralProcessor::applyGain(int frameIndex, float minFreq, float maxFreq, float gainDB) {
    if (frameIndex < 0 || frameIndex >= static_cast<int>(frames_.size())) {
        return;
    }

    int minBin = getBinForFrequency(minFreq);
    int maxBin = getBinForFrequency(maxFreq);
    float gainLinear = std::pow(10.0f, gainDB / 20.0f);

    SpectralFrame& frame = frames_[frameIndex];
    for (int bin = minBin; bin <= maxBin && bin < static_cast<int>(frame.bins.size()); ++bin) {
        frame.bins[bin] *= gainLinear;
        frame.magnitudes[bin] *= gainLinear;
    }
}

void SpectralProcessor::suppressNoise(float thresholdDB) {
    float thresholdLinear = std::pow(10.0f, thresholdDB / 20.0f);

    for (auto& frame : frames_) {
        for (size_t bin = 0; bin < frame.bins.size(); ++bin) {
            if (frame.magnitudes[bin] < thresholdLinear) {
                frame.bins[bin] = std::complex<float>(0.0f, 0.0f);
                frame.magnitudes[bin] = 0.0f;
            }
        }
    }
}

void SpectralProcessor::harmonicEnhancement(float fundamentalFreq, int numHarmonics, float gainDB) {
    float gainLinear = std::pow(10.0f, gainDB / 20.0f);

    for (auto& frame : frames_) {
        for (int h = 1; h <= numHarmonics; ++h) {
            float harmonicFreq = fundamentalFreq * h;
            int bin = getBinForFrequency(harmonicFreq);

            if (bin < static_cast<int>(frame.bins.size())) {
                frame.bins[bin] *= gainLinear;
                frame.magnitudes[bin] *= gainLinear;
            }
        }
    }
}

void SpectralProcessor::spectralGate(float thresholdDB) {
    suppressNoise(thresholdDB);
}

void SpectralProcessor::denoiseSpectralSubtraction(const Core::AudioBuffer& noiseProfile) {
    // Analyze noise profile
    SpectralProcessor noiseAnalyzer;
    noiseAnalyzer.setFFTSize(fftSize_);
    noiseAnalyzer.setHopSize(hopSize_);
    noiseAnalyzer.setSampleRate(sampleRate_);
    noiseAnalyzer.analyze(noiseProfile);

    if (noiseAnalyzer.getFrames().empty()) {
        return;
    }

    // Compute average noise spectrum
    std::vector<float> avgNoiseSpectrum(fftSize_ / 2 + 1, 0.0f);
    for (const auto& noiseFrame : noiseAnalyzer.getFrames()) {
        for (size_t bin = 0; bin < avgNoiseSpectrum.size() && bin < noiseFrame.magnitudes.size(); ++bin) {
            avgNoiseSpectrum[bin] += noiseFrame.magnitudes[bin];
        }
    }

    float scale = 1.0f / noiseAnalyzer.getFrames().size();
    for (auto& val : avgNoiseSpectrum) {
        val *= scale;
    }

    // Apply spectral subtraction
    const float overSubtraction = 2.0f;
    for (auto& frame : frames_) {
        for (size_t bin = 0; bin < frame.magnitudes.size() && bin < avgNoiseSpectrum.size(); ++bin) {
            float cleanMagnitude = frame.magnitudes[bin] - overSubtraction * avgNoiseSpectrum[bin];
            cleanMagnitude = std::max(0.0f, cleanMagnitude);

            // Reconstruct complex number with original phase
            frame.magnitudes[bin] = cleanMagnitude;
            frame.bins[bin] = std::polar(cleanMagnitude, frame.phases[bin]);
        }
    }

    std::cout << "[SpectralProcessor] Denoising applied using spectral subtraction\n";
}

SpectralProcessor::ForensicReport SpectralProcessor::performForensicAnalysis() {
    ForensicReport report;

    report.spectralCentroid.reserve(frames_.size());
    report.spectralRolloff.reserve(frames_.size());
    report.spectralFlux.reserve(frames_.size());

    std::vector<float> prevMagnitudes;

    for (size_t f = 0; f < frames_.size(); ++f) {
        const auto& frame = frames_[f];

        // Spectral Centroid
        float weightedSum = 0.0f;
        float magnitudeSum = 0.0f;
        for (size_t bin = 0; bin < frame.magnitudes.size(); ++bin) {
            float freq = getFrequencyForBin(bin);
            weightedSum += freq * frame.magnitudes[bin];
            magnitudeSum += frame.magnitudes[bin];
        }
        float centroid = (magnitudeSum > 0.0f) ? (weightedSum / magnitudeSum) : 0.0f;
        report.spectralCentroid.push_back(centroid);

        // Spectral Rolloff (95% of energy)
        float targetEnergy = magnitudeSum * 0.95f;
        float cumulativeEnergy = 0.0f;
        float rolloff = 0.0f;
        for (size_t bin = 0; bin < frame.magnitudes.size(); ++bin) {
            cumulativeEnergy += frame.magnitudes[bin];
            if (cumulativeEnergy >= targetEnergy) {
                rolloff = getFrequencyForBin(bin);
                break;
            }
        }
        report.spectralRolloff.push_back(rolloff);

        // Spectral Flux (change from previous frame)
        if (!prevMagnitudes.empty()) {
            float flux = 0.0f;
            for (size_t bin = 0; bin < frame.magnitudes.size() && bin < prevMagnitudes.size(); ++bin) {
                float diff = frame.magnitudes[bin] - prevMagnitudes[bin];
                flux += diff * diff;
            }
            report.spectralFlux.push_back(std::sqrt(flux));
        } else {
            report.spectralFlux.push_back(0.0f);
        }

        prevMagnitudes = frame.magnitudes;
    }

    // Average pitch estimation (simplified)
    report.averagePitch = 0.0f;
    if (!report.spectralCentroid.empty()) {
        report.averagePitch = std::accumulate(report.spectralCentroid.begin(),
                                               report.spectralCentroid.end(), 0.0f) / report.spectralCentroid.size();
    }

    // Editing detection (look for anomalies in spectral flux)
    report.editingDetected = false;
    if (!report.spectralFlux.empty()) {
        float meanFlux = std::accumulate(report.spectralFlux.begin(),
                                         report.spectralFlux.end(), 0.0f) / report.spectralFlux.size();

        float variance = 0.0f;
        for (float flux : report.spectralFlux) {
            float diff = flux - meanFlux;
            variance += diff * diff;
        }
        variance /= report.spectralFlux.size();
        float stddev = std::sqrt(variance);

        // Flag frames with flux > mean + 3*stddev as potential edits
        for (size_t i = 0; i < report.spectralFlux.size(); ++i) {
            if (report.spectralFlux[i] > meanFlux + 3.0f * stddev) {
                report.editingDetected = true;
                report.anomalyFrames.push_back(static_cast<int>(i));
            }
        }
    }

    std::cout << "[SpectralProcessor] Forensic analysis complete:\n";
    std::cout << "  - Average pitch: " << report.averagePitch << " Hz\n";
    std::cout << "  - Editing detected: " << (report.editingDetected ? "YES" : "NO") << "\n";
    std::cout << "  - Anomaly frames: " << report.anomalyFrames.size() << "\n";

    return report;
}

void SpectralProcessor::synthesize(Core::AudioBuffer& output) {
    if (frames_.empty()) {
        std::cerr << "[SpectralProcessor] No frames to synthesize\n";
        return;
    }

    int totalSamples = (frames_.size() - 1) * hopSize_ + fftSize_;
    output = Core::AudioBuffer(1, totalSamples);
    output.clear();

    float* outputData = output.getWritePointer(0);
    std::vector<float> overlapAdd(totalSamples, 0.0f);
    std::vector<float> windowSum(totalSamples, 0.0f);

    // Overlap-add synthesis
    for (size_t f = 0; f < frames_.size(); ++f) {
        auto& frame = frames_[f];

        // IFFT
        std::vector<std::complex<float>> complexFrame = frame.bins;
        fft(complexFrame, true);

        // Extract real part and apply window
        std::vector<float> frameData(fftSize_);
        for (int i = 0; i < fftSize_; ++i) {
            frameData[i] = complexFrame[i].real() * window_[i];
        }

        // Overlap-add
        int outputPos = f * hopSize_;
        for (int i = 0; i < fftSize_ && outputPos + i < totalSamples; ++i) {
            overlapAdd[outputPos + i] += frameData[i];
            windowSum[outputPos + i] += window_[i] * window_[i];
        }
    }

    // Normalize by window sum
    for (int i = 0; i < totalSamples; ++i) {
        outputData[i] = (windowSum[i] > 0.001f) ? (overlapAdd[i] / windowSum[i]) : 0.0f;
    }

    std::cout << "[SpectralProcessor] Synthesized " << totalSamples << " samples from " << frames_.size() << " frames\n";
}

// ============================================================================
// SpectralNoiseReduction Implementation
// ============================================================================

SpectralNoiseReduction::SpectralNoiseReduction()
    : processor_(std::make_unique<SpectralProcessor>())
    , thresholdDB_(-40.0f)
    , reductionDB_(-20.0f)
    , smoothingFrames_(3)
    , profileLearned_(false)
{
}

void SpectralNoiseReduction::learnNoiseProfile(const Core::AudioBuffer& noiseSection) {
    processor_->analyze(noiseSection);
    const auto& frames = processor_->getFrames();

    if (frames.empty()) {
        std::cerr << "[SpectralNoiseReduction] Failed to learn noise profile\n";
        return;
    }

    // Average all frames to get noise profile
    int numBins = frames[0].magnitudes.size();
    noiseProfile_.assign(numBins, 0.0f);

    for (const auto& frame : frames) {
        for (size_t bin = 0; bin < noiseProfile_.size() && bin < frame.magnitudes.size(); ++bin) {
            noiseProfile_[bin] += frame.magnitudes[bin];
        }
    }

    float scale = 1.0f / frames.size();
    for (auto& val : noiseProfile_) {
        val *= scale;
    }

    profileLearned_ = true;
    std::cout << "[SpectralNoiseReduction] Noise profile learned (" << numBins << " bins)\n";
}

void SpectralNoiseReduction::process(Core::AudioBuffer& buffer, float reductionAmount) {
    if (!profileLearned_) {
        std::cerr << "[SpectralNoiseReduction] No noise profile learned\n";
        return;
    }

    // Create temporary buffer for processing
    Core::AudioBuffer tempBuffer(buffer.getNumChannels(), buffer.getNumSamples());
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        tempBuffer.copyFrom(buffer, ch, ch);
    }

    processor_->analyze(tempBuffer);

    // Apply noise reduction using the learned profile
    auto& frames = processor_->getFrames();
    for (auto& frame : frames) {
        for (size_t bin = 0; bin < frame.magnitudes.size() && bin < noiseProfile_.size(); ++bin) {
            float noiseMag = noiseProfile_[bin] * reductionAmount;
            float cleanMag = std::max(0.0f, frame.magnitudes[bin] - noiseMag);

            frame.magnitudes[bin] = cleanMag;
            frame.bins[bin] = std::polar(cleanMag, frame.phases[bin]);
        }
    }

    processor_->synthesize(buffer);

    std::cout << "[SpectralNoiseReduction] Noise reduction applied (amount: " << reductionAmount << ")\n";
}

void SpectralNoiseReduction::setThreshold(float thresholdDB) {
    thresholdDB_ = thresholdDB;
}

void SpectralNoiseReduction::setReduction(float reductionDB) {
    reductionDB_ = reductionDB;
}

void SpectralNoiseReduction::setSmoothingFrames(int frames) {
    smoothingFrames_ = frames;
}

// ============================================================================
// HarmonicPercussiveSeparation Implementation
// ============================================================================

HarmonicPercussiveSeparation::HarmonicPercussiveSeparation()
    : processor_(std::make_unique<SpectralProcessor>())
    , harmonicFilterLength_(17)
    , percussiveFilterLength_(17)
{
}

void HarmonicPercussiveSeparation::medianFilter(std::vector<std::vector<float>>& spectrogram,
                                                 int filterLength, bool horizontal) {
    int rows = spectrogram.size();
    int cols = spectrogram[0].size();

    std::vector<std::vector<float>> filtered = spectrogram;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            std::vector<float> values;

            if (horizontal) {
                int start = std::max(0, c - filterLength / 2);
                int end = std::min(cols - 1, c + filterLength / 2);
                for (int i = start; i <= end; ++i) {
                    values.push_back(spectrogram[r][i]);
                }
            } else {
                int start = std::max(0, r - filterLength / 2);
                int end = std::min(rows - 1, r + filterLength / 2);
                for (int i = start; i <= end; ++i) {
                    values.push_back(spectrogram[i][c]);
                }
            }

            std::nth_element(values.begin(), values.begin() + values.size() / 2, values.end());
            filtered[r][c] = values[values.size() / 2];
        }
    }

    spectrogram = filtered;
}

HarmonicPercussiveSeparation::SeparationResult HarmonicPercussiveSeparation::separate(const Core::AudioBuffer& input) {
    processor_->analyze(input);
    auto& frames = processor_->getFrames();

    if (frames.empty()) {
        std::cerr << "[HarmonicPercussiveSeparation] No frames to process\n";
        return {};
    }

    // Build spectrogram
    int numFrames = frames.size();
    int numBins = frames[0].magnitudes.size();
    std::vector<std::vector<float>> spectrogram(numFrames, std::vector<float>(numBins));

    for (int f = 0; f < numFrames; ++f) {
        spectrogram[f] = frames[f].magnitudes;
    }

    // Apply median filtering
    auto harmonicSpec = spectrogram;
    auto percussiveSpec = spectrogram;

    medianFilter(harmonicSpec, harmonicFilterLength_, true);  // Horizontal for harmonic
    medianFilter(percussiveSpec, percussiveFilterLength_, false); // Vertical for percussive

    // Create masks and separate
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            float total = harmonicSpec[f][b] + percussiveSpec[f][b];
            if (total > 0.0001f) {
                float harmonicMask = harmonicSpec[f][b] / total;
                float percussiveMask = percussiveSpec[f][b] / total;

                harmonicSpec[f][b] = frames[f].magnitudes[b] * harmonicMask;
                percussiveSpec[f][b] = frames[f].magnitudes[b] * percussiveMask;
            }
        }
    }

    // Reconstruct buffers
    SeparationResult result;

    // Harmonic component
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            frames[f].bins[b] = std::polar(harmonicSpec[f][b], frames[f].phases[b]);
        }
    }
    result.harmonic = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    processor_->synthesize(*result.harmonic);

    // Percussive component
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            frames[f].bins[b] = std::polar(percussiveSpec[f][b], frames[f].phases[b]);
        }
    }
    result.percussive = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    processor_->synthesize(*result.percussive);

    // Residual (if needed)
    result.residual = std::make_unique<Core::AudioBuffer>(1, input.getNumSamples());
    result.residual->clear();

    std::cout << "[HarmonicPercussiveSeparation] Separation complete\n";
    return result;
}

void HarmonicPercussiveSeparation::setHarmonicFilterLength(int length) {
    harmonicFilterLength_ = length | 1; // Ensure odd
}

void HarmonicPercussiveSeparation::setPercussiveFilterLength(int length) {
    percussiveFilterLength_ = length | 1; // Ensure odd
}

} // namespace DSP
} // namespace MolinAntro
