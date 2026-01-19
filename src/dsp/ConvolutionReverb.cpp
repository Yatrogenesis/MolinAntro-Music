#include "dsp/ConvolutionReverb.h"
#include "dsp/AudioFile.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <random>
#include <numeric>

namespace MolinAntro {
namespace DSP {

// ============================================================================
// Constructor / Destructor
// ============================================================================

ConvolutionReverb::ConvolutionReverb() {
    irInfo_.sampleRate = 0;
    irInfo_.numChannels = 0;
    irInfo_.numSamples = 0;
    irInfo_.lengthSeconds = 0.0f;
    irInfo_.isTrueStereo = false;
    irInfo_.name = "None";
}

ConvolutionReverb::~ConvolutionReverb() = default;

// ============================================================================
// AudioEffect Interface
// ============================================================================

void ConvolutionReverb::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    // Allocate temp buffers
    tempL_.resize(maxBlockSize);
    tempR_.resize(maxBlockSize);
    wetL_.resize(maxBlockSize);
    wetR_.resize(maxBlockSize);

    // Pre-delay buffer (max 500ms)
    int maxPreDelaySamples = static_cast<int>(0.5f * sampleRate);
    preDelayBufferL_.resize(maxPreDelaySamples, 0.0f);
    preDelayBufferR_.resize(maxPreDelaySamples, 0.0f);
    preDelayWritePos_ = 0;

    // Update pre-delay samples
    preDelaySamples_ = static_cast<int>(preDelayMs_ * sampleRate / 1000.0f);

    // Reprocess IR if loaded (for sample rate conversion)
    if (irLoaded_) {
        reprocessIR();
    }

    std::cout << "[ConvolutionReverb] Prepared: " << sampleRate << " Hz, "
              << "block size: " << maxBlockSize << std::endl;
}

void ConvolutionReverb::process(Core::AudioBuffer& buffer) {
    if (bypassed_ || !irLoaded_) return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    if (numChannels < 2) return;

    float* left = buffer.getWritePointer(0);
    float* right = buffer.getWritePointer(1);

    // Copy dry signal
    std::copy(left, left + numSamples, tempL_.begin());
    std::copy(right, right + numSamples, tempR_.begin());

    // Clear wet buffers
    std::fill(wetL_.begin(), wetL_.begin() + numSamples, 0.0f);
    std::fill(wetR_.begin(), wetR_.begin() + numSamples, 0.0f);

    // Process convolution
    std::lock_guard<std::mutex> lock(irMutex_);
    processPartitionedConvolution(wetL_.data(), wetR_.data(), numSamples);

    // Apply pre-delay
    if (preDelaySamples_ > 0) {
        processPreDelay(wetL_.data(), wetR_.data(), numSamples);
    }

    // Apply gain
    float gain = std::pow(10.0f, gainDB_ / 20.0f);

    // Mix dry/wet
    float wetGain = mix_ * gain;
    float dryGain = 1.0f - mix_;

    for (int i = 0; i < numSamples; ++i) {
        left[i] = tempL_[i] * dryGain + wetL_[i] * wetGain;
        right[i] = tempR_[i] * dryGain + wetR_[i] * wetGain;
    }
}

void ConvolutionReverb::reset() {
    // Reset convolution state
    if (!stateL_.inputBuffer.empty()) {
        std::fill(stateL_.inputBuffer.begin(), stateL_.inputBuffer.end(), 0.0f);
        std::fill(stateL_.outputBuffer.begin(), stateL_.outputBuffer.end(), 0.0f);
        std::fill(stateL_.accumulator.begin(), stateL_.accumulator.end(),
                  std::complex<float>(0.0f, 0.0f));
        stateL_.inputWritePos = 0;
        stateL_.outputReadPos = 0;
        stateL_.partitionIndex = 0;
    }

    if (!stateR_.inputBuffer.empty()) {
        std::fill(stateR_.inputBuffer.begin(), stateR_.inputBuffer.end(), 0.0f);
        std::fill(stateR_.outputBuffer.begin(), stateR_.outputBuffer.end(), 0.0f);
        std::fill(stateR_.accumulator.begin(), stateR_.accumulator.end(),
                  std::complex<float>(0.0f, 0.0f));
        stateR_.inputWritePos = 0;
        stateR_.outputReadPos = 0;
        stateR_.partitionIndex = 0;
    }

    // Reset pre-delay
    std::fill(preDelayBufferL_.begin(), preDelayBufferL_.end(), 0.0f);
    std::fill(preDelayBufferR_.begin(), preDelayBufferR_.end(), 0.0f);
    preDelayWritePos_ = 0;
}

// ============================================================================
// IR Management
// ============================================================================

bool ConvolutionReverb::loadIR(const std::string& filepath) {
    AudioFile loader;
    if (!loader.load(filepath)) {
        std::cerr << "[ConvolutionReverb] Failed to load IR: " << filepath << std::endl;
        return false;
    }

    Core::AudioBuffer irBuffer(loader.getNumChannels(),
                               static_cast<int>(loader.getSamples().size() / loader.getNumChannels()));

    // Copy samples to buffer
    const auto& samples = loader.getSamples();
    int numChannels = loader.getNumChannels();
    int numFrames = static_cast<int>(samples.size() / numChannels);

    for (int ch = 0; ch < numChannels; ++ch) {
        float* dest = irBuffer.getWritePointer(ch);
        for (int i = 0; i < numFrames; ++i) {
            dest[i] = samples[i * numChannels + ch];
        }
    }

    loadIR(irBuffer, loader.getSampleRate());

    // Extract filename for info
    size_t lastSlash = filepath.find_last_of("/\\");
    irInfo_.name = (lastSlash != std::string::npos)
                   ? filepath.substr(lastSlash + 1)
                   : filepath;

    std::cout << "[ConvolutionReverb] Loaded IR: " << irInfo_.name
              << " (" << irInfo_.lengthSeconds << "s, "
              << irInfo_.numChannels << " channels)" << std::endl;

    return true;
}

void ConvolutionReverb::loadIR(const Core::AudioBuffer& ir, int irSampleRate) {
    std::lock_guard<std::mutex> lock(irMutex_);

    trueStereo_ = false;
    irInfo_.sampleRate = irSampleRate;
    irInfo_.numChannels = ir.getNumChannels();
    irInfo_.numSamples = ir.getNumSamples();
    irInfo_.lengthSeconds = static_cast<float>(ir.getNumSamples()) / irSampleRate;
    irInfo_.isTrueStereo = false;

    // Limit IR length
    size_t maxSamples = std::min(static_cast<size_t>(ir.getNumSamples()), MAX_IR_SAMPLES);

    // Extract IR data
    irL_.resize(maxSamples);
    std::copy(ir.getReadPointer(0), ir.getReadPointer(0) + maxSamples, irL_.begin());

    if (ir.getNumChannels() >= 2) {
        irR_.resize(maxSamples);
        std::copy(ir.getReadPointer(1), ir.getReadPointer(1) + maxSamples, irR_.begin());
    } else {
        irR_ = irL_;  // Mono IR
    }

    irLoaded_ = true;
    reprocessIR();
}

void ConvolutionReverb::loadTrueStereoIR(const Core::AudioBuffer& ir, int irSampleRate) {
    if (ir.getNumChannels() < 4) {
        std::cerr << "[ConvolutionReverb] True stereo IR requires 4 channels" << std::endl;
        loadIR(ir, irSampleRate);
        return;
    }

    std::lock_guard<std::mutex> lock(irMutex_);

    trueStereo_ = true;
    irInfo_.sampleRate = irSampleRate;
    irInfo_.numChannels = 4;
    irInfo_.numSamples = ir.getNumSamples();
    irInfo_.lengthSeconds = static_cast<float>(ir.getNumSamples()) / irSampleRate;
    irInfo_.isTrueStereo = true;

    size_t maxSamples = std::min(static_cast<size_t>(ir.getNumSamples()), MAX_IR_SAMPLES);

    // LL, LR, RL, RR
    irL_.resize(maxSamples);
    irLR_.resize(maxSamples);
    irRL_.resize(maxSamples);
    irR_.resize(maxSamples);

    std::copy(ir.getReadPointer(0), ir.getReadPointer(0) + maxSamples, irL_.begin());
    std::copy(ir.getReadPointer(1), ir.getReadPointer(1) + maxSamples, irLR_.begin());
    std::copy(ir.getReadPointer(2), ir.getReadPointer(2) + maxSamples, irRL_.begin());
    std::copy(ir.getReadPointer(3), ir.getReadPointer(3) + maxSamples, irR_.begin());

    irLoaded_ = true;
    reprocessIR();
}

// ============================================================================
// Parameters
// ============================================================================

void ConvolutionReverb::setPreDelay(float ms) {
    preDelayMs_ = std::clamp(ms, 0.0f, 500.0f);
    preDelaySamples_ = static_cast<int>(preDelayMs_ * sampleRate_ / 1000.0f);
}

void ConvolutionReverb::setLowCPUMode(bool enable) {
    lowCPUMode_ = enable;
    if (irLoaded_) {
        reprocessIR();
    }
}

void ConvolutionReverb::setIRTrim(float startSec, float endSec) {
    trimStartSec_ = std::max(0.0f, startSec);
    trimEndSec_ = endSec;
    if (irLoaded_) {
        reprocessIR();
    }
}

// ============================================================================
// FFT Implementation
// ============================================================================

int ConvolutionReverb::nextPowerOfTwo(int n) {
    int power = 1;
    while (power < n) power <<= 1;
    return power;
}

void ConvolutionReverb::fft(std::vector<std::complex<float>>& data, bool inverse) {
    fftIterative(data.data(), static_cast<int>(data.size()), inverse);
}

void ConvolutionReverb::fftIterative(std::complex<float>* data, int n, bool inverse) {
    // Bit reversal
    int j = 0;
    for (int i = 0; i < n - 1; ++i) {
        if (i < j) std::swap(data[i], data[j]);
        int k = n >> 1;
        while (k <= j) {
            j -= k;
            k >>= 1;
        }
        j += k;
    }

    // Cooley-Tukey iterative FFT
    const float pi = 3.14159265358979323846f;
    float sign = inverse ? 1.0f : -1.0f;

    for (int len = 2; len <= n; len <<= 1) {
        float theta = sign * 2.0f * pi / len;
        std::complex<float> wn(std::cos(theta), std::sin(theta));

        for (int i = 0; i < n; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (int jj = 0; jj < len / 2; ++jj) {
                std::complex<float> u = data[i + jj];
                std::complex<float> t = w * data[i + jj + len / 2];
                data[i + jj] = u + t;
                data[i + jj + len / 2] = u - t;
                w *= wn;
            }
        }
    }

    // Normalize for inverse
    if (inverse) {
        float scale = 1.0f / n;
        for (int i = 0; i < n; ++i) {
            data[i] *= scale;
        }
    }
}

// ============================================================================
// IR Processing
// ============================================================================

void ConvolutionReverb::reprocessIR() {
    if (!irLoaded_) return;

    // Apply trim
    size_t startSample = static_cast<size_t>(trimStartSec_ * irInfo_.sampleRate);
    size_t endSample = (trimEndSec_ > 0)
                       ? static_cast<size_t>(trimEndSec_ * irInfo_.sampleRate)
                       : irL_.size();

    startSample = std::min(startSample, irL_.size());
    endSample = std::min(endSample, irL_.size());

    if (startSample >= endSample) {
        startSample = 0;
        endSample = irL_.size();
    }

    // Create trimmed copies
    std::vector<float> irLTrimmed(irL_.begin() + startSample, irL_.begin() + endSample);
    std::vector<float> irRTrimmed(irR_.begin() + startSample, irR_.begin() + endSample);

    // Resample if needed
    if (irInfo_.sampleRate != sampleRate_ && sampleRate_ > 0) {
        std::vector<float> resampledL, resampledR;
        resampleIR(irLTrimmed, irInfo_.sampleRate, resampledL, sampleRate_);
        resampleIR(irRTrimmed, irInfo_.sampleRate, resampledR, sampleRate_);
        irLTrimmed = std::move(resampledL);
        irRTrimmed = std::move(resampledR);
    }

    // Normalize
    if (normalize_) {
        normalizeIR(irLTrimmed);
        normalizeIR(irRTrimmed);
    }

    // Apply damping
    if (damping_ > 0.0f) {
        // Simple one-pole lowpass applied progressively
        float dampCoeff = 1.0f - damping_ * 0.5f;
        float prevL = 0.0f, prevR = 0.0f;
        for (size_t i = 0; i < irLTrimmed.size(); ++i) {
            float progress = static_cast<float>(i) / irLTrimmed.size();
            float localDamp = 1.0f - progress * damping_;

            irLTrimmed[i] = irLTrimmed[i] * localDamp + prevL * (1.0f - localDamp);
            irRTrimmed[i] = irRTrimmed[i] * localDamp + prevR * (1.0f - localDamp);

            prevL = irLTrimmed[i];
            prevR = irRTrimmed[i];
        }
    }

    // Store processed IR for partitioning
    irL_ = std::move(irLTrimmed);
    irR_ = std::move(irRTrimmed);

    // Partition for FFT convolution
    partitionIR();

    // Initialize state
    reset();
}

void ConvolutionReverb::normalizeIR(std::vector<float>& ir) {
    float peak = 0.0f;
    for (float s : ir) {
        peak = std::max(peak, std::abs(s));
    }
    if (peak > 0.0001f) {
        float scale = 1.0f / peak;
        for (float& s : ir) {
            s *= scale;
        }
    }
}

void ConvolutionReverb::resampleIR(const std::vector<float>& input, int inputRate,
                                    std::vector<float>& output, int outputRate) {
    if (inputRate == outputRate) {
        output = input;
        return;
    }

    double ratio = static_cast<double>(outputRate) / inputRate;
    size_t newLength = static_cast<size_t>(input.size() * ratio);
    output.resize(newLength);

    // Linear interpolation resampling
    for (size_t i = 0; i < newLength; ++i) {
        double srcPos = i / ratio;
        size_t srcIndex = static_cast<size_t>(srcPos);
        float frac = static_cast<float>(srcPos - srcIndex);

        if (srcIndex + 1 < input.size()) {
            output[i] = input[srcIndex] * (1.0f - frac) + input[srcIndex + 1] * frac;
        } else if (srcIndex < input.size()) {
            output[i] = input[srcIndex];
        } else {
            output[i] = 0.0f;
        }
    }
}

void ConvolutionReverb::partitionIR() {
    // Determine FFT size based on CPU mode
    if (lowCPUMode_) {
        fftSize_ = MAX_FFT_SIZE;
    } else {
        fftSize_ = 2048;  // Good balance for low latency
    }
    partitionSize_ = fftSize_ / 2;

    // Calculate number of partitions
    size_t irLength = std::max(irL_.size(), irR_.size());
    numPartitions_ = static_cast<int>((irLength + partitionSize_ - 1) / partitionSize_);

    partitions_.resize(numPartitions_);

    // FFT each partition
    for (int p = 0; p < numPartitions_; ++p) {
        Partition& part = partitions_[p];

        part.irFreqL.resize(fftSize_);
        part.irFreqR.resize(fftSize_);

        // Zero-pad and copy IR segment
        std::fill(part.irFreqL.begin(), part.irFreqL.end(), std::complex<float>(0.0f, 0.0f));
        std::fill(part.irFreqR.begin(), part.irFreqR.end(), std::complex<float>(0.0f, 0.0f));

        size_t startSample = p * partitionSize_;
        size_t samplesToProcess = std::min(static_cast<size_t>(partitionSize_),
                                           irL_.size() - startSample);

        if (startSample < irL_.size()) {
            for (size_t i = 0; i < samplesToProcess; ++i) {
                part.irFreqL[i] = std::complex<float>(irL_[startSample + i], 0.0f);
            }
        }

        if (startSample < irR_.size()) {
            samplesToProcess = std::min(static_cast<size_t>(partitionSize_),
                                        irR_.size() - startSample);
            for (size_t i = 0; i < samplesToProcess; ++i) {
                part.irFreqR[i] = std::complex<float>(irR_[startSample + i], 0.0f);
            }
        }

        // Transform to frequency domain
        fft(part.irFreqL, false);
        fft(part.irFreqR, false);

        // True stereo partitions
        if (trueStereo_) {
            part.irFreqLR.resize(fftSize_);
            part.irFreqRL.resize(fftSize_);
            std::fill(part.irFreqLR.begin(), part.irFreqLR.end(), std::complex<float>(0.0f, 0.0f));
            std::fill(part.irFreqRL.begin(), part.irFreqRL.end(), std::complex<float>(0.0f, 0.0f));

            if (startSample < irLR_.size()) {
                samplesToProcess = std::min(static_cast<size_t>(partitionSize_),
                                            irLR_.size() - startSample);
                for (size_t i = 0; i < samplesToProcess; ++i) {
                    part.irFreqLR[i] = std::complex<float>(irLR_[startSample + i], 0.0f);
                    part.irFreqRL[i] = std::complex<float>(irRL_[startSample + i], 0.0f);
                }
            }
            fft(part.irFreqLR, false);
            fft(part.irFreqRL, false);
        }
    }

    // Initialize convolution state
    stateL_.inputBuffer.resize(partitionSize_, 0.0f);
    stateL_.outputBuffer.resize(partitionSize_ * 2, 0.0f);
    stateL_.fftBuffer.resize(fftSize_);
    stateL_.accumulator.resize(fftSize_, std::complex<float>(0.0f, 0.0f));
    stateL_.inputWritePos = 0;
    stateL_.outputReadPos = 0;
    stateL_.partitionIndex = 0;

    stateR_.inputBuffer.resize(partitionSize_, 0.0f);
    stateR_.outputBuffer.resize(partitionSize_ * 2, 0.0f);
    stateR_.fftBuffer.resize(fftSize_);
    stateR_.accumulator.resize(fftSize_, std::complex<float>(0.0f, 0.0f));
    stateR_.inputWritePos = 0;
    stateR_.outputReadPos = 0;
    stateR_.partitionIndex = 0;

    std::cout << "[ConvolutionReverb] IR partitioned: " << numPartitions_
              << " partitions, FFT size: " << fftSize_ << std::endl;
}

// ============================================================================
// Convolution Processing
// ============================================================================

void ConvolutionReverb::processPartitionedConvolution(float* left, float* right, int numSamples) {
    if (numPartitions_ == 0) return;

    for (int i = 0; i < numSamples; ++i) {
        // Store input sample
        stateL_.inputBuffer[stateL_.inputWritePos] = left[i];
        stateR_.inputBuffer[stateR_.inputWritePos] = right[i];

        // Output from buffer
        left[i] = stateL_.outputBuffer[stateL_.outputReadPos];
        right[i] = stateR_.outputBuffer[stateR_.outputReadPos];

        // Clear output position for next overlap-add
        stateL_.outputBuffer[stateL_.outputReadPos] = 0.0f;
        stateR_.outputBuffer[stateR_.outputReadPos] = 0.0f;

        stateL_.inputWritePos++;
        stateR_.inputWritePos++;
        stateL_.outputReadPos++;
        stateR_.outputReadPos++;

        // Process partition when buffer is full
        if (stateL_.inputWritePos >= partitionSize_) {
            // Prepare FFT input (zero-padded)
            std::fill(stateL_.fftBuffer.begin(), stateL_.fftBuffer.end(),
                      std::complex<float>(0.0f, 0.0f));
            std::fill(stateR_.fftBuffer.begin(), stateR_.fftBuffer.end(),
                      std::complex<float>(0.0f, 0.0f));

            for (int j = 0; j < partitionSize_; ++j) {
                stateL_.fftBuffer[j] = std::complex<float>(stateL_.inputBuffer[j], 0.0f);
                stateR_.fftBuffer[j] = std::complex<float>(stateR_.inputBuffer[j], 0.0f);
            }

            // FFT the input
            fft(stateL_.fftBuffer, false);
            fft(stateR_.fftBuffer, false);

            // Multiply-accumulate with each partition
            for (int p = 0; p < numPartitions_; ++p) {
                int partIdx = (stateL_.partitionIndex - p + numPartitions_) % numPartitions_;
                const Partition& part = partitions_[p];

                // Complex multiplication
                for (int k = 0; k < fftSize_; ++k) {
                    if (trueStereo_) {
                        // True stereo: L = LL*L + LR*R, R = RL*L + RR*R
                        stateL_.accumulator[k] += stateL_.fftBuffer[k] * part.irFreqL[k] +
                                                  stateR_.fftBuffer[k] * part.irFreqLR[k];
                        stateR_.accumulator[k] += stateL_.fftBuffer[k] * part.irFreqRL[k] +
                                                  stateR_.fftBuffer[k] * part.irFreqR[k];
                    } else {
                        stateL_.accumulator[k] += stateL_.fftBuffer[k] * part.irFreqL[k];
                        stateR_.accumulator[k] += stateR_.fftBuffer[k] * part.irFreqR[k];
                    }
                }
            }

            // IFFT
            fft(stateL_.accumulator, true);
            fft(stateR_.accumulator, true);

            // Overlap-add to output buffer
            for (int j = 0; j < fftSize_; ++j) {
                int outIdx = j % (partitionSize_ * 2);
                stateL_.outputBuffer[outIdx] += stateL_.accumulator[j].real();
                stateR_.outputBuffer[outIdx] += stateR_.accumulator[j].real();
            }

            // Clear accumulator
            std::fill(stateL_.accumulator.begin(), stateL_.accumulator.end(),
                      std::complex<float>(0.0f, 0.0f));
            std::fill(stateR_.accumulator.begin(), stateR_.accumulator.end(),
                      std::complex<float>(0.0f, 0.0f));

            // Reset for next partition
            stateL_.inputWritePos = 0;
            stateR_.inputWritePos = 0;
            stateL_.partitionIndex = (stateL_.partitionIndex + 1) % numPartitions_;
            stateR_.partitionIndex = (stateR_.partitionIndex + 1) % numPartitions_;

            // Wrap output read position
            if (stateL_.outputReadPos >= partitionSize_ * 2) {
                stateL_.outputReadPos = 0;
                stateR_.outputReadPos = 0;
            }
        }
    }
}

void ConvolutionReverb::processPreDelay(float* left, float* right, int numSamples) {
    int bufferSize = static_cast<int>(preDelayBufferL_.size());

    for (int i = 0; i < numSamples; ++i) {
        // Write current sample
        preDelayBufferL_[preDelayWritePos_] = left[i];
        preDelayBufferR_[preDelayWritePos_] = right[i];

        // Read delayed sample
        int readPos = (preDelayWritePos_ - preDelaySamples_ + bufferSize) % bufferSize;
        left[i] = preDelayBufferL_[readPos];
        right[i] = preDelayBufferR_[readPos];

        preDelayWritePos_ = (preDelayWritePos_ + 1) % bufferSize;
    }
}

// ============================================================================
// Synthetic IR Generation
// ============================================================================

void ConvolutionReverb::generateSyntheticIR(float decayTime, float roomSize, float brightness) {
    std::vector<float> ir = IRFactory::generateRoom(sampleRate_, roomSize, decayTime, brightness);

    Core::AudioBuffer irBuffer(1, static_cast<int>(ir.size()));
    std::copy(ir.begin(), ir.end(), irBuffer.getWritePointer(0));

    loadIR(irBuffer, sampleRate_);
    irInfo_.name = "Synthetic Room";
}

// ============================================================================
// IRFactory Implementation
// ============================================================================

std::vector<float> IRFactory::generateExponentialDecay(int sampleRate, float decayTime,
                                                        float density) {
    int numSamples = static_cast<int>(decayTime * sampleRate);
    std::vector<float> ir(numSamples);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float decayRate = -6.907755f / (decayTime * sampleRate);  // -60dB decay

    for (int i = 0; i < numSamples; ++i) {
        float noise = dist(rng);
        float envelope = std::exp(decayRate * i);

        // Add density variation
        if (density < 1.0f) {
            if (dist(rng) > density) noise = 0.0f;
        }

        ir[i] = noise * envelope;
    }

    // Initial spike
    if (!ir.empty()) {
        ir[0] = 1.0f;
    }

    return ir;
}

std::vector<float> IRFactory::generateRoom(int sampleRate, float roomSizeMeters,
                                            float rt60, float brightness) {
    int numSamples = static_cast<int>(rt60 * sampleRate);
    std::vector<float> ir(numSamples, 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Speed of sound
    const float speedOfSound = 343.0f;

    // Early reflections (6 surfaces)
    float distances[] = {
        roomSizeMeters,           // Front
        roomSizeMeters * 0.8f,    // Back
        roomSizeMeters * 0.5f,    // Left
        roomSizeMeters * 0.5f,    // Right
        roomSizeMeters * 0.3f,    // Ceiling
        roomSizeMeters * 0.25f    // Floor
    };

    for (float distance : distances) {
        int delaySamples = static_cast<int>(distance / speedOfSound * sampleRate);
        if (delaySamples < numSamples) {
            float attenuation = 1.0f / (1.0f + distance * 0.1f);
            ir[delaySamples] += attenuation * (dist(rng) * 0.5f + 0.5f);
        }
    }

    // Late reverb tail
    float decayRate = -6.907755f / (rt60 * sampleRate);
    int lateStart = static_cast<int>(0.08f * sampleRate);  // 80ms

    for (int i = lateStart; i < numSamples; ++i) {
        float noise = dist(rng);
        float envelope = std::exp(decayRate * (i - lateStart));

        // High-frequency rolloff for darker sound
        float hfRolloff = 1.0f - (1.0f - brightness) * static_cast<float>(i) / numSamples;

        ir[i] += noise * envelope * hfRolloff * 0.3f;
    }

    // Direct sound
    ir[0] = 1.0f;

    // Normalize
    float peak = 0.0f;
    for (float s : ir) peak = std::max(peak, std::abs(s));
    if (peak > 0.0f) {
        for (float& s : ir) s /= peak;
    }

    return ir;
}

std::vector<float> IRFactory::generatePlate(int sampleRate, float decayTime, float damping) {
    int numSamples = static_cast<int>(decayTime * sampleRate);
    std::vector<float> ir(numSamples, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float decayRate = -6.907755f / (decayTime * sampleRate);

    // Dense, bright reflections
    for (int i = 0; i < numSamples; ++i) {
        float envelope = std::exp(decayRate * i);
        float damp = 1.0f - damping * static_cast<float>(i) / numSamples;

        ir[i] = dist(rng) * envelope * damp;
    }

    // Metallic initial transient
    ir[0] = 1.0f;
    for (int i = 1; i < std::min(500, numSamples); ++i) {
        ir[i] += std::sin(i * 0.05f) * std::exp(-i * 0.01f) * 0.3f;
    }

    return ir;
}

std::vector<float> IRFactory::generateSpring(int sampleRate, float decayTime, float tension) {
    int numSamples = static_cast<int>(decayTime * sampleRate);
    std::vector<float> ir(numSamples, 0.0f);

    std::mt19937 rng(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float decayRate = -4.0f / (decayTime * sampleRate);

    // Spring characteristic: delay + flutter
    float springFreq = 200.0f + tension * 800.0f;  // Hz

    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;
        float envelope = std::exp(decayRate * i);

        // Dispersive delay characteristic
        float dispersion = std::sin(2.0f * 3.14159f * springFreq * t +
                                    std::sin(t * 50.0f) * (1.0f - tension));

        ir[i] = (dispersion * 0.5f + dist(rng) * 0.5f) * envelope;
    }

    ir[0] = 1.0f;

    return ir;
}

std::vector<float> IRFactory::generateHall(int sampleRate, float rt60, float diffusion) {
    int numSamples = static_cast<int>(rt60 * sampleRate);
    std::vector<float> ir(numSamples, 0.0f);

    std::mt19937 rng(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    float decayRate = -6.907755f / (rt60 * sampleRate);

    // Build-up time for hall
    float buildUp = 0.1f;  // 100ms build-up
    int buildUpSamples = static_cast<int>(buildUp * sampleRate);

    for (int i = 0; i < numSamples; ++i) {
        float envelope = std::exp(decayRate * i);

        // Build-up envelope
        float buildEnv = (i < buildUpSamples)
                         ? static_cast<float>(i) / buildUpSamples
                         : 1.0f;

        // Diffusion affects noise density
        float noise = 0.0f;
        int diffusionSteps = static_cast<int>(1 + diffusion * 5);
        for (int d = 0; d < diffusionSteps; ++d) {
            noise += dist(rng);
        }
        noise /= diffusionSteps;

        ir[i] = noise * envelope * buildEnv;
    }

    // Direct sound (reduced for large hall)
    ir[0] = 0.7f;

    // Normalize
    float peak = 0.0f;
    for (float s : ir) peak = std::max(peak, std::abs(s));
    if (peak > 0.0f) {
        for (float& s : ir) s /= peak;
    }

    return ir;
}

} // namespace DSP
} // namespace MolinAntro
