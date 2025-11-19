#include "dsp/Effects.h"
#include <cmath>
#include <iostream>

namespace MolinAntro {
namespace DSP {

// ============================================================================
// PARAMETRIC EQ IMPLEMENTATION
// ============================================================================

ParametricEQ::ParametricEQ() {
    std::cout << "[ParametricEQ] Constructed" << std::endl;

    // Initialize default bands
    bands_[0].type = FilterType::LowShelf;
    bands_[0].frequency = 100.0f;
    bands_[0].gain = 0.0f;
    bands_[0].Q = 0.707f;

    bands_[1].type = FilterType::Peak;
    bands_[1].frequency = 1000.0f;
    bands_[1].gain = 0.0f;
    bands_[1].Q = 1.0f;

    bands_[2].type = FilterType::Peak;
    bands_[2].frequency = 5000.0f;
    bands_[2].gain = 0.0f;
    bands_[2].Q = 1.0f;

    bands_[3].type = FilterType::HighShelf;
    bands_[3].frequency = 10000.0f;
    bands_[3].gain = 0.0f;
    bands_[3].Q = 0.707f;
}

void ParametricEQ::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    // Calculate initial coefficients for all bands
    for (int i = 0; i < NUM_BANDS; ++i) {
        calculateCoefficients(i);
    }

    reset();
}

void ParametricEQ::process(Core::AudioBuffer& buffer) {
    if (bypassed_) return;

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        float* channelData = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            float sample = channelData[i];

            // Process through each enabled band
            for (int band = 0; band < NUM_BANDS; ++band) {
                if (bands_[band].enabled) {
                    sample = processSample(sample, band, ch);
                }
            }

            channelData[i] = sample;
        }
    }
}

void ParametricEQ::reset() {
    for (int band = 0; band < NUM_BANDS; ++band) {
        for (int ch = 0; ch < 2; ++ch) {
            state_[band][ch].x1 = 0.0f;
            state_[band][ch].x2 = 0.0f;
            state_[band][ch].y1 = 0.0f;
            state_[band][ch].y2 = 0.0f;
        }
    }
}

void ParametricEQ::setBand(int bandIndex, const Band& band) {
    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return;

    bands_[bandIndex] = band;
    calculateCoefficients(bandIndex);
}

ParametricEQ::Band ParametricEQ::getBand(int bandIndex) const {
    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return Band();
    return bands_[bandIndex];
}

void ParametricEQ::calculateCoefficients(int bandIndex) {
    const Band& band = bands_[bandIndex];

    const float omega = 2.0f * M_PI * band.frequency / sampleRate_;
    const float sinOmega = std::sin(omega);
    const float cosOmega = std::cos(omega);
    const float alpha = sinOmega / (2.0f * band.Q);
    const float A = std::pow(10.0f, band.gain / 40.0f);

    BiquadCoeffs& c = coeffs_[bandIndex][0]; // Same for both channels

    switch (band.type) {
        case FilterType::Peak:
            c.b0 = 1.0f + alpha * A;
            c.b1 = -2.0f * cosOmega;
            c.b2 = 1.0f - alpha * A;
            c.a1 = -2.0f * cosOmega;
            c.a2 = 1.0f - alpha / A;
            break;

        case FilterType::LowShelf:
            c.b0 = A * ((A + 1.0f) - (A - 1.0f) * cosOmega + 2.0f * std::sqrt(A) * alpha);
            c.b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosOmega);
            c.b2 = A * ((A + 1.0f) - (A - 1.0f) * cosOmega - 2.0f * std::sqrt(A) * alpha);
            c.a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosOmega);
            c.a2 = (A + 1.0f) + (A - 1.0f) * cosOmega - 2.0f * std::sqrt(A) * alpha;
            break;

        case FilterType::HighShelf:
            c.b0 = A * ((A + 1.0f) + (A - 1.0f) * cosOmega + 2.0f * std::sqrt(A) * alpha);
            c.b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosOmega);
            c.b2 = A * ((A + 1.0f) + (A - 1.0f) * cosOmega - 2.0f * std::sqrt(A) * alpha);
            c.a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cosOmega);
            c.a2 = (A + 1.0f) - (A - 1.0f) * cosOmega - 2.0f * std::sqrt(A) * alpha;
            break;

        case FilterType::LowPass:
            c.b0 = (1.0f - cosOmega) / 2.0f;
            c.b1 = 1.0f - cosOmega;
            c.b2 = (1.0f - cosOmega) / 2.0f;
            c.a1 = -2.0f * cosOmega;
            c.a2 = 1.0f - alpha;
            break;

        case FilterType::HighPass:
            c.b0 = (1.0f + cosOmega) / 2.0f;
            c.b1 = -(1.0f + cosOmega);
            c.b2 = (1.0f + cosOmega) / 2.0f;
            c.a1 = -2.0f * cosOmega;
            c.a2 = 1.0f - alpha;
            break;

        case FilterType::Notch:
            c.b0 = 1.0f;
            c.b1 = -2.0f * cosOmega;
            c.b2 = 1.0f;
            c.a1 = -2.0f * cosOmega;
            c.a2 = 1.0f - alpha;
            break;
    }

    // Normalize
    const float a0 = 1.0f + alpha;
    c.b0 /= a0;
    c.b1 /= a0;
    c.b2 /= a0;
    c.a1 /= a0;
    c.a2 /= a0;

    // Copy to second channel
    coeffs_[bandIndex][1] = c;
}

float ParametricEQ::processSample(float input, int bandIndex, int channel) {
    const BiquadCoeffs& c = coeffs_[bandIndex][channel];
    BiquadState& s = state_[bandIndex][channel];

    const float output = c.b0 * input + c.b1 * s.x1 + c.b2 * s.x2
                       - c.a1 * s.y1 - c.a2 * s.y2;

    s.x2 = s.x1;
    s.x1 = input;
    s.y2 = s.y1;
    s.y1 = output;

    return output;
}

// ============================================================================
// COMPRESSOR IMPLEMENTATION
// ============================================================================

Compressor::Compressor() {
    std::cout << "[Compressor] Constructed" << std::endl;
}

void Compressor::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;
    updateEnvelope();
    reset();
}

void Compressor::process(Core::AudioBuffer& buffer) {
    if (bypassed_) return;

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    const float makeupGainLinear = std::pow(10.0f, makeupGain_ / 20.0f);

    for (int i = 0; i < numSamples; ++i) {
        // Get peak level from all channels
        float peak = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch) {
            peak = std::max(peak, std::abs(buffer.getReadPointer(ch)[i]));
        }

        // Convert to dB
        const float inputLevel = 20.0f * std::log10(peak + 1e-10f);

        // Compute gain reduction
        const float gain = computeGain(inputLevel);

        // Smooth with envelope follower
        const float coeff = (gain < envelope_) ? attackCoeff_ : releaseCoeff_;
        envelope_ = envelope_ * coeff + gain * (1.0f - coeff);

        // Convert back to linear
        const float gainLinear = std::pow(10.0f, envelope_ / 20.0f);

        // Apply gain to all channels
        for (int ch = 0; ch < numChannels; ++ch) {
            float* channelData = buffer.getWritePointer(ch);
            channelData[i] *= gainLinear * makeupGainLinear;
        }

        gainReduction_ = -envelope_;
    }
}

void Compressor::reset() {
    envelope_ = 0.0f;
    gainReduction_ = 0.0f;
}

void Compressor::updateEnvelope() {
    attackCoeff_ = std::exp(-1.0f / (attackTime_ * 0.001f * sampleRate_));
    releaseCoeff_ = std::exp(-1.0f / (releaseTime_ * 0.001f * sampleRate_));
}

float Compressor::computeGain(float inputLevel) {
    const float overThreshold = inputLevel - threshold_;

    if (overThreshold <= -knee_ / 2.0f) {
        return 0.0f; // Below threshold
    } else if (overThreshold >= knee_ / 2.0f) {
        // Above knee - full compression
        return (1.0f - 1.0f / ratio_) * overThreshold;
    } else {
        // In knee - soft transition
        const float kneeValue = overThreshold + knee_ / 2.0f;
        return (1.0f - 1.0f / ratio_) * kneeValue * kneeValue / (2.0f * knee_);
    }
}

// ============================================================================
// DELAY IMPLEMENTATION
// ============================================================================

Delay::Delay() {
    std::cout << "[Delay] Constructed" << std::endl;
}

void Delay::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    // Allocate delay buffers (max 5 seconds)
    const int maxDelaySamples = sampleRate * 5;
    for (int ch = 0; ch < 2; ++ch) {
        delayBuffer_[ch].resize(maxDelaySamples, 0.0f);
    }

    updateDelay();
    reset();
}

void Delay::process(Core::AudioBuffer& buffer) {
    if (bypassed_) return;

    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    for (int ch = 0; ch < numChannels; ++ch) {
        float* channelData = buffer.getWritePointer(ch);
        std::vector<float>& delayBuf = delayBuffer_[ch];
        int& writeIdx = writeIndex_[ch];

        for (int i = 0; i < numSamples; ++i) {
            const float input = channelData[i];

            // Read from delay buffer
            int readIdx = writeIdx - delayInSamples_;
            if (readIdx < 0) readIdx += delayBuf.size();

            float delayed = delayBuf[readIdx];

            // Ping-pong: swap channels on feedback
            if (pingPong_ && ch < 1) {
                int otherReadIdx = writeIndex_[1 - ch] - delayInSamples_;
                if (otherReadIdx < 0) otherReadIdx += delayBuffer_[1 - ch].size();
                delayed = delayBuffer_[1 - ch][otherReadIdx];
            }

            // Write to delay buffer with feedback
            delayBuf[writeIdx] = input + delayed * feedback_;

            // Mix dry and wet
            channelData[i] = input * (1.0f - mix_) + delayed * mix_;

            writeIdx = (writeIdx + 1) % delayBuf.size();
        }
    }
}

void Delay::reset() {
    for (int ch = 0; ch < 2; ++ch) {
        std::fill(delayBuffer_[ch].begin(), delayBuffer_[ch].end(), 0.0f);
        writeIndex_[ch] = 0;
    }
}

void Delay::updateDelay() {
    delayInSamples_ = static_cast<int>(delayTime_ * 0.001f * sampleRate_);
    delayInSamples_ = std::clamp(delayInSamples_, 1, static_cast<int>(delayBuffer_[0].size() - 1));
}

// ============================================================================
// LIMITER IMPLEMENTATION
// ============================================================================

Limiter::Limiter() {
    std::cout << "[Limiter] Constructed" << std::endl;
}

void Limiter::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;
    releaseCoeff_ = std::exp(-1.0f / (releaseTime_ * 0.001f * sampleRate));
    reset();
}

void Limiter::process(Core::AudioBuffer& buffer) {
    if (bypassed_) return;

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    const float thresholdLinear = std::pow(10.0f, threshold_ / 20.0f);
    const float ceilingLinear = std::pow(10.0f, ceiling_ / 20.0f);

    for (int i = 0; i < numSamples; ++i) {
        // Get peak level
        float peak = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch) {
            peak = std::max(peak, std::abs(buffer.getReadPointer(ch)[i]));
        }

        // Compute gain
        float targetGain = 1.0f;
        if (peak > thresholdLinear) {
            targetGain = thresholdLinear / peak;
        }

        // Envelope follower (instant attack)
        if (targetGain < envelope_) {
            envelope_ = targetGain;
        } else {
            envelope_ = envelope_ * releaseCoeff_ + targetGain * (1.0f - releaseCoeff_);
        }

        // Apply gain and ceiling
        for (int ch = 0; ch < numChannels; ++ch) {
            float* channelData = buffer.getWritePointer(ch);
            channelData[i] *= envelope_;
            channelData[i] = std::clamp(channelData[i], -ceilingLinear, ceilingLinear);
        }
    }
}

void Limiter::reset() {
    envelope_ = 1.0f;
}

// ============================================================================
// SATURATOR IMPLEMENTATION
// ============================================================================

Saturator::Saturator() {
    std::cout << "[Saturator] Constructed" << std::endl;
}

void Saturator::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;
    reset();
}

void Saturator::process(Core::AudioBuffer& buffer) {
    if (bypassed_) return;

    const int numChannels = buffer.getNumChannels();
    const int numSamples = buffer.getNumSamples();
    const float driveAmount = drive_ / 10.0f;

    for (int ch = 0; ch < numChannels; ++ch) {
        float* channelData = buffer.getWritePointer(ch);

        for (int i = 0; i < numSamples; ++i) {
            const float dry = channelData[i];
            const float wet = processSample(dry * driveAmount) / driveAmount;
            channelData[i] = dry * (1.0f - mix_) + wet * mix_;
        }
    }
}

void Saturator::reset() {
    // Nothing to reset
}

float Saturator::processSample(float input) {
    switch (mode_) {
        case Mode::Soft: return softClip(input);
        case Mode::Hard: return hardClip(input);
        case Mode::Tube: return tubeDistortion(input);
        case Mode::Tape: return tapeSimulation(input);
        case Mode::Digital: return hardClip(input);
        default: return input;
    }
}

float Saturator::softClip(float input) {
    return std::tanh(input);
}

float Saturator::hardClip(float input) {
    return std::clamp(input, -1.0f, 1.0f);
}

float Saturator::tubeDistortion(float input) {
    const float x = input * 0.5f;
    if (x > 1.0f) return 0.666f;
    if (x < -1.0f) return -0.666f;
    return x - (x * x * x) / 3.0f;
}

float Saturator::tapeSimulation(float input) {
    const float x = input * 0.7f;
    return x / (1.0f + std::abs(x));
}

// ============================================================================
// REVERB IMPLEMENTATION (Simplified Freeverb-style)
// ============================================================================

Reverb::Reverb() {
    std::cout << "[Reverb] Constructed" << std::endl;
}

void Reverb::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    // Freeverb comb delays (in samples)
    const int combTunings[NUM_COMBS] = {1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617};
    const int allpassTunings[NUM_ALLPASS] = {556, 441, 341, 225};

    // Allocate buffers
    for (int i = 0; i < NUM_COMBS; ++i) {
        for (int ch = 0; ch < 2; ++ch) {
            int size = combTunings[i];
            if (ch == 1) size += 23; // Stereo spread
            combBuffers_[i][ch].resize(size, 0.0f);
            combIndices_[i][ch] = 0;
            combFilters_[i][ch] = 0.0f;
        }
    }

    for (int i = 0; i < NUM_ALLPASS; ++i) {
        for (int ch = 0; ch < 2; ++ch) {
            int size = allpassTunings[i];
            if (ch == 1) size += 23;
            allpassBuffers_[i][ch].resize(size, 0.0f);
            allpassIndices_[i][ch] = 0;
        }
    }

    reset();
}

void Reverb::process(Core::AudioBuffer& buffer) {
    if (bypassed_) return;

    const int numChannels = std::min(buffer.getNumChannels(), 2);
    const int numSamples = buffer.getNumSamples();

    for (int i = 0; i < numSamples; ++i) {
        float inputL = numChannels > 0 ? buffer.getReadPointer(0)[i] : 0.0f;
        float inputR = numChannels > 1 ? buffer.getReadPointer(1)[i] : inputL;

        // Mono input to stereo reverb
        const float input = (inputL + inputR) * 0.5f;

        float outL = 0.0f, outR = 0.0f;

        // Process through comb filters
        for (int c = 0; c < NUM_COMBS; ++c) {
            outL += combBuffers_[c][0][combIndices_[c][0]];
            outR += combBuffers_[c][1][combIndices_[c][1]];
        }

        // Process through allpass filters
        float tempL = outL, tempR = outR;
        for (int a = 0; a < NUM_ALLPASS; ++a) {
            const float tmp = tempL;
            tempL = -tempL + allpassBuffers_[a][0][allpassIndices_[a][0]];
            allpassBuffers_[a][0][allpassIndices_[a][0]] = tmp + tempL * 0.5f;
            allpassIndices_[a][0] = (allpassIndices_[a][0] + 1) % allpassBuffers_[a][0].size();

            const float tmp2 = tempR;
            tempR = -tempR + allpassBuffers_[a][1][allpassIndices_[a][1]];
            allpassBuffers_[a][1][allpassIndices_[a][1]] = tmp2 + tempR * 0.5f;
            allpassIndices_[a][1] = (allpassIndices_[a][1] + 1) % allpassBuffers_[a][1].size();
        }

        // Mix dry/wet
        if (numChannels > 0) {
            buffer.getWritePointer(0)[i] = inputL * dryLevel_ + tempL * wetLevel_;
        }
        if (numChannels > 1) {
            buffer.getWritePointer(1)[i] = inputR * dryLevel_ + tempR * wetLevel_;
        }

        // Update comb filters
        for (int c = 0; c < NUM_COMBS; ++c) {
            combBuffers_[c][0][combIndices_[c][0]] = input + combBuffers_[c][0][combIndices_[c][0]] * roomSize_;
            combIndices_[c][0] = (combIndices_[c][0] + 1) % combBuffers_[c][0].size();

            combBuffers_[c][1][combIndices_[c][1]] = input + combBuffers_[c][1][combIndices_[c][1]] * roomSize_;
            combIndices_[c][1] = (combIndices_[c][1] + 1) % combBuffers_[c][1].size();
        }
    }
}

void Reverb::reset() {
    for (int i = 0; i < NUM_COMBS; ++i) {
        for (int ch = 0; ch < 2; ++ch) {
            std::fill(combBuffers_[i][ch].begin(), combBuffers_[i][ch].end(), 0.0f);
            combIndices_[i][ch] = 0;
            combFilters_[i][ch] = 0.0f;
        }
    }

    for (int i = 0; i < NUM_ALLPASS; ++i) {
        for (int ch = 0; ch < 2; ++ch) {
            std::fill(allpassBuffers_[i][ch].begin(), allpassBuffers_[i][ch].end(), 0.0f);
            allpassIndices_[i][ch] = 0;
        }
    }
}

void Reverb::processComb(float* /*buffer*/, int /*combIndex*/, int /*channel*/, int /*numSamples*/) {
    // Implemented inline in process()
}

void Reverb::processAllpass(float* /*buffer*/, int /*allpassIndex*/, int /*channel*/, int /*numSamples*/) {
    // Implemented inline in process()
}

} // namespace DSP
} // namespace MolinAntro
