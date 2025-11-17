#pragma once

#include "core/AudioBuffer.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace MolinAntro {
namespace DSP {

/**
 * @brief Base class for all audio effects
 */
class AudioEffect {
public:
    virtual ~AudioEffect() = default;

    virtual void prepare(int sampleRate, int maxBlockSize) = 0;
    virtual void process(Core::AudioBuffer& buffer) = 0;
    virtual void reset() = 0;

    void setBypass(bool bypass) { bypassed_ = bypass; }
    bool isBypassed() const { return bypassed_; }

protected:
    bool bypassed_ = false;
    int sampleRate_ = 48000;
    int maxBlockSize_ = 512;
};

/**
 * @brief Parametric EQ (4-band)
 */
class ParametricEQ : public AudioEffect {
public:
    enum class FilterType {
        LowShelf,
        HighShelf,
        Peak,
        Notch,
        LowPass,
        HighPass
    };

    struct Band {
        FilterType type = FilterType::Peak;
        float frequency = 1000.0f;  // Hz
        float gain = 0.0f;           // dB
        float Q = 1.0f;              // Quality factor
        bool enabled = true;
    };

    ParametricEQ();
    ~ParametricEQ() override = default;

    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    void setBand(int bandIndex, const Band& band);
    Band getBand(int bandIndex) const;

private:
    static const int NUM_BANDS = 4;
    Band bands_[NUM_BANDS];

    // Biquad filter coefficients
    struct BiquadCoeffs {
        float b0, b1, b2, a1, a2;
    };

    struct BiquadState {
        float x1 = 0, x2 = 0;
        float y1 = 0, y2 = 0;
    };

    BiquadCoeffs coeffs_[NUM_BANDS][2];  // [band][channel]
    BiquadState state_[NUM_BANDS][2];

    void calculateCoefficients(int bandIndex);
    float processSample(float input, int bandIndex, int channel);
};

/**
 * @brief Dynamic Range Compressor
 */
class Compressor : public AudioEffect {
public:
    Compressor();
    ~Compressor() override = default;

    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    // Parameters
    void setThreshold(float dB) { threshold_ = dB; }
    void setRatio(float ratio) { ratio_ = ratio; }
    void setAttack(float ms) { attackTime_ = ms; updateEnvelope(); }
    void setRelease(float ms) { releaseTime_ = ms; updateEnvelope(); }
    void setKnee(float dB) { knee_ = dB; }
    void setMakeupGain(float dB) { makeupGain_ = dB; }

    float getGainReduction() const { return gainReduction_; }

private:
    float threshold_ = -20.0f;  // dB
    float ratio_ = 4.0f;
    float attackTime_ = 10.0f;  // ms
    float releaseTime_ = 100.0f; // ms
    float knee_ = 6.0f;         // dB
    float makeupGain_ = 0.0f;   // dB

    float attackCoeff_ = 0.0f;
    float releaseCoeff_ = 0.0f;
    float envelope_ = 0.0f;
    float gainReduction_ = 0.0f;

    void updateEnvelope();
    float computeGain(float inputLevel);
};

/**
 * @brief Algorithmic Reverb
 */
class Reverb : public AudioEffect {
public:
    Reverb();
    ~Reverb() override = default;

    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    // Parameters
    void setRoomSize(float size) { roomSize_ = std::clamp(size, 0.0f, 1.0f); }
    void setDamping(float damping) { damping_ = std::clamp(damping, 0.0f, 1.0f); }
    void setWetLevel(float wet) { wetLevel_ = std::clamp(wet, 0.0f, 1.0f); }
    void setDryLevel(float dry) { dryLevel_ = std::clamp(dry, 0.0f, 1.0f); }
    void setWidth(float width) { width_ = std::clamp(width, 0.0f, 1.0f); }

private:
    float roomSize_ = 0.5f;
    float damping_ = 0.5f;
    float wetLevel_ = 0.3f;
    float dryLevel_ = 0.7f;
    float width_ = 1.0f;

    // Freeverb-style comb and allpass filters
    static const int NUM_COMBS = 8;
    static const int NUM_ALLPASS = 4;

    std::vector<float> combBuffers_[NUM_COMBS][2];
    std::vector<float> allpassBuffers_[NUM_ALLPASS][2];

    int combIndices_[NUM_COMBS][2];
    int allpassIndices_[NUM_ALLPASS][2];

    float combFilters_[NUM_COMBS][2];

    void processComb(float* buffer, int combIndex, int channel, int numSamples);
    void processAllpass(float* buffer, int allpassIndex, int channel, int numSamples);
};

/**
 * @brief Stereo Delay
 */
class Delay : public AudioEffect {
public:
    Delay();
    ~Delay() override = default;

    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    // Parameters
    void setDelayTime(float ms) { delayTime_ = ms; updateDelay(); }
    void setFeedback(float feedback) { feedback_ = std::clamp(feedback, 0.0f, 0.95f); }
    void setMix(float mix) { mix_ = std::clamp(mix, 0.0f, 1.0f); }
    void setPingPong(bool enable) { pingPong_ = enable; }

private:
    float delayTime_ = 250.0f;  // ms
    float feedback_ = 0.5f;
    float mix_ = 0.5f;
    bool pingPong_ = false;

    std::vector<float> delayBuffer_[2];
    int writeIndex_[2] = {0, 0};
    int delayInSamples_ = 0;

    void updateDelay();
};

/**
 * @brief Limiter (brick-wall)
 */
class Limiter : public AudioEffect {
public:
    Limiter();
    ~Limiter() override = default;

    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    void setThreshold(float dB) { threshold_ = dB; }
    void setRelease(float ms) { releaseTime_ = ms; }
    void setCeiling(float dB) { ceiling_ = dB; }

private:
    float threshold_ = -0.1f;  // dB
    float releaseTime_ = 50.0f; // ms
    float ceiling_ = 0.0f;      // dB

    float envelope_ = 0.0f;
    float releaseCoeff_ = 0.0f;
};

/**
 * @brief Saturation/Distortion
 */
class Saturator : public AudioEffect {
public:
    enum class Mode {
        Soft,
        Hard,
        Tube,
        Tape,
        Digital
    };

    Saturator();
    ~Saturator() override = default;

    void prepare(int sampleRate, int maxBlockSize) override;
    void process(Core::AudioBuffer& buffer) override;
    void reset() override;

    void setDrive(float drive) { drive_ = std::clamp(drive, 0.0f, 100.0f); }
    void setMode(Mode mode) { mode_ = mode; }
    void setMix(float mix) { mix_ = std::clamp(mix, 0.0f, 1.0f); }

private:
    float drive_ = 10.0f;
    Mode mode_ = Mode::Soft;
    float mix_ = 1.0f;

    float processSample(float input);
    float softClip(float input);
    float hardClip(float input);
    float tubeDistortion(float input);
    float tapeSimulation(float input);
};

} // namespace DSP
} // namespace MolinAntro
