#pragma once

#include "core/AudioBuffer.h"
#include <vector>
#include <complex>
#include <memory>
#include <functional>

namespace MolinAntro {
namespace DSP {

/**
 * @brief Spectral Noise Gate - Real-time noise reduction
 * SOTA implementation using spectral subtraction with psychoacoustic masking
 */
class SpectralNoiseGate {
public:
    SpectralNoiseGate();
    ~SpectralNoiseGate();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    // Learn noise profile from current audio
    void learnNoiseProfile(const Core::AudioBuffer& noiseOnly);
    void clearNoiseProfile();

    // Parameters
    void setReduction(float dB) { reductionDb_ = dB; }        // 0 to -60 dB
    void setThreshold(float dB) { thresholdDb_ = dB; }        // Noise floor threshold
    void setSmoothing(float ms) { smoothingMs_ = ms; }        // Temporal smoothing
    void setAttack(float ms) { attackMs_ = ms; }
    void setRelease(float ms) { releaseMs_ = ms; }

    bool hasNoiseProfile() const { return hasProfile_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    float reductionDb_ = -20.0f;
    float thresholdDb_ = -60.0f;
    float smoothingMs_ = 50.0f;
    float attackMs_ = 5.0f;
    float releaseMs_ = 100.0f;
    bool hasProfile_ = false;
};

/**
 * @brief Adaptive Noise Reduction - AI-powered noise profiling
 * Automatically detects and removes background noise
 */
class AdaptiveNoiseReduction {
public:
    enum class Mode {
        Light,          // Minimal processing, preserve quality
        Standard,       // Balanced noise reduction
        Heavy,          // Aggressive noise removal
        Broadcast,      // Optimized for speech/podcast
        Music           // Preserve musical content
    };

    AdaptiveNoiseReduction();
    ~AdaptiveNoiseReduction();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setMode(Mode mode) { mode_ = mode; }
    void setStrength(float strength) { strength_ = std::clamp(strength, 0.0f, 1.0f); }
    void setPreserveTone(bool enable) { preserveTone_ = enable; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Mode mode_ = Mode::Standard;
    float strength_ = 0.5f;
    bool preserveTone_ = true;
};

/**
 * @brief DeClicker - Remove clicks and pops from audio
 * Uses interpolation and spectral repair
 */
class DeClicker {
public:
    DeClicker();
    ~DeClicker();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setSensitivity(float sens) { sensitivity_ = std::clamp(sens, 0.0f, 1.0f); }
    void setMaxClickLength(float ms) { maxClickLengthMs_ = ms; }
    void setRepairMethod(int method) { repairMethod_ = method; }  // 0=interpolate, 1=spectral

    int getClicksDetected() const { return clicksDetected_; }
    int getClicksRepaired() const { return clicksRepaired_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    float sensitivity_ = 0.5f;
    float maxClickLengthMs_ = 10.0f;
    int repairMethod_ = 0;
    int clicksDetected_ = 0;
    int clicksRepaired_ = 0;
};

/**
 * @brief DeClipper - Restore clipped/distorted audio
 * Uses cubic spline interpolation and harmonic regeneration
 */
class DeClipper {
public:
    DeClipper();
    ~DeClipper();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setThreshold(float threshold) { threshold_ = std::clamp(threshold, 0.9f, 1.0f); }
    void setStrength(float strength) { strength_ = std::clamp(strength, 0.0f, 1.0f); }
    void setOversample(int factor) { oversampleFactor_ = std::clamp(factor, 1, 8); }

    float getClippingPercentage() const { return clippingPercent_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    float threshold_ = 0.95f;
    float strength_ = 0.8f;
    int oversampleFactor_ = 4;
    float clippingPercent_ = 0.0f;
};

/**
 * @brief DeHummer - Remove AC hum and harmonics (50/60 Hz)
 * Adaptive notch filter bank with harmonic tracking
 */
class DeHummer {
public:
    enum class Region {
        Auto,           // Detect automatically
        Europe_50Hz,    // 50 Hz fundamental
        USA_60Hz        // 60 Hz fundamental
    };

    DeHummer();
    ~DeHummer();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setRegion(Region region) { region_ = region; }
    void setNumHarmonics(int num) { numHarmonics_ = std::clamp(num, 1, 16); }
    void setNotchWidth(float hz) { notchWidthHz_ = hz; }
    void setReduction(float dB) { reductionDb_ = dB; }

    float getDetectedFrequency() const { return detectedFreq_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Region region_ = Region::Auto;
    int numHarmonics_ = 8;
    float notchWidthHz_ = 2.0f;
    float reductionDb_ = -40.0f;
    float detectedFreq_ = 0.0f;
};

/**
 * @brief DeReverb - Remove or reduce reverb from recordings
 * Uses spectral dereverberation and blind source separation
 */
class DeReverb {
public:
    DeReverb();
    ~DeReverb();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setReduction(float amount) { reduction_ = std::clamp(amount, 0.0f, 1.0f); }
    void setPreserveAmbience(float amount) { preserveAmbience_ = std::clamp(amount, 0.0f, 1.0f); }
    void setTailSuppression(float dB) { tailSuppressionDb_ = dB; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    float reduction_ = 0.5f;
    float preserveAmbience_ = 0.3f;
    float tailSuppressionDb_ = -20.0f;
};

/**
 * @brief DeEsser - Remove sibilance from vocals
 * Multiband dynamics with spectral detection
 */
class DeEsser {
public:
    enum class Mode {
        Broadband,      // Reduce entire signal when sibilance detected
        Multiband,      // Only reduce sibilant frequencies
        Dynamic         // Adaptive frequency tracking
    };

    DeEsser();
    ~DeEsser();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setMode(Mode mode) { mode_ = mode; }
    void setFrequency(float hz) { centerFreq_ = hz; }          // Sibilance center (4-10 kHz)
    void setBandwidth(float octaves) { bandwidth_ = octaves; }
    void setThreshold(float dB) { thresholdDb_ = dB; }
    void setReduction(float dB) { reductionDb_ = dB; }

    float getGainReduction() const { return gainReduction_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Mode mode_ = Mode::Multiband;
    float centerFreq_ = 6000.0f;
    float bandwidth_ = 1.0f;
    float thresholdDb_ = -20.0f;
    float reductionDb_ = -10.0f;
    float gainReduction_ = 0.0f;
};

/**
 * @brief SpeechEnhancer - Improve speech clarity and intelligibility
 * Combines multiple processing stages
 */
class SpeechEnhancer {
public:
    struct Settings {
        float noiseReduction = 0.5f;    // 0-1
        float clarity = 0.5f;           // High frequency boost
        float presence = 0.5f;          // Mid-range enhancement
        float deEssing = 0.3f;          // Sibilance reduction
        float compression = 0.4f;       // Dynamic range control
        bool gateEnabled = true;        // Noise gate
        float gateThreshold = -40.0f;   // dB
    };

    SpeechEnhancer();
    ~SpeechEnhancer();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setSettings(const Settings& settings) { settings_ = settings; }
    Settings getSettings() const { return settings_; }

    // Presets
    void loadPreset(const std::string& name);
    static std::vector<std::string> getPresetNames();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    Settings settings_;
};

/**
 * @brief SpectralRepair - Repair damaged audio using spectral interpolation
 * Can fix gaps, scratches, and corrupted sections
 */
class SpectralRepair {
public:
    enum class RepairMode {
        Interpolate,    // Simple interpolation
        Pattern,        // Pattern-based repair
        AI              // Neural network inpainting
    };

    SpectralRepair();
    ~SpectralRepair();

    void prepare(int sampleRate, int maxBlockSize);

    // Repair a specific region
    void repairRegion(Core::AudioBuffer& buffer, int startSample, int endSample,
                      RepairMode mode = RepairMode::Pattern);

    // Automatic detection and repair
    void autoRepair(Core::AudioBuffer& buffer, float sensitivity = 0.5f);

    // Mark regions for repair
    void markRegion(int startSample, int endSample);
    void clearMarkedRegions();
    void repairMarkedRegions(Core::AudioBuffer& buffer, RepairMode mode);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    std::vector<std::pair<int, int>> markedRegions_;
};

/**
 * @brief AudioRestoration - Complete restoration pipeline
 * Combines all tools for full audio restoration workflow
 */
class AudioRestoration {
public:
    struct Pipeline {
        bool declick = true;
        bool declip = true;
        bool dehum = true;
        bool denoise = true;
        bool dereverb = false;
        bool enhance = true;
    };

    AudioRestoration();
    ~AudioRestoration();

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void setPipeline(const Pipeline& pipeline) { pipeline_ = pipeline; }
    void setQuality(int quality) { quality_ = std::clamp(quality, 1, 5); }  // 1=fast, 5=best

    // Access individual processors for fine-tuning
    DeClicker& getDeClicker() { return *deClicker_; }
    DeClipper& getDeClipper() { return *deClipper_; }
    DeHummer& getDeHummer() { return *deHummer_; }
    SpectralNoiseGate& getNoiseGate() { return *noiseGate_; }
    DeReverb& getDeReverb() { return *deReverb_; }
    SpeechEnhancer& getSpeechEnhancer() { return *speechEnhancer_; }

private:
    Pipeline pipeline_;
    int quality_ = 3;

    std::unique_ptr<DeClicker> deClicker_;
    std::unique_ptr<DeClipper> deClipper_;
    std::unique_ptr<DeHummer> deHummer_;
    std::unique_ptr<SpectralNoiseGate> noiseGate_;
    std::unique_ptr<DeReverb> deReverb_;
    std::unique_ptr<SpeechEnhancer> speechEnhancer_;
};

} // namespace DSP
} // namespace MolinAntro
