#pragma once

#include "core/AudioBuffer.h"
#include "instruments/Sampler.h"
#include <array>
#include <vector>
#include <string>
#include <functional>
#include <memory>

namespace MolinAntro {
namespace Instruments {

/**
 * @brief Professional Drum Machine with 16 pads, pattern sequencing
 *
 * Features:
 * - 16 velocity-sensitive pads
 * - Multi-layered samples per pad (velocity layers)
 * - Pattern sequencer with swing
 * - Real-time effects per pad
 * - Kit management (save/load)
 *
 * Author: F. Molina-Burgos, MolinAntro Technologies
 */
class DrumMachine {
public:
    static constexpr int NUM_PADS = 16;
    static constexpr int MAX_VELOCITY_LAYERS = 8;
    static constexpr int MAX_PATTERN_STEPS = 64;

    /**
     * @brief A single drum pad with sample layers
     */
    struct Pad {
        std::string name = "Empty";
        std::vector<Sample> velocityLayers;  // Samples for different velocities
        float volume = 1.0f;
        float pan = 0.0f;                    // -1.0 (left) to 1.0 (right)
        float pitch = 0.0f;                  // Semitones
        float decay = 1.0f;                  // Envelope decay multiplier
        bool mute = false;
        bool solo = false;
        int midiNote = 36;                   // Default: C1 (kick)
        int group = 0;                       // Choke group (0 = none)

        // Effects per pad
        float lowCut = 20.0f;                // Hz
        float highCut = 20000.0f;            // Hz
        float attack = 0.0f;                 // Seconds
        float release = 0.1f;                // Seconds
    };

    /**
     * @brief A step in the pattern sequencer
     */
    struct PatternStep {
        bool enabled = false;
        float velocity = 1.0f;               // 0.0 - 1.0
        float probability = 1.0f;            // 0.0 - 1.0 (for humanization)
        int flamOffset = 0;                  // Samples for flam effect
    };

    /**
     * @brief A complete drum pattern
     */
    struct Pattern {
        std::string name = "Pattern 1";
        int numSteps = 16;
        int division = 4;                    // Steps per beat (4 = 16th notes)
        float swing = 0.0f;                  // 0.0 - 1.0
        std::array<std::array<PatternStep, MAX_PATTERN_STEPS>, NUM_PADS> steps;
    };

    /**
     * @brief Drum kit preset
     */
    struct Kit {
        std::string name = "Default Kit";
        std::string author;
        std::array<Pad, NUM_PADS> pads;
    };

    DrumMachine();
    ~DrumMachine();

    // ========================================================================
    // Audio Processing
    // ========================================================================

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    // ========================================================================
    // Pad Triggering
    // ========================================================================

    void triggerPad(int padIndex, float velocity = 1.0f);
    void releasePad(int padIndex);
    void noteOn(int midiNote, float velocity);
    void noteOff(int midiNote);
    void allNotesOff();

    // ========================================================================
    // Pattern Sequencer
    // ========================================================================

    void setPlaying(bool playing);
    bool isPlaying() const { return playing_; }

    void setTempo(float bpm);
    float getTempo() const { return tempo_; }

    void setCurrentPattern(int index);
    int getCurrentPattern() const { return currentPattern_; }

    Pattern& getPattern(int index);
    const Pattern& getPattern(int index) const;

    void toggleStep(int padIndex, int stepIndex);
    void setStepVelocity(int padIndex, int stepIndex, float velocity);
    void clearPattern(int patternIndex);

    // ========================================================================
    // Kit Management
    // ========================================================================

    bool loadKit(const std::string& path);
    bool saveKit(const std::string& path) const;
    void loadDefaultKit();

    Pad& getPad(int index);
    const Pad& getPad(int index) const;

    bool loadSampleToPad(int padIndex, const std::string& filepath, int velocityLayer = 0);
    void clearPad(int padIndex);

    // ========================================================================
    // Configuration
    // ========================================================================

    void setMasterVolume(float volume) { masterVolume_ = volume; }
    float getMasterVolume() const { return masterVolume_; }

    void setSwing(float swing);
    float getSwing() const { return swing_; }

    // ========================================================================
    // Callbacks
    // ========================================================================

    using PadTriggerCallback = std::function<void(int padIndex, float velocity)>;
    using StepCallback = std::function<void(int stepIndex)>;

    void setPadTriggerCallback(PadTriggerCallback callback) { padTriggerCallback_ = callback; }
    void setStepCallback(StepCallback callback) { stepCallback_ = callback; }

private:
    // ========================================================================
    // Voice Management
    // ========================================================================

    struct Voice {
        int padIndex = -1;
        int sampleIndex = 0;
        double playbackPosition = 0.0;
        float velocity = 0.0f;
        float envLevel = 1.0f;
        bool active = false;
        bool releasing = false;

        void reset() {
            padIndex = -1;
            playbackPosition = 0.0;
            velocity = 0.0f;
            envLevel = 1.0f;
            active = false;
            releasing = false;
        }
    };

    static constexpr int MAX_VOICES = 32;
    std::array<Voice, MAX_VOICES> voices_;

    Voice* allocateVoice(int padIndex);
    void processVoice(Voice& voice, float* leftOut, float* rightOut, int numSamples);

    // ========================================================================
    // Sequencer State
    // ========================================================================

    void advanceSequencer(int numSamples);
    void triggerStep(int stepIndex);

    bool playing_ = false;
    float tempo_ = 120.0f;
    float swing_ = 0.0f;
    double samplesPerStep_ = 0.0;
    double stepAccumulator_ = 0.0;
    int currentStep_ = 0;
    int currentPattern_ = 0;

    // ========================================================================
    // Data
    // ========================================================================

    Kit currentKit_;
    std::array<Pattern, 16> patterns_;  // 16 patterns

    int sampleRate_ = 48000;
    int maxBlockSize_ = 512;
    float masterVolume_ = 1.0f;

    // Callbacks
    PadTriggerCallback padTriggerCallback_;
    StepCallback stepCallback_;

    // Random for humanization
    float randomFloat();
};

/**
 * @brief Standard drum pad mappings
 */
namespace DrumPads {
    constexpr int KICK = 0;
    constexpr int SNARE = 1;
    constexpr int CLOSED_HAT = 2;
    constexpr int OPEN_HAT = 3;
    constexpr int LOW_TOM = 4;
    constexpr int MID_TOM = 5;
    constexpr int HIGH_TOM = 6;
    constexpr int CRASH = 7;
    constexpr int RIDE = 8;
    constexpr int CLAP = 9;
    constexpr int RIM = 10;
    constexpr int COWBELL = 11;
    constexpr int PERC_1 = 12;
    constexpr int PERC_2 = 13;
    constexpr int PERC_3 = 14;
    constexpr int PERC_4 = 15;
}

} // namespace Instruments
} // namespace MolinAntro
