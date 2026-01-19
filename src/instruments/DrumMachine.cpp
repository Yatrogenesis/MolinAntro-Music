#include "instruments/DrumMachine.h"
#include "dsp/AudioFile.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

namespace MolinAntro {
namespace Instruments {

// ============================================================================
// Constructor / Destructor
// ============================================================================

DrumMachine::DrumMachine() {
    std::cout << "[DrumMachine] Constructed" << std::endl;
    loadDefaultKit();

    // Initialize patterns
    for (auto& pattern : patterns_) {
        pattern.numSteps = 16;
        pattern.division = 4;
        pattern.swing = 0.0f;
    }
}

DrumMachine::~DrumMachine() = default;

// ============================================================================
// Audio Processing
// ============================================================================

void DrumMachine::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    // Calculate samples per step at current tempo
    float beatsPerSecond = tempo_ / 60.0f;
    float stepsPerSecond = beatsPerSecond * patterns_[currentPattern_].division;
    samplesPerStep_ = sampleRate_ / stepsPerSecond;

    std::cout << "[DrumMachine] Prepared: " << sampleRate << " Hz, "
              << tempo_ << " BPM" << std::endl;

    reset();
}

void DrumMachine::process(Core::AudioBuffer& buffer) {
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    if (numChannels < 2) return;

    float* leftOut = buffer.getWritePointer(0);
    float* rightOut = buffer.getWritePointer(1);

    // Process sequencer
    if (playing_) {
        advanceSequencer(numSamples);
    }

    // Process all active voices
    for (auto& voice : voices_) {
        if (voice.active) {
            processVoice(voice, leftOut, rightOut, numSamples);
        }
    }

    // Apply master volume
    if (masterVolume_ != 1.0f) {
        buffer.applyGain(0, masterVolume_);
        buffer.applyGain(1, masterVolume_);
    }
}

void DrumMachine::reset() {
    for (auto& voice : voices_) {
        voice.reset();
    }
    stepAccumulator_ = 0.0;
    currentStep_ = 0;
}

// ============================================================================
// Voice Processing
// ============================================================================

DrumMachine::Voice* DrumMachine::allocateVoice(int padIndex) {
    // First, find inactive voice
    for (auto& voice : voices_) {
        if (!voice.active) {
            return &voice;
        }
    }

    // Check for choke group - stop voices in same group
    int group = currentKit_.pads[padIndex].group;
    if (group > 0) {
        for (auto& voice : voices_) {
            if (voice.active && voice.padIndex >= 0) {
                if (currentKit_.pads[voice.padIndex].group == group) {
                    voice.reset();
                    return &voice;
                }
            }
        }
    }

    // Steal oldest voice
    return &voices_[0];
}

void DrumMachine::processVoice(Voice& voice, float* leftOut, float* rightOut, int numSamples) {
    if (voice.padIndex < 0 || voice.padIndex >= NUM_PADS) {
        voice.reset();
        return;
    }

    const Pad& pad = currentKit_.pads[voice.padIndex];

    if (pad.mute || pad.velocityLayers.empty()) {
        voice.reset();
        return;
    }

    // Get the sample for this velocity layer
    const Sample& sample = pad.velocityLayers[std::min(voice.sampleIndex,
                                               static_cast<int>(pad.velocityLayers.size()) - 1)];

    if (sample.data.empty()) {
        voice.reset();
        return;
    }

    const int numChannels = sample.numChannels;
    const size_t totalSamples = sample.data.size() / numChannels;

    // Calculate pitch shift
    float pitchRatio = std::pow(2.0f, pad.pitch / 12.0f);

    // Calculate pan gains
    float panAngle = (pad.pan + 1.0f) * 0.5f * 3.14159f * 0.5f;
    float leftGain = std::cos(panAngle);
    float rightGain = std::sin(panAngle);

    for (int i = 0; i < numSamples; ++i) {
        // Check if we've reached the end
        if (voice.playbackPosition >= totalSamples) {
            voice.reset();
            return;
        }

        // Process envelope
        if (voice.releasing) {
            voice.envLevel -= 1.0f / (pad.release * sampleRate_);
            if (voice.envLevel <= 0.0f) {
                voice.reset();
                return;
            }
        } else {
            // Apply decay
            voice.envLevel *= std::pow(pad.decay, 1.0f / sampleRate_);
        }

        // Linear interpolation for pitch shifting
        size_t pos0 = static_cast<size_t>(voice.playbackPosition);
        size_t pos1 = pos0 + 1;
        float frac = static_cast<float>(voice.playbackPosition - pos0);

        if (pos1 >= totalSamples) pos1 = pos0;

        float sampleLeft, sampleRight;

        if (numChannels == 2) {
            // Stereo sample
            float l0 = sample.data[pos0 * 2];
            float l1 = sample.data[pos1 * 2];
            float r0 = sample.data[pos0 * 2 + 1];
            float r1 = sample.data[pos1 * 2 + 1];

            sampleLeft = l0 + frac * (l1 - l0);
            sampleRight = r0 + frac * (r1 - r0);
        } else {
            // Mono sample
            float s0 = sample.data[pos0];
            float s1 = sample.data[pos1];
            float mono = s0 + frac * (s1 - s0);
            sampleLeft = mono;
            sampleRight = mono;
        }

        // Apply velocity, envelope, and volume
        float gain = voice.velocity * voice.envLevel * pad.volume;

        // Apply pan and add to output
        leftOut[i] += sampleLeft * gain * leftGain;
        rightOut[i] += sampleRight * gain * rightGain;

        // Advance playback position
        voice.playbackPosition += pitchRatio;
    }
}

// ============================================================================
// Pad Triggering
// ============================================================================

void DrumMachine::triggerPad(int padIndex, float velocity) {
    if (padIndex < 0 || padIndex >= NUM_PADS) return;

    const Pad& pad = currentKit_.pads[padIndex];
    if (pad.mute || pad.velocityLayers.empty()) return;

    // Solo check
    bool hasSolo = false;
    for (const auto& p : currentKit_.pads) {
        if (p.solo) { hasSolo = true; break; }
    }
    if (hasSolo && !pad.solo) return;

    // Select velocity layer
    int layerIndex = 0;
    if (pad.velocityLayers.size() > 1) {
        layerIndex = static_cast<int>(velocity * (pad.velocityLayers.size() - 1));
    }

    // Allocate voice
    Voice* voice = allocateVoice(padIndex);
    if (!voice) return;

    voice->padIndex = padIndex;
    voice->sampleIndex = layerIndex;
    voice->playbackPosition = 0.0;
    voice->velocity = velocity;
    voice->envLevel = 1.0f;
    voice->active = true;
    voice->releasing = false;

    // Callback
    if (padTriggerCallback_) {
        padTriggerCallback_(padIndex, velocity);
    }

    std::cout << "[DrumMachine] Trigger pad " << padIndex
              << " (" << pad.name << ") vel: " << velocity << std::endl;
}

void DrumMachine::releasePad(int padIndex) {
    for (auto& voice : voices_) {
        if (voice.active && voice.padIndex == padIndex) {
            voice.releasing = true;
        }
    }
}

void DrumMachine::noteOn(int midiNote, float velocity) {
    // Find pad with matching MIDI note
    for (int i = 0; i < NUM_PADS; ++i) {
        if (currentKit_.pads[i].midiNote == midiNote) {
            triggerPad(i, velocity);
            return;
        }
    }
}

void DrumMachine::noteOff(int midiNote) {
    for (int i = 0; i < NUM_PADS; ++i) {
        if (currentKit_.pads[i].midiNote == midiNote) {
            releasePad(i);
            return;
        }
    }
}

void DrumMachine::allNotesOff() {
    for (auto& voice : voices_) {
        if (voice.active) {
            voice.releasing = true;
        }
    }
    std::cout << "[DrumMachine] All notes off" << std::endl;
}

// ============================================================================
// Pattern Sequencer
// ============================================================================

void DrumMachine::setPlaying(bool playing) {
    playing_ = playing;
    if (!playing) {
        currentStep_ = 0;
        stepAccumulator_ = 0.0;
    }
    std::cout << "[DrumMachine] " << (playing ? "Play" : "Stop") << std::endl;
}

void DrumMachine::setTempo(float bpm) {
    tempo_ = std::clamp(bpm, 20.0f, 300.0f);

    float beatsPerSecond = tempo_ / 60.0f;
    float stepsPerSecond = beatsPerSecond * patterns_[currentPattern_].division;
    samplesPerStep_ = sampleRate_ / stepsPerSecond;

    std::cout << "[DrumMachine] Tempo: " << tempo_ << " BPM" << std::endl;
}

void DrumMachine::setCurrentPattern(int index) {
    if (index >= 0 && index < static_cast<int>(patterns_.size())) {
        currentPattern_ = index;
        setTempo(tempo_); // Recalculate samples per step
    }
}

DrumMachine::Pattern& DrumMachine::getPattern(int index) {
    return patterns_[std::clamp(index, 0, static_cast<int>(patterns_.size()) - 1)];
}

const DrumMachine::Pattern& DrumMachine::getPattern(int index) const {
    return patterns_[std::clamp(index, 0, static_cast<int>(patterns_.size()) - 1)];
}

void DrumMachine::advanceSequencer(int numSamples) {
    stepAccumulator_ += numSamples;

    const Pattern& pattern = patterns_[currentPattern_];

    while (stepAccumulator_ >= samplesPerStep_) {
        triggerStep(currentStep_);

        if (stepCallback_) {
            stepCallback_(currentStep_);
        }

        currentStep_ = (currentStep_ + 1) % pattern.numSteps;
        stepAccumulator_ -= samplesPerStep_;

        // Apply swing to even steps
        if (currentStep_ % 2 == 0 && swing_ > 0.0f) {
            stepAccumulator_ -= samplesPerStep_ * swing_ * 0.5;
        }
    }
}

void DrumMachine::triggerStep(int stepIndex) {
    const Pattern& pattern = patterns_[currentPattern_];

    for (int padIndex = 0; padIndex < NUM_PADS; ++padIndex) {
        const PatternStep& step = pattern.steps[padIndex][stepIndex];

        if (step.enabled) {
            // Apply probability
            if (step.probability < 1.0f) {
                if (randomFloat() > step.probability) {
                    continue;
                }
            }

            triggerPad(padIndex, step.velocity);
        }
    }
}

void DrumMachine::toggleStep(int padIndex, int stepIndex) {
    if (padIndex < 0 || padIndex >= NUM_PADS) return;
    if (stepIndex < 0 || stepIndex >= MAX_PATTERN_STEPS) return;

    Pattern& pattern = patterns_[currentPattern_];
    pattern.steps[padIndex][stepIndex].enabled = !pattern.steps[padIndex][stepIndex].enabled;
}

void DrumMachine::setStepVelocity(int padIndex, int stepIndex, float velocity) {
    if (padIndex < 0 || padIndex >= NUM_PADS) return;
    if (stepIndex < 0 || stepIndex >= MAX_PATTERN_STEPS) return;

    Pattern& pattern = patterns_[currentPattern_];
    pattern.steps[padIndex][stepIndex].velocity = std::clamp(velocity, 0.0f, 1.0f);
}

void DrumMachine::clearPattern(int patternIndex) {
    if (patternIndex < 0 || patternIndex >= static_cast<int>(patterns_.size())) return;

    Pattern& pattern = patterns_[patternIndex];
    for (auto& padSteps : pattern.steps) {
        for (auto& step : padSteps) {
            step.enabled = false;
            step.velocity = 1.0f;
            step.probability = 1.0f;
        }
    }
}

void DrumMachine::setSwing(float swing) {
    swing_ = std::clamp(swing, 0.0f, 1.0f);
    patterns_[currentPattern_].swing = swing_;
}

// ============================================================================
// Kit Management
// ============================================================================

void DrumMachine::loadDefaultKit() {
    currentKit_.name = "Default Kit";
    currentKit_.author = "MolinAntro";

    // Standard GM drum mapping
    const std::array<std::pair<std::string, int>, NUM_PADS> defaultPads = {{
        {"Kick", 36},
        {"Snare", 38},
        {"Closed HH", 42},
        {"Open HH", 46},
        {"Low Tom", 45},
        {"Mid Tom", 47},
        {"High Tom", 50},
        {"Crash", 49},
        {"Ride", 51},
        {"Clap", 39},
        {"Rim", 37},
        {"Cowbell", 56},
        {"Perc 1", 60},
        {"Perc 2", 61},
        {"Perc 3", 62},
        {"Perc 4", 63}
    }};

    for (int i = 0; i < NUM_PADS; ++i) {
        currentKit_.pads[i].name = defaultPads[i].first;
        currentKit_.pads[i].midiNote = defaultPads[i].second;
        currentKit_.pads[i].volume = 1.0f;
        currentKit_.pads[i].pan = 0.0f;
        currentKit_.pads[i].pitch = 0.0f;
        currentKit_.pads[i].decay = 0.999f;
        currentKit_.pads[i].mute = false;
        currentKit_.pads[i].solo = false;
        currentKit_.pads[i].group = 0;
    }

    // Set hi-hat choke group
    currentKit_.pads[DrumPads::CLOSED_HAT].group = 1;
    currentKit_.pads[DrumPads::OPEN_HAT].group = 1;

    std::cout << "[DrumMachine] Default kit loaded" << std::endl;
}

DrumMachine::Pad& DrumMachine::getPad(int index) {
    return currentKit_.pads[std::clamp(index, 0, NUM_PADS - 1)];
}

const DrumMachine::Pad& DrumMachine::getPad(int index) const {
    return currentKit_.pads[std::clamp(index, 0, NUM_PADS - 1)];
}

bool DrumMachine::loadSampleToPad(int padIndex, const std::string& filepath, int velocityLayer) {
    if (padIndex < 0 || padIndex >= NUM_PADS) return false;

    DSP::AudioFile loader;
    if (!loader.load(filepath)) {
        std::cerr << "[DrumMachine] Failed to load sample: " << filepath << std::endl;
        return false;
    }

    Sample sample;
    sample.path = filepath;
    sample.name = filepath.substr(filepath.find_last_of("/\\") + 1);
    sample.numChannels = loader.getNumChannels();
    sample.sampleRate = loader.getSampleRate();
    sample.data = loader.getSamples();
    sample.rootNote = 60;

    Pad& pad = currentKit_.pads[padIndex];

    // Resize velocity layers if needed
    if (velocityLayer >= static_cast<int>(pad.velocityLayers.size())) {
        pad.velocityLayers.resize(velocityLayer + 1);
    }

    pad.velocityLayers[velocityLayer] = std::move(sample);

    std::cout << "[DrumMachine] Loaded sample to pad " << padIndex
              << " layer " << velocityLayer << ": " << filepath << std::endl;

    return true;
}

void DrumMachine::clearPad(int padIndex) {
    if (padIndex < 0 || padIndex >= NUM_PADS) return;

    Pad& pad = currentKit_.pads[padIndex];
    pad.velocityLayers.clear();
    pad.name = "Empty";
}

bool DrumMachine::loadKit(const std::string& /*path*/) {
    // TODO: Implement kit loading (JSON or custom format)
    std::cerr << "[DrumMachine] Kit loading not yet implemented" << std::endl;
    return false;
}

bool DrumMachine::saveKit(const std::string& /*path*/) const {
    // TODO: Implement kit saving
    std::cerr << "[DrumMachine] Kit saving not yet implemented" << std::endl;
    return false;
}

// ============================================================================
// Utility
// ============================================================================

float DrumMachine::randomFloat() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    return dist(gen);
}

} // namespace Instruments
} // namespace MolinAntro
