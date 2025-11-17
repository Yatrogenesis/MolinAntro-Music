#include "instruments/Synthesizer.h"
#include <algorithm>
#include <random>
#include <iostream>

namespace MolinAntro {
namespace Instruments {

// ============================================================================
// ADSR Implementation
// ============================================================================

float Synthesizer::ADSR::process(float& level, bool gate, float sampleRate) {
    const float attackRate = 1.0f / (attack * sampleRate);
    const float decayRate = 1.0f / (decay * sampleRate);
    const float releaseRate = 1.0f / (release * sampleRate);

    if (gate) {
        if (level < 1.0f) {
            // Attack phase
            level += attackRate;
            if (level >= 1.0f) {
                level = 1.0f;
            }
        } else if (level > sustain) {
            // Decay phase
            level -= decayRate;
            if (level < sustain) {
                level = sustain;
            }
        }
    } else {
        // Release phase
        if (level > 0.0f) {
            level -= releaseRate;
            if (level < 0.0f) {
                level = 0.0f;
            }
        }
    }

    return level;
}

// ============================================================================
// Synthesizer Implementation
// ============================================================================

Synthesizer::Synthesizer() {
    std::cout << "[Synthesizer] Constructed" << std::endl;

    // Initialize templates with good defaults
    osc1Template_.waveform = Waveform::Saw;
    osc1Template_.level = 0.7f;
    osc1Template_.pitch = 0.0f;
    osc1Template_.detune = 0.0f;

    osc2Template_.waveform = Waveform::Saw;
    osc2Template_.level = 0.7f;
    osc2Template_.pitch = 12.0f; // One octave up
    osc2Template_.detune = 5.0f; // Slight detune

    subLevel_ = 0.3f;
    noiseLevel_ = 0.0f;

    filterTemplate_.type = Filter::Type::LowPass;
    filterTemplate_.cutoff = 2000.0f;
    filterTemplate_.resonance = 0.3f;
    filterTemplate_.envAmount = 0.5f;

    ampEnvTemplate_.attack = 0.01f;
    ampEnvTemplate_.decay = 0.1f;
    ampEnvTemplate_.sustain = 0.7f;
    ampEnvTemplate_.release = 0.3f;

    filterEnvTemplate_.attack = 0.05f;
    filterEnvTemplate_.decay = 0.2f;
    filterEnvTemplate_.sustain = 0.5f;
    filterEnvTemplate_.release = 0.5f;

    lfo1_.waveform = Waveform::Sine;
    lfo1_.rate = 4.0f;
    lfo1_.amount = 0.2f;
    lfo1_.enabled = false;
}

void Synthesizer::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    std::cout << "[Synthesizer] Prepared: " << sampleRate << " Hz, "
              << maxVoices_ << " voices" << std::endl;

    reset();
}

void Synthesizer::process(Core::AudioBuffer& buffer) {
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    // Clear buffer
    buffer.clear();

    // Process each active voice
    for (int v = 0; v < MAX_VOICES; ++v) {
        Voice& voice = voices_[v];
        if (!voice.active) continue;

        for (int i = 0; i < numSamples; ++i) {
            float sample = voice.process(*this, sampleRate_);

            // Add to output (mono to stereo)
            if (numChannels > 0) {
                buffer.getWritePointer(0)[i] += sample;
            }
            if (numChannels > 1) {
                buffer.getWritePointer(1)[i] += sample;
            }
        }

        // Deactivate voice if envelope finished
        if (!voice.gate && voice.ampEnvLevel <= 0.0f) {
            voice.active = false;
        }
    }

    // Normalize to prevent clipping with multiple voices
    const float normalization = 1.0f / std::sqrt(static_cast<float>(getActiveVoiceCount() + 1));
    buffer.applyGain(0, normalization);
    if (numChannels > 1) {
        buffer.applyGain(1, normalization);
    }
}

void Synthesizer::reset() {
    for (auto& voice : voices_) {
        voice.active = false;
        voice.gate = false;
        voice.ampEnvLevel = 0.0f;
        voice.filterEnvLevel = 0.0f;
        voice.osc1.phase = 0.0f;
        voice.osc2.phase = 0.0f;

        // Reset filter state
        for (int i = 0; i < 4; ++i) voice.filter.stage[i] = 0.0f;
        for (int i = 0; i < 3; ++i) voice.filter.stageTanh[i] = 0.0f;
        for (int i = 0; i < 6; ++i) voice.filter.delay[i] = 0.0f;
    }

    lfo1_.phase = 0.0f;
    lfo2_.phase = 0.0f;
}

void Synthesizer::noteOn(int note, float velocity) {
    Voice* voice = allocateVoice(note);
    if (!voice) {
        std::cerr << "[Synthesizer] No voices available" << std::endl;
        return;
    }

    voice->note = note;
    voice->velocity = velocity;
    voice->active = true;
    voice->gate = true;

    // Initialize oscillators from templates
    voice->osc1 = osc1Template_;
    voice->osc2 = osc2Template_;
    voice->subLevel = subLevel_;
    voice->noiseLevel = noiseLevel_;

    // Initialize envelopes
    voice->ampEnv = ampEnvTemplate_;
    voice->filterEnv = filterEnvTemplate_;
    voice->ampEnvLevel = 0.0f;
    voice->filterEnvLevel = 0.0f;

    // Initialize filter
    voice->filter = filterTemplate_;

    std::cout << "[Synthesizer] Note ON: " << note << " velocity: " << velocity << std::endl;
}

void Synthesizer::noteOff(int note) {
    Voice* voice = findVoice(note);
    if (voice) {
        voice->gate = false;
        std::cout << "[Synthesizer] Note OFF: " << note << std::endl;
    }
}

void Synthesizer::allNotesOff() {
    for (auto& voice : voices_) {
        voice.gate = false;
    }
    std::cout << "[Synthesizer] All notes off" << std::endl;
}

void Synthesizer::setMaxVoices(int voices) {
    maxVoices_ = std::clamp(voices, 1, MAX_VOICES);
}

int Synthesizer::getActiveVoiceCount() const {
    int count = 0;
    for (const auto& voice : voices_) {
        if (voice.active) count++;
    }
    return count;
}

float Synthesizer::Voice::process(const Synthesizer& synth, float sampleRate) {
    // Process envelopes
    ampEnv.process(ampEnvLevel, gate, sampleRate);
    filterEnv.process(filterEnvLevel, gate, sampleRate);

    // Calculate note frequency
    const float baseFreq = synth.midiNoteToFrequency(note);

    // Oscillator 1
    float osc1Out = 0.0f;
    if (osc1.enabled) {
        const float freq1 = baseFreq * std::pow(2.0f, (osc1.pitch + osc1.detune / 100.0f) / 12.0f);
        osc1Out = synth.generateWaveform(osc1.waveform, osc1.phase) * osc1.level;
        osc1.phase += freq1 / sampleRate;
        if (osc1.phase >= 1.0f) osc1.phase -= 1.0f;
    }

    // Oscillator 2
    float osc2Out = 0.0f;
    if (osc2.enabled) {
        const float freq2 = baseFreq * std::pow(2.0f, (osc2.pitch + osc2.detune / 100.0f) / 12.0f);
        osc2Out = synth.generateWaveform(osc2.waveform, osc2.phase) * osc2.level;
        osc2.phase += freq2 / sampleRate;
        if (osc2.phase >= 1.0f) osc2.phase -= 1.0f;
    }

    // Sub oscillator (one octave down, sine wave)
    float subOut = 0.0f;
    if (subLevel > 0.0f) {
        static float subPhase = 0.0f;
        const float subFreq = baseFreq * 0.5f;
        subOut = std::sin(2.0f * M_PI * subPhase) * subLevel;
        subPhase += subFreq / sampleRate;
        if (subPhase >= 1.0f) subPhase -= 1.0f;
    }

    // Noise generator
    float noiseOut = 0.0f;
    if (noiseLevel > 0.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        noiseOut = dist(gen) * noiseLevel;
    }

    // Mix oscillators
    float sample = (osc1Out + osc2Out + subOut + noiseOut) * velocity;

    // Apply filter (simplified Moog ladder)
    const float cutoffMod = filter.cutoff * (1.0f + filter.envAmount * filterEnvLevel);
    const float cutoffNorm = std::clamp(cutoffMod / (sampleRate * 0.5f), 0.0f, 0.99f);
    const float resonanceMod = filter.resonance * 4.0f;

    // Simple one-pole lowpass (for performance)
    filter.stage[0] += cutoffNorm * (sample - filter.stage[0]);
    filter.stage[1] += cutoffNorm * (filter.stage[0] - filter.stage[1]);
    filter.stage[2] += cutoffNorm * (filter.stage[1] - filter.stage[2]);
    filter.stage[3] += cutoffNorm * (filter.stage[2] - filter.stage[3]);

    sample = filter.stage[3];

    // Apply resonance feedback
    sample += resonanceMod * (sample - filter.delay[0]);
    filter.delay[0] = sample;

    // Apply amplitude envelope
    sample *= ampEnvLevel;

    return sample;
}

float Synthesizer::generateWaveform(Waveform wf, float phase) const {
    switch (wf) {
        case Waveform::Sine:
            return std::sin(2.0f * M_PI * phase);

        case Waveform::Saw:
            return 2.0f * phase - 1.0f;

        case Waveform::Square:
            return (phase < 0.5f) ? 1.0f : -1.0f;

        case Waveform::Triangle:
            return (phase < 0.5f) ? (4.0f * phase - 1.0f) : (3.0f - 4.0f * phase);

        case Waveform::Noise: {
            static std::random_device rd;
            static std::mt19937 gen(rd());
            static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            return dist(gen);
        }

        default:
            return 0.0f;
    }
}

float Synthesizer::midiNoteToFrequency(int note, float pitchBend) const {
    return 440.0f * std::pow(2.0f, (note - 69 + pitchBend) / 12.0f);
}

void Synthesizer::processFilter(Voice& /*voice*/, float& /*sample*/) {
    // Implemented inline in Voice::process() for performance
}

Synthesizer::Voice* Synthesizer::allocateVoice(int /*note*/) {
    // First, try to find an inactive voice
    for (int i = 0; i < maxVoices_; ++i) {
        if (!voices_[i].active) {
            return &voices_[i];
        }
    }

    // If all voices are active, steal the oldest one
    int oldestIndex = 0;
    float lowestLevel = 1.0f;

    for (int i = 0; i < maxVoices_; ++i) {
        if (voices_[i].ampEnvLevel < lowestLevel) {
            lowestLevel = voices_[i].ampEnvLevel;
            oldestIndex = i;
        }
    }

    return &voices_[oldestIndex];
}

Synthesizer::Voice* Synthesizer::findVoice(int note) {
    for (int i = 0; i < MAX_VOICES; ++i) {
        if (voices_[i].active && voices_[i].gate && voices_[i].note == note) {
            return &voices_[i];
        }
    }
    return nullptr;
}

} // namespace Instruments
} // namespace MolinAntro
