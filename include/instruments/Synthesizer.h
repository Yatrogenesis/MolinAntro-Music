#pragma once

#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"
#include <vector>
#include <array>
#include <cmath>

namespace MolinAntro {
namespace Instruments {

/**
 * @brief Professional Subtractive Synthesizer
 *
 * Features:
 * - 2 Oscillators + Sub + Noise
 * - Multiple waveforms (Saw, Square, Triangle, Sine)
 * - Moog-style 24dB/oct ladder filter
 * - 2 ADSR envelopes (amp + filter)
 * - 2 LFOs with multiple destinations
 * - Up to 128 voice polyphony
 * - Unison/detune
 */
class Synthesizer {
public:
    enum class Waveform {
        Sine,
        Saw,
        Square,
        Triangle,
        Noise
    };

    struct ADSR {
        float attack = 0.01f;   // seconds
        float decay = 0.1f;
        float sustain = 0.7f;   // 0-1
        float release = 0.3f;

        float process(float& level, bool gate, float sampleRate);
    };

    struct Oscillator {
        Waveform waveform = Waveform::Saw;
        float level = 1.0f;
        float pitch = 0.0f;     // semitones
        float detune = 0.0f;    // cents
        float phase = 0.0f;
        bool enabled = true;
    };

    struct Filter {
        enum class Type {
            LowPass,
            HighPass,
            BandPass,
            Notch
        };

        Type type = Type::LowPass;
        float cutoff = 1000.0f; // Hz
        float resonance = 0.5f; // 0-1
        float envAmount = 0.5f; // -1 to 1

        // Moog ladder filter state
        float stage[4] = {0, 0, 0, 0};
        float stageTanh[3] = {0, 0, 0};
        float delay[6] = {0, 0, 0, 0, 0, 0};
    };

    struct LFO {
        Waveform waveform = Waveform::Sine;
        float rate = 2.0f;      // Hz
        float amount = 0.5f;    // 0-1
        float phase = 0.0f;
        bool enabled = false;
    };

    struct Voice {
        int note = 0;
        float velocity = 0.0f;
        bool active = false;
        bool gate = false;

        Oscillator osc1;
        Oscillator osc2;
        float subLevel = 0.0f;
        float noiseLevel = 0.0f;

        ADSR ampEnv;
        ADSR filterEnv;
        float ampEnvLevel = 0.0f;
        float filterEnvLevel = 0.0f;

        Filter filter;

        float process(const Synthesizer& synth, float sampleRate);
    };

    Synthesizer();
    ~Synthesizer() = default;

    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    // MIDI interface
    void noteOn(int note, float velocity);
    void noteOff(int note);
    void allNotesOff();

    // Oscillator controls
    void setOsc1Waveform(Waveform wf) { osc1Template_.waveform = wf; }
    void setOsc1Level(float level) { osc1Template_.level = level; }
    void setOsc1Pitch(float semitones) { osc1Template_.pitch = semitones; }
    void setOsc1Detune(float cents) { osc1Template_.detune = cents; }

    void setOsc2Waveform(Waveform wf) { osc2Template_.waveform = wf; }
    void setOsc2Level(float level) { osc2Template_.level = level; }
    void setOsc2Pitch(float semitones) { osc2Template_.pitch = semitones; }
    void setOsc2Detune(float cents) { osc2Template_.detune = cents; }

    void setSubLevel(float level) { subLevel_ = level; }
    void setNoiseLevel(float level) { noiseLevel_ = level; }

    // Filter controls
    void setFilterType(Filter::Type type) { filterTemplate_.type = type; }
    void setFilterCutoff(float hz) { filterTemplate_.cutoff = hz; }
    void setFilterResonance(float res) { filterTemplate_.resonance = res; }
    void setFilterEnvAmount(float amount) { filterTemplate_.envAmount = amount; }

    // Envelope controls
    void setAmpAttack(float sec) { ampEnvTemplate_.attack = sec; }
    void setAmpDecay(float sec) { ampEnvTemplate_.decay = sec; }
    void setAmpSustain(float level) { ampEnvTemplate_.sustain = level; }
    void setAmpRelease(float sec) { ampEnvTemplate_.release = sec; }

    void setFilterAttack(float sec) { filterEnvTemplate_.attack = sec; }
    void setFilterDecay(float sec) { filterEnvTemplate_.decay = sec; }
    void setFilterSustain(float level) { filterEnvTemplate_.sustain = level; }
    void setFilterRelease(float sec) { filterEnvTemplate_.release = sec; }

    // LFO controls
    void setLFO1Waveform(Waveform wf) { lfo1_.waveform = wf; }
    void setLFO1Rate(float hz) { lfo1_.rate = hz; }
    void setLFO1Amount(float amount) { lfo1_.amount = amount; }
    void setLFO1Enabled(bool enabled) { lfo1_.enabled = enabled; }

    // Polyphony
    void setMaxVoices(int voices);
    int getActiveVoiceCount() const;

private:
    static const int MAX_VOICES = 128;
    std::array<Voice, MAX_VOICES> voices_;
    int maxVoices_ = 32;

    int sampleRate_ = 48000;
    int maxBlockSize_ = 512;

    // Templates for new voices
    Oscillator osc1Template_;
    Oscillator osc2Template_;
    float subLevel_ = 0.0f;
    float noiseLevel_ = 0.0f;
    Filter filterTemplate_;
    ADSR ampEnvTemplate_;
    ADSR filterEnvTemplate_;
    LFO lfo1_;
    LFO lfo2_;

    // Helper functions
    float generateWaveform(Waveform wf, float phase) const;
    float midiNoteToFrequency(int note, float pitchBend = 0.0f) const;
    void processFilter(Voice& voice, float& sample);

    // Voice allocation
    Voice* allocateVoice(int note);
    Voice* findVoice(int note);
};

} // namespace Instruments
} // namespace MolinAntro
