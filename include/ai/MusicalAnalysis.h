#pragma once

#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"
#include <string>
#include <vector>
#include <memory>

namespace MolinAntro {
namespace AI {

/**
 * @brief Chord detection and analysis
 */
class ChordDetector {
public:
    /**
     * @brief Detected chord information
     */
    struct Chord {
        std::string name;           ///< Chord name (e.g., "Cmaj7", "Dm", "G7sus4")
        std::string root;           ///< Root note ("C", "D#", etc.)
        std::string quality;        ///< "major", "minor", "diminished", "augmented"
        std::vector<std::string> extensions; ///< "7", "9", "sus4", etc.

        float startTime{0.0f};      ///< Start time in seconds
        float duration{0.0f};       ///< Duration in seconds
        float confidence{0.0f};     ///< Detection confidence (0-1)

        std::vector<int> notes;     ///< MIDI note numbers
        std::vector<float> weights; ///< Note weights/probabilities
    };

    /**
     * @brief Chord progression analysis
     */
    struct Progression {
        std::vector<Chord> chords;
        std::string key;            ///< Detected key (e.g., "C major", "A minor")
        std::string mode;           ///< "major", "minor", "dorian", etc.
        float tempo{120.0f};        ///< Detected tempo (BPM)
        std::string timeSignature{"4/4"};

        // Harmonic analysis
        std::vector<std::string> romanNumerals; ///< "I", "ii", "V7", etc.
        std::vector<std::string> functions;     ///< "tonic", "subdominant", "dominant"
    };

    ChordDetector();
    ~ChordDetector();

    /**
     * @brief Detect chords from audio
     *
     * @param audio Input audio buffer
     * @param minChordDuration Minimum chord duration (seconds)
     * @return Detected chords
     */
    std::vector<Chord> detectChords(const Core::AudioBuffer& audio,
                                    float minChordDuration = 0.2f);

    /**
     * @brief Analyze full chord progression
     */
    Progression analyzeProgression(const Core::AudioBuffer& audio);

    /**
     * @brief Convert chords to MIDI
     *
     * @param chords Chord sequence
     * @param voicing "close", "open", "drop2", "drop3"
     * @return MIDI sequence
     */
    MIDI::Sequence toMIDI(const std::vector<Chord>& chords,
                         const std::string& voicing = "close");

    /**
     * @brief Suggest chord progressions
     *
     * @param key Musical key
     * @param style "pop", "jazz", "classical", "blues"
     * @param numChords Number of chords to generate
     * @return Chord names
     */
    std::vector<std::string> suggestProgression(const std::string& key,
                                               const std::string& style,
                                               int numChords = 4);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Beat detection and tempo analysis
 */
class BeatAnalyzer {
public:
    /**
     * @brief Beat detection result
     */
    struct BeatMap {
        float globalBPM{120.0f};            ///< Global tempo (BPM)
        std::vector<float> beatTimes;       ///< Beat positions (seconds)
        std::vector<float> downbeatTimes;   ///< Downbeat positions (bar starts)
        std::vector<int> beatNumbers;       ///< Beat number within bar (1-4 for 4/4)

        // Time signature changes
        struct TimeSignature {
            float time{0.0f};               ///< Start time
            int numerator{4};               ///< Beats per bar
            int denominator{4};             ///< Beat unit
        };
        std::vector<TimeSignature> timeSignatures;

        // Tempo variations
        struct TempoChange {
            float time{0.0f};
            float bpm{120.0f};
        };
        std::vector<TempoChange> tempoChanges;
        bool hasTempoVariations{false};

        // Rhythm analysis
        float rhythmicComplexity{0.0f};     ///< 0-1 (simple to complex)
        std::string groove;                 ///< "straight", "swing", "shuffle"
    };

    /**
     * @brief Onset detection result
     */
    struct OnsetMap {
        std::vector<float> onsetTimes;      ///< Transient positions
        std::vector<float> onsetStrengths;  ///< Onset strength (0-1)
        std::vector<std::string> onsetTypes; ///< "percussive", "harmonic", "soft"
    };

    BeatAnalyzer();
    ~BeatAnalyzer();

    /**
     * @brief Detect beats in audio
     *
     * @param audio Input audio
     * @param sensitivity Detection sensitivity (0-1)
     * @return Beat map with timing information
     */
    BeatMap analyze(const Core::AudioBuffer& audio,
                   float sensitivity = 0.7f);

    /**
     * @brief Detect onsets (transients)
     */
    OnsetMap detectOnsets(const Core::AudioBuffer& audio);

    /**
     * @brief Time-stretch audio to match target tempo
     *
     * @param audio Input audio
     * @param currentBPM Detected or known BPM
     * @param targetBPM Desired BPM
     * @param preservePitch Keep pitch constant
     * @return Time-stretched audio
     */
    Core::AudioBuffer warpToTempo(const Core::AudioBuffer& audio,
                                  float currentBPM,
                                  float targetBPM,
                                  bool preservePitch = true);

    /**
     * @brief Quantize audio to grid
     *
     * @param audio Input audio
     * @param beatMap Beat timing information
     * @param strength Quantization strength (0-1)
     * @return Quantized audio
     */
    Core::AudioBuffer quantize(const Core::AudioBuffer& audio,
                              const BeatMap& beatMap,
                              float strength = 0.8f);

    /**
     * @brief Extract groove template
     *
     * Captures timing deviations from strict grid
     *
     * @param audio Input audio with groove
     * @param beatMap Beat map
     * @return Groove template (timing offsets per beat)
     */
    std::vector<float> extractGroove(const Core::AudioBuffer& audio,
                                     const BeatMap& beatMap);

    /**
     * @brief Apply groove template
     */
    Core::AudioBuffer applyGroove(const Core::AudioBuffer& audio,
                                 const BeatMap& beatMap,
                                 const std::vector<float>& groove,
                                 float strength = 0.5f);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Key detection
 */
class KeyDetector {
public:
    KeyDetector();
    ~KeyDetector();

    /**
     * @brief Detected key
     */
    struct Key {
        std::string tonic;          ///< Root note ("C", "F#", etc.)
        std::string mode;           ///< "major", "minor", "dorian", etc.
        float confidence{0.0f};     ///< Detection confidence (0-1)

        // Alternative keys
        std::vector<std::pair<std::string, float>> alternatives; ///< (key, confidence)
    };

    /**
     * @brief Detect musical key
     */
    Key detect(const Core::AudioBuffer& audio);

    /**
     * @brief Detect key changes
     */
    struct KeyChange {
        float time{0.0f};
        Key key;
    };
    std::vector<KeyChange> detectModulations(const Core::AudioBuffer& audio);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Melody extraction
 */
class MelodyExtractor {
public:
    MelodyExtractor();
    ~MelodyExtractor();

    /**
     * @brief Extracted melody
     */
    struct Melody {
        std::vector<MIDI::Note> notes;
        float confidence{0.0f};
    };

    /**
     * @brief Extract melody from polyphonic audio
     *
     * Isolates the dominant melodic line
     */
    Melody extract(const Core::AudioBuffer& audio);

    /**
     * @brief Extract melody to MIDI
     */
    MIDI::Sequence toMIDI(const Core::AudioBuffer& audio,
                         int channel = 0);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Audio-to-MIDI transcription
 */
class AudioToMIDI {
public:
    AudioToMIDI();
    ~AudioToMIDI();

    /**
     * @brief Transcription settings
     */
    struct Settings {
        std::string instrumentType; ///< "piano", "guitar", "bass", "drums", "vocals"
        bool polyphonic{true};      ///< Allow multiple simultaneous notes
        float sensitivity{0.7f};    ///< Note detection sensitivity (0-1)
        int minNoteLength{50};      ///< Minimum note length (ms)
    };

    /**
     * @brief Transcribe audio to MIDI
     *
     * @param audio Input audio
     * @param settings Transcription parameters
     * @return MIDI sequence
     */
    MIDI::Sequence transcribe(const Core::AudioBuffer& audio,
                             const Settings& settings);

    /**
     * @brief Transcribe drums to MIDI
     *
     * Maps drum hits to MIDI notes (GM drum map)
     */
    MIDI::Sequence transcribeDrums(const Core::AudioBuffer& audio);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace AI
} // namespace MolinAntro
