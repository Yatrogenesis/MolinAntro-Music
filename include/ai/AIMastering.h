#pragma once

#include "core/AudioBuffer.h"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace MolinAntro {
namespace AI {

/**
 * @brief AI-powered mastering engine
 * Inspired by iZotope Ozone 10
 *
 * Features:
 * - Genre-aware processing
 * - Reference track matching
 * - LUFS normalization (streaming standards)
 * - Intelligent multiband processing
 * - Text prompt control
 */
class AIMasteringEngine {
public:
    /**
     * @brief Mastering configuration
     */
    struct MasteringSettings {
        std::string genre;                  ///< "Rock", "EDM", "Classical", "Jazz", "Hip-Hop"
        std::string targetPlatform;         ///< "Streaming", "CD", "Vinyl", "Broadcast"

        // Loudness targets
        float targetLUFS{-14.0f};          ///< Target integrated LUFS (-23 to -8)
        float targetTruePeak{-1.0f};       ///< True peak limit (dB)
        float targetDynamicRange{8.0f};    ///< Target DR (LU)

        // Processing modes
        enum class Mode {
            Transparent,                    ///< Minimal coloration
            Warm,                          ///< Analog-style warmth
            Modern,                        ///< Loud and bright
            Vintage,                       ///< Tape/vinyl emulation
            Broadcast                      ///< Broadcast standards
        };
        Mode mode{Mode::Modern};

        // References (optional)
        std::vector<std::string> referenceTracks; ///< Paths to reference tracks
        float referenceMatchStrength{0.7f};       ///< How closely to match (0-1)

        // Processing options
        bool enableEQ{true};
        bool enableCompression{true};
        bool enableStereoEnhancement{true};
        bool enableLimiting{true};
        bool enableExciter{true};
    };

    /**
     * @brief Mix analysis results
     */
    struct MixAnalysis {
        // Loudness metrics
        float integratedLUFS{0.0f};
        float shortTermLUFS{0.0f};
        float momentaryLUFS{0.0f};
        float truePeak{0.0f};
        float dynamicRange{0.0f};              ///< PLR (Peak-to-Loudness Ratio)

        // Frequency balance (relative levels)
        std::map<std::string, float> frequencyBalance;  // "sub", "bass", "low-mid", "mid", "high-mid", "presence", "brilliance"

        // Stereo metrics
        float stereoWidth{0.0f};               ///< Average stereo width (0-1)
        float phaseCorrelation{0.0f};          ///< Phase correlation (-1 to +1)

        // Quality indicators
        bool hasClipping{false};
        bool hasDCOffset{false};
        float noiseFloor{0.0f};                ///< dBFS

        // Recommendations
        std::vector<std::string> issues;       ///< Detected problems
        std::vector<std::string> recommendations; ///< Suggested fixes
    };

    AIMasteringEngine();
    ~AIMasteringEngine();

    /**
     * @brief Analyze a mix
     *
     * @param mix Input audio to analyze
     * @return Detailed analysis
     */
    MixAnalysis analyze(const Core::AudioBuffer& mix);

    /**
     * @brief Apply automatic mastering
     *
     * @param mix Input stereo mix
     * @param settings Mastering configuration
     * @return Mastered audio
     */
    Core::AudioBuffer master(const Core::AudioBuffer& mix,
                            const MasteringSettings& settings);

    /**
     * @brief Apply text prompt (iZotope-style)
     *
     * Examples:
     * - "make vocals brighter"
     * - "reduce muddiness"
     * - "add warmth"
     * - "boost presence"
     * - "tighten low end"
     *
     * @param audio Input audio
     * @param prompt Natural language instruction
     * @return Processed audio
     */
    Core::AudioBuffer applyPrompt(const Core::AudioBuffer& audio,
                                  const std::string& prompt);

    /**
     * @brief Match reference track
     *
     * Analyzes reference and applies similar tonal balance
     *
     * @param mix Input mix
     * @param referencePath Path to reference track
     * @param matchStrength How closely to match (0-1)
     * @return Processed audio
     */
    Core::AudioBuffer matchReference(const Core::AudioBuffer& mix,
                                     const std::string& referencePath,
                                     float matchStrength = 0.7f);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Neural pitch correction
 * Beyond traditional Auto-Tune - intent-aware, formant-preserving
 */
class NeuralPitchCorrector {
public:
    /**
     * @brief Correction settings
     */
    struct CorrectionSettings {
        float strength{100.0f};             ///< Correction strength (0-100%)
        float speed{10.0f};                 ///< Attack time (ms)

        // Melodyne-style preservation
        bool preserveVibrato{true};         ///< Keep natural vibrato
        bool preserveExpression{true};      ///< Keep emotional dynamics
        bool preserveFormants{true};        ///< Avoid "chipmunk" effect

        // Scale/key
        std::string scale{"chromatic"};     ///< "chromatic", "major", "minor", "pentatonic"
        std::string key{"C"};               ///< Root note

        // Neural harmony generation
        bool enableHarmony{false};
        std::vector<int> harmonyIntervals;  ///< Semitones: {3, 7} = third + seventh
        std::string chordProgression;       ///< "C-Am-F-G" for chord-aware tuning

        // Performance
        bool lowLatencyMode{false};         ///< <10ms latency (lower quality)
    };

    /**
     * @brief Pitch analysis result
     */
    struct PitchAnalysis {
        std::vector<float> pitchCurve;      ///< Detected pitch (Hz) per frame
        std::vector<float> confidence;      ///< Detection confidence (0-1)
        std::vector<bool> voiced;           ///< Voiced/unvoiced classification
        std::vector<float> vibrato;         ///< Vibrato intensity per frame
        float avgPitch{0.0f};               ///< Average pitch (Hz)
    };

    NeuralPitchCorrector();
    ~NeuralPitchCorrector();

    /**
     * @brief Analyze pitch
     */
    PitchAnalysis analyzePitch(const Core::AudioBuffer& vocal);

    /**
     * @brief Apply pitch correction
     *
     * @param vocal Input vocal track
     * @param settings Correction parameters
     * @param targetNotes Optional MIDI notes to snap to
     * @return Corrected vocal
     */
    Core::AudioBuffer correct(const Core::AudioBuffer& vocal,
                             const CorrectionSettings& settings,
                             const std::vector<MIDI::Note>* targetNotes = nullptr);

    /**
     * @brief Generate harmonies
     *
     * @param vocal Input vocal
     * @param chordProgression Chord sequence (e.g., "C-Am-F-G")
     * @param numVoices Number of harmony voices (1-4)
     * @return Vector of harmony tracks
     */
    std::vector<Core::AudioBuffer> generateHarmonies(
        const Core::AudioBuffer& vocal,
        const std::string& chordProgression,
        int numVoices = 2);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Smart EQ with AI
 */
class SmartEQ {
public:
    /**
     * @brief Instrument-specific auto EQ
     */
    void autoEQ(Core::AudioBuffer& audio,
               const std::string& instrumentType,
               float intensity = 0.7f);

    /**
     * @brief Detect and remove frequency masking
     *
     * Analyzes two tracks and adjusts EQ to reduce masking
     *
     * @param track1 First audio track
     * @param track2 Second audio track
     */
    void removeMasking(Core::AudioBuffer& track1,
                      Core::AudioBuffer& track2);

    /**
     * @brief Apply text prompt
     *
     * Examples:
     * - "make vocals brighter"
     * - "reduce harshness"
     * - "boost bass"
     * - "cut muddiness"
     *
     * @param audio Audio to process
     * @param prompt Natural language instruction
     */
    void applyPrompt(Core::AudioBuffer& audio,
                    const std::string& prompt);

    /**
     * @brief Match EQ to reference
     */
    void matchEQ(Core::AudioBuffer& audio,
                const Core::AudioBuffer& reference,
                float strength = 0.7f);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Smart compression with AI
 */
class SmartCompressor {
public:
    /**
     * @brief Compression style
     */
    enum class Style {
        Transparent,    ///< Invisible gain reduction
        Gentle,         ///< Subtle, musical
        Moderate,       ///< Standard compression
        Aggressive,     ///< Heavy, pumping
        Limiting        ///< Brick-wall style
    };

    /**
     * @brief Auto-compress based on content
     *
     * @param audio Audio to compress
     * @param instrumentType "vocals", "drums", "bass", "guitar", etc.
     * @param style Compression style
     */
    void autoCompress(Core::AudioBuffer& audio,
                     const std::string& instrumentType,
                     Style style = Style::Moderate);

    /**
     * @brief Intelligent multiband compression
     *
     * @param audio Audio to process
     * @param numBands Number of frequency bands (2-6)
     */
    void multibandCompress(Core::AudioBuffer& audio,
                          int numBands = 4);

    /**
     * @brief Sidechain compression (ducking)
     */
    void sidechainCompress(Core::AudioBuffer& audio,
                          const Core::AudioBuffer& sidechain,
                          float amount = 0.7f);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace AI
} // namespace MolinAntro
