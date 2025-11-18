#pragma once

#include "core/AudioBuffer.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace MolinAntro {
namespace AI {

/**
 * @brief RVC (Retrieval-based Voice Conversion) Implementation
 *
 * State-of-the-art voice cloning system using:
 * - HuBERT for feature extraction
 * - Retrieval system for voice matching
 * - Generator network (net_g) for synthesis
 * - RMVPE for pitch estimation
 *
 * Requirements:
 * - ~18 minutes of clean audio for training (minimum)
 * - GPU recommended (CUDA/Metal) for real-time performance
 * - CPU fallback available with ONNX Runtime
 */
class RVCVoiceCloner {
public:
    /**
     * @brief Voice model structure
     */
    struct VoiceModel {
        std::string modelPath;              ///< Path to trained model file
        std::string speakerName;            ///< Name/ID of the speaker
        int sampleRate{48000};              ///< Model sample rate
        std::vector<float> speakerEmbedding; ///< Speaker embedding vector

        // Model metadata
        int trainingDuration{0};            ///< Seconds of training audio
        float quality{0.0f};                ///< Quality score (0-1)
        std::string language;               ///< Primary language

        // Model parameters
        int hopLength{512};                 ///< Hop length for analysis
        int fftSize{2048};                  ///< FFT size
    };

    /**
     * @brief Training configuration
     */
    struct TrainingConfig {
        int epochs{100};
        float learningRate{0.0001f};
        int batchSize{8};

        // Data augmentation
        bool enablePitchAugmentation{true};
        bool enableNoiseAugmentation{true};
        float augmentationStrength{0.3f};

        // Quality settings
        enum class Quality {
            Fast,        ///< Fast training, lower quality
            Balanced,    ///< Balanced speed/quality
            High         ///< Slow training, best quality
        };
        Quality quality{Quality::Balanced};
    };

    /**
     * @brief Conversion settings
     */
    struct ConversionSettings {
        float pitchShift{0.0f};           ///< Semitones (-12 to +12)
        float indexRate{0.75f};           ///< Retrieval feature ratio (0-1)
        float filterRadius{3.0f};         ///< Median filter radius
        float rmsNormalize{0.0f};         ///< RMS normalization (0=off)
        bool preserveFormants{true};      ///< Preserve vocal formants

        // Performance
        int blockSize{512};               ///< Processing block size
        bool useGPU{true};                ///< Enable GPU acceleration
    };

    /**
     * @brief Constructor
     */
    RVCVoiceCloner();
    ~RVCVoiceCloner();

    /**
     * @brief Train a voice model from reference audio
     *
     * @param referenceAudio Clean audio of target voice (min 18 minutes)
     * @param config Training configuration
     * @param outputModelPath Path to save trained model
     * @param progressCallback Progress updates (0.0-1.0)
     * @return true if training succeeded
     */
    bool trainModel(const Core::AudioBuffer& referenceAudio,
                   const TrainingConfig& config,
                   const std::string& outputModelPath,
                   std::function<void(float, const std::string&)> progressCallback = nullptr);

    /**
     * @brief Load a pre-trained voice model
     *
     * @param modelPath Path to .pth or .onnx model file
     * @return VoiceModel structure
     */
    VoiceModel loadModel(const std::string& modelPath);

    /**
     * @brief Convert voice in real-time
     *
     * @param sourceAudio Input audio to convert
     * @param targetVoice Voice model to clone
     * @param settings Conversion settings
     * @return Converted audio buffer
     */
    Core::AudioBuffer convert(const Core::AudioBuffer& sourceAudio,
                             const VoiceModel& targetVoice,
                             const ConversionSettings& settings);

    /**
     * @brief Extract HuBERT features from audio
     *
     * @param audio Input audio buffer
     * @return Feature vector (soft-speech units)
     */
    std::vector<float> extractFeatures(const Core::AudioBuffer& audio);

    /**
     * @brief Extract pitch using RMVPE
     *
     * @param audio Input audio buffer
     * @return Pitch curve (Hz per frame)
     */
    std::vector<float> extractPitch(const Core::AudioBuffer& audio);

    /**
     * @brief Synthesize audio from features and voice model
     *
     * @param features HuBERT features
     * @param pitch Pitch curve
     * @param voice Target voice model
     * @param settings Conversion settings
     * @return Synthesized audio
     */
    Core::AudioBuffer synthesize(const std::vector<float>& features,
                                const std::vector<float>& pitch,
                                const VoiceModel& voice,
                                const ConversionSettings& settings);

    /**
     * @brief Get available voice models
     *
     * @return List of loaded models
     */
    std::vector<VoiceModel> getAvailableModels() const;

    /**
     * @brief Check if GPU acceleration is available
     */
    bool isGPUAvailable() const;

    /**
     * @brief Get performance statistics
     */
    struct PerformanceStats {
        float avgLatency{0.0f};        ///< Average latency (ms)
        float realTimeFactor{0.0f};    ///< <1.0 = real-time capable
        int processedSamples{0};
        bool usingGPU{false};
    };
    PerformanceStats getStats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Text-to-Speech using Tacotron2 + HiFi-GAN
 */
class TTSEngine {
public:
    /**
     * @brief Voice configuration for TTS
     */
    struct Voice {
        std::string voiceID;
        std::string name;
        std::string language;      // "en-US", "es-ES", etc.
        std::string gender;        // "male", "female", "neutral"
        std::string style;         // "news", "casual", "formal"
        std::string modelPath;
    };

    /**
     * @brief Prosody control
     */
    struct ProsodySettings {
        float speed{1.0f};         ///< Speech rate (0.5-2.0)
        float pitch{1.0f};         ///< Pitch multiplier (0.5-2.0)
        float energy{1.0f};        ///< Energy/volume (0.5-2.0)
        float emotion{0.0f};       ///< Emotional expression (-1=sad, 0=neutral, 1=happy)
    };

    TTSEngine();
    ~TTSEngine();

    /**
     * @brief Synthesize speech from text
     *
     * @param text Input text (supports SSML)
     * @param voice Voice configuration
     * @param prosody Prosody settings
     * @return Generated speech audio
     */
    Core::AudioBuffer synthesize(const std::string& text,
                                const Voice& voice,
                                const ProsodySettings& prosody = ProsodySettings());

    /**
     * @brief Get available voices
     */
    std::vector<Voice> getAvailableVoices() const;

    /**
     * @brief Convert text to phonemes
     */
    std::vector<std::string> textToPhonemes(const std::string& text,
                                           const std::string& language);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief AI Vocal Synthesis (Singing Voice)
 * Similar to ACE Studio / Vocaloid
 */
class VocalSynthesizer {
public:
    /**
     * @brief Singing voice styles
     */
    enum class VoiceStyle {
        Pop_Female,
        Pop_Male,
        Rock_Female,
        Rock_Male,
        Opera_Soprano,
        Opera_Tenor,
        Opera_Alto,
        Opera_Bass,
        Jazz_Female,
        Jazz_Male,
        RnB_Female,
        RnB_Male,
        Country_Female,
        Country_Male
    };

    /**
     * @brief Vocal expression parameters
     */
    struct Expression {
        float vibrato{0.5f};        ///< Vibrato intensity (0-1)
        float breathiness{0.3f};    ///< Breath noise (0-1)
        float tension{0.5f};        ///< Vocal tension (0-1)
        float growl{0.0f};          ///< Distortion/growl (0-1)
        float nasality{0.5f};       ///< Nasal resonance (0-1)
        float brightness{0.5f};     ///< Vocal brightness (0-1)
    };

    /**
     * @brief Phoneme with timing
     */
    struct Phoneme {
        std::string symbol;         ///< IPA phoneme symbol
        float duration{0.1f};       ///< Duration in seconds
        int midiNote{60};          ///< MIDI note number
        Expression expression;      ///< Expression parameters
    };

    VocalSynthesizer();
    ~VocalSynthesizer();

    /**
     * @brief Synthesize singing voice from MIDI + lyrics
     *
     * @param midiNotes MIDI note sequence
     * @param lyrics Text lyrics (syllable-aligned)
     * @param style Voice style
     * @param defaultExpression Default expression settings
     * @return Synthesized vocal audio
     */
    Core::AudioBuffer synthesize(const std::vector<MIDI::Note>& midiNotes,
                                const std::string& lyrics,
                                VoiceStyle style,
                                const Expression& defaultExpression = Expression());

    /**
     * @brief Synthesize from phoneme sequence (advanced)
     */
    Core::AudioBuffer synthesizePhonemes(const std::vector<Phoneme>& phonemes,
                                        VoiceStyle style);

    /**
     * @brief Align lyrics to MIDI notes automatically
     */
    std::vector<Phoneme> alignLyrics(const std::vector<MIDI::Note>& notes,
                                     const std::string& lyrics,
                                     const std::string& language = "en");

    /**
     * @brief Get available voice styles
     */
    std::vector<VoiceStyle> getAvailableStyles() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace AI
} // namespace MolinAntro
