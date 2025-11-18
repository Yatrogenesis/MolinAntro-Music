// AITests.cpp - Comprehensive tests for AI features
// MolinAntro DAW ACME Edition v3.0.0

#include <gtest/gtest.h>
#include "ai/VoiceCloning.h"
#include "ai/AIMastering.h"
#include "ai/MusicalAnalysis.h"
#include "ai/GPUAccelerator.h"
#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"

using namespace MolinAntro;

// ============================================================================
// Voice Cloning Tests
// ============================================================================

class VoiceCloningTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test audio (440 Hz sine wave)
        testAudio = std::make_unique<Core::AudioBuffer>(1, 48000);
        float* samples = testAudio->getWritePointer(0);

        for (int i = 0; i < testAudio->getNumSamples(); ++i) {
            float t = static_cast<float>(i) / 48000.0f;
            samples[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);
        }
    }

    std::unique_ptr<Core::AudioBuffer> testAudio;
};

TEST_F(VoiceCloningTest, ExtractFeatures) {
    AI::RVCVoiceCloner cloner;

    auto features = cloner.extractFeatures(*testAudio);

    EXPECT_GT(features.size(), 0);
    std::cout << "Extracted " << features.size() << " features" << std::endl;
}

TEST_F(VoiceCloningTest, ExtractPitch) {
    AI::RVCVoiceCloner cloner;

    auto pitch = cloner.extractPitch(*testAudio);

    ASSERT_GT(pitch.size(), 0);

    // Check if detected pitch is close to 440 Hz
    float avgPitch = 0.0f;
    int count = 0;
    for (float f : pitch) {
        if (f > 0.0f) {
            avgPitch += f;
            count++;
        }
    }
    avgPitch /= count;

    EXPECT_NEAR(avgPitch, 440.0f, 50.0f); // Within 50 Hz tolerance
    std::cout << "Detected pitch: " << avgPitch << " Hz" << std::endl;
}

TEST_F(VoiceCloningTest, TrainAndLoadModel) {
    AI::RVCVoiceCloner cloner;

    AI::RVCVoiceCloner::TrainingConfig config;
    config.epochs = 1; // Quick test
    config.quality = AI::RVCVoiceCloner::TrainingConfig::Quality::Fast;

    bool success = cloner.trainModel(*testAudio, config, "/tmp/test_voice.rvc",
        [](float progress, const std::string& msg) {
            std::cout << "[" << static_cast<int>(progress * 100) << "%] " << msg << std::endl;
        });

    EXPECT_TRUE(success);

    // Load model
    auto model = cloner.loadModel("/tmp/test_voice.rvc");
    EXPECT_FALSE(model.modelPath.empty());
    EXPECT_GT(model.speakerEmbedding.size(), 0);
}

TEST_F(VoiceCloningTest, VoiceConversion) {
    AI::RVCVoiceCloner cloner;

    // Create and load model
    AI::RVCVoiceCloner::TrainingConfig config;
    config.epochs = 1;
    cloner.trainModel(*testAudio, config, "/tmp/test_voice2.rvc");

    auto model = cloner.loadModel("/tmp/test_voice2.rvc");

    // Convert voice
    AI::RVCVoiceCloner::ConversionSettings settings;
    settings.pitchShift = 2.0f; // +2 semitones

    auto converted = cloner.convert(*testAudio, model, settings);

    EXPECT_EQ(converted.getNumSamples(), testAudio->getNumSamples());
    EXPECT_GT(converted.getRMSLevel(0), 0.0f);
}

TEST_F(VoiceCloningTest, TTSEngine) {
    AI::TTSEngine tts;

    auto voices = tts.getAvailableVoices();
    EXPECT_GT(voices.size(), 0);

    AI::TTSEngine::ProsodySettings prosody;
    prosody.speed = 1.0f;
    prosody.pitch = 1.0f;

    auto speech = tts.synthesize("Hello world", voices[0], prosody);

    EXPECT_GT(speech.getNumSamples(), 0);
    EXPECT_GT(speech.getRMSLevel(0), 0.0f);
    std::cout << "Generated TTS audio: " << speech.getNumSamples() << " samples" << std::endl;
}

TEST_F(VoiceCloningTest, VocalSynthesizer) {
    AI::VocalSynthesizer synth;

    // Create MIDI melody
    std::vector<MIDI::Note> melody = {
        {60, 100, 0.0f, 0.5f, 0}, // C4
        {64, 100, 0.5f, 0.5f, 0}, // E4
        {67, 100, 1.0f, 0.5f, 0}, // G4
        {72, 100, 1.5f, 1.0f, 0}  // C5
    };

    AI::VocalSynthesizer::Expression expr;
    auto vocal = synth.synthesize(melody, "La la la la",
                                  AI::VocalSynthesizer::VoiceStyle::Pop_Female,
                                  expr);

    EXPECT_GT(vocal.getNumSamples(), 0);
    EXPECT_GT(vocal.getRMSLevel(0), 0.0f);
    std::cout << "Generated vocal: " << vocal.getNumSamples() << " samples" << std::endl;
}

// ============================================================================
// AI Mastering Tests
// ============================================================================

class AIMasteringTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test mix (sine wave)
        testMix = std::make_unique<Core::AudioBuffer>(2, 48000);

        for (int ch = 0; ch < 2; ++ch) {
            float* samples = testMix->getWritePointer(ch);
            for (int i = 0; i < testMix->getNumSamples(); ++i) {
                float t = static_cast<float>(i) / 48000.0f;
                samples[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * t);
            }
        }
    }

    std::unique_ptr<Core::AudioBuffer> testMix;
};

TEST_F(AIMasteringTest, MixAnalysis) {
    AI::AIMasteringEngine engine;

    auto analysis = engine.analyze(*testMix);

    std::cout << "LUFS: " << analysis.integratedLUFS << " dB" << std::endl;
    std::cout << "True Peak: " << analysis.truePeak << " dBFS" << std::endl;
    std::cout << "Dynamic Range: " << analysis.dynamicRange << " dB" << std::endl;

    EXPECT_LT(analysis.integratedLUFS, 0.0f); // Should be negative
    EXPECT_LT(analysis.truePeak, 0.0f);       // Should be below 0 dBFS
}

TEST_F(AIMasteringTest, AutoMastering) {
    AI::AIMasteringEngine engine;

    AI::AIMasteringEngine::MasteringSettings settings;
    settings.genre = "Rock";
    settings.targetLUFS = -14.0f; // Spotify standard
    settings.targetTruePeak = -1.0f;

    auto mastered = engine.master(*testMix, settings);

    EXPECT_EQ(mastered.getNumSamples(), testMix->getNumSamples());

    auto analysis = engine.analyze(mastered);
    EXPECT_NEAR(analysis.integratedLUFS, -14.0f, 3.0f); // Within 3 dB

    std::cout << "Mastered LUFS: " << analysis.integratedLUFS << " dB" << std::endl;
}

TEST_F(AIMasteringTest, NeuralPitchCorrection) {
    AI::NeuralPitchCorrector corrector;

    // Create vocal-like signal
    Core::AudioBuffer vocal(1, 48000);
    float* samples = vocal.getWritePointer(0);

    for (int i = 0; i < vocal.getNumSamples(); ++i) {
        float t = static_cast<float>(i) / 48000.0f;
        // Slightly detuned (262 Hz instead of 261.63 Hz = C4)
        samples[i] = 0.5f * std::sin(2.0f * M_PI * 262.0f * t);
    }

    auto analysis = corrector.analyzePitch(vocal);

    EXPECT_GT(analysis.pitchCurve.size(), 0);
    EXPECT_GT(analysis.avgPitch, 0.0f);

    std::cout << "Average pitch: " << analysis.avgPitch << " Hz" << std::endl;

    // Correct pitch
    AI::NeuralPitchCorrector::CorrectionSettings settings;
    settings.strength = 100.0f;

    auto corrected = corrector.correct(vocal, settings);

    EXPECT_EQ(corrected.getNumSamples(), vocal.getNumSamples());
}

TEST_F(AIMasteringTest, HarmonyGeneration) {
    AI::NeuralPitchCorrector corrector;

    Core::AudioBuffer vocal(1, 24000);
    float* samples = vocal.getWritePointer(0);

    for (int i = 0; i < vocal.getNumSamples(); ++i) {
        float t = static_cast<float>(i) / 48000.0f;
        samples[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);
    }

    auto harmonies = corrector.generateHarmonies(vocal, "C-Am-F-G", 2);

    EXPECT_EQ(harmonies.size(), 2);

    for (size_t i = 0; i < harmonies.size(); ++i) {
        EXPECT_GT(harmonies[i].getNumSamples(), 0);
        std::cout << "Harmony " << i << ": " << harmonies[i].getRMSLevel(0) << " RMS" << std::endl;
    }
}

// ============================================================================
// Musical Analysis Tests
// ============================================================================

class MusicalAnalysisTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test audio with C major chord (C-E-G)
        testAudio = std::make_unique<Core::AudioBuffer>(1, 48000);
        float* samples = testAudio->getWritePointer(0);

        for (int i = 0; i < testAudio->getNumSamples(); ++i) {
            float t = static_cast<float>(i) / 48000.0f;
            samples[i] = 0.3f * (
                std::sin(2.0f * M_PI * 261.63f * t) + // C4
                std::sin(2.0f * M_PI * 329.63f * t) + // E4
                std::sin(2.0f * M_PI * 392.00f * t)   // G4
            );
        }
    }

    std::unique_ptr<Core::AudioBuffer> testAudio;
};

TEST_F(MusicalAnalysisTest, ChordDetection) {
    AI::ChordDetector detector;

    auto chords = detector.detectChords(*testAudio, 1.0f);

    EXPECT_GT(chords.size(), 0);

    for (const auto& chord : chords) {
        std::cout << "Chord: " << chord.name
                  << " at " << chord.startTime << "s"
                  << " (confidence: " << chord.confidence << ")"
                  << std::endl;
    }
}

TEST_F(MusicalAnalysisTest, BeatDetection) {
    AI::BeatAnalyzer analyzer;

    // Create rhythmic audio
    Core::AudioBuffer rhythm(1, 96000); // 2 seconds
    float* samples = rhythm.getWritePointer(0);

    for (int beat = 0; beat < 8; ++beat) {
        int startSample = beat * 12000; // 120 BPM
        for (int i = 0; i < 1000; ++i) {
            if (startSample + i < rhythm.getNumSamples()) {
                samples[startSample + i] = 0.8f * std::exp(-i / 100.0f);
            }
        }
    }

    auto beatMap = analyzer.analyze(rhythm, 0.5f);

    std::cout << "Detected BPM: " << beatMap.globalBPM << std::endl;
    std::cout << "Number of beats: " << beatMap.beatTimes.size() << std::endl;

    EXPECT_GT(beatMap.beatTimes.size(), 0);
    EXPECT_GT(beatMap.globalBPM, 0.0f);
}

TEST_F(MusicalAnalysisTest, KeyDetection) {
    AI::KeyDetector detector;

    auto key = detector.detect(*testAudio);

    std::cout << "Detected key: " << key.tonic << " " << key.mode
              << " (confidence: " << key.confidence << ")"
              << std::endl;

    EXPECT_FALSE(key.tonic.empty());
    EXPECT_GT(key.confidence, 0.0f);
}

TEST_F(MusicalAnalysisTest, MelodyExtraction) {
    AI::MelodyExtractor extractor;

    auto melody = extractor.extract(*testAudio);

    EXPECT_GT(melody.notes.size(), 0);

    std::cout << "Extracted " << melody.notes.size() << " notes" << std::endl;
}

TEST_F(MusicalAnalysisTest, AudioToMIDI) {
    AI::AudioToMIDI transcriber;

    AI::AudioToMIDI::Settings settings;
    settings.instrumentType = "piano";
    settings.polyphonic = true;

    auto sequence = transcriber.transcribe(*testAudio, settings);

    EXPECT_GT(sequence.getNotes().size(), 0);

    std::cout << "Transcribed " << sequence.getNotes().size() << " MIDI notes" << std::endl;
}

// ============================================================================
// GPU Accelerator Tests
// ============================================================================

class GPUAcceleratorTest : public ::testing::Test {
protected:
    void SetUp() override {
        gpu.initialize(AI::GPUAccelerator::detectBestBackend());
    }

    AI::GPUAccelerator gpu;
};

TEST_F(GPUAcceleratorTest, Initialization) {
    auto backend = gpu.getBackend();
    auto deviceInfo = gpu.getDeviceInfo();

    std::cout << "Backend: ";
    switch (backend) {
        case AI::GPUAccelerator::Backend::CPU:
            std::cout << "CPU"; break;
        case AI::GPUAccelerator::Backend::CUDA:
            std::cout << "CUDA"; break;
        case AI::GPUAccelerator::Backend::Metal:
            std::cout << "Metal"; break;
        case AI::GPUAccelerator::Backend::OpenCL:
            std::cout << "OpenCL"; break;
    }
    std::cout << std::endl;

    std::cout << "Device: " << deviceInfo.name << std::endl;
}

TEST_F(GPUAcceleratorTest, FFT) {
    const int size = 1024;
    std::vector<float> input(size);
    std::vector<std::complex<float>> output(size);

    // Create sine wave
    for (int i = 0; i < size; ++i) {
        input[i] = std::sin(2.0f * M_PI * 10.0f * i / size);
    }

    gpu.fft(input.data(), output.data(), size);

    // Check output is non-zero
    float magnitude = 0.0f;
    for (const auto& val : output) {
        magnitude += std::abs(val);
    }

    EXPECT_GT(magnitude, 0.0f);
    std::cout << "FFT magnitude sum: " << magnitude << std::endl;
}

TEST_F(GPUAcceleratorTest, Convolution) {
    std::vector<float> signal = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> kernel = {0.5f, 0.5f};
    std::vector<float> output(signal.size() + kernel.size() - 1);

    gpu.convolve(signal.data(), kernel.data(), output.data(),
                signal.size(), kernel.size());

    EXPECT_GT(output[0], 0.0f);
    std::cout << "Convolution result[0]: " << output[0] << std::endl;
}

TEST_F(GPUAcceleratorTest, MatrixMultiplication) {
    const int M = 3, N = 3, K = 3;

    std::vector<float> A = {1, 2, 3,  4, 5, 6,  7, 8, 9};
    std::vector<float> B = {1, 0, 0,  0, 1, 0,  0, 0, 1}; // Identity
    std::vector<float> C(M * N);

    gpu.matmul(A.data(), B.data(), C.data(), M, N, K);

    // C should equal A (identity multiplication)
    for (int i = 0; i < M * N; ++i) {
        EXPECT_FLOAT_EQ(C[i], A[i]);
    }
}

TEST_F(GPUAcceleratorTest, PerformanceStats) {
    std::vector<float> input(1024);
    std::vector<std::complex<float>> output(1024);

    gpu.resetStats();

    for (int i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i);
    }

    gpu.fft(input.data(), output.data(), 1024);

    auto stats = gpu.getStats();

    EXPECT_GT(stats.operationCount, 0);
    EXPECT_GT(stats.lastOperationTime, 0.0f);

    std::cout << "Last operation: " << stats.lastOperationTime << " ms" << std::endl;
    std::cout << "Operations: " << stats.operationCount << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
