// demo_ai.cpp - AI Features Demo
// MolinAntro DAW ACME Edition v3.0.0

#include "ai/AIMastering.h"
#include "ai/GPUAccelerator.h"
#include "ai/MusicalAnalysis.h"
#include "ai/VoiceCloning.h"
#include "core/AudioBuffer.h"
#include "core/AudioEngine.h"
#include "midi/MIDIEngine.h"
#include <iomanip>
#include <iostream>

using namespace MolinAntro;

void printHeader(const std::string &title) {
  std::cout
      << "\n╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║ " << std::left << std::setw(56) << title << " ║\n";
  std::cout
      << "╚══════════════════════════════════════════════════════════╝\n\n";
}

void demoVoiceCloning() {
  printHeader("VOICE CLONING & TTS DEMONSTRATION");

  std::cout << "━━━ [1] RVC Voice Cloner ━━━\n";

  AI::RVCVoiceCloner cloner;

  // Create training audio (1 second of 440 Hz)
  Core::AudioBuffer trainingAudio(1, 48000);
  float *samples = trainingAudio.getWritePointer(0);

  for (int i = 0; i < trainingAudio.getNumSamples(); ++i) {
    float t = static_cast<float>(i) / 48000.0f;
    samples[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);
  }

  std::cout << "✓ Created training audio (1 second @ 440 Hz)\n";

  // Train model
  AI::RVCVoiceCloner::TrainingConfig config;
  config.epochs = 10;
  config.quality = AI::RVCVoiceCloner::TrainingConfig::Quality::Balanced;

  std::cout << "Training voice model...\n";

  bool success = cloner.trainModel(trainingAudio, config, "/tmp/demo_voice.rvc",
                                   [](float progress, const std::string &msg) {
                                     std::cout
                                         << "  [" << std::setw(3)
                                         << static_cast<int>(progress * 100)
                                         << "%] " << msg << std::endl;
                                   });

  if (success) {
    std::cout << "✓ Voice model trained successfully!\n";

    // Load and use model
    auto model = cloner.loadModel("/tmp/demo_voice.rvc");
    std::cout << "✓ Model loaded: " << model.speakerEmbedding.size()
              << " features\n";

    // Extract features
    auto features = cloner.extractFeatures(trainingAudio);
    auto pitch = cloner.extractPitch(trainingAudio);

    std::cout << "✓ Features extracted: " << features.size() << " dimensions\n";
    std::cout << "✓ Pitch extracted: " << pitch.size() << " frames\n";

    // Calculate average pitch
    float avgPitch = 0.0f;
    int count = 0;
    for (float f : pitch) {
      if (f > 0.0f) {
        avgPitch += f;
        count++;
      }
    }
    if (count > 0)
      avgPitch /= count;
    std::cout << "  Average pitch: " << avgPitch << " Hz\n";

    // Voice conversion
    AI::RVCVoiceCloner::ConversionSettings settings;
    settings.pitchShift = 3.5f; // +3.5 semitones

    auto converted = cloner.convert(trainingAudio, model, settings);
    std::cout << "✓ Voice converted with +" << settings.pitchShift
              << " semitones pitch shift\n";
    std::cout << "  Output RMS: " << converted.getRMSLevel(0) << "\n";
  }

  std::cout << "\n━━━ [2] Text-to-Speech Engine ━━━\n";

  AI::TTSEngine tts;

  auto voices = tts.getAvailableVoices();
  std::cout << "✓ Available voices: " << voices.size() << "\n";

  for (const auto &voice : voices) {
    std::cout << "  - " << voice.name << " (" << voice.language << ", "
              << voice.gender << ")\n";
  }

  if (!voices.empty()) {
    AI::TTSEngine::ProsodySettings prosody;
    prosody.speed = 1.0f;
    prosody.pitch = 1.2f;
    prosody.energy = 0.8f;

    auto speech = tts.synthesize("Welcome to MolinAntro DAW ACME Edition",
                                 voices[0], prosody);
    std::cout << "✓ Generated TTS: " << speech.getNumSamples() << " samples\n";
    std::cout << "  Duration: " << speech.getNumSamples() / 48000.0f
              << " seconds\n";
  }

  std::cout << "\n━━━ [3] AI Vocal Synthesizer ━━━\n";

  AI::VocalSynthesizer synth;

  auto styles = synth.getAvailableStyles();
  std::cout << "✓ Available vocal styles: " << styles.size() << "\n";

  // Create melody
  std::vector<MIDI::Note> melody = {
      {60, 100, 0.0f, 0.5f, 0}, // C4
      {62, 100, 0.5f, 0.5f, 0}, // D4
      {64, 100, 1.0f, 0.5f, 0}, // E4
      {65, 100, 1.5f, 0.5f, 0}, // F4
      {67, 100, 2.0f, 1.0f, 0}  // G4
  };

  std::cout << "Synthesizing vocal melody (5 notes)...\n";

  AI::VocalSynthesizer::Expression expr;
  expr.vibrato = 0.6f;
  expr.breathiness = 0.2f;

  auto vocal =
      synth.synthesize(melody, "Do re mi fa sol",
                       AI::VocalSynthesizer::VoiceStyle::Pop_Female, expr);

  std::cout << "✓ Vocal synthesized: " << vocal.getNumSamples() << " samples\n";
  std::cout << "  Duration: " << vocal.getNumSamples() / 48000.0f
            << " seconds\n";
  std::cout << "  RMS level: " << vocal.getRMSLevel(0) << "\n";
}

void demoAIMastering() {
  printHeader("AI MASTERING & PITCH CORRECTION");

  std::cout << "━━━ [1] AI Mastering Engine ━━━\n";

  AI::AIMasteringEngine engine;

  // Create test mix
  Core::AudioBuffer mix(2, 96000); // 2 seconds stereo

  for (int ch = 0; ch < 2; ++ch) {
    float *samples = mix.getWritePointer(ch);
    for (int i = 0; i < mix.getNumSamples(); ++i) {
      float t = static_cast<float>(i) / 48000.0f;
      samples[i] = 0.3f * std::sin(2.0f * M_PI * 440.0f * t);
    }
  }

  std::cout << "Analyzing mix...\n";

  auto analysis = engine.analyze(mix);

  std::cout << "\n  Mix Analysis Results:\n";
  std::cout << "  ├─ Integrated LUFS: " << std::fixed << std::setprecision(2)
            << analysis.integratedLUFS << " dB\n";
  std::cout << "  ├─ True Peak:       " << analysis.truePeak << " dBFS\n";
  std::cout << "  ├─ Dynamic Range:   " << analysis.dynamicRange << " dB\n";
  std::cout << "  ├─ Clipping:        " << (analysis.hasClipping ? "YES" : "NO")
            << "\n";
  std::cout << "  └─ Stereo Width:    " << analysis.stereoWidth << "\n";

  if (!analysis.recommendations.empty()) {
    std::cout << "\n  Recommendations:\n";
    for (const auto &rec : analysis.recommendations) {
      std::cout << "  • " << rec << "\n";
    }
  }

  // Apply mastering
  std::cout << "\nApplying AI mastering...\n";

  AI::AIMasteringEngine::MasteringSettings settings;
  settings.genre = "Pop";
  settings.targetLUFS = -14.0f; // Spotify standard
  settings.targetTruePeak = -1.0f;
  settings.mode = AI::AIMasteringEngine::MasteringSettings::Mode::Modern;

  auto mastered = engine.master(mix, settings);

  auto masteredAnalysis = engine.analyze(mastered);

  std::cout << "✓ Mastering complete!\n";
  std::cout << "  Final LUFS: " << masteredAnalysis.integratedLUFS << " dB\n";
  std::cout << "  Final Peak: " << masteredAnalysis.truePeak << " dBFS\n";

  std::cout << "\n━━━ [2] Neural Pitch Correction ━━━\n";

  AI::NeuralPitchCorrector corrector;

  // Create vocal with pitch variations
  Core::AudioBuffer vocal(1, 48000);
  float *samples = vocal.getWritePointer(0);

  for (int i = 0; i < vocal.getNumSamples(); ++i) {
    float t = static_cast<float>(i) / 48000.0f;
    float freq = 262.0f + 5.0f * std::sin(2.0f * M_PI * 5.0f * t); // Vibrato
    samples[i] = 0.5f * std::sin(2.0f * M_PI * freq * t);
  }

  std::cout << "Analyzing pitch...\n";

  auto pitchAnalysis = corrector.analyzePitch(vocal);

  std::cout << "  Average pitch: " << pitchAnalysis.avgPitch << " Hz\n";
  std::cout << "  Confidence:    " << pitchAnalysis.confidence[0] << "\n";

  // Correct pitch
  AI::NeuralPitchCorrector::CorrectionSettings corrSettings;
  corrSettings.strength = 50.0f; // 50% correction
  corrSettings.preserveVibrato = true;
  corrSettings.preserveFormants = true;

  auto corrected = corrector.correct(vocal, corrSettings);

  std::cout << "✓ Pitch corrected with " << corrSettings.strength
            << "% strength\n";

  // Generate harmonies
  std::cout << "\nGenerating harmonies...\n";

  auto harmonies = corrector.generateHarmonies(vocal, "C-F-G-C", 2);

  std::cout << "✓ Generated " << harmonies.size() << " harmony voices\n";
  for (size_t i = 0; i < harmonies.size(); ++i) {
    std::cout << "  Harmony " << (i + 1) << ": " << harmonies[i].getRMSLevel(0)
              << " RMS\n";
  }
}

void demoMusicalAnalysis() {
  printHeader("MUSICAL ANALYSIS");

  // Create test audio with C major chord
  Core::AudioBuffer audio(1, 96000); // 2 seconds
  float *samples = audio.getWritePointer(0);

  for (int i = 0; i < audio.getNumSamples(); ++i) {
    float t = static_cast<float>(i) / 48000.0f;
    samples[i] = 0.3f * (std::sin(2.0f * M_PI * 261.63f * t) + // C4
                         std::sin(2.0f * M_PI * 329.63f * t) + // E4
                         std::sin(2.0f * M_PI * 392.00f * t)   // G4
                        );
  }

  std::cout << "━━━ [1] Chord Detection ━━━\n";

  AI::ChordDetector chordDetector;

  auto chords = chordDetector.detectChords(audio, 1.0f);

  std::cout << "✓ Detected " << chords.size() << " chord(s):\n";
  for (const auto &chord : chords) {
    std::cout << "  • " << chord.name << " at " << std::fixed
              << std::setprecision(2) << chord.startTime
              << "s (confidence: " << std::setprecision(1)
              << (chord.confidence * 100) << "%)\n";
  }

  // Analyze progression
  auto progression = chordDetector.analyzeProgression(audio);

  std::cout << "\nProgression Analysis:\n";
  std::cout << "  Key:  " << progression.key << "\n";
  std::cout << "  Mode: " << progression.mode << "\n";

  std::cout << "\n━━━ [2] Beat Detection ━━━\n";

  AI::BeatAnalyzer beatAnalyzer;

  // Create rhythmic audio (4/4 at 120 BPM)
  Core::AudioBuffer rhythm(1, 96000);
  float *rhythmSamples = rhythm.getWritePointer(0);

  for (int beat = 0; beat < 8; ++beat) {
    int startSample = beat * 12000; // 120 BPM = 0.5s per beat
    for (int i = 0; i < 1000; ++i) {
      if (startSample + i < rhythm.getNumSamples()) {
        rhythmSamples[startSample + i] = 0.8f * std::exp(-i / 100.0f);
      }
    }
  }

  auto beatMap = beatAnalyzer.analyze(rhythm, 0.5f);

  std::cout << "✓ Tempo detected: " << std::fixed << std::setprecision(1)
            << beatMap.globalBPM << " BPM\n";
  std::cout << "  Beats found: " << beatMap.beatTimes.size() << "\n";
  std::cout << "  Downbeats:   " << beatMap.downbeatTimes.size() << "\n";

  std::cout << "\n━━━ [3] Key Detection ━━━\n";

  AI::KeyDetector keyDetector;

  auto key = keyDetector.detect(audio);

  std::cout << "✓ Detected key: " << key.tonic << " " << key.mode << "\n";
  std::cout << "  Confidence:   " << std::setprecision(1)
            << (key.confidence * 100) << "%\n";

  std::cout << "\n━━━ [4] Audio-to-MIDI Transcription ━━━\n";

  AI::AudioToMIDI transcriber;

  AI::AudioToMIDI::Settings settings;
  settings.instrumentType = "piano";
  settings.polyphonic = true;
  settings.sensitivity = 0.7f;

  auto sequence = transcriber.transcribe(audio, settings);

  std::cout << "✓ Transcribed " << sequence.getNotes().size()
            << " MIDI notes\n";

  if (!sequence.getNotes().empty()) {
    std::cout << "\n  First 3 notes:\n";
    for (size_t i = 0; i < std::min(size_t(3), sequence.getNotes().size());
         ++i) {
      const auto &note = sequence.getNotes()[i];
      std::cout << "  • MIDI " << note.note << " at " << note.timestamp
                << "s, vel=" << note.velocity << "\n";
    }
  }
}

void demoGPUAcceleration() {
  printHeader("GPU ACCELERATION");

  AI::GPUAccelerator gpu;

  // Initialize
  std::cout << "Detecting best backend...\n";

  auto backend = AI::GPUAccelerator::detectBestBackend();

  std::string backendName;
  switch (backend) {
  case AI::GPUAccelerator::Backend::CPU:
    backendName = "CPU (SIMD Optimized)";
    break;
  case AI::GPUAccelerator::Backend::CUDA:
    backendName = "NVIDIA CUDA";
    break;
  case AI::GPUAccelerator::Backend::Metal:
    backendName = "Apple Metal";
    break;
  case AI::GPUAccelerator::Backend::OpenCL:
    backendName = "OpenCL";
    break;
  }

  std::cout << "✓ Selected backend: " << backendName << "\n";

  gpu.initialize(backend);

  auto deviceInfo = gpu.getDeviceInfo();

  std::cout << "\nDevice Information:\n";
  std::cout << "  Device name:    " << deviceInfo.name << "\n";
  std::cout << "  Total memory:   " << (deviceInfo.totalMemory / (1024 * 1024))
            << " MB\n";
  std::cout << "  Free memory:    " << (deviceInfo.freeMemory / (1024 * 1024))
            << " MB\n";
  std::cout << "  Float16 support: "
            << (deviceInfo.supportsFloat16 ? "Yes" : "No") << "\n";
  std::cout << "  Float64 support: "
            << (deviceInfo.supportsFloat64 ? "Yes" : "No") << "\n";

  std::cout << "\n━━━ FFT Benchmark ━━━\n";

  const int fftSize = 8192;
  std::vector<float> input(fftSize);
  std::vector<std::complex<float>> output(fftSize);

  // Generate test signal
  for (int i = 0; i < fftSize; ++i) {
    input[i] = std::sin(2.0f * M_PI * 10.0f * i / fftSize);
  }

  gpu.resetStats();

  // Perform FFT
  gpu.fft(input.data(), output.data(), fftSize);

  auto stats = gpu.getStats();

  std::cout << "✓ FFT computed (" << fftSize << " points)\n";
  std::cout << "  Time: " << std::fixed << std::setprecision(3)
            << stats.lastOperationTime << " ms\n";

  std::cout << "\n━━━ Convolution Benchmark ━━━\n";

  std::vector<float> signal(10000);
  std::vector<float> kernel(1000);
  std::vector<float> result(signal.size() + kernel.size() - 1);

  for (size_t i = 0; i < signal.size(); ++i) {
    signal[i] = static_cast<float>(i) / signal.size();
  }

  for (size_t i = 0; i < kernel.size(); ++i) {
    kernel[i] = std::exp(-static_cast<float>(i) / 100.0f);
  }

  gpu.resetStats();
  gpu.fastConvolve(signal.data(), kernel.data(), result.data(), signal.size(),
                   kernel.size());

  stats = gpu.getStats();

  std::cout << "✓ Fast convolution completed\n";
  std::cout << "  Signal: " << signal.size() << " samples\n";
  std::cout << "  Kernel: " << kernel.size() << " samples\n";
  std::cout << "  Time:   " << stats.lastOperationTime << " ms\n";

  std::cout << "\n━━━ Performance Summary ━━━\n";
  std::cout << "  Operations:     " << stats.operationCount << "\n";
  std::cout << "  Avg time:       " << stats.avgOperationTime << " ms\n";
  std::cout << "  GPU accelerated: " << (stats.usingGPU ? "Yes" : "No") << "\n";
}

int main() {
  std::cout << "\n";
  std::cout << "╔══════════════════════════════════════════════════════════╗\n";
  std::cout << "║       MolinAntro DAW - ACME Edition v3.0.0              ║\n";
  std::cout << "║         AI Features Demonstration                        ║\n";
  std::cout << "╚══════════════════════════════════════════════════════════╝\n";

  std::cout << "\nThis demo showcases all AI features:\n";
  std::cout << "  [1] Voice Cloning (RVC), TTS, Vocal Synthesis\n";
  std::cout << "  [2] AI Mastering & Neural Pitch Correction\n";
  std::cout << "  [3] Musical Analysis (Chords, Beat, Key, Transcription)\n";
  std::cout << "  [4] GPU Acceleration\n\n";

  std::cout << "Press ENTER to continue...\n";
  std::cin.get();

  demoVoiceCloning();
  std::cout << "\nPress ENTER to continue...\n";
  std::cin.get();

  demoAIMastering();
  std::cout << "\nPress ENTER to continue...\n";
  std::cin.get();

  demoMusicalAnalysis();
  std::cout << "\nPress ENTER to continue...\n";
  std::cin.get();

  demoGPUAcceleration();

  std::cout << "\n━━━ DEMONSTRATION COMPLETE ━━━\n\n";

  std::cout << "MolinAntro DAW ACME Edition - Status Report:\n";
  std::cout << "--------------------------------------------\n";
  std::cout << "  ✓ All modules initialized.\n";
  std::cout << "  ✓ Detailed logs above.\n";

  return 0;
}
