#include "core/AudioEngine.h"
#include "core/AudioBuffer.h"
#include "dsp/AudioFile.h"
#include "dsp/SpectralProcessor.h"
#include "instruments/Synthesizer.h"
#include <iostream>
#include <iomanip>

using namespace MolinAntro;

void printHeader() {
    std::cout << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     MolinAntro DAW - Spectral Processing Demo           ║\n";
    std::cout << "║     SOTA Spectral Editor & Forensic Analysis             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n\n";
}

void demonstrateSpectralAnalysis() {
    std::cout << "━━━ [1] SPECTRAL ANALYSIS & EDITING ━━━\n";

    const int sampleRate = 48000;
    const int duration = 2; // seconds
    const int numSamples = sampleRate * duration;

    // Generate test signal: 440Hz + 880Hz + noise
    Core::AudioBuffer testSignal(1, numSamples);
    float* data = testSignal.getWritePointer(0);

    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;
        data[i] = 0.5f * std::sin(2.0f * M_PI * 440.0f * t)  // A4
                + 0.3f * std::sin(2.0f * M_PI * 880.0f * t)  // A5
                + 0.05f * (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f); // Noise
    }

    std::cout << "✓ Generated test signal (440Hz + 880Hz + noise)\n";

    // Spectral analysis
    DSP::SpectralProcessor processor;
    processor.setFFTSize(2048);
    processor.setHopSize(512);
    processor.setSampleRate(sampleRate);
    processor.setWindowType(DSP::SpectralProcessor::WindowType::BlackmanHarris);

    processor.analyze(testSignal);
    std::cout << "✓ Spectral analysis complete\n";
    std::cout << "  - Frames analyzed: " << processor.getFrames().size() << "\n";
    std::cout << "  - FFT size: " << processor.getFFTSize() << "\n";
    std::cout << "  - Hop size: " << processor.getHopSize() << "\n";

    // Spectral editing: Boost harmonics
    processor.harmonicEnhancement(440.0f, 8, 6.0f);
    std::cout << "✓ Enhanced harmonics (440Hz fundamental, 8 harmonics, +6dB)\n";

    // Spectral gate
    processor.spectralGate(-40.0f);
    std::cout << "✓ Applied spectral gate (-40dB threshold)\n";

    // Synthesize back
    Core::AudioBuffer processedSignal(1, numSamples);
    processor.synthesize(processedSignal);
    std::cout << "✓ Synthesized processed audio\n";

    // Save result
    DSP::AudioFile file;
    file.save("/tmp/spectral_enhanced.wav", processedSignal, sampleRate, 24);
    std::cout << "✓ Saved to: /tmp/spectral_enhanced.wav\n\n";
}

void demonstrateNoiseReduction() {
    std::cout << "━━━ [2] PROFESSIONAL NOISE REDUCTION ━━━\n";

    const int sampleRate = 48000;
    const int noiseDuration = 1; // 1 second noise profile
    const int signalDuration = 3; // 3 seconds noisy signal
    const int noiseSamples = sampleRate * noiseDuration;
    const int signalSamples = sampleRate * signalDuration;

    // Generate noise profile (pink noise)
    Core::AudioBuffer noiseProfile(1, noiseSamples);
    float* noiseData = noiseProfile.getWritePointer(0);
    for (int i = 0; i < noiseSamples; ++i) {
        noiseData[i] = 0.1f * (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
    }
    std::cout << "✓ Generated noise profile (1 second)\n";

    // Generate noisy signal (music + noise)
    Core::AudioBuffer noisySignal(1, signalSamples);
    float* signalData = noisySignal.getWritePointer(0);
    for (int i = 0; i < signalSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;
        // Clean music signal
        float music = 0.6f * std::sin(2.0f * M_PI * 440.0f * t)
                    + 0.4f * std::sin(2.0f * M_PI * 554.37f * t) // C#5
                    + 0.3f * std::sin(2.0f * M_PI * 659.25f * t); // E5
        // Add noise
        float noise = 0.1f * (static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f);
        signalData[i] = music + noise;
    }
    std::cout << "✓ Generated noisy signal (3 seconds)\n";

    // Apply noise reduction
    DSP::SpectralNoiseReduction noiseReducer;
    noiseReducer.learnNoiseProfile(noiseProfile);
    noiseReducer.process(noisySignal, 1.0f);

    // Save results
    DSP::AudioFile file;
    file.save("/tmp/denoised.wav", noisySignal, sampleRate, 24);
    std::cout << "✓ Noise reduction applied\n";
    std::cout << "✓ Saved to: /tmp/denoised.wav\n\n";
}

void demonstrateHarmonicPercussiveSeparation() {
    std::cout << "━━━ [3] HARMONIC/PERCUSSIVE SEPARATION ━━━\n";

    const int sampleRate = 48000;
    const int duration = 2;
    const int numSamples = sampleRate * duration;

    // Generate mixed signal: tonal + percussive
    Core::AudioBuffer mixedSignal(1, numSamples);
    float* data = mixedSignal.getWritePointer(0);

    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;

        // Harmonic component (sustained tone)
        float harmonic = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);

        // Percussive component (short bursts every 0.25s)
        float percussive = 0.0f;
        if (std::fmod(t, 0.25f) < 0.01f) {
            percussive = 0.8f * std::exp(-50.0f * std::fmod(t, 0.25f));
        }

        data[i] = harmonic + percussive;
    }

    std::cout << "✓ Generated mixed signal (harmonic + percussive)\n";

    // Perform separation
    DSP::HarmonicPercussiveSeparation separator;
    auto result = separator.separate(mixedSignal);

    if (result.harmonic && result.percussive) {
        // Save separated components
        DSP::AudioFile file;
        file.save("/tmp/harmonic.wav", *result.harmonic, sampleRate, 24);
        file.save("/tmp/percussive.wav", *result.percussive, sampleRate, 24);

        std::cout << "✓ Separation complete\n";
        std::cout << "✓ Harmonic component: /tmp/harmonic.wav\n";
        std::cout << "✓ Percussive component: /tmp/percussive.wav\n\n";
    }
}

void demonstrateForensicAnalysis() {
    std::cout << "━━━ [4] FORENSIC AUDIO ANALYSIS ━━━\n";

    const int sampleRate = 48000;
    const int duration = 3;
    const int numSamples = sampleRate * duration;

    // Generate signal with potential editing artifact
    Core::AudioBuffer signal(1, numSamples);
    float* data = signal.getWritePointer(0);

    for (int i = 0; i < numSamples; ++i) {
        float t = static_cast<float>(i) / sampleRate;

        // Normal signal
        float value = 0.5f * std::sin(2.0f * M_PI * 440.0f * t);

        // Introduce "edit" at 1.5 seconds (spectral discontinuity)
        if (std::abs(t - 1.5f) < 0.01f) {
            value += 0.5f; // Abrupt change
        }

        data[i] = value;
    }

    std::cout << "✓ Generated test signal with editing artifact at t=1.5s\n";

    // Forensic analysis
    DSP::SpectralProcessor processor;
    processor.setFFTSize(4096); // Higher resolution for forensics
    processor.setHopSize(1024);
    processor.setSampleRate(sampleRate);
    processor.analyze(signal);

    auto forensicReport = processor.performForensicAnalysis();

    std::cout << "\n📊 FORENSIC ANALYSIS REPORT:\n";
    std::cout << "  ┌────────────────────────────────────┐\n";
    std::cout << "  │ Average Pitch: " << std::fixed << std::setprecision(2)
              << forensicReport.averagePitch << " Hz           │\n";
    std::cout << "  │ Editing Detected: " << (forensicReport.editingDetected ? "YES ⚠️" : "NO ✓")
              << "            │\n";
    std::cout << "  │ Anomaly Frames: " << std::setw(4) << forensicReport.anomalyFrames.size()
              << "                │\n";
    std::cout << "  └────────────────────────────────────┘\n";

    if (!forensicReport.anomalyFrames.empty()) {
        std::cout << "\n  Detected anomalies at frames: ";
        for (size_t i = 0; i < std::min(size_t(10), forensicReport.anomalyFrames.size()); ++i) {
            std::cout << forensicReport.anomalyFrames[i];
            if (i < forensicReport.anomalyFrames.size() - 1) std::cout << ", ";
        }
        if (forensicReport.anomalyFrames.size() > 10) {
            std::cout << " (...+" << (forensicReport.anomalyFrames.size() - 10) << " more)";
        }
        std::cout << "\n";
    }

    std::cout << "\n✓ Forensic analysis complete\n\n";
}

int main(int /*argc*/, char* /*argv*/[]) {
    printHeader();

    try {
        std::cout << "This demo showcases SOTA spectral processing capabilities:\n";
        std::cout << "  [1] Spectral Analysis & Editing (FFT/IFFT)\n";
        std::cout << "  [2] Professional Noise Reduction\n";
        std::cout << "  [3] Harmonic/Percussive Separation\n";
        std::cout << "  [4] Forensic Audio Analysis\n\n";

        std::cout << "Starting demonstrations...\n\n";

        demonstrateSpectralAnalysis();
        demonstrateNoiseReduction();
        demonstrateHarmonicPercussiveSeparation();
        demonstrateForensicAnalysis();

        std::cout << "╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║     ALL SPECTRAL PROCESSING DEMOS COMPLETED ✓            ║\n";
        std::cout << "║                                                          ║\n";
        std::cout << "║  Output files in /tmp/:                                  ║\n";
        std::cout << "║    - spectral_enhanced.wav                               ║\n";
        std::cout << "║    - denoised.wav                                        ║\n";
        std::cout << "║    - harmonic.wav                                        ║\n";
        std::cout << "║    - percussive.wav                                      ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
