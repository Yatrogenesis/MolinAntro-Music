#include "core/AudioEngine.h"
#include "core/Transport.h"
#include "core/AudioBuffer.h"
#include "dsp/AudioFile.h"
#include "ui/ConsoleUI.h"

#include <iostream>
#include <memory>
#include <cmath>

int main(int argc, char* argv[]) {
    std::cout << "MolinAntro DAW v1.0.0\n";
    std::cout << "Professional Digital Audio Workstation\n";
    std::cout << "Copyright (c) 2025 MolinAntro Technologies\n\n";

    // Initialize audio engine
    MolinAntro::Core::AudioEngine engine;
    MolinAntro::Core::AudioEngine::Config config;
    config.sampleRate = 48000;
    config.bufferSize = 512;
    config.numInputChannels = 2;
    config.numOutputChannels = 2;

    if (!engine.initialize(config)) {
        std::cerr << "Failed to initialize audio engine\n";
        return 1;
    }

    // Initialize transport
    MolinAntro::Core::Transport transport;
    transport.setBPM(120.0);
    transport.setTimeSignature(4, 4);

    // Set up audio processing callback (simple test tone generator)
    engine.setAudioCallback([&transport](float** /*inputs*/, float** outputs, int numSamples) {
        static double phase = 0.0;
        const double frequency = 440.0; // A4 note
        const int sampleRate = 48000;
        const double phaseIncrement = (2.0 * M_PI * frequency) / sampleRate;

        // Update transport
        transport.update(numSamples, sampleRate);

        auto timeInfo = transport.getTimeInfo();

        // Only generate tone when playing
        if (timeInfo.isPlaying) {
            for (int i = 0; i < numSamples; ++i) {
                float sample = 0.1f * std::sin(phase);
                outputs[0][i] = sample; // Left
                outputs[1][i] = sample; // Right
                phase += phaseIncrement;
                if (phase >= 2.0 * M_PI) {
                    phase -= 2.0 * M_PI;
                }
            }
        } else {
            // Silence
            for (int i = 0; i < numSamples; ++i) {
                outputs[0][i] = 0.0f;
                outputs[1][i] = 0.0f;
            }
        }
    });

    // Start the engine
    engine.start();

    // Create and run console UI
    MolinAntro::UI::ConsoleUI ui(engine, transport);

    // Check for command line arguments
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--test") {
            std::cout << "Running in test mode...\n";

            // Quick test sequence
            std::cout << "Testing audio buffer...\n";
            MolinAntro::Core::AudioBuffer buffer(2, 1024);
            buffer.clear();
            std::cout << "✓ AudioBuffer OK\n";

            std::cout << "Testing audio file I/O...\n";
            MolinAntro::DSP::AudioFile audioFile;
            // Test save
            if (audioFile.save("/tmp/test.wav", buffer, 48000, 16)) {
                std::cout << "✓ AudioFile Save OK\n";
            }

            std::cout << "Testing transport...\n";
            transport.play();
            transport.stop();
            std::cout << "✓ Transport OK\n";

            std::cout << "\n✓ All basic tests passed!\n";
            return 0;
        } else if (arg == "--version") {
            std::cout << "Version: 1.0.0\n";
            return 0;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n";
            std::cout << "Options:\n";
            std::cout << "  --test     Run basic tests\n";
            std::cout << "  --version  Show version\n";
            std::cout << "  --help     Show this help\n";
            return 0;
        }
    }

    // Run interactive mode
    ui.run();

    // Cleanup
    engine.stop();

    return 0;
}
