#include <gtest/gtest.h>
#include "core/AudioEngine.h"
#include "core/Transport.h"
#include "core/AudioBuffer.h"
#include <thread>
#include <chrono>
#include <cmath>

using namespace MolinAntro::Core;

class EngineTransportIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<AudioEngine>();
        transport = std::make_unique<Transport>();

        AudioEngine::Config config;
        config.sampleRate = 48000;
        config.bufferSize = 512;
        config.numInputChannels = 2;
        config.numOutputChannels = 2;

        ASSERT_TRUE(engine->initialize(config));

        transport->setBPM(120.0);
        transport->setTimeSignature(4, 4);
    }

    std::unique_ptr<AudioEngine> engine;
    std::unique_ptr<Transport> transport;
};

TEST_F(EngineTransportIntegrationTest, PlaybackWorkflow) {
    // Start playback
    transport->play();
    engine->start();

    EXPECT_TRUE(transport->isPlaying());
    EXPECT_EQ(engine->getState(), AudioEngine::State::Playing);

    // Simulate processing
    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    // Process several buffers
    for (int i = 0; i < 10; ++i) {
        transport->update(512, 48000);
        engine->processAudio(outputs, 512);
    }

    auto timeInfo = transport->getTimeInfo();
    EXPECT_GT(timeInfo.samplePosition, 0.0);

    // Stop
    transport->stop();
    engine->stop();

    EXPECT_FALSE(transport->isPlaying());
    EXPECT_EQ(engine->getState(), AudioEngine::State::Stopped);
}

TEST_F(EngineTransportIntegrationTest, RecordingWorkflow) {
    // Start recording
    transport->record();
    engine->start();

    EXPECT_TRUE(transport->isPlaying());
    EXPECT_TRUE(transport->isRecording());
    EXPECT_EQ(engine->getState(), AudioEngine::State::Playing);

    // Stop recording
    transport->stop();
    engine->stop();

    EXPECT_FALSE(transport->isRecording());
}

TEST_F(EngineTransportIntegrationTest, BPMChangeWhilePlaying) {
    transport->play();
    engine->start();

    // Change BPM during playback
    transport->setBPM(140.0);

    EXPECT_DOUBLE_EQ(transport->getBPM(), 140.0);
    EXPECT_TRUE(transport->isPlaying());
}

TEST_F(EngineTransportIntegrationTest, AudioCallbackWithTransport) {
    int samplesProcessed = 0;
    bool transportWasPlaying = false;

    // Set audio callback that checks transport state
    engine->setAudioCallback([&](float** /*inputs*/, float** outputs, int numSamples) {
        samplesProcessed += numSamples;

        auto timeInfo = transport->getTimeInfo();
        if (timeInfo.isPlaying) {
            transportWasPlaying = true;
            // Generate audio when playing
            for (int i = 0; i < numSamples; ++i) {
                outputs[0][i] = 0.1f;
                outputs[1][i] = 0.1f;
            }
        } else {
            // Silence when not playing
            for (int i = 0; i < numSamples; ++i) {
                outputs[0][i] = 0.0f;
                outputs[1][i] = 0.0f;
            }
        }
    });

    transport->play();
    engine->start();

    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    // Process
    transport->update(512, 48000);
    engine->processAudio(outputs, 512);

    EXPECT_EQ(samplesProcessed, 512);
    EXPECT_TRUE(transportWasPlaying);

    transport->stop();
    engine->stop();
}

TEST_F(EngineTransportIntegrationTest, PauseResumeWorkflow) {
    transport->play();
    engine->start();

    // Process some audio
    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    for (int i = 0; i < 5; ++i) {
        transport->update(512, 48000);
        engine->processAudio(outputs, 512);
    }

    auto timeInfo = transport->getTimeInfo();
    double pausePosition = timeInfo.samplePosition;

    // Pause
    transport->pause();
    engine->pause();

    EXPECT_FALSE(transport->isPlaying());
    EXPECT_EQ(engine->getState(), AudioEngine::State::Paused);

    // Position should not change while paused
    for (int i = 0; i < 5; ++i) {
        transport->update(512, 48000);
        engine->processAudio(outputs, 512);
    }

    timeInfo = transport->getTimeInfo();
    EXPECT_DOUBLE_EQ(timeInfo.samplePosition, pausePosition);

    // Resume
    transport->play();
    engine->resume();

    EXPECT_TRUE(transport->isPlaying());
    EXPECT_EQ(engine->getState(), AudioEngine::State::Playing);

    transport->stop();
    engine->stop();
}

TEST_F(EngineTransportIntegrationTest, CPUUsageMonitoring) {
    // Set a heavy processing callback
    engine->setAudioCallback([](float** /*inputs*/, float** outputs, int numSamples) {
        // Simulate processing
        for (int i = 0; i < numSamples; ++i) {
            float sample = 0.0f;
            for (int j = 0; j < 100; ++j) {
                sample += std::sin(static_cast<float>(i * j) * 0.001f);
            }
            outputs[0][i] = sample * 0.01f;
            outputs[1][i] = sample * 0.01f;
        }
    });

    engine->start();

    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    // Process several buffers to get stable CPU usage
    for (int i = 0; i < 20; ++i) {
        engine->processAudio(outputs, 512);
    }

    float cpuUsage = engine->getCPUUsage();
    EXPECT_GT(cpuUsage, 0.0f);
    EXPECT_LT(cpuUsage, 100.0f);

    engine->stop();
}
