#include <gtest/gtest.h>
#include "core/AudioEngine.h"
#include <thread>
#include <chrono>

using namespace MolinAntro::Core;

class AudioEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<AudioEngine>();

        config.sampleRate = 48000;
        config.bufferSize = 512;
        config.numInputChannels = 2;
        config.numOutputChannels = 2;
    }

    std::unique_ptr<AudioEngine> engine;
    AudioEngine::Config config;
};

TEST_F(AudioEngineTest, Initialize) {
    EXPECT_TRUE(engine->initialize(config));
    EXPECT_EQ(engine->getSampleRate(), 48000);
    EXPECT_EQ(engine->getBufferSize(), 512);
}

TEST_F(AudioEngineTest, StateTransitions) {
    engine->initialize(config);

    EXPECT_EQ(engine->getState(), AudioEngine::State::Stopped);

    engine->start();
    EXPECT_EQ(engine->getState(), AudioEngine::State::Playing);

    engine->pause();
    EXPECT_EQ(engine->getState(), AudioEngine::State::Paused);

    engine->resume();
    EXPECT_EQ(engine->getState(), AudioEngine::State::Playing);

    engine->stop();
    EXPECT_EQ(engine->getState(), AudioEngine::State::Stopped);
}

TEST_F(AudioEngineTest, ProcessAudio) {
    engine->initialize(config);
    engine->start();

    // Prepare output buffer
    float outputLeft[512] = {0};
    float outputRight[512] = {0};
    float* outputs[2] = {outputLeft, outputRight};

    // Process audio
    engine->processAudio(outputs, 512);

    // When playing with no callback, should output silence
    for (int i = 0; i < 512; ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], 0.0f);
        EXPECT_FLOAT_EQ(outputs[1][i], 0.0f);
    }

    engine->stop();
}

TEST_F(AudioEngineTest, AudioCallback) {
    engine->initialize(config);

    bool callbackCalled = false;

    // Set callback that generates test tone
    engine->setAudioCallback([&callbackCalled](float** /*inputs*/, float** outputs, int numSamples) {
        callbackCalled = true;
        for (int i = 0; i < numSamples; ++i) {
            outputs[0][i] = 0.5f;
            outputs[1][i] = 0.5f;
        }
    });

    engine->start();

    // Process audio
    float outputLeft[512] = {0};
    float outputRight[512] = {0};
    float* outputs[2] = {outputLeft, outputRight};

    engine->processAudio(outputs, 512);

    EXPECT_TRUE(callbackCalled);

    // Verify callback output
    for (int i = 0; i < 512; ++i) {
        EXPECT_FLOAT_EQ(outputs[0][i], 0.5f);
        EXPECT_FLOAT_EQ(outputs[1][i], 0.5f);
    }

    engine->stop();
}

TEST_F(AudioEngineTest, CPUUsageTracking) {
    engine->initialize(config);
    engine->start();

    float outputLeft[512] = {0};
    float outputRight[512] = {0};
    float* outputs[2] = {outputLeft, outputRight};

    // Process several buffers
    for (int i = 0; i < 10; ++i) {
        engine->processAudio(outputs, 512);
    }

    // CPU usage should be tracked
    float cpuUsage = engine->getCPUUsage();
    EXPECT_GE(cpuUsage, 0.0f);
    EXPECT_LE(cpuUsage, 100.0f);

    engine->stop();
}
