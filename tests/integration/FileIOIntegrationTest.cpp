#include <gtest/gtest.h>
#include "dsp/AudioFile.h"
#include "core/AudioBuffer.h"
#include "core/AudioEngine.h"
#include <filesystem>
#include <cmath>

using namespace MolinAntro::DSP;
using namespace MolinAntro::Core;

class FileIOIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = "/tmp/molinantro_tests";
        std::filesystem::create_directories(testDir);
    }

    void TearDown() override {
        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::string testDir;
};

TEST_F(FileIOIntegrationTest, SaveLoadRoundTrip) {
    std::string filepath = testDir + "/roundtrip.wav";

    // Create buffer with test data
    const int numChannels = 2;
    const int numSamples = 48000;
    AudioBuffer buffer(numChannels, numSamples);

    // Generate stereo sine waves with different frequencies
    for (int i = 0; i < numSamples; ++i) {
        buffer.getWritePointer(0)[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / 48000.0);
        buffer.getWritePointer(1)[i] = 0.5f * std::sin(2.0 * M_PI * 880.0 * i / 48000.0);
    }

    // Save
    AudioFile saveFile;
    ASSERT_TRUE(saveFile.save(filepath, buffer, 48000, 24));

    // Load
    AudioFile loadFile;
    ASSERT_TRUE(loadFile.load(filepath));

    // Verify
    const AudioBuffer& loadedBuffer = loadFile.getBuffer();
    EXPECT_EQ(loadedBuffer.getNumChannels(), numChannels);
    EXPECT_EQ(loadedBuffer.getNumSamples(), numSamples);

    // Check audio data (with tolerance for 24-bit quantization)
    for (int ch = 0; ch < numChannels; ++ch) {
        const float* original = buffer.getReadPointer(ch);
        const float* loaded = loadedBuffer.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            EXPECT_NEAR(loaded[i], original[i], 2.0f / 8388608.0f); // 24-bit tolerance (conservative)
        }
    }
}

TEST_F(FileIOIntegrationTest, MultipleFileOperations) {
    // Save multiple files
    for (int i = 0; i < 5; ++i) {
        std::string filepath = testDir + "/file" + std::to_string(i) + ".wav";

        AudioBuffer buffer(1, 1000);
        float* data = buffer.getWritePointer(0);
        for (int j = 0; j < 1000; ++j) {
            data[j] = static_cast<float>(i) / 10.0f;
        }

        AudioFile audioFile;
        EXPECT_TRUE(audioFile.save(filepath, buffer, 48000, 16));
    }

    // Verify all files exist
    for (int i = 0; i < 5; ++i) {
        std::string filepath = testDir + "/file" + std::to_string(i) + ".wav";
        EXPECT_TRUE(std::filesystem::exists(filepath));
    }

    // Load and verify each file
    for (int i = 0; i < 5; ++i) {
        std::string filepath = testDir + "/file" + std::to_string(i) + ".wav";

        AudioFile audioFile;
        ASSERT_TRUE(audioFile.load(filepath));

        const AudioBuffer& buffer = audioFile.getBuffer();
        const float* data = buffer.getReadPointer(0);

        float expectedValue = static_cast<float>(i) / 10.0f;
        for (int j = 0; j < 1000; ++j) {
            EXPECT_NEAR(data[j], expectedValue, 2.0f / 32768.0f);  // 16-bit tolerance
        }
    }
}

TEST_F(FileIOIntegrationTest, ProcessAndSave) {
    std::string inputFile = testDir + "/input.wav";
    std::string outputFile = testDir + "/output.wav";

    // Create input file
    AudioBuffer inputBuffer(2, 10000);
    for (int ch = 0; ch < 2; ++ch) {
        float* data = inputBuffer.getWritePointer(ch);
        for (int i = 0; i < 10000; ++i) {
            data[i] = 0.5f;
        }
    }

    AudioFile saveFile;
    ASSERT_TRUE(saveFile.save(inputFile, inputBuffer, 48000, 16));

    // Load and process
    AudioFile loadFile;
    ASSERT_TRUE(loadFile.load(inputFile));

    AudioBuffer& buffer = loadFile.getBuffer();

    // Apply gain reduction
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        buffer.applyGain(ch, 0.5f);
    }

    // Save processed audio
    AudioFile processedFile;
    ASSERT_TRUE(processedFile.save(outputFile, buffer, 48000, 16));

    // Load and verify
    AudioFile verifyFile;
    ASSERT_TRUE(verifyFile.load(outputFile));

    const AudioBuffer& verifyBuffer = verifyFile.getBuffer();
    const float* data = verifyBuffer.getReadPointer(0);

    for (int i = 0; i < 10000; ++i) {
        EXPECT_NEAR(data[i], 0.25f, 2.0f / 32768.0f);  // 16-bit tolerance
    }
}

TEST_F(FileIOIntegrationTest, DifferentBitDepths) {
    std::vector<int> bitDepths = {16, 24, 32};

    for (int bitDepth : bitDepths) {
        std::string filepath = testDir + "/bitdepth_" + std::to_string(bitDepth) + ".wav";

        AudioBuffer buffer(1, 1000);
        float* data = buffer.getWritePointer(0);
        for (int i = 0; i < 1000; ++i) {
            data[i] = std::sin(2.0 * M_PI * i / 1000.0);
        }

        AudioFile saveFile;
        ASSERT_TRUE(saveFile.save(filepath, buffer, 48000, bitDepth));

        AudioFile loadFile;
        ASSERT_TRUE(loadFile.load(filepath));

        auto info = loadFile.getInfo();
        EXPECT_EQ(info.bitDepth, bitDepth);
    }
}
