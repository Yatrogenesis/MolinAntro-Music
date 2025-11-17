#include <gtest/gtest.h>
#include "dsp/AudioFile.h"
#include "core/AudioBuffer.h"
#include <filesystem>
#include <cmath>

using namespace MolinAntro::DSP;
using namespace MolinAntro::Core;

class AudioFileTest : public ::testing::Test {
protected:
    void SetUp() override {
        testFilePath = "/tmp/molinantro_test.wav";
    }

    void TearDown() override {
        if (std::filesystem::exists(testFilePath)) {
            std::filesystem::remove(testFilePath);
        }
    }

    std::string testFilePath;
};

TEST_F(AudioFileTest, DetectFormat) {
    EXPECT_EQ(AudioFile::detectFormat("test.wav"), AudioFile::Format::WAV);
    EXPECT_EQ(AudioFile::detectFormat("test.aiff"), AudioFile::Format::AIFF);
    EXPECT_EQ(AudioFile::detectFormat("test.flac"), AudioFile::Format::FLAC);
    EXPECT_EQ(AudioFile::detectFormat("test.mp3"), AudioFile::Format::MP3);
    EXPECT_EQ(AudioFile::detectFormat("test.unknown"), AudioFile::Format::Unknown);
}

TEST_F(AudioFileTest, SaveAndLoadWAV) {
    AudioFile audioFile;

    // Create test buffer with sine wave
    const int numChannels = 2;
    const int numSamples = 48000; // 1 second at 48kHz
    AudioBuffer buffer(numChannels, numSamples);

    // Generate 440Hz sine wave
    for (int ch = 0; ch < numChannels; ++ch) {
        float* data = buffer.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            data[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / 48000.0);
        }
    }

    // Save
    EXPECT_TRUE(audioFile.save(testFilePath, buffer, 48000, 16));
    EXPECT_TRUE(std::filesystem::exists(testFilePath));

    // Load
    AudioFile loadedFile;
    EXPECT_TRUE(loadedFile.load(testFilePath));

    // Verify file info
    auto info = loadedFile.getInfo();
    EXPECT_EQ(info.format, AudioFile::Format::WAV);
    EXPECT_EQ(info.sampleRate, 48000);
    EXPECT_EQ(info.numChannels, 2);
    EXPECT_EQ(info.bitDepth, 16);
    EXPECT_EQ(info.numSamples, numSamples);

    // Verify audio data (with tolerance for 16-bit quantization)
    const AudioBuffer& loadedBuffer = loadedFile.getBuffer();
    for (int ch = 0; ch < numChannels; ++ch) {
        const float* original = buffer.getReadPointer(ch);
        const float* loaded = loadedBuffer.getReadPointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            EXPECT_NEAR(loaded[i], original[i], 2.0f / 32768.0f); // 16-bit tolerance (conservative)
        }
    }
}

TEST_F(AudioFileTest, Save24Bit) {
    AudioFile audioFile;

    AudioBuffer buffer(1, 1000);
    float* data = buffer.getWritePointer(0);
    for (int i = 0; i < 1000; ++i) {
        data[i] = static_cast<float>(i) / 1000.0f;
    }

    // Save as 24-bit
    EXPECT_TRUE(audioFile.save(testFilePath, buffer, 48000, 24));

    // Load and verify bit depth
    AudioFile loadedFile;
    EXPECT_TRUE(loadedFile.load(testFilePath));

    auto info = loadedFile.getInfo();
    EXPECT_EQ(info.bitDepth, 24);
}

TEST_F(AudioFileTest, Save32BitFloat) {
    AudioFile audioFile;

    AudioBuffer buffer(2, 2000);
    for (int ch = 0; ch < 2; ++ch) {
        float* data = buffer.getWritePointer(ch);
        for (int i = 0; i < 2000; ++i) {
            data[i] = 0.25f;
        }
    }

    // Save as 32-bit float
    EXPECT_TRUE(audioFile.save(testFilePath, buffer, 96000, 32));

    // Load
    AudioFile loadedFile;
    EXPECT_TRUE(loadedFile.load(testFilePath));

    auto info = loadedFile.getInfo();
    EXPECT_EQ(info.sampleRate, 96000);
    EXPECT_EQ(info.bitDepth, 32);

    // 32-bit float should have no quantization error
    const AudioBuffer& loadedBuffer = loadedFile.getBuffer();
    const float* loaded = loadedBuffer.getReadPointer(0);
    for (int i = 0; i < 2000; ++i) {
        EXPECT_FLOAT_EQ(loaded[i], 0.25f);
    }
}

TEST_F(AudioFileTest, LoadNonExistentFile) {
    AudioFile audioFile;
    EXPECT_FALSE(audioFile.load("/nonexistent/path/file.wav"));
    EXPECT_FALSE(audioFile.isLoaded());
}

TEST_F(AudioFileTest, DurationCalculation) {
    AudioFile audioFile;

    AudioBuffer buffer(2, 96000); // 2 seconds at 48kHz
    buffer.clear();

    EXPECT_TRUE(audioFile.save(testFilePath, buffer, 48000, 16));

    AudioFile loadedFile;
    EXPECT_TRUE(loadedFile.load(testFilePath));

    auto info = loadedFile.getInfo();
    EXPECT_DOUBLE_EQ(info.durationSeconds, 2.0);
}
