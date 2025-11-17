#include <gtest/gtest.h>
#include "core/AudioBuffer.h"

using namespace MolinAntro::Core;

class AudioBufferTest : public ::testing::Test {
protected:
    void SetUp() override {
        buffer = std::make_unique<AudioBuffer>(2, 1024);
    }

    std::unique_ptr<AudioBuffer> buffer;
};

TEST_F(AudioBufferTest, Construction) {
    EXPECT_EQ(buffer->getNumChannels(), 2);
    EXPECT_EQ(buffer->getNumSamples(), 1024);
}

TEST_F(AudioBufferTest, ClearBuffer) {
    // Fill with data
    for (int ch = 0; ch < buffer->getNumChannels(); ++ch) {
        float* data = buffer->getWritePointer(ch);
        for (int i = 0; i < buffer->getNumSamples(); ++i) {
            data[i] = 1.0f;
        }
    }

    // Clear
    buffer->clear();

    // Verify all zeros
    for (int ch = 0; ch < buffer->getNumChannels(); ++ch) {
        const float* data = buffer->getReadPointer(ch);
        for (int i = 0; i < buffer->getNumSamples(); ++i) {
            EXPECT_FLOAT_EQ(data[i], 0.0f);
        }
    }
}

TEST_F(AudioBufferTest, ApplyGain) {
    // Fill with 1.0
    for (int ch = 0; ch < buffer->getNumChannels(); ++ch) {
        float* data = buffer->getWritePointer(ch);
        for (int i = 0; i < buffer->getNumSamples(); ++i) {
            data[i] = 1.0f;
        }
    }

    // Apply 0.5 gain
    buffer->applyGain(0, 0.5f);

    // Verify
    const float* data = buffer->getReadPointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        EXPECT_FLOAT_EQ(data[i], 0.5f);
    }
}

TEST_F(AudioBufferTest, GetRMSLevel) {
    // Fill with known pattern
    float* data = buffer->getWritePointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        data[i] = 0.5f;
    }

    float rms = buffer->getRMSLevel(0);
    EXPECT_FLOAT_EQ(rms, 0.5f);
}

TEST_F(AudioBufferTest, GetPeakLevel) {
    // Fill with known pattern
    float* data = buffer->getWritePointer(0);
    data[0] = 0.0f;
    data[1] = 0.5f;
    data[2] = -0.8f;
    data[3] = 0.3f;

    float peak = buffer->getPeakLevel(0);
    EXPECT_FLOAT_EQ(peak, 0.8f);
}

TEST_F(AudioBufferTest, CopyFrom) {
    AudioBuffer source(2, 1024);

    // Fill source with data
    float* srcData = source.getWritePointer(0);
    for (int i = 0; i < source.getNumSamples(); ++i) {
        srcData[i] = 0.75f;
    }

    // Copy to destination
    buffer->copyFrom(source, 0, 0);

    // Verify
    const float* dstData = buffer->getReadPointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        EXPECT_FLOAT_EQ(dstData[i], 0.75f);
    }
}

TEST_F(AudioBufferTest, AddFrom) {
    AudioBuffer source(2, 1024);

    // Fill source with 0.5
    float* srcData = source.getWritePointer(0);
    for (int i = 0; i < source.getNumSamples(); ++i) {
        srcData[i] = 0.5f;
    }

    // Fill destination with 0.3
    float* dstData = buffer->getWritePointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        dstData[i] = 0.3f;
    }

    // Add with gain 2.0
    buffer->addFrom(source, 0, 0, 2.0f);

    // Verify (0.3 + 0.5 * 2.0 = 1.3)
    const float* resultData = buffer->getReadPointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        EXPECT_FLOAT_EQ(resultData[i], 1.3f);
    }
}

TEST_F(AudioBufferTest, Clone) {
    // Fill with data
    float* data = buffer->getWritePointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        data[i] = static_cast<float>(i) / buffer->getNumSamples();
    }

    // Clone
    auto cloned = buffer->clone();

    // Verify dimensions
    EXPECT_EQ(cloned->getNumChannels(), buffer->getNumChannels());
    EXPECT_EQ(cloned->getNumSamples(), buffer->getNumSamples());

    // Verify data
    const float* originalData = buffer->getReadPointer(0);
    const float* clonedData = cloned->getReadPointer(0);
    for (int i = 0; i < buffer->getNumSamples(); ++i) {
        EXPECT_FLOAT_EQ(clonedData[i], originalData[i]);
    }
}

// RingBuffer tests
TEST(RingBufferTest, WriteRead) {
    RingBuffer ringBuffer(2, 1024);

    // Prepare data
    float writeData[2][512];
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            writeData[ch][i] = static_cast<float>(i);
        }
    }

    const float* writePointers[2] = {writeData[0], writeData[1]};

    // Write
    int written = ringBuffer.write(writePointers, 512);
    EXPECT_EQ(written, 512);

    // Read
    float readData[2][512];
    float* readPointers[2] = {readData[0], readData[1]};
    int read = ringBuffer.read(readPointers, 512);
    EXPECT_EQ(read, 512);

    // Verify
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < 512; ++i) {
            EXPECT_FLOAT_EQ(readData[ch][i], writeData[ch][i]);
        }
    }
}

TEST(RingBufferTest, AvailableSpace) {
    RingBuffer ringBuffer(1, 1024);

    EXPECT_GT(ringBuffer.getAvailableWrite(), 1000);
    EXPECT_EQ(ringBuffer.getAvailableRead(), 0);

    float data[256];
    const float* writePtr = data;
    ringBuffer.write(&writePtr, 256);

    EXPECT_EQ(ringBuffer.getAvailableRead(), 256);
}
