#include <gtest/gtest.h>
#include "core/Transport.h"

using namespace MolinAntro::Core;

class TransportTest : public ::testing::Test {
protected:
    void SetUp() override {
        transport = std::make_unique<Transport>();
        transport->setBPM(120.0);
        transport->setTimeSignature(4, 4);
    }

    std::unique_ptr<Transport> transport;
};

TEST_F(TransportTest, InitialState) {
    auto info = transport->getTimeInfo();

    EXPECT_FALSE(info.isPlaying);
    EXPECT_FALSE(info.isRecording);
    EXPECT_DOUBLE_EQ(info.bpm, 120.0);
    EXPECT_EQ(info.numerator, 4);
    EXPECT_EQ(info.denominator, 4);
    EXPECT_EQ(info.bar, 1);
    EXPECT_EQ(info.beat, 1);
}

TEST_F(TransportTest, PlayStop) {
    EXPECT_FALSE(transport->isPlaying());

    transport->play();
    EXPECT_TRUE(transport->isPlaying());

    transport->stop();
    EXPECT_FALSE(transport->isPlaying());

    auto info = transport->getTimeInfo();
    EXPECT_DOUBLE_EQ(info.samplePosition, 0.0);
}

TEST_F(TransportTest, Record) {
    transport->record();

    EXPECT_TRUE(transport->isPlaying());
    EXPECT_TRUE(transport->isRecording());
}

TEST_F(TransportTest, PauseResume) {
    transport->play();
    EXPECT_TRUE(transport->isPlaying());

    transport->pause();
    EXPECT_FALSE(transport->isPlaying());

    // Position should be preserved
    auto info = transport->getTimeInfo();
    double pausedPosition = info.samplePosition;

    // Update shouldn't change position when paused
    transport->update(512, 48000);
    info = transport->getTimeInfo();
    EXPECT_DOUBLE_EQ(info.samplePosition, pausedPosition);
}

TEST_F(TransportTest, BPMChange) {
    transport->setBPM(140.0);
    EXPECT_DOUBLE_EQ(transport->getBPM(), 140.0);

    // Invalid BPM should be rejected
    transport->setBPM(5.0);  // Too low
    EXPECT_DOUBLE_EQ(transport->getBPM(), 140.0);  // Should remain unchanged

    transport->setBPM(1000.0);  // Too high
    EXPECT_DOUBLE_EQ(transport->getBPM(), 140.0);  // Should remain unchanged
}

TEST_F(TransportTest, TimeSignature) {
    transport->setTimeSignature(3, 4);

    auto info = transport->getTimeInfo();
    EXPECT_EQ(info.numerator, 3);
    EXPECT_EQ(info.denominator, 4);
}

TEST_F(TransportTest, PositionUpdate) {
    transport->play();

    const int sampleRate = 48000;
    const int bufferSize = 512;

    // Update several times
    for (int i = 0; i < 10; ++i) {
        transport->update(bufferSize, sampleRate);
    }

    auto info = transport->getTimeInfo();

    // Should have advanced
    EXPECT_GT(info.samplePosition, 0.0);
    EXPECT_GT(info.timeInSeconds, 0.0);

    // Calculate expected samples
    double expectedSamples = 10.0 * bufferSize;
    EXPECT_DOUBLE_EQ(info.samplePosition, expectedSamples);
}

TEST_F(TransportTest, BarBeatCalculation) {
    transport->setBPM(120.0);  // 120 BPM
    transport->setTimeSignature(4, 4);  // 4/4 time
    transport->play();

    const int sampleRate = 48000;

    // At 120 BPM, one beat = 0.5 seconds = 24000 samples
    // One bar (4 beats) = 96000 samples

    // Advance just past one bar
    transport->update(96001, sampleRate);

    auto info = transport->getTimeInfo();

    // Should be in bar 2, beat 1
    EXPECT_EQ(info.bar, 2);
    EXPECT_EQ(info.beat, 1);
}

TEST_F(TransportTest, SetPosition) {
    transport->setPositionSamples(48000.0);

    auto info = transport->getTimeInfo();
    EXPECT_DOUBLE_EQ(info.samplePosition, 48000.0);

    transport->setPositionSeconds(2.0, 48000);
    info = transport->getTimeInfo();
    EXPECT_DOUBLE_EQ(info.timeInSeconds, 2.0);
    EXPECT_DOUBLE_EQ(info.samplePosition, 96000.0);
}

TEST_F(TransportTest, StateCallback) {
    bool callbackCalled = false;
    double lastBPM = 0.0;

    transport->setStateCallback([&](const Transport::TimeInfo& info) {
        callbackCalled = true;
        lastBPM = info.bpm;
    });

    transport->setBPM(130.0);

    EXPECT_TRUE(callbackCalled);
    EXPECT_DOUBLE_EQ(lastBPM, 130.0);
}
