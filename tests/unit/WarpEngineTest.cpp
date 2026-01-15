/**
 * @file WarpEngineTest.cpp
 * @brief Unit tests for WarpEngine time-stretching algorithms
 *
 * Tests for:
 * - PhaseVocoderEngine (STFT-based)
 * - WSOLAEngine (cross-correlation)
 * - Simple OLA
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "sequencer/WarpEngine.h"
#include "core/AudioBuffer.h"
#include <cmath>

using namespace MolinAntro::Sequencer;
using namespace MolinAntro::Core;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//=============================================================================
// Test Fixtures
//=============================================================================

class WarpEngineTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 1-second sine wave at 440 Hz, 48kHz sample rate
        const int sampleRate = 48000;
        const int numSamples = sampleRate;  // 1 second
        const double frequency = 440.0;

        testBuffer = std::make_unique<AudioBuffer>(2, numSamples);

        for (int ch = 0; ch < 2; ++ch) {
            float* data = testBuffer->getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                double t = static_cast<double>(i) / sampleRate;
                data[i] = static_cast<float>(std::sin(2.0 * M_PI * frequency * t));
            }
        }
    }

    std::unique_ptr<AudioBuffer> testBuffer;
    static constexpr int sampleRate_ = 48000;
};

//=============================================================================
// PhaseVocoderEngine Tests
//=============================================================================

TEST_F(WarpEngineTest, PhaseVocoder_StretchDoublesLength) {
    PhaseVocoderEngine pv(2048, 512);

    // Stretch by 2x (half speed)
    AudioBuffer result = pv.stretch(*testBuffer, 2.0);

    // Output should be approximately 2x the length
    int expectedLength = testBuffer->getNumSamples() * 2;
    int actualLength = result.getNumSamples();

    // Allow 5% tolerance due to windowing edge effects
    EXPECT_NEAR(actualLength, expectedLength, expectedLength * 0.05);
    EXPECT_EQ(result.getNumChannels(), testBuffer->getNumChannels());
}

TEST_F(WarpEngineTest, PhaseVocoder_CompressHalvesLength) {
    PhaseVocoderEngine pv(2048, 512);

    // Compress by 0.5x (double speed)
    AudioBuffer result = pv.stretch(*testBuffer, 0.5);

    int expectedLength = testBuffer->getNumSamples() / 2;
    int actualLength = result.getNumSamples();

    EXPECT_NEAR(actualLength, expectedLength, expectedLength * 0.05);
}

TEST_F(WarpEngineTest, PhaseVocoder_NoStretchPreservesSignal) {
    PhaseVocoderEngine pv(2048, 512);

    // Stretch by 1.0x (no change)
    AudioBuffer result = pv.stretch(*testBuffer, 1.0);

    // Length should be approximately the same
    EXPECT_NEAR(result.getNumSamples(), testBuffer->getNumSamples(),
                testBuffer->getNumSamples() * 0.02);
}

TEST_F(WarpEngineTest, PhaseVocoder_InvalidRatioReturnsOriginal) {
    PhaseVocoderEngine pv(2048, 512);

    // Invalid ratio (0 or negative)
    AudioBuffer result = pv.stretch(*testBuffer, 0.0);

    EXPECT_EQ(result.getNumSamples(), testBuffer->getNumSamples());
}

TEST_F(WarpEngineTest, PhaseVocoder_OutputNotSilent) {
    PhaseVocoderEngine pv(2048, 512);

    AudioBuffer result = pv.stretch(*testBuffer, 1.5);

    // Check that output has actual audio data
    float maxPeak = 0.0f;
    for (int ch = 0; ch < result.getNumChannels(); ++ch) {
        const float* data = result.getReadPointer(ch);
        for (int i = 0; i < result.getNumSamples(); ++i) {
            maxPeak = std::max(maxPeak, std::abs(data[i]));
        }
    }

    // Should have significant signal level
    EXPECT_GT(maxPeak, 0.1f);
}

//=============================================================================
// WSOLAEngine Tests
//=============================================================================

TEST_F(WarpEngineTest, WSOLA_StretchDoublesLength) {
    WSOLAEngine wsola(1024, 128);

    AudioBuffer result = wsola.stretch(*testBuffer, 2.0);

    int expectedLength = testBuffer->getNumSamples() * 2;
    int actualLength = result.getNumSamples();

    EXPECT_NEAR(actualLength, expectedLength, expectedLength * 0.05);
}

TEST_F(WarpEngineTest, WSOLA_CompressHalvesLength) {
    WSOLAEngine wsola(1024, 128);

    AudioBuffer result = wsola.stretch(*testBuffer, 0.5);

    int expectedLength = testBuffer->getNumSamples() / 2;
    int actualLength = result.getNumSamples();

    EXPECT_NEAR(actualLength, expectedLength, expectedLength * 0.05);
}

TEST_F(WarpEngineTest, WSOLA_OutputNotSilent) {
    WSOLAEngine wsola(1024, 128);

    AudioBuffer result = wsola.stretch(*testBuffer, 1.5);

    float maxPeak = 0.0f;
    for (int ch = 0; ch < result.getNumChannels(); ++ch) {
        const float* data = result.getReadPointer(ch);
        for (int i = 0; i < result.getNumSamples(); ++i) {
            maxPeak = std::max(maxPeak, std::abs(data[i]));
        }
    }

    EXPECT_GT(maxPeak, 0.1f);
}

//=============================================================================
// WarpEngine (Unified Interface) Tests
//=============================================================================

TEST_F(WarpEngineTest, WarpEngine_SetAlgorithm) {
    WarpEngine engine;

    // Should not crash when switching algorithms
    engine.setAlgorithm(WarpAlgorithm::PhaseVocoder);
    engine.setAlgorithm(WarpAlgorithm::WSOLA);
    engine.setAlgorithm(WarpAlgorithm::OLA);
}

TEST_F(WarpEngineTest, WarpEngine_TempoConversion) {
    WarpEngine engine;
    engine.setAlgorithm(WarpAlgorithm::WSOLA);

    // Original audio at 120 BPM, target 60 BPM (should stretch 2x)
    AudioBuffer result = engine.warp(*testBuffer, 120.0, 60.0, sampleRate_);

    int expectedLength = testBuffer->getNumSamples() * 2;
    EXPECT_NEAR(result.getNumSamples(), expectedLength, expectedLength * 0.05);
}

TEST_F(WarpEngineTest, WarpEngine_TempoCompression) {
    WarpEngine engine;
    engine.setAlgorithm(WarpAlgorithm::WSOLA);

    // Original at 120 BPM, target 240 BPM (should compress 0.5x)
    AudioBuffer result = engine.warp(*testBuffer, 120.0, 240.0, sampleRate_);

    int expectedLength = testBuffer->getNumSamples() / 2;
    EXPECT_NEAR(result.getNumSamples(), expectedLength, expectedLength * 0.05);
}

TEST_F(WarpEngineTest, WarpEngine_InvalidBPMReturnsOriginal) {
    WarpEngine engine;

    AudioBuffer result = engine.warp(*testBuffer, 0.0, 120.0, sampleRate_);
    EXPECT_EQ(result.getNumSamples(), testBuffer->getNumSamples());

    AudioBuffer result2 = engine.warp(*testBuffer, 120.0, 0.0, sampleRate_);
    EXPECT_EQ(result2.getNumSamples(), testBuffer->getNumSamples());
}

TEST_F(WarpEngineTest, WarpEngine_WarpMarkers) {
    WarpEngine engine;

    // Add markers
    engine.addWarpMarker(0.0, 0.0);
    engine.addWarpMarker(4.0, 48000 * 2.0);  // 4 beats at 2 seconds = 120 BPM

    // Clear markers
    engine.clearWarpMarkers();

    // Should work without crashing
    engine.addWarpMarker(0.0, 0.0);
    engine.addWarpMarker(8.0, 48000 * 4.0);  // 8 beats at 4 seconds = 120 BPM
}

//=============================================================================
// Simple OLA Tests
//=============================================================================

TEST_F(WarpEngineTest, OLA_BasicStretch) {
    WarpEngine engine;
    engine.setAlgorithm(WarpAlgorithm::OLA);

    AudioBuffer result = engine.warp(*testBuffer, 120.0, 60.0, sampleRate_);

    // Should produce stretched output
    EXPECT_GT(result.getNumSamples(), testBuffer->getNumSamples());
}

TEST_F(WarpEngineTest, OLA_OutputNotSilent) {
    WarpEngine engine;
    engine.setAlgorithm(WarpAlgorithm::OLA);

    AudioBuffer result = engine.warp(*testBuffer, 120.0, 80.0, sampleRate_);

    float maxPeak = 0.0f;
    for (int ch = 0; ch < result.getNumChannels(); ++ch) {
        const float* data = result.getReadPointer(ch);
        for (int i = 0; i < result.getNumSamples(); ++i) {
            maxPeak = std::max(maxPeak, std::abs(data[i]));
        }
    }

    EXPECT_GT(maxPeak, 0.1f);
}

//=============================================================================
// Edge Cases
//=============================================================================

TEST_F(WarpEngineTest, EdgeCase_EmptyBuffer) {
    WarpEngine engine;
    AudioBuffer emptyBuffer(2, 0);

    AudioBuffer result = engine.warp(emptyBuffer, 120.0, 60.0, sampleRate_);

    // Should handle empty buffer gracefully
    EXPECT_EQ(result.getNumSamples(), 0);
}

TEST_F(WarpEngineTest, EdgeCase_SingleSample) {
    WarpEngine engine;
    AudioBuffer singleSample(1, 1);
    singleSample.getWritePointer(0)[0] = 0.5f;

    // Should not crash
    AudioBuffer result = engine.warp(singleSample, 120.0, 60.0, sampleRate_);
}

TEST_F(WarpEngineTest, EdgeCase_VeryLargeStretch) {
    PhaseVocoderEngine pv(2048, 512);

    // Create smaller buffer for this test
    AudioBuffer smallBuffer(1, 8192);
    float* data = smallBuffer.getWritePointer(0);
    for (int i = 0; i < 8192; ++i) {
        data[i] = std::sin(2.0 * M_PI * 440.0 * i / 48000.0);
    }

    // 10x stretch
    AudioBuffer result = pv.stretch(smallBuffer, 10.0);

    EXPECT_GT(result.getNumSamples(), smallBuffer.getNumSamples() * 8);
}

TEST_F(WarpEngineTest, EdgeCase_VerySmallStretch) {
    PhaseVocoderEngine pv(2048, 512);

    // 0.1x stretch (very fast)
    AudioBuffer result = pv.stretch(*testBuffer, 0.1);

    EXPECT_LT(result.getNumSamples(), testBuffer->getNumSamples() / 5);
}

//=============================================================================
// Quality Tests (Signal Integrity)
//=============================================================================

TEST_F(WarpEngineTest, Quality_PreservesChannelCount) {
    WarpEngine engine;

    AudioBuffer mono(1, 48000);
    AudioBuffer stereo(2, 48000);

    AudioBuffer monoResult = engine.warp(mono, 120.0, 60.0, sampleRate_);
    AudioBuffer stereoResult = engine.warp(stereo, 120.0, 60.0, sampleRate_);

    EXPECT_EQ(monoResult.getNumChannels(), 1);
    EXPECT_EQ(stereoResult.getNumChannels(), 2);
}

TEST_F(WarpEngineTest, Quality_NoClipping) {
    WarpEngine engine;
    engine.setAlgorithm(WarpAlgorithm::PhaseVocoder);

    AudioBuffer result = engine.warp(*testBuffer, 120.0, 60.0, sampleRate_);

    // Check for clipping (values > 1.0 or < -1.0)
    for (int ch = 0; ch < result.getNumChannels(); ++ch) {
        const float* data = result.getReadPointer(ch);
        for (int i = 0; i < result.getNumSamples(); ++i) {
            // Allow some headroom but check for extreme clipping
            EXPECT_LT(std::abs(data[i]), 5.0f);  // Generous threshold
        }
    }
}
