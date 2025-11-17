#include <gtest/gtest.h>
#include "core/AudioEngine.h"
#include "core/Transport.h"
#include "core/AudioBuffer.h"
#include "dsp/AudioFile.h"
#include <filesystem>
#include <cmath>

using namespace MolinAntro;

/**
 * @brief End-to-End test simulating complete DAW workflow
 *
 * This test validates the entire system from initialization to teardown,
 * including:
 * - Engine initialization
 * - Transport control
 * - Audio processing
 * - File I/O
 * - Recording simulation
 * - Playback simulation
 * - Export workflow
 */
class CompleteWorkflowTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir = "/tmp/molinantro_e2e";
        std::filesystem::create_directories(testDir);

        // Initialize engine
        engine = std::make_unique<Core::AudioEngine>();
        transport = std::make_unique<Core::Transport>();

        Core::AudioEngine::Config config;
        config.sampleRate = 48000;
        config.bufferSize = 512;
        config.numInputChannels = 2;
        config.numOutputChannels = 2;

        ASSERT_TRUE(engine->initialize(config));

        transport->setBPM(120.0);
        transport->setTimeSignature(4, 4);
    }

    void TearDown() override {
        engine->stop();
        engine.reset();
        transport.reset();

        if (std::filesystem::exists(testDir)) {
            std::filesystem::remove_all(testDir);
        }
    }

    std::unique_ptr<Core::AudioEngine> engine;
    std::unique_ptr<Core::Transport> transport;
    std::string testDir;
};

TEST_F(CompleteWorkflowTest, E2E_RecordPlaybackExport) {
    std::cout << "\n=== E2E Test: Record -> Playback -> Export ===\n";

    // STEP 1: Simulate Recording
    std::cout << "Step 1: Recording simulation...\n";

    transport->record();
    engine->start();

    ASSERT_TRUE(transport->isRecording());

    // Create recorded buffer (simulate microphone input)
    Core::AudioBuffer recordedBuffer(2, 48000); // 1 second

    // Generate test signal (440Hz sine wave)
    for (int ch = 0; ch < 2; ++ch) {
        float* data = recordedBuffer.getWritePointer(ch);
        for (int i = 0; i < 48000; ++i) {
            data[i] = 0.5f * std::sin(2.0 * M_PI * 440.0 * i / 48000.0);
        }
    }

    std::cout << "  ✓ Recorded 1 second of audio\n";

    // STEP 2: Stop Recording
    transport->stop();
    ASSERT_FALSE(transport->isRecording());
    std::cout << "  ✓ Stopped recording\n";

    // STEP 3: Save Recorded Audio
    std::cout << "\nStep 2: Saving recorded audio...\n";

    std::string recordedFile = testDir + "/recorded.wav";
    DSP::AudioFile saveFile;
    ASSERT_TRUE(saveFile.save(recordedFile, recordedBuffer, 48000, 24));
    ASSERT_TRUE(std::filesystem::exists(recordedFile));

    std::cout << "  ✓ Saved to: " << recordedFile << "\n";

    // STEP 4: Load Audio for Playback
    std::cout << "\nStep 3: Loading audio for playback...\n";

    DSP::AudioFile loadFile;
    ASSERT_TRUE(loadFile.load(recordedFile));

    auto info = loadFile.getInfo();
    EXPECT_EQ(info.sampleRate, 48000);
    EXPECT_EQ(info.numChannels, 2);
    EXPECT_EQ(info.numSamples, 48000);

    std::cout << "  ✓ Loaded: " << info.numChannels << " channels, "
              << info.sampleRate << " Hz, "
              << info.durationSeconds << " seconds\n";

    // STEP 5: Playback Simulation
    std::cout << "\nStep 4: Playback simulation...\n";

    const Core::AudioBuffer& playbackBuffer = loadFile.getBuffer();

    transport->play();
    engine->start();

    ASSERT_TRUE(transport->isPlaying());

    // Simulate playback processing
    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    int totalSamplesProcessed = 0;
    int playbackPosition = 0;

    while (playbackPosition < playbackBuffer.getNumSamples()) {
        // Process buffer
        transport->update(512, 48000);
        engine->processAudio(outputs, 512);

        int samplesToProcess = std::min(512, playbackBuffer.getNumSamples() - playbackPosition);

        // Copy from playback buffer to output
        for (int ch = 0; ch < 2; ++ch) {
            const float* src = playbackBuffer.getReadPointer(ch) + playbackPosition;
            std::copy(src, src + samplesToProcess, outputs[ch]);
        }

        playbackPosition += samplesToProcess;
        totalSamplesProcessed += samplesToProcess;
    }

    EXPECT_EQ(totalSamplesProcessed, 48000);
    std::cout << "  ✓ Played back " << totalSamplesProcessed << " samples\n";

    auto timeInfo = transport->getTimeInfo();
    std::cout << "  ✓ Final position: " << timeInfo.timeInSeconds << " seconds\n";

    transport->stop();

    // STEP 6: Apply Processing and Export
    std::cout << "\nStep 5: Processing and export...\n";

    auto processedBufferPtr = playbackBuffer.clone();
    Core::AudioBuffer& processedBuffer = *processedBufferPtr;

    // Apply gain reduction (mastering simulation)
    for (int ch = 0; ch < processedBuffer.getNumChannels(); ++ch) {
        processedBuffer.applyGain(ch, 0.8f);
    }

    std::cout << "  ✓ Applied gain reduction (0.8x)\n";

    // Export processed audio
    std::string exportedFile = testDir + "/exported.wav";
    DSP::AudioFile exportFile;
    ASSERT_TRUE(exportFile.save(exportedFile, processedBuffer, 48000, 16));
    ASSERT_TRUE(std::filesystem::exists(exportedFile));

    std::cout << "  ✓ Exported to: " << exportedFile << "\n";

    // STEP 7: Verify Export
    std::cout << "\nStep 6: Verifying export...\n";

    DSP::AudioFile verifyFile;
    ASSERT_TRUE(verifyFile.load(exportedFile));

    const Core::AudioBuffer& verifiedBuffer = verifyFile.getBuffer();

    // Check that gain was applied correctly (with reasonable tolerance)
    float originalRMS = recordedBuffer.getRMSLevel(0);
    float processedRMS = verifiedBuffer.getRMSLevel(0);

    EXPECT_NEAR(processedRMS, originalRMS * 0.8f, 0.02f);  // Conservative tolerance

    std::cout << "  ✓ Original RMS: " << originalRMS << "\n";
    std::cout << "  ✓ Processed RMS: " << processedRMS << "\n";
    std::cout << "  ✓ Gain ratio verified\n";

    std::cout << "\n=== E2E Test PASSED ✓ ===\n\n";
}

TEST_F(CompleteWorkflowTest, E2E_MultiTrackSession) {
    std::cout << "\n=== E2E Test: Multi-Track Session ===\n";

    const int numTracks = 4;
    const int numSamples = 24000; // 0.5 seconds
    std::vector<Core::AudioBuffer> tracks;

    // STEP 1: Create multiple tracks with different content
    std::cout << "Step 1: Creating " << numTracks << " tracks...\n";

    for (int track = 0; track < numTracks; ++track) {
        tracks.emplace_back(2, numSamples);

        // Generate different frequency for each track
        double frequency = 220.0 * (track + 1); // A3, A4, A5, A6

        for (int ch = 0; ch < 2; ++ch) {
            float* data = tracks[track].getWritePointer(ch);
            for (int i = 0; i < numSamples; ++i) {
                data[i] = 0.25f * std::sin(2.0 * M_PI * frequency * i / 48000.0);
            }
        }

        std::cout << "  ✓ Track " << (track + 1) << ": " << frequency << " Hz\n";
    }

    // STEP 2: Mix all tracks
    std::cout << "\nStep 2: Mixing tracks...\n";

    Core::AudioBuffer mixBuffer(2, numSamples);
    mixBuffer.clear();

    for (const auto& track : tracks) {
        mixBuffer.addFrom(track, 0, 0, 1.0f);
        mixBuffer.addFrom(track, 1, 1, 1.0f);
    }

    std::cout << "  ✓ Mixed " << numTracks << " tracks\n";

    // STEP 3: Apply master limiter (prevent clipping)
    float peak = mixBuffer.getPeakLevel(0);
    if (peak > 1.0f) {
        float gain = 0.99f / peak;
        mixBuffer.applyGain(0, gain);
        mixBuffer.applyGain(1, gain);
        std::cout << "  ✓ Applied limiter (gain: " << gain << ")\n";
    }

    // STEP 4: Export master
    std::cout << "\nStep 3: Exporting master...\n";

    std::string masterFile = testDir + "/master.wav";
    DSP::AudioFile exportFile;
    ASSERT_TRUE(exportFile.save(masterFile, mixBuffer, 48000, 24));

    std::cout << "  ✓ Exported master to: " << masterFile << "\n";

    // STEP 5: Verify master
    DSP::AudioFile verifyFile;
    ASSERT_TRUE(verifyFile.load(masterFile));

    const Core::AudioBuffer& masterBuffer = verifyFile.getBuffer();
    float masterPeak = masterBuffer.getPeakLevel(0);

    EXPECT_LE(masterPeak, 1.0f);
    std::cout << "  ✓ Master peak level: " << masterPeak << " (OK)\n";

    std::cout << "\n=== E2E Test PASSED ✓ ===\n\n";
}

TEST_F(CompleteWorkflowTest, E2E_TempoChangeDuringPlayback) {
    std::cout << "\n=== E2E Test: Tempo Change During Playback ===\n";

    // Start playback
    transport->setBPM(120.0);
    transport->play();
    engine->start();

    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    // Process at 120 BPM
    std::cout << "Processing at 120 BPM...\n";
    for (int i = 0; i < 5; ++i) {
        transport->update(512, 48000);
        engine->processAudio(outputs, 512);
    }

    auto timeInfo1 = transport->getTimeInfo();
    std::cout << "  Position: Bar " << timeInfo1.bar << ", Beat " << timeInfo1.beat << "\n";

    // Change tempo
    std::cout << "\nChanging tempo to 140 BPM...\n";
    transport->setBPM(140.0);

    // Continue processing
    for (int i = 0; i < 5; ++i) {
        transport->update(512, 48000);
        engine->processAudio(outputs, 512);
    }

    auto timeInfo2 = transport->getTimeInfo();
    std::cout << "  Position: Bar " << timeInfo2.bar << ", Beat " << timeInfo2.beat << "\n";

    EXPECT_DOUBLE_EQ(transport->getBPM(), 140.0);
    EXPECT_GT(timeInfo2.samplePosition, timeInfo1.samplePosition);

    transport->stop();
    engine->stop();

    std::cout << "\n=== E2E Test PASSED ✓ ===\n\n";
}
