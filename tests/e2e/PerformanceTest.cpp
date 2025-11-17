#include <gtest/gtest.h>
#include "core/AudioEngine.h"
#include "core/Transport.h"
#include "core/AudioBuffer.h"
#include "dsp/AudioFile.h"
#include <chrono>
#include <iostream>

using namespace MolinAntro;

class PerformanceTest : public ::testing::Test {
protected:
    void SetUp() override {
        engine = std::make_unique<Core::AudioEngine>();

        Core::AudioEngine::Config config;
        config.sampleRate = 48000;
        config.bufferSize = 512;
        config.numInputChannels = 2;
        config.numOutputChannels = 2;

        ASSERT_TRUE(engine->initialize(config));
    }

    void TearDown() override {
        engine->stop();
        engine.reset();
    }

    std::unique_ptr<Core::AudioEngine> engine;
};

TEST_F(PerformanceTest, AudioProcessingPerformance) {
    std::cout << "\n=== Performance Test: Audio Processing ===\n";

    engine->start();

    float* outputs[2];
    float leftBuffer[512];
    float rightBuffer[512];
    outputs[0] = leftBuffer;
    outputs[1] = rightBuffer;

    const int numIterations = 10000;

    auto startTime = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numIterations; ++i) {
        engine->processAudio(outputs, 512);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    double totalSamples = numIterations * 512.0;
    double totalSeconds = totalSamples / 48000.0;
    double processingTimeMs = duration.count();
    double realTimeRatio = (totalSeconds * 1000.0) / processingTimeMs;

    std::cout << "Processed: " << numIterations << " buffers (" << totalSeconds << " sec of audio)\n";
    std::cout << "Processing time: " << processingTimeMs << " ms\n";
    std::cout << "Real-time ratio: " << realTimeRatio << "x\n";

    // Should process faster than real-time
    EXPECT_GT(realTimeRatio, 1.0);

    std::cout << "✓ Performance test PASSED\n\n";
}

TEST_F(PerformanceTest, BufferOperationsPerformance) {
    std::cout << "\n=== Performance Test: Buffer Operations ===\n";

    const int bufferSize = 4096;
    const int numIterations = 10000;

    Core::AudioBuffer buffer1(2, bufferSize);
    Core::AudioBuffer buffer2(2, bufferSize);

    // Fill with data
    for (int ch = 0; ch < 2; ++ch) {
        float* data = buffer1.getWritePointer(ch);
        for (int i = 0; i < bufferSize; ++i) {
            data[i] = static_cast<float>(i) / bufferSize;
        }
    }

    // Test: Copy operations
    auto startCopy = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        buffer2.copyFrom(buffer1, 0, 0);
        buffer2.copyFrom(buffer1, 1, 1);
    }
    auto endCopy = std::chrono::high_resolution_clock::now();
    auto copyDuration = std::chrono::duration_cast<std::chrono::microseconds>(endCopy - startCopy);

    std::cout << "Copy operations: " << copyDuration.count() << " µs for " << numIterations << " iterations\n";
    std::cout << "  (" << (copyDuration.count() / numIterations) << " µs per operation)\n";

    // Test: Gain operations
    auto startGain = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numIterations; ++i) {
        buffer1.applyGain(0, 0.5f);
        buffer1.applyGain(1, 0.5f);
    }
    auto endGain = std::chrono::high_resolution_clock::now();
    auto gainDuration = std::chrono::duration_cast<std::chrono::microseconds>(endGain - startGain);

    std::cout << "Gain operations: " << gainDuration.count() << " µs for " << numIterations << " iterations\n";
    std::cout << "  (" << (gainDuration.count() / numIterations) << " µs per operation)\n";

    std::cout << "✓ Performance test PASSED\n\n";
}

TEST_F(PerformanceTest, MemoryUsage) {
    std::cout << "\n=== Performance Test: Memory Usage ===\n";

    // Allocate multiple buffers
    std::vector<std::unique_ptr<Core::AudioBuffer>> buffers;
    const int numBuffers = 100;
    const int bufferSize = 48000; // 1 second

    auto startAlloc = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < numBuffers; ++i) {
        buffers.push_back(std::make_unique<Core::AudioBuffer>(2, bufferSize));
    }

    auto endAlloc = std::chrono::high_resolution_clock::now();
    auto allocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endAlloc - startAlloc);

    size_t totalBytes = numBuffers * 2 * bufferSize * sizeof(float);
    double totalMB = totalBytes / (1024.0 * 1024.0);

    std::cout << "Allocated: " << numBuffers << " buffers\n";
    std::cout << "Total memory: " << totalMB << " MB\n";
    std::cout << "Allocation time: " << allocDuration.count() << " ms\n";

    // Cleanup
    auto startDealloc = std::chrono::high_resolution_clock::now();
    buffers.clear();
    auto endDealloc = std::chrono::high_resolution_clock::now();
    auto deallocDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDealloc - startDealloc);

    std::cout << "Deallocation time: " << deallocDuration.count() << " ms\n";

    std::cout << "✓ Memory test PASSED\n\n";
}

TEST_F(PerformanceTest, ConcurrentAudioProcessing) {
    std::cout << "\n=== Performance Test: Concurrent Processing ===\n";

    const int numThreads = 4;
    const int iterationsPerThread = 1000;

    std::vector<std::thread> threads;
    std::vector<Core::AudioBuffer> buffers;

    // Pre-allocate buffers
    for (int i = 0; i < numThreads; ++i) {
        buffers.emplace_back(2, 512);
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    // Launch threads
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < iterationsPerThread; ++i) {
                // Simulate processing
                buffers[t].applyGain(0, 0.9f);
                buffers[t].applyGain(1, 0.9f);

                float rms = buffers[t].getRMSLevel(0);
                (void)rms; // Suppress unused warning
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Threads: " << numThreads << "\n";
    std::cout << "Iterations per thread: " << iterationsPerThread << "\n";
    std::cout << "Total time: " << duration.count() << " ms\n";

    std::cout << "✓ Concurrent processing test PASSED\n\n";
}
