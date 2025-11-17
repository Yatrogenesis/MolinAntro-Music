#pragma once

#include <atomic>
#include <memory>
#include <vector>
#include <functional>
#include <mutex>
#include <thread>

namespace MolinAntro {
namespace Core {

/**
 * @brief Main audio engine responsible for real-time audio processing
 *
 * Thread-safe audio engine with lock-free processing for real-time performance.
 * Supports sample rates from 8kHz to 384kHz with configurable buffer sizes.
 */
class AudioEngine {
public:
    struct Config {
        int sampleRate = 48000;
        int bufferSize = 512;
        int numInputChannels = 2;
        int numOutputChannels = 2;
        int bitDepth = 24;
    };

    enum class State {
        Stopped,
        Playing,
        Recording,
        Paused
    };

    AudioEngine();
    ~AudioEngine();

    // Prevent copying
    AudioEngine(const AudioEngine&) = delete;
    AudioEngine& operator=(const AudioEngine&) = delete;

    /**
     * @brief Initialize the audio engine with given configuration
     * @param config Engine configuration
     * @return true if initialization succeeded
     */
    bool initialize(const Config& config);

    /**
     * @brief Start the audio engine
     * @return true if started successfully
     */
    bool start();

    /**
     * @brief Stop the audio engine
     */
    void stop();

    /**
     * @brief Pause audio processing
     */
    void pause();

    /**
     * @brief Resume audio processing
     */
    void resume();

    /**
     * @brief Get current engine state
     */
    State getState() const { return state_.load(); }

    /**
     * @brief Get current sample rate
     */
    int getSampleRate() const { return config_.sampleRate; }

    /**
     * @brief Get current buffer size
     */
    int getBufferSize() const { return config_.bufferSize; }

    /**
     * @brief Get CPU usage percentage (0-100)
     */
    float getCPUUsage() const { return cpuUsage_.load(); }

    /**
     * @brief Process audio callback (called from audio thread)
     * @param outputBuffer Output audio buffer
     * @param numSamples Number of samples to process
     */
    void processAudio(float** outputBuffer, int numSamples);

    /**
     * @brief Set audio processing callback
     */
    using AudioCallback = std::function<void(float**, float**, int)>;
    void setAudioCallback(AudioCallback callback);

private:
    Config config_;
    std::atomic<State> state_{State::Stopped};
    std::atomic<float> cpuUsage_{0.0f};

    std::unique_ptr<float[]> inputBuffer_;
    std::unique_ptr<float[]> outputBuffer_;

    AudioCallback audioCallback_;
    std::mutex callbackMutex_;

    // Performance monitoring
    std::chrono::high_resolution_clock::time_point lastProcessTime_;

    void updateCPUUsage(std::chrono::microseconds processingTime);
};

} // namespace Core
} // namespace MolinAntro
