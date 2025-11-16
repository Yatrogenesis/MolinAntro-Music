#pragma once

#include <atomic>
#include <chrono>
#include <functional>

namespace MolinAntro {
namespace Core {

/**
 * @brief Transport controller for playback, recording, and tempo management
 */
class Transport {
public:
    struct TimeInfo {
        double samplePosition = 0.0;
        double timeInSeconds = 0.0;
        int bar = 1;
        int beat = 1;
        int tick = 0;
        double bpm = 120.0;
        int numerator = 4;
        int denominator = 4;
        bool isPlaying = false;
        bool isRecording = false;
    };

    Transport();
    ~Transport() = default;

    /**
     * @brief Start playback
     */
    void play();

    /**
     * @brief Stop playback and reset to start
     */
    void stop();

    /**
     * @brief Pause playback at current position
     */
    void pause();

    /**
     * @brief Start recording
     */
    void record();

    /**
     * @brief Set tempo in BPM
     */
    void setBPM(double bpm);

    /**
     * @brief Get current tempo
     */
    double getBPM() const { return timeInfo_.bpm; }

    /**
     * @brief Set time signature
     */
    void setTimeSignature(int numerator, int denominator);

    /**
     * @brief Set playback position in samples
     */
    void setPositionSamples(double samples);

    /**
     * @brief Set playback position in seconds
     */
    void setPositionSeconds(double seconds, int sampleRate);

    /**
     * @brief Set playback position in bars
     */
    void setPositionBars(int bar, int beat);

    /**
     * @brief Update transport state (call from audio thread)
     */
    void update(int numSamples, int sampleRate);

    /**
     * @brief Get current time information
     */
    TimeInfo getTimeInfo() const { return timeInfo_; }

    /**
     * @brief Check if transport is playing
     */
    bool isPlaying() const { return timeInfo_.isPlaying; }

    /**
     * @brief Check if transport is recording
     */
    bool isRecording() const { return timeInfo_.isRecording; }

    /**
     * @brief Set transport state change callback
     */
    using StateCallback = std::function<void(const TimeInfo&)>;
    void setStateCallback(StateCallback callback);

private:
    TimeInfo timeInfo_;
    StateCallback stateCallback_;
    std::atomic<bool> playRequested_{false};
    std::atomic<bool> stopRequested_{false};
    std::atomic<bool> pauseRequested_{false};

    void calculateBarBeatTick(int sampleRate);
    void notifyStateChange();
};

} // namespace Core
} // namespace MolinAntro
