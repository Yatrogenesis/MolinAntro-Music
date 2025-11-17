#include "core/AudioEngine.h"
#include <iostream>
#include <cmath>

namespace MolinAntro {
namespace Core {

AudioEngine::AudioEngine() {
    std::cout << "[AudioEngine] Constructed" << std::endl;
}

AudioEngine::~AudioEngine() {
    if (state_.load() != State::Stopped) {
        stop();
    }
    std::cout << "[AudioEngine] Destroyed" << std::endl;
}

bool AudioEngine::initialize(const Config& config) {
    if (state_.load() != State::Stopped) {
        std::cerr << "[AudioEngine] Cannot initialize while running" << std::endl;
        return false;
    }

    config_ = config;

    // Allocate audio buffers (aligned for SIMD)
    inputBuffer_ = std::make_unique<float[]>(config_.bufferSize * config_.numInputChannels);
    outputBuffer_ = std::make_unique<float[]>(config_.bufferSize * config_.numOutputChannels);

    std::fill_n(inputBuffer_.get(), config_.bufferSize * config_.numInputChannels, 0.0f);
    std::fill_n(outputBuffer_.get(), config_.bufferSize * config_.numOutputChannels, 0.0f);

    std::cout << "[AudioEngine] Initialized: " << config_.sampleRate << "Hz, "
              << config_.bufferSize << " samples, "
              << config_.numInputChannels << " in, "
              << config_.numOutputChannels << " out" << std::endl;

    return true;
}

bool AudioEngine::start() {
    State expected = State::Stopped;
    if (!state_.compare_exchange_strong(expected, State::Playing)) {
        expected = State::Paused;
        if (!state_.compare_exchange_strong(expected, State::Playing)) {
            std::cerr << "[AudioEngine] Cannot start from current state" << std::endl;
            return false;
        }
    }

    std::cout << "[AudioEngine] Started" << std::endl;
    return true;
}

void AudioEngine::stop() {
    state_.store(State::Stopped);
    std::cout << "[AudioEngine] Stopped" << std::endl;
}

void AudioEngine::pause() {
    State expected = State::Playing;
    if (state_.compare_exchange_strong(expected, State::Paused)) {
        std::cout << "[AudioEngine] Paused" << std::endl;
    }
}

void AudioEngine::resume() {
    State expected = State::Paused;
    if (state_.compare_exchange_strong(expected, State::Playing)) {
        std::cout << "[AudioEngine] Resumed" << std::endl;
    }
}

void AudioEngine::processAudio(float** outputBuffer, int numSamples) {
    auto startTime = std::chrono::high_resolution_clock::now();

    if (state_.load() != State::Playing && state_.load() != State::Recording) {
        // Clear output buffer when not playing
        for (int ch = 0; ch < config_.numOutputChannels; ++ch) {
            std::fill_n(outputBuffer[ch], numSamples, 0.0f);
        }
        return;
    }

    // Call user-provided audio callback if set
    {
        std::lock_guard<std::mutex> lock(callbackMutex_);
        if (audioCallback_) {
            float* inputs[2] = {inputBuffer_.get(), inputBuffer_.get() + config_.bufferSize};
            audioCallback_(inputs, outputBuffer, numSamples);
        } else {
            // Default: silence
            for (int ch = 0; ch < config_.numOutputChannels; ++ch) {
                std::fill_n(outputBuffer[ch], numSamples, 0.0f);
            }
        }
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);
    updateCPUUsage(duration);
}

void AudioEngine::setAudioCallback(AudioCallback callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    audioCallback_ = std::move(callback);
}

void AudioEngine::updateCPUUsage(std::chrono::microseconds processingTime) {
    // Calculate available time for processing
    double availableTime = (config_.bufferSize * 1000000.0) / config_.sampleRate;

    // CPU usage percentage
    float usage = (processingTime.count() / availableTime) * 100.0f;

    // Smooth CPU usage with simple low-pass filter
    float smoothed = cpuUsage_.load() * 0.9f + usage * 0.1f;
    cpuUsage_.store(smoothed);
}

} // namespace Core
} // namespace MolinAntro
