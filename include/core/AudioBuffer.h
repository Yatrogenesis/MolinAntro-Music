#pragma once

#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <atomic>

namespace MolinAntro {
namespace Core {

/**
 * @brief Multi-channel audio buffer with RAII and alignment for SIMD
 */
class AudioBuffer {
public:
    /**
     * @brief Construct audio buffer with specified channels and size
     */
    AudioBuffer(int numChannels, int numSamples);

    /**
     * @brief Destructor
     */
    ~AudioBuffer();

    // Move semantics
    AudioBuffer(AudioBuffer&& other) noexcept;
    AudioBuffer& operator=(AudioBuffer&& other) noexcept;

    // Disable copying (use clone() instead)
    AudioBuffer(const AudioBuffer&) = delete;
    AudioBuffer& operator=(const AudioBuffer&) = delete;

    /**
     * @brief Get number of channels
     */
    int getNumChannels() const { return numChannels_; }

    /**
     * @brief Get number of samples per channel
     */
    int getNumSamples() const { return numSamples_; }

    /**
     * @brief Get read pointer for a channel
     */
    const float* getReadPointer(int channel) const;

    /**
     * @brief Get write pointer for a channel
     */
    float* getWritePointer(int channel);

    /**
     * @brief Clear all channels to zero
     */
    void clear();

    /**
     * @brief Clear a specific channel to zero
     */
    void clearChannel(int channel);

    /**
     * @brief Copy data from another buffer
     */
    void copyFrom(const AudioBuffer& source, int destChannel, int sourceChannel);

    /**
     * @brief Add data from another buffer
     */
    void addFrom(const AudioBuffer& source, int destChannel, int sourceChannel, float gain = 1.0f);

    /**
     * @brief Apply gain to a channel
     */
    void applyGain(int channel, float gain);

    /**
     * @brief Apply gain ramp to a channel
     */
    void applyGainRamp(int channel, float startGain, float endGain);

    /**
     * @brief Get RMS (Root Mean Square) level for a channel
     */
    float getRMSLevel(int channel) const;

    /**
     * @brief Get peak level for a channel
     */
    float getPeakLevel(int channel) const;

    /**
     * @brief Resize the buffer (will clear existing data)
     */
    void resize(int newNumChannels, int newNumSamples);

    /**
     * @brief Create a deep copy of this buffer
     */
    std::unique_ptr<AudioBuffer> clone() const;

private:
    int numChannels_;
    int numSamples_;
    std::vector<float*> channels_;
    std::unique_ptr<float[]> data_;

    void allocate();
    void deallocate();
};

/**
 * @brief Ring buffer for lock-free audio streaming
 */
class RingBuffer {
public:
    RingBuffer(int numChannels, int capacity);
    ~RingBuffer() = default;

    /**
     * @brief Write samples to the ring buffer
     * @return Number of samples actually written
     */
    int write(const float* const* data, int numSamples);

    /**
     * @brief Read samples from the ring buffer
     * @return Number of samples actually read
     */
    int read(float** data, int numSamples);

    /**
     * @brief Get number of samples available to read
     */
    int getAvailableRead() const;

    /**
     * @brief Get number of samples available to write
     */
    int getAvailableWrite() const;

    /**
     * @brief Clear the buffer
     */
    void clear();

private:
    int numChannels_;
    int capacity_;
    std::atomic<int> writePosition_{0};
    std::atomic<int> readPosition_{0};
    std::vector<std::vector<float>> buffers_;
};

} // namespace Core
} // namespace MolinAntro
