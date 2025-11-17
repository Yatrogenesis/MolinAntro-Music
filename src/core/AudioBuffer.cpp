#include "core/AudioBuffer.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace MolinAntro {
namespace Core {

// AudioBuffer implementation
AudioBuffer::AudioBuffer(int numChannels, int numSamples)
    : numChannels_(numChannels), numSamples_(numSamples) {
    if (numChannels <= 0 || numSamples <= 0) {
        throw std::invalid_argument("Invalid buffer dimensions");
    }
    allocate();
}

AudioBuffer::~AudioBuffer() {
    deallocate();
}

AudioBuffer::AudioBuffer(AudioBuffer&& other) noexcept
    : numChannels_(other.numChannels_),
      numSamples_(other.numSamples_),
      channels_(std::move(other.channels_)),
      data_(std::move(other.data_)) {
    other.numChannels_ = 0;
    other.numSamples_ = 0;
}

AudioBuffer& AudioBuffer::operator=(AudioBuffer&& other) noexcept {
    if (this != &other) {
        deallocate();
        numChannels_ = other.numChannels_;
        numSamples_ = other.numSamples_;
        channels_ = std::move(other.channels_);
        data_ = std::move(other.data_);
        other.numChannels_ = 0;
        other.numSamples_ = 0;
    }
    return *this;
}

const float* AudioBuffer::getReadPointer(int channel) const {
    if (channel < 0 || channel >= numChannels_) {
        throw std::out_of_range("Channel index out of range");
    }
    return channels_[channel];
}

float* AudioBuffer::getWritePointer(int channel) {
    if (channel < 0 || channel >= numChannels_) {
        throw std::out_of_range("Channel index out of range");
    }
    return channels_[channel];
}

void AudioBuffer::clear() {
    std::fill_n(data_.get(), numChannels_ * numSamples_, 0.0f);
}

void AudioBuffer::clearChannel(int channel) {
    std::fill_n(getWritePointer(channel), numSamples_, 0.0f);
}

void AudioBuffer::copyFrom(const AudioBuffer& source, int destChannel, int sourceChannel) {
    if (numSamples_ != source.numSamples_) {
        throw std::invalid_argument("Buffer size mismatch");
    }

    const float* src = source.getReadPointer(sourceChannel);
    float* dst = getWritePointer(destChannel);
    std::copy_n(src, numSamples_, dst);
}

void AudioBuffer::addFrom(const AudioBuffer& source, int destChannel, int sourceChannel, float gain) {
    if (numSamples_ != source.numSamples_) {
        throw std::invalid_argument("Buffer size mismatch");
    }

    const float* src = source.getReadPointer(sourceChannel);
    float* dst = getWritePointer(destChannel);

    for (int i = 0; i < numSamples_; ++i) {
        dst[i] += src[i] * gain;
    }
}

void AudioBuffer::applyGain(int channel, float gain) {
    float* data = getWritePointer(channel);
    for (int i = 0; i < numSamples_; ++i) {
        data[i] *= gain;
    }
}

void AudioBuffer::applyGainRamp(int channel, float startGain, float endGain) {
    float* data = getWritePointer(channel);
    float increment = (endGain - startGain) / numSamples_;
    float currentGain = startGain;

    for (int i = 0; i < numSamples_; ++i) {
        data[i] *= currentGain;
        currentGain += increment;
    }
}

float AudioBuffer::getRMSLevel(int channel) const {
    const float* data = getReadPointer(channel);
    double sum = 0.0;

    for (int i = 0; i < numSamples_; ++i) {
        sum += data[i] * data[i];
    }

    return static_cast<float>(std::sqrt(sum / numSamples_));
}

float AudioBuffer::getPeakLevel(int channel) const {
    const float* data = getReadPointer(channel);
    float peak = 0.0f;

    for (int i = 0; i < numSamples_; ++i) {
        peak = std::max(peak, std::abs(data[i]));
    }

    return peak;
}

void AudioBuffer::resize(int newNumChannels, int newNumSamples) {
    if (newNumChannels <= 0 || newNumSamples <= 0) {
        throw std::invalid_argument("Invalid buffer dimensions");
    }

    deallocate();
    numChannels_ = newNumChannels;
    numSamples_ = newNumSamples;
    allocate();
}

std::unique_ptr<AudioBuffer> AudioBuffer::clone() const {
    auto copy = std::make_unique<AudioBuffer>(numChannels_, numSamples_);
    std::copy_n(data_.get(), numChannels_ * numSamples_, copy->data_.get());
    return copy;
}

void AudioBuffer::allocate() {
    size_t totalSamples = numChannels_ * numSamples_;
    data_ = std::make_unique<float[]>(totalSamples);
    std::fill_n(data_.get(), totalSamples, 0.0f);

    channels_.resize(numChannels_);
    for (int i = 0; i < numChannels_; ++i) {
        channels_[i] = data_.get() + (i * numSamples_);
    }
}

void AudioBuffer::deallocate() {
    channels_.clear();
    data_.reset();
}

// RingBuffer implementation
RingBuffer::RingBuffer(int numChannels, int capacity)
    : numChannels_(numChannels), capacity_(capacity) {
    buffers_.resize(numChannels);
    for (auto& buffer : buffers_) {
        buffer.resize(capacity, 0.0f);
    }
}

int RingBuffer::write(const float* const* data, int numSamples) {
    int available = getAvailableWrite();
    int toWrite = std::min(numSamples, available);

    int writePos = writePosition_.load();

    for (int ch = 0; ch < numChannels_; ++ch) {
        for (int i = 0; i < toWrite; ++i) {
            int pos = (writePos + i) % capacity_;
            buffers_[ch][pos] = data[ch][i];
        }
    }

    writePosition_.store((writePos + toWrite) % capacity_);
    return toWrite;
}

int RingBuffer::read(float** data, int numSamples) {
    int available = getAvailableRead();
    int toRead = std::min(numSamples, available);

    int readPos = readPosition_.load();

    for (int ch = 0; ch < numChannels_; ++ch) {
        for (int i = 0; i < toRead; ++i) {
            int pos = (readPos + i) % capacity_;
            data[ch][i] = buffers_[ch][pos];
        }
    }

    readPosition_.store((readPos + toRead) % capacity_);
    return toRead;
}

int RingBuffer::getAvailableRead() const {
    int write = writePosition_.load();
    int read = readPosition_.load();
    return (write - read + capacity_) % capacity_;
}

int RingBuffer::getAvailableWrite() const {
    return capacity_ - getAvailableRead() - 1;
}

void RingBuffer::clear() {
    writePosition_.store(0);
    readPosition_.store(0);
    for (auto& buffer : buffers_) {
        std::fill(buffer.begin(), buffer.end(), 0.0f);
    }
}

} // namespace Core
} // namespace MolinAntro
