#pragma once

#include "core/AudioBuffer.h"
#include <string>
#include <memory>
#include <vector>

namespace MolinAntro {
namespace DSP {

/**
 * @brief Audio file reader/writer supporting WAV, AIFF, FLAC
 */
class AudioFile {
public:
    enum class Format {
        WAV,
        AIFF,
        FLAC,
        MP3,
        Unknown
    };

    struct FileInfo {
        Format format = Format::Unknown;
        int sampleRate = 0;
        int numChannels = 0;
        int bitDepth = 0;
        int64_t numSamples = 0;
        double durationSeconds = 0.0;
    };

    AudioFile() = default;
    ~AudioFile() = default;

    /**
     * @brief Load an audio file
     * @param filepath Path to audio file
     * @return true if successful
     */
    bool load(const std::string& filepath);

    /**
     * @brief Save audio buffer to file
     * @param filepath Output path
     * @param buffer Audio data to write
     * @param sampleRate Sample rate
     * @param bitDepth Bit depth (16, 24, or 32)
     * @return true if successful
     */
    bool save(const std::string& filepath,
              const Core::AudioBuffer& buffer,
              int sampleRate,
              int bitDepth = 24);

    /**
     * @brief Get file information
     */
    const FileInfo& getInfo() const { return info_; }

    /**
     * @brief Get audio data
     */
    const Core::AudioBuffer& getBuffer() const { return *buffer_; }

    /**
     * @brief Get audio data (non-const)
     */
    Core::AudioBuffer& getBuffer() { return *buffer_; }

    /**
     * @brief Check if file is loaded
     */
    bool isLoaded() const { return buffer_ != nullptr; }

    /**
     * @brief Detect file format from extension
     */
    static Format detectFormat(const std::string& filepath);

    /**
     * @brief Get samples as flat vector (for compatibility)
     */
    std::vector<float> getSamples() const {
        if (!buffer_) return {};
        std::vector<float> samples;
        samples.reserve(info_.numSamples * info_.numChannels);
        for (int64_t i = 0; i < info_.numSamples; ++i) {
            for (int ch = 0; ch < info_.numChannels; ++ch) {
                samples.push_back(buffer_->getReadPointer(ch)[i]);
            }
        }
        return samples;
    }

    int getNumChannels() const { return info_.numChannels; }
    int getSampleRate() const { return info_.sampleRate; }

private:
    FileInfo info_;
    std::unique_ptr<Core::AudioBuffer> buffer_;

    // WAV format
    bool loadWAV(const std::string& filepath);
    bool saveWAV(const std::string& filepath, const Core::AudioBuffer& buffer, int sampleRate, int bitDepth);

    // AIFF format
    bool loadAIFF(const std::string& filepath);
    bool saveAIFF(const std::string& filepath, const Core::AudioBuffer& buffer, int sampleRate, int bitDepth);

    // FLAC format (lossless compression)
    bool loadFLAC(const std::string& filepath);
    bool saveFLAC(const std::string& filepath, const Core::AudioBuffer& buffer, int sampleRate, int bitDepth);

    // MP3 format (read-only, lossy)
    bool loadMP3(const std::string& filepath);

    // Helper functions
    static uint32_t readBigEndian32(const uint8_t* data);
    static uint16_t readBigEndian16(const uint8_t* data);
    static void writeBigEndian32(uint8_t* data, uint32_t value);
    static void writeBigEndian16(uint8_t* data, uint16_t value);
    static double readIEEE80(const uint8_t* data);
    static void writeIEEE80(uint8_t* data, double value);
};

} // namespace DSP
} // namespace MolinAntro
