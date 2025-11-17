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

private:
    FileInfo info_;
    std::unique_ptr<Core::AudioBuffer> buffer_;

    bool loadWAV(const std::string& filepath);
    bool saveWAV(const std::string& filepath, const Core::AudioBuffer& buffer, int sampleRate, int bitDepth);
};

} // namespace DSP
} // namespace MolinAntro
