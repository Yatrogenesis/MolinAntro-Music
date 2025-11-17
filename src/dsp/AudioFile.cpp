#include "dsp/AudioFile.h"
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>

namespace MolinAntro {
namespace DSP {

// Simple WAV file header structure
struct WAVHeader {
    char riff[4];           // "RIFF"
    uint32_t fileSize;      // Total file size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmtSize;       // Format chunk size
    uint16_t audioFormat;   // Audio format (1 = PCM)
    uint16_t numChannels;   // Number of channels
    uint32_t sampleRate;    // Sample rate
    uint32_t byteRate;      // Byte rate
    uint16_t blockAlign;    // Block align
    uint16_t bitsPerSample; // Bits per sample
    char data[4];           // "data"
    uint32_t dataSize;      // Data chunk size
};

bool AudioFile::load(const std::string& filepath) {
    Format format = detectFormat(filepath);

    if (format == Format::WAV) {
        return loadWAV(filepath);
    }

    std::cerr << "[AudioFile] Unsupported format: " << filepath << std::endl;
    return false;
}

bool AudioFile::save(const std::string& filepath,
                     const Core::AudioBuffer& buffer,
                     int sampleRate,
                     int bitDepth) {
    Format format = detectFormat(filepath);

    if (format == Format::WAV) {
        return saveWAV(filepath, buffer, sampleRate, bitDepth);
    }

    std::cerr << "[AudioFile] Unsupported format: " << filepath << std::endl;
    return false;
}

AudioFile::Format AudioFile::detectFormat(const std::string& filepath) {
    size_t dotPos = filepath.find_last_of('.');
    if (dotPos == std::string::npos) {
        return Format::Unknown;
    }

    std::string ext = filepath.substr(dotPos + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == "wav") return Format::WAV;
    if (ext == "aif" || ext == "aiff") return Format::AIFF;
    if (ext == "flac") return Format::FLAC;
    if (ext == "mp3") return Format::MP3;

    return Format::Unknown;
}

bool AudioFile::loadWAV(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[AudioFile] Failed to open: " << filepath << std::endl;
        return false;
    }

    // Read header
    WAVHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

    // Validate WAV file
    if (std::strncmp(header.riff, "RIFF", 4) != 0 ||
        std::strncmp(header.wave, "WAVE", 4) != 0) {
        std::cerr << "[AudioFile] Invalid WAV file" << std::endl;
        return false;
    }

    // Set file info
    info_.format = Format::WAV;
    info_.sampleRate = header.sampleRate;
    info_.numChannels = header.numChannels;
    info_.bitDepth = header.bitsPerSample;
    info_.numSamples = header.dataSize / (header.numChannels * (header.bitsPerSample / 8));
    info_.durationSeconds = static_cast<double>(info_.numSamples) / info_.sampleRate;

    // Allocate buffer
    buffer_ = std::make_unique<Core::AudioBuffer>(info_.numChannels, info_.numSamples);

    // Read audio data
    if (info_.bitDepth == 16) {
        std::vector<int16_t> tempBuffer(info_.numSamples * info_.numChannels);
        file.read(reinterpret_cast<char*>(tempBuffer.data()), header.dataSize);

        // Convert to float and deinterleave
        for (int ch = 0; ch < info_.numChannels; ++ch) {
            float* channelData = buffer_->getWritePointer(ch);
            for (int64_t i = 0; i < info_.numSamples; ++i) {
                channelData[i] = tempBuffer[i * info_.numChannels + ch] / 32768.0f;
            }
        }
    } else if (info_.bitDepth == 24) {
        // 24-bit handling (simplified)
        std::vector<uint8_t> tempBuffer(header.dataSize);
        file.read(reinterpret_cast<char*>(tempBuffer.data()), header.dataSize);

        int bytesPerSample = 3;
        for (int ch = 0; ch < info_.numChannels; ++ch) {
            float* channelData = buffer_->getWritePointer(ch);
            for (int64_t i = 0; i < info_.numSamples; ++i) {
                int offset = (i * info_.numChannels + ch) * bytesPerSample;
                int32_t sample = (tempBuffer[offset + 2] << 16) |
                                (tempBuffer[offset + 1] << 8) |
                                tempBuffer[offset];

                // Sign extend
                if (sample & 0x800000) {
                    sample |= 0xFF000000;
                }

                channelData[i] = sample / 8388608.0f;
            }
        }
    } else if (info_.bitDepth == 32) {
        std::vector<float> tempBuffer(info_.numSamples * info_.numChannels);
        file.read(reinterpret_cast<char*>(tempBuffer.data()), header.dataSize);

        // Deinterleave
        for (int ch = 0; ch < info_.numChannels; ++ch) {
            float* channelData = buffer_->getWritePointer(ch);
            for (int64_t i = 0; i < info_.numSamples; ++i) {
                channelData[i] = tempBuffer[i * info_.numChannels + ch];
            }
        }
    }

    std::cout << "[AudioFile] Loaded: " << filepath << " ("
              << info_.numChannels << " ch, "
              << info_.sampleRate << " Hz, "
              << info_.bitDepth << " bit, "
              << info_.durationSeconds << " sec)" << std::endl;

    return true;
}

bool AudioFile::saveWAV(const std::string& filepath,
                        const Core::AudioBuffer& buffer,
                        int sampleRate,
                        int bitDepth) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[AudioFile] Failed to create: " << filepath << std::endl;
        return false;
    }

    int numChannels = buffer.getNumChannels();
    int numSamples = buffer.getNumSamples();
    int bytesPerSample = bitDepth / 8;
    uint32_t dataSize = numSamples * numChannels * bytesPerSample;

    // Prepare WAV header
    WAVHeader header;
    std::memcpy(header.riff, "RIFF", 4);
    header.fileSize = sizeof(WAVHeader) - 8 + dataSize;
    std::memcpy(header.wave, "WAVE", 4);
    std::memcpy(header.fmt, "fmt ", 4);
    header.fmtSize = 16;
    header.audioFormat = (bitDepth == 32) ? 3 : 1; // 3 = float, 1 = PCM
    header.numChannels = numChannels;
    header.sampleRate = sampleRate;
    header.byteRate = sampleRate * numChannels * bytesPerSample;
    header.blockAlign = numChannels * bytesPerSample;
    header.bitsPerSample = bitDepth;
    std::memcpy(header.data, "data", 4);
    header.dataSize = dataSize;

    // Write header
    file.write(reinterpret_cast<const char*>(&header), sizeof(WAVHeader));

    // Write audio data
    if (bitDepth == 16) {
        std::vector<int16_t> tempBuffer(numSamples * numChannels);

        // Interleave and convert to int16
        for (int i = 0; i < numSamples; ++i) {
            for (int ch = 0; ch < numChannels; ++ch) {
                float sample = buffer.getReadPointer(ch)[i];
                tempBuffer[i * numChannels + ch] = static_cast<int16_t>(std::clamp(sample, -1.0f, 1.0f) * 32767.0f);
            }
        }

        file.write(reinterpret_cast<const char*>(tempBuffer.data()), dataSize);
    } else if (bitDepth == 24) {
        std::vector<uint8_t> tempBuffer(dataSize);

        // Interleave and convert to 24-bit
        int bytesPerSample = 3;
        for (int i = 0; i < numSamples; ++i) {
            for (int ch = 0; ch < numChannels; ++ch) {
                float sample = std::clamp(buffer.getReadPointer(ch)[i], -1.0f, 1.0f);
                int32_t value = static_cast<int32_t>(sample * 8388607.0f);

                int offset = (i * numChannels + ch) * bytesPerSample;
                tempBuffer[offset] = value & 0xFF;
                tempBuffer[offset + 1] = (value >> 8) & 0xFF;
                tempBuffer[offset + 2] = (value >> 16) & 0xFF;
            }
        }

        file.write(reinterpret_cast<const char*>(tempBuffer.data()), dataSize);
    } else if (bitDepth == 32) {
        std::vector<float> tempBuffer(numSamples * numChannels);

        // Interleave
        for (int i = 0; i < numSamples; ++i) {
            for (int ch = 0; ch < numChannels; ++ch) {
                tempBuffer[i * numChannels + ch] = buffer.getReadPointer(ch)[i];
            }
        }

        file.write(reinterpret_cast<const char*>(tempBuffer.data()), dataSize);
    }

    std::cout << "[AudioFile] Saved: " << filepath << " ("
              << numChannels << " ch, "
              << sampleRate << " Hz, "
              << bitDepth << " bit)" << std::endl;

    return true;
}

} // namespace DSP
} // namespace MolinAntro
