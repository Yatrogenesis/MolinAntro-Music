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

    switch (format) {
        case Format::WAV:
            return loadWAV(filepath);
        case Format::AIFF:
            return loadAIFF(filepath);
        case Format::FLAC:
            return loadFLAC(filepath);
        case Format::MP3:
            return loadMP3(filepath);
        default:
            std::cerr << "[AudioFile] Unsupported format: " << filepath << std::endl;
            return false;
    }
}

bool AudioFile::save(const std::string& filepath,
                     const Core::AudioBuffer& buffer,
                     int sampleRate,
                     int bitDepth) {
    Format format = detectFormat(filepath);

    switch (format) {
        case Format::WAV:
            return saveWAV(filepath, buffer, sampleRate, bitDepth);
        case Format::AIFF:
            return saveAIFF(filepath, buffer, sampleRate, bitDepth);
        case Format::FLAC:
            return saveFLAC(filepath, buffer, sampleRate, bitDepth);
        case Format::MP3:
            std::cerr << "[AudioFile] MP3 writing not supported (lossy format)" << std::endl;
            return false;
        default:
            std::cerr << "[AudioFile] Unsupported format: " << filepath << std::endl;
            return false;
    }
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

// ============================================================================
// Big-Endian Helper Functions (for AIFF)
// ============================================================================

uint32_t AudioFile::readBigEndian32(const uint8_t* data) {
    return (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3];
}

uint16_t AudioFile::readBigEndian16(const uint8_t* data) {
    return (data[0] << 8) | data[1];
}

void AudioFile::writeBigEndian32(uint8_t* data, uint32_t value) {
    data[0] = (value >> 24) & 0xFF;
    data[1] = (value >> 16) & 0xFF;
    data[2] = (value >> 8) & 0xFF;
    data[3] = value & 0xFF;
}

void AudioFile::writeBigEndian16(uint8_t* data, uint16_t value) {
    data[0] = (value >> 8) & 0xFF;
    data[1] = value & 0xFF;
}

// IEEE 80-bit extended precision (used for AIFF sample rate)
double AudioFile::readIEEE80(const uint8_t* data) {
    int sign = (data[0] >> 7) & 1;
    int exp = ((data[0] & 0x7F) << 8) | data[1];
    uint64_t mantissa = 0;
    for (int i = 0; i < 8; ++i) {
        mantissa = (mantissa << 8) | data[2 + i];
    }

    if (exp == 0 && mantissa == 0) return 0.0;
    if (exp == 0x7FFF) return sign ? -INFINITY : INFINITY;

    double value = static_cast<double>(mantissa) / (1ULL << 63);
    value = std::ldexp(value, exp - 16383);
    return sign ? -value : value;
}

void AudioFile::writeIEEE80(uint8_t* data, double value) {
    if (value == 0.0) {
        std::memset(data, 0, 10);
        return;
    }

    int sign = value < 0 ? 1 : 0;
    if (sign) value = -value;

    int exp;
    double mantissa = std::frexp(value, &exp);
    exp += 16382;

    uint64_t mant = static_cast<uint64_t>(mantissa * (1ULL << 63));

    data[0] = (sign << 7) | ((exp >> 8) & 0x7F);
    data[1] = exp & 0xFF;
    for (int i = 0; i < 8; ++i) {
        data[9 - i] = mant & 0xFF;
        mant >>= 8;
    }
}

// ============================================================================
// AIFF Format Implementation
// ============================================================================

bool AudioFile::loadAIFF(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[AudioFile] Failed to open: " << filepath << std::endl;
        return false;
    }

    // Read FORM header
    char formHeader[12];
    file.read(formHeader, 12);

    if (std::strncmp(formHeader, "FORM", 4) != 0) {
        std::cerr << "[AudioFile] Invalid AIFF file (no FORM)" << std::endl;
        return false;
    }

    bool isAIFC = (std::strncmp(formHeader + 8, "AIFC", 4) == 0);
    if (!isAIFC && std::strncmp(formHeader + 8, "AIFF", 4) != 0) {
        std::cerr << "[AudioFile] Invalid AIFF file (not AIFF/AIFC)" << std::endl;
        return false;
    }

    int numChannels = 0;
    int64_t numSampleFrames = 0;
    int bitsPerSample = 0;
    double sampleRate = 0;
    std::vector<uint8_t> audioData;

    // Parse chunks
    while (file.good()) {
        char chunkId[4];
        uint8_t chunkSizeBytes[4];

        file.read(chunkId, 4);
        file.read(reinterpret_cast<char*>(chunkSizeBytes), 4);

        if (!file.good()) break;

        uint32_t chunkSize = readBigEndian32(chunkSizeBytes);

        if (std::strncmp(chunkId, "COMM", 4) == 0) {
            // Common chunk
            uint8_t commData[18];
            file.read(reinterpret_cast<char*>(commData), std::min(chunkSize, 18u));

            numChannels = readBigEndian16(commData);
            numSampleFrames = readBigEndian32(commData + 2);
            bitsPerSample = readBigEndian16(commData + 6);
            sampleRate = readIEEE80(commData + 8);

            // Skip rest of chunk if AIFC (compression info)
            if (chunkSize > 18) {
                file.seekg(chunkSize - 18, std::ios::cur);
            }
        } else if (std::strncmp(chunkId, "SSND", 4) == 0) {
            // Sound data chunk
            uint8_t ssndHeader[8];
            file.read(reinterpret_cast<char*>(ssndHeader), 8);
            // offset and blockSize (usually 0)

            uint32_t audioSize = chunkSize - 8;
            audioData.resize(audioSize);
            file.read(reinterpret_cast<char*>(audioData.data()), audioSize);
        } else {
            // Skip unknown chunk
            file.seekg(chunkSize, std::ios::cur);
        }

        // Chunks are padded to even byte boundaries
        if (chunkSize & 1) {
            file.seekg(1, std::ios::cur);
        }
    }

    if (numChannels == 0 || audioData.empty()) {
        std::cerr << "[AudioFile] Invalid AIFF: missing COMM or SSND chunk" << std::endl;
        return false;
    }

    // Set file info
    info_.format = Format::AIFF;
    info_.sampleRate = static_cast<int>(sampleRate);
    info_.numChannels = numChannels;
    info_.bitDepth = bitsPerSample;
    info_.numSamples = numSampleFrames;
    info_.durationSeconds = static_cast<double>(numSampleFrames) / sampleRate;

    // Allocate buffer
    buffer_ = std::make_unique<Core::AudioBuffer>(numChannels, static_cast<int>(numSampleFrames));

    // Convert audio data (big-endian to native float)
    int bytesPerSample = bitsPerSample / 8;

    for (int ch = 0; ch < numChannels; ++ch) {
        float* channelData = buffer_->getWritePointer(ch);

        for (int64_t i = 0; i < numSampleFrames; ++i) {
            size_t byteOffset = (i * numChannels + ch) * bytesPerSample;

            if (bitsPerSample == 16) {
                int16_t sample = (audioData[byteOffset] << 8) | audioData[byteOffset + 1];
                channelData[i] = sample / 32768.0f;
            } else if (bitsPerSample == 24) {
                int32_t sample = (audioData[byteOffset] << 24) |
                                (audioData[byteOffset + 1] << 16) |
                                (audioData[byteOffset + 2] << 8);
                sample >>= 8; // Sign extend
                channelData[i] = sample / 8388608.0f;
            } else if (bitsPerSample == 32) {
                int32_t sample = (audioData[byteOffset] << 24) |
                                (audioData[byteOffset + 1] << 16) |
                                (audioData[byteOffset + 2] << 8) |
                                audioData[byteOffset + 3];
                channelData[i] = sample / 2147483648.0f;
            }
        }
    }

    std::cout << "[AudioFile] Loaded AIFF: " << filepath << " ("
              << numChannels << " ch, "
              << info_.sampleRate << " Hz, "
              << bitsPerSample << " bit, "
              << info_.durationSeconds << " sec)" << std::endl;

    return true;
}

bool AudioFile::saveAIFF(const std::string& filepath,
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
    uint32_t audioDataSize = numSamples * numChannels * bytesPerSample;

    // FORM header
    file.write("FORM", 4);
    uint8_t formSize[4];
    writeBigEndian32(formSize, 4 + 26 + 8 + audioDataSize + 8); // AIFF + COMM + SSND header + data
    file.write(reinterpret_cast<char*>(formSize), 4);
    file.write("AIFF", 4);

    // COMM chunk
    file.write("COMM", 4);
    uint8_t commSize[4];
    writeBigEndian32(commSize, 18);
    file.write(reinterpret_cast<char*>(commSize), 4);

    uint8_t commData[18];
    writeBigEndian16(commData, numChannels);
    writeBigEndian32(commData + 2, numSamples);
    writeBigEndian16(commData + 6, bitDepth);
    writeIEEE80(commData + 8, static_cast<double>(sampleRate));
    file.write(reinterpret_cast<char*>(commData), 18);

    // SSND chunk
    file.write("SSND", 4);
    uint8_t ssndSize[4];
    writeBigEndian32(ssndSize, audioDataSize + 8);
    file.write(reinterpret_cast<char*>(ssndSize), 4);

    // Offset and block size (both 0)
    uint8_t zeros[8] = {0};
    file.write(reinterpret_cast<char*>(zeros), 8);

    // Write audio data (interleaved, big-endian)
    std::vector<uint8_t> audioData(audioDataSize);

    for (int i = 0; i < numSamples; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) {
            float sample = std::clamp(buffer.getReadPointer(ch)[i], -1.0f, 1.0f);
            size_t byteOffset = (i * numChannels + ch) * bytesPerSample;

            if (bitDepth == 16) {
                int16_t intSample = static_cast<int16_t>(sample * 32767.0f);
                audioData[byteOffset] = (intSample >> 8) & 0xFF;
                audioData[byteOffset + 1] = intSample & 0xFF;
            } else if (bitDepth == 24) {
                int32_t intSample = static_cast<int32_t>(sample * 8388607.0f);
                audioData[byteOffset] = (intSample >> 16) & 0xFF;
                audioData[byteOffset + 1] = (intSample >> 8) & 0xFF;
                audioData[byteOffset + 2] = intSample & 0xFF;
            } else if (bitDepth == 32) {
                int32_t intSample = static_cast<int32_t>(sample * 2147483647.0f);
                audioData[byteOffset] = (intSample >> 24) & 0xFF;
                audioData[byteOffset + 1] = (intSample >> 16) & 0xFF;
                audioData[byteOffset + 2] = (intSample >> 8) & 0xFF;
                audioData[byteOffset + 3] = intSample & 0xFF;
            }
        }
    }

    file.write(reinterpret_cast<char*>(audioData.data()), audioDataSize);

    std::cout << "[AudioFile] Saved AIFF: " << filepath << " ("
              << numChannels << " ch, "
              << sampleRate << " Hz, "
              << bitDepth << " bit)" << std::endl;

    return true;
}

// ============================================================================
// FLAC Format Implementation (Simplified - No Compression for Writing)
// ============================================================================

bool AudioFile::loadFLAC(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[AudioFile] Failed to open: " << filepath << std::endl;
        return false;
    }

    // Check FLAC signature
    char signature[4];
    file.read(signature, 4);
    if (std::strncmp(signature, "fLaC", 4) != 0) {
        std::cerr << "[AudioFile] Invalid FLAC file" << std::endl;
        return false;
    }

    // Parse metadata blocks
    bool lastBlock = false;
    int sampleRate = 0;
    int numChannels = 0;
    int bitsPerSample = 0;
    int64_t totalSamples = 0;

    while (!lastBlock && file.good()) {
        uint8_t blockHeader[4];
        file.read(reinterpret_cast<char*>(blockHeader), 4);

        lastBlock = (blockHeader[0] & 0x80) != 0;
        int blockType = blockHeader[0] & 0x7F;
        uint32_t blockLength = (blockHeader[1] << 16) | (blockHeader[2] << 8) | blockHeader[3];

        if (blockType == 0) {
            // STREAMINFO block
            uint8_t streamInfo[34];
            file.read(reinterpret_cast<char*>(streamInfo), std::min(blockLength, 34u));

            // Parse STREAMINFO
            // Bytes 10-13: sample rate (20 bits), channels (3 bits), bits per sample (5 bits)
            sampleRate = (streamInfo[10] << 12) | (streamInfo[11] << 4) | (streamInfo[12] >> 4);
            numChannels = ((streamInfo[12] >> 1) & 0x07) + 1;
            bitsPerSample = ((streamInfo[12] & 0x01) << 4) | (streamInfo[13] >> 4) + 1;

            // Bytes 13-17: total samples (36 bits)
            totalSamples = (static_cast<int64_t>(streamInfo[13] & 0x0F) << 32) |
                          (static_cast<int64_t>(streamInfo[14]) << 24) |
                          (streamInfo[15] << 16) |
                          (streamInfo[16] << 8) |
                          streamInfo[17];

            if (blockLength > 34) {
                file.seekg(blockLength - 34, std::ios::cur);
            }
        } else {
            // Skip other blocks
            file.seekg(blockLength, std::ios::cur);
        }
    }

    if (sampleRate == 0 || numChannels == 0) {
        std::cerr << "[AudioFile] Invalid FLAC: missing STREAMINFO" << std::endl;
        return false;
    }

    // Set file info
    info_.format = Format::FLAC;
    info_.sampleRate = sampleRate;
    info_.numChannels = numChannels;
    info_.bitDepth = bitsPerSample;
    info_.numSamples = totalSamples;
    info_.durationSeconds = static_cast<double>(totalSamples) / sampleRate;

    // NOTE: Full FLAC decoding requires implementing the FLAC frame decoder
    // which is complex (LPC prediction, Rice coding, etc.)
    // For production use, integrate libFLAC or dr_flac

    std::cerr << "[AudioFile] FLAC decoding requires libFLAC integration" << std::endl;
    std::cerr << "[AudioFile] File info: " << numChannels << " ch, "
              << sampleRate << " Hz, " << bitsPerSample << " bit, "
              << totalSamples << " samples" << std::endl;

    // Create empty buffer with correct dimensions
    buffer_ = std::make_unique<Core::AudioBuffer>(numChannels, static_cast<int>(totalSamples));
    buffer_->clear();

    std::cout << "[AudioFile] FLAC metadata loaded (audio data requires libFLAC): " << filepath << std::endl;

    return true;
}

bool AudioFile::saveFLAC(const std::string& filepath,
                          const Core::AudioBuffer& buffer,
                          int sampleRate,
                          int bitDepth) {
    // FLAC encoding is complex - requires LPC analysis, Rice coding, etc.
    // For production use, integrate libFLAC

    std::cerr << "[AudioFile] FLAC encoding requires libFLAC integration" << std::endl;
    std::cerr << "[AudioFile] Saving as WAV instead..." << std::endl;

    // Fallback to WAV
    std::string wavPath = filepath.substr(0, filepath.find_last_of('.')) + ".wav";
    return saveWAV(wavPath, buffer, sampleRate, bitDepth);
}

// ============================================================================
// MP3 Format Implementation (Simplified - Header Parsing Only)
// ============================================================================

bool AudioFile::loadMP3(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[AudioFile] Failed to open: " << filepath << std::endl;
        return false;
    }

    // Check for ID3v2 tag
    char id3Header[10];
    file.read(id3Header, 10);

    size_t audioDataStart = 0;

    if (std::strncmp(id3Header, "ID3", 3) == 0) {
        // Parse ID3v2 tag size (syncsafe integer)
        uint32_t id3Size = ((id3Header[6] & 0x7F) << 21) |
                          ((id3Header[7] & 0x7F) << 14) |
                          ((id3Header[8] & 0x7F) << 7) |
                          (id3Header[9] & 0x7F);
        audioDataStart = 10 + id3Size;
    } else {
        file.seekg(0);
    }

    // Find MP3 frame sync
    file.seekg(audioDataStart);

    uint8_t frameHeader[4];
    bool foundSync = false;

    while (file.good() && !foundSync) {
        file.read(reinterpret_cast<char*>(frameHeader), 4);
        if (frameHeader[0] == 0xFF && (frameHeader[1] & 0xE0) == 0xE0) {
            foundSync = true;
        } else {
            file.seekg(-3, std::ios::cur);
        }
    }

    if (!foundSync) {
        std::cerr << "[AudioFile] No MP3 frame sync found" << std::endl;
        return false;
    }

    // Parse frame header
    int version = (frameHeader[1] >> 3) & 0x03;
    int layer = (frameHeader[1] >> 1) & 0x03;
    int bitrateIndex = (frameHeader[2] >> 4) & 0x0F;
    int sampleRateIndex = (frameHeader[2] >> 2) & 0x03;
    int channelMode = (frameHeader[3] >> 6) & 0x03;

    // MPEG version
    static const int versions[] = {3, -1, 2, 1}; // 2.5, reserved, 2, 1
    int mpegVersion = versions[version];

    // Sample rates (Hz)
    static const int sampleRates[4][3] = {
        {44100, 48000, 32000},  // MPEG 1
        {22050, 24000, 16000},  // MPEG 2
        {11025, 12000, 8000}    // MPEG 2.5
    };

    // Bitrates (kbps) for Layer III
    static const int bitrates[16] = {
        0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0
    };

    int sampleRate = (mpegVersion > 0 && sampleRateIndex < 3)
                     ? sampleRates[mpegVersion == 1 ? 0 : (mpegVersion == 2 ? 1 : 2)][sampleRateIndex]
                     : 44100;

    int bitrate = bitrates[bitrateIndex];
    int numChannels = (channelMode == 3) ? 1 : 2;

    // Estimate duration from file size
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();

    int64_t totalSamples = 0;
    if (bitrate > 0) {
        double durationSec = static_cast<double>(fileSize - audioDataStart) * 8.0 / (bitrate * 1000.0);
        totalSamples = static_cast<int64_t>(durationSec * sampleRate);
    }

    // Set file info
    info_.format = Format::MP3;
    info_.sampleRate = sampleRate;
    info_.numChannels = numChannels;
    info_.bitDepth = 16; // MP3 decodes to 16-bit typically
    info_.numSamples = totalSamples;
    info_.durationSeconds = static_cast<double>(totalSamples) / sampleRate;

    // NOTE: Full MP3 decoding requires implementing the MP3 decoder
    // (Huffman decoding, IMDCT, synthesis filterbank)
    // For production use, integrate minimp3 or libmpg123

    std::cerr << "[AudioFile] MP3 decoding requires libmpg123/minimp3 integration" << std::endl;
    std::cerr << "[AudioFile] File info: " << numChannels << " ch, "
              << sampleRate << " Hz, " << bitrate << " kbps, "
              << info_.durationSeconds << " sec" << std::endl;

    // Create empty buffer with correct dimensions
    buffer_ = std::make_unique<Core::AudioBuffer>(numChannels, static_cast<int>(totalSamples));
    buffer_->clear();

    std::cout << "[AudioFile] MP3 metadata loaded (audio data requires decoder library): " << filepath << std::endl;

    return true;
}

} // namespace DSP
} // namespace MolinAntro
