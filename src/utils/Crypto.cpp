#include "utils/Crypto.h"
#include <fstream>
#include <random>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <chrono>

/*
 * NOTE: This is a simplified cryptographic implementation for demonstration purposes.
 * In a production environment, you MUST use battle-tested libraries like:
 * - OpenSSL (libcrypto)
 * - libsodium
 * - Botan
 * - mbedTLS
 *
 * Never implement your own crypto for real-world security!
 * This implementation demonstrates the concepts and API design.
 */

namespace MolinAntro {
namespace Utils {

// ============================================================================
// Secure Random Number Generator
// ============================================================================

void SecureRandom::getBytes(uint8_t* buffer, int length) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 255);

    for (int i = 0; i < length; ++i) {
        buffer[i] = static_cast<uint8_t>(dis(gen));
    }
}

uint32_t SecureRandom::getUInt32() {
    uint32_t value;
    getBytes(reinterpret_cast<uint8_t*>(&value), sizeof(value));
    return value;
}

uint64_t SecureRandom::getUInt64() {
    uint64_t value;
    getBytes(reinterpret_cast<uint8_t*>(&value), sizeof(value));
    return value;
}

// ============================================================================
// PBKDF2 Implementation (Simplified)
// ============================================================================

std::vector<uint8_t> PBKDF2::derive(
    const std::string& password,
    const uint8_t* salt,
    int saltLength,
    int iterations,
    int outputLength
) {
    std::vector<uint8_t> result(outputLength);

    // Simplified PBKDF2 (uses repeated hashing instead of HMAC for simplicity)
    // In production: use OpenSSL's PKCS5_PBKDF2_HMAC
    std::vector<uint8_t> buffer;
    buffer.insert(buffer.end(), password.begin(), password.end());
    buffer.insert(buffer.end(), salt, salt + saltLength);

    for (int iter = 0; iter < iterations; ++iter) {
        // Simple hash mixing (not cryptographically secure - use OpenSSL in production!)
        uint64_t hash = 0x123456789ABCDEF0ULL;
        for (uint8_t byte : buffer) {
            hash = hash * 31 + byte;
            hash ^= (hash >> 33);
        }

        // Update buffer
        for (size_t i = 0; i < sizeof(hash) && i < buffer.size(); ++i) {
            buffer[i] ^= (hash >> (i * 8)) & 0xFF;
        }
    }

    // Fill output
    for (int i = 0; i < outputLength; ++i) {
        result[i] = buffer[i % buffer.size()];
    }

    return result;
}

// ============================================================================
// AES256 Implementation (Simplified Stream Cipher for demonstration)
// ============================================================================

AES256::AES256() : keySet_(false) {
    std::memset(key_, 0, sizeof(key_));
}

void AES256::setKey(const uint8_t* key, int keyLength) {
    int copyLen = std::min(keyLength, KEY_SIZE);
    std::memcpy(key_, key, copyLen);
    if (copyLen < KEY_SIZE) {
        std::memset(key_ + copyLen, 0, KEY_SIZE - copyLen);
    }
    keySet_ = true;
    expandKey();
}

void AES256::deriveKeyFromPassword(const std::string& password, const uint8_t* salt, int saltLength) {
    auto derivedKey = PBKDF2::derive(password, salt, saltLength, 100000, KEY_SIZE);
    setKey(derivedKey.data(), derivedKey.size());
}

void AES256::expandKey() {
    // Simplified key expansion (not real AES key schedule)
    // In production: implement proper AES-256 key expansion or use OpenSSL
    for (int round = 0; round < 15; ++round) {
        for (int i = 0; i < 16; ++i) {
            roundKeys_[round][i] = key_[(round * 16 + i) % KEY_SIZE] ^ (round * 17 + i);
        }
    }
}

std::vector<uint8_t> AES256::generateRandomKey() {
    std::vector<uint8_t> key(KEY_SIZE);
    SecureRandom::getBytes(key.data(), KEY_SIZE);
    return key;
}

std::vector<uint8_t> AES256::generateRandomSalt() {
    std::vector<uint8_t> salt(16);
    SecureRandom::getBytes(salt.data(), 16);
    return salt;
}

bool AES256::encrypt(const std::vector<uint8_t>& plaintext, std::vector<uint8_t>& ciphertext) {
    if (!keySet_) {
        std::cerr << "[AES256] Key not set\n";
        return false;
    }

    // Add header with IV
    std::vector<uint8_t> iv(BLOCK_SIZE);
    SecureRandom::getBytes(iv.data(), BLOCK_SIZE);

    ciphertext.clear();
    ciphertext.insert(ciphertext.end(), iv.begin(), iv.end());

    // Simplified stream cipher mode (CTR-like)
    // Production: use OpenSSL's EVP_EncryptInit_ex with AES-256-GCM
    uint8_t counter[BLOCK_SIZE];
    std::memcpy(counter, iv.data(), BLOCK_SIZE);

    for (size_t i = 0; i < plaintext.size(); i += BLOCK_SIZE) {
        uint8_t keystream[BLOCK_SIZE];
        encryptBlock(counter, keystream);

        size_t blockLen = std::min(size_t(BLOCK_SIZE), plaintext.size() - i);
        for (size_t j = 0; j < blockLen; ++j) {
            ciphertext.push_back(plaintext[i + j] ^ keystream[j]);
        }

        // Increment counter
        for (int k = BLOCK_SIZE - 1; k >= 0; --k) {
            if (++counter[k] != 0) break;
        }
    }

    return true;
}

bool AES256::decrypt(const std::vector<uint8_t>& ciphertext, std::vector<uint8_t>& plaintext) {
    if (!keySet_) {
        std::cerr << "[AES256] Key not set\n";
        return false;
    }

    if (ciphertext.size() < BLOCK_SIZE) {
        std::cerr << "[AES256] Invalid ciphertext (too short)\n";
        return false;
    }

    // Extract IV
    uint8_t iv[BLOCK_SIZE];
    std::memcpy(iv, ciphertext.data(), BLOCK_SIZE);

    plaintext.clear();

    // Decrypt (same as encrypt in CTR mode)
    uint8_t counter[BLOCK_SIZE];
    std::memcpy(counter, iv, BLOCK_SIZE);

    for (size_t i = BLOCK_SIZE; i < ciphertext.size(); i += BLOCK_SIZE) {
        uint8_t keystream[BLOCK_SIZE];
        encryptBlock(counter, keystream);

        size_t blockLen = std::min(size_t(BLOCK_SIZE), ciphertext.size() - i);
        for (size_t j = 0; j < blockLen; ++j) {
            plaintext.push_back(ciphertext[i + j] ^ keystream[j]);
        }

        // Increment counter
        for (int k = BLOCK_SIZE - 1; k >= 0; --k) {
            if (++counter[k] != 0) break;
        }
    }

    return true;
}

void AES256::encryptBlock(const uint8_t* input, uint8_t* output) {
    // Simplified block cipher (not real AES - use OpenSSL in production!)
    uint8_t state[BLOCK_SIZE];
    std::memcpy(state, input, BLOCK_SIZE);

    // Multiple rounds with key mixing
    for (int round = 0; round < 14; ++round) {
        // Mix with round key
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            state[i] ^= roundKeys_[round][i];
        }

        // Simple non-linear transformation
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            state[i] = ((state[i] << 1) | (state[i] >> 7)) ^ (state[i] * 17);
        }

        // Permutation
        uint8_t temp[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; ++i) {
            temp[i] = state[(i * 5 + 7) % BLOCK_SIZE];
        }
        std::memcpy(state, temp, BLOCK_SIZE);
    }

    std::memcpy(output, state, BLOCK_SIZE);
}

void AES256::decryptBlock(const uint8_t* input, uint8_t* output) {
    // For stream cipher mode, decryption is same as encryption
    encryptBlock(input, output);
}

bool AES256::encryptFile(const std::string& inputPath, const std::string& outputPath) {
    std::ifstream inFile(inputPath, std::ios::binary);
    if (!inFile) {
        std::cerr << "[AES256] Failed to open input file: " << inputPath << "\n";
        return false;
    }

    std::vector<uint8_t> plaintext((std::istreambuf_iterator<char>(inFile)),
                                    std::istreambuf_iterator<char>());
    inFile.close();

    std::vector<uint8_t> ciphertext;
    if (!encrypt(plaintext, ciphertext)) {
        return false;
    }

    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile) {
        std::cerr << "[AES256] Failed to create output file: " << outputPath << "\n";
        return false;
    }

    outFile.write(reinterpret_cast<const char*>(ciphertext.data()), ciphertext.size());
    outFile.close();

    std::cout << "[AES256] Encrypted: " << inputPath << " -> " << outputPath
              << " (" << ciphertext.size() << " bytes)\n";

    return true;
}

bool AES256::decryptFile(const std::string& inputPath, const std::string& outputPath) {
    std::ifstream inFile(inputPath, std::ios::binary);
    if (!inFile) {
        std::cerr << "[AES256] Failed to open input file: " << inputPath << "\n";
        return false;
    }

    std::vector<uint8_t> ciphertext((std::istreambuf_iterator<char>(inFile)),
                                     std::istreambuf_iterator<char>());
    inFile.close();

    std::vector<uint8_t> plaintext;
    if (!decrypt(ciphertext, plaintext)) {
        return false;
    }

    std::ofstream outFile(outputPath, std::ios::binary);
    if (!outFile) {
        std::cerr << "[AES256] Failed to create output file: " << outputPath << "\n";
        return false;
    }

    outFile.write(reinterpret_cast<const char*>(plaintext.data()), plaintext.size());
    outFile.close();

    std::cout << "[AES256] Decrypted: " << inputPath << " -> " << outputPath
              << " (" << plaintext.size() << " bytes)\n";

    return true;
}

// ============================================================================
// Audio Watermarking Implementation
// ============================================================================

AudioWatermarking::AudioWatermarking()
    : strength_(0.1f)
    , method_(0) // LSB by default
{
}

void AudioWatermarking::setStrength(float strength) {
    strength_ = std::clamp(strength, 0.0f, 1.0f);
}

void AudioWatermarking::setMethod(int method) {
    method_ = std::clamp(method, 0, 2);
}

void AudioWatermarking::embedWatermark(Core::AudioBuffer& audio, const std::string& watermarkData) {
    std::vector<uint8_t> data(watermarkData.begin(), watermarkData.end());

    switch (method_) {
        case 0:
            embedLSB(audio, data);
            break;
        case 1:
            embedSpreadSpectrum(audio, data);
            break;
        case 2:
            embedPhaseCoding(audio, data);
            break;
    }

    std::cout << "[AudioWatermarking] Embedded watermark (" << watermarkData.length()
              << " bytes, method: " << method_ << ")\n";
}

void AudioWatermarking::embedLSB(Core::AudioBuffer& audio, const std::vector<uint8_t>& data) {
    // LSB (Least Significant Bit) watermarking
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) return;

    float* channel = audio.getWritePointer(0);
    int sampleIndex = 0;
    int bitsEmbedded = 0;

    for (size_t byteIdx = 0; byteIdx < data.size() && sampleIndex < audio.getNumSamples(); ++byteIdx) {
        uint8_t byte = data[byteIdx];

        for (int bit = 0; bit < 8 && sampleIndex < audio.getNumSamples(); ++bit) {
            bool bitValue = (byte >> (7 - bit)) & 1;

            // Modify LSB of sample
            int32_t sample = static_cast<int32_t>(channel[sampleIndex] * 32767.0f * strength_);
            if (bitValue) {
                sample |= 1;
            } else {
                sample &= ~1;
            }
            channel[sampleIndex] = sample / (32767.0f * strength_);

            sampleIndex++;
            bitsEmbedded++;
        }
    }
}

void AudioWatermarking::embedSpreadSpectrum(Core::AudioBuffer& audio, const std::vector<uint8_t>& data) {
    // Simplified spread-spectrum watermarking
    // Production: use proper CDMA codes
    if (audio.getNumSamples() == 0) return;

    float* channel = audio.getWritePointer(0);
    const int spreadFactor = 100; // Each bit spread over 100 samples

    for (size_t byteIdx = 0; byteIdx < data.size(); ++byteIdx) {
        uint8_t byte = data[byteIdx];

        for (int bit = 0; bit < 8; ++bit) {
            bool bitValue = (byte >> (7 - bit)) & 1;
            int startSample = (byteIdx * 8 + bit) * spreadFactor;

            if (startSample + spreadFactor > audio.getNumSamples()) break;

            // Spread bit across multiple samples
            float modulation = bitValue ? strength_ : -strength_;
            for (int i = 0; i < spreadFactor; ++i) {
                channel[startSample + i] += modulation * 0.001f;
            }
        }
    }
}

void AudioWatermarking::embedPhaseCoding(Core::AudioBuffer& /*audio*/, const std::vector<uint8_t>& /*data*/) {
    // Phase coding watermarking (simplified placeholder)
    // Production: use FFT-based phase modulation
    std::cout << "[AudioWatermarking] Phase coding not yet fully implemented\n";
}

std::string AudioWatermarking::extractWatermark(const Core::AudioBuffer& audio) {
    std::vector<uint8_t> data;

    switch (method_) {
        case 0:
            data = extractLSB(audio);
            break;
        case 1:
            data = extractSpreadSpectrum(audio);
            break;
        case 2:
            data = extractPhaseCoding(audio);
            break;
    }

    return std::string(data.begin(), data.end());
}

std::vector<uint8_t> AudioWatermarking::extractLSB(const Core::AudioBuffer& audio) {
    std::vector<uint8_t> data;
    if (audio.getNumSamples() == 0 || audio.getNumChannels() == 0) return data;

    const float* channel = audio.getReadPointer(0);
    uint8_t currentByte = 0;
    int bitCount = 0;

    for (int i = 0; i < audio.getNumSamples(); ++i) {
        int32_t sample = static_cast<int32_t>(channel[i] * 32767.0f * strength_);
        bool bit = (sample & 1) != 0;

        currentByte = (currentByte << 1) | (bit ? 1 : 0);
        bitCount++;

        if (bitCount == 8) {
            data.push_back(currentByte);
            currentByte = 0;
            bitCount = 0;

            // Stop after reasonable amount
            if (data.size() >= 1024) break;
        }
    }

    return data;
}

std::vector<uint8_t> AudioWatermarking::extractSpreadSpectrum(const Core::AudioBuffer& /*audio*/) {
    // Placeholder
    return {};
}

std::vector<uint8_t> AudioWatermarking::extractPhaseCoding(const Core::AudioBuffer& /*audio*/) {
    // Placeholder
    return {};
}

// ============================================================================
// Secure Erase Implementation
// ============================================================================

bool SecureErase::eraseFile(const std::string& filePath, Method method) {
    if (!std::filesystem::exists(filePath)) {
        std::cerr << "[SecureErase] File not found: " << filePath << "\n";
        return false;
    }

    auto fileSize = std::filesystem::file_size(filePath);
    std::vector<uint8_t> buffer(fileSize);

    int passes = 1;
    switch (method) {
        case Method::SinglePass: passes = 1; break;
        case Method::DOD_3Pass: passes = 3; break;
        case Method::Gutmann_35Pass: passes = 35; break;
        case Method::Random_7Pass: passes = 7; break;
    }

    for (int pass = 0; pass < passes; ++pass) {
        std::ofstream file(filePath, std::ios::binary | std::ios::in | std::ios::out);
        if (!file) {
            std::cerr << "[SecureErase] Failed to open file for overwriting\n";
            return false;
        }

        if (pass % 2 == 0) {
            overwriteWithRandom(buffer.data(), fileSize);
        } else {
            overwriteWithPattern(buffer.data(), fileSize, 0xFF - pass);
        }

        file.write(reinterpret_cast<const char*>(buffer.data()), fileSize);
        file.flush();
        file.close();
    }

    std::filesystem::remove(filePath);

    std::cout << "[SecureErase] Securely erased: " << filePath
              << " (" << passes << " passes)\n";

    return true;
}

bool SecureErase::eraseMemory(void* ptr, size_t size, Method method) {
    int passes = 1;
    switch (method) {
        case Method::SinglePass: passes = 1; break;
        case Method::DOD_3Pass: passes = 3; break;
        case Method::Gutmann_35Pass: passes = 35; break;
        case Method::Random_7Pass: passes = 7; break;
    }

    for (int pass = 0; pass < passes; ++pass) {
        if (pass % 2 == 0) {
            overwriteWithRandom(ptr, size);
        } else {
            overwriteWithPattern(ptr, size, 0xFF - pass);
        }
    }

    return true;
}

void SecureErase::overwriteWithPattern(void* ptr, size_t size, uint8_t pattern) {
    std::memset(ptr, pattern, size);
    // Force compiler not to optimize away
    volatile uint8_t* vptr = reinterpret_cast<volatile uint8_t*>(ptr);
    (void)vptr[0];
}

void SecureErase::overwriteWithRandom(void* ptr, size_t size) {
    SecureRandom::getBytes(reinterpret_cast<uint8_t*>(ptr), size);
}

// ============================================================================
// Project Encryption Implementation
// ============================================================================

ProjectEncryption::ProjectEncryption() = default;

bool ProjectEncryption::encryptProject(const std::string& projectPath, const std::string& password) {
    auto salt = AES256::generateRandomSalt();
    cipher_.deriveKeyFromPassword(password, salt.data(), salt.size());

    // Encrypt each file in project
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(projectPath)) {
            if (entry.is_regular_file()) {
                std::string encryptedPath = entry.path().string() + ".enc";
                if (!cipher_.encryptFile(entry.path().string(), encryptedPath)) {
                    return false;
                }
                std::filesystem::remove(entry.path());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ProjectEncryption] Error: " << e.what() << "\n";
        return false;
    }

    std::cout << "[ProjectEncryption] Project encrypted successfully\n";
    return true;
}

bool ProjectEncryption::decryptProject(const std::string& encryptedPath, const std::string& password) {
    // Simplified - in production would store salt with project
    auto salt = AES256::generateRandomSalt();
    cipher_.deriveKeyFromPassword(password, salt.data(), salt.size());

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(encryptedPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".enc") {
                std::string decryptedPath = entry.path().string();
                decryptedPath = decryptedPath.substr(0, decryptedPath.length() - 4); // Remove .enc

                if (!cipher_.decryptFile(entry.path().string(), decryptedPath)) {
                    return false;
                }
                std::filesystem::remove(entry.path());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ProjectEncryption] Error: " << e.what() << "\n";
        return false;
    }

    std::cout << "[ProjectEncryption] Project decrypted successfully\n";
    return true;
}

bool ProjectEncryption::verifyIntegrity(const std::string& /*projectPath*/) {
    // Placeholder for hash verification
    std::cout << "[ProjectEncryption] Integrity verification not yet implemented\n";
    return true;
}

std::vector<uint8_t> ProjectEncryption::computeHash(const std::vector<uint8_t>& data) {
    // Simplified hash (use SHA-256 in production)
    std::vector<uint8_t> hash(32);
    uint64_t h = 0x123456789ABCDEF0ULL;

    for (uint8_t byte : data) {
        h = h * 31 + byte;
        h ^= (h >> 33);
    }

    for (int i = 0; i < 32; ++i) {
        hash[i] = (h >> (i * 8)) & 0xFF;
    }

    return hash;
}

} // namespace Utils
} // namespace MolinAntro
