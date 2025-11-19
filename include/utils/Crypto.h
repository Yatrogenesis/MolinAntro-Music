#pragma once

#include "core/AudioBuffer.h"
#include <vector>
#include <string>
#include <cstdint>
#include <memory>

namespace MolinAntro {
namespace Utils {

/**
 * Military-Grade Cryptography & Security
 * Features: AES-256, PBKDF2, audio watermarking, secure erase
 */

/**
 * AES-256 Encryption (Simplified implementation for demonstration)
 * In production, would use OpenSSL or similar library
 */
class AES256 {
public:
    static constexpr int KEY_SIZE = 32;  // 256 bits
    static constexpr int BLOCK_SIZE = 16; // 128 bits

    AES256();

    // Key management
    void setKey(const uint8_t* key, int keyLength);
    void deriveKeyFromPassword(const std::string& password, const uint8_t* salt, int saltLength);

    // Encryption/Decryption
    bool encrypt(const std::vector<uint8_t>& plaintext, std::vector<uint8_t>& ciphertext);
    bool decrypt(const std::vector<uint8_t>& ciphertext, std::vector<uint8_t>& plaintext);

    // File encryption
    bool encryptFile(const std::string& inputPath, const std::string& outputPath);
    bool decryptFile(const std::string& inputPath, const std::string& outputPath);

    // Generate random key
    static std::vector<uint8_t> generateRandomKey();
    static std::vector<uint8_t> generateRandomSalt();

private:
    uint8_t key_[KEY_SIZE];
    uint8_t roundKeys_[15][16]; // AES-256 has 14 rounds
    bool keySet_;

    void expandKey();
    void encryptBlock(const uint8_t* input, uint8_t* output);
    void decryptBlock(const uint8_t* input, uint8_t* output);

    // AES operations
    void subBytes(uint8_t* state);
    void invSubBytes(uint8_t* state);
    void shiftRows(uint8_t* state);
    void invShiftRows(uint8_t* state);
    void mixColumns(uint8_t* state);
    void invMixColumns(uint8_t* state);
    void addRoundKey(uint8_t* state, int round);
};

/**
 * PBKDF2 - Password-Based Key Derivation Function 2
 * Used for secure key generation from passwords
 */
class PBKDF2 {
public:
    static std::vector<uint8_t> derive(
        const std::string& password,
        const uint8_t* salt,
        int saltLength,
        int iterations,
        int outputLength
    );

private:
    static void hmacSHA256(
        const uint8_t* key, int keyLen,
        const uint8_t* data, int dataLen,
        uint8_t* output
    );
};

/**
 * Audio Watermarking - Invisible copyright protection
 * Embeds imperceptible watermark in audio signal
 */
class AudioWatermarking {
public:
    AudioWatermarking();

    // Embed watermark
    void embedWatermark(Core::AudioBuffer& audio, const std::string& watermarkData);

    // Extract watermark
    std::string extractWatermark(const Core::AudioBuffer& audio);

    // Configuration
    void setStrength(float strength); // 0.0 (subtle) to 1.0 (strong)
    void setMethod(int method); // 0=LSB, 1=Spread-Spectrum, 2=Phase-Coding

private:
    float strength_;
    int method_;

    void embedLSB(Core::AudioBuffer& audio, const std::vector<uint8_t>& data);
    void embedSpreadSpectrum(Core::AudioBuffer& audio, const std::vector<uint8_t>& data);
    void embedPhaseCoding(Core::AudioBuffer& audio, const std::vector<uint8_t>& data);

    std::vector<uint8_t> extractLSB(const Core::AudioBuffer& audio);
    std::vector<uint8_t> extractSpreadSpectrum(const Core::AudioBuffer& audio);
    std::vector<uint8_t> extractPhaseCoding(const Core::AudioBuffer& audio);
};

/**
 * Secure Data Erasure
 * DOD 5220.22-M compliant secure data deletion
 */
class SecureErase {
public:
    enum class Method {
        SinglePass,      // Single overwrite with zeros
        DOD_3Pass,       // DOD 5220.22-M (3 passes)
        Gutmann_35Pass,  // Gutmann method (35 passes)
        Random_7Pass     // 7 passes with random data
    };

    static bool eraseFile(const std::string& filePath, Method method = Method::DOD_3Pass);
    static bool eraseMemory(void* ptr, size_t size, Method method = Method::DOD_3Pass);

private:
    static void overwriteWithPattern(void* ptr, size_t size, uint8_t pattern);
    static void overwriteWithRandom(void* ptr, size_t size);
};

/**
 * Project Encryption - Encrypt entire DAW projects
 */
class ProjectEncryption {
public:
    ProjectEncryption();

    // Encrypt project directory
    bool encryptProject(const std::string& projectPath, const std::string& password);

    // Decrypt project directory
    bool decryptProject(const std::string& encryptedPath, const std::string& password);

    // Verify project integrity
    bool verifyIntegrity(const std::string& projectPath);

private:
    AES256 cipher_;

    std::vector<uint8_t> computeHash(const std::vector<uint8_t>& data);
};

/**
 * Secure Random Number Generator (CSPRNG)
 */
class SecureRandom {
public:
    static void getBytes(uint8_t* buffer, int length);
    static uint32_t getUInt32();
    static uint64_t getUInt64();

private:
    static void initializeEntropy();
};

} // namespace Utils
} // namespace MolinAntro
