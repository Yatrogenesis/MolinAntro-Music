/**
 * MolinAntro DAW - Forensic Audio Analysis Module
 * SOTA x5 Implementation - Military-Grade Audio Forensics
 *
 * Features:
 * - ENF (Electrical Network Frequency) Analysis
 * - Audio watermarking (inaudible, robust)
 * - Tampering detection (splicing, editing, AI generation)
 * - Speaker identification / Voice biometrics
 * - Gunshot acoustic analysis
 * - Chain of custody (cryptographic audit trail)
 * - Audio authentication (hash verification)
 * - Spectral anomaly detection (AI-based)
 * - Metadata forensics
 * - Room acoustics analysis
 *
 * Author: F. Molina-Burgos / MolinAntro Technologies
 * Copyright (C) 2026 - All Rights Reserved
 *
 * EXPORT CONTROLLED - ITAR/EAR regulations may apply
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <optional>
#include <functional>

namespace MolinAntro {
namespace Forensic {

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================

class ENFAnalyzer;
class WatermarkEngine;
class TamperDetector;
class SpeakerIdentifier;
class GunshotAnalyzer;
class ChainOfCustody;
class AudioAuthenticator;
class AnomalyDetector;
class MetadataForensics;
class RoomAcousticsAnalyzer;

// =============================================================================
// ANALYSIS RESULT TYPES
// =============================================================================

struct TimeRange {
    double startSeconds;
    double endSeconds;
};

struct ConfidenceInterval {
    double value;
    double lowerBound;
    double upperBound;
    double confidence;  // 0-1
};

enum class EvidenceQuality {
    Excellent,    // >99% confidence
    Good,         // 90-99% confidence
    Moderate,     // 70-90% confidence
    Poor,         // 50-70% confidence
    Unreliable    // <50% confidence
};

// =============================================================================
// ENF ANALYSIS (Electrical Network Frequency)
// =============================================================================

struct ENFResult {
    bool detected = false;
    EvidenceQuality quality = EvidenceQuality::Unreliable;

    // Estimated recording time based on ENF matching
    std::optional<std::chrono::system_clock::time_point> estimatedTimestamp;
    double timestampConfidence = 0.0;

    // ENF frequency measurements
    std::vector<double> enfFrequencies;        // Hz (typically around 50/60 Hz)
    std::vector<double> enfTimestamps;         // Seconds
    double nominalFrequency = 0.0;             // 50 Hz or 60 Hz

    // Geographic region estimate
    std::string estimatedRegion;               // "Europe", "North America", etc.
    double regionConfidence = 0.0;

    // Anomalies (potential tampering indicators)
    std::vector<TimeRange> discontinuities;
    std::vector<TimeRange> frequencyAnomalies;
};

class ENFAnalyzer {
public:
    ENFAnalyzer();
    ~ENFAnalyzer();

    // Load ENF reference database (historical power grid recordings)
    bool loadReferenceDatabase(const std::string& path);
    bool addReferenceRecording(const std::string& region,
                               std::chrono::system_clock::time_point timestamp,
                               const float* enfData, size_t numSamples, double sampleRate);

    // Analyze recording for ENF
    ENFResult analyze(const float* audio, size_t numSamples, double sampleRate);

    // Match against reference database
    ENFResult matchToReference(const float* audio, size_t numSamples, double sampleRate,
                               const std::string& region = "",
                               std::optional<std::chrono::system_clock::time_point> startTime = std::nullopt,
                               std::optional<std::chrono::system_clock::time_point> endTime = std::nullopt);

    // Extract ENF signal
    void extractENF(const float* audio, size_t numSamples, double sampleRate,
                    std::vector<double>& enfFrequencies, std::vector<double>& timestamps);

    // Settings
    void setNominalFrequency(double freq) { nominalFrequency_ = freq; }  // 50 or 60 Hz
    void setAnalysisWindow(double seconds) { analysisWindow_ = seconds; }
    void setFrequencyTolerance(double hz) { frequencyTolerance_ = hz; }

private:
    double nominalFrequency_ = 50.0;
    double analysisWindow_ = 0.5;  // seconds
    double frequencyTolerance_ = 0.1;  // Hz

    class Database;
    std::unique_ptr<Database> database_;
};

// =============================================================================
// AUDIO WATERMARKING
// =============================================================================

struct WatermarkPayload {
    std::vector<uint8_t> data;       // Binary payload (up to 1KB)
    std::string textMessage;          // Human-readable message
    std::string ownerIdentifier;      // Owner/creator ID
    std::chrono::system_clock::time_point timestamp;
    std::string checksum;             // SHA-256 of payload
};

struct WatermarkDetectionResult {
    bool detected = false;
    WatermarkPayload payload;
    EvidenceQuality quality = EvidenceQuality::Unreliable;
    double ber = 1.0;                 // Bit Error Rate (0 = perfect)
    std::vector<TimeRange> locations; // Where watermark was found
};

class WatermarkEngine {
public:
    enum class Method {
        SpreadSpectrum,      // Robust to compression, moderate capacity
        EchoHiding,          // Robust to filtering, lower capacity
        PatchworkDCT,        // Robust to JPEG-like compression
        WaveletDomain,       // High capacity, moderate robustness
        PhaseModulation,     // Very robust, low capacity
        DeepLearning         // AI-based, highest robustness
    };

    WatermarkEngine(Method method = Method::SpreadSpectrum);
    ~WatermarkEngine();

    // Embed watermark
    bool embed(float* audio, size_t numSamples, double sampleRate,
               const WatermarkPayload& payload, float strength = 0.5f);

    // Detect watermark
    WatermarkDetectionResult detect(const float* audio, size_t numSamples, double sampleRate);

    // Verify watermark authenticity
    bool verify(const float* audio, size_t numSamples, double sampleRate,
                const WatermarkPayload& expectedPayload);

    // Remove watermark (if known)
    bool remove(float* audio, size_t numSamples, double sampleRate,
                const WatermarkPayload& payload);

    // Settings
    void setMethod(Method method) { method_ = method; }
    void setKey(const std::vector<uint8_t>& key) { encryptionKey_ = key; }

private:
    Method method_;
    std::vector<uint8_t> encryptionKey_;

    void embedSpreadSpectrum(float* audio, size_t numSamples, double sampleRate,
                            const std::vector<uint8_t>& bits, float strength);
    std::vector<uint8_t> detectSpreadSpectrum(const float* audio, size_t numSamples,
                                               double sampleRate);
    // ... other methods
};

// =============================================================================
// TAMPERING DETECTION
// =============================================================================

struct TamperEvidence {
    enum class Type {
        Splice,           // Cut and paste editing
        Insert,           // Inserted audio
        Delete,           // Deleted audio
        Copy,             // Copied segment
        TimeStretch,      // Time manipulation
        PitchShift,       // Pitch manipulation
        Compression,      // Lossy compression artifacts
        AIGenerated,      // Synthetic audio
        Deepfake,         // Voice cloning / deepfake
        NoiseReduction,   // Heavy noise reduction applied
        Reverb,           // Artificial reverb added
        Unknown
    };

    Type type = Type::Unknown;
    TimeRange location;
    double confidence = 0.0;
    std::string description;
    std::vector<std::pair<std::string, double>> evidenceFactors;
};

struct TamperAnalysisResult {
    bool authentic = true;
    EvidenceQuality overallQuality = EvidenceQuality::Unreliable;
    std::vector<TamperEvidence> evidences;
    std::string summary;
    double integrityScore = 0.0;  // 0-100
};

class TamperDetector {
public:
    TamperDetector();
    ~TamperDetector();

    // Load AI models for detection
    bool loadModels(const std::string& modelsPath);

    // Full analysis
    TamperAnalysisResult analyze(const float* audio, size_t numSamples, double sampleRate);

    // Specific detections
    std::vector<TamperEvidence> detectSplices(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> detectCopyMove(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> detectAIGenerated(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> detectCompression(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> detectTimeManipulation(const float* audio, size_t numSamples, double sampleRate);

    // Feature extraction for ML
    std::vector<float> extractFeatures(const float* audio, size_t numSamples, double sampleRate);

private:
    class MLModels;
    std::unique_ptr<MLModels> models_;

    std::vector<TamperEvidence> analyzePhaseConsistency(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> analyzeEnvelopeConsistency(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> analyzeSpectralConsistency(const float* audio, size_t numSamples, double sampleRate);
    std::vector<TamperEvidence> analyzeNoiseFloor(const float* audio, size_t numSamples, double sampleRate);
};

// =============================================================================
// SPEAKER IDENTIFICATION
// =============================================================================

struct SpeakerProfile {
    std::string id;
    std::string name;
    std::vector<float> voiceprint;  // Embedding vector
    std::map<std::string, std::string> metadata;
    std::chrono::system_clock::time_point enrolled;
    int sampleCount = 0;
};

struct SpeakerMatch {
    std::string profileId;
    std::string profileName;
    double similarity = 0.0;    // 0-1
    double confidence = 0.0;    // 0-1
    bool isMatch = false;
    TimeRange location;
};

struct SpeakerAnalysisResult {
    int numSpeakers = 0;
    std::vector<SpeakerMatch> identifiedSpeakers;
    std::vector<TimeRange> speakerSegments;
    std::vector<int> segmentSpeakerIds;  // -1 = unknown
};

class SpeakerIdentifier {
public:
    SpeakerIdentifier();
    ~SpeakerIdentifier();

    // Load embedding model (ResNet/ECAPA-TDNN)
    bool loadModel(const std::string& modelPath);

    // Enroll speaker
    std::string enrollSpeaker(const std::string& name,
                              const float* audio, size_t numSamples, double sampleRate);

    // Update speaker profile with more samples
    bool updateSpeaker(const std::string& profileId,
                       const float* audio, size_t numSamples, double sampleRate);

    // Identify speaker(s) in audio
    SpeakerAnalysisResult identify(const float* audio, size_t numSamples, double sampleRate);

    // Verify speaker identity (1:1 matching)
    SpeakerMatch verify(const std::string& claimedProfileId,
                        const float* audio, size_t numSamples, double sampleRate);

    // Speaker diarization (who spoke when)
    SpeakerAnalysisResult diarize(const float* audio, size_t numSamples, double sampleRate,
                                   int expectedSpeakers = -1);

    // Profile management
    bool loadProfiles(const std::string& databasePath);
    bool saveProfiles(const std::string& databasePath);
    void clearProfiles();

    // Settings
    void setThreshold(double threshold) { matchThreshold_ = threshold; }

private:
    class EmbeddingModel;
    std::unique_ptr<EmbeddingModel> model_;
    std::vector<SpeakerProfile> profiles_;
    double matchThreshold_ = 0.7;

    std::vector<float> extractVoiceprint(const float* audio, size_t numSamples, double sampleRate);
};

// =============================================================================
// GUNSHOT ACOUSTIC ANALYSIS
// =============================================================================

struct GunshotEvent {
    TimeRange location;
    double confidence = 0.0;

    // Classification
    std::string weaponType;       // "Handgun", "Rifle", "Shotgun", etc.
    std::string caliber;          // ".45 ACP", "5.56 NATO", etc.
    double classificationConfidence = 0.0;

    // Localization (if multi-channel)
    double estimatedDistance = 0.0;     // meters
    double estimatedAzimuth = 0.0;      // degrees
    double estimatedElevation = 0.0;    // degrees
    double localizationError = 0.0;     // meters

    // Acoustic characteristics
    double muzzleBlastDuration = 0.0;   // ms
    double shockwaveDuration = 0.0;     // ms (if supersonic)
    double peakLevel = 0.0;             // dB SPL
    double riseTime = 0.0;              // ms
};

struct GunshotAnalysisResult {
    std::vector<GunshotEvent> events;
    int totalShots = 0;
    bool multipleWeapons = false;
    std::vector<std::string> weaponTypes;
};

class GunshotAnalyzer {
public:
    GunshotAnalyzer();
    ~GunshotAnalyzer();

    // Load classification model and weapon database
    bool loadModels(const std::string& modelsPath);
    bool loadWeaponDatabase(const std::string& databasePath);

    // Detect and analyze gunshots
    GunshotAnalysisResult analyze(const float* audio, size_t numSamples, double sampleRate);

    // Multi-channel analysis for localization
    GunshotAnalysisResult analyzeMultichannel(const float* const* channels, int numChannels,
                                               size_t numSamples, double sampleRate,
                                               const std::vector<std::array<float, 3>>& micPositions);

    // Shot counting
    int countShots(const float* audio, size_t numSamples, double sampleRate);

    // Time-of-arrival difference calculation
    std::vector<double> calculateTOAD(const float* const* channels, int numChannels,
                                      size_t numSamples, double sampleRate,
                                      const TimeRange& eventWindow);

private:
    class WeaponDatabase;
    std::unique_ptr<WeaponDatabase> database_;

    class ClassificationModel;
    std::unique_ptr<ClassificationModel> model_;
};

// =============================================================================
// CHAIN OF CUSTODY
// =============================================================================

struct CustodyEvent {
    std::chrono::system_clock::time_point timestamp;
    std::string action;           // "Created", "Modified", "Accessed", "Exported", etc.
    std::string actor;            // Who performed the action
    std::string description;
    std::string previousHash;     // Hash of previous state
    std::string currentHash;      // Hash of current state
    std::string signature;        // Digital signature
};

class ChainOfCustody {
public:
    ChainOfCustody();
    ~ChainOfCustody();

    // Initialize chain for a file
    std::string initializeChain(const std::string& filePath,
                                const std::string& creator,
                                const std::string& privateKeyPath);

    // Log an event
    bool logEvent(const std::string& chainId,
                  const std::string& action,
                  const std::string& actor,
                  const std::string& description,
                  const float* currentAudio = nullptr,
                  size_t numSamples = 0);

    // Verify chain integrity
    bool verifyChain(const std::string& chainId);

    // Get chain history
    std::vector<CustodyEvent> getHistory(const std::string& chainId);

    // Export chain to court-admissible format
    bool exportReport(const std::string& chainId, const std::string& outputPath);

    // Import/export chain
    bool exportChain(const std::string& chainId, const std::string& outputPath);
    std::string importChain(const std::string& inputPath);

private:
    std::map<std::string, std::vector<CustodyEvent>> chains_;

    std::string computeHash(const float* audio, size_t numSamples);
    std::string signData(const std::string& data, const std::string& privateKeyPath);
    bool verifySignature(const std::string& data, const std::string& signature,
                         const std::string& publicKeyPath);
};

// =============================================================================
// AUDIO AUTHENTICATOR
// =============================================================================

struct AuthenticationResult {
    bool authentic = false;
    EvidenceQuality quality = EvidenceQuality::Unreliable;
    std::string originalHash;
    std::string currentHash;
    bool hashMatch = false;
    bool signatureValid = false;
    std::string signedBy;
    std::chrono::system_clock::time_point signedAt;
    std::vector<std::string> issues;
};

class AudioAuthenticator {
public:
    AudioAuthenticator();
    ~AudioAuthenticator();

    // Sign audio file
    bool sign(const std::string& filePath,
              const std::string& privateKeyPath,
              const std::string& signerId);

    // Verify audio file
    AuthenticationResult verify(const std::string& filePath,
                                const std::string& publicKeyPath);

    // Verify audio data directly
    AuthenticationResult verifyData(const float* audio, size_t numSamples,
                                    double sampleRate,
                                    const std::string& expectedHash,
                                    const std::string& signature,
                                    const std::string& publicKeyPath);

    // Generate perceptual hash (robust to minor changes)
    std::string generatePerceptualHash(const float* audio, size_t numSamples,
                                        double sampleRate);

    // Compare perceptual hashes
    double comparePerceptualHashes(const std::string& hash1, const std::string& hash2);

private:
    std::string computeCryptographicHash(const float* audio, size_t numSamples);
};

// =============================================================================
// ANOMALY DETECTOR (AI-based)
// =============================================================================

struct Anomaly {
    TimeRange location;
    std::string type;
    std::string description;
    double severity = 0.0;      // 0-1
    double confidence = 0.0;    // 0-1
    std::vector<float> spectrogram;  // For visualization
};

struct AnomalyAnalysisResult {
    std::vector<Anomaly> anomalies;
    double overallAnomalyScore = 0.0;  // 0-1
    bool requiresReview = false;
};

class AnomalyDetector {
public:
    AnomalyDetector();
    ~AnomalyDetector();

    // Load trained anomaly detection model
    bool loadModel(const std::string& modelPath);

    // Analyze for anomalies
    AnomalyAnalysisResult analyze(const float* audio, size_t numSamples, double sampleRate);

    // Train on normal audio (unsupervised learning)
    void trainOnNormal(const float* audio, size_t numSamples, double sampleRate);

    // Save/load trained model
    bool saveModel(const std::string& path);

    // Set detection sensitivity
    void setSensitivity(double sensitivity) { sensitivity_ = sensitivity; }

private:
    class AutoencoderModel;
    std::unique_ptr<AutoencoderModel> model_;
    double sensitivity_ = 0.5;
};

// =============================================================================
// METADATA FORENSICS
// =============================================================================

struct MetadataReport {
    std::map<std::string, std::string> metadata;
    std::vector<std::string> inconsistencies;
    std::vector<std::string> manipulationIndicators;
    std::optional<std::chrono::system_clock::time_point> creationTime;
    std::optional<std::chrono::system_clock::time_point> modificationTime;
    std::string recordingDevice;
    std::string software;
    std::string gpsLocation;
    bool metadataComplete = false;
    bool metadataTrusted = false;
};

class MetadataForensics {
public:
    MetadataForensics();
    ~MetadataForensics();

    // Analyze file metadata
    MetadataReport analyze(const std::string& filePath);

    // Extract all metadata
    std::map<std::string, std::string> extractMetadata(const std::string& filePath);

    // Verify metadata consistency
    std::vector<std::string> verifyConsistency(const std::string& filePath);

    // Check for metadata tampering
    std::vector<std::string> checkTampering(const std::string& filePath);

    // Compare with known recording device profiles
    bool matchDeviceProfile(const std::string& filePath, const std::string& deviceProfile);
};

// =============================================================================
// ROOM ACOUSTICS ANALYZER
// =============================================================================

struct RoomAcousticsResult {
    // Room characteristics
    double estimatedVolume = 0.0;           // cubic meters
    double estimatedRT60 = 0.0;             // seconds
    double clarityC50 = 0.0;                // dB
    double clarityC80 = 0.0;                // dB
    double definitionD50 = 0.0;             // 0-1
    double centreTime = 0.0;                // seconds

    // Consistency analysis
    bool roomConsistent = true;             // Same room throughout?
    std::vector<TimeRange> roomChanges;     // Where room changed
    double consistencyScore = 0.0;          // 0-1

    // Estimated room type
    std::string estimatedRoomType;          // "Small room", "Large hall", "Outdoor", etc.
    double roomTypeConfidence = 0.0;
};

class RoomAcousticsAnalyzer {
public:
    RoomAcousticsAnalyzer();
    ~RoomAcousticsAnalyzer();

    // Analyze room acoustics
    RoomAcousticsResult analyze(const float* audio, size_t numSamples, double sampleRate);

    // Check room consistency throughout recording
    bool checkConsistency(const float* audio, size_t numSamples, double sampleRate);

    // Estimate room impulse response
    std::vector<float> estimateIR(const float* audio, size_t numSamples, double sampleRate);

    // Compare room acoustics between two recordings
    double compareRooms(const float* audio1, size_t numSamples1,
                        const float* audio2, size_t numSamples2,
                        double sampleRate);

private:
    std::vector<float> extractRoomFeatures(const float* audio, size_t numSamples, double sampleRate);
};

// =============================================================================
// FORENSIC REPORT GENERATOR
// =============================================================================

class ForensicReportGenerator {
public:
    struct ReportOptions {
        bool includeENF = true;
        bool includeTampering = true;
        bool includeSpeakers = true;
        bool includeMetadata = true;
        bool includeRoomAcoustics = true;
        bool includeWaveform = true;
        bool includeSpectrogram = true;
        bool includeChainOfCustody = true;
        std::string outputFormat = "PDF";  // PDF, HTML, JSON, XML
        std::string language = "en";       // en, es, fr, de, etc.
    };

    ForensicReportGenerator();
    ~ForensicReportGenerator();

    // Generate comprehensive forensic report
    bool generateReport(const std::string& audioFilePath,
                        const std::string& outputPath,
                        const ReportOptions& options);

    // Generate court-ready evidence package
    bool generateEvidencePackage(const std::string& audioFilePath,
                                 const std::string& outputDir,
                                 const std::string& caseNumber,
                                 const std::string& examinerName);

private:
    std::unique_ptr<ENFAnalyzer> enf_;
    std::unique_ptr<TamperDetector> tamper_;
    std::unique_ptr<SpeakerIdentifier> speaker_;
    std::unique_ptr<MetadataForensics> metadata_;
    std::unique_ptr<RoomAcousticsAnalyzer> room_;
    std::unique_ptr<ChainOfCustody> custody_;
};

} // namespace Forensic
} // namespace MolinAntro
