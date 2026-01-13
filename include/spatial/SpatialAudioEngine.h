/**
 * MolinAntro DAW - Spatial Audio Engine
 * SOTA x5 Implementation - Immersive 3D Audio
 *
 * Features:
 * - Ambisonics (1st to 7th order HOA)
 * - Dolby Atmos object-based audio
 * - Binaural rendering with HRTF
 * - Distance attenuation with air absorption
 * - Room simulation (early reflections + late reverb)
 * - 360° VR/AR audio
 * - Apple Spatial Audio compatible
 * - Sony 360 Reality Audio compatible
 *
 * Author: F. Molina-Burgos / MolinAntro Technologies
 * Copyright (C) 2026 - All Rights Reserved
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <mutex>
#include <atomic>

namespace MolinAntro {
namespace Spatial {

// =============================================================================
// CONSTANTS AND TYPES
// =============================================================================

// Maximum supported Ambisonics order
constexpr int MAX_AMBISONIC_ORDER = 7;
// Number of channels for Nth order = (N+1)^2
constexpr int MAX_AMBISONIC_CHANNELS = (MAX_AMBISONIC_ORDER + 1) * (MAX_AMBISONIC_ORDER + 1);

// Maximum simultaneous audio objects
constexpr int MAX_AUDIO_OBJECTS = 128;

// Maximum room dimensions (meters)
constexpr float MAX_ROOM_SIZE = 100.0f;

// Speed of sound at 20°C (m/s)
constexpr float SPEED_OF_SOUND = 343.0f;

// 3D Vector
struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Vec3() = default;
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

    float length() const { return std::sqrt(x*x + y*y + z*z); }
    Vec3 normalized() const {
        float len = length();
        if (len < 1e-6f) return Vec3(0, 0, 1);
        return Vec3(x/len, y/len, z/len);
    }

    Vec3 operator+(const Vec3& other) const { return Vec3(x + other.x, y + other.y, z + other.z); }
    Vec3 operator-(const Vec3& other) const { return Vec3(x - other.x, y - other.y, z - other.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }

    static float dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return Vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
    }
};

// Quaternion for rotations
struct Quaternion {
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;

    Quaternion() = default;
    Quaternion(float w_, float x_, float y_, float z_) : w(w_), x(x_), y(y_), z(z_) {}

    static Quaternion fromEuler(float yaw, float pitch, float roll);
    static Quaternion fromAxisAngle(const Vec3& axis, float angle);

    Quaternion operator*(const Quaternion& other) const;
    Vec3 rotate(const Vec3& v) const;

    Quaternion conjugate() const { return Quaternion(w, -x, -y, -z); }
    Quaternion normalized() const;
};

// Spherical coordinates (azimuth, elevation, distance)
struct Spherical {
    float azimuth = 0.0f;      // -180 to +180 degrees (0 = front, +90 = left)
    float elevation = 0.0f;    // -90 to +90 degrees (0 = horizon, +90 = up)
    float distance = 1.0f;     // meters

    Vec3 toCartesian() const;
    static Spherical fromCartesian(const Vec3& v);
};

// =============================================================================
// HRTF DATABASE
// =============================================================================

class HRTFDatabase {
public:
    struct Measurement {
        float azimuth;
        float elevation;
        std::vector<float> leftIR;
        std::vector<float> rightIR;
        size_t irLength;
    };

    HRTFDatabase();
    ~HRTFDatabase();

    // Load SOFA format HRTF
    bool loadSOFA(const std::string& path);

    // Load built-in HRTF (MIT KEMAR, CIPIC, etc.)
    bool loadBuiltIn(const std::string& name);

    // Get interpolated HRTF for any direction
    void getHRTF(float azimuth, float elevation,
                 std::vector<float>& leftIR, std::vector<float>& rightIR) const;

    // Get minimum phase HRTF (lower latency)
    void getMinPhaseHRTF(float azimuth, float elevation,
                         std::vector<float>& leftIR, std::vector<float>& rightIR) const;

    size_t getIRLength() const { return irLength_; }
    int getSampleRate() const { return sampleRate_; }

private:
    std::vector<Measurement> measurements_;
    size_t irLength_ = 0;
    int sampleRate_ = 48000;

    // Spatial lookup table for fast nearest-neighbor search
    std::vector<int> spatialIndex_;
    int azimuthResolution_ = 360;
    int elevationResolution_ = 180;

    void buildSpatialIndex();
    void findNearestMeasurements(float azimuth, float elevation,
                                  int& idx1, int& idx2, int& idx3, int& idx4,
                                  float& w1, float& w2, float& w3, float& w4) const;
};

// =============================================================================
// BINAURAL RENDERER
// =============================================================================

class BinauralRenderer {
public:
    BinauralRenderer(int sampleRate = 48000, size_t blockSize = 512);
    ~BinauralRenderer();

    void setHRTF(std::shared_ptr<HRTFDatabase> hrtf);

    // Render mono source to binaural stereo
    void renderMono(const float* input, float* outputL, float* outputR,
                    size_t numSamples, float azimuth, float elevation, float distance);

    // Render with position interpolation (for moving sources)
    void renderMonoInterpolated(const float* input, float* outputL, float* outputR,
                                size_t numSamples,
                                float startAz, float startEl, float startDist,
                                float endAz, float endEl, float endDist);

    // Render Ambisonics to binaural
    void renderAmbisonics(const float* const* ambiChannels, int order,
                          float* outputL, float* outputR, size_t numSamples,
                          const Quaternion& headRotation);

    void setEnableNearField(bool enable) { nearFieldEnabled_ = enable; }
    void setEnableRoomSimulation(bool enable) { roomSimEnabled_ = enable; }

private:
    int sampleRate_;
    size_t blockSize_;
    std::shared_ptr<HRTFDatabase> hrtf_;

    // Partitioned convolution for HRTF
    struct ConvolutionState {
        std::vector<float> fdlReal;
        std::vector<float> fdlImag;
        std::vector<float> overlap;
        size_t fdlIndex = 0;
    };
    ConvolutionState convStateL_;
    ConvolutionState convStateR_;

    // Previous position for interpolation
    float prevAzimuth_ = 0.0f;
    float prevElevation_ = 0.0f;
    float prevDistance_ = 1.0f;

    bool nearFieldEnabled_ = true;
    bool roomSimEnabled_ = true;

    void applyDistanceAttenuation(float* buffer, size_t numSamples, float distance);
    void applyAirAbsorption(float* buffer, size_t numSamples, float distance);
    void applyNearFieldILD(float* left, float* right, size_t numSamples,
                           float azimuth, float distance);
};

// =============================================================================
// AMBISONICS ENCODER/DECODER
// =============================================================================

class AmbisonicsProcessor {
public:
    enum class Normalization {
        SN3D,   // Schmidt semi-normalized (default)
        N3D,    // Full 3D normalization
        FuMa    // Furse-Malham (legacy B-format)
    };

    enum class ChannelOrdering {
        ACN,    // Ambisonics Channel Numbering (standard)
        FuMa    // Furse-Malham ordering (legacy)
    };

    AmbisonicsProcessor(int order = 3, int sampleRate = 48000);
    ~AmbisonicsProcessor();

    void setNormalization(Normalization norm) { normalization_ = norm; }
    void setChannelOrdering(ChannelOrdering order) { channelOrdering_ = order; }

    // Encode mono source to Ambisonics
    void encode(const float* input, float* const* ambiChannels, size_t numSamples,
                float azimuth, float elevation, float spread = 0.0f);

    // Encode with position in Cartesian coordinates
    void encodeCartesian(const float* input, float* const* ambiChannels, size_t numSamples,
                         const Vec3& position, float spread = 0.0f);

    // Decode to speaker array
    void decode(const float* const* ambiChannels, float* const* speakerOutputs,
                size_t numSamples, size_t numSpeakers);

    // Rotate soundfield
    void rotate(float* const* ambiChannels, size_t numSamples, const Quaternion& rotation);

    // Set speaker layout for decoding
    void setSpeakerLayout(const std::vector<Spherical>& positions);
    void setStandardLayout(const std::string& name);  // "5.1", "7.1", "7.1.4", "9.1.6", etc.

    int getOrder() const { return order_; }
    int getNumChannels() const { return (order_ + 1) * (order_ + 1); }

private:
    int order_;
    int sampleRate_;
    Normalization normalization_ = Normalization::SN3D;
    ChannelOrdering channelOrdering_ = ChannelOrdering::ACN;

    // Spherical harmonic coefficients for encoding
    std::vector<float> shCoeffs_;

    // Decoder matrix (speakers x ambi channels)
    std::vector<std::vector<float>> decoderMatrix_;
    std::vector<Spherical> speakerPositions_;

    // Rotation matrices for each order
    std::vector<std::vector<std::vector<float>>> rotationMatrices_;

    void calculateSHCoeffs(float azimuth, float elevation);
    void buildDecoderMatrix();
    void buildRotationMatrix(const Quaternion& rotation);

    float getSphericalHarmonic(int order, int degree, float azimuth, float elevation) const;
    float getNormalizationFactor(int order, int degree) const;
};

// =============================================================================
// AUDIO OBJECT (for object-based audio)
// =============================================================================

struct AudioObject {
    int id = -1;
    bool active = false;

    Vec3 position;
    Vec3 velocity;         // For Doppler effect
    Quaternion orientation;

    float gain = 1.0f;
    float spread = 0.0f;   // 0 = point source, 1 = omnidirectional
    float size = 0.0f;     // Object size in meters (for extent panning)

    // Directivity
    bool directivityEnabled = false;
    float directivityPattern = 0.0f;  // 0 = omni, 1 = cardioid, 2 = figure-8

    // Distance model
    enum class DistanceModel {
        Linear,
        Inverse,
        Exponential,
        Custom
    };
    DistanceModel distanceModel = DistanceModel::Inverse;
    float referenceDistance = 1.0f;
    float maxDistance = 100.0f;
    float rolloffFactor = 1.0f;

    // Effects
    bool enableDoppler = true;
    bool enableAirAbsorption = true;
    bool enableOcclusion = false;
    float occlusionFactor = 0.0f;
};

// =============================================================================
// ROOM MODEL
// =============================================================================

class RoomModel {
public:
    struct Material {
        std::string name;
        std::array<float, 8> absorption;  // Absorption coefficients at 8 octave bands
        std::array<float, 8> scattering;  // Scattering coefficients

        static Material Concrete();
        static Material Wood();
        static Material Glass();
        static Material Carpet();
        static Material Curtain();
        static Material Acoustic();
    };

    RoomModel();
    ~RoomModel();

    void setDimensions(float width, float height, float depth);
    void setMaterials(Material floor, Material ceiling,
                      Material frontWall, Material backWall,
                      Material leftWall, Material rightWall);

    // Set listener position
    void setListenerPosition(const Vec3& pos, const Quaternion& orientation);

    // Calculate early reflections for a source
    void calculateEarlyReflections(const Vec3& sourcePos,
                                   std::vector<Vec3>& reflectionPositions,
                                   std::vector<float>& reflectionGains,
                                   std::vector<float>& reflectionDelays,
                                   int maxOrder = 2) const;

    // Get RT60 (reverberation time) for each frequency band
    std::array<float, 8> getRT60() const;

    // Get room impulse response
    void getRoomIR(const Vec3& sourcePos, float* ir, size_t irLength, int sampleRate) const;

private:
    Vec3 dimensions_;
    Material materials_[6];  // Floor, ceiling, front, back, left, right
    Vec3 listenerPosition_;
    Quaternion listenerOrientation_;

    void imageSourceMethod(const Vec3& sourcePos, int order,
                           std::vector<Vec3>& images,
                           std::vector<float>& gains,
                           std::vector<int>& reflectionCounts) const;
};

// =============================================================================
// SPATIAL AUDIO SCENE
// =============================================================================

class SpatialAudioScene {
public:
    SpatialAudioScene(int sampleRate = 48000, size_t blockSize = 512);
    ~SpatialAudioScene();

    // Object management
    int createObject();
    void destroyObject(int objectId);
    AudioObject& getObject(int objectId);
    const AudioObject& getObject(int objectId) const;

    // Set object audio input
    void setObjectInput(int objectId, const float* input, size_t numSamples);

    // Listener
    void setListenerPosition(const Vec3& position);
    void setListenerOrientation(const Quaternion& orientation);
    void setListenerVelocity(const Vec3& velocity);  // For Doppler

    // Room
    void setRoom(std::shared_ptr<RoomModel> room);
    void setRoomEnabled(bool enabled) { roomEnabled_ = enabled; }

    // Rendering
    enum class OutputFormat {
        Stereo,
        Binaural,
        Quad,
        Surround51,
        Surround71,
        Surround714,
        Ambisonics1,
        Ambisonics2,
        Ambisonics3,
        Ambisonics4,
        Ambisonics5,
        Ambisonics6,
        Ambisonics7
    };

    void setOutputFormat(OutputFormat format);
    void render(float* const* outputs, size_t numSamples);

    // Reverb
    void setReverbEnabled(bool enabled) { reverbEnabled_ = enabled; }
    void setReverbMix(float wet, float dry) { reverbWet_ = wet; reverbDry_ = dry; }

    // HRTF for binaural
    void setHRTF(std::shared_ptr<HRTFDatabase> hrtf);

private:
    int sampleRate_;
    size_t blockSize_;

    std::array<AudioObject, MAX_AUDIO_OBJECTS> objects_;
    std::array<std::vector<float>, MAX_AUDIO_OBJECTS> objectBuffers_;

    Vec3 listenerPosition_;
    Quaternion listenerOrientation_;
    Vec3 listenerVelocity_;

    std::shared_ptr<RoomModel> room_;
    bool roomEnabled_ = true;
    bool reverbEnabled_ = true;
    float reverbWet_ = 0.3f;
    float reverbDry_ = 1.0f;

    OutputFormat outputFormat_ = OutputFormat::Binaural;
    int numOutputChannels_ = 2;

    std::unique_ptr<AmbisonicsProcessor> ambisonics_;
    std::unique_ptr<BinauralRenderer> binaural_;
    std::shared_ptr<HRTFDatabase> hrtf_;

    std::vector<float> ambiBuffer_;
    std::vector<float> reverbBuffer_;

    std::mutex mutex_;

    void renderToAmbisonics(float* const* ambiOutputs, size_t numSamples);
    void renderToBinaural(float* outputL, float* outputR, size_t numSamples);
    void renderToSpeakers(float* const* outputs, size_t numSamples, int numSpeakers);

    float calculateDistanceGain(const AudioObject& obj, float distance) const;
    float calculateDopplerShift(const AudioObject& obj, const Vec3& toListener) const;
};

// =============================================================================
// DOLBY ATMOS ENCODER (ADM BWF compatible)
// =============================================================================

class DolbyAtmosEncoder {
public:
    DolbyAtmosEncoder(int sampleRate = 48000);
    ~DolbyAtmosEncoder();

    // Create bed channels (7.1.2, 7.1.4, 9.1.6)
    void createBed(const std::string& name, int channels);

    // Add object to mix
    int addObject(const std::string& name);

    // Set object position (normalized -1 to 1)
    void setObjectPosition(int objectId, float x, float y, float z);
    void setObjectSize(int objectId, float size);
    void setObjectGain(int objectId, float gainDB);

    // Render to speaker layout
    void render(const float* const* objectInputs, const float* const* bedInputs,
                float* const* outputs, size_t numSamples,
                const std::string& speakerLayout);

    // Export ADM BWF metadata
    void exportADM(const std::string& path);

private:
    int sampleRate_;
    // ... implementation details
};

} // namespace Spatial
} // namespace MolinAntro
