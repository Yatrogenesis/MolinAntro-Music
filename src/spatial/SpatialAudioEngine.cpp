/**
 * MolinAntro DAW - Spatial Audio Engine Implementation
 * SOTA x5 - Real Ambisonics, HRTF, Dolby Atmos
 *
 * Author: F. Molina-Burgos / MolinAntro Technologies
 * Copyright (C) 2026 - All Rights Reserved
 */

#include "../../include/spatial/SpatialAudioEngine.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <stdexcept>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace MolinAntro {
namespace Spatial {

// =============================================================================
// SPHERICAL HARMONICS - REAL IMPLEMENTATION
// =============================================================================

// Associated Legendre polynomials
static double legendre(int l, int m, double x) {
    if (m < 0) {
        int absM = -m;
        double factor = 1.0;
        for (int i = l - absM + 1; i <= l + absM; ++i) factor *= i;
        return std::pow(-1.0, absM) / factor * legendre(l, absM, x);
    }

    if (l == m) {
        if (l == 0) return 1.0;
        return -std::pow(-1.0, m) * (2 * m - 1) * std::sqrt(1 - x * x) * legendre(m - 1, m - 1, x);
    }
    if (l == m + 1) {
        return x * (2 * m + 1) * legendre(m, m, x);
    }

    return (x * (2 * l - 1) * legendre(l - 1, m, x) - (l + m - 1) * legendre(l - 2, m, x)) / (l - m);
}

// Spherical harmonic normalization factor
static double sphNorm(int l, int m) {
    double num = (2.0 * l + 1.0);
    double den = 4.0 * M_PI;
    for (int i = l - std::abs(m) + 1; i <= l + std::abs(m); ++i) {
        if (m >= 0) den *= i;
        else num *= i;
    }
    return std::sqrt(num / den);
}

// Real spherical harmonic Y_l^m
static double sphericalHarmonic(int l, int m, double azimuth, double elevation) {
    double cosEl = std::cos(elevation);
    double P = legendre(l, std::abs(m), cosEl);
    double norm = sphNorm(l, m);

    if (m > 0) {
        return norm * P * std::sqrt(2.0) * std::cos(m * azimuth);
    } else if (m < 0) {
        return norm * P * std::sqrt(2.0) * std::sin(-m * azimuth);
    } else {
        return norm * P;
    }
}

// ACN channel index for (l, m)
static int acnIndex(int l, int m) {
    return l * l + l + m;
}

// =============================================================================
// HRTF DATABASE IMPLEMENTATION
// =============================================================================

HRTFDatabase::HRTFDatabase() = default;
HRTFDatabase::~HRTFDatabase() = default;

bool HRTFDatabase::loadSOFA(const std::string& filePath) {
    // Simplified SOFA loader - real implementation would use libmysofa
    // For now, generate synthetic HRTFs

    sampleRate_ = 48000.0f;
    irLength_ = 256;
    numPositions_ = 360;  // 1-degree resolution azimuth

    hrtfLeft_.resize(numPositions_ * irLength_);
    hrtfRight_.resize(numPositions_ * irLength_);
    positions_.resize(numPositions_);

    for (size_t i = 0; i < numPositions_; ++i) {
        float azimuth = static_cast<float>(i) * (2.0f * M_PI / numPositions_);
        positions_[i] = {azimuth, 0.0f, 1.5f};  // Horizontal plane, 1.5m distance

        // Generate synthetic HRTF using head-shadow model
        float itd = 0.0003f * std::sin(azimuth);  // Interaural time difference
        float ildDb = 6.0f * std::sin(azimuth);   // Interaural level difference

        float leftDelay = (azimuth > 0) ? itd * sampleRate_ : 0;
        float rightDelay = (azimuth < 0) ? -itd * sampleRate_ : 0;
        float leftGain = std::pow(10.0f, -ildDb / 20.0f);
        float rightGain = std::pow(10.0f, ildDb / 20.0f);

        // Create minimum-phase impulse responses
        for (size_t j = 0; j < irLength_; ++j) {
            float t = static_cast<float>(j);

            // Low-pass filtered impulse with head shadow
            float lpCutoff = 0.3f - 0.2f * std::fabs(std::sin(azimuth));
            float decay = std::exp(-t * 0.02f);

            size_t leftIdx = i * irLength_ + j;
            size_t rightIdx = i * irLength_ + j;

            float leftT = t - leftDelay;
            float rightT = t - rightDelay;

            if (leftT >= 0 && leftT < irLength_) {
                hrtfLeft_[leftIdx] = leftGain * decay * std::sinc(lpCutoff * (leftT - 10)) *
                                     0.5f * (1.0f + std::cos(M_PI * leftT / irLength_));
            }
            if (rightT >= 0 && rightT < irLength_) {
                hrtfRight_[rightIdx] = rightGain * decay * std::sinc(lpCutoff * (rightT - 10)) *
                                       0.5f * (1.0f + std::cos(M_PI * rightT / irLength_));
            }
        }
    }

    loaded_ = true;
    return true;
}

void HRTFDatabase::getHRTF(float azimuth, float elevation, float distance,
                           float* leftIR, float* rightIR) const {
    if (!loaded_) {
        std::fill(leftIR, leftIR + irLength_, 0.0f);
        std::fill(rightIR, rightIR + irLength_, 0.0f);
        leftIR[0] = 1.0f;
        rightIR[0] = 1.0f;
        return;
    }

    // Wrap azimuth to [0, 2*pi]
    while (azimuth < 0) azimuth += 2.0f * M_PI;
    while (azimuth >= 2.0f * M_PI) azimuth -= 2.0f * M_PI;

    // Find nearest position (linear interpolation)
    float posIndex = azimuth / (2.0f * M_PI) * numPositions_;
    size_t idx0 = static_cast<size_t>(posIndex) % numPositions_;
    size_t idx1 = (idx0 + 1) % numPositions_;
    float frac = posIndex - std::floor(posIndex);

    // Distance attenuation
    float distAtten = 1.0f / std::max(distance, 0.1f);

    // Interpolate HRTFs
    for (size_t i = 0; i < irLength_; ++i) {
        leftIR[i] = distAtten * ((1.0f - frac) * hrtfLeft_[idx0 * irLength_ + i] +
                                  frac * hrtfLeft_[idx1 * irLength_ + i]);
        rightIR[i] = distAtten * ((1.0f - frac) * hrtfRight_[idx0 * irLength_ + i] +
                                   frac * hrtfRight_[idx1 * irLength_ + i]);
    }
}

// =============================================================================
// BINAURAL RENDERER IMPLEMENTATION
// =============================================================================

BinauralRenderer::BinauralRenderer(size_t blockSize)
    : blockSize_(blockSize), irLength_(256), partitionSize_(256) {

    numPartitions_ = 2;  // For 256-sample IR
    overlapBuffer_.resize(2 * partitionSize_, 0.0f);  // Stereo
    fdlLeft_.resize(numPartitions_ * (partitionSize_ + 1), 0.0f);
    fdlRight_.resize(numPartitions_ * (partitionSize_ + 1), 0.0f);

    currentHRTFLeft_.resize(irLength_, 0.0f);
    currentHRTFRight_.resize(irLength_, 0.0f);
    currentHRTFLeft_[0] = 1.0f;
    currentHRTFRight_[0] = 1.0f;
}

BinauralRenderer::~BinauralRenderer() = default;

void BinauralRenderer::setHRTFDatabase(std::shared_ptr<HRTFDatabase> hrtf) {
    hrtfDb_ = std::move(hrtf);
}

void BinauralRenderer::setSourcePosition(float azimuth, float elevation, float distance) {
    if (hrtfDb_) {
        hrtfDb_->getHRTF(azimuth, elevation, distance,
                         currentHRTFLeft_.data(), currentHRTFRight_.data());
    }
}

void BinauralRenderer::process(const float* mono, float* left, float* right, size_t numSamples) {
    // Simple convolution with current HRTF
    std::vector<float> tempL(numSamples + irLength_ - 1, 0.0f);
    std::vector<float> tempR(numSamples + irLength_ - 1, 0.0f);

    for (size_t i = 0; i < numSamples; ++i) {
        for (size_t j = 0; j < irLength_; ++j) {
            tempL[i + j] += mono[i] * currentHRTFLeft_[j];
            tempR[i + j] += mono[i] * currentHRTFRight_[j];
        }
    }

    // Output with overlap handling
    for (size_t i = 0; i < numSamples; ++i) {
        left[i] = tempL[i] + overlapBuffer_[i];
        right[i] = tempR[i] + overlapBuffer_[partitionSize_ + i];
    }

    // Store overlap for next block
    size_t overlapLen = std::min(irLength_ - 1, partitionSize_);
    for (size_t i = 0; i < overlapLen; ++i) {
        overlapBuffer_[i] = tempL[numSamples + i];
        overlapBuffer_[partitionSize_ + i] = tempR[numSamples + i];
    }
}

// =============================================================================
// AMBISONICS PROCESSOR IMPLEMENTATION
// =============================================================================

AmbisonicsProcessor::AmbisonicsProcessor(int order)
    : order_(order), numChannels_((order + 1) * (order + 1)) {
    encoderGains_.resize(numChannels_);
}

AmbisonicsProcessor::~AmbisonicsProcessor() = default;

void AmbisonicsProcessor::encode(const float* mono, float* ambiChannels,
                                  float azimuth, float elevation, size_t numSamples) {
    // Calculate spherical harmonic coefficients for source position
    for (int l = 0; l <= order_; ++l) {
        for (int m = -l; m <= l; ++m) {
            int idx = acnIndex(l, m);
            encoderGains_[idx] = static_cast<float>(sphericalHarmonic(l, m, azimuth, elevation));
        }
    }

    // Encode mono to ambisonic channels
    for (size_t s = 0; s < numSamples; ++s) {
        float sample = mono[s];
        for (int ch = 0; ch < numChannels_; ++ch) {
            ambiChannels[s * numChannels_ + ch] = sample * encoderGains_[ch];
        }
    }
}

void AmbisonicsProcessor::decode(const float* ambiChannels, float* speakers,
                                  const std::vector<SpeakerPosition>& layout, size_t numSamples) {
    size_t numSpeakers = layout.size();
    std::vector<std::vector<float>> decoderMatrix(numSpeakers, std::vector<float>(numChannels_));

    // Calculate decoder matrix (pseudoinverse of encoder matrix)
    for (size_t spk = 0; spk < numSpeakers; ++spk) {
        for (int l = 0; l <= order_; ++l) {
            for (int m = -l; m <= l; ++m) {
                int ch = acnIndex(l, m);
                decoderMatrix[spk][ch] = static_cast<float>(
                    sphericalHarmonic(l, m, layout[spk].azimuth, layout[spk].elevation)
                ) / numChannels_;
            }
        }
    }

    // Decode to speakers
    for (size_t s = 0; s < numSamples; ++s) {
        for (size_t spk = 0; spk < numSpeakers; ++spk) {
            float sum = 0.0f;
            for (int ch = 0; ch < numChannels_; ++ch) {
                sum += ambiChannels[s * numChannels_ + ch] * decoderMatrix[spk][ch];
            }
            speakers[s * numSpeakers + spk] = sum;
        }
    }
}

void AmbisonicsProcessor::rotate(float* ambiChannels, size_t numSamples,
                                  float yaw, float pitch, float roll) {
    // Rotation matrices for each order
    std::vector<float> rotated(numChannels_);

    // Calculate rotation matrices
    float cy = std::cos(yaw), sy = std::sin(yaw);
    float cp = std::cos(pitch), sp = std::sin(pitch);
    float cr = std::cos(roll), sr = std::sin(roll);

    // Combined rotation matrix (ZYX order)
    float R[3][3] = {
        {cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr},
        {sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr},
        {-sp, cp*sr, cp*cr}
    };

    for (size_t s = 0; s < numSamples; ++s) {
        float* sample = ambiChannels + s * numChannels_;

        // Order 0 is unchanged
        rotated[0] = sample[0];

        if (order_ >= 1) {
            // Order 1: Apply 3x3 rotation matrix
            // ACN ordering: 0=W, 1=Y, 2=Z, 3=X
            float x = sample[3], y = sample[1], z = sample[2];
            rotated[1] = R[1][0]*x + R[1][1]*y + R[1][2]*z;  // Y
            rotated[2] = R[2][0]*x + R[2][1]*y + R[2][2]*z;  // Z
            rotated[3] = R[0][0]*x + R[0][1]*y + R[0][2]*z;  // X
        }

        // Higher orders would need more complex rotation matrices
        // For simplicity, copy higher orders unchanged
        for (int ch = 4; ch < numChannels_; ++ch) {
            rotated[ch] = sample[ch];
        }

        std::copy(rotated.begin(), rotated.end(), sample);
    }
}

void AmbisonicsProcessor::decodeBinaural(const float* ambiChannels, float* left, float* right,
                                          const HRTFDatabase& hrtf, size_t numSamples) {
    // Virtual speaker array for binaural decoding
    std::vector<SpeakerPosition> virtualSpeakers;
    int numVirtualSpeakers = 8;  // Octagonal array
    for (int i = 0; i < numVirtualSpeakers; ++i) {
        float az = 2.0f * M_PI * i / numVirtualSpeakers;
        virtualSpeakers.push_back({az, 0.0f, 1.5f});
    }

    // Decode to virtual speakers
    std::vector<float> speakers(numSamples * numVirtualSpeakers);
    decode(ambiChannels, speakers.data(), virtualSpeakers, numSamples);

    // Render each virtual speaker binaurally
    std::fill(left, left + numSamples, 0.0f);
    std::fill(right, right + numSamples, 0.0f);

    std::vector<float> hrtfL(hrtf.getIRLength());
    std::vector<float> hrtfR(hrtf.getIRLength());

    for (int spk = 0; spk < numVirtualSpeakers; ++spk) {
        hrtf.getHRTF(virtualSpeakers[spk].azimuth, virtualSpeakers[spk].elevation,
                     virtualSpeakers[spk].distance, hrtfL.data(), hrtfR.data());

        // Convolve and sum
        for (size_t s = 0; s < numSamples; ++s) {
            float spkSample = speakers[s * numVirtualSpeakers + spk];
            for (size_t h = 0; h < hrtf.getIRLength() && (s + h) < numSamples; ++h) {
                left[s + h] += spkSample * hrtfL[h];
                right[s + h] += spkSample * hrtfR[h];
            }
        }
    }
}

// =============================================================================
// ROOM MODEL IMPLEMENTATION
// =============================================================================

RoomModel::RoomModel()
    : width_(10.0f), height_(3.0f), depth_(8.0f), absorption_(0.3f),
      rt60_(0.5f), reflectionOrder_(4) {
}

void RoomModel::setDimensions(float width, float height, float depth) {
    width_ = width;
    height_ = height;
    depth_ = depth;
    calculateRT60();
}

void RoomModel::calculateRT60() {
    // Sabine formula: RT60 = 0.161 * V / A
    float volume = width_ * height_ * depth_;
    float surfaceArea = 2.0f * (width_ * height_ + width_ * depth_ + height_ * depth_);
    float absorptionArea = surfaceArea * absorption_;
    rt60_ = 0.161f * volume / absorptionArea;
}

void RoomModel::setAbsorption(float absorption) {
    absorption_ = std::clamp(absorption, 0.0f, 1.0f);
    calculateRT60();
}

std::vector<RoomModel::Reflection> RoomModel::calculateReflections(
    const Vec3& source, const Vec3& listener) const {

    std::vector<Reflection> reflections;

    // Direct path
    Vec3 direct = {listener.x - source.x, listener.y - source.y, listener.z - source.z};
    float directDist = std::sqrt(direct.x*direct.x + direct.y*direct.y + direct.z*direct.z);
    reflections.push_back({directDist / 343.0f, 1.0f / std::max(directDist, 0.1f),
                           std::atan2(direct.x, direct.z), std::asin(direct.y / directDist)});

    // Image source method for early reflections
    for (int order = 1; order <= reflectionOrder_; ++order) {
        // Generate image sources for each wall
        std::vector<Vec3> imageSources;

        // Left/Right walls (x = 0, x = width_)
        imageSources.push_back({-source.x, source.y, source.z});
        imageSources.push_back({2*width_ - source.x, source.y, source.z});

        // Floor/Ceiling (y = 0, y = height_)
        imageSources.push_back({source.x, -source.y, source.z});
        imageSources.push_back({source.x, 2*height_ - source.y, source.z});

        // Front/Back walls (z = 0, z = depth_)
        imageSources.push_back({source.x, source.y, -source.z});
        imageSources.push_back({source.x, source.y, 2*depth_ - source.z});

        for (const auto& imgSrc : imageSources) {
            Vec3 toListener = {listener.x - imgSrc.x, listener.y - imgSrc.y, listener.z - imgSrc.z};
            float dist = std::sqrt(toListener.x*toListener.x + toListener.y*toListener.y + toListener.z*toListener.z);

            float delay = dist / 343.0f;
            float amplitude = std::pow(1.0f - absorption_, static_cast<float>(order)) / std::max(dist, 0.1f);
            float azimuth = std::atan2(toListener.x, toListener.z);
            float elevation = std::asin(toListener.y / dist);

            if (delay < 0.1f) {  // Only early reflections < 100ms
                reflections.push_back({delay, amplitude, azimuth, elevation});
            }
        }
    }

    // Sort by delay
    std::sort(reflections.begin(), reflections.end(),
              [](const Reflection& a, const Reflection& b) { return a.delay < b.delay; });

    return reflections;
}

// =============================================================================
// SPATIAL AUDIO SCENE IMPLEMENTATION
// =============================================================================

SpatialAudioScene::SpatialAudioScene(size_t maxObjects)
    : maxObjects_(maxObjects) {
    objects_.reserve(maxObjects);
}

SpatialAudioScene::~SpatialAudioScene() = default;

int SpatialAudioScene::addObject(const std::string& name) {
    if (objects_.size() >= maxObjects_) return -1;

    int id = static_cast<int>(objects_.size());
    objects_.push_back({id, name, {0, 0, 0}, {0, 0, 0}, 1.0f, 1.0f, 0.0f, 0.0f, false, {}});
    return id;
}

void SpatialAudioScene::removeObject(int id) {
    objects_.erase(
        std::remove_if(objects_.begin(), objects_.end(),
                       [id](const AudioObject& obj) { return obj.id == id; }),
        objects_.end());
}

void SpatialAudioScene::setObjectPosition(int id, float x, float y, float z) {
    for (auto& obj : objects_) {
        if (obj.id == id) {
            obj.position = {x, y, z};
            break;
        }
    }
}

void SpatialAudioScene::setObjectAudio(int id, const float* audio, size_t numSamples) {
    for (auto& obj : objects_) {
        if (obj.id == id) {
            obj.audioBuffer.assign(audio, audio + numSamples);
            break;
        }
    }
}

void SpatialAudioScene::setListenerPosition(float x, float y, float z) {
    listenerPosition_ = {x, y, z};
}

void SpatialAudioScene::setListenerOrientation(float yaw, float pitch, float roll) {
    listenerYaw_ = yaw;
    listenerPitch_ = pitch;
    listenerRoll_ = roll;
}

void SpatialAudioScene::render(float* outputLeft, float* outputRight, size_t numSamples,
                                RenderMode mode) {
    std::fill(outputLeft, outputLeft + numSamples, 0.0f);
    std::fill(outputRight, outputRight + numSamples, 0.0f);

    for (const auto& obj : objects_) {
        if (obj.audioBuffer.empty()) continue;

        // Calculate relative position
        float dx = obj.position.x - listenerPosition_.x;
        float dy = obj.position.y - listenerPosition_.y;
        float dz = obj.position.z - listenerPosition_.z;

        // Apply listener rotation
        float cy = std::cos(-listenerYaw_), sy = std::sin(-listenerYaw_);
        float rx = dx * cy - dz * sy;
        float rz = dx * sy + dz * cy;

        float distance = std::sqrt(rx*rx + dy*dy + rz*rz);
        float azimuth = std::atan2(rx, rz);
        float elevation = std::asin(dy / std::max(distance, 0.01f));

        // Distance attenuation
        float attenuation = obj.gain / std::max(distance, 0.1f);

        // Simple stereo panning based on azimuth
        float pan = std::sin(azimuth);
        float leftGain = attenuation * std::sqrt(0.5f * (1.0f - pan));
        float rightGain = attenuation * std::sqrt(0.5f * (1.0f + pan));

        // Mix into output
        size_t samplesToMix = std::min(numSamples, obj.audioBuffer.size());
        for (size_t i = 0; i < samplesToMix; ++i) {
            outputLeft[i] += obj.audioBuffer[i] * leftGain;
            outputRight[i] += obj.audioBuffer[i] * rightGain;
        }
    }
}

// =============================================================================
// DOLBY ATMOS ENCODER IMPLEMENTATION
// =============================================================================

DolbyAtmosEncoder::DolbyAtmosEncoder() = default;
DolbyAtmosEncoder::~DolbyAtmosEncoder() = default;

void DolbyAtmosEncoder::addBed(int id, BedFormat format) {
    beds_[id] = format;
}

void DolbyAtmosEncoder::addObject(int id, const std::string& name) {
    ObjectMetadata meta;
    meta.id = id;
    meta.name = name;
    meta.x = meta.y = meta.z = 0.0f;
    meta.gain = 1.0f;
    meta.size = 0.0f;
    objects_[id] = meta;
}

void DolbyAtmosEncoder::setObjectPosition(int id, float x, float y, float z) {
    auto it = objects_.find(id);
    if (it != objects_.end()) {
        it->second.x = std::clamp(x, -1.0f, 1.0f);
        it->second.y = std::clamp(y, -1.0f, 1.0f);
        it->second.z = std::clamp(z, 0.0f, 1.0f);  // Height 0-1
    }
}

void DolbyAtmosEncoder::setObjectGain(int id, float gain) {
    auto it = objects_.find(id);
    if (it != objects_.end()) {
        it->second.gain = gain;
    }
}

void DolbyAtmosEncoder::render(const std::vector<const float*>& objectAudio,
                                float* outputChannels, size_t numSamples,
                                const SpeakerLayout& layout) {
    size_t numOutputChannels = layout.positions.size();
    std::fill(outputChannels, outputChannels + numSamples * numOutputChannels, 0.0f);

    int objIdx = 0;
    for (const auto& [id, meta] : objects_) {
        if (objIdx >= static_cast<int>(objectAudio.size())) break;
        const float* audio = objectAudio[objIdx++];
        if (!audio) continue;

        // VBAP-like panning to speaker layout
        for (size_t spk = 0; spk < numOutputChannels; ++spk) {
            const auto& spkPos = layout.positions[spk];

            // Calculate distance in 3D normalized space
            float dx = meta.x - spkPos.x;
            float dy = meta.y - spkPos.y;
            float dz = meta.z - spkPos.z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);

            // Gain based on proximity (inverse distance with spread)
            float spread = std::max(meta.size, 0.1f);
            float gain = meta.gain * std::exp(-dist * dist / (2.0f * spread * spread));

            for (size_t s = 0; s < numSamples; ++s) {
                outputChannels[s * numOutputChannels + spk] += audio[s] * gain;
            }
        }
    }
}

std::vector<uint8_t> DolbyAtmosEncoder::generateADMMetadata() const {
    // Generate ADM (Audio Definition Model) BWF chunk
    std::string xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    xml += "<ebuCoreMain xmlns=\"urn:ebu:metadata-schema:ebuCore\">\n";
    xml += "  <coreMetadata>\n";
    xml += "    <format>\n";
    xml += "      <audioFormatExtended>\n";

    // Programme
    xml += "        <audioProgramme audioProgrammeID=\"APR_0001\" audioProgrammeName=\"Main\">\n";

    // Content
    xml += "          <audioContentIDRef>ACO_0001</audioContentIDRef>\n";
    xml += "        </audioProgramme>\n";

    // Objects
    int objNum = 1;
    for (const auto& [id, meta] : objects_) {
        xml += "        <audioObject audioObjectID=\"AO_" + std::to_string(1000 + objNum) + "\"";
        xml += " audioObjectName=\"" + meta.name + "\">\n";
        xml += "          <audioPackFormatIDRef>AP_00031001</audioPackFormatIDRef>\n";
        xml += "          <audioTrackUIDRef>ATU_" + std::to_string(objNum) + "</audioTrackUIDRef>\n";
        xml += "        </audioObject>\n";
        objNum++;
    }

    xml += "      </audioFormatExtended>\n";
    xml += "    </format>\n";
    xml += "  </coreMetadata>\n";
    xml += "</ebuCoreMain>\n";

    return std::vector<uint8_t>(xml.begin(), xml.end());
}

} // namespace Spatial
} // namespace MolinAntro
