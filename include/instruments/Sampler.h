#pragma once

#include "core/AudioBuffer.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <cmath>

namespace MolinAntro {
namespace Instruments {

/**
 * @brief Sample data container with metadata
 */
struct Sample {
    std::vector<float> data;
    int numChannels = 1;
    int sampleRate = 48000;
    int rootNote = 60;              // MIDI note this sample was recorded at
    int loopStart = 0;
    int loopEnd = 0;
    bool loopEnabled = false;
    float gainDb = 0.0f;
    std::string name;
    std::string path;
};

/**
 * @brief Sample zone mapping (velocity layers, key ranges)
 */
struct SampleZone {
    int sampleIndex = -1;
    int lowKey = 0;
    int highKey = 127;
    int lowVelocity = 1;
    int highVelocity = 127;
    int rootNote = 60;
    float pan = 0.0f;               // -1.0 to 1.0
    float tuning = 0.0f;            // cents
    float volume = 1.0f;
};

/**
 * @brief Voice for playing a sample with envelope
 */
class SamplerVoice {
public:
    enum class State {
        Idle,
        Attack,
        Hold,
        Decay,
        Sustain,
        Release
    };

    struct AHDSR {
        float attack = 0.005f;      // seconds
        float hold = 0.0f;
        float decay = 0.1f;
        float sustain = 1.0f;       // 0.0 - 1.0
        float release = 0.2f;       // seconds
    };

    void noteOn(int note, float velocity, const Sample* sample, int rootNote);
    void noteOff();
    void process(float* leftOut, float* rightOut, int numSamples, int sampleRate);
    void reset();

    bool isActive() const { return state_ != State::Idle; }
    int getNote() const { return note_; }

    AHDSR envelope;

private:
    float processEnvelope(int sampleRate);
    float interpolateSample(const Sample* sample, double position);

    const Sample* sample_ = nullptr;
    State state_ = State::Idle;
    int note_ = -1;
    float velocity_ = 0.0f;
    int rootNote_ = 60;

    double playbackPosition_ = 0.0;
    double playbackRate_ = 1.0;

    float envLevel_ = 0.0f;
    float envTarget_ = 1.0f;
    float envCoeff_ = 0.0f;
    int holdSamples_ = 0;
};

/**
 * @brief Multi-sample instrument with velocity layers and key zones
 */
class Sampler {
public:
    static constexpr int MAX_VOICES = 64;

    Sampler();
    ~Sampler();

    // Sample management
    int loadSample(const std::string& path);
    int addSample(const Sample& sample);
    void removeSample(int index);
    Sample* getSample(int index);
    const std::vector<Sample>& getSamples() const { return samples_; }

    // Zone mapping
    void addZone(const SampleZone& zone);
    void clearZones();
    const std::vector<SampleZone>& getZones() const { return zones_; }

    // Playback
    void prepare(int sampleRate, int maxBlockSize);
    void process(Core::AudioBuffer& buffer);
    void reset();

    void noteOn(int note, float velocity);
    void noteOff(int note);
    void allNotesOff();

    // Settings
    void setMaxVoices(int voices) { maxVoices_ = std::min(voices, MAX_VOICES); }
    void setEnvelope(const SamplerVoice::AHDSR& env) { globalEnvelope_ = env; }
    void setGlobalTuning(float cents) { globalTuning_ = cents; }
    void setGlobalVolume(float db) { globalVolume_ = db; }

    int getActiveVoiceCount() const;

private:
    const SampleZone* findZone(int note, int velocity) const;
    SamplerVoice* allocateVoice(int note);
    SamplerVoice* findVoice(int note);

    std::vector<Sample> samples_;
    std::vector<SampleZone> zones_;
    SamplerVoice voices_[MAX_VOICES];

    int sampleRate_ = 48000;
    int maxBlockSize_ = 512;
    int maxVoices_ = 32;

    SamplerVoice::AHDSR globalEnvelope_;
    float globalTuning_ = 0.0f;
    float globalVolume_ = 0.0f;
};

/**
 * @brief SFZ format loader
 */
class SFZLoader {
public:
    struct Region {
        std::string sample;
        int lokey = 0;
        int hikey = 127;
        int lovel = 1;
        int hivel = 127;
        int pitch_keycenter = 60;
        float volume = 0.0f;
        float pan = 0.0f;
        float tune = 0.0f;
        int loop_start = 0;
        int loop_end = 0;
        std::string loop_mode = "no_loop";
        float ampeg_attack = 0.001f;
        float ampeg_hold = 0.0f;
        float ampeg_decay = 0.0f;
        float ampeg_sustain = 100.0f;
        float ampeg_release = 0.001f;
    };

    bool load(const std::string& path, Sampler& sampler);

private:
    void parseRegion(const std::string& content, Region& region);
    std::string basePath_;
};

/**
 * @brief Sample library manager
 */
class SampleLibrary {
public:
    struct LibraryEntry {
        std::string name;
        std::string path;
        std::string category;
        std::string format;         // "sfz", "wav", "sf2"
        int numSamples = 0;
        size_t sizeBytes = 0;
    };

    void scanDirectory(const std::string& path, bool recursive = true);
    std::vector<LibraryEntry> getEntries() const { return entries_; }
    std::vector<LibraryEntry> search(const std::string& query) const;
    std::vector<LibraryEntry> filterByCategory(const std::string& category) const;

    bool loadIntoSampler(const std::string& path, Sampler& sampler);

private:
    std::vector<LibraryEntry> entries_;
};

} // namespace Instruments
} // namespace MolinAntro
