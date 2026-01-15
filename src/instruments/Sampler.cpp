#include "instruments/Sampler.h"
#include "dsp/AudioFile.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <filesystem>
#include <regex>

namespace MolinAntro {
namespace Instruments {

// ============================================================================
// SamplerVoice Implementation
// ============================================================================

void SamplerVoice::noteOn(int note, float velocity, const Sample* sample, int rootNote) {
    sample_ = sample;
    note_ = note;
    velocity_ = velocity;
    rootNote_ = rootNote;

    playbackPosition_ = 0.0;

    // Calculate playback rate for pitch transposition
    float semitones = static_cast<float>(note - rootNote);
    playbackRate_ = std::pow(2.0, semitones / 12.0);

    // Compensate for sample rate differences
    if (sample_ && sample_->sampleRate != 48000) {
        playbackRate_ *= static_cast<double>(sample_->sampleRate) / 48000.0;
    }

    state_ = State::Attack;
    envLevel_ = 0.0f;
    envTarget_ = 1.0f;
    holdSamples_ = 0;

    std::cout << "[Sampler] Voice noteOn: " << note << " vel: " << velocity
              << " rate: " << playbackRate_ << std::endl;
}

void SamplerVoice::noteOff() {
    if (state_ != State::Idle && state_ != State::Release) {
        state_ = State::Release;
        envTarget_ = 0.0f;
    }
}

void SamplerVoice::reset() {
    state_ = State::Idle;
    sample_ = nullptr;
    note_ = -1;
    velocity_ = 0.0f;
    playbackPosition_ = 0.0;
    envLevel_ = 0.0f;
}

void SamplerVoice::process(float* leftOut, float* rightOut, int numSamples, int sampleRate) {
    if (!sample_ || state_ == State::Idle) {
        return;
    }

    const int numChannels = sample_->numChannels;
    const size_t totalSamples = sample_->data.size() / numChannels;

    for (int i = 0; i < numSamples; ++i) {
        // Process envelope
        float envValue = processEnvelope(sampleRate);

        if (state_ == State::Idle) {
            break;
        }

        // Check if we've reached the end of the sample
        if (playbackPosition_ >= totalSamples) {
            if (sample_->loopEnabled && sample_->loopEnd > sample_->loopStart) {
                // Loop back
                playbackPosition_ = sample_->loopStart;
            } else {
                // End of sample
                reset();
                break;
            }
        }

        // Interpolate sample
        float sampleValue = interpolateSample(sample_, playbackPosition_);

        // Apply velocity and envelope
        float outputValue = sampleValue * velocity_ * envValue;

        // Output (mono or stereo)
        if (numChannels == 2) {
            float left = interpolateSample(sample_, playbackPosition_);
            float right = 0.0f;

            // Interleaved stereo
            size_t pos = static_cast<size_t>(playbackPosition_) * 2;
            if (pos + 1 < sample_->data.size()) {
                left = sample_->data[pos];
                right = sample_->data[pos + 1];
            }

            leftOut[i] += left * velocity_ * envValue;
            rightOut[i] += right * velocity_ * envValue;
        } else {
            leftOut[i] += outputValue;
            rightOut[i] += outputValue;
        }

        // Advance playback position
        playbackPosition_ += playbackRate_;
    }
}

float SamplerVoice::processEnvelope(int sampleRate) {
    switch (state_) {
        case State::Attack: {
            float attackSamples = envelope.attack * sampleRate;
            if (attackSamples > 0) {
                envLevel_ += 1.0f / attackSamples;
            } else {
                envLevel_ = 1.0f;
            }

            if (envLevel_ >= 1.0f) {
                envLevel_ = 1.0f;
                if (envelope.hold > 0) {
                    state_ = State::Hold;
                    holdSamples_ = static_cast<int>(envelope.hold * sampleRate);
                } else {
                    state_ = State::Decay;
                }
            }
            break;
        }

        case State::Hold: {
            holdSamples_--;
            if (holdSamples_ <= 0) {
                state_ = State::Decay;
            }
            break;
        }

        case State::Decay: {
            float decaySamples = envelope.decay * sampleRate;
            float target = envelope.sustain;
            if (decaySamples > 0) {
                envLevel_ -= (1.0f - target) / decaySamples;
            } else {
                envLevel_ = target;
            }

            if (envLevel_ <= target) {
                envLevel_ = target;
                state_ = State::Sustain;
            }
            break;
        }

        case State::Sustain:
            // Hold at sustain level
            envLevel_ = envelope.sustain;
            break;

        case State::Release: {
            float releaseSamples = envelope.release * sampleRate;
            if (releaseSamples > 0) {
                envLevel_ -= envLevel_ / releaseSamples;
            } else {
                envLevel_ = 0.0f;
            }

            if (envLevel_ <= 0.001f) {
                reset();
            }
            break;
        }

        case State::Idle:
            return 0.0f;
    }

    return envLevel_;
}

float SamplerVoice::interpolateSample(const Sample* sample, double position) {
    if (!sample || sample->data.empty()) {
        return 0.0f;
    }

    size_t numSamples = sample->data.size() / sample->numChannels;

    size_t pos0 = static_cast<size_t>(position);
    size_t pos1 = pos0 + 1;

    if (pos0 >= numSamples) {
        return 0.0f;
    }
    if (pos1 >= numSamples) {
        pos1 = pos0;
    }

    float frac = static_cast<float>(position - pos0);

    // Linear interpolation (can upgrade to cubic for better quality)
    float s0 = sample->data[pos0 * sample->numChannels];
    float s1 = sample->data[pos1 * sample->numChannels];

    return s0 + frac * (s1 - s0);
}

// ============================================================================
// Sampler Implementation
// ============================================================================

Sampler::Sampler() {
    std::cout << "[Sampler] Constructed" << std::endl;

    // Default envelope
    globalEnvelope_.attack = 0.005f;
    globalEnvelope_.hold = 0.0f;
    globalEnvelope_.decay = 0.1f;
    globalEnvelope_.sustain = 1.0f;
    globalEnvelope_.release = 0.2f;
}

Sampler::~Sampler() = default;

int Sampler::loadSample(const std::string& path) {
    DSP::AudioFile loader;
    if (!loader.load(path)) {
        std::cerr << "[Sampler] Failed to load sample: " << path << std::endl;
        return -1;
    }

    Sample sample;
    sample.path = path;
    sample.name = std::filesystem::path(path).stem().string();
    sample.numChannels = loader.getNumChannels();
    sample.sampleRate = loader.getSampleRate();
    sample.data = loader.getSamples();
    sample.rootNote = 60;  // Default to middle C
    sample.loopEnabled = false;

    samples_.push_back(std::move(sample));

    int index = static_cast<int>(samples_.size()) - 1;
    std::cout << "[Sampler] Loaded sample: " << sample.name
              << " (" << sample.data.size() / sample.numChannels << " samples, "
              << sample.numChannels << " channels)" << std::endl;

    return index;
}

int Sampler::addSample(const Sample& sample) {
    samples_.push_back(sample);
    return static_cast<int>(samples_.size()) - 1;
}

void Sampler::removeSample(int index) {
    if (index >= 0 && index < static_cast<int>(samples_.size())) {
        samples_.erase(samples_.begin() + index);

        // Update zone references
        for (auto& zone : zones_) {
            if (zone.sampleIndex == index) {
                zone.sampleIndex = -1;
            } else if (zone.sampleIndex > index) {
                zone.sampleIndex--;
            }
        }
    }
}

Sample* Sampler::getSample(int index) {
    if (index >= 0 && index < static_cast<int>(samples_.size())) {
        return &samples_[index];
    }
    return nullptr;
}

void Sampler::addZone(const SampleZone& zone) {
    zones_.push_back(zone);
}

void Sampler::clearZones() {
    zones_.clear();
}

void Sampler::prepare(int sampleRate, int maxBlockSize) {
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    std::cout << "[Sampler] Prepared: " << sampleRate << " Hz, "
              << maxVoices_ << " voices" << std::endl;

    reset();
}

void Sampler::process(Core::AudioBuffer& buffer) {
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    // Process all active voices
    for (int v = 0; v < MAX_VOICES; ++v) {
        if (!voices_[v].isActive()) continue;

        voices_[v].process(
            buffer.getWritePointer(0),
            numChannels > 1 ? buffer.getWritePointer(1) : buffer.getWritePointer(0),
            numSamples,
            sampleRate_
        );
    }

    // Normalize to prevent clipping with multiple voices
    int activeCount = getActiveVoiceCount();
    if (activeCount > 1) {
        float normalization = 1.0f / std::sqrt(static_cast<float>(activeCount));
        for (int ch = 0; ch < numChannels; ++ch) {
            buffer.applyGain(ch, normalization);
        }
    }

    // Apply global volume
    if (globalVolume_ != 0.0f) {
        float gain = std::pow(10.0f, globalVolume_ / 20.0f);
        for (int ch = 0; ch < numChannels; ++ch) {
            buffer.applyGain(ch, gain);
        }
    }
}

void Sampler::reset() {
    for (auto& voice : voices_) {
        voice.reset();
    }
}

void Sampler::noteOn(int note, float velocity) {
    // Find matching zone
    int intVelocity = static_cast<int>(velocity * 127.0f);
    const SampleZone* zone = findZone(note, intVelocity);

    if (!zone || zone->sampleIndex < 0 || zone->sampleIndex >= static_cast<int>(samples_.size())) {
        // No zone found, try direct sample mapping (one sample per octave spread)
        if (!samples_.empty()) {
            // Use first sample for all notes
            SamplerVoice* voice = allocateVoice(note);
            if (voice) {
                voice->envelope = globalEnvelope_;
                voice->noteOn(note, velocity, &samples_[0], samples_[0].rootNote);
            }
        }
        return;
    }

    SamplerVoice* voice = allocateVoice(note);
    if (!voice) {
        std::cerr << "[Sampler] No voices available" << std::endl;
        return;
    }

    voice->envelope = globalEnvelope_;
    voice->noteOn(note, velocity * zone->volume, &samples_[zone->sampleIndex], zone->rootNote);
}

void Sampler::noteOff(int note) {
    SamplerVoice* voice = findVoice(note);
    if (voice) {
        voice->noteOff();
        std::cout << "[Sampler] Note OFF: " << note << std::endl;
    }
}

void Sampler::allNotesOff() {
    for (auto& voice : voices_) {
        if (voice.isActive()) {
            voice.noteOff();
        }
    }
    std::cout << "[Sampler] All notes off" << std::endl;
}

int Sampler::getActiveVoiceCount() const {
    int count = 0;
    for (const auto& voice : voices_) {
        if (voice.isActive()) count++;
    }
    return count;
}

const SampleZone* Sampler::findZone(int note, int velocity) const {
    for (const auto& zone : zones_) {
        if (note >= zone.lowKey && note <= zone.highKey &&
            velocity >= zone.lowVelocity && velocity <= zone.highVelocity) {
            return &zone;
        }
    }
    return nullptr;
}

SamplerVoice* Sampler::allocateVoice(int note) {
    // First, try to find an idle voice
    for (int i = 0; i < maxVoices_; ++i) {
        if (!voices_[i].isActive()) {
            return &voices_[i];
        }
    }

    // If all voices are active, steal the oldest one (simple voice stealing)
    // In a production sampler, you'd use more sophisticated algorithms
    return &voices_[0];
}

SamplerVoice* Sampler::findVoice(int note) {
    for (int i = 0; i < MAX_VOICES; ++i) {
        if (voices_[i].isActive() && voices_[i].getNote() == note) {
            return &voices_[i];
        }
    }
    return nullptr;
}

// ============================================================================
// SFZLoader Implementation
// ============================================================================

bool SFZLoader::load(const std::string& path, Sampler& sampler) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[SFZLoader] Failed to open: " << path << std::endl;
        return false;
    }

    basePath_ = std::filesystem::path(path).parent_path().string();

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    // Simple SFZ parser
    std::regex regionRegex(R"(<region>([^<]*))");
    std::regex globalRegex(R"(<global>([^<]*))");

    Region globalDefaults;

    // Parse global section
    std::smatch globalMatch;
    if (std::regex_search(content, globalMatch, globalRegex)) {
        parseRegion(globalMatch[1].str(), globalDefaults);
    }

    // Parse regions
    std::sregex_iterator iter(content.begin(), content.end(), regionRegex);
    std::sregex_iterator end;

    int regionCount = 0;
    while (iter != end) {
        Region region = globalDefaults;  // Start with global defaults
        parseRegion((*iter)[1].str(), region);

        if (!region.sample.empty()) {
            // Load the sample
            std::string samplePath = basePath_ + "/" + region.sample;
            int sampleIndex = sampler.loadSample(samplePath);

            if (sampleIndex >= 0) {
                // Create zone
                SampleZone zone;
                zone.sampleIndex = sampleIndex;
                zone.lowKey = region.lokey;
                zone.highKey = region.hikey;
                zone.lowVelocity = region.lovel;
                zone.highVelocity = region.hivel;
                zone.rootNote = region.pitch_keycenter;
                zone.volume = std::pow(10.0f, region.volume / 20.0f);
                zone.pan = region.pan / 100.0f;
                zone.tuning = region.tune;

                // Set sample loop points
                Sample* sample = sampler.getSample(sampleIndex);
                if (sample) {
                    sample->rootNote = region.pitch_keycenter;
                    sample->loopStart = region.loop_start;
                    sample->loopEnd = region.loop_end;
                    sample->loopEnabled = (region.loop_mode != "no_loop");
                }

                sampler.addZone(zone);
                regionCount++;
            }
        }

        ++iter;
    }

    std::cout << "[SFZLoader] Loaded " << regionCount << " regions from: " << path << std::endl;
    return regionCount > 0;
}

void SFZLoader::parseRegion(const std::string& content, Region& region) {
    std::regex kvRegex(R"((\w+)=([^\s]+))");
    std::sregex_iterator iter(content.begin(), content.end(), kvRegex);
    std::sregex_iterator end;

    while (iter != end) {
        std::string key = (*iter)[1].str();
        std::string value = (*iter)[2].str();

        if (key == "sample") region.sample = value;
        else if (key == "lokey") region.lokey = std::stoi(value);
        else if (key == "hikey") region.hikey = std::stoi(value);
        else if (key == "lovel") region.lovel = std::stoi(value);
        else if (key == "hivel") region.hivel = std::stoi(value);
        else if (key == "pitch_keycenter") region.pitch_keycenter = std::stoi(value);
        else if (key == "volume") region.volume = std::stof(value);
        else if (key == "pan") region.pan = std::stof(value);
        else if (key == "tune") region.tune = std::stof(value);
        else if (key == "loop_start") region.loop_start = std::stoi(value);
        else if (key == "loop_end") region.loop_end = std::stoi(value);
        else if (key == "loop_mode") region.loop_mode = value;
        else if (key == "ampeg_attack") region.ampeg_attack = std::stof(value);
        else if (key == "ampeg_hold") region.ampeg_hold = std::stof(value);
        else if (key == "ampeg_decay") region.ampeg_decay = std::stof(value);
        else if (key == "ampeg_sustain") region.ampeg_sustain = std::stof(value);
        else if (key == "ampeg_release") region.ampeg_release = std::stof(value);

        ++iter;
    }
}

// ============================================================================
// SampleLibrary Implementation
// ============================================================================

void SampleLibrary::scanDirectory(const std::string& path, bool recursive) {
    try {
        auto iterator = recursive
            ? std::filesystem::recursive_directory_iterator(path)
            : std::filesystem::recursive_directory_iterator(path);

        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (!entry.is_regular_file()) continue;

            std::string ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

            if (ext == ".sfz" || ext == ".wav" || ext == ".flac" ||
                ext == ".aiff" || ext == ".sf2") {

                LibraryEntry libEntry;
                libEntry.name = entry.path().stem().string();
                libEntry.path = entry.path().string();
                libEntry.format = ext.substr(1);  // Remove the dot
                libEntry.sizeBytes = entry.file_size();

                // Try to detect category from path
                std::string pathStr = entry.path().string();
                if (pathStr.find("drum") != std::string::npos ||
                    pathStr.find("Drum") != std::string::npos) {
                    libEntry.category = "Drums";
                } else if (pathStr.find("bass") != std::string::npos ||
                           pathStr.find("Bass") != std::string::npos) {
                    libEntry.category = "Bass";
                } else if (pathStr.find("piano") != std::string::npos ||
                           pathStr.find("Piano") != std::string::npos) {
                    libEntry.category = "Keys";
                } else if (pathStr.find("synth") != std::string::npos ||
                           pathStr.find("Synth") != std::string::npos) {
                    libEntry.category = "Synth";
                } else if (pathStr.find("string") != std::string::npos ||
                           pathStr.find("String") != std::string::npos) {
                    libEntry.category = "Strings";
                } else {
                    libEntry.category = "Other";
                }

                entries_.push_back(libEntry);
            }
        }

        std::cout << "[SampleLibrary] Scanned " << entries_.size()
                  << " samples from: " << path << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[SampleLibrary] Error scanning: " << e.what() << std::endl;
    }
}

std::vector<SampleLibrary::LibraryEntry> SampleLibrary::search(const std::string& query) const {
    std::vector<LibraryEntry> results;
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);

    for (const auto& entry : entries_) {
        std::string lowerName = entry.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

        if (lowerName.find(lowerQuery) != std::string::npos) {
            results.push_back(entry);
        }
    }

    return results;
}

std::vector<SampleLibrary::LibraryEntry> SampleLibrary::filterByCategory(const std::string& category) const {
    std::vector<LibraryEntry> results;

    for (const auto& entry : entries_) {
        if (entry.category == category) {
            results.push_back(entry);
        }
    }

    return results;
}

bool SampleLibrary::loadIntoSampler(const std::string& path, Sampler& sampler) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (ext == ".sfz") {
        SFZLoader loader;
        return loader.load(path, sampler);
    } else if (ext == ".wav" || ext == ".flac" || ext == ".aiff") {
        int index = sampler.loadSample(path);
        if (index >= 0) {
            // Create a simple zone mapping the sample across all keys
            SampleZone zone;
            zone.sampleIndex = index;
            zone.lowKey = 0;
            zone.highKey = 127;
            zone.lowVelocity = 1;
            zone.highVelocity = 127;
            zone.rootNote = 60;
            sampler.addZone(zone);
            return true;
        }
    }

    return false;
}

} // namespace Instruments
} // namespace MolinAntro
