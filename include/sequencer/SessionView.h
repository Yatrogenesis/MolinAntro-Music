#pragma once

#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <atomic>
#include <mutex>

namespace MolinAntro {
namespace Sequencer {

// Forward declarations
class Clip;
class Track;
class Scene;
class SessionView;

/**
 * @brief Launch quantization options
 */
enum class LaunchQuantize {
    None,
    Bar,
    HalfBar,
    Beat,
    HalfBeat,
    Quarter,
    Eighth,
    Sixteenth
};

/**
 * @brief Clip launch modes (Ableton-style)
 */
enum class LaunchMode {
    Trigger,        // Play once, retrigger on click
    Gate,           // Play while held
    Toggle,         // Toggle on/off
    Repeat          // Repeat at quantize interval
};

/**
 * @brief Follow action types
 */
enum class FollowAction {
    None,
    Stop,
    PlayAgain,
    PlayPrevious,
    PlayNext,
    PlayFirst,
    PlayLast,
    PlayRandom,
    PlayOther
};

/**
 * @brief Clip - Audio or MIDI content container
 */
class Clip {
public:
    enum class Type { Audio, MIDI };
    enum class State { Stopped, Playing, Recording, Queued };

    Clip(const std::string& name = "Clip", Type type = Type::Audio);
    ~Clip();

    // Properties
    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }
    Type getType() const { return type_; }
    State getState() const { return state_.load(); }

    // Timing
    void setLength(double beats) { lengthBeats_ = beats; }
    double getLength() const { return lengthBeats_; }
    void setStartOffset(double beats) { startOffset_ = beats; }
    double getStartOffset() const { return startOffset_; }
    void setLoopEnabled(bool enable) { loopEnabled_ = enable; }
    bool isLoopEnabled() const { return loopEnabled_; }

    // Launch settings
    void setLaunchMode(LaunchMode mode) { launchMode_ = mode; }
    LaunchMode getLaunchMode() const { return launchMode_; }
    void setLaunchQuantize(LaunchQuantize quant) { launchQuantize_ = quant; }
    LaunchQuantize getLaunchQuantize() const { return launchQuantize_; }

    // Follow actions
    void setFollowAction(FollowAction action, double probability = 1.0);
    void setFollowTime(double beats) { followTime_ = beats; }

    // Audio content
    void setAudioBuffer(std::shared_ptr<Core::AudioBuffer> buffer) { audioBuffer_ = buffer; }
    Core::AudioBuffer* getAudioBuffer() { return audioBuffer_.get(); }

    // MIDI content
    void addMIDINote(const MIDI::Note& note);
    void removeMIDINote(int index);
    const std::vector<MIDI::Note>& getMIDINotes() const { return midiNotes_; }
    void clearMIDI() { midiNotes_.clear(); }

    // Warping (time-stretching)
    void setWarpEnabled(bool enable) { warpEnabled_ = enable; }
    bool isWarpEnabled() const { return warpEnabled_; }
    void setOriginalTempo(double bpm) { originalTempo_ = bpm; }
    double getOriginalTempo() const { return originalTempo_; }

    // Playback control
    void launch();
    void stop();
    void record();

    // Color for UI
    void setColor(uint32_t color) { color_ = color; }
    uint32_t getColor() const { return color_; }

private:
    std::string name_;
    Type type_;
    std::atomic<State> state_{State::Stopped};

    double lengthBeats_ = 4.0;
    double startOffset_ = 0.0;
    bool loopEnabled_ = true;
    double playPosition_ = 0.0;

    LaunchMode launchMode_ = LaunchMode::Trigger;
    LaunchQuantize launchQuantize_ = LaunchQuantize::Bar;

    FollowAction followAction_ = FollowAction::None;
    double followProbability_ = 1.0;
    double followTime_ = 0.0;

    std::shared_ptr<Core::AudioBuffer> audioBuffer_;
    std::vector<MIDI::Note> midiNotes_;

    bool warpEnabled_ = true;
    double originalTempo_ = 120.0;

    uint32_t color_ = 0xFF4444FF;  // Default red
};

/**
 * @brief ClipSlot - Container for a clip in the session grid
 */
class ClipSlot {
public:
    enum class State { Empty, HasClip, Playing, Recording, Queued };

    ClipSlot();
    ~ClipSlot();

    void setClip(std::shared_ptr<Clip> clip);
    Clip* getClip() { return clip_.get(); }
    bool isEmpty() const { return clip_ == nullptr; }
    State getState() const;

    void launch();
    void stop();
    void record();

    void setStopButton(bool hasStop) { hasStopButton_ = hasStop; }
    bool hasStopButton() const { return hasStopButton_; }

private:
    std::shared_ptr<Clip> clip_;
    bool hasStopButton_ = true;
};

/**
 * @brief Track - Vertical column of clip slots
 */
class Track {
public:
    enum class Type { Audio, MIDI, Return, Master };
    enum class Arm { Off, On, Exclusive };

    Track(const std::string& name = "Track", Type type = Type::Audio);
    ~Track();

    // Properties
    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }
    Type getType() const { return type_; }

    // Clip slots
    void setNumSlots(int num);
    int getNumSlots() const { return static_cast<int>(slots_.size()); }
    ClipSlot* getSlot(int index);
    void insertClip(int slotIndex, std::shared_ptr<Clip> clip);

    // Mixer
    void setVolume(float dB) { volumeDb_ = dB; }
    float getVolume() const { return volumeDb_; }
    void setPan(float pan) { pan_ = std::clamp(pan, -1.0f, 1.0f); }
    float getPan() const { return pan_; }
    void setMute(bool mute) { muted_ = mute; }
    bool isMuted() const { return muted_; }
    void setSolo(bool solo) { soloed_ = solo; }
    bool isSoloed() const { return soloed_; }

    // Recording
    void setArm(Arm arm) { arm_ = arm; }
    Arm getArm() const { return arm_; }

    // Monitoring
    void setMonitoring(bool enable) { monitoring_ = enable; }
    bool isMonitoring() const { return monitoring_; }

    // Routing
    void setInput(const std::string& input) { inputRouting_ = input; }
    void setOutput(const std::string& output) { outputRouting_ = output; }

    // Color
    void setColor(uint32_t color) { color_ = color; }
    uint32_t getColor() const { return color_; }

    // Audio processing
    void processAudio(Core::AudioBuffer& buffer, double bpm);
    void processMIDI(std::vector<MIDI::MIDIMessage>& messages, double bpm);

    // Stop all clips
    void stopAllClips();

private:
    std::string name_;
    Type type_;
    std::vector<std::unique_ptr<ClipSlot>> slots_;

    float volumeDb_ = 0.0f;
    float pan_ = 0.0f;
    bool muted_ = false;
    bool soloed_ = false;

    Arm arm_ = Arm::Off;
    bool monitoring_ = false;

    std::string inputRouting_ = "Default";
    std::string outputRouting_ = "Master";

    uint32_t color_ = 0xFF888888;

    int activeSlotIndex_ = -1;
};

/**
 * @brief Scene - Horizontal row of clips that can be launched together
 */
class Scene {
public:
    Scene(const std::string& name = "Scene");
    ~Scene();

    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    void setTempo(double bpm) { tempo_ = bpm; hasTempo_ = true; }
    double getTempo() const { return tempo_; }
    bool hasTempo() const { return hasTempo_; }
    void clearTempo() { hasTempo_ = false; }

    void setTimeSignature(int numerator, int denominator);
    int getTimeSignatureNumerator() const { return timeSignatureNum_; }
    int getTimeSignatureDenominator() const { return timeSignatureDenom_; }

    void setColor(uint32_t color) { color_ = color; }
    uint32_t getColor() const { return color_; }

    // Launch all clips in this scene row
    void launch(SessionView* session);
    void stop(SessionView* session);

private:
    std::string name_;
    double tempo_ = 120.0;
    bool hasTempo_ = false;
    int timeSignatureNum_ = 4;
    int timeSignatureDenom_ = 4;
    uint32_t color_ = 0xFF444444;
};

/**
 * @brief SessionView - Main session controller (Ableton-style clip launcher)
 */
class SessionView {
public:
    SessionView();
    ~SessionView();

    // Track management
    Track* createTrack(const std::string& name, Track::Type type = Track::Type::Audio);
    void deleteTrack(int index);
    Track* getTrack(int index);
    int getNumTracks() const { return static_cast<int>(tracks_.size()); }

    // Scene management
    Scene* createScene(const std::string& name = "");
    void deleteScene(int index);
    Scene* getScene(int index);
    int getNumScenes() const { return static_cast<int>(scenes_.size()); }

    // Clip access
    Clip* getClip(int trackIndex, int sceneIndex);
    void setClip(int trackIndex, int sceneIndex, std::shared_ptr<Clip> clip);

    // Launching
    void launchClip(int trackIndex, int sceneIndex);
    void stopClip(int trackIndex, int sceneIndex);
    void launchScene(int sceneIndex);
    void stopScene(int sceneIndex);
    void stopAllClips();

    // Global settings
    void setLaunchQuantize(LaunchQuantize quant) { globalLaunchQuantize_ = quant; }
    LaunchQuantize getLaunchQuantize() const { return globalLaunchQuantize_; }
    void setRecordQuantize(LaunchQuantize quant) { recordQuantize_ = quant; }
    void setSelectOnLaunch(bool enable) { selectOnLaunch_ = enable; }

    // Recording
    void startRecording(int trackIndex, int sceneIndex);
    void stopRecording();

    // Transport sync
    void setTransport(double bpm, double positionBeats, bool playing);
    double getBPM() const { return bpm_; }
    double getPositionBeats() const { return positionBeats_; }
    bool isPlaying() const { return playing_; }

    // Audio processing (called from audio thread)
    void processAudio(Core::AudioBuffer& buffer);
    void processMIDI(std::vector<MIDI::MIDIMessage>& messages);

    // Callbacks
    using ClipStateCallback = std::function<void(int track, int scene, Clip::State)>;
    void setClipStateCallback(ClipStateCallback callback) { clipStateCallback_ = callback; }

private:
    void updateQuantization();
    double getNextQuantizePoint(LaunchQuantize quant);

    std::vector<std::unique_ptr<Track>> tracks_;
    std::vector<std::unique_ptr<Scene>> scenes_;

    LaunchQuantize globalLaunchQuantize_ = LaunchQuantize::Bar;
    LaunchQuantize recordQuantize_ = LaunchQuantize::None;
    bool selectOnLaunch_ = true;

    double bpm_ = 120.0;
    double positionBeats_ = 0.0;
    bool playing_ = false;

    int sampleRate_ = 48000;
    int blockSize_ = 512;

    std::mutex mutex_;
    ClipStateCallback clipStateCallback_;
};

/**
 * @brief ArrangementView - Linear timeline for arrangement
 */
class ArrangementView {
public:
    struct Region {
        int trackIndex;
        double startBeats;
        double endBeats;
        std::shared_ptr<Clip> clip;
        double clipOffset = 0.0;
        bool muted = false;
    };

    ArrangementView();
    ~ArrangementView();

    // Regions
    int addRegion(const Region& region);
    void removeRegion(int regionId);
    void moveRegion(int regionId, double newStartBeats);
    void resizeRegion(int regionId, double newEndBeats);
    void splitRegion(int regionId, double splitPoint);
    std::vector<Region> getRegionsInRange(double startBeats, double endBeats);

    // Loop
    void setLoopRange(double startBeats, double endBeats);
    void setLoopEnabled(bool enable) { loopEnabled_ = enable; }
    bool isLoopEnabled() const { return loopEnabled_; }
    double getLoopStart() const { return loopStart_; }
    double getLoopEnd() const { return loopEnd_; }

    // Markers
    void addMarker(double position, const std::string& name);
    void removeMarker(int index);

    // Locators
    void setLocator(int id, double position);
    double getLocator(int id);

    // Time selection
    void setTimeSelection(double start, double end);
    std::pair<double, double> getTimeSelection() const;

    // Export from session view
    void captureFromSession(SessionView& session, double durationBeats);

    // Playback
    void processAudio(Core::AudioBuffer& buffer, double positionBeats, double bpm);

private:
    std::vector<Region> regions_;
    int nextRegionId_ = 0;

    bool loopEnabled_ = false;
    double loopStart_ = 0.0;
    double loopEnd_ = 4.0;

    std::vector<std::pair<double, std::string>> markers_;
    std::map<int, double> locators_;

    double selectionStart_ = 0.0;
    double selectionEnd_ = 0.0;
};

} // namespace Sequencer
} // namespace MolinAntro
