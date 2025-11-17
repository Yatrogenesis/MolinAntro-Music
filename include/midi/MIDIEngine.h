#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <functional>
#include <atomic>
#include <queue>
#include <mutex>

namespace MolinAntro {
namespace MIDI {

/**
 * @brief MIDI Message structure
 */
struct MIDIMessage {
    enum class Type {
        NoteOff = 0x80,
        NoteOn = 0x90,
        PolyPressure = 0xA0,
        ControlChange = 0xB0,
        ProgramChange = 0xC0,
        ChannelPressure = 0xD0,
        PitchBend = 0xE0,
        SystemExclusive = 0xF0
    };

    Type type;
    uint8_t channel;      // 0-15
    uint8_t data1;        // Note number, CC number, etc.
    uint8_t data2;        // Velocity, CC value, etc.
    double timestamp;     // Sample-accurate timestamp

    // MPE (MIDI Polyphonic Expression) support
    bool isMPE = false;
    uint8_t mpeZone = 0;  // 0 = lower zone, 1 = upper zone

    MIDIMessage() : type(Type::NoteOn), channel(0), data1(0), data2(0), timestamp(0.0) {}

    MIDIMessage(Type t, uint8_t ch, uint8_t d1, uint8_t d2, double ts = 0.0)
        : type(t), channel(ch), data1(d1), data2(d2), timestamp(ts) {}

    // Helper constructors
    static MIDIMessage noteOn(uint8_t channel, uint8_t note, uint8_t velocity, double timestamp = 0.0);
    static MIDIMessage noteOff(uint8_t channel, uint8_t note, uint8_t velocity = 0, double timestamp = 0.0);
    static MIDIMessage controlChange(uint8_t channel, uint8_t controller, uint8_t value, double timestamp = 0.0);
    static MIDIMessage pitchBend(uint8_t channel, int16_t value, double timestamp = 0.0);

    // MIDI 2.0 compatibility
    bool isMIDI2 = false;
    uint32_t data32 = 0;  // For MIDI 2.0 extended data
};

/**
 * @brief Thread-safe MIDI message queue
 */
class MIDIMessageQueue {
public:
    void push(const MIDIMessage& msg);
    bool pop(MIDIMessage& msg);
    void clear();
    size_t size() const;
    bool empty() const;

private:
    std::queue<MIDIMessage> queue_;
    mutable std::mutex mutex_;
};

/**
 * @brief MIDI Input/Output Device
 */
struct MIDIDevice {
    std::string name;
    std::string id;
    bool isInput;
    bool isOutput;
    bool isAvailable;
};

/**
 * @brief MIDI Engine - handles all MIDI I/O and processing
 */
class MIDIEngine {
public:
    MIDIEngine();
    ~MIDIEngine();

    /**
     * @brief Initialize MIDI engine
     */
    bool initialize();

    /**
     * @brief Shutdown MIDI engine
     */
    void shutdown();

    /**
     * @brief Get list of available MIDI devices
     */
    std::vector<MIDIDevice> getDevices() const;

    /**
     * @brief Open MIDI input device
     */
    bool openInputDevice(const std::string& deviceId);

    /**
     * @brief Open MIDI output device
     */
    bool openOutputDevice(const std::string& deviceId);

    /**
     * @brief Close MIDI input device
     */
    void closeInputDevice();

    /**
     * @brief Close MIDI output device
     */
    void closeOutputDevice();

    /**
     * @brief Send MIDI message
     */
    void sendMessage(const MIDIMessage& msg);

    /**
     * @brief Get incoming MIDI messages
     */
    std::vector<MIDIMessage> getMessages();

    /**
     * @brief Set MIDI input callback
     */
    using MIDICallback = std::function<void(const MIDIMessage&)>;
    void setInputCallback(MIDICallback callback);

    /**
     * @brief Enable MPE (MIDI Polyphonic Expression)
     */
    void enableMPE(bool enable, uint8_t lowerZoneChannels = 8, uint8_t upperZoneChannels = 0);

    /**
     * @brief Get MPE status
     */
    bool isMPEEnabled() const { return mpeEnabled_; }

    /**
     * @brief All notes off (panic button)
     */
    void allNotesOff();

private:
    std::atomic<bool> initialized_{false};
    std::atomic<bool> mpeEnabled_{false};
    uint8_t mpeLowerZoneChannels_ = 8;
    uint8_t mpeUpperZoneChannels_ = 0;

    MIDIMessageQueue inputQueue_;
    MIDIMessageQueue outputQueue_;

    MIDICallback inputCallback_;
    std::mutex callbackMutex_;

    std::string inputDeviceId_;
    std::string outputDeviceId_;

    void processMIDIInput();
    void processMIDIOutput();
};

/**
 * @brief MIDI Sequencer - for recording and playback
 */
class MIDISequencer {
public:
    struct MIDIEvent {
        MIDIMessage message;
        double positionInBeats;
        int track;
    };

    MIDISequencer();
    ~MIDISequencer() = default;

    /**
     * @brief Start recording
     */
    void startRecording(int track = 0);

    /**
     * @brief Stop recording
     */
    void stopRecording();

    /**
     * @brief Is recording
     */
    bool isRecording() const { return recording_.load(); }

    /**
     * @brief Record MIDI message
     */
    void recordMessage(const MIDIMessage& msg, double positionInBeats);

    /**
     * @brief Get events for playback
     */
    std::vector<MIDIEvent> getEventsInRange(double startBeat, double endBeat) const;

    /**
     * @brief Clear all events
     */
    void clear();

    /**
     * @brief Get event count
     */
    size_t getEventCount() const { return events_.size(); }

    /**
     * @brief Quantize events to grid
     */
    void quantize(double gridSize);

private:
    std::vector<MIDIEvent> events_;
    std::atomic<bool> recording_{false};
    int recordingTrack_ = 0;
    mutable std::mutex eventsMutex_;
};

} // namespace MIDI
} // namespace MolinAntro
