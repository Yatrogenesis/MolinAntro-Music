#include "midi/MIDIEngine.h"
#include <iostream>
#include <algorithm>
#include <cmath>

namespace MolinAntro {
namespace MIDI {

// MIDIMessage helper constructors
MIDIMessage MIDIMessage::noteOn(uint8_t channel, uint8_t note, uint8_t velocity, double timestamp) {
    return MIDIMessage(Type::NoteOn, channel, note, velocity, timestamp);
}

MIDIMessage MIDIMessage::noteOff(uint8_t channel, uint8_t note, uint8_t velocity, double timestamp) {
    return MIDIMessage(Type::NoteOff, channel, note, velocity, timestamp);
}

MIDIMessage MIDIMessage::controlChange(uint8_t channel, uint8_t controller, uint8_t value, double timestamp) {
    return MIDIMessage(Type::ControlChange, channel, controller, value, timestamp);
}

MIDIMessage MIDIMessage::pitchBend(uint8_t channel, int16_t value, double timestamp) {
    // Pitch bend is 14-bit (0-16383, center = 8192)
    uint16_t unsignedValue = static_cast<uint16_t>(value + 8192);
    uint8_t lsb = unsignedValue & 0x7F;
    uint8_t msb = (unsignedValue >> 7) & 0x7F;
    return MIDIMessage(Type::PitchBend, channel, lsb, msb, timestamp);
}

// MIDIMessageQueue implementation
void MIDIMessageQueue::push(const MIDIMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(msg);
}

bool MIDIMessageQueue::pop(MIDIMessage& msg) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queue_.empty()) {
        return false;
    }
    msg = queue_.front();
    queue_.pop();
    return true;
}

void MIDIMessageQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        queue_.pop();
    }
}

size_t MIDIMessageQueue::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

bool MIDIMessageQueue::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

// MIDIEngine implementation
MIDIEngine::MIDIEngine() {
    std::cout << "[MIDIEngine] Constructed" << std::endl;
}

MIDIEngine::~MIDIEngine() {
    shutdown();
    std::cout << "[MIDIEngine] Destroyed" << std::endl;
}

bool MIDIEngine::initialize() {
    if (initialized_.load()) {
        std::cerr << "[MIDIEngine] Already initialized" << std::endl;
        return false;
    }

    std::cout << "[MIDIEngine] Initializing..." << std::endl;

    // Initialize MIDI subsystem (platform-specific)
    // For MVP, we'll use a virtual MIDI interface

    initialized_.store(true);
    std::cout << "[MIDIEngine] Initialized successfully" << std::endl;
    return true;
}

void MIDIEngine::shutdown() {
    if (!initialized_.load()) {
        return;
    }

    closeInputDevice();
    closeOutputDevice();

    inputQueue_.clear();
    outputQueue_.clear();

    initialized_.store(false);
    std::cout << "[MIDIEngine] Shutdown complete" << std::endl;
}

std::vector<MIDIDevice> MIDIEngine::getDevices() const {
    std::vector<MIDIDevice> devices;

    // Virtual MIDI device for MVP
    MIDIDevice virtualIn;
    virtualIn.name = "MolinAntro Virtual MIDI In";
    virtualIn.id = "virtual_midi_in";
    virtualIn.isInput = true;
    virtualIn.isOutput = false;
    virtualIn.isAvailable = true;
    devices.push_back(virtualIn);

    MIDIDevice virtualOut;
    virtualOut.name = "MolinAntro Virtual MIDI Out";
    virtualOut.id = "virtual_midi_out";
    virtualOut.isInput = false;
    virtualOut.isOutput = true;
    virtualOut.isAvailable = true;
    devices.push_back(virtualOut);

    return devices;
}

bool MIDIEngine::openInputDevice(const std::string& deviceId) {
    if (!initialized_.load()) {
        std::cerr << "[MIDIEngine] Not initialized" << std::endl;
        return false;
    }

    inputDeviceId_ = deviceId;
    std::cout << "[MIDIEngine] Opened input device: " << deviceId << std::endl;
    return true;
}

bool MIDIEngine::openOutputDevice(const std::string& deviceId) {
    if (!initialized_.load()) {
        std::cerr << "[MIDIEngine] Not initialized" << std::endl;
        return false;
    }

    outputDeviceId_ = deviceId;
    std::cout << "[MIDIEngine] Opened output device: " << deviceId << std::endl;
    return true;
}

void MIDIEngine::closeInputDevice() {
    if (!inputDeviceId_.empty()) {
        std::cout << "[MIDIEngine] Closed input device: " << inputDeviceId_ << std::endl;
        inputDeviceId_.clear();
    }
}

void MIDIEngine::closeOutputDevice() {
    if (!outputDeviceId_.empty()) {
        std::cout << "[MIDIEngine] Closed output device: " << outputDeviceId_ << std::endl;
        outputDeviceId_.clear();
    }
}

void MIDIEngine::sendMessage(const MIDIMessage& msg) {
    if (!initialized_.load()) {
        return;
    }

    outputQueue_.push(msg);

    // Process MPE if enabled
    if (mpeEnabled_ && msg.isMPE) {
        // MPE processing logic here
    }
}

std::vector<MIDIMessage> MIDIEngine::getMessages() {
    std::vector<MIDIMessage> messages;

    MIDIMessage msg;
    while (inputQueue_.pop(msg)) {
        messages.push_back(msg);
    }

    return messages;
}

void MIDIEngine::setInputCallback(MIDICallback callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    inputCallback_ = std::move(callback);
}

void MIDIEngine::enableMPE(bool enable, uint8_t lowerZoneChannels, uint8_t upperZoneChannels) {
    mpeEnabled_.store(enable);
    mpeLowerZoneChannels_ = lowerZoneChannels;
    mpeUpperZoneChannels_ = upperZoneChannels;

    std::cout << "[MIDIEngine] MPE " << (enable ? "enabled" : "disabled");
    if (enable) {
        std::cout << " (Lower: " << static_cast<int>(lowerZoneChannels)
                  << ", Upper: " << static_cast<int>(upperZoneChannels) << ")";
    }
    std::cout << std::endl;
}

void MIDIEngine::allNotesOff() {
    std::cout << "[MIDIEngine] All notes off (panic)" << std::endl;

    // Send note off for all channels and all notes
    for (uint8_t channel = 0; channel < 16; ++channel) {
        for (uint8_t note = 0; note < 128; ++note) {
            sendMessage(MIDIMessage::noteOff(channel, note, 0));
        }
        // Also send All Notes Off CC
        sendMessage(MIDIMessage::controlChange(channel, 123, 0));
    }
}

void MIDIEngine::processMIDIInput() {
    // Called from audio thread to process incoming MIDI
    std::lock_guard<std::mutex> lock(callbackMutex_);

    MIDIMessage msg;
    while (inputQueue_.pop(msg)) {
        if (inputCallback_) {
            inputCallback_(msg);
        }
    }
}

void MIDIEngine::processMIDIOutput() {
    // Called from audio thread to send outgoing MIDI
    MIDIMessage msg;
    while (outputQueue_.pop(msg)) {
        // Send to physical MIDI device
        // Platform-specific implementation would go here
    }
}

// MIDISequencer implementation
MIDISequencer::MIDISequencer() {
    std::cout << "[MIDISequencer] Constructed" << std::endl;
}

void MIDISequencer::startRecording(int track) {
    recordingTrack_ = track;
    recording_.store(true);
    std::cout << "[MIDISequencer] Recording started on track " << track << std::endl;
}

void MIDISequencer::stopRecording() {
    recording_.store(false);
    std::cout << "[MIDISequencer] Recording stopped ("
              << events_.size() << " events recorded)" << std::endl;
}

void MIDISequencer::recordMessage(const MIDIMessage& msg, double positionInBeats) {
    if (!recording_.load()) {
        return;
    }

    std::lock_guard<std::mutex> lock(eventsMutex_);

    MIDIEvent event;
    event.message = msg;
    event.positionInBeats = positionInBeats;
    event.track = recordingTrack_;

    events_.push_back(event);
}

std::vector<MIDISequencer::MIDIEvent> MIDISequencer::getEventsInRange(
    double startBeat, double endBeat) const {

    std::lock_guard<std::mutex> lock(eventsMutex_);

    std::vector<MIDIEvent> result;
    for (const auto& event : events_) {
        if (event.positionInBeats >= startBeat && event.positionInBeats < endBeat) {
            result.push_back(event);
        }
    }

    return result;
}

void MIDISequencer::clear() {
    std::lock_guard<std::mutex> lock(eventsMutex_);
    events_.clear();
    std::cout << "[MIDISequencer] Cleared all events" << std::endl;
}

void MIDISequencer::quantize(double gridSize) {
    std::lock_guard<std::mutex> lock(eventsMutex_);

    for (auto& event : events_) {
        double quantized = std::round(event.positionInBeats / gridSize) * gridSize;
        event.positionInBeats = quantized;
    }

    std::cout << "[MIDISequencer] Quantized " << events_.size()
              << " events to grid size " << gridSize << std::endl;
}

} // namespace MIDI
} // namespace MolinAntro
