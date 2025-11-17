#include "core/Transport.h"
#include <cmath>
#include <iostream>

namespace MolinAntro {
namespace Core {

Transport::Transport() {
    std::cout << "[Transport] Constructed" << std::endl;
}

void Transport::play() {
    timeInfo_.isPlaying = true;
    timeInfo_.isRecording = false;
    playRequested_.store(true);
    std::cout << "[Transport] Play requested" << std::endl;
    notifyStateChange();
}

void Transport::stop() {
    timeInfo_.isPlaying = false;
    timeInfo_.isRecording = false;
    timeInfo_.samplePosition = 0.0;
    timeInfo_.timeInSeconds = 0.0;
    timeInfo_.bar = 1;
    timeInfo_.beat = 1;
    timeInfo_.tick = 0;
    stopRequested_.store(true);
    std::cout << "[Transport] Stop requested" << std::endl;
    notifyStateChange();
}

void Transport::pause() {
    timeInfo_.isPlaying = false;
    pauseRequested_.store(true);
    std::cout << "[Transport] Pause requested" << std::endl;
    notifyStateChange();
}

void Transport::record() {
    timeInfo_.isPlaying = true;
    timeInfo_.isRecording = true;
    std::cout << "[Transport] Record requested" << std::endl;
    notifyStateChange();
}

void Transport::setBPM(double bpm) {
    if (bpm < 20.0 || bpm > 999.0) {
        std::cerr << "[Transport] BPM out of range: " << bpm << std::endl;
        return;
    }
    timeInfo_.bpm = bpm;
    std::cout << "[Transport] BPM set to " << bpm << std::endl;
    notifyStateChange();
}

void Transport::setTimeSignature(int numerator, int denominator) {
    if (numerator < 1 || numerator > 16 || denominator < 1 || denominator > 16) {
        std::cerr << "[Transport] Invalid time signature" << std::endl;
        return;
    }
    timeInfo_.numerator = numerator;
    timeInfo_.denominator = denominator;
    std::cout << "[Transport] Time signature set to " << numerator << "/" << denominator << std::endl;
    notifyStateChange();
}

void Transport::setPositionSamples(double samples) {
    timeInfo_.samplePosition = samples;
    notifyStateChange();
}

void Transport::setPositionSeconds(double seconds, int sampleRate) {
    timeInfo_.timeInSeconds = seconds;
    timeInfo_.samplePosition = seconds * sampleRate;
    notifyStateChange();
}

void Transport::setPositionBars(int bar, int beat) {
    if (bar < 1 || beat < 1 || beat > timeInfo_.numerator) {
        std::cerr << "[Transport] Invalid bar/beat position" << std::endl;
        return;
    }
    timeInfo_.bar = bar;
    timeInfo_.beat = beat;
    timeInfo_.tick = 0;
    notifyStateChange();
}

void Transport::update(int numSamples, int sampleRate) {
    if (!timeInfo_.isPlaying) {
        return;
    }

    // Update sample position
    timeInfo_.samplePosition += numSamples;
    timeInfo_.timeInSeconds = timeInfo_.samplePosition / static_cast<double>(sampleRate);

    // Calculate bar, beat, tick
    calculateBarBeatTick(sampleRate);
}

void Transport::setStateCallback(StateCallback callback) {
    stateCallback_ = std::move(callback);
}

void Transport::calculateBarBeatTick(int sampleRate) {
    // Calculate musical time from sample position
    double samplesPerBeat = (60.0 / timeInfo_.bpm) * sampleRate;

    double totalBeats = timeInfo_.samplePosition / samplesPerBeat;

    timeInfo_.bar = static_cast<int>(totalBeats / timeInfo_.numerator) + 1;
    timeInfo_.beat = static_cast<int>(std::fmod(totalBeats, timeInfo_.numerator)) + 1;

    // Calculate ticks (960 ppqn - pulses per quarter note)
    double beatFraction = std::fmod(totalBeats, 1.0);
    timeInfo_.tick = static_cast<int>(beatFraction * 960.0);
}

void Transport::notifyStateChange() {
    if (stateCallback_) {
        stateCallback_(timeInfo_);
    }
}

} // namespace Core
} // namespace MolinAntro
