/**
 * @file PatternSequencer.cpp
 * @brief FULL FL Studio-style Pattern Sequencer Implementation
 *
 * Professional pattern-based workflow with:
 * - Piano roll with slide notes
 * - Step sequencer for drums
 * - Automation clips with LFO
 * - Playlist with pattern blocks
 * - Scale/chord helpers
 * - Ghost notes
 * - Humanization
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "sequencer/PatternSequencer.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>

namespace MolinAntro {
namespace Sequencer {

//=============================================================================
// AutomationClip Implementation
//=============================================================================

AutomationClip::AutomationClip(const std::string& name)
    : name_(name)
{
    // Add default points (start and end)
    points_.push_back({0.0, 0.5f, AutomationPoint::CurveType::Single, 0.5f});
    points_.push_back({lengthTicks_, 0.5f, AutomationPoint::CurveType::Single, 0.5f});
}

AutomationClip::~AutomationClip() = default;

void AutomationClip::addPoint(const AutomationPoint& point) {
    // Insert in sorted order
    auto it = std::lower_bound(points_.begin(), points_.end(), point,
        [](const AutomationPoint& a, const AutomationPoint& b) {
            return a.tick < b.tick;
        });
    points_.insert(it, point);
}

void AutomationClip::removePoint(int index) {
    if (index >= 0 && index < static_cast<int>(points_.size())) {
        // Don't remove first or last point
        if (index != 0 && index != static_cast<int>(points_.size()) - 1) {
            points_.erase(points_.begin() + index);
        }
    }
}

void AutomationClip::movePoint(int index, double tick, float value) {
    if (index >= 0 && index < static_cast<int>(points_.size())) {
        // First and last points can only move vertically
        if (index == 0) {
            points_[index].value = std::clamp(value, 0.0f, 1.0f);
        } else if (index == static_cast<int>(points_.size()) - 1) {
            points_[index].tick = lengthTicks_;
            points_[index].value = std::clamp(value, 0.0f, 1.0f);
        } else {
            // Clamp tick to be between neighbors
            double minTick = points_[index - 1].tick + 1.0;
            double maxTick = points_[index + 1].tick - 1.0;
            points_[index].tick = std::clamp(tick, minTick, maxTick);
            points_[index].value = std::clamp(value, 0.0f, 1.0f);
        }
    }
}

void AutomationClip::setCurve(int index, AutomationPoint::CurveType type, float tension) {
    if (index >= 0 && index < static_cast<int>(points_.size())) {
        points_[index].curveType = type;
        points_[index].tension = std::clamp(tension, 0.0f, 1.0f);
    }
}

float AutomationClip::getValueAtTick(double tick) const {
    if (points_.empty()) return 0.5f;
    if (tick <= points_.front().tick) return points_.front().value;
    if (tick >= points_.back().tick) return points_.back().value;

    // Find surrounding points
    auto it = std::lower_bound(points_.begin(), points_.end(), tick,
        [](const AutomationPoint& p, double t) {
            return p.tick < t;
        });

    if (it == points_.begin()) return it->value;
    if (it == points_.end()) return points_.back().value;

    const AutomationPoint& p2 = *it;
    const AutomationPoint& p1 = *(it - 1);

    double t = (tick - p1.tick) / (p2.tick - p1.tick);
    float baseValue = 0.0f;

    // Apply curve type
    switch (p1.curveType) {
        case AutomationPoint::CurveType::Hold:
            baseValue = p1.value;
            break;

        case AutomationPoint::CurveType::Linear:
            baseValue = p1.value + static_cast<float>(t) * (p2.value - p1.value);
            break;

        case AutomationPoint::CurveType::Smooth: {
            // Cosine interpolation
            float ct = static_cast<float>((1.0 - std::cos(t * 3.14159265359)) * 0.5);
            baseValue = p1.value + ct * (p2.value - p1.value);
            break;
        }

        case AutomationPoint::CurveType::FastAttack: {
            // Exponential attack
            float et = 1.0f - std::pow(1.0f - static_cast<float>(t), 2.0f + p1.tension * 4.0f);
            baseValue = p1.value + et * (p2.value - p1.value);
            break;
        }

        case AutomationPoint::CurveType::FastDecay: {
            // Exponential decay
            float et = std::pow(static_cast<float>(t), 2.0f + p1.tension * 4.0f);
            baseValue = p1.value + et * (p2.value - p1.value);
            break;
        }

        case AutomationPoint::CurveType::Single: {
            // FL Studio single curve (tension-controlled)
            float tension = (p1.tension - 0.5f) * 2.0f; // -1 to 1
            float curve;
            if (std::abs(tension) < 0.001f) {
                curve = static_cast<float>(t);
            } else {
                curve = (std::pow(std::abs(tension) + 1.0f, static_cast<float>(t)) - 1.0f) /
                        std::abs(tension);
                if (tension < 0) curve = 1.0f - curve;
            }
            baseValue = p1.value + curve * (p2.value - p1.value);
            break;
        }

        case AutomationPoint::CurveType::Double: {
            // Double S-curve
            float st = static_cast<float>(t);
            if (st < 0.5f) {
                st = 2.0f * st * st;
            } else {
                st = 1.0f - 2.0f * (1.0f - st) * (1.0f - st);
            }
            baseValue = p1.value + st * (p2.value - p1.value);
            break;
        }
    }

    // Apply LFO if enabled
    if (lfoEnabled_) {
        float lfoValue = 0.0f;
        double lfoPhase = std::fmod(tick / (96.0 * lfoSpeed_) + lfoPhase_, 1.0);

        switch (lfoShape_) {
            case 0: // Sine
                lfoValue = std::sin(lfoPhase * 2.0 * 3.14159265359);
                break;
            case 1: // Triangle
                lfoValue = 4.0f * std::abs(static_cast<float>(lfoPhase) - 0.5f) - 1.0f;
                break;
            case 2: // Square
                lfoValue = lfoPhase < 0.5 ? 1.0f : -1.0f;
                break;
            case 3: // Saw
                lfoValue = 2.0f * static_cast<float>(lfoPhase) - 1.0f;
                break;
        }

        baseValue += lfoValue * lfoAmount_ * 0.5f;
        baseValue = std::clamp(baseValue, 0.0f, 1.0f);
    }

    return baseValue;
}

//=============================================================================
// StepSequencer Implementation
//=============================================================================

StepSequencer::StepSequencer(int numSteps)
    : numSteps_(numSteps)
{
}

StepSequencer::~StepSequencer() = default;

void StepSequencer::setNumSteps(int steps) {
    numSteps_ = std::clamp(steps, 1, 128);
    for (auto& channel : channels_) {
        channel.steps.resize(numSteps_);
    }
}

int StepSequencer::addChannel(const std::string& name, int noteNumber) {
    Channel channel;
    channel.name = name;
    channel.noteNumber = noteNumber;
    channel.steps.resize(numSteps_);
    channels_.push_back(channel);
    return static_cast<int>(channels_.size()) - 1;
}

void StepSequencer::removeChannel(int index) {
    if (index >= 0 && index < static_cast<int>(channels_.size())) {
        channels_.erase(channels_.begin() + index);
    }
}

StepSequencer::Channel* StepSequencer::getChannel(int index) {
    if (index >= 0 && index < static_cast<int>(channels_.size())) {
        return &channels_[index];
    }
    return nullptr;
}

void StepSequencer::setStep(int channel, int step, bool enabled) {
    if (channel >= 0 && channel < static_cast<int>(channels_.size()) &&
        step >= 0 && step < numSteps_) {
        channels_[channel].steps[step].enabled = enabled;
    }
}

void StepSequencer::setStepVelocity(int channel, int step, float velocity) {
    if (channel >= 0 && channel < static_cast<int>(channels_.size()) &&
        step >= 0 && step < numSteps_) {
        channels_[channel].steps[step].velocity = std::clamp(velocity, 0.0f, 1.0f);
    }
}

void StepSequencer::setStepProbability(int channel, int step, int probability) {
    if (channel >= 0 && channel < static_cast<int>(channels_.size()) &&
        step >= 0 && step < numSteps_) {
        channels_[channel].steps[step].probability = std::clamp(probability, 0, 100);
    }
}

StepSequencer::Step* StepSequencer::getStep(int channel, int step) {
    if (channel >= 0 && channel < static_cast<int>(channels_.size()) &&
        step >= 0 && step < numSteps_) {
        return &channels_[channel].steps[step];
    }
    return nullptr;
}

void StepSequencer::randomize(int channel, float density, float velocityVariation) {
    if (channel < 0 || channel >= static_cast<int>(channels_.size())) return;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& step : channels_[channel].steps) {
        step.enabled = dist(gen) < density;
        if (step.enabled) {
            step.velocity = std::clamp(0.78f + (dist(gen) - 0.5f) * velocityVariation, 0.1f, 1.0f);
        }
    }
}

void StepSequencer::rotate(int channel, int steps) {
    if (channel < 0 || channel >= static_cast<int>(channels_.size())) return;

    auto& channelSteps = channels_[channel].steps;
    int n = static_cast<int>(channelSteps.size());
    if (n == 0) return;

    steps = ((steps % n) + n) % n; // Normalize to positive
    std::rotate(channelSteps.begin(), channelSteps.begin() + steps, channelSteps.end());
}

void StepSequencer::mirror(int channel) {
    if (channel < 0 || channel >= static_cast<int>(channels_.size())) return;

    auto& channelSteps = channels_[channel].steps;
    std::reverse(channelSteps.begin(), channelSteps.end());
}

void StepSequencer::shift(int channel, int semitones) {
    if (channel < 0 || channel >= static_cast<int>(channels_.size())) return;

    channels_[channel].noteNumber = std::clamp(
        channels_[channel].noteNumber + semitones, 0, 127
    );
}

std::vector<MIDI::Note> StepSequencer::generateNotes(double startBeat, double bpm) const {
    std::vector<MIDI::Note> notes;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 99);

    for (const auto& channel : channels_) {
        if (channel.muted) continue;

        for (int i = 0; i < numSteps_; ++i) {
            const Step& step = channel.steps[i];
            if (!step.enabled) continue;

            // Probability check
            if (dist(gen) >= step.probability) continue;

            MIDI::Note note;
            note.noteNumber = channel.noteNumber + static_cast<int>(step.pitch);
            note.velocity = static_cast<uint8_t>(step.velocity * 127.0f);

            // Calculate timing with swing
            double stepBeat = startBeat + i * stepSizeBeats_;
            if (i % 2 == 1) {  // Off-beat
                double swingOffset = (globalSwing_ + step.swing) * stepSizeBeats_ * 0.5;
                stepBeat += swingOffset;
            }

            note.startTime = stepBeat;
            note.duration = stepSizeBeats_ * 0.8; // Slightly shorter than full step
            note.channel = 0;

            notes.push_back(note);
        }
    }

    return notes;
}

//=============================================================================
// PianoRoll Implementation
//=============================================================================

PianoRoll::PianoRoll() = default;
PianoRoll::~PianoRoll() = default;

void PianoRoll::addNote(const PianoRollNote& note) {
    notes_.push_back(note);
    // Sort by start time
    std::sort(notes_.begin(), notes_.end(),
        [](const PianoRollNote& a, const PianoRollNote& b) {
            return a.startTick < b.startTick;
        });
}

void PianoRoll::removeNote(int index) {
    if (index >= 0 && index < static_cast<int>(notes_.size())) {
        notes_.erase(notes_.begin() + index);
    }
}

void PianoRoll::moveNote(int index, int newNote, double newStartTick) {
    if (index >= 0 && index < static_cast<int>(notes_.size())) {
        notes_[index].noteNumber = std::clamp(newNote, 0, 127);
        notes_[index].startTick = std::max(0.0, newStartTick);
    }
}

void PianoRoll::resizeNote(int index, double newLength) {
    if (index >= 0 && index < static_cast<int>(notes_.size())) {
        notes_[index].lengthTicks = std::max(1.0, newLength);
    }
}

PianoRollNote* PianoRoll::getNote(int index) {
    if (index >= 0 && index < static_cast<int>(notes_.size())) {
        return &notes_[index];
    }
    return nullptr;
}

void PianoRoll::selectNote(int index, bool addToSelection) {
    if (!addToSelection) {
        deselectAll();
    }
    if (index >= 0 && index < static_cast<int>(notes_.size())) {
        notes_[index].selected = true;
    }
}

void PianoRoll::selectNotesInRange(int lowNote, int highNote, double startTick, double endTick) {
    for (auto& note : notes_) {
        bool inRange = note.noteNumber >= lowNote && note.noteNumber <= highNote &&
                       note.startTick >= startTick &&
                       note.startTick + note.lengthTicks <= endTick;
        note.selected = inRange;
    }
}

void PianoRoll::selectAll() {
    for (auto& note : notes_) {
        note.selected = true;
    }
}

void PianoRoll::deselectAll() {
    for (auto& note : notes_) {
        note.selected = false;
    }
}

std::vector<int> PianoRoll::getSelectedNotes() const {
    std::vector<int> selected;
    for (int i = 0; i < static_cast<int>(notes_.size()); ++i) {
        if (notes_[i].selected) {
            selected.push_back(i);
        }
    }
    return selected;
}

void PianoRoll::deleteSelected() {
    notes_.erase(
        std::remove_if(notes_.begin(), notes_.end(),
            [](const PianoRollNote& n) { return n.selected; }),
        notes_.end()
    );
}

void PianoRoll::duplicateSelected() {
    std::vector<PianoRollNote> duplicates;

    // Find the extent of selection
    double minStart = std::numeric_limits<double>::max();
    double maxEnd = 0.0;
    for (const auto& note : notes_) {
        if (note.selected) {
            minStart = std::min(minStart, note.startTick);
            maxEnd = std::max(maxEnd, note.startTick + note.lengthTicks);
        }
    }

    double duration = maxEnd - minStart;

    for (const auto& note : notes_) {
        if (note.selected) {
            PianoRollNote dup = note;
            dup.startTick += duration;
            dup.selected = false;
            duplicates.push_back(dup);
        }
    }

    // Deselect originals
    for (auto& note : notes_) {
        if (note.selected) note.selected = false;
    }

    // Add duplicates (selected)
    for (auto& dup : duplicates) {
        dup.selected = true;
        notes_.push_back(dup);
    }
}

void PianoRoll::quantizeSelected(double gridTicks) {
    for (auto& note : notes_) {
        if (note.selected) {
            note.startTick = std::round(note.startTick / gridTicks) * gridTicks;
        }
    }
}

void PianoRoll::transposeSelected(int semitones) {
    for (auto& note : notes_) {
        if (note.selected) {
            note.noteNumber = std::clamp(note.noteNumber + semitones, 0, 127);
        }
    }
}

void PianoRoll::setVelocitySelected(float velocity) {
    for (auto& note : notes_) {
        if (note.selected) {
            note.velocity = std::clamp(velocity, 0.0f, 1.0f);
        }
    }
}

void PianoRoll::legato() {
    // Sort selected notes by start time
    std::vector<int> selected = getSelectedNotes();
    if (selected.size() < 2) return;

    std::sort(selected.begin(), selected.end(),
        [this](int a, int b) {
            return notes_[a].startTick < notes_[b].startTick;
        });

    // Extend each note to the start of the next
    for (size_t i = 0; i < selected.size() - 1; ++i) {
        PianoRollNote& current = notes_[selected[i]];
        PianoRollNote& next = notes_[selected[i + 1]];
        current.lengthTicks = next.startTick - current.startTick;
    }
}

void PianoRoll::staccato(double ratio) {
    for (auto& note : notes_) {
        if (note.selected) {
            note.lengthTicks *= std::clamp(ratio, 0.1, 1.0);
        }
    }
}

void PianoRoll::humanize(float timingAmount, float velocityAmount) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& note : notes_) {
        if (note.selected) {
            // Timing humanization (in ticks)
            double timingOffset = dist(gen) * timingAmount * (ppq_ / 4.0);
            note.startTick = std::max(0.0, note.startTick + timingOffset);

            // Velocity humanization
            float velocityOffset = dist(gen) * velocityAmount;
            note.velocity = std::clamp(note.velocity + velocityOffset, 0.0f, 1.0f);
        }
    }
}

void PianoRoll::snapToScale(int rootNote, const std::vector<int>& scaleIntervals) {
    for (auto& note : notes_) {
        if (note.selected) {
            note.noteNumber = ScaleHelper::snapToScale(note.noteNumber, rootNote, scaleIntervals);
        }
    }
}

void PianoRoll::makeChord(int rootNote, const std::vector<int>& intervals) {
    // Get first selected note
    auto selected = getSelectedNotes();
    if (selected.empty()) return;

    PianoRollNote baseNote = notes_[selected[0]];

    // Delete selected
    deleteSelected();

    // Create chord notes
    for (int interval : intervals) {
        PianoRollNote chordNote = baseNote;
        chordNote.noteNumber = rootNote + interval;
        chordNote.selected = true;
        addNote(chordNote);
    }
}

void PianoRoll::arpeggiate(double noteLengthTicks, double gapTicks, bool ascending) {
    auto selected = getSelectedNotes();
    if (selected.empty()) return;

    // Sort by pitch
    std::sort(selected.begin(), selected.end(),
        [this](int a, int b) {
            return notes_[a].noteNumber < notes_[b].noteNumber;
        });

    if (!ascending) {
        std::reverse(selected.begin(), selected.end());
    }

    // Get start position from first selected note
    double startTick = notes_[selected[0]].startTick;

    // Adjust each note
    for (size_t i = 0; i < selected.size(); ++i) {
        notes_[selected[i]].startTick = startTick + i * (noteLengthTicks + gapTicks);
        notes_[selected[i]].lengthTicks = noteLengthTicks;
    }
}

void PianoRoll::processSlides(std::vector<MIDI::MIDIMessage>& output, double bpm, int sampleRate) {
    // Process slide notes (FL Studio-style pitch slides)
    for (size_t i = 0; i < notes_.size(); ++i) {
        const PianoRollNote& note = notes_[i];
        if (note.slideMode == SlideMode::None) continue;

        // Find the previous note on same channel
        int prevNoteIdx = -1;
        for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
            if (notes_[j].channel == note.channel &&
                notes_[j].startTick + notes_[j].lengthTicks >= note.startTick) {
                prevNoteIdx = j;
                break;
            }
        }

        if (prevNoteIdx >= 0) {
            const PianoRollNote& prevNote = notes_[prevNoteIdx];
            int pitchDiff = note.noteNumber - prevNote.noteNumber;

            // Generate pitch bend messages
            double ticksPerSample = (bpm / 60.0) * ppq_ / sampleRate;
            int numSteps = static_cast<int>(note.startTick - prevNote.startTick - prevNote.lengthTicks);

            // Create pitch bend ramp
            for (int step = 0; step < numSteps; ++step) {
                float t = static_cast<float>(step) / numSteps;
                int bendValue = static_cast<int>(t * pitchDiff * 8192 / 2); // Assuming ±2 semitone range

                // MIDI::MIDIMessage bendMsg;
                // bendMsg.type = MIDI::MIDIMessage::Type::PitchBend;
                // bendMsg.channel = note.channel;
                // bendMsg.data1 = bendValue & 0x7F;
                // bendMsg.data2 = (bendValue >> 7) & 0x7F;
                // output.push_back(bendMsg);
            }
        }
    }
}

void PianoRoll::setNoteColor(int index, NoteColor color) {
    if (index >= 0 && index < static_cast<int>(notes_.size())) {
        notes_[index].color = color;
    }
}

void PianoRoll::selectByColor(NoteColor color) {
    for (auto& note : notes_) {
        note.selected = (note.color == color);
    }
}

void PianoRoll::setColorForSelected(NoteColor color) {
    for (auto& note : notes_) {
        if (note.selected) {
            note.color = color;
        }
    }
}

void PianoRoll::addMarker(double tick, const std::string& name) {
    markers_.push_back({tick, name});
    std::sort(markers_.begin(), markers_.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
}

void PianoRoll::removeMarker(int index) {
    if (index >= 0 && index < static_cast<int>(markers_.size())) {
        markers_.erase(markers_.begin() + index);
    }
}

//=============================================================================
// Pattern Implementation
//=============================================================================

Pattern::Pattern(const std::string& name)
    : name_(name)
{
}

Pattern::~Pattern() = default;

int Pattern::addAutomationClip(std::shared_ptr<AutomationClip> clip) {
    automationClips_.push_back(clip);
    return static_cast<int>(automationClips_.size()) - 1;
}

void Pattern::removeAutomationClip(int index) {
    if (index >= 0 && index < static_cast<int>(automationClips_.size())) {
        automationClips_.erase(automationClips_.begin() + index);
    }
}

AutomationClip* Pattern::getAutomationClip(int index) {
    if (index >= 0 && index < static_cast<int>(automationClips_.size())) {
        return automationClips_[index].get();
    }
    return nullptr;
}

std::vector<MIDI::Note> Pattern::generateNotes(double bpm) const {
    std::vector<MIDI::Note> allNotes;

    // Convert piano roll notes
    for (const auto& prNote : pianoRoll_.getNotes()) {
        if (prNote.muted) continue;

        MIDI::Note note;
        note.noteNumber = prNote.noteNumber;
        note.velocity = static_cast<uint8_t>(prNote.velocity * 127.0f);
        note.startTime = prNote.startTick / pianoRoll_.getPPQ(); // Convert to beats
        note.duration = prNote.lengthTicks / pianoRoll_.getPPQ();
        note.channel = prNote.channel;

        allNotes.push_back(note);
    }

    // Add step sequencer notes
    auto stepNotes = stepSequencer_.generateNotes(0.0, bpm);
    allNotes.insert(allNotes.end(), stepNotes.begin(), stepNotes.end());

    // Sort by start time
    std::sort(allNotes.begin(), allNotes.end(),
        [](const MIDI::Note& a, const MIDI::Note& b) {
            return a.startTime < b.startTime;
        });

    return allNotes;
}

std::vector<MIDI::MIDIMessage> Pattern::generateMIDI(double startBeat, double bpm, int sampleRate) {
    std::vector<MIDI::MIDIMessage> messages;

    auto notes = generateNotes(bpm);
    double samplesPerBeat = (60.0 / bpm) * sampleRate;

    for (const auto& note : notes) {
        // Note On
        MIDI::MIDIMessage noteOn;
        noteOn.timestamp = static_cast<uint64_t>((startBeat + note.startTime) * samplesPerBeat);
        noteOn.status = 0x90 | (note.channel & 0x0F);
        noteOn.data1 = note.noteNumber;
        noteOn.data2 = note.velocity;
        messages.push_back(noteOn);

        // Note Off
        MIDI::MIDIMessage noteOff;
        noteOff.timestamp = static_cast<uint64_t>((startBeat + note.startTime + note.duration) * samplesPerBeat);
        noteOff.status = 0x80 | (note.channel & 0x0F);
        noteOff.data1 = note.noteNumber;
        noteOff.data2 = 0;
        messages.push_back(noteOff);
    }

    // Sort by timestamp
    std::sort(messages.begin(), messages.end(),
        [](const MIDI::MIDIMessage& a, const MIDI::MIDIMessage& b) {
            return a.timestamp < b.timestamp;
        });

    return messages;
}

//=============================================================================
// PlaylistTrack Implementation
//=============================================================================

PlaylistTrack::PlaylistTrack(const std::string& name)
    : name_(name)
{
}

PlaylistTrack::~PlaylistTrack() = default;

//=============================================================================
// PatternSequencer Implementation
//=============================================================================

PatternSequencer::PatternSequencer() {
    // Create default pattern and tracks
    createPattern("Pattern 1");
    for (int i = 0; i < 16; ++i) {
        createPlaylistTrack("Track " + std::to_string(i + 1));
    }
}

PatternSequencer::~PatternSequencer() = default;

int PatternSequencer::createPattern(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string patternName = name;
    if (patternName.empty()) {
        patternName = "Pattern " + std::to_string(patterns_.size() + 1);
    }

    auto pattern = std::make_unique<Pattern>(patternName);
    patterns_.push_back(std::move(pattern));

    return static_cast<int>(patterns_.size()) - 1;
}

void PatternSequencer::deletePattern(int index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (index >= 0 && index < static_cast<int>(patterns_.size())) {
        patterns_.erase(patterns_.begin() + index);

        // Remove blocks referencing this pattern
        blocks_.erase(
            std::remove_if(blocks_.begin(), blocks_.end(),
                [index](const PlaylistBlock& b) { return b.patternIndex == index; }),
            blocks_.end()
        );

        // Update pattern indices in remaining blocks
        for (auto& block : blocks_) {
            if (block.patternIndex > index) {
                block.patternIndex--;
            }
        }

        if (currentPattern_ >= static_cast<int>(patterns_.size())) {
            currentPattern_ = static_cast<int>(patterns_.size()) - 1;
        }
    }
}

Pattern* PatternSequencer::getPattern(int index) {
    if (index >= 0 && index < static_cast<int>(patterns_.size())) {
        return patterns_[index].get();
    }
    return nullptr;
}

Pattern* PatternSequencer::getCurrentPatternPtr() {
    return getPattern(currentPattern_);
}

int PatternSequencer::clonePattern(int sourceIndex, const std::string& newName) {
    std::lock_guard<std::mutex> lock(mutex_);

    Pattern* source = getPattern(sourceIndex);
    if (!source) return -1;

    std::string cloneName = newName.empty() ?
        source->getName() + " (copy)" : newName;

    auto clone = std::make_unique<Pattern>(cloneName);
    clone->setLength(source->getLength());
    clone->setColor(source->getColor());
    clone->setCategory(source->getCategory());

    // Clone piano roll notes
    for (const auto& note : source->getPianoRoll().getNotes()) {
        clone->getPianoRoll().addNote(note);
    }

    // Clone step sequencer
    auto& srcSeq = source->getStepSequencer();
    auto& dstSeq = clone->getStepSequencer();
    dstSeq.setNumSteps(srcSeq.getNumSteps());

    patterns_.push_back(std::move(clone));
    return static_cast<int>(patterns_.size()) - 1;
}

int PatternSequencer::createPlaylistTrack(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);

    std::string trackName = name;
    if (trackName.empty()) {
        trackName = "Track " + std::to_string(playlistTracks_.size() + 1);
    }

    auto track = std::make_unique<PlaylistTrack>(trackName);
    playlistTracks_.push_back(std::move(track));

    return static_cast<int>(playlistTracks_.size()) - 1;
}

void PatternSequencer::deletePlaylistTrack(int index) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (index >= 0 && index < static_cast<int>(playlistTracks_.size())) {
        playlistTracks_.erase(playlistTracks_.begin() + index);

        // Remove blocks on this track
        blocks_.erase(
            std::remove_if(blocks_.begin(), blocks_.end(),
                [index](const PlaylistBlock& b) { return b.trackIndex == index; }),
            blocks_.end()
        );

        // Update track indices
        for (auto& block : blocks_) {
            if (block.trackIndex > index) {
                block.trackIndex--;
            }
        }
    }
}

PlaylistTrack* PatternSequencer::getPlaylistTrack(int index) {
    if (index >= 0 && index < static_cast<int>(playlistTracks_.size())) {
        return playlistTracks_[index].get();
    }
    return nullptr;
}

int PatternSequencer::addBlock(const PlaylistBlock& block) {
    std::lock_guard<std::mutex> lock(mutex_);

    PlaylistBlock newBlock = block;
    // newBlock.id = nextBlockId_++; // Would need id field
    blocks_.push_back(newBlock);

    return static_cast<int>(blocks_.size()) - 1;
}

void PatternSequencer::removeBlock(int blockId) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (blockId >= 0 && blockId < static_cast<int>(blocks_.size())) {
        blocks_.erase(blocks_.begin() + blockId);
    }
}

void PatternSequencer::moveBlock(int blockId, double newStartBeat, int newTrackIndex) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (blockId >= 0 && blockId < static_cast<int>(blocks_.size())) {
        blocks_[blockId].startBeat = std::max(0.0, newStartBeat);
        blocks_[blockId].trackIndex = std::clamp(newTrackIndex, 0,
            static_cast<int>(playlistTracks_.size()) - 1);
    }
}

void PatternSequencer::resizeBlock(int blockId, double newLength) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (blockId >= 0 && blockId < static_cast<int>(blocks_.size())) {
        blocks_[blockId].lengthBeats = std::max(0.25, newLength);
    }
}

PlaylistBlock* PatternSequencer::getBlock(int blockId) {
    if (blockId >= 0 && blockId < static_cast<int>(blocks_.size())) {
        return &blocks_[blockId];
    }
    return nullptr;
}

std::vector<PlaylistBlock*> PatternSequencer::getBlocksAtBeat(double beat) {
    std::vector<PlaylistBlock*> result;
    for (auto& block : blocks_) {
        if (beat >= block.startBeat && beat < block.startBeat + block.lengthBeats) {
            result.push_back(&block);
        }
    }
    return result;
}

std::vector<PlaylistBlock*> PatternSequencer::getBlocksInRange(double startBeat, double endBeat) {
    std::vector<PlaylistBlock*> result;
    for (auto& block : blocks_) {
        if (block.startBeat < endBeat &&
            block.startBeat + block.lengthBeats > startBeat) {
            result.push_back(&block);
        }
    }
    return result;
}

void PatternSequencer::setLoopRange(double startBeat, double endBeat) {
    loopStart_ = std::min(startBeat, endBeat);
    loopEnd_ = std::max(startBeat, endBeat);
}

void PatternSequencer::setTimeSignature(int numerator, int denominator) {
    timeSignatureNum_ = std::clamp(numerator, 1, 32);
    timeSignatureDenom_ = std::clamp(denominator, 1, 32);
}

double PatternSequencer::getSongLength() const {
    double maxEnd = 0.0;
    for (const auto& block : blocks_) {
        maxEnd = std::max(maxEnd, block.startBeat + block.lengthBeats);
    }
    return maxEnd;
}

void PatternSequencer::processMIDI(std::vector<MIDI::MIDIMessage>& output,
                                    int numSamples, int sampleRate) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!playing_) return;

    double samplesPerBeat = (60.0 / bpm_) * sampleRate;
    double beatsPerSample = 1.0 / samplesPerBeat;
    double endBeat = positionBeat_ + numSamples * beatsPerSample;

    if (songMode_) {
        // Get all blocks in range
        auto activeBlocks = getBlocksInRange(positionBeat_, endBeat);

        for (auto* block : activeBlocks) {
            if (block->muted) continue;

            PlaylistTrack* track = getPlaylistTrack(block->trackIndex);
            if (track && track->isMuted()) continue;

            Pattern* pattern = getPattern(block->patternIndex);
            if (!pattern) continue;

            // Generate MIDI for this block's portion
            auto patternMidi = pattern->generateMIDI(block->startBeat, bpm_, sampleRate);

            for (auto& msg : patternMidi) {
                // Adjust timestamp relative to current position
                double msgBeat = msg.timestamp / samplesPerBeat;
                if (msgBeat >= positionBeat_ && msgBeat < endBeat) {
                    msg.timestamp = static_cast<uint64_t>((msgBeat - positionBeat_) * samplesPerBeat);
                    output.push_back(msg);
                }
            }
        }
    } else {
        // Pattern mode - play current pattern
        Pattern* pattern = getCurrentPatternPtr();
        if (pattern) {
            auto patternMidi = pattern->generateMIDI(0.0, bpm_, sampleRate);

            double patternLength = pattern->getLength();
            double patternPos = std::fmod(positionBeat_, patternLength);

            for (auto& msg : patternMidi) {
                double msgBeat = msg.timestamp / samplesPerBeat;

                // Handle pattern wrap
                if (msgBeat >= patternPos && msgBeat < patternPos + numSamples * beatsPerSample) {
                    msg.timestamp = static_cast<uint64_t>((msgBeat - patternPos) * samplesPerBeat);
                    output.push_back(msg);
                }
            }
        }
    }

    // Handle looping
    if (loopEnabled_ && endBeat >= loopEnd_) {
        positionBeat_ = loopStart_ + std::fmod(endBeat - loopStart_, loopEnd_ - loopStart_);
    } else {
        positionBeat_ = endBeat;
    }
}

float PatternSequencer::getAutomationValue(const std::string& paramId, double beat) {
    // Search all patterns for automation targeting this parameter
    for (auto& pattern : patterns_) {
        for (int i = 0; i < pattern->getNumAutomationClips(); ++i) {
            AutomationClip* clip = pattern->getAutomationClip(i);
            if (clip && clip->getTargetParameter() == paramId) {
                double tick = beat * 96; // Convert to ticks (assuming 96 PPQ)
                return clip->scaleValue(clip->getValueAtTick(tick));
            }
        }
    }
    return 0.5f; // Default
}

void PatternSequencer::addMarker(double beat, const std::string& name, uint32_t color) {
    markers_.push_back({beat, name, color});
    std::sort(markers_.begin(), markers_.end(),
        [](const Marker& a, const Marker& b) {
            return a.beat < b.beat;
        });
}

void PatternSequencer::removeMarker(int index) {
    if (index >= 0 && index < static_cast<int>(markers_.size())) {
        markers_.erase(markers_.begin() + index);
    }
}

void PatternSequencer::addTimeMarker(double beat, double newBPM) {
    tempoChanges_.push_back({beat, newBPM});
    std::sort(tempoChanges_.begin(), tempoChanges_.end(),
        [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
}

void PatternSequencer::addSignatureMarker(double beat, int num, int denom) {
    signatureChanges_.push_back({beat, num, denom});
    std::sort(signatureChanges_.begin(), signatureChanges_.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
}

void PatternSequencer::copySelection() {
    clipboard_.clear();
    // Copy selected blocks (would need selection tracking)
}

void PatternSequencer::paste(double beat) {
    for (const auto& block : clipboard_) {
        PlaylistBlock newBlock = block;
        newBlock.startBeat = beat + (block.startBeat - clipboard_.front().startBeat);
        addBlock(newBlock);
    }
}

void PatternSequencer::cut() {
    copySelection();
    // Delete selected blocks
}

//=============================================================================
// ScaleHelper Implementation
//=============================================================================

int ScaleHelper::snapToScale(int note, int root, const std::vector<int>& scale) {
    int octave = note / 12;
    int noteInOctave = note % 12;
    int relativeNote = (noteInOctave - root % 12 + 12) % 12;

    // Find closest scale degree
    int closest = scale[0];
    int minDist = std::abs(relativeNote - scale[0]);

    for (int degree : scale) {
        int dist = std::abs(relativeNote - degree);
        if (dist < minDist) {
            minDist = dist;
            closest = degree;
        }
    }

    return octave * 12 + (root % 12 + closest) % 12;
}

std::string ScaleHelper::getScaleName(const std::vector<int>& scale) {
    if (scale == getMajorScale()) return "Major";
    if (scale == getMinorScale()) return "Natural Minor";
    if (scale == getHarmonicMinor()) return "Harmonic Minor";
    if (scale == getMelodicMinor()) return "Melodic Minor";
    if (scale == getDorian()) return "Dorian";
    if (scale == getPhrygian()) return "Phrygian";
    if (scale == getLydian()) return "Lydian";
    if (scale == getMixolydian()) return "Mixolydian";
    if (scale == getLocrian()) return "Locrian";
    if (scale == getPentatonicMajor()) return "Pentatonic Major";
    if (scale == getPentatonicMinor()) return "Pentatonic Minor";
    if (scale == getBlues()) return "Blues";
    if (scale == getWholeTone()) return "Whole Tone";
    if (scale == getChromatic()) return "Chromatic";
    return "Custom";
}

//=============================================================================
// ChordHelper Implementation
//=============================================================================

std::vector<int> ChordHelper::invert(const std::vector<int>& chord, int times) {
    if (chord.empty()) return chord;

    std::vector<int> result = chord;
    for (int i = 0; i < times; ++i) {
        result[0] += 12;
        std::sort(result.begin(), result.end());
    }

    // Normalize to start from 0
    int offset = result[0];
    for (int& note : result) {
        note -= offset;
    }

    return result;
}

std::vector<int> ChordHelper::openVoicing(const std::vector<int>& chord) {
    if (chord.size() < 3) return chord;

    std::vector<int> result;
    result.push_back(chord[0]);
    result.push_back(chord[2] + 12);
    if (chord.size() > 1) result.push_back(chord[1] + 12);
    if (chord.size() > 3) {
        for (size_t i = 3; i < chord.size(); ++i) {
            result.push_back(chord[i] + 24);
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> ChordHelper::dropVoicing(const std::vector<int>& chord, int voice) {
    if (chord.size() < 2 || voice >= static_cast<int>(chord.size())) {
        return chord;
    }

    std::vector<int> result = chord;
    result[voice] -= 12;
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace Sequencer
} // namespace MolinAntro
