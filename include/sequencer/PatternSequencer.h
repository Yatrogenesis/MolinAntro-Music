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
#include <optional>

namespace MolinAntro {
namespace Sequencer {

// Forward declarations
class Pattern;
class PianoRoll;
class StepSequencer;
class AutomationClip;
class PatternSequencer;

/**
 * @brief Note color groups (FL Studio-style)
 */
enum class NoteColor {
    Default = 0,
    Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink,
    Custom1, Custom2, Custom3, Custom4
};

/**
 * @brief Note slide mode (FL Studio specialty)
 */
enum class SlideMode {
    None,           // Normal note
    Slide,          // Pitch slides to this note
    Porta,          // Portamento to this note
    OctaveUp,       // +12 semitones
    OctaveDown      // -12 semitones
};

/**
 * @brief Piano roll note with full FL Studio features
 */
struct PianoRollNote {
    // Core properties
    int noteNumber = 60;        // MIDI note (0-127)
    double startTick = 0.0;     // Position in ticks (PPQ-based)
    double lengthTicks = 96.0;  // Duration in ticks
    float velocity = 0.78f;     // 0.0-1.0
    float pan = 0.5f;           // 0.0=left, 0.5=center, 1.0=right
    float finePitch = 0.0f;     // -100 to +100 cents

    // FL Studio features
    float release = 0.5f;       // Note release velocity
    float modX = 0.5f;          // Mod X value (0-1)
    float modY = 0.5f;          // Mod Y value (0-1)
    NoteColor color = NoteColor::Default;
    SlideMode slideMode = SlideMode::None;
    int group = 0;              // Note group for selection

    // Filtering
    float cutoff = 1.0f;        // Filter cutoff (0-1)
    float resonance = 0.0f;     // Filter resonance (0-1)

    // Advanced
    bool muted = false;
    bool selected = false;
    int channel = 0;            // For multi-timbral instruments
    std::string name;           // Optional note label
};

/**
 * @brief Automation point with curve options
 */
struct AutomationPoint {
    double tick = 0.0;
    float value = 0.5f;         // 0.0-1.0 normalized

    enum class CurveType {
        Hold,       // Step/hold until next point
        Linear,     // Linear interpolation
        Smooth,     // Smooth curve (sine-based)
        FastAttack, // Exponential attack
        FastDecay,  // Exponential decay
        Single,     // Single curve (FL default)
        Double      // Double S-curve
    } curveType = CurveType::Single;

    float tension = 0.5f;       // Curve tension (0-1)
};

/**
 * @brief Automation clip for parameter control
 */
class AutomationClip {
public:
    AutomationClip(const std::string& name = "Automation");
    ~AutomationClip();

    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    void setLength(double ticks) { lengthTicks_ = ticks; }
    double getLength() const { return lengthTicks_; }

    // Point management
    void addPoint(const AutomationPoint& point);
    void removePoint(int index);
    void movePoint(int index, double tick, float value);
    void setCurve(int index, AutomationPoint::CurveType type, float tension);
    const std::vector<AutomationPoint>& getPoints() const { return points_; }
    void clearPoints() { points_.clear(); }

    // Value interpolation
    float getValueAtTick(double tick) const;

    // LFO mode
    void setLFOEnabled(bool enable) { lfoEnabled_ = enable; }
    void setLFOShape(int shape) { lfoShape_ = shape; } // 0=sine, 1=tri, 2=square, 3=saw
    void setLFOSpeed(double beatsPerCycle) { lfoSpeed_ = beatsPerCycle; }
    void setLFOPhase(float phase) { lfoPhase_ = phase; }
    void setLFOAmount(float amount) { lfoAmount_ = amount; }

    // Parameter link
    void setTargetParameter(const std::string& target) { targetParam_ = target; }
    const std::string& getTargetParameter() const { return targetParam_; }

    // Range
    void setRange(float min, float max) { minValue_ = min; maxValue_ = max; }
    float getMinValue() const { return minValue_; }
    float getMaxValue() const { return maxValue_; }

    // Scaling
    float scaleValue(float normalized) const {
        return minValue_ + normalized * (maxValue_ - minValue_);
    }

private:
    std::string name_;
    double lengthTicks_ = 384.0; // 1 bar at 96 PPQ

    std::vector<AutomationPoint> points_;
    std::string targetParam_;

    float minValue_ = 0.0f;
    float maxValue_ = 1.0f;

    // LFO settings
    bool lfoEnabled_ = false;
    int lfoShape_ = 0;
    double lfoSpeed_ = 1.0;
    float lfoPhase_ = 0.0f;
    float lfoAmount_ = 1.0f;
};

/**
 * @brief Step sequencer for drum patterns
 */
class StepSequencer {
public:
    struct Step {
        bool enabled = false;
        float velocity = 0.78f;
        float pan = 0.5f;
        float pitch = 0.0f;     // Pitch offset in semitones
        float swing = 0.0f;     // Step-specific swing offset
        bool accent = false;
        bool slide = false;
        int probability = 100;  // 0-100% chance of playing
    };

    struct Channel {
        std::string name = "Channel";
        int noteNumber = 36;    // MIDI note for this channel
        std::vector<Step> steps;
        bool muted = false;
        bool soloed = false;
        float volume = 1.0f;
        float pan = 0.5f;
        uint32_t color = 0xFF4444FF;
    };

    StepSequencer(int numSteps = 16);
    ~StepSequencer();

    // Steps
    void setNumSteps(int steps);
    int getNumSteps() const { return numSteps_; }

    // Channels
    int addChannel(const std::string& name, int noteNumber);
    void removeChannel(int index);
    Channel* getChannel(int index);
    int getNumChannels() const { return static_cast<int>(channels_.size()); }

    // Step manipulation
    void setStep(int channel, int step, bool enabled);
    void setStepVelocity(int channel, int step, float velocity);
    void setStepProbability(int channel, int step, int probability);
    Step* getStep(int channel, int step);

    // Swing
    void setGlobalSwing(float amount) { globalSwing_ = std::clamp(amount, 0.0f, 1.0f); }
    float getGlobalSwing() const { return globalSwing_; }

    // Grid size
    void setStepSize(double beats) { stepSizeBeats_ = beats; }
    double getStepSize() const { return stepSizeBeats_; }

    // Pattern variations
    void randomize(int channel, float density, float velocityVariation);
    void rotate(int channel, int steps);  // Rotate pattern left/right
    void mirror(int channel);             // Mirror pattern
    void shift(int channel, int semitones); // Transpose

    // Generate MIDI notes
    std::vector<MIDI::Note> generateNotes(double startBeat, double bpm) const;

private:
    int numSteps_;
    std::vector<Channel> channels_;
    float globalSwing_ = 0.0f;
    double stepSizeBeats_ = 0.25; // 16th notes
};

/**
 * @brief Piano roll for note editing (FL Studio-style)
 */
class PianoRoll {
public:
    PianoRoll();
    ~PianoRoll();

    // Notes
    void addNote(const PianoRollNote& note);
    void removeNote(int index);
    void moveNote(int index, int newNote, double newStartTick);
    void resizeNote(int index, double newLength);
    PianoRollNote* getNote(int index);
    const std::vector<PianoRollNote>& getNotes() const { return notes_; }
    void clearNotes() { notes_.clear(); }
    int getNumNotes() const { return static_cast<int>(notes_.size()); }

    // Selection
    void selectNote(int index, bool addToSelection = false);
    void selectNotesInRange(int lowNote, int highNote, double startTick, double endTick);
    void selectAll();
    void deselectAll();
    std::vector<int> getSelectedNotes() const;

    // Editing operations
    void deleteSelected();
    void duplicateSelected();
    void quantizeSelected(double gridTicks);
    void transposeSelected(int semitones);
    void setVelocitySelected(float velocity);
    void legato();  // Make notes connect
    void staccato(double ratio);  // Shorten notes
    void humanize(float timingAmount, float velocityAmount);

    // Scale/chord helpers
    void snapToScale(int rootNote, const std::vector<int>& scaleIntervals);
    void makeChord(int rootNote, const std::vector<int>& intervals);
    void arpeggiate(double noteLengthTicks, double gapTicks, bool ascending);

    // Ghost notes (show notes from other patterns)
    void setGhostNotes(const std::vector<PianoRollNote>& ghostNotes) { ghostNotes_ = ghostNotes; }
    const std::vector<PianoRollNote>& getGhostNotes() const { return ghostNotes_; }
    void clearGhostNotes() { ghostNotes_.clear(); }

    // Slide notes processing
    void processSlides(std::vector<MIDI::MIDIMessage>& output, double bpm, int sampleRate);

    // View settings
    void setGridSize(double ticks) { gridTicks_ = ticks; }
    double getGridSize() const { return gridTicks_; }
    void setTripletGrid(bool enable) { tripletGrid_ = enable; }
    bool isTripletGrid() const { return tripletGrid_; }

    // PPQ (pulses per quarter note)
    void setPPQ(int ppq) { ppq_ = ppq; }
    int getPPQ() const { return ppq_; }

    // Color grouping
    void setNoteColor(int index, NoteColor color);
    void selectByColor(NoteColor color);
    void setColorForSelected(NoteColor color);

    // Markers
    void addMarker(double tick, const std::string& name);
    void removeMarker(int index);

private:
    std::vector<PianoRollNote> notes_;
    std::vector<PianoRollNote> ghostNotes_;
    std::vector<std::pair<double, std::string>> markers_;

    int ppq_ = 96;           // Pulses per quarter note
    double gridTicks_ = 24;  // 16th notes at 96 PPQ
    bool tripletGrid_ = false;
};

/**
 * @brief Pattern - Container for notes, automation, and step sequences
 */
class Pattern {
public:
    Pattern(const std::string& name = "Pattern");
    ~Pattern();

    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    // Length
    void setLength(double beats) { lengthBeats_ = beats; }
    double getLength() const { return lengthBeats_; }

    // Piano Roll
    PianoRoll& getPianoRoll() { return pianoRoll_; }
    const PianoRoll& getPianoRoll() const { return pianoRoll_; }

    // Step Sequencer
    StepSequencer& getStepSequencer() { return stepSequencer_; }
    const StepSequencer& getStepSequencer() const { return stepSequencer_; }

    // Automation clips
    int addAutomationClip(std::shared_ptr<AutomationClip> clip);
    void removeAutomationClip(int index);
    AutomationClip* getAutomationClip(int index);
    int getNumAutomationClips() const { return static_cast<int>(automationClips_.size()); }

    // Generate all MIDI for playback
    std::vector<MIDI::Note> generateNotes(double bpm) const;
    std::vector<MIDI::MIDIMessage> generateMIDI(double startBeat, double bpm, int sampleRate);

    // Color
    void setColor(uint32_t color) { color_ = color; }
    uint32_t getColor() const { return color_; }

    // Lock
    void setLocked(bool locked) { locked_ = locked; }
    bool isLocked() const { return locked_; }

    // Category/group
    void setCategory(const std::string& category) { category_ = category; }
    const std::string& getCategory() const { return category_; }

private:
    std::string name_;
    double lengthBeats_ = 4.0;

    PianoRoll pianoRoll_;
    StepSequencer stepSequencer_;
    std::vector<std::shared_ptr<AutomationClip>> automationClips_;

    uint32_t color_ = 0xFF5577DD;
    bool locked_ = false;
    std::string category_;
};

/**
 * @brief Playlist block - A pattern instance placed in the playlist
 */
struct PlaylistBlock {
    int patternIndex = 0;
    int trackIndex = 0;
    double startBeat = 0.0;
    double lengthBeats = 4.0;     // Can be different from pattern length
    double clipStartBeat = 0.0;   // Offset into pattern
    bool muted = false;
    float gain = 1.0f;
    uint32_t color = 0;           // 0 = use pattern color
};

/**
 * @brief Playlist track
 */
class PlaylistTrack {
public:
    PlaylistTrack(const std::string& name = "Track");
    ~PlaylistTrack();

    const std::string& getName() const { return name_; }
    void setName(const std::string& name) { name_ = name; }

    void setMute(bool mute) { muted_ = mute; }
    bool isMuted() const { return muted_; }

    void setSolo(bool solo) { soloed_ = solo; }
    bool isSoloed() const { return soloed_; }

    void setHeight(int height) { height_ = height; }
    int getHeight() const { return height_; }

    void setColor(uint32_t color) { color_ = color; }
    uint32_t getColor() const { return color_; }

    // Grouping
    void setGroup(int group) { groupIndex_ = group; }
    int getGroup() const { return groupIndex_; }
    void setGroupCollapsed(bool collapsed) { groupCollapsed_ = collapsed; }

    // Lock
    void setLocked(bool locked) { locked_ = locked; }
    bool isLocked() const { return locked_; }

private:
    std::string name_;
    bool muted_ = false;
    bool soloed_ = false;
    int height_ = 64;
    uint32_t color_ = 0xFF888888;
    int groupIndex_ = -1;
    bool groupCollapsed_ = false;
    bool locked_ = false;
};

/**
 * @brief PatternSequencer - FL Studio-style pattern-based workflow
 */
class PatternSequencer {
public:
    PatternSequencer();
    ~PatternSequencer();

    // Patterns
    int createPattern(const std::string& name = "");
    void deletePattern(int index);
    Pattern* getPattern(int index);
    int getNumPatterns() const { return static_cast<int>(patterns_.size()); }
    void setCurrentPattern(int index) { currentPattern_ = index; }
    int getCurrentPattern() const { return currentPattern_; }
    Pattern* getCurrentPatternPtr();

    // Clone pattern
    int clonePattern(int sourceIndex, const std::string& newName = "");

    // Playlist tracks
    int createPlaylistTrack(const std::string& name = "");
    void deletePlaylistTrack(int index);
    PlaylistTrack* getPlaylistTrack(int index);
    int getNumPlaylistTracks() const { return static_cast<int>(playlistTracks_.size()); }

    // Playlist blocks
    int addBlock(const PlaylistBlock& block);
    void removeBlock(int blockId);
    void moveBlock(int blockId, double newStartBeat, int newTrackIndex);
    void resizeBlock(int blockId, double newLength);
    PlaylistBlock* getBlock(int blockId);
    std::vector<PlaylistBlock*> getBlocksAtBeat(double beat);
    std::vector<PlaylistBlock*> getBlocksInRange(double startBeat, double endBeat);

    // Playback
    void setPlaying(bool playing) { playing_ = playing; }
    bool isPlaying() const { return playing_; }
    void setPosition(double beat) { positionBeat_ = beat; }
    double getPosition() const { return positionBeat_; }
    void setBPM(double bpm) { bpm_ = bpm; }
    double getBPM() const { return bpm_; }

    // Loop
    void setLoopEnabled(bool enable) { loopEnabled_ = enable; }
    bool isLoopEnabled() const { return loopEnabled_; }
    void setLoopRange(double startBeat, double endBeat);
    double getLoopStart() const { return loopStart_; }
    double getLoopEnd() const { return loopEnd_; }

    // Song mode vs Pattern mode
    void setSongMode(bool song) { songMode_ = song; }
    bool isSongMode() const { return songMode_; }

    // Time signature
    void setTimeSignature(int numerator, int denominator);
    int getTimeSignatureNumerator() const { return timeSignatureNum_; }
    int getTimeSignatureDenominator() const { return timeSignatureDenom_; }

    // Song length
    double getSongLength() const;

    // Generate MIDI for playback
    void processMIDI(std::vector<MIDI::MIDIMessage>& output, int numSamples, int sampleRate);

    // Automation
    float getAutomationValue(const std::string& paramId, double beat);

    // Markers
    void addMarker(double beat, const std::string& name, uint32_t color = 0xFF00FF00);
    void removeMarker(int index);
    struct Marker { double beat; std::string name; uint32_t color; };
    const std::vector<Marker>& getMarkers() const { return markers_; }

    // Time markers
    void addTimeMarker(double beat, double newBPM);  // Tempo change
    void addSignatureMarker(double beat, int num, int denom);  // Time sig change

    // Clipboard
    void copySelection();
    void paste(double beat);
    void cut();

private:
    std::vector<std::unique_ptr<Pattern>> patterns_;
    std::vector<std::unique_ptr<PlaylistTrack>> playlistTracks_;
    std::vector<PlaylistBlock> blocks_;
    int nextBlockId_ = 0;

    int currentPattern_ = 0;

    bool playing_ = false;
    double positionBeat_ = 0.0;
    double bpm_ = 140.0;

    bool loopEnabled_ = true;
    double loopStart_ = 0.0;
    double loopEnd_ = 4.0;

    bool songMode_ = false;

    int timeSignatureNum_ = 4;
    int timeSignatureDenom_ = 4;

    std::vector<Marker> markers_;
    std::vector<std::pair<double, double>> tempoChanges_;  // beat -> BPM
    std::vector<std::tuple<double, int, int>> signatureChanges_;  // beat -> num, denom

    std::mutex mutex_;

    // Clipboard
    std::vector<PlaylistBlock> clipboard_;
};

/**
 * @brief Scale helper for piano roll
 */
class ScaleHelper {
public:
    static std::vector<int> getMajorScale() { return {0, 2, 4, 5, 7, 9, 11}; }
    static std::vector<int> getMinorScale() { return {0, 2, 3, 5, 7, 8, 10}; }
    static std::vector<int> getHarmonicMinor() { return {0, 2, 3, 5, 7, 8, 11}; }
    static std::vector<int> getMelodicMinor() { return {0, 2, 3, 5, 7, 9, 11}; }
    static std::vector<int> getDorian() { return {0, 2, 3, 5, 7, 9, 10}; }
    static std::vector<int> getPhrygian() { return {0, 1, 3, 5, 7, 8, 10}; }
    static std::vector<int> getLydian() { return {0, 2, 4, 6, 7, 9, 11}; }
    static std::vector<int> getMixolydian() { return {0, 2, 4, 5, 7, 9, 10}; }
    static std::vector<int> getLocrian() { return {0, 1, 3, 5, 6, 8, 10}; }
    static std::vector<int> getPentatonicMajor() { return {0, 2, 4, 7, 9}; }
    static std::vector<int> getPentatonicMinor() { return {0, 3, 5, 7, 10}; }
    static std::vector<int> getBlues() { return {0, 3, 5, 6, 7, 10}; }
    static std::vector<int> getWholeTone() { return {0, 2, 4, 6, 8, 10}; }
    static std::vector<int> getChromatic() { return {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}; }

    // Snap note to scale
    static int snapToScale(int note, int root, const std::vector<int>& scale);

    // Get scale name
    static std::string getScaleName(const std::vector<int>& scale);
};

/**
 * @brief Chord helper for piano roll
 */
class ChordHelper {
public:
    // Basic triads
    static std::vector<int> getMajorTriad() { return {0, 4, 7}; }
    static std::vector<int> getMinorTriad() { return {0, 3, 7}; }
    static std::vector<int> getDiminished() { return {0, 3, 6}; }
    static std::vector<int> getAugmented() { return {0, 4, 8}; }

    // Seventh chords
    static std::vector<int> getMajor7() { return {0, 4, 7, 11}; }
    static std::vector<int> getMinor7() { return {0, 3, 7, 10}; }
    static std::vector<int> getDominant7() { return {0, 4, 7, 10}; }
    static std::vector<int> getDim7() { return {0, 3, 6, 9}; }
    static std::vector<int> getHalfDim7() { return {0, 3, 6, 10}; }
    static std::vector<int> getMinMaj7() { return {0, 3, 7, 11}; }
    static std::vector<int> getAug7() { return {0, 4, 8, 10}; }

    // Extended chords
    static std::vector<int> get9th() { return {0, 4, 7, 10, 14}; }
    static std::vector<int> getAdd9() { return {0, 4, 7, 14}; }
    static std::vector<int> get11th() { return {0, 4, 7, 10, 14, 17}; }
    static std::vector<int> get13th() { return {0, 4, 7, 10, 14, 17, 21}; }

    // Suspensions
    static std::vector<int> getSus2() { return {0, 2, 7}; }
    static std::vector<int> getSus4() { return {0, 5, 7}; }
    static std::vector<int> get7sus4() { return {0, 5, 7, 10}; }

    // Power chords
    static std::vector<int> getPower() { return {0, 7}; }
    static std::vector<int> getPower8() { return {0, 7, 12}; }

    // Inversions
    static std::vector<int> invert(const std::vector<int>& chord, int times);

    // Voicing
    static std::vector<int> openVoicing(const std::vector<int>& chord);
    static std::vector<int> dropVoicing(const std::vector<int>& chord, int voice);
};

} // namespace Sequencer
} // namespace MolinAntro
