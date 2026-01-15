#pragma once

#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <functional>
#include <random>
#include <array>

namespace MolinAntro {
namespace AI {

/**
 * @brief AI Drummer - GarageBand-style intelligent drummer
 * Generates realistic drum patterns based on genre, complexity, and feel
 */
class AIDrummer {
public:
    enum class Genre {
        Rock,
        Pop,
        Jazz,
        Blues,
        Country,
        RnB,
        HipHop,
        Electronic,
        Latin,
        Reggae,
        Metal,
        Punk,
        Funk,
        Soul,
        World
    };

    enum class DrumKit {
        Acoustic,
        Electronic,
        Vintage,
        Modern,
        Brush,
        Percussion,
        TR808,
        TR909,
        LinDrum,
        Custom
    };

    struct DrummerSettings {
        Genre genre = Genre::Rock;
        DrumKit kit = DrumKit::Acoustic;

        float complexity = 0.5f;        // 0-1: Simple to complex patterns
        float loudness = 0.7f;          // 0-1: Soft to loud
        float swing = 0.0f;             // 0-1: Straight to swung
        float fills = 0.3f;             // 0-1: Fill frequency
        float ghost = 0.2f;             // 0-1: Ghost note amount
        float humanize = 0.15f;         // 0-1: Timing/velocity variation

        bool followBass = true;         // Lock to bass pattern
        bool followKeys = false;        // React to keyboard
        bool halfTime = false;          // Half-time feel
        bool doubleTime = false;        // Double-time feel

        int barsPerSection = 8;         // Section length for variations
    };

    struct DrumMap {
        int kick = 36;
        int snare = 38;
        int rimshot = 37;
        int closedHat = 42;
        int openHat = 46;
        int ride = 51;
        int rideBell = 53;
        int crash = 49;
        int crash2 = 57;
        int tomHigh = 48;
        int tomMid = 45;
        int tomLow = 41;
        int floorTom = 43;
        int china = 52;
        int splash = 55;
        int cowbell = 56;
        int clap = 39;
    };

    AIDrummer();
    ~AIDrummer();

    void setSettings(const DrummerSettings& settings) { settings_ = settings; }
    DrummerSettings getSettings() const { return settings_; }

    void setDrumMap(const DrumMap& map) { drumMap_ = map; }
    DrumMap getDrumMap() const { return drumMap_; }

    // Generate a pattern for given number of bars
    std::vector<MIDI::Note> generatePattern(int bars, double bpm);

    // Generate a fill
    std::vector<MIDI::Note> generateFill(double startBeat, double lengthBeats);

    // React to input (bass/keys)
    void feedInput(const std::vector<MIDI::Note>& inputNotes);

    // Real-time generation
    void processBlock(std::vector<MIDI::Note>& output, double startBeat,
                      double endBeat, double bpm);

    // Get available genres
    static std::vector<std::string> getGenreNames();
    static Genre getGenreByName(const std::string& name);

private:
    struct PatternTemplate {
        std::vector<std::pair<double, int>> kickPattern;   // beat, velocity
        std::vector<std::pair<double, int>> snarePattern;
        std::vector<std::pair<double, int>> hatPattern;
        std::vector<std::pair<double, int>> ridePattern;
        std::vector<int> fillNotes;
    };

    PatternTemplate getTemplateForGenre(Genre genre);
    void addGhostNotes(std::vector<MIDI::Note>& notes, double startBeat, double endBeat);
    void humanizePattern(std::vector<MIDI::Note>& notes);
    bool shouldPlayFill(double beat);

    DrummerSettings settings_;
    DrumMap drumMap_;

    std::vector<MIDI::Note> bassNotes_;
    std::vector<MIDI::Note> keyNotes_;

    std::mt19937 rng_;
    double lastFillBeat_ = -100.0;
    int currentBar_ = 0;
};

/**
 * @brief Chord detection and analysis
 */
class ChordAnalyzer {
public:
    struct ChordInfo {
        int rootNote = 60;              // MIDI note
        std::string name = "C";         // Chord name (C, Dm, G7, etc.)
        std::string quality = "maj";    // maj, min, dim, aug, 7, etc.
        std::vector<int> notes;         // All notes in chord
        double startBeat = 0.0;
        double duration = 4.0;
        float confidence = 1.0f;        // Detection confidence
    };

    ChordAnalyzer();
    ~ChordAnalyzer();

    // Analyze MIDI notes and detect chords
    std::vector<ChordInfo> analyzeNotes(const std::vector<MIDI::Note>& notes);

    // Analyze audio and detect chords
    std::vector<ChordInfo> analyzeAudio(const Core::AudioBuffer& buffer,
                                         int sampleRate, double bpm);

    // Get chord at specific beat
    ChordInfo getChordAtBeat(double beat) const;

    // Suggest next chords
    std::vector<ChordInfo> suggestNextChords(const ChordInfo& current, int count = 4);

    // Common progressions
    static std::vector<std::string> getCommonProgressions(const std::string& key);

private:
    std::string detectChordName(const std::vector<int>& notes);
    std::vector<ChordInfo> detectedChords_;
};

/**
 * @brief Smart Bass - Generates bass lines that follow chord progression
 */
class SmartBass {
public:
    enum class Style {
        Root,           // Root notes only
        Simple,         // Root + fifth
        Walking,        // Walking bass
        Groovy,         // Funk/groove style
        Synth,          // Synth bass patterns
        Slap,           // Slap bass style
        Reggae,         // Reggae bass
        Latin           // Latin bass patterns
    };

    struct BassSettings {
        Style style = Style::Simple;
        float complexity = 0.5f;
        float swing = 0.0f;
        float octaveRange = 1.0f;       // 0=stay in octave, 1=use octave jumps
        bool followDrums = true;
        int baseOctave = 2;             // C2 = MIDI 36
    };

    SmartBass();
    ~SmartBass();

    void setSettings(const BassSettings& settings) { settings_ = settings; }

    // Generate bass line for chord progression
    std::vector<MIDI::Note> generateBassLine(
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        double bpm);

    // Real-time generation following chord
    void processBlock(std::vector<MIDI::Note>& output,
                      const ChordAnalyzer::ChordInfo& currentChord,
                      double startBeat, double endBeat, double bpm);

private:
    BassSettings settings_;
    std::mt19937 rng_;
};

/**
 * @brief Smart Keys - Piano/keyboard accompaniment
 */
class SmartKeys {
public:
    enum class Style {
        Block,          // Block chords
        Arpeggio,       // Arpeggiated
        Comping,        // Jazz comping
        Pop,            // Pop piano
        Classical,      // Classical style
        Pad,            // Sustained pads
        Strum           // Guitar-like strumming
    };

    enum class Voicing {
        Close,          // Close voicing
        Open,           // Open voicing
        Drop2,          // Drop 2
        Drop3,          // Drop 3
        Quartal,        // Quartal harmony
        Shell           // Shell voicings (3rds and 7ths)
    };

    struct KeysSettings {
        Style style = Style::Pop;
        Voicing voicing = Voicing::Close;
        float complexity = 0.5f;
        float swing = 0.0f;
        float velocity = 0.7f;
        int baseOctave = 4;             // C4 = MIDI 60
        bool addBassNote = false;       // Add bass note in left hand
    };

    SmartKeys();
    ~SmartKeys();

    void setSettings(const KeysSettings& settings) { settings_ = settings; }

    // Generate accompaniment for chord progression
    std::vector<MIDI::Note> generateAccompaniment(
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        double bpm);

    // Get chord voicing
    std::vector<int> getVoicing(const ChordAnalyzer::ChordInfo& chord);

private:
    KeysSettings settings_;
    std::mt19937 rng_;
};

/**
 * @brief Smart Strings - String section accompaniment
 */
class SmartStrings {
public:
    enum class Style {
        Sustain,        // Long sustained notes
        Tremolo,        // Tremolo effect
        Pizzicato,      // Pizzicato
        Spiccato,       // Spiccato bowing
        Legato,         // Smooth legato
        Staccato,       // Short notes
        Crescendo       // Building dynamics
    };

    struct StringsSettings {
        Style style = Style::Sustain;
        int voices = 4;                 // Number of parts
        float spread = 0.5f;            // Voice spacing
        float humanize = 0.1f;
        float dynamics = 0.7f;          // Overall volume
        bool divisi = true;             // Use divisi voicings
    };

    SmartStrings();
    ~SmartStrings();

    void setSettings(const StringsSettings& settings) { settings_ = settings; }

    std::vector<MIDI::Note> generateParts(
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        double bpm);

private:
    StringsSettings settings_;
    std::mt19937 rng_;
};

/**
 * @brief Live Loops controller - GarageBand-style loop triggering
 */
class LiveLoops {
public:
    struct LoopCell {
        std::string name;
        int row = 0;
        int column = 0;

        enum class Type { Empty, Audio, MIDI, Pattern } type = Type::Empty;

        // Audio loop
        std::shared_ptr<Core::AudioBuffer> audioBuffer;
        double originalTempo = 120.0;

        // MIDI loop
        std::vector<MIDI::Note> midiNotes;

        // Pattern reference
        int patternIndex = -1;

        // Playback
        double lengthBeats = 4.0;
        bool queued = false;
        bool playing = false;
        float gain = 1.0f;

        // Color for UI
        uint32_t color = 0xFF5577DD;
    };

    struct Row {
        std::string name;
        std::vector<LoopCell> cells;
        bool muted = false;
        bool soloed = false;
        float volume = 1.0f;

        // Instrument assignment
        int instrumentId = -1;
    };

    LiveLoops(int numRows = 8, int numColumns = 8);
    ~LiveLoops();

    // Grid management
    void setGridSize(int rows, int columns);
    int getNumRows() const { return static_cast<int>(rows_.size()); }
    int getNumColumns() const { return numColumns_; }

    // Cell operations
    LoopCell* getCell(int row, int column);
    void setCell(int row, int column, const LoopCell& cell);
    void clearCell(int row, int column);

    // Row operations
    Row* getRow(int index);
    void setRowMute(int row, bool mute);
    void setRowSolo(int row, bool solo);

    // Playback
    void triggerCell(int row, int column);
    void stopCell(int row, int column);
    void triggerColumn(int column);       // Scene launch
    void stopAll();

    // Transport
    void setPlaying(bool playing) { playing_ = playing; }
    bool isPlaying() const { return playing_; }
    void setPosition(double beat) { positionBeat_ = beat; }
    double getPosition() const { return positionBeat_; }
    void setBPM(double bpm) { bpm_ = bpm; }
    double getBPM() const { return bpm_; }

    // Quantization
    enum class QuantizeMode { None, Beat, Bar, TwoBars };
    void setQuantizeMode(QuantizeMode mode) { quantize_ = mode; }

    // Process
    void processAudio(Core::AudioBuffer& output);
    void processMIDI(std::vector<MIDI::MIDIMessage>& output, int numSamples, int sampleRate);

private:
    std::vector<Row> rows_;
    int numColumns_;

    bool playing_ = false;
    double positionBeat_ = 0.0;
    double bpm_ = 120.0;

    QuantizeMode quantize_ = QuantizeMode::Bar;
};

/**
 * @brief Melody generator - Creates melodies following chord progression
 */
class MelodyGenerator {
public:
    enum class Style {
        Pop,            // Pop melody style
        Jazz,           // Jazz improvisation
        Classical,      // Classical phrasing
        EDM,            // Electronic melody
        Blues,          // Blues licks
        Country,        // Country melody
        RnB,            // R&B style
        Rock            // Rock melody
    };

    struct MelodySettings {
        Style style = Style::Pop;
        float complexity = 0.5f;
        float rhythmVariation = 0.3f;
        float rangeOctaves = 1.5f;
        int baseOctave = 4;
        float restProbability = 0.2f;
        bool useScale = true;
        std::vector<int> scale;         // Scale intervals
        int rootNote = 0;               // C
    };

    MelodyGenerator();
    ~MelodyGenerator();

    void setSettings(const MelodySettings& settings) { settings_ = settings; }

    // Generate melody for chord progression
    std::vector<MIDI::Note> generateMelody(
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        double bpm, int bars);

    // Continue melody from given notes
    std::vector<MIDI::Note> continueMelody(
        const std::vector<MIDI::Note>& existing,
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        double bpm, int bars);

private:
    MelodySettings settings_;
    std::mt19937 rng_;

    int getNextNote(int currentNote, const ChordAnalyzer::ChordInfo& chord);
    double getNextDuration(double currentBeat, double barEnd);
};

/**
 * @brief Auto-Accompaniment - Complete backing track generation
 */
class AutoAccompaniment {
public:
    struct AccompanimentParts {
        bool drums = true;
        bool bass = true;
        bool keys = true;
        bool strings = false;
        bool guitar = false;
        bool pad = false;
    };

    struct AccompanimentSettings {
        std::string genre = "Pop";
        AccompanimentParts parts;
        float energy = 0.5f;            // Overall energy level
        float complexity = 0.5f;
        bool buildUp = true;            // Gradually build up parts
        int introBeats = 8;
        int outroBeats = 8;
    };

    AutoAccompaniment();
    ~AutoAccompaniment();

    void setSettings(const AccompanimentSettings& settings) { settings_ = settings; }

    // Generate full accompaniment
    struct AccompanimentTracks {
        std::vector<MIDI::Note> drums;
        std::vector<MIDI::Note> bass;
        std::vector<MIDI::Note> keys;
        std::vector<MIDI::Note> strings;
        std::vector<MIDI::Note> melody;
    };

    AccompanimentTracks generate(
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        double bpm, int bars);

    // Real-time generation
    void processBlock(AccompanimentTracks& output,
                      const ChordAnalyzer::ChordInfo& currentChord,
                      double startBeat, double endBeat, double bpm);

private:
    AccompanimentSettings settings_;

    std::unique_ptr<AIDrummer> drummer_;
    std::unique_ptr<SmartBass> bass_;
    std::unique_ptr<SmartKeys> keys_;
    std::unique_ptr<SmartStrings> strings_;
    std::unique_ptr<MelodyGenerator> melody_;
};

/**
 * @brief Chord suggester - Suggests chord progressions
 */
class ChordSuggester {
public:
    struct ChordSuggestion {
        std::string chord;
        float probability;
        std::string reason;             // Why this chord is suggested
    };

    ChordSuggester();
    ~ChordSuggester();

    // Get suggestions based on current progression
    std::vector<ChordSuggestion> suggest(
        const std::vector<std::string>& currentProgression,
        const std::string& key,
        int count = 4);

    // Get common progressions in key
    std::vector<std::vector<std::string>> getCommonProgressions(
        const std::string& key,
        int length = 4);

    // Analyze emotional character of progression
    std::string analyzeProgression(const std::vector<std::string>& progression);

private:
    struct ChordTransition {
        std::string from;
        std::string to;
        float probability;
    };

    std::vector<ChordTransition> transitions_;
    void loadTransitionData();
};

/**
 * @brief Song structure analyzer and generator
 */
class SongStructure {
public:
    struct Section {
        std::string name;               // Intro, Verse, Chorus, etc.
        int startBar = 0;
        int lengthBars = 8;
        float energy = 0.5f;
        std::vector<std::string> chords;
        uint32_t color = 0xFF4444FF;
    };

    SongStructure();
    ~SongStructure();

    // Analyze existing song structure
    std::vector<Section> analyzeSong(
        const std::vector<ChordAnalyzer::ChordInfo>& chords,
        const Core::AudioBuffer* audioRef = nullptr);

    // Generate structure for genre
    std::vector<Section> generateStructure(
        const std::string& genre,
        int totalBars);

    // Get section at bar
    const Section* getSectionAtBar(int bar) const;

    // Common structures
    static std::vector<std::string> getCommonStructures();

private:
    std::vector<Section> sections_;
};

} // namespace AI
} // namespace MolinAntro
