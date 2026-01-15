/**
 * @file SmartInstruments.cpp
 * @brief FULL GarageBand-style AI Smart Instruments Implementation
 *
 * Professional AI-powered music generation with:
 * - AI Drummer with genre-specific patterns
 * - Chord detection and analysis
 * - Smart Bass following chords
 * - Smart Keys/Piano accompaniment
 * - Auto-accompaniment
 * - Melody generation
 * - Live Loops
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "ai/SmartInstruments.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace MolinAntro {
namespace AI {

//=============================================================================
// AIDrummer Implementation
//=============================================================================

AIDrummer::AIDrummer()
    : rng_(std::random_device{}())
{
}

AIDrummer::~AIDrummer() = default;

std::vector<std::string> AIDrummer::getGenreNames() {
    return {
        "Rock", "Pop", "Jazz", "Blues", "Country",
        "R&B", "Hip Hop", "Electronic", "Latin", "Reggae",
        "Metal", "Punk", "Funk", "Soul", "World"
    };
}

AIDrummer::Genre AIDrummer::getGenreByName(const std::string& name) {
    static const std::map<std::string, Genre> genreMap = {
        {"Rock", Genre::Rock}, {"Pop", Genre::Pop}, {"Jazz", Genre::Jazz},
        {"Blues", Genre::Blues}, {"Country", Genre::Country}, {"R&B", Genre::RnB},
        {"Hip Hop", Genre::HipHop}, {"Electronic", Genre::Electronic},
        {"Latin", Genre::Latin}, {"Reggae", Genre::Reggae}, {"Metal", Genre::Metal},
        {"Punk", Genre::Punk}, {"Funk", Genre::Funk}, {"Soul", Genre::Soul},
        {"World", Genre::World}
    };

    auto it = genreMap.find(name);
    return it != genreMap.end() ? it->second : Genre::Rock;
}

AIDrummer::PatternTemplate AIDrummer::getTemplateForGenre(Genre genre) {
    PatternTemplate tpl;

    switch (genre) {
        case Genre::Rock:
            // Standard 4/4 rock beat
            tpl.kickPattern = {{0.0, 100}, {1.5, 80}, {2.0, 100}, {3.5, 70}};
            tpl.snarePattern = {{1.0, 100}, {3.0, 100}};
            tpl.hatPattern = {{0.0, 80}, {0.5, 60}, {1.0, 80}, {1.5, 60},
                              {2.0, 80}, {2.5, 60}, {3.0, 80}, {3.5, 60}};
            tpl.fillNotes = {drumMap_.tomHigh, drumMap_.tomMid, drumMap_.tomLow,
                             drumMap_.snare, drumMap_.crash};
            break;

        case Genre::Pop:
            // Clean pop beat with stronger hats
            tpl.kickPattern = {{0.0, 100}, {2.0, 100}};
            tpl.snarePattern = {{1.0, 90}, {3.0, 90}};
            tpl.hatPattern = {{0.0, 70}, {0.5, 50}, {1.0, 70}, {1.5, 50},
                              {2.0, 70}, {2.5, 50}, {3.0, 70}, {3.5, 50}};
            tpl.fillNotes = {drumMap_.snare, drumMap_.tomHigh, drumMap_.crash};
            break;

        case Genre::Jazz:
            // Swing jazz pattern with ride
            tpl.kickPattern = {{0.0, 70}, {2.5, 60}};
            tpl.snarePattern = {}; // Ghost notes instead
            tpl.ridePattern = {{0.0, 80}, {0.67, 60}, {1.0, 80}, {1.67, 60},
                               {2.0, 80}, {2.67, 60}, {3.0, 80}, {3.67, 60}};
            tpl.hatPattern = {{1.0, 50}};
            tpl.fillNotes = {drumMap_.snare, drumMap_.crash};
            break;

        case Genre::HipHop:
            // Boom bap pattern
            tpl.kickPattern = {{0.0, 110}, {0.75, 90}, {2.0, 110}, {2.75, 80}};
            tpl.snarePattern = {{1.0, 100}, {3.0, 100}};
            tpl.hatPattern = {{0.0, 70}, {0.25, 50}, {0.5, 70}, {0.75, 50},
                              {1.0, 70}, {1.25, 50}, {1.5, 70}, {1.75, 50},
                              {2.0, 70}, {2.25, 50}, {2.5, 70}, {2.75, 50},
                              {3.0, 70}, {3.25, 50}, {3.5, 70}, {3.75, 50}};
            tpl.fillNotes = {drumMap_.snare, drumMap_.clap, drumMap_.openHat};
            break;

        case Genre::Funk:
            // Syncopated funk groove
            tpl.kickPattern = {{0.0, 100}, {0.5, 80}, {2.0, 100}, {2.75, 90}, {3.5, 70}};
            tpl.snarePattern = {{1.0, 100}, {2.5, 80}, {3.0, 100}};
            tpl.hatPattern = {{0.0, 80}, {0.25, 60}, {0.5, 80}, {0.75, 60},
                              {1.0, 80}, {1.25, 60}, {1.5, 80}, {1.75, 60},
                              {2.0, 80}, {2.25, 60}, {2.5, 80}, {2.75, 60},
                              {3.0, 80}, {3.25, 60}, {3.5, 80}, {3.75, 60}};
            tpl.fillNotes = {drumMap_.snare, drumMap_.tomHigh, drumMap_.cowbell};
            break;

        case Genre::Metal:
            // Double bass metal
            tpl.kickPattern = {{0.0, 120}, {0.25, 110}, {0.5, 120}, {0.75, 110},
                               {1.0, 120}, {1.25, 110}, {1.5, 120}, {1.75, 110},
                               {2.0, 120}, {2.25, 110}, {2.5, 120}, {2.75, 110},
                               {3.0, 120}, {3.25, 110}, {3.5, 120}, {3.75, 110}};
            tpl.snarePattern = {{1.0, 110}, {3.0, 110}};
            tpl.hatPattern = {{0.0, 90}, {0.5, 90}, {1.0, 90}, {1.5, 90},
                              {2.0, 90}, {2.5, 90}, {3.0, 90}, {3.5, 90}};
            tpl.fillNotes = {drumMap_.tomHigh, drumMap_.tomMid, drumMap_.tomLow,
                             drumMap_.floorTom, drumMap_.china};
            break;

        case Genre::Electronic:
            // Four-on-the-floor
            tpl.kickPattern = {{0.0, 110}, {1.0, 110}, {2.0, 110}, {3.0, 110}};
            tpl.snarePattern = {{1.0, 90}};
            tpl.hatPattern = {{0.5, 80}, {1.5, 80}, {2.5, 80}, {3.5, 80}};
            tpl.fillNotes = {drumMap_.clap, drumMap_.openHat, drumMap_.crash};
            break;

        case Genre::Reggae:
            // One drop
            tpl.kickPattern = {{2.5, 100}};
            tpl.snarePattern = {{3.0, 90}};
            tpl.hatPattern = {{0.0, 60}, {0.5, 80}, {1.0, 60}, {1.5, 80},
                              {2.0, 60}, {2.5, 80}, {3.0, 60}, {3.5, 80}};
            tpl.fillNotes = {drumMap_.snare, drumMap_.rimshot};
            break;

        case Genre::Latin:
            // Basic Latin clave pattern
            tpl.kickPattern = {{0.0, 90}, {1.5, 80}, {3.0, 90}};
            tpl.snarePattern = {{1.0, 85}, {2.5, 85}};
            tpl.hatPattern = {{0.0, 70}, {0.5, 70}, {1.0, 70}, {1.5, 70},
                              {2.0, 70}, {2.5, 70}, {3.0, 70}, {3.5, 70}};
            tpl.fillNotes = {drumMap_.cowbell, drumMap_.tomHigh, drumMap_.tomMid};
            break;

        default:
            // Default rock pattern
            tpl = getTemplateForGenre(Genre::Rock);
            break;
    }

    return tpl;
}

std::vector<MIDI::Note> AIDrummer::generatePattern(int bars, double bpm) {
    std::vector<MIDI::Note> notes;
    PatternTemplate tpl = getTemplateForGenre(settings_.genre);

    std::uniform_real_distribution<float> velVar(-0.1f, 0.1f);
    std::uniform_real_distribution<float> timeVar(-0.02f, 0.02f);
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);

    for (int bar = 0; bar < bars; ++bar) {
        double barOffset = bar * 4.0;

        // Check for fill
        bool playFill = bar > 0 && (bar % settings_.barsPerSection == settings_.barsPerSection - 1);
        if (playFill && prob(rng_) < settings_.fills) {
            auto fill = generateFill(barOffset + 2.0, 2.0);
            notes.insert(notes.end(), fill.begin(), fill.end());
            continue; // Skip regular pattern for fill bars
        }

        // Kick pattern
        for (const auto& hit : tpl.kickPattern) {
            if (prob(rng_) > settings_.complexity * 0.3f + 0.7f) continue;

            MIDI::Note note;
            note.noteNumber = drumMap_.kick;
            note.startTime = barOffset + hit.first;
            note.duration = 0.1;
            note.velocity = static_cast<uint8_t>(std::clamp(
                hit.second * settings_.loudness * (1.0f + velVar(rng_) * settings_.humanize),
                1.0f, 127.0f));

            if (settings_.humanize > 0) {
                note.startTime += timeVar(rng_) * settings_.humanize;
            }

            notes.push_back(note);
        }

        // Snare pattern
        for (const auto& hit : tpl.snarePattern) {
            MIDI::Note note;
            note.noteNumber = drumMap_.snare;
            note.startTime = barOffset + hit.first;
            note.duration = 0.1;
            note.velocity = static_cast<uint8_t>(std::clamp(
                hit.second * settings_.loudness * (1.0f + velVar(rng_) * settings_.humanize),
                1.0f, 127.0f));

            if (settings_.humanize > 0) {
                note.startTime += timeVar(rng_) * settings_.humanize * 0.5f;
            }

            notes.push_back(note);
        }

        // Hi-hat pattern
        for (const auto& hit : tpl.hatPattern) {
            if (settings_.complexity < 0.3f && (static_cast<int>(hit.first * 4) % 2 != 0)) {
                continue; // Skip off-beat hats at low complexity
            }

            MIDI::Note note;
            note.noteNumber = drumMap_.closedHat;

            // Apply swing to off-beats
            double swing = (static_cast<int>(hit.first * 4) % 2 == 1) ?
                           settings_.swing * 0.1 : 0.0;

            note.startTime = barOffset + hit.first + swing;
            note.duration = 0.05;
            note.velocity = static_cast<uint8_t>(std::clamp(
                hit.second * settings_.loudness * (1.0f + velVar(rng_) * settings_.humanize),
                1.0f, 127.0f));

            notes.push_back(note);
        }

        // Ride pattern (for jazz, etc.)
        for (const auto& hit : tpl.ridePattern) {
            MIDI::Note note;
            note.noteNumber = drumMap_.ride;
            note.startTime = barOffset + hit.first;
            note.duration = 0.1;
            note.velocity = static_cast<uint8_t>(std::clamp(
                hit.second * settings_.loudness * (1.0f + velVar(rng_) * settings_.humanize),
                1.0f, 127.0f));

            notes.push_back(note);
        }

        // Ghost notes
        if (settings_.ghost > 0) {
            addGhostNotes(notes, barOffset, barOffset + 4.0);
        }
    }

    // Add crash on beat 1 of first bar
    if (!notes.empty()) {
        MIDI::Note crash;
        crash.noteNumber = drumMap_.crash;
        crash.startTime = 0.0;
        crash.duration = 0.5;
        crash.velocity = static_cast<uint8_t>(90 * settings_.loudness);
        notes.insert(notes.begin(), crash);
    }

    return notes;
}

std::vector<MIDI::Note> AIDrummer::generateFill(double startBeat, double lengthBeats) {
    std::vector<MIDI::Note> notes;
    PatternTemplate tpl = getTemplateForGenre(settings_.genre);

    std::uniform_real_distribution<float> velVar(0.8f, 1.0f);
    std::uniform_int_distribution<int> noteChoice(0, static_cast<int>(tpl.fillNotes.size()) - 1);

    // Number of hits based on complexity
    int numHits = static_cast<int>(4 + settings_.complexity * 12);
    double spacing = lengthBeats / numHits;

    for (int i = 0; i < numHits; ++i) {
        MIDI::Note note;
        note.noteNumber = tpl.fillNotes[noteChoice(rng_)];
        note.startTime = startBeat + i * spacing;
        note.duration = 0.1;
        note.velocity = static_cast<uint8_t>(
            std::clamp(80.0f + i * 3.0f, 80.0f, 120.0f) * velVar(rng_) * settings_.loudness
        );
        notes.push_back(note);
    }

    // End fill with crash
    MIDI::Note crash;
    crash.noteNumber = drumMap_.crash;
    crash.startTime = startBeat + lengthBeats;
    crash.duration = 0.5;
    crash.velocity = static_cast<uint8_t>(100 * settings_.loudness);
    notes.push_back(crash);

    return notes;
}

void AIDrummer::addGhostNotes(std::vector<MIDI::Note>& notes, double startBeat, double endBeat) {
    std::uniform_real_distribution<float> prob(0.0f, 1.0f);
    std::uniform_real_distribution<float> velVar(0.2f, 0.4f);

    // Add ghost snares on 16th note positions
    for (double beat = startBeat; beat < endBeat; beat += 0.25) {
        // Skip positions where main hits occur
        bool hasMainHit = false;
        for (const auto& note : notes) {
            if (std::abs(note.startTime - beat) < 0.1) {
                hasMainHit = true;
                break;
            }
        }

        if (!hasMainHit && prob(rng_) < settings_.ghost * 0.5f) {
            MIDI::Note ghost;
            ghost.noteNumber = drumMap_.snare;
            ghost.startTime = beat;
            ghost.duration = 0.05;
            ghost.velocity = static_cast<uint8_t>(velVar(rng_) * 127 * settings_.loudness);
            notes.push_back(ghost);
        }
    }
}

void AIDrummer::humanizePattern(std::vector<MIDI::Note>& notes) {
    std::uniform_real_distribution<float> timeVar(-0.02f, 0.02f);
    std::uniform_real_distribution<float> velVar(-10.0f, 10.0f);

    for (auto& note : notes) {
        note.startTime += timeVar(rng_) * settings_.humanize;
        int newVel = note.velocity + static_cast<int>(velVar(rng_) * settings_.humanize);
        note.velocity = static_cast<uint8_t>(std::clamp(newVel, 1, 127));
    }
}

void AIDrummer::feedInput(const std::vector<MIDI::Note>& inputNotes) {
    // Store for follow mode
    bassNotes_ = inputNotes;
}

void AIDrummer::processBlock(std::vector<MIDI::Note>& output, double startBeat,
                              double endBeat, double bpm) {
    // Real-time pattern generation
    auto pattern = generatePattern(1, bpm);

    for (const auto& note : pattern) {
        double noteBeat = std::fmod(note.startTime, 4.0);
        double patternStart = std::floor(startBeat / 4.0) * 4.0;

        if (patternStart + noteBeat >= startBeat && patternStart + noteBeat < endBeat) {
            MIDI::Note outputNote = note;
            outputNote.startTime = patternStart + noteBeat;
            output.push_back(outputNote);
        }
    }
}

//=============================================================================
// ChordAnalyzer Implementation
//=============================================================================

ChordAnalyzer::ChordAnalyzer() = default;
ChordAnalyzer::~ChordAnalyzer() = default;

std::string ChordAnalyzer::detectChordName(const std::vector<int>& noteNumbers) {
    if (noteNumbers.empty()) return "N.C.";

    // Normalize to pitch classes
    std::vector<int> pitchClasses;
    for (int note : noteNumbers) {
        int pc = note % 12;
        if (std::find(pitchClasses.begin(), pitchClasses.end(), pc) == pitchClasses.end()) {
            pitchClasses.push_back(pc);
        }
    }
    std::sort(pitchClasses.begin(), pitchClasses.end());

    if (pitchClasses.size() < 2) {
        static const char* noteNames[] = {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"};
        return noteNames[pitchClasses[0]];
    }

    // Find root (assume lowest note for simplicity)
    int root = pitchClasses[0];
    static const char* noteNames[] = {"C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"};

    // Get intervals from root
    std::vector<int> intervals;
    for (int pc : pitchClasses) {
        intervals.push_back((pc - root + 12) % 12);
    }
    std::sort(intervals.begin(), intervals.end());

    // Detect chord quality
    std::string quality;
    bool has3 = std::find(intervals.begin(), intervals.end(), 4) != intervals.end();
    bool hasMinor3 = std::find(intervals.begin(), intervals.end(), 3) != intervals.end();
    bool has5 = std::find(intervals.begin(), intervals.end(), 7) != intervals.end();
    bool hasDim5 = std::find(intervals.begin(), intervals.end(), 6) != intervals.end();
    bool hasAug5 = std::find(intervals.begin(), intervals.end(), 8) != intervals.end();
    bool has7 = std::find(intervals.begin(), intervals.end(), 10) != intervals.end();
    bool hasMaj7 = std::find(intervals.begin(), intervals.end(), 11) != intervals.end();
    bool has9 = std::find(intervals.begin(), intervals.end(), 2) != intervals.end();
    bool has2 = has9;
    bool has4 = std::find(intervals.begin(), intervals.end(), 5) != intervals.end();

    if (has3 && has5) {
        quality = "";  // Major
    } else if (hasMinor3 && has5) {
        quality = "m"; // Minor
    } else if (hasMinor3 && hasDim5) {
        quality = "dim";
    } else if (has3 && hasAug5) {
        quality = "aug";
    } else if (!has3 && !hasMinor3 && has2 && has5) {
        quality = "sus2";
    } else if (!has3 && !hasMinor3 && has4 && has5) {
        quality = "sus4";
    } else if (has3 && has5) {
        quality = "";
    }

    // Add 7th
    if (hasMaj7) {
        quality += "maj7";
    } else if (has7) {
        quality += "7";
    }

    return std::string(noteNames[root]) + quality;
}

std::vector<ChordAnalyzer::ChordInfo> ChordAnalyzer::analyzeNotes(
    const std::vector<MIDI::Note>& notes) {

    detectedChords_.clear();
    if (notes.empty()) return detectedChords_;

    // Group notes by time windows (e.g., 1 beat)
    double windowSize = 1.0;
    double minStart = notes.front().startTime;
    double maxEnd = minStart;

    for (const auto& note : notes) {
        maxEnd = std::max(maxEnd, note.startTime + note.duration);
    }

    for (double windowStart = minStart; windowStart < maxEnd; windowStart += windowSize) {
        double windowEnd = windowStart + windowSize;

        // Collect notes in this window
        std::vector<int> windowNotes;
        for (const auto& note : notes) {
            if (note.startTime >= windowStart && note.startTime < windowEnd) {
                windowNotes.push_back(note.noteNumber);
            }
        }

        if (windowNotes.size() >= 2) {
            ChordInfo chord;
            chord.notes = windowNotes;
            chord.name = detectChordName(windowNotes);
            chord.rootNote = windowNotes[0] % 12;
            chord.startBeat = windowStart;
            chord.duration = windowSize;

            // Merge with previous if same chord
            if (!detectedChords_.empty() &&
                detectedChords_.back().name == chord.name) {
                detectedChords_.back().duration += windowSize;
            } else {
                detectedChords_.push_back(chord);
            }
        }
    }

    return detectedChords_;
}

std::vector<ChordAnalyzer::ChordInfo> ChordAnalyzer::analyzeAudio(
    const Core::AudioBuffer& buffer, int sampleRate, double bpm) {

    // Simplified audio chord detection using FFT bins
    // Full implementation would use chromagram analysis
    detectedChords_.clear();

    // For now, return empty - full implementation requires FFT
    return detectedChords_;
}

ChordAnalyzer::ChordInfo ChordAnalyzer::getChordAtBeat(double beat) const {
    for (const auto& chord : detectedChords_) {
        if (beat >= chord.startBeat && beat < chord.startBeat + chord.duration) {
            return chord;
        }
    }
    return ChordInfo(); // Return empty/default chord
}

std::vector<ChordAnalyzer::ChordInfo> ChordAnalyzer::suggestNextChords(
    const ChordInfo& current, int count) {

    std::vector<ChordInfo> suggestions;

    // Common chord progressions based on current chord
    std::map<std::string, std::vector<std::string>> transitions = {
        {"C", {"G", "Am", "F", "Dm"}},
        {"G", {"C", "D", "Em", "Am"}},
        {"Am", {"F", "G", "Dm", "E"}},
        {"F", {"C", "G", "Dm", "Am"}},
        {"Dm", {"G", "Am", "F", "Bb"}},
        {"Em", {"Am", "C", "G", "D"}},
        {"D", {"G", "A", "Bm", "Em"}},
    };

    auto it = transitions.find(current.name);
    if (it != transitions.end()) {
        for (int i = 0; i < std::min(count, static_cast<int>(it->second.size())); ++i) {
            ChordInfo suggestion;
            suggestion.name = it->second[i];
            suggestion.confidence = 1.0f - i * 0.2f;
            suggestions.push_back(suggestion);
        }
    }

    return suggestions;
}

std::vector<std::string> ChordAnalyzer::getCommonProgressions(const std::string& key) {
    // Return common progressions for the key
    return {
        "I-IV-V-I",
        "I-V-vi-IV",
        "I-vi-IV-V",
        "ii-V-I",
        "I-IV-vi-V"
    };
}

//=============================================================================
// SmartBass Implementation
//=============================================================================

SmartBass::SmartBass()
    : rng_(std::random_device{}())
{
}

SmartBass::~SmartBass() = default;

std::vector<MIDI::Note> SmartBass::generateBassLine(
    const std::vector<ChordAnalyzer::ChordInfo>& chords, double bpm) {

    std::vector<MIDI::Note> notes;
    std::uniform_real_distribution<float> velVar(0.9f, 1.0f);

    int baseNote = settings_.baseOctave * 12; // C2 = 36

    for (const auto& chord : chords) {
        int rootNote = chord.rootNote + baseNote;
        double beat = chord.startBeat;
        double endBeat = beat + chord.duration;

        switch (settings_.style) {
            case Style::Root: {
                // Just root notes
                MIDI::Note note;
                note.noteNumber = rootNote;
                note.startTime = beat;
                note.duration = chord.duration * 0.9;
                note.velocity = static_cast<uint8_t>(100 * velVar(rng_));
                notes.push_back(note);
                break;
            }

            case Style::Simple: {
                // Root and fifth
                for (double b = beat; b < endBeat; b += 2.0) {
                    MIDI::Note root;
                    root.noteNumber = rootNote;
                    root.startTime = b;
                    root.duration = 0.9;
                    root.velocity = static_cast<uint8_t>(100 * velVar(rng_));
                    notes.push_back(root);

                    if (b + 1.0 < endBeat) {
                        MIDI::Note fifth;
                        fifth.noteNumber = rootNote + 7;
                        fifth.startTime = b + 1.0;
                        fifth.duration = 0.9;
                        fifth.velocity = static_cast<uint8_t>(90 * velVar(rng_));
                        notes.push_back(fifth);
                    }
                }
                break;
            }

            case Style::Walking: {
                // Walking bass (jazz style)
                std::vector<int> walkPattern = {0, 2, 4, 5, 7, 5, 4, 2};
                int idx = 0;
                for (double b = beat; b < endBeat; b += 0.5) {
                    MIDI::Note note;
                    note.noteNumber = rootNote + walkPattern[idx % walkPattern.size()];
                    note.startTime = b;
                    note.duration = 0.45;
                    note.velocity = static_cast<uint8_t>(85 * velVar(rng_));
                    notes.push_back(note);
                    idx++;
                }
                break;
            }

            case Style::Groovy: {
                // Funk/groove style
                std::vector<std::pair<double, int>> pattern = {
                    {0.0, 0}, {0.5, -12}, {0.75, 0}, {1.5, 7},
                    {2.0, 0}, {2.75, 5}, {3.0, 0}, {3.5, 7}
                };
                for (const auto& hit : pattern) {
                    if (beat + hit.first < endBeat) {
                        MIDI::Note note;
                        note.noteNumber = rootNote + hit.second;
                        note.startTime = beat + hit.first;
                        note.duration = 0.2;
                        note.velocity = static_cast<uint8_t>(95 * velVar(rng_));
                        notes.push_back(note);
                    }
                }
                break;
            }

            case Style::Synth: {
                // Synth bass - 16th notes
                for (double b = beat; b < endBeat; b += 0.25) {
                    bool isAccent = (static_cast<int>((b - beat) * 4) % 4 == 0);
                    MIDI::Note note;
                    note.noteNumber = rootNote;
                    note.startTime = b;
                    note.duration = 0.2;
                    note.velocity = static_cast<uint8_t>((isAccent ? 100 : 70) * velVar(rng_));
                    notes.push_back(note);
                }
                break;
            }

            case Style::Reggae: {
                // Reggae - off-beat emphasis
                for (double b = beat + 0.5; b < endBeat; b += 1.0) {
                    MIDI::Note note;
                    note.noteNumber = rootNote;
                    note.startTime = b;
                    note.duration = 0.4;
                    note.velocity = static_cast<uint8_t>(90 * velVar(rng_));
                    notes.push_back(note);
                }
                break;
            }

            default:
                // Default to root notes
                {
                    MIDI::Note note;
                    note.noteNumber = rootNote;
                    note.startTime = beat;
                    note.duration = chord.duration * 0.9;
                    note.velocity = static_cast<uint8_t>(100 * velVar(rng_));
                    notes.push_back(note);
                }
                break;
        }
    }

    return notes;
}

void SmartBass::processBlock(std::vector<MIDI::Note>& output,
                              const ChordAnalyzer::ChordInfo& currentChord,
                              double startBeat, double endBeat, double bpm) {
    std::vector<ChordAnalyzer::ChordInfo> chords = {currentChord};
    chords[0].startBeat = startBeat;
    chords[0].duration = endBeat - startBeat;

    auto generated = generateBassLine(chords, bpm);
    output.insert(output.end(), generated.begin(), generated.end());
}

//=============================================================================
// SmartKeys Implementation
//=============================================================================

SmartKeys::SmartKeys()
    : rng_(std::random_device{}())
{
}

SmartKeys::~SmartKeys() = default;

std::vector<int> SmartKeys::getVoicing(const ChordAnalyzer::ChordInfo& chord) {
    std::vector<int> voicing;
    int baseNote = settings_.baseOctave * 12;

    // Basic intervals for common chord types
    std::vector<int> intervals;
    if (chord.quality == "m" || chord.name.find("m") != std::string::npos) {
        intervals = {0, 3, 7};  // Minor
    } else if (chord.name.find("dim") != std::string::npos) {
        intervals = {0, 3, 6};  // Diminished
    } else if (chord.name.find("aug") != std::string::npos) {
        intervals = {0, 4, 8};  // Augmented
    } else if (chord.name.find("7") != std::string::npos) {
        intervals = {0, 4, 7, 10};  // Dominant 7
    } else if (chord.name.find("maj7") != std::string::npos) {
        intervals = {0, 4, 7, 11};  // Major 7
    } else {
        intervals = {0, 4, 7};  // Major
    }

    // Apply voicing
    switch (settings_.voicing) {
        case Voicing::Close:
            for (int i : intervals) {
                voicing.push_back(chord.rootNote + baseNote + i);
            }
            break;

        case Voicing::Open:
            if (intervals.size() >= 3) {
                voicing.push_back(chord.rootNote + baseNote + intervals[0]);
                voicing.push_back(chord.rootNote + baseNote + intervals[2] + 12);
                voicing.push_back(chord.rootNote + baseNote + intervals[1] + 12);
            }
            break;

        case Voicing::Drop2:
            if (intervals.size() >= 3) {
                voicing.push_back(chord.rootNote + baseNote + intervals[1] - 12);
                voicing.push_back(chord.rootNote + baseNote + intervals[0]);
                voicing.push_back(chord.rootNote + baseNote + intervals[2]);
            }
            break;

        case Voicing::Shell:
            // Just 3rd and 7th
            voicing.push_back(chord.rootNote + baseNote);
            if (intervals.size() >= 2) {
                voicing.push_back(chord.rootNote + baseNote + intervals[1]);
            }
            if (intervals.size() >= 4) {
                voicing.push_back(chord.rootNote + baseNote + intervals[3]);
            }
            break;

        default:
            for (int i : intervals) {
                voicing.push_back(chord.rootNote + baseNote + i);
            }
            break;
    }

    return voicing;
}

std::vector<MIDI::Note> SmartKeys::generateAccompaniment(
    const std::vector<ChordAnalyzer::ChordInfo>& chords, double bpm) {

    std::vector<MIDI::Note> notes;
    std::uniform_real_distribution<float> velVar(0.85f, 1.0f);

    for (const auto& chord : chords) {
        auto voicing = getVoicing(chord);
        double beat = chord.startBeat;
        double endBeat = beat + chord.duration;

        switch (settings_.style) {
            case Style::Block: {
                // Block chords on each beat
                for (double b = beat; b < endBeat; b += 1.0) {
                    for (int noteNum : voicing) {
                        MIDI::Note note;
                        note.noteNumber = noteNum;
                        note.startTime = b;
                        note.duration = 0.9;
                        note.velocity = static_cast<uint8_t>(
                            settings_.velocity * 127 * velVar(rng_));
                        notes.push_back(note);
                    }
                }
                break;
            }

            case Style::Arpeggio: {
                // Arpeggiated pattern
                int idx = 0;
                for (double b = beat; b < endBeat; b += 0.5) {
                    MIDI::Note note;
                    note.noteNumber = voicing[idx % voicing.size()];
                    note.startTime = b;
                    note.duration = 0.45;
                    note.velocity = static_cast<uint8_t>(
                        settings_.velocity * 127 * velVar(rng_));
                    notes.push_back(note);
                    idx++;
                }
                break;
            }

            case Style::Comping: {
                // Jazz comping - syncopated
                std::vector<double> compPattern = {0.0, 1.5, 2.5};
                for (double offset : compPattern) {
                    if (beat + offset < endBeat) {
                        for (int noteNum : voicing) {
                            MIDI::Note note;
                            note.noteNumber = noteNum;
                            note.startTime = beat + offset;
                            note.duration = 0.7;
                            note.velocity = static_cast<uint8_t>(
                                settings_.velocity * 100 * velVar(rng_));
                            notes.push_back(note);
                        }
                    }
                }
                break;
            }

            case Style::Pop: {
                // Pop piano - mix of sustained and rhythmic
                // First beat - sustained
                for (int noteNum : voicing) {
                    MIDI::Note note;
                    note.noteNumber = noteNum;
                    note.startTime = beat;
                    note.duration = chord.duration * 0.9;
                    note.velocity = static_cast<uint8_t>(
                        settings_.velocity * 100 * velVar(rng_));
                    notes.push_back(note);
                }
                break;
            }

            case Style::Pad: {
                // Long sustained pads
                for (int noteNum : voicing) {
                    MIDI::Note note;
                    note.noteNumber = noteNum;
                    note.startTime = beat;
                    note.duration = chord.duration;
                    note.velocity = static_cast<uint8_t>(
                        settings_.velocity * 90 * velVar(rng_));
                    notes.push_back(note);
                }
                break;
            }

            default:
                // Default to block chords
                for (int noteNum : voicing) {
                    MIDI::Note note;
                    note.noteNumber = noteNum;
                    note.startTime = beat;
                    note.duration = chord.duration * 0.9;
                    note.velocity = static_cast<uint8_t>(
                        settings_.velocity * 127 * velVar(rng_));
                    notes.push_back(note);
                }
                break;
        }

        // Add bass note if enabled
        if (settings_.addBassNote) {
            MIDI::Note bass;
            bass.noteNumber = chord.rootNote + (settings_.baseOctave - 1) * 12;
            bass.startTime = beat;
            bass.duration = chord.duration * 0.9;
            bass.velocity = static_cast<uint8_t>(settings_.velocity * 110 * velVar(rng_));
            notes.push_back(bass);
        }
    }

    return notes;
}

//=============================================================================
// SmartStrings Implementation
//=============================================================================

SmartStrings::SmartStrings()
    : rng_(std::random_device{}())
{
}

SmartStrings::~SmartStrings() = default;

std::vector<MIDI::Note> SmartStrings::generateParts(
    const std::vector<ChordAnalyzer::ChordInfo>& chords, double bpm) {

    std::vector<MIDI::Note> notes;
    std::uniform_real_distribution<float> velVar(0.9f, 1.0f);
    std::uniform_real_distribution<float> timeVar(-0.05f, 0.05f);

    for (const auto& chord : chords) {
        // Generate voicing for strings (spread across 4 parts)
        std::vector<int> baseIntervals;
        if (chord.name.find("m") != std::string::npos) {
            baseIntervals = {0, 3, 7, 12};
        } else {
            baseIntervals = {0, 4, 7, 12};
        }

        // Spread across octaves for string section
        std::vector<int> stringVoicing;
        int baseNote = 48; // C3 - cello range

        for (int v = 0; v < settings_.voices; ++v) {
            int interval = baseIntervals[v % baseIntervals.size()];
            int octaveOffset = (v / 4) * 12;
            stringVoicing.push_back(chord.rootNote + baseNote + interval + octaveOffset);
        }

        switch (settings_.style) {
            case Style::Sustain: {
                for (int noteNum : stringVoicing) {
                    MIDI::Note note;
                    note.noteNumber = noteNum;
                    note.startTime = chord.startBeat;
                    note.duration = chord.duration;
                    note.velocity = static_cast<uint8_t>(settings_.dynamics * 100 * velVar(rng_));

                    // Add humanization
                    if (settings_.humanize > 0) {
                        note.startTime += timeVar(rng_) * settings_.humanize;
                    }

                    notes.push_back(note);
                }
                break;
            }

            case Style::Tremolo: {
                // Fast repeated notes
                for (int noteNum : stringVoicing) {
                    for (double b = chord.startBeat; b < chord.startBeat + chord.duration; b += 0.125) {
                        MIDI::Note note;
                        note.noteNumber = noteNum;
                        note.startTime = b;
                        note.duration = 0.1;
                        note.velocity = static_cast<uint8_t>(settings_.dynamics * 80 * velVar(rng_));
                        notes.push_back(note);
                    }
                }
                break;
            }

            case Style::Staccato: {
                // Short detached notes
                for (double b = chord.startBeat; b < chord.startBeat + chord.duration; b += 0.5) {
                    for (int noteNum : stringVoicing) {
                        MIDI::Note note;
                        note.noteNumber = noteNum;
                        note.startTime = b;
                        note.duration = 0.2;
                        note.velocity = static_cast<uint8_t>(settings_.dynamics * 90 * velVar(rng_));
                        notes.push_back(note);
                    }
                }
                break;
            }

            default:
                // Default sustain
                for (int noteNum : stringVoicing) {
                    MIDI::Note note;
                    note.noteNumber = noteNum;
                    note.startTime = chord.startBeat;
                    note.duration = chord.duration;
                    note.velocity = static_cast<uint8_t>(settings_.dynamics * 100 * velVar(rng_));
                    notes.push_back(note);
                }
                break;
        }
    }

    return notes;
}

//=============================================================================
// LiveLoops Implementation
//=============================================================================

LiveLoops::LiveLoops(int numRows, int numColumns)
    : numColumns_(numColumns)
{
    rows_.resize(numRows);
    for (int i = 0; i < numRows; ++i) {
        rows_[i].name = "Track " + std::to_string(i + 1);
        rows_[i].cells.resize(numColumns);
    }
}

LiveLoops::~LiveLoops() = default;

void LiveLoops::setGridSize(int rows, int columns) {
    rows_.resize(rows);
    numColumns_ = columns;
    for (auto& row : rows_) {
        row.cells.resize(columns);
    }
}

LiveLoops::LoopCell* LiveLoops::getCell(int row, int column) {
    if (row >= 0 && row < static_cast<int>(rows_.size()) &&
        column >= 0 && column < numColumns_) {
        return &rows_[row].cells[column];
    }
    return nullptr;
}

void LiveLoops::setCell(int row, int column, const LoopCell& cell) {
    if (auto* c = getCell(row, column)) {
        *c = cell;
        c->row = row;
        c->column = column;
    }
}

void LiveLoops::clearCell(int row, int column) {
    if (auto* c = getCell(row, column)) {
        *c = LoopCell();
        c->row = row;
        c->column = column;
    }
}

LiveLoops::Row* LiveLoops::getRow(int index) {
    if (index >= 0 && index < static_cast<int>(rows_.size())) {
        return &rows_[index];
    }
    return nullptr;
}

void LiveLoops::setRowMute(int row, bool mute) {
    if (auto* r = getRow(row)) {
        r->muted = mute;
    }
}

void LiveLoops::setRowSolo(int row, bool solo) {
    if (auto* r = getRow(row)) {
        r->soloed = solo;
    }
}

void LiveLoops::triggerCell(int row, int column) {
    auto* cell = getCell(row, column);
    if (!cell || cell->type == LoopCell::Type::Empty) return;

    // Stop other cells in same row
    for (auto& c : rows_[row].cells) {
        c.playing = false;
        c.queued = false;
    }

    // Queue this cell
    cell->queued = true;
}

void LiveLoops::stopCell(int row, int column) {
    if (auto* cell = getCell(row, column)) {
        cell->playing = false;
        cell->queued = false;
    }
}

void LiveLoops::triggerColumn(int column) {
    // Scene launch - trigger all cells in column
    for (int row = 0; row < static_cast<int>(rows_.size()); ++row) {
        triggerCell(row, column);
    }
}

void LiveLoops::stopAll() {
    for (auto& row : rows_) {
        for (auto& cell : row.cells) {
            cell.playing = false;
            cell.queued = false;
        }
    }
}

void LiveLoops::processAudio(Core::AudioBuffer& output) {
    if (!playing_) return;

    // Process each playing cell
    for (auto& row : rows_) {
        if (row.muted) continue;

        for (auto& cell : row.cells) {
            if (!cell.playing || cell.type != LoopCell::Type::Audio) continue;
            if (!cell.audioBuffer) continue;

            // Mix audio into output
            // TODO: Time-stretch to match tempo
        }
    }
}

void LiveLoops::processMIDI(std::vector<MIDI::MIDIMessage>& output,
                            int numSamples, int sampleRate) {
    if (!playing_) return;

    double samplesPerBeat = (60.0 / bpm_) * sampleRate;

    // Check for queued cells at quantize point
    double quantizeBeats = 4.0;
    switch (quantize_) {
        case QuantizeMode::None: quantizeBeats = 0; break;
        case QuantizeMode::Beat: quantizeBeats = 1.0; break;
        case QuantizeMode::Bar: quantizeBeats = 4.0; break;
        case QuantizeMode::TwoBars: quantizeBeats = 8.0; break;
    }

    if (quantizeBeats > 0) {
        double nextQuantize = std::ceil(positionBeat_ / quantizeBeats) * quantizeBeats;
        double endBeat = positionBeat_ + numSamples / samplesPerBeat;

        if (endBeat >= nextQuantize) {
            // Start queued cells
            for (auto& row : rows_) {
                for (auto& cell : row.cells) {
                    if (cell.queued) {
                        cell.playing = true;
                        cell.queued = false;
                    }
                }
            }
        }
    }

    // Generate MIDI from playing cells
    for (auto& row : rows_) {
        if (row.muted) continue;

        for (auto& cell : row.cells) {
            if (!cell.playing || cell.type != LoopCell::Type::MIDI) continue;

            // Find notes in current time window
            double cellPos = std::fmod(positionBeat_, cell.lengthBeats);
            double endPos = cellPos + numSamples / samplesPerBeat;

            for (const auto& note : cell.midiNotes) {
                if (note.startTime >= cellPos && note.startTime < endPos) {
                    MIDI::MIDIMessage noteOn;
                    noteOn.status = 0x90;
                    noteOn.data1 = note.noteNumber;
                    noteOn.data2 = note.velocity;
                    noteOn.timestamp = static_cast<uint64_t>(
                        (note.startTime - cellPos) * samplesPerBeat);
                    output.push_back(noteOn);
                }
            }
        }
    }

    positionBeat_ += numSamples / samplesPerBeat;
}

//=============================================================================
// MelodyGenerator Implementation
//=============================================================================

MelodyGenerator::MelodyGenerator()
    : rng_(std::random_device{}())
{
    // Default major scale
    settings_.scale = {0, 2, 4, 5, 7, 9, 11};
}

MelodyGenerator::~MelodyGenerator() = default;

int MelodyGenerator::getNextNote(int currentNote, const ChordAnalyzer::ChordInfo& chord) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> stepDist(-2, 2);

    // Prefer chord tones and scale tones
    if (dist(rng_) < 0.6f && !chord.notes.empty()) {
        // Use chord tone
        int idx = static_cast<int>(dist(rng_) * chord.notes.size());
        int octaveOffset = (settings_.baseOctave - 4) * 12;
        return chord.notes[idx] % 12 + settings_.baseOctave * 12;
    }

    // Step-wise motion in scale
    int currentScaleDegree = 0;
    int currentPitch = currentNote % 12;

    // Find closest scale degree
    for (int i = 0; i < static_cast<int>(settings_.scale.size()); ++i) {
        int scalePitch = (settings_.rootNote + settings_.scale[i]) % 12;
        if (scalePitch == currentPitch) {
            currentScaleDegree = i;
            break;
        }
    }

    int newDegree = currentScaleDegree + stepDist(rng_);
    newDegree = ((newDegree % static_cast<int>(settings_.scale.size())) +
                  settings_.scale.size()) % settings_.scale.size();

    int newPitch = settings_.rootNote + settings_.scale[newDegree];
    int octave = currentNote / 12;

    // Keep in range
    int newNote = octave * 12 + newPitch;
    int minNote = settings_.baseOctave * 12;
    int maxNote = static_cast<int>(minNote + settings_.rangeOctaves * 12);

    while (newNote < minNote) newNote += 12;
    while (newNote > maxNote) newNote -= 12;

    return newNote;
}

double MelodyGenerator::getNextDuration(double currentBeat, double barEnd) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Rhythm options based on complexity
    std::vector<double> durations;
    if (settings_.complexity < 0.3f) {
        durations = {1.0, 2.0};
    } else if (settings_.complexity < 0.6f) {
        durations = {0.5, 1.0, 1.5, 2.0};
    } else {
        durations = {0.25, 0.5, 0.75, 1.0, 1.5};
    }

    int idx = static_cast<int>(dist(rng_) * durations.size());
    double duration = durations[idx];

    // Don't exceed bar end
    if (currentBeat + duration > barEnd) {
        duration = barEnd - currentBeat;
    }

    return std::max(0.25, duration);
}

std::vector<MIDI::Note> MelodyGenerator::generateMelody(
    const std::vector<ChordAnalyzer::ChordInfo>& chords,
    double bpm, int bars) {

    std::vector<MIDI::Note> notes;
    std::uniform_real_distribution<float> velVar(0.7f, 0.95f);
    std::uniform_real_distribution<float> restProb(0.0f, 1.0f);

    int currentNote = settings_.baseOctave * 12 + settings_.rootNote;

    for (const auto& chord : chords) {
        double beat = chord.startBeat;
        double endBeat = beat + chord.duration;

        while (beat < endBeat) {
            // Rest probability
            if (restProb(rng_) < settings_.restProbability) {
                beat += 0.5;
                continue;
            }

            currentNote = getNextNote(currentNote, chord);
            double duration = getNextDuration(beat, endBeat);

            MIDI::Note note;
            note.noteNumber = currentNote;
            note.startTime = beat;
            note.duration = duration * 0.9;
            note.velocity = static_cast<uint8_t>(100 * velVar(rng_));

            notes.push_back(note);
            beat += duration;
        }
    }

    return notes;
}

std::vector<MIDI::Note> MelodyGenerator::continueMelody(
    const std::vector<MIDI::Note>& existing,
    const std::vector<ChordAnalyzer::ChordInfo>& chords,
    double bpm, int bars) {

    // Analyze existing melody patterns
    // For simplicity, just generate new melody
    return generateMelody(chords, bpm, bars);
}

//=============================================================================
// AutoAccompaniment Implementation
//=============================================================================

AutoAccompaniment::AutoAccompaniment() {
    drummer_ = std::make_unique<AIDrummer>();
    bass_ = std::make_unique<SmartBass>();
    keys_ = std::make_unique<SmartKeys>();
    strings_ = std::make_unique<SmartStrings>();
    melody_ = std::make_unique<MelodyGenerator>();
}

AutoAccompaniment::~AutoAccompaniment() = default;

AutoAccompaniment::AccompanimentTracks AutoAccompaniment::generate(
    const std::vector<ChordAnalyzer::ChordInfo>& chords,
    double bpm, int bars) {

    AccompanimentTracks tracks;

    // Set up instruments based on genre
    AIDrummer::DrummerSettings drumSettings;
    if (settings_.genre == "Rock") {
        drumSettings.genre = AIDrummer::Genre::Rock;
    } else if (settings_.genre == "Jazz") {
        drumSettings.genre = AIDrummer::Genre::Jazz;
    } else if (settings_.genre == "Electronic") {
        drumSettings.genre = AIDrummer::Genre::Electronic;
    } else {
        drumSettings.genre = AIDrummer::Genre::Pop;
    }
    drumSettings.complexity = settings_.complexity;
    drummer_->setSettings(drumSettings);

    SmartBass::BassSettings bassSettings;
    bassSettings.complexity = settings_.complexity;
    bass_->setSettings(bassSettings);

    SmartKeys::KeysSettings keysSettings;
    keysSettings.complexity = settings_.complexity;
    keys_->setSettings(keysSettings);

    // Generate parts
    if (settings_.parts.drums) {
        tracks.drums = drummer_->generatePattern(bars, bpm);
    }

    if (settings_.parts.bass) {
        tracks.bass = bass_->generateBassLine(chords, bpm);
    }

    if (settings_.parts.keys) {
        tracks.keys = keys_->generateAccompaniment(chords, bpm);
    }

    if (settings_.parts.strings) {
        tracks.strings = strings_->generateParts(chords, bpm);
    }

    return tracks;
}

void AutoAccompaniment::processBlock(AccompanimentTracks& output,
                                      const ChordAnalyzer::ChordInfo& currentChord,
                                      double startBeat, double endBeat, double bpm) {

    if (settings_.parts.drums) {
        drummer_->processBlock(output.drums, startBeat, endBeat, bpm);
    }

    if (settings_.parts.bass) {
        bass_->processBlock(output.bass, currentChord, startBeat, endBeat, bpm);
    }
}

//=============================================================================
// ChordSuggester Implementation
//=============================================================================

ChordSuggester::ChordSuggester() {
    loadTransitionData();
}

ChordSuggester::~ChordSuggester() = default;

void ChordSuggester::loadTransitionData() {
    // Common chord transitions (simplified)
    transitions_ = {
        {"C", "G", 0.25f}, {"C", "Am", 0.2f}, {"C", "F", 0.25f}, {"C", "Dm", 0.15f},
        {"G", "C", 0.3f}, {"G", "D", 0.2f}, {"G", "Em", 0.2f}, {"G", "Am", 0.15f},
        {"Am", "F", 0.25f}, {"Am", "G", 0.25f}, {"Am", "Dm", 0.2f}, {"Am", "E", 0.15f},
        {"F", "C", 0.3f}, {"F", "G", 0.25f}, {"F", "Dm", 0.2f}, {"F", "Am", 0.15f},
        {"Dm", "G", 0.3f}, {"Dm", "Am", 0.25f}, {"Dm", "F", 0.2f}, {"Dm", "Bb", 0.15f},
        {"Em", "Am", 0.25f}, {"Em", "C", 0.25f}, {"Em", "G", 0.2f}, {"Em", "D", 0.15f},
    };
}

std::vector<ChordSuggester::ChordSuggestion> ChordSuggester::suggest(
    const std::vector<std::string>& currentProgression,
    const std::string& key, int count) {

    std::vector<ChordSuggestion> suggestions;

    if (currentProgression.empty()) {
        // Suggest common starting chords for key
        suggestions.push_back({"C", 0.9f, "Tonic - stable starting point"});
        suggestions.push_back({"Am", 0.7f, "Relative minor - emotional"});
        suggestions.push_back({"F", 0.6f, "Subdominant - soft start"});
        suggestions.push_back({"G", 0.5f, "Dominant - creates tension"});
        return suggestions;
    }

    std::string lastChord = currentProgression.back();

    // Find matching transitions
    for (const auto& trans : transitions_) {
        if (trans.from == lastChord) {
            ChordSuggestion sug;
            sug.chord = trans.to;
            sug.probability = trans.probability;
            sug.reason = "Common transition from " + lastChord;
            suggestions.push_back(sug);
        }
    }

    // Sort by probability
    std::sort(suggestions.begin(), suggestions.end(),
        [](const ChordSuggestion& a, const ChordSuggestion& b) {
            return a.probability > b.probability;
        });

    // Limit count
    if (static_cast<int>(suggestions.size()) > count) {
        suggestions.resize(count);
    }

    return suggestions;
}

std::vector<std::vector<std::string>> ChordSuggester::getCommonProgressions(
    const std::string& key, int length) {

    // Common 4-chord progressions in C (transpose for other keys)
    return {
        {"C", "G", "Am", "F"},     // Pop progression
        {"Am", "F", "C", "G"},     // Sad but hopeful
        {"C", "Am", "F", "G"},     // 50s progression
        {"F", "G", "C", "Am"},     // Variation
        {"Am", "Dm", "G", "C"},    // Minor start
        {"C", "F", "Am", "G"},     // Uplifting
    };
}

std::string ChordSuggester::analyzeProgression(const std::vector<std::string>& progression) {
    if (progression.empty()) return "Empty progression";

    bool hasMinor = false;
    bool hasMajor = false;

    for (const auto& chord : progression) {
        if (chord.find("m") != std::string::npos && chord.find("maj") == std::string::npos) {
            hasMinor = true;
        } else {
            hasMajor = true;
        }
    }

    if (hasMinor && !hasMajor) return "Dark/melancholic mood";
    if (!hasMinor && hasMajor) return "Bright/happy mood";
    return "Mixed emotional character";
}

//=============================================================================
// SongStructure Implementation
//=============================================================================

SongStructure::SongStructure() = default;
SongStructure::~SongStructure() = default;

std::vector<SongStructure::Section> SongStructure::analyzeSong(
    const std::vector<ChordAnalyzer::ChordInfo>& chords,
    const Core::AudioBuffer* audioRef) {

    sections_.clear();

    // Simple analysis based on chord repetition
    // Full implementation would analyze energy, harmony changes, etc.

    int currentBar = 0;
    int sectionStart = 0;
    std::vector<std::string> currentChords;

    for (const auto& chord : chords) {
        currentChords.push_back(chord.name);

        // Look for section boundaries (every 8 bars or chord pattern repeat)
        if (currentChords.size() >= 8) {
            Section section;
            section.startBar = sectionStart;
            section.lengthBars = 8;
            section.chords = currentChords;

            // Guess section type based on position
            if (sectionStart == 0) {
                section.name = "Intro";
            } else if (sections_.size() % 2 == 0) {
                section.name = "Verse";
            } else {
                section.name = "Chorus";
            }

            sections_.push_back(section);
            sectionStart += 8;
            currentChords.clear();
        }
    }

    return sections_;
}

std::vector<SongStructure::Section> SongStructure::generateStructure(
    const std::string& genre, int totalBars) {

    sections_.clear();

    // Standard pop structure: Intro - Verse - Chorus - Verse - Chorus - Bridge - Chorus - Outro
    struct SectionTemplate {
        std::string name;
        int bars;
        float energy;
    };

    std::vector<SectionTemplate> structure;

    if (genre == "Pop" || genre == "Rock") {
        structure = {
            {"Intro", 8, 0.3f},
            {"Verse 1", 16, 0.5f},
            {"Chorus", 8, 0.8f},
            {"Verse 2", 16, 0.5f},
            {"Chorus", 8, 0.85f},
            {"Bridge", 8, 0.6f},
            {"Chorus", 8, 0.9f},
            {"Outro", 8, 0.4f}
        };
    } else if (genre == "Electronic") {
        structure = {
            {"Intro", 16, 0.3f},
            {"Build", 8, 0.6f},
            {"Drop", 16, 0.9f},
            {"Breakdown", 8, 0.4f},
            {"Build", 8, 0.7f},
            {"Drop", 16, 0.95f},
            {"Outro", 8, 0.3f}
        };
    } else {
        // Default structure
        structure = {
            {"Intro", 8, 0.3f},
            {"Section A", 16, 0.6f},
            {"Section B", 16, 0.8f},
            {"Section A", 16, 0.6f},
            {"Outro", 8, 0.3f}
        };
    }

    int currentBar = 0;
    for (const auto& tpl : structure) {
        if (currentBar >= totalBars) break;

        Section section;
        section.name = tpl.name;
        section.startBar = currentBar;
        section.lengthBars = std::min(tpl.bars, totalBars - currentBar);
        section.energy = tpl.energy;

        sections_.push_back(section);
        currentBar += section.lengthBars;
    }

    return sections_;
}

const SongStructure::Section* SongStructure::getSectionAtBar(int bar) const {
    for (const auto& section : sections_) {
        if (bar >= section.startBar && bar < section.startBar + section.lengthBars) {
            return &section;
        }
    }
    return nullptr;
}

std::vector<std::string> SongStructure::getCommonStructures() {
    return {
        "Intro-Verse-Chorus-Verse-Chorus-Bridge-Chorus-Outro (Pop)",
        "Intro-Build-Drop-Breakdown-Build-Drop-Outro (EDM)",
        "AABA (Jazz Standard)",
        "12-Bar Blues",
        "Verse-Verse-Chorus-Verse-Chorus (Folk)",
        "Intro-Theme-Variation-Theme-Coda (Classical)"
    };
}

} // namespace AI
} // namespace MolinAntro
