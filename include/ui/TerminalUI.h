#pragma once

#include "core/AudioEngine.h"
#include "core/Transport.h"
#include "dsp/AudioFile.h"
#include "instruments/Synthesizer.h"
#include "plugins/PluginHost.h"
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace MolinAntro {
namespace UI {

/**
 * Advanced Terminal User Interface
 * Professional DAW interface in the terminal (ncurses-like but portable)
 */

class TerminalUI {
public:
    TerminalUI();
    ~TerminalUI();

    // Main interface
    void run();
    void shutdown();

    // Configuration
    void setAudioEngine(Core::AudioEngine* engine);
    void setTransport(Core::Transport* transport);

private:
    Core::AudioEngine* engine_;
    Core::Transport* transport_;
    bool running_;

    // UI state
    int selectedTrack_;
    int selectedPlugin_;
    float masterVolume_;

    // Display functions
    void drawMainScreen();
    void drawTransportBar();
    void drawTrackList();
    void drawPluginChain();
    void drawVUMeters();
    void drawStatusBar();
    void clearScreen();
    void drawBox(int x, int y, int width, int height, const std::string& title);
    void drawText(int x, int y, const std::string& text);
    void drawProgressBar(int x, int y, int width, float value, const std::string& label);

    // Input handling
    void handleKeypress(char key);
    char getKeypress();

    // Navigation
    void navigateUp();
    void navigateDown();
    void navigateLeft();
    void navigateRight();

    // Transport controls
    void playPause();
    void stopTransport();
    void record();

    // Track operations
    void soloTrack();
    void muteTrack();
    void adjustTrackVolume(float delta);
};

/**
 * Simple ASCII-based UI (no ncurses dependency)
 * Provides professional terminal interface using ANSI escape codes
 */
class SimpleTerminalUI {
public:
    SimpleTerminalUI();
    ~SimpleTerminalUI();

    void run();
    void setEngineStatus(const std::string& status);
    void setTransportStatus(const std::string& status);
    void setCPUUsage(float cpu);
    void addLogMessage(const std::string& message);

    // Menu system
    enum class MenuOption {
        NewProject,
        LoadProject,
        SaveProject,
        ImportAudio,
        ExportAudio,
        Preferences,
        Quit
    };

    MenuOption showMainMenu();
    void showAbout();
    void showHelp();

private:
    std::vector<std::string> logMessages_;
    std::string engineStatus_;
    std::string transportStatus_;
    float cpuUsage_;
    bool running_;

    // ANSI escape codes
    void moveCursor(int row, int col);
    void clearScreen();
    void setColor(int foreground, int background = 0);
    void resetColor();
    void clearLine();
    void hideCursor();
    void showCursor();
    void enableRawMode();
    void disableRawMode();

    // Display helpers
    void drawHeader();
    void drawFooter();
    void drawMenu(const std::vector<std::string>& options, int selected);
    void drawPanel(int x, int y, int width, int height, const std::string& title);
    void drawSeparator(int width);

    // Input
    int getMenuSelection(const std::vector<std::string>& options);
    std::string getInput(const std::string& prompt);
    bool confirm(const std::string& message);
};

/**
 * Progress Display for Long Operations
 */
class ProgressDisplay {
public:
    ProgressDisplay(const std::string& title, int total);
    ~ProgressDisplay();

    void update(int current, const std::string& message = "");
    void finish(const std::string& message = "");

private:
    std::string title_;
    int total_;
    int current_;
    void draw();
};

/**
 * Live Mixer View
 */
class MixerView {
public:
    MixerView();

    struct Track {
        std::string name;
        float volume;
        float pan;
        bool muted;
        bool solo;
        float peakL;
        float peakR;
    };

    void addTrack(const Track& track);
    void updateTrack(int index, const Track& track);
    void display();

private:
    std::vector<Track> tracks_;
    void drawFader(int x, int y, float value, const std::string& label);
    void drawVUMeter(int x, int y, float peak);
};

} // namespace UI
} // namespace MolinAntro
