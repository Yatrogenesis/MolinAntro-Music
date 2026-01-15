#pragma once

/**
 * @file MainWindow.h
 * @brief Professional Qt6 GUI Architecture for MolinAntro DAW
 *
 * Complete GUI framework matching professional DAWs:
 * - Multi-dock window system
 * - Mixer view
 * - Piano roll
 * - Arrangement view
 * - Session view
 * - Browser panel
 * - Plugin editor windows
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 *
 * NOTE: Actual implementation requires Qt6 Framework
 * Install Qt6: https://www.qt.io/download
 * CMake: find_package(Qt6 COMPONENTS Widgets REQUIRED)
 */

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <map>

// Forward declarations for Qt types (would be actual Qt includes)
// #include <QMainWindow>
// #include <QDockWidget>
// #include <QWidget>

namespace MolinAntro {
namespace UI {

/**
 * @brief Theme configuration
 */
struct Theme {
    std::string name;

    // Colors (as 0xAARRGGBB)
    uint32_t background = 0xFF1E1E1E;
    uint32_t backgroundAlt = 0xFF252525;
    uint32_t foreground = 0xFFE0E0E0;
    uint32_t accent = 0xFF0078D7;
    uint32_t accentLight = 0xFF3399FF;
    uint32_t selection = 0xFF264F78;
    uint32_t border = 0xFF3C3C3C;

    // Track colors
    uint32_t trackAudio = 0xFF5588CC;
    uint32_t trackMIDI = 0xFF88CC55;
    uint32_t trackReturn = 0xFFCC8855;
    uint32_t trackMaster = 0xFFCC5555;

    // Meters
    uint32_t meterGreen = 0xFF44CC44;
    uint32_t meterYellow = 0xFFCCCC44;
    uint32_t meterRed = 0xFFCC4444;

    // Piano roll
    uint32_t noteDefault = 0xFF6688BB;
    uint32_t noteSelected = 0xFF88AADD;
    uint32_t grid = 0xFF2A2A2A;
    uint32_t gridBar = 0xFF404040;

    // Waveform
    uint32_t waveform = 0xFF5599DD;
    uint32_t waveformRMS = 0xFF88BBEE;

    static Theme getDarkTheme();
    static Theme getLightTheme();
    static Theme getAbletonTheme();
    static Theme getFLStudioTheme();
    static Theme getLogicTheme();
};

/**
 * @brief Panel/Dock configuration
 */
struct PanelConfig {
    std::string id;
    std::string title;
    bool visible = true;
    bool floating = false;
    int dockArea = 0;  // 1=Left, 2=Right, 4=Top, 8=Bottom
    int width = 300;
    int height = 200;
    int x = 0;
    int y = 0;
};

/**
 * @brief Keyboard shortcut
 */
struct KeyShortcut {
    std::string action;
    std::string key;  // e.g., "Ctrl+S", "Space", "F5"
    std::string category;
    std::string description;
};

/**
 * @brief Transport bar widget
 */
class TransportBar {
public:
    TransportBar();
    ~TransportBar();

    // Transport controls
    void play();
    void stop();
    void pause();
    void record();
    void rewind();
    void fastForward();
    void loop();
    void metronome();

    // Display
    void setPosition(double beats);
    void setTempo(double bpm);
    void setTimeSignature(int num, int denom);
    void setLoopRange(double start, double end);

    // Mode
    void setSongMode(bool song);
    bool isSongMode() const { return songMode_; }

    // Callbacks
    using TransportCallback = std::function<void()>;
    void setPlayCallback(TransportCallback cb) { onPlay_ = cb; }
    void setStopCallback(TransportCallback cb) { onStop_ = cb; }
    void setRecordCallback(TransportCallback cb) { onRecord_ = cb; }

private:
    double position_ = 0.0;
    double tempo_ = 120.0;
    int timeSignatureNum_ = 4;
    int timeSignatureDenom_ = 4;
    bool songMode_ = true;
    bool loopEnabled_ = false;
    bool metronomeEnabled_ = true;

    TransportCallback onPlay_;
    TransportCallback onStop_;
    TransportCallback onRecord_;
};

/**
 * @brief Mixer view panel
 */
class MixerView {
public:
    struct ChannelStrip {
        int trackIndex;
        std::string name;
        float volume = 0.0f;    // dB
        float pan = 0.0f;       // -1 to 1
        bool mute = false;
        bool solo = false;
        bool arm = false;
        float meterL = 0.0f;    // 0-1
        float meterR = 0.0f;
        uint32_t color = 0xFF888888;

        // Sends
        std::vector<std::pair<int, float>> sends;  // returnIndex, amount
    };

    MixerView();
    ~MixerView();

    void setNumChannels(int count);
    ChannelStrip* getChannel(int index);
    void updateMeters(const std::vector<std::pair<float, float>>& levels);

    // Callbacks
    using VolumeCallback = std::function<void(int track, float dB)>;
    using PanCallback = std::function<void(int track, float pan)>;
    using MuteCallback = std::function<void(int track, bool mute)>;
    using SoloCallback = std::function<void(int track, bool solo)>;

    void setVolumeCallback(VolumeCallback cb) { onVolume_ = cb; }
    void setPanCallback(PanCallback cb) { onPan_ = cb; }
    void setMuteCallback(MuteCallback cb) { onMute_ = cb; }
    void setSoloCallback(SoloCallback cb) { onSolo_ = cb; }

private:
    std::vector<ChannelStrip> channels_;
    VolumeCallback onVolume_;
    PanCallback onPan_;
    MuteCallback onMute_;
    SoloCallback onSolo_;
};

/**
 * @brief Piano roll editor
 */
class PianoRollView {
public:
    struct ViewSettings {
        double horizontalZoom = 1.0;
        double verticalZoom = 1.0;
        double scrollX = 0.0;
        int scrollY = 60;  // Center on middle C
        double gridSize = 0.25;  // 16th notes
        bool tripletGrid = false;
        bool snapToGrid = true;
        bool showGhostNotes = true;
        bool showVelocity = true;
    };

    enum class Tool {
        Select,
        Draw,
        Erase,
        Slice,
        Mute,
        Velocity,
        Pan,
        Pitch
    };

    PianoRollView();
    ~PianoRollView();

    void setViewSettings(const ViewSettings& settings) { settings_ = settings; }
    ViewSettings getViewSettings() const { return settings_; }

    void setTool(Tool tool) { currentTool_ = tool; }
    Tool getTool() const { return currentTool_; }

    // Note editing callbacks
    using NoteCallback = std::function<void(int note, double start, double length, float velocity)>;
    void setAddNoteCallback(NoteCallback cb) { onAddNote_ = cb; }
    void setEditNoteCallback(NoteCallback cb) { onEditNote_ = cb; }
    void setDeleteNoteCallback(std::function<void(int index)> cb) { onDeleteNote_ = cb; }

    // Scale highlighting
    void setScale(int root, const std::vector<int>& intervals);
    void clearScale();

    // Ghost notes from other patterns
    void setGhostPattern(int patternIndex);
    void clearGhostPattern();

private:
    ViewSettings settings_;
    Tool currentTool_ = Tool::Draw;

    NoteCallback onAddNote_;
    NoteCallback onEditNote_;
    std::function<void(int)> onDeleteNote_;

    std::vector<int> highlightedScale_;
    int scaleRoot_ = 0;
    int ghostPatternIndex_ = -1;
};

/**
 * @brief Arrangement/Timeline view
 */
class ArrangementView {
public:
    struct TrackLane {
        int trackIndex;
        std::string name;
        int height = 80;
        bool collapsed = false;
        bool locked = false;
        uint32_t color = 0xFF888888;
    };

    ArrangementView();
    ~ArrangementView();

    void setNumTracks(int count);
    TrackLane* getTrackLane(int index);

    // View
    void setHorizontalZoom(double zoom);
    void setVerticalZoom(double zoom);
    void scrollTo(double beat);
    void scrollToTrack(int trackIndex);

    // Selection
    void selectRegion(double startBeat, double endBeat, int startTrack, int endTrack);
    void selectAll();
    void deselectAll();

    // Markers
    void addMarker(double beat, const std::string& name, uint32_t color);
    void removeMarker(int index);

    // Loop
    void setLoopRange(double start, double end);
    void setLoopEnabled(bool enabled);

    // Callbacks
    using RegionCallback = std::function<void(int track, double start, double end, int patternIndex)>;
    void setAddRegionCallback(RegionCallback cb) { onAddRegion_ = cb; }
    void setMoveRegionCallback(RegionCallback cb) { onMoveRegion_ = cb; }

private:
    std::vector<TrackLane> tracks_;
    double horizontalZoom_ = 1.0;
    double verticalZoom_ = 1.0;

    RegionCallback onAddRegion_;
    RegionCallback onMoveRegion_;
};

/**
 * @brief Session/Clip launcher view (Ableton-style)
 */
class SessionView {
public:
    struct ClipCell {
        int trackIndex;
        int sceneIndex;
        std::string name;
        uint32_t color = 0xFF5577DD;

        enum class State { Empty, Stopped, Playing, Recording, Queued } state = State::Empty;
    };

    SessionView();
    ~SessionView();

    void setGridSize(int tracks, int scenes);
    ClipCell* getCell(int track, int scene);

    void updateCellState(int track, int scene, ClipCell::State state);

    // Callbacks
    using CellCallback = std::function<void(int track, int scene)>;
    void setLaunchCallback(CellCallback cb) { onLaunch_ = cb; }
    void setStopCallback(CellCallback cb) { onStop_ = cb; }
    void setRecordCallback(CellCallback cb) { onRecord_ = cb; }
    void setSceneLaunchCallback(std::function<void(int scene)> cb) { onSceneLaunch_ = cb; }

private:
    std::vector<std::vector<ClipCell>> grid_;
    int numTracks_ = 8;
    int numScenes_ = 8;

    CellCallback onLaunch_;
    CellCallback onStop_;
    CellCallback onRecord_;
    std::function<void(int)> onSceneLaunch_;
};

/**
 * @brief Browser panel for samples, presets, plugins
 */
class BrowserPanel {
public:
    struct BrowserItem {
        std::string id;
        std::string name;
        std::string path;
        enum class Type { Folder, AudioFile, MIDIFile, Preset, Plugin, Project } type;
        bool isFavorite = false;
        std::vector<std::string> tags;
    };

    BrowserPanel();
    ~BrowserPanel();

    // Navigation
    void setRootPath(const std::string& path);
    void navigateTo(const std::string& path);
    void navigateUp();
    std::string getCurrentPath() const { return currentPath_; }

    // Search
    void search(const std::string& query);
    void setFilter(BrowserItem::Type type);
    void clearFilter();

    // Favorites
    void addFavorite(const std::string& path);
    void removeFavorite(const std::string& path);
    std::vector<BrowserItem> getFavorites() const;

    // Recent
    std::vector<BrowserItem> getRecent(int limit = 20) const;
    void addRecent(const std::string& path);

    // Preview
    void previewAudio(const std::string& path);
    void stopPreview();
    bool isPreviewing() const { return previewing_; }

    // Callbacks
    using ItemCallback = std::function<void(const BrowserItem&)>;
    void setSelectCallback(ItemCallback cb) { onSelect_ = cb; }
    void setDoubleClickCallback(ItemCallback cb) { onDoubleClick_ = cb; }
    void setDragCallback(ItemCallback cb) { onDrag_ = cb; }

private:
    std::string currentPath_;
    std::string searchQuery_;
    std::vector<BrowserItem> items_;
    std::vector<std::string> favorites_;
    std::vector<std::string> recentFiles_;
    bool previewing_ = false;

    ItemCallback onSelect_;
    ItemCallback onDoubleClick_;
    ItemCallback onDrag_;
};

/**
 * @brief Plugin editor window
 */
class PluginEditorWindow {
public:
    PluginEditorWindow();
    ~PluginEditorWindow();

    void setPluginId(const std::string& pluginId);
    void setTitle(const std::string& title);

    // For generic UI (when plugin has no custom UI)
    void showGenericUI();
    void showPluginUI(void* nativeWindowHandle);

    // Preset handling
    void loadPreset(const std::string& path);
    void savePreset(const std::string& path);
    std::vector<std::string> getPresetList() const;

    // A/B comparison
    void setStateA();
    void setStateB();
    void compareAB();

    // Callbacks
    using ParameterCallback = std::function<void(int paramIndex, float value)>;
    void setParameterCallback(ParameterCallback cb) { onParameter_ = cb; }

private:
    std::string pluginId_;
    std::string title_;
    ParameterCallback onParameter_;
};

/**
 * @brief Main DAW window
 */
class MainWindow {
public:
    MainWindow();
    ~MainWindow();

    // Initialization
    bool initialize();
    void show();
    void hide();
    bool isVisible() const { return visible_; }

    // Panels
    TransportBar* getTransport() { return transport_.get(); }
    MixerView* getMixer() { return mixer_.get(); }
    PianoRollView* getPianoRoll() { return pianoRoll_.get(); }
    ArrangementView* getArrangement() { return arrangement_.get(); }
    SessionView* getSession() { return session_.get(); }
    BrowserPanel* getBrowser() { return browser_.get(); }

    // Panel visibility
    void showPanel(const std::string& panelId);
    void hidePanel(const std::string& panelId);
    void togglePanel(const std::string& panelId);
    bool isPanelVisible(const std::string& panelId) const;

    // Layout
    void saveLayout(const std::string& name);
    void loadLayout(const std::string& name);
    std::vector<std::string> getLayoutNames() const;
    void resetLayout();

    // Theme
    void setTheme(const Theme& theme);
    Theme getTheme() const { return theme_; }

    // Keyboard shortcuts
    void setShortcut(const std::string& action, const std::string& key);
    std::string getShortcut(const std::string& action) const;
    std::vector<KeyShortcut> getAllShortcuts() const;
    void resetShortcuts();

    // Plugin editors
    PluginEditorWindow* openPluginEditor(const std::string& pluginId);
    void closePluginEditor(const std::string& pluginId);
    void closeAllPluginEditors();

    // Dialogs
    void showPreferences();
    void showAbout();
    void showExportDialog();
    void showProjectSettings();

    // Status bar
    void setStatusMessage(const std::string& message, int timeoutMs = 3000);
    void setCPUUsage(float percentage);
    void setMemoryUsage(size_t bytes);
    void setLatency(float ms);

    // Undo/Redo indicators
    void setUndoEnabled(bool enabled);
    void setRedoEnabled(bool enabled);
    void setUndoText(const std::string& text);
    void setRedoText(const std::string& text);

    // Window state
    void setTitle(const std::string& title);
    void setModified(bool modified);
    bool isModified() const { return modified_; }

    // Event loop
    int run();  // Returns exit code
    void quit();

private:
    bool visible_ = false;
    bool modified_ = false;
    Theme theme_;

    std::unique_ptr<TransportBar> transport_;
    std::unique_ptr<MixerView> mixer_;
    std::unique_ptr<PianoRollView> pianoRoll_;
    std::unique_ptr<ArrangementView> arrangement_;
    std::unique_ptr<SessionView> session_;
    std::unique_ptr<BrowserPanel> browser_;

    std::map<std::string, std::unique_ptr<PluginEditorWindow>> pluginEditors_;
    std::map<std::string, PanelConfig> panels_;
    std::map<std::string, std::string> shortcuts_;
};

/**
 * @brief Waveform display widget
 */
class WaveformDisplay {
public:
    WaveformDisplay();
    ~WaveformDisplay();

    void setAudioData(const float* data, int numSamples, int numChannels);
    void setZoom(double samplesPerPixel);
    void setScroll(int sampleOffset);

    // Display options
    void setShowRMS(bool show) { showRMS_ = show; }
    void setShowPeaks(bool show) { showPeaks_ = show; }
    void setColor(uint32_t color) { waveformColor_ = color; }

    // Selection
    void setSelection(int startSample, int endSample);
    void clearSelection();

    // Markers
    void addMarker(int sample, const std::string& label, uint32_t color);
    void removeMarker(int index);

private:
    std::vector<float> audioData_;
    int numChannels_ = 2;
    double samplesPerPixel_ = 100.0;
    int scrollOffset_ = 0;

    bool showRMS_ = true;
    bool showPeaks_ = true;
    uint32_t waveformColor_ = 0xFF5599DD;

    int selectionStart_ = -1;
    int selectionEnd_ = -1;
};

/**
 * @brief Automation curve editor
 */
class AutomationEditor {
public:
    struct Point {
        double time;        // In beats
        float value;        // 0-1 normalized
        int curveType;      // 0=hold, 1=linear, 2=smooth, etc.
        float tension;
    };

    AutomationEditor();
    ~AutomationEditor();

    void setPoints(const std::vector<Point>& points);
    std::vector<Point> getPoints() const { return points_; }

    void setParameterName(const std::string& name) { paramName_ = name; }
    void setRange(float min, float max);

    // Editing
    void addPoint(double time, float value);
    void movePoint(int index, double time, float value);
    void removePoint(int index);
    void setCurve(int index, int curveType, float tension);

    // Callbacks
    using PointCallback = std::function<void(int index, double time, float value)>;
    void setPointCallback(PointCallback cb) { onPoint_ = cb; }

private:
    std::vector<Point> points_;
    std::string paramName_;
    float minValue_ = 0.0f;
    float maxValue_ = 1.0f;

    PointCallback onPoint_;
};

} // namespace UI
} // namespace MolinAntro
