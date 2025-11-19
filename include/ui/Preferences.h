#pragma once

#include "ui/Theme.h"
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <functional>

namespace MolinAntro {
namespace UI {

/**
 * @brief User preferences and settings system
 */
class Preferences {
public:
    static Preferences& getInstance();

    // ============================================
    // Theme & Appearance
    // ============================================

    struct AppearanceSettings {
        std::string themeName{"Dark"};
        bool followSystemTheme{false};
        float uiScale{1.0f};             ///< UI scaling (0.8 - 2.0)
        bool animationsEnabled{true};
        float animationSpeed{1.0f};      ///< Animation speed multiplier
        bool showTooltips{true};
        float tooltipDelay{0.5f};        ///< Seconds before showing tooltip
    };

    AppearanceSettings& appearance() { return appearance_; }
    const AppearanceSettings& appearance() const { return appearance_; }

    // ============================================
    // Accessibility
    // ============================================

    struct AccessibilitySettings {
        bool highContrast{false};
        bool reducedMotion{false};
        bool largeText{false};
        float textScale{1.0f};
        bool screenReaderMode{false};
        bool dyslexiaMode{false};
        Accessibility::ColorblindMode colorblindMode{Accessibility::ColorblindMode::None};
        bool keyboardNavigation{true};
        bool focusIndicators{true};
        Accessibility::WCAGLevel wcagLevel{Accessibility::WCAGLevel::AAA};
    };

    AccessibilitySettings& accessibility() { return accessibility_; }
    const AccessibilitySettings& accessibility() const { return accessibility_; }

    // ============================================
    // Audio Settings
    // ============================================

    struct AudioSettings {
        std::string audioDevice;
        int sampleRate{48000};
        int bufferSize{512};
        int inputChannels{2};
        int outputChannels{2};
        bool lowLatencyMode{false};
        float masterVolume{0.8f};
        bool autoSave{true};
        int autoSaveInterval{300};  ///< Seconds
    };

    AudioSettings& audio() { return audio_; }
    const AudioSettings& audio() const { return audio_; }

    // ============================================
    // MIDI Settings
    // ============================================

    struct MIDISettings {
        std::vector<std::string> inputDevices;
        std::vector<std::string> outputDevices;
        bool midiThru{false};
        int midiChannel{1};
        bool velocitySensitive{true};
        float velocityCurve{1.0f};  ///< 0.5 = soft, 1.0 = linear, 2.0 = hard
    };

    MIDISettings& midi() { return midi_; }
    const MIDISettings& midi() const { return midi_; }

    // ============================================
    // Workflow & Behavior
    // ============================================

    struct WorkflowSettings {
        bool snapToGrid{true};
        float gridDivision{1.0f};     ///< 1/4, 1/8, 1/16, etc.
        bool followPlayhead{true};
        bool recordOnPlay{false};
        bool countIn{false};
        int countInBars{1};
        bool autoQuantize{false};
        bool loopEnabled{false};
        float defaultTempo{120.0f};
        int defaultTimeSignature{4};  ///< 4/4, 3/4, etc.
    };

    WorkflowSettings& workflow() { return workflow_; }
    const WorkflowSettings& workflow() const { return workflow_; }

    // ============================================
    // Keyboard Shortcuts
    // ============================================

    struct Shortcut {
        std::string action;
        int keyCode;
        bool ctrl{false};
        bool shift{false};
        bool alt{false};
        bool cmd{false};  ///< macOS Command key

        std::string toString() const;
        static Shortcut fromString(const std::string& str);
    };

    void setShortcut(const std::string& action, const Shortcut& shortcut);
    Shortcut getShortcut(const std::string& action) const;
    std::map<std::string, Shortcut> getAllShortcuts() const { return shortcuts_; }
    void resetShortcutsToDefault();

    // ============================================
    // Window & Layout
    // ============================================

    struct WindowSettings {
        int x{100};
        int y{100};
        int width{1280};
        int height{720};
        bool maximized{false};
        bool fullscreen{false};
        std::string activeLayout{"Default"};
        std::map<std::string, std::string> layoutStates;  ///< Saved layout states
    };

    WindowSettings& window() { return window_; }
    const WindowSettings& window() const { return window_; }

    // ============================================
    // Recent Files & Projects
    // ============================================

    void addRecentFile(const std::string& path);
    std::vector<std::string> getRecentFiles() const { return recentFiles_; }
    void clearRecentFiles();
    void setMaxRecentFiles(int max) { maxRecentFiles_ = max; }

    void addRecentProject(const std::string& path);
    std::vector<std::string> getRecentProjects() const { return recentProjects_; }
    void clearRecentProjects();

    // ============================================
    // Plugin Preferences
    // ============================================

    struct PluginSettings {
        std::vector<std::string> scanPaths;
        std::vector<std::string> blacklist;
        bool autoRescan{false};
        bool bridgeUnsafe{true};  ///< Bridge 32-bit or incompatible plugins
        int bridgeTimeout{30000}; ///< Milliseconds
    };

    PluginSettings& plugins() { return plugins_; }
    const PluginSettings& plugins() const { return plugins_; }

    // ============================================
    // Advanced Settings
    // ============================================

    struct AdvancedSettings {
        bool enableOpenGL{true};
        bool enableMetal{true};  ///< macOS
        bool enableVulkan{false};
        bool enableMultithreading{true};
        int maxThreads{0};  ///< 0 = auto
        bool enableTelemetry{false};
        bool betaFeatures{false};
        int logLevel{2};  ///< 0=off, 1=error, 2=warn, 3=info, 4=debug
    };

    AdvancedSettings& advanced() { return advanced_; }
    const AdvancedSettings& advanced() const { return advanced_; }

    // ============================================
    // Persistence
    // ============================================

    bool load();  ///< Load from disk
    bool save();  ///< Save to disk
    void reset(); ///< Reset to defaults

    std::string getPreferencesPath() const;
    std::string toJSON() const;
    bool fromJSON(const std::string& json);

    // ============================================
    // Change Notifications
    // ============================================

    using ChangeCallback = std::function<void(const std::string& category)>;
    void onChange(ChangeCallback callback);
    void notifyChange(const std::string& category);

    // ============================================
    // Import/Export
    // ============================================

    bool exportToFile(const std::string& path) const;
    bool importFromFile(const std::string& path);

private:
    Preferences();
    ~Preferences() = default;
    Preferences(const Preferences&) = delete;
    Preferences& operator=(const Preferences&) = delete;

    void initializeDefaults();
    void initializeDefaultShortcuts();

    // Settings categories
    AppearanceSettings appearance_;
    AccessibilitySettings accessibility_;
    AudioSettings audio_;
    MIDISettings midi_;
    WorkflowSettings workflow_;
    WindowSettings window_;
    PluginSettings plugins_;
    AdvancedSettings advanced_;

    // Shortcuts
    std::map<std::string, Shortcut> shortcuts_;

    // Recent files
    std::vector<std::string> recentFiles_;
    std::vector<std::string> recentProjects_;
    int maxRecentFiles_{10};

    // Change callbacks
    std::vector<ChangeCallback> callbacks_;
};

/**
 * @brief Keyboard shortcut manager
 */
class ShortcutManager {
public:
    static ShortcutManager& getInstance();

    // Register actions
    void registerAction(const std::string& action, const std::string& description,
                       std::function<void()> callback);
    void unregisterAction(const std::string& action);

    // Execute action by name
    bool executeAction(const std::string& action);

    // Execute action by shortcut
    bool handleKeyPress(int keyCode, bool ctrl, bool shift, bool alt, bool cmd);

    // Get all available actions
    std::vector<std::string> getAvailableActions() const;
    std::string getActionDescription(const std::string& action) const;

    // Default shortcuts (ergonomic, industry-standard)
    static std::map<std::string, Preferences::Shortcut> getDefaultShortcuts();

private:
    ShortcutManager() = default;

    struct Action {
        std::string name;
        std::string description;
        std::function<void()> callback;
    };

    std::map<std::string, Action> actions_;
};

/**
 * @brief Layout manager for customizable workspace layouts
 */
class LayoutManager {
public:
    static LayoutManager& getInstance();

    struct PanelState {
        std::string panelId;
        bool visible{true};
        float x{0.0f};
        float y{0.0f};
        float width{100.0f};
        float height{100.0f};
        bool docked{true};
        std::string dockSide{"left"};  ///< left, right, top, bottom
    };

    struct Layout {
        std::string name;
        std::string description;
        std::vector<PanelState> panels;

        std::string toJSON() const;
        static Layout fromJSON(const std::string& json);
    };

    // Layout management
    void saveLayout(const std::string& name, const Layout& layout);
    Layout loadLayout(const std::string& name) const;
    void deleteLayout(const std::string& name);
    std::vector<std::string> getAvailableLayouts() const;

    void setActiveLayout(const std::string& name);
    std::string getActiveLayout() const { return activeLayout_; }

    // Panel management
    void setPanelState(const std::string& panelId, const PanelState& state);
    PanelState getPanelState(const std::string& panelId) const;

    // Default layouts
    static Layout createMixingLayout();
    static Layout createEditingLayout();
    static Layout createMasteringLayout();
    static Layout createComposingLayout();

    // Callbacks
    using LayoutChangeCallback = std::function<void(const std::string& layoutName)>;
    void onLayoutChange(LayoutChangeCallback callback);

private:
    LayoutManager();

    std::map<std::string, Layout> layouts_;
    std::map<std::string, PanelState> currentPanelStates_;
    std::string activeLayout_;
    std::vector<LayoutChangeCallback> callbacks_;

    void registerDefaultLayouts();
};

} // namespace UI
} // namespace MolinAntro
