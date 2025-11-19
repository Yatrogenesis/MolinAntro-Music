// Preferences.cpp - User Preferences and Settings System
// MolinAntro DAW ACME Edition v3.0.0

#include "ui/Preferences.h"
#include "ui/Theme.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace MolinAntro {
namespace UI {

// ============================================================================
// Shortcut Implementation
// ============================================================================

std::string Preferences::Shortcut::toString() const {
    std::stringstream ss;
    if (ctrl) ss << "Ctrl+";
    if (shift) ss << "Shift+";
    if (alt) ss << "Alt+";
    if (cmd) ss << "Cmd+";
    ss << keyCode;
    return ss.str();
}

Preferences::Shortcut Preferences::Shortcut::fromString(const std::string& str) {
    Shortcut shortcut;
    // Simple parser - in production, use proper string parsing
    shortcut.ctrl = (str.find("Ctrl") != std::string::npos);
    shortcut.shift = (str.find("Shift") != std::string::npos);
    shortcut.alt = (str.find("Alt") != std::string::npos);
    shortcut.cmd = (str.find("Cmd") != std::string::npos);
    // Parse key code from end of string
    return shortcut;
}

// ============================================================================
// Preferences Implementation
// ============================================================================

Preferences::Preferences() {
    initializeDefaults();
    initializeDefaultShortcuts();
}

Preferences& Preferences::getInstance() {
    static Preferences instance;
    return instance;
}

void Preferences::initializeDefaults() {
    // Appearance defaults
    appearance_.themeName = "Dark";
    appearance_.followSystemTheme = false;
    appearance_.uiScale = 1.0f;
    appearance_.animationsEnabled = true;
    appearance_.animationSpeed = 1.0f;
    appearance_.showTooltips = true;
    appearance_.tooltipDelay = 0.5f;

    // Accessibility defaults
    accessibility_.highContrast = false;
    accessibility_.reducedMotion = false;
    accessibility_.largeText = false;
    accessibility_.textScale = 1.0f;
    accessibility_.screenReaderMode = false;
    accessibility_.dyslexiaMode = false;
    accessibility_.colorblindMode = Accessibility::ColorblindMode::None;
    accessibility_.keyboardNavigation = true;
    accessibility_.focusIndicators = true;
    accessibility_.wcagLevel = Accessibility::WCAGLevel::AAA;

    // Audio defaults
    audio_.sampleRate = 48000;
    audio_.bufferSize = 512;
    audio_.inputChannels = 2;
    audio_.outputChannels = 2;
    audio_.lowLatencyMode = false;
    audio_.masterVolume = 0.8f;
    audio_.autoSave = true;
    audio_.autoSaveInterval = 300;

    // MIDI defaults
    midi_.midiThru = false;
    midi_.midiChannel = 1;
    midi_.velocitySensitive = true;
    midi_.velocityCurve = 1.0f;

    // Workflow defaults
    workflow_.snapToGrid = true;
    workflow_.gridDivision = 1.0f;
    workflow_.followPlayhead = true;
    workflow_.recordOnPlay = false;
    workflow_.countIn = false;
    workflow_.countInBars = 1;
    workflow_.autoQuantize = false;
    workflow_.loopEnabled = false;
    workflow_.defaultTempo = 120.0f;
    workflow_.defaultTimeSignature = 4;

    // Window defaults
    window_.x = 100;
    window_.y = 100;
    window_.width = 1280;
    window_.height = 720;
    window_.maximized = false;
    window_.fullscreen = false;
    window_.activeLayout = "Default";

    // Plugin defaults
    plugins_.autoRescan = false;
    plugins_.bridgeUnsafe = true;
    plugins_.bridgeTimeout = 30000;

    // Advanced defaults
    advanced_.enableOpenGL = true;
    advanced_.enableMetal = true;
    advanced_.enableVulkan = false;
    advanced_.enableMultithreading = true;
    advanced_.maxThreads = 0;  // Auto
    advanced_.enableTelemetry = false;
    advanced_.betaFeatures = false;
    advanced_.logLevel = 2;  // Warn

    // Recent files
    maxRecentFiles_ = 10;
}

void Preferences::initializeDefaultShortcuts() {
    shortcuts_ = ShortcutManager::getDefaultShortcuts();
}

void Preferences::setShortcut(const std::string& action, const Shortcut& shortcut) {
    shortcuts_[action] = shortcut;
    notifyChange("shortcuts");
}

Preferences::Shortcut Preferences::getShortcut(const std::string& action) const {
    auto it = shortcuts_.find(action);
    if (it != shortcuts_.end()) {
        return it->second;
    }
    return Shortcut();
}

void Preferences::resetShortcutsToDefault() {
    initializeDefaultShortcuts();
    notifyChange("shortcuts");
}

void Preferences::addRecentFile(const std::string& path) {
    // Remove if already exists
    recentFiles_.erase(
        std::remove(recentFiles_.begin(), recentFiles_.end(), path),
        recentFiles_.end()
    );

    // Add to front
    recentFiles_.insert(recentFiles_.begin(), path);

    // Trim to max size
    if (static_cast<int>(recentFiles_.size()) > maxRecentFiles_) {
        recentFiles_.resize(maxRecentFiles_);
    }

    notifyChange("recent_files");
}

void Preferences::clearRecentFiles() {
    recentFiles_.clear();
    notifyChange("recent_files");
}

void Preferences::addRecentProject(const std::string& path) {
    recentProjects_.erase(
        std::remove(recentProjects_.begin(), recentProjects_.end(), path),
        recentProjects_.end()
    );

    recentProjects_.insert(recentProjects_.begin(), path);

    if (static_cast<int>(recentProjects_.size()) > maxRecentFiles_) {
        recentProjects_.resize(maxRecentFiles_);
    }

    notifyChange("recent_projects");
}

void Preferences::clearRecentProjects() {
    recentProjects_.clear();
    notifyChange("recent_projects");
}

bool Preferences::load() {
    std::string path = getPreferencesPath();
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    return fromJSON(buffer.str());
}

bool Preferences::save() {
    std::string path = getPreferencesPath();
    std::string json = toJSON();

    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << json;
    file.close();

    return true;
}

void Preferences::reset() {
    initializeDefaults();
    initializeDefaultShortcuts();
    recentFiles_.clear();
    recentProjects_.clear();
    notifyChange("all");
}

std::string Preferences::getPreferencesPath() const {
    #ifdef _WIN32
        const char* appdata = std::getenv("APPDATA");
        return std::string(appdata ? appdata : "") + "/MolinAntro/preferences.json";
    #elif __APPLE__
        const char* home = std::getenv("HOME");
        return std::string(home ? home : "") +
               "/Library/Application Support/MolinAntro/preferences.json";
    #else
        const char* home = std::getenv("HOME");
        return std::string(home ? home : "") + "/.config/molinantro/preferences.json";
    #endif
}

std::string Preferences::toJSON() const {
    // Simplified JSON export - in production, use a proper JSON library
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"version\": \"3.0.0\",\n";
    ss << "  \"appearance\": {\n";
    ss << "    \"theme\": \"" << appearance_.themeName << "\",\n";
    ss << "    \"uiScale\": " << appearance_.uiScale << ",\n";
    ss << "    \"animationsEnabled\": " << (appearance_.animationsEnabled ? "true" : "false") << "\n";
    ss << "  },\n";
    ss << "  \"audio\": {\n";
    ss << "    \"sampleRate\": " << audio_.sampleRate << ",\n";
    ss << "    \"bufferSize\": " << audio_.bufferSize << "\n";
    ss << "  }\n";
    ss << "}\n";
    return ss.str();
}

bool Preferences::fromJSON(const std::string& /*json*/) {
    // TODO: Implement JSON parsing
    // For now, return false to use defaults
    return false;
}

void Preferences::onChange(ChangeCallback callback) {
    callbacks_.push_back(callback);
}

void Preferences::notifyChange(const std::string& category) {
    for (auto& callback : callbacks_) {
        callback(category);
    }

    // Auto-save on changes
    save();
}

bool Preferences::exportToFile(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }

    file << toJSON();
    file.close();

    return true;
}

bool Preferences::importFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        return false;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    file.close();

    bool success = fromJSON(buffer.str());
    if (success) {
        notifyChange("all");
    }

    return success;
}

// ============================================================================
// ShortcutManager Implementation
// ============================================================================

ShortcutManager& ShortcutManager::getInstance() {
    static ShortcutManager instance;
    return instance;
}

void ShortcutManager::registerAction(const std::string& action,
                                    const std::string& description,
                                    std::function<void()> callback) {
    actions_[action] = {action, description, callback};
}

void ShortcutManager::unregisterAction(const std::string& action) {
    actions_.erase(action);
}

bool ShortcutManager::executeAction(const std::string& action) {
    auto it = actions_.find(action);
    if (it != actions_.end() && it->second.callback) {
        it->second.callback();
        return true;
    }
    return false;
}

bool ShortcutManager::handleKeyPress(int keyCode, bool ctrl, bool shift,
                                    bool alt, bool cmd) {
    auto& prefs = Preferences::getInstance();
    auto shortcuts = prefs.getAllShortcuts();

    for (const auto& pair : shortcuts) {
        const auto& shortcut = pair.second;
        if (shortcut.keyCode == keyCode &&
            shortcut.ctrl == ctrl &&
            shortcut.shift == shift &&
            shortcut.alt == alt &&
            shortcut.cmd == cmd) {
            return executeAction(pair.first);
        }
    }

    return false;
}

std::vector<std::string> ShortcutManager::getAvailableActions() const {
    std::vector<std::string> actions;
    for (const auto& pair : actions_) {
        actions.push_back(pair.first);
    }
    return actions;
}

std::string ShortcutManager::getActionDescription(const std::string& action) const {
    auto it = actions_.find(action);
    return (it != actions_.end()) ? it->second.description : "";
}

std::map<std::string, Preferences::Shortcut> ShortcutManager::getDefaultShortcuts() {
    std::map<std::string, Preferences::Shortcut> shortcuts;

    // Transport controls
    shortcuts["transport.play"] = {"transport.play", 32, false, false, false, false};  // Space
    shortcuts["transport.stop"] = {"transport.stop", 0, false, false, false, false};   // 0 key
    shortcuts["transport.record"] = {"transport.record", 'R', false, false, false, false};
    shortcuts["transport.loop"] = {"transport.loop", 'L', false, false, false, false};

    // Edit operations
    shortcuts["edit.undo"] = {"edit.undo", 'Z', true, false, false, false};   // Ctrl+Z
    shortcuts["edit.redo"] = {"edit.redo", 'Y', true, false, false, false};   // Ctrl+Y
    shortcuts["edit.cut"] = {"edit.cut", 'X', true, false, false, false};    // Ctrl+X
    shortcuts["edit.copy"] = {"edit.copy", 'C', true, false, false, false};   // Ctrl+C
    shortcuts["edit.paste"] = {"edit.paste", 'V', true, false, false, false};  // Ctrl+V
    shortcuts["edit.delete"] = {"edit.delete", 127, false, false, false, false}; // Delete
    shortcuts["edit.selectAll"] = {"edit.selectAll", 'A', true, false, false, false}; // Ctrl+A

    // File operations
    shortcuts["file.new"] = {"file.new", 'N', true, false, false, false};    // Ctrl+N
    shortcuts["file.open"] = {"file.open", 'O', true, false, false, false};   // Ctrl+O
    shortcuts["file.save"] = {"file.save", 'S', true, false, false, false};   // Ctrl+S
    shortcuts["file.saveAs"] = {"file.saveAs", 'S', true, true, false, false}; // Ctrl+Shift+S

    // View controls
    shortcuts["view.zoomIn"] = {"view.zoomIn", '=', true, false, false, false};   // Ctrl+=
    shortcuts["view.zoomOut"] = {"view.zoomOut", '-', true, false, false, false};  // Ctrl+-
    shortcuts["view.zoomFit"] = {"view.zoomFit", '0', true, false, false, false};  // Ctrl+0

    // Track operations
    shortcuts["track.new"] = {"track.new", 'T', true, false, false, false};    // Ctrl+T
    shortcuts["track.duplicate"] = {"track.duplicate", 'D', true, false, false, false}; // Ctrl+D
    shortcuts["track.mute"] = {"track.mute", 'M', false, false, false, false};  // M
    shortcuts["track.solo"] = {"track.solo", 'S', false, false, false, false};  // S
    shortcuts["track.arm"] = {"track.arm", 'A', false, false, false, false};   // A

    // Mixing
    shortcuts["mix.normalize"] = {"mix.normalize", 'N', true, true, false, false}; // Ctrl+Shift+N
    shortcuts["mix.reverse"] = {"mix.reverse", 'R', true, true, false, false};  // Ctrl+Shift+R
    shortcuts["mix.fadeIn"] = {"mix.fadeIn", 'I', true, false, false, false};  // Ctrl+I
    shortcuts["mix.fadeOut"] = {"mix.fadeOut", 'O', true, true, false, false};  // Ctrl+Shift+O

    return shortcuts;
}

// ============================================================================
// LayoutManager Implementation
// ============================================================================

LayoutManager::LayoutManager() {
    registerDefaultLayouts();
}

LayoutManager& LayoutManager::getInstance() {
    static LayoutManager instance;
    return instance;
}

void LayoutManager::saveLayout(const std::string& name, const Layout& layout) {
    layouts_[name] = layout;

    // Save to disk
    auto& prefs = Preferences::getInstance();
    prefs.window().layoutStates[name] = layout.toJSON();
    prefs.save();
}

LayoutManager::Layout LayoutManager::loadLayout(const std::string& name) const {
    auto it = layouts_.find(name);
    if (it != layouts_.end()) {
        return it->second;
    }

    // Try to load from preferences
    auto& prefs = Preferences::getInstance();
    auto stateIt = prefs.window().layoutStates.find(name);
    if (stateIt != prefs.window().layoutStates.end()) {
        return Layout::fromJSON(stateIt->second);
    }

    return Layout();
}

void LayoutManager::deleteLayout(const std::string& name) {
    layouts_.erase(name);

    auto& prefs = Preferences::getInstance();
    prefs.window().layoutStates.erase(name);
    prefs.save();
}

std::vector<std::string> LayoutManager::getAvailableLayouts() const {
    std::vector<std::string> names;
    for (const auto& pair : layouts_) {
        names.push_back(pair.first);
    }
    return names;
}

void LayoutManager::setActiveLayout(const std::string& name) {
    activeLayout_ = name;

    auto layout = loadLayout(name);

    // Apply panel states
    for (const auto& panel : layout.panels) {
        currentPanelStates_[panel.panelId] = panel;
    }

    // Notify callbacks
    for (auto& callback : callbacks_) {
        callback(name);
    }

    // Save to preferences
    auto& prefs = Preferences::getInstance();
    prefs.window().activeLayout = name;
    prefs.save();
}

void LayoutManager::setPanelState(const std::string& panelId, const PanelState& state) {
    currentPanelStates_[panelId] = state;
}

LayoutManager::PanelState LayoutManager::getPanelState(const std::string& panelId) const {
    auto it = currentPanelStates_.find(panelId);
    if (it != currentPanelStates_.end()) {
        return it->second;
    }
    return PanelState();
}

void LayoutManager::onLayoutChange(LayoutChangeCallback callback) {
    callbacks_.push_back(callback);
}

void LayoutManager::registerDefaultLayouts() {
    layouts_["Default"] = createMixingLayout();
    layouts_["Mixing"] = createMixingLayout();
    layouts_["Editing"] = createEditingLayout();
    layouts_["Mastering"] = createMasteringLayout();
    layouts_["Composing"] = createComposingLayout();
}

LayoutManager::Layout LayoutManager::createMixingLayout() {
    Layout layout;
    layout.name = "Mixing";
    layout.description = "Optimized for mixing with mixer, meters, and effects";

    // Mixer panel (left)
    layout.panels.push_back({"mixer", true, 0, 0, 200, 600, true, "left"});

    // Main view (center)
    layout.panels.push_back({"tracks", true, 200, 0, 800, 600, true, "center"});

    // Effects/Inspector (right)
    layout.panels.push_back({"inspector", true, 1000, 0, 280, 600, true, "right"});

    // Transport (bottom)
    layout.panels.push_back({"transport", true, 0, 600, 1280, 120, true, "bottom"});

    return layout;
}

LayoutManager::Layout LayoutManager::createEditingLayout() {
    Layout layout;
    layout.name = "Editing";
    layout.description = "Optimized for editing with waveform and MIDI editors";

    layout.panels.push_back({"browser", true, 0, 0, 200, 400, true, "left"});
    layout.panels.push_back({"editor", true, 200, 0, 880, 600, true, "center"});
    layout.panels.push_back({"tools", true, 1080, 0, 200, 400, true, "right"});
    layout.panels.push_back({"transport", true, 0, 600, 1280, 120, true, "bottom"});

    return layout;
}

LayoutManager::Layout LayoutManager::createMasteringLayout() {
    Layout layout;
    layout.name = "Mastering";
    layout.description = "Optimized for mastering with spectrum and metering";

    layout.panels.push_back({"waveform", true, 0, 0, 1280, 300, true, "top"});
    layout.panels.push_back({"spectrum", true, 0, 300, 640, 300, true, "left"});
    layout.panels.push_back({"meters", true, 640, 300, 640, 300, true, "right"});
    layout.panels.push_back({"transport", true, 0, 600, 1280, 120, true, "bottom"});

    return layout;
}

LayoutManager::Layout LayoutManager::createComposingLayout() {
    Layout layout;
    layout.name = "Composing";
    layout.description = "Optimized for composing with MIDI editor and piano roll";

    layout.panels.push_back({"instruments", true, 0, 0, 250, 600, true, "left"});
    layout.panels.push_back({"pianoroll", true, 250, 0, 780, 400, true, "center"});
    layout.panels.push_back({"mixer", true, 1030, 0, 250, 600, true, "right"});
    layout.panels.push_back({"arrangement", true, 250, 400, 780, 200, true, "center_bottom"});
    layout.panels.push_back({"transport", true, 0, 600, 1280, 120, true, "bottom"});

    return layout;
}

std::string LayoutManager::Layout::toJSON() const {
    // Simplified JSON - in production use proper JSON library
    std::stringstream ss;
    ss << "{\"name\":\"" << name << "\",\"description\":\"" << description << "\"}";
    return ss.str();
}

LayoutManager::Layout LayoutManager::Layout::fromJSON(const std::string& /*json*/) {
    // TODO: Implement JSON parsing
    return Layout();
}

} // namespace UI
} // namespace MolinAntro
