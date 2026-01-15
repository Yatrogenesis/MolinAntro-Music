#pragma once

/**
 * @file PluginFormats.h
 * @brief Multi-format plugin support (VST3, AU, CLAP, LV2)
 *
 * Professional plugin hosting supporting:
 * - VST3 (Steinberg)
 * - Audio Unit (Apple)
 * - CLAP (Clever Audio Plugin)
 * - LV2 (Linux Audio)
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 *
 * NOTE: Requires respective SDKs:
 * - VST3: https://github.com/steinbergmedia/vst3sdk
 * - CLAP: https://github.com/free-audio/clap
 * - LV2: https://lv2plug.in/
 */

#include "core/AudioBuffer.h"
#include "midi/MIDIEngine.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <optional>

namespace MolinAntro {
namespace Plugins {

/**
 * @brief Plugin format types
 */
enum class PluginFormat {
    VST3,       // Steinberg VST3
    AU,         // Apple Audio Unit
    CLAP,       // Clever Audio Plugin
    LV2,        // Linux Audio
    Internal    // Built-in plugins
};

/**
 * @brief Plugin category
 */
enum class PluginCategory {
    Unknown,
    Effect,
    Instrument,
    Analyzer,
    Generator,
    Spatial,
    Dynamics,
    EQ,
    Filter,
    Reverb,
    Delay,
    Modulation,
    Distortion,
    Utility,
    Restoration,
    Network,
    Synth,
    Sampler,
    Drum
};

/**
 * @brief Plugin parameter
 */
struct PluginParameter {
    int id;
    std::string name;
    std::string label;
    std::string unit;

    float value = 0.0f;
    float defaultValue = 0.0f;
    float minValue = 0.0f;
    float maxValue = 1.0f;

    int numSteps = 0;  // 0 = continuous
    bool isAutomatable = true;
    bool isBypass = false;

    // For list parameters
    std::vector<std::string> valueStrings;

    // Parameter groups
    std::string groupName;
    int groupId = -1;
};

/**
 * @brief Plugin information
 */
struct PluginInfo {
    std::string id;             // Unique identifier
    std::string name;
    std::string vendor;
    std::string version;
    std::string sdkVersion;

    PluginFormat format;
    PluginCategory category;
    std::vector<PluginCategory> subCategories;

    std::string filePath;

    // Capabilities
    int numAudioInputs = 0;
    int numAudioOutputs = 2;
    int numMIDIInputs = 0;
    int numMIDIOutputs = 0;
    bool hasEditor = false;
    bool canProcessDouble = false;
    bool isSynth = false;

    // Latency
    int latencySamples = 0;

    // Tags for organization
    std::vector<std::string> tags;

    // Rating/favorite
    int rating = 0;
    bool isFavorite = false;
};

/**
 * @brief Plugin preset
 */
struct PluginPreset {
    std::string name;
    std::string category;
    std::string author;
    std::string description;
    std::string filePath;

    // Parameter values
    std::map<int, float> parameters;

    // Raw preset data (for format-specific presets)
    std::vector<uint8_t> data;
};

/**
 * @brief Plugin state for save/restore
 */
struct PluginState {
    std::string pluginId;
    std::map<int, float> parameters;
    std::vector<uint8_t> chunk;  // Opaque plugin state
    bool bypassState = false;
};

/**
 * @brief Abstract plugin instance interface
 */
class IPluginInstance {
public:
    virtual ~IPluginInstance() = default;

    // Info
    virtual PluginInfo getInfo() const = 0;
    virtual PluginFormat getFormat() const = 0;

    // Lifecycle
    virtual bool initialize(int sampleRate, int maxBlockSize) = 0;
    virtual void terminate() = 0;
    virtual bool isInitialized() const = 0;

    // Processing
    virtual void process(Core::AudioBuffer& buffer,
                         std::vector<MIDI::MIDIMessage>& midiIn,
                         std::vector<MIDI::MIDIMessage>& midiOut) = 0;

    virtual void setProcessing(bool active) = 0;
    virtual bool isProcessing() const = 0;

    // Parameters
    virtual int getNumParameters() const = 0;
    virtual PluginParameter getParameter(int index) const = 0;
    virtual float getParameterValue(int index) const = 0;
    virtual void setParameterValue(int index, float value) = 0;
    virtual std::string getParameterDisplay(int index) const = 0;

    // State
    virtual PluginState getState() const = 0;
    virtual bool setState(const PluginState& state) = 0;

    // Presets
    virtual std::vector<PluginPreset> getPresets() const = 0;
    virtual bool loadPreset(const PluginPreset& preset) = 0;
    virtual PluginPreset savePreset(const std::string& name) const = 0;

    // Editor
    virtual bool hasEditor() const = 0;
    virtual void* createEditor(void* parentWindow) = 0;
    virtual void destroyEditor() = 0;
    virtual void getEditorSize(int& width, int& height) const = 0;
    virtual void resizeEditor(int width, int height) = 0;

    // Bypass
    virtual void setBypass(bool bypass) = 0;
    virtual bool isBypassed() const = 0;

    // Latency
    virtual int getLatency() const = 0;
    virtual int getTailLength() const = 0;
};

/**
 * @brief VST3 plugin wrapper
 */
class VST3Plugin : public IPluginInstance {
public:
    VST3Plugin();
    ~VST3Plugin() override;

    static bool load(const std::string& path, VST3Plugin& plugin);

    // IPluginInstance implementation
    PluginInfo getInfo() const override;
    PluginFormat getFormat() const override { return PluginFormat::VST3; }

    bool initialize(int sampleRate, int maxBlockSize) override;
    void terminate() override;
    bool isInitialized() const override;

    void process(Core::AudioBuffer& buffer,
                 std::vector<MIDI::MIDIMessage>& midiIn,
                 std::vector<MIDI::MIDIMessage>& midiOut) override;

    void setProcessing(bool active) override;
    bool isProcessing() const override;

    int getNumParameters() const override;
    PluginParameter getParameter(int index) const override;
    float getParameterValue(int index) const override;
    void setParameterValue(int index, float value) override;
    std::string getParameterDisplay(int index) const override;

    PluginState getState() const override;
    bool setState(const PluginState& state) override;

    std::vector<PluginPreset> getPresets() const override;
    bool loadPreset(const PluginPreset& preset) override;
    PluginPreset savePreset(const std::string& name) const override;

    bool hasEditor() const override;
    void* createEditor(void* parentWindow) override;
    void destroyEditor() override;
    void getEditorSize(int& width, int& height) const override;
    void resizeEditor(int width, int height) override;

    void setBypass(bool bypass) override;
    bool isBypassed() const override;

    int getLatency() const override;
    int getTailLength() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Audio Unit plugin wrapper (macOS only)
 */
class AudioUnitPlugin : public IPluginInstance {
public:
    AudioUnitPlugin();
    ~AudioUnitPlugin() override;

    static bool load(const std::string& identifier, AudioUnitPlugin& plugin);

    // Same interface as VST3Plugin...
    PluginInfo getInfo() const override;
    PluginFormat getFormat() const override { return PluginFormat::AU; }

    bool initialize(int sampleRate, int maxBlockSize) override;
    void terminate() override;
    bool isInitialized() const override { return initialized_; }

    void process(Core::AudioBuffer& buffer,
                 std::vector<MIDI::MIDIMessage>& midiIn,
                 std::vector<MIDI::MIDIMessage>& midiOut) override;

    void setProcessing(bool active) override;
    bool isProcessing() const override { return processing_; }

    int getNumParameters() const override;
    PluginParameter getParameter(int index) const override;
    float getParameterValue(int index) const override;
    void setParameterValue(int index, float value) override;
    std::string getParameterDisplay(int index) const override;

    PluginState getState() const override;
    bool setState(const PluginState& state) override;

    std::vector<PluginPreset> getPresets() const override;
    bool loadPreset(const PluginPreset& preset) override;
    PluginPreset savePreset(const std::string& name) const override;

    bool hasEditor() const override;
    void* createEditor(void* parentWindow) override;
    void destroyEditor() override;
    void getEditorSize(int& width, int& height) const override;
    void resizeEditor(int width, int height) override;

    void setBypass(bool bypass) override;
    bool isBypassed() const override { return bypassed_; }

    int getLatency() const override;
    int getTailLength() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool initialized_ = false;
    bool processing_ = false;
    bool bypassed_ = false;
};

/**
 * @brief CLAP plugin wrapper
 */
class CLAPPlugin : public IPluginInstance {
public:
    CLAPPlugin();
    ~CLAPPlugin() override;

    static bool load(const std::string& path, CLAPPlugin& plugin);

    // Same interface...
    PluginInfo getInfo() const override;
    PluginFormat getFormat() const override { return PluginFormat::CLAP; }

    bool initialize(int sampleRate, int maxBlockSize) override;
    void terminate() override;
    bool isInitialized() const override { return initialized_; }

    void process(Core::AudioBuffer& buffer,
                 std::vector<MIDI::MIDIMessage>& midiIn,
                 std::vector<MIDI::MIDIMessage>& midiOut) override;

    void setProcessing(bool active) override;
    bool isProcessing() const override { return processing_; }

    int getNumParameters() const override;
    PluginParameter getParameter(int index) const override;
    float getParameterValue(int index) const override;
    void setParameterValue(int index, float value) override;
    std::string getParameterDisplay(int index) const override;

    PluginState getState() const override;
    bool setState(const PluginState& state) override;

    std::vector<PluginPreset> getPresets() const override;
    bool loadPreset(const PluginPreset& preset) override;
    PluginPreset savePreset(const std::string& name) const override;

    bool hasEditor() const override;
    void* createEditor(void* parentWindow) override;
    void destroyEditor() override;
    void getEditorSize(int& width, int& height) const override;
    void resizeEditor(int width, int height) override;

    void setBypass(bool bypass) override;
    bool isBypassed() const override { return bypassed_; }

    int getLatency() const override;
    int getTailLength() const override;

    // CLAP-specific features
    bool supportsThreadSafeParameterAccess() const;
    bool supportsNoteExpression() const;
    bool supportsRemoteControls() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool initialized_ = false;
    bool processing_ = false;
    bool bypassed_ = false;
};

/**
 * @brief LV2 plugin wrapper (Linux)
 */
class LV2Plugin : public IPluginInstance {
public:
    LV2Plugin();
    ~LV2Plugin() override;

    static bool load(const std::string& uri, LV2Plugin& plugin);

    PluginInfo getInfo() const override;
    PluginFormat getFormat() const override { return PluginFormat::LV2; }

    bool initialize(int sampleRate, int maxBlockSize) override;
    void terminate() override;
    bool isInitialized() const override { return initialized_; }

    void process(Core::AudioBuffer& buffer,
                 std::vector<MIDI::MIDIMessage>& midiIn,
                 std::vector<MIDI::MIDIMessage>& midiOut) override;

    void setProcessing(bool active) override;
    bool isProcessing() const override { return processing_; }

    int getNumParameters() const override;
    PluginParameter getParameter(int index) const override;
    float getParameterValue(int index) const override;
    void setParameterValue(int index, float value) override;
    std::string getParameterDisplay(int index) const override;

    PluginState getState() const override;
    bool setState(const PluginState& state) override;

    std::vector<PluginPreset> getPresets() const override;
    bool loadPreset(const PluginPreset& preset) override;
    PluginPreset savePreset(const std::string& name) const override;

    bool hasEditor() const override;
    void* createEditor(void* parentWindow) override;
    void destroyEditor() override;
    void getEditorSize(int& width, int& height) const override;
    void resizeEditor(int width, int height) override;

    void setBypass(bool bypass) override;
    bool isBypassed() const override { return bypassed_; }

    int getLatency() const override;
    int getTailLength() const override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool initialized_ = false;
    bool processing_ = false;
    bool bypassed_ = false;
};

/**
 * @brief Plugin manager - scans, loads, and manages plugins
 */
class PluginManager {
public:
    PluginManager();
    ~PluginManager();

    // Scanning
    void addSearchPath(const std::string& path, PluginFormat format);
    void removeSearchPath(const std::string& path);
    std::vector<std::string> getSearchPaths(PluginFormat format) const;

    void scanPlugins(PluginFormat format = PluginFormat::VST3);
    void scanAllFormats();
    void rescanPlugin(const std::string& pluginId);

    bool isScanning() const { return scanning_; }
    float getScanProgress() const { return scanProgress_; }
    std::string getCurrentScanPlugin() const { return currentScanPlugin_; }

    // Plugin list
    std::vector<PluginInfo> getPlugins() const;
    std::vector<PluginInfo> getPluginsByFormat(PluginFormat format) const;
    std::vector<PluginInfo> getPluginsByCategory(PluginCategory category) const;
    std::vector<PluginInfo> searchPlugins(const std::string& query) const;
    PluginInfo getPluginInfo(const std::string& pluginId) const;

    // Loading
    std::unique_ptr<IPluginInstance> createInstance(const std::string& pluginId);
    bool unloadPlugin(const std::string& pluginId);

    // Blacklist (for problematic plugins)
    void blacklistPlugin(const std::string& pluginId);
    void removeFromBlacklist(const std::string& pluginId);
    bool isBlacklisted(const std::string& pluginId) const;
    std::vector<std::string> getBlacklist() const;

    // Favorites and ratings
    void setFavorite(const std::string& pluginId, bool favorite);
    void setRating(const std::string& pluginId, int rating);

    // Preset management
    std::vector<PluginPreset> getPresetsForPlugin(const std::string& pluginId) const;
    void savePreset(const std::string& pluginId, const PluginPreset& preset);
    void deletePreset(const std::string& pluginId, const std::string& presetName);

    // Factory presets
    void scanFactoryPresets(const std::string& pluginId);

    // Callbacks
    using ScanCallback = std::function<void(float progress, const std::string& currentPlugin)>;
    using PluginFoundCallback = std::function<void(const PluginInfo& plugin)>;
    void setScanCallback(ScanCallback cb) { onScan_ = cb; }
    void setPluginFoundCallback(PluginFoundCallback cb) { onPluginFound_ = cb; }

    // Persistence
    void saveDatabase(const std::string& path);
    void loadDatabase(const std::string& path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    bool scanning_ = false;
    float scanProgress_ = 0.0f;
    std::string currentScanPlugin_;

    ScanCallback onScan_;
    PluginFoundCallback onPluginFound_;
};

/**
 * @brief Plugin chain - series of plugins on a track
 */
class PluginChain {
public:
    PluginChain();
    ~PluginChain();

    // Chain management
    int addPlugin(std::unique_ptr<IPluginInstance> plugin);
    void removePlugin(int index);
    void movePlugin(int fromIndex, int toIndex);
    void clearChain();

    int getNumPlugins() const { return static_cast<int>(plugins_.size()); }
    IPluginInstance* getPlugin(int index);

    // Processing
    void process(Core::AudioBuffer& buffer,
                 std::vector<MIDI::MIDIMessage>& midiIn,
                 std::vector<MIDI::MIDIMessage>& midiOut);

    void setActive(bool active) { active_ = active; }
    bool isActive() const { return active_; }

    // Bypass individual plugins
    void setPluginBypass(int index, bool bypass);
    bool isPluginBypassed(int index) const;

    // State
    struct ChainState {
        std::vector<PluginState> pluginStates;
        std::vector<bool> bypassStates;
    };
    ChainState getState() const;
    bool setState(const ChainState& state);

    // Latency
    int getTotalLatency() const;

private:
    std::vector<std::unique_ptr<IPluginInstance>> plugins_;
    bool active_ = true;
};

/**
 * @brief Send/Return routing for auxiliary effects
 */
class SendReturn {
public:
    struct Send {
        int sourceTrackIndex;
        int returnTrackIndex;
        float amount = 0.0f;    // 0-1
        bool preFader = false;
    };

    SendReturn();
    ~SendReturn();

    void addSend(const Send& send);
    void removeSend(int sourceTrack, int returnTrack);
    void setSendAmount(int sourceTrack, int returnTrack, float amount);
    float getSendAmount(int sourceTrack, int returnTrack) const;

    std::vector<Send> getSendsForTrack(int sourceTrack) const;
    std::vector<Send> getReturnsForTrack(int returnTrack) const;

    // Process sends
    void processSends(const std::vector<Core::AudioBuffer*>& trackBuffers,
                      std::vector<Core::AudioBuffer>& returnBuffers);

private:
    std::vector<Send> sends_;
};

} // namespace Plugins
} // namespace MolinAntro
