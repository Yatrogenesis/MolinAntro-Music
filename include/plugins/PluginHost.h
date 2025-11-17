#pragma once

#include "core/AudioBuffer.h"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace MolinAntro {
namespace Plugins {

/**
 * Plugin Parameter
 */
struct PluginParameter {
    std::string id;
    std::string name;
    float value;          // Normalized 0.0 - 1.0
    float defaultValue;
    float minValue;
    float maxValue;
    std::string unit;
    bool isAutomatable;
};

/**
 * Plugin Interface (simplified VST3-like)
 */
class Plugin {
public:
    virtual ~Plugin() = default;

    // Lifecycle
    virtual bool initialize(float sampleRate, int maxBlockSize) = 0;
    virtual void terminate() = 0;

    // Processing
    virtual void process(Core::AudioBuffer& buffer) = 0;
    virtual void processReplacing(float** inputs, float** outputs, int numSamples) = 0;

    // Parameters
    virtual int getParameterCount() const = 0;
    virtual PluginParameter getParameterInfo(int index) const = 0;
    virtual void setParameterValue(int index, float value) = 0;
    virtual float getParameterValue(int index) const = 0;

    // Info
    virtual std::string getName() const = 0;
    virtual std::string getVendor() const = 0;
    virtual std::string getVersion() const = 0;
    virtual int getNumInputs() const = 0;
    virtual int getNumOutputs() const = 0;

    // MIDI (optional)
    virtual bool acceptsMidi() const { return false; }
    virtual bool producesMidi() const { return false; }

    // State management
    virtual bool saveState(std::vector<uint8_t>& /*data*/) { return false; }
    virtual bool loadState(const std::vector<uint8_t>& /*data*/) { return false; }
};

/**
 * Plugin Scanner - scans system for VST3 plugins
 */
class PluginScanner {
public:
    struct PluginInfo {
        std::string path;
        std::string name;
        std::string vendor;
        std::string category;
        int numInputs;
        int numOutputs;
        bool isSynth;
    };

    PluginScanner();

    void scanDefaultLocations();
    void scanPath(const std::string& path);
    std::vector<PluginInfo> getScannedPlugins() const { return plugins_; }

private:
    std::vector<PluginInfo> plugins_;
    void scanDirectory(const std::string& dir);
    bool isPluginFile(const std::string& path);
};

/**
 * Plugin Host - loads and manages plugins
 */
class PluginHost {
public:
    PluginHost();
    ~PluginHost();

    // Plugin management
    int loadPlugin(const std::string& path);
    void unloadPlugin(int pluginId);
    Plugin* getPlugin(int pluginId);

    // Audio processing
    void processPluginChain(Core::AudioBuffer& buffer);
    void processPlugin(int pluginId, Core::AudioBuffer& buffer);

    // Chain management
    void addPluginToChain(int pluginId);
    void removePluginFromChain(int pluginId);
    void reorderChain(const std::vector<int>& newOrder);
    std::vector<int> getPluginChain() const { return pluginChain_; }

    // Configuration
    void setSampleRate(float sampleRate);
    void setBlockSize(int blockSize);

    // Built-in plugins
    void registerBuiltInPlugins();

private:
    std::map<int, std::unique_ptr<Plugin>> plugins_;
    std::vector<int> pluginChain_;
    int nextPluginId_;
    float sampleRate_;
    int blockSize_;
};

/**
 * Built-in Utility Plugins
 */
class GainPlugin : public Plugin {
public:
    GainPlugin();

    bool initialize(float sampleRate, int maxBlockSize) override;
    void terminate() override;
    void process(Core::AudioBuffer& buffer) override;
    void processReplacing(float** inputs, float** outputs, int numSamples) override;

    int getParameterCount() const override { return 1; }
    PluginParameter getParameterInfo(int index) const override;
    void setParameterValue(int index, float value) override;
    float getParameterValue(int index) const override;

    std::string getName() const override { return "Gain Utility"; }
    std::string getVendor() const override { return "MolinAntro"; }
    std::string getVersion() const override { return "1.0.0"; }
    int getNumInputs() const override { return 2; }
    int getNumOutputs() const override { return 2; }

private:
    float gainDB_;
    float gainLinear_;
    void updateGainLinear();
};

class PanPlugin : public Plugin {
public:
    PanPlugin();

    bool initialize(float sampleRate, int maxBlockSize) override;
    void terminate() override;
    void process(Core::AudioBuffer& buffer) override;
    void processReplacing(float** inputs, float** outputs, int numSamples) override;

    int getParameterCount() const override { return 1; }
    PluginParameter getParameterInfo(int index) const override;
    void setParameterValue(int index, float value) override;
    float getParameterValue(int index) const override;

    std::string getName() const override { return "Pan Utility"; }
    std::string getVendor() const override { return "MolinAntro"; }
    std::string getVersion() const override { return "1.0.0"; }
    int getNumInputs() const override { return 2; }
    int getNumOutputs() const override { return 2; }

private:
    float pan_; // -1.0 (left) to 1.0 (right)
    float leftGain_;
    float rightGain_;
    void updatePanGains();
};

/**
 * Plugin Wrapper for DSP effects (adapts existing effects to plugin interface)
 */
template<typename EffectType>
class EffectPluginWrapper : public Plugin {
public:
    EffectPluginWrapper(const std::string& name) : name_(name), effect_() {}

    bool initialize(float sampleRate, int /*maxBlockSize*/) override {
        effect_.prepare(sampleRate);
        return true;
    }

    void terminate() override {}

    void process(Core::AudioBuffer& buffer) override {
        effect_.process(buffer);
    }

    void processReplacing(float** inputs, float** outputs, int numSamples) override {
        Core::AudioBuffer buffer(2, numSamples);
        for (int ch = 0; ch < 2; ++ch) {
            std::copy(inputs[ch], inputs[ch] + numSamples, buffer.getWritePointer(ch));
        }
        effect_.process(buffer);
        for (int ch = 0; ch < 2; ++ch) {
            std::copy(buffer.getReadPointer(ch), buffer.getReadPointer(ch) + numSamples, outputs[ch]);
        }
    }

    int getParameterCount() const override { return 0; } // Simplified
    PluginParameter getParameterInfo(int /*index*/) const override { return {}; }
    void setParameterValue(int /*index*/, float /*value*/) override {}
    float getParameterValue(int /*index*/) const override { return 0.0f; }

    std::string getName() const override { return name_; }
    std::string getVendor() const override { return "MolinAntro"; }
    std::string getVersion() const override { return "1.0.0"; }
    int getNumInputs() const override { return 2; }
    int getNumOutputs() const override { return 2; }

private:
    std::string name_;
    EffectType effect_;
};

} // namespace Plugins
} // namespace MolinAntro
