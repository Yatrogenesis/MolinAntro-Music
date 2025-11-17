#include "plugins/PluginHost.h"
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <cmath>

namespace MolinAntro {
namespace Plugins {

// ============================================================================
// PluginScanner Implementation
// ============================================================================

PluginScanner::PluginScanner() = default;

void PluginScanner::scanDefaultLocations() {
    std::vector<std::string> defaultPaths;

#ifdef __linux__
    defaultPaths.push_back("/usr/lib/vst3");
    defaultPaths.push_back("/usr/local/lib/vst3");
    defaultPaths.push_back(std::string(getenv("HOME")) + "/.vst3");
#elif __APPLE__
    defaultPaths.push_back("/Library/Audio/Plug-Ins/VST3");
    defaultPaths.push_back(std::string(getenv("HOME")) + "/Library/Audio/Plug-Ins/VST3");
#elif _WIN32
    defaultPaths.push_back("C:\\Program Files\\Common Files\\VST3");
    defaultPaths.push_back("C:\\Program Files (x86)\\Common Files\\VST3");
#endif

    for (const auto& path : defaultPaths) {
        if (std::filesystem::exists(path)) {
            scanPath(path);
        }
    }

    std::cout << "[PluginScanner] Found " << plugins_.size() << " plugins\n";
}

void PluginScanner::scanPath(const std::string& path) {
    if (std::filesystem::is_directory(path)) {
        scanDirectory(path);
    } else if (isPluginFile(path)) {
        PluginInfo info;
        info.path = path;
        info.name = std::filesystem::path(path).stem().string();
        info.vendor = "Unknown";
        info.category = "Effect";
        info.numInputs = 2;
        info.numOutputs = 2;
        info.isSynth = false;
        plugins_.push_back(info);
    }
}

void PluginScanner::scanDirectory(const std::string& dir) {
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file() && isPluginFile(entry.path().string())) {
                PluginInfo info;
                info.path = entry.path().string();
                info.name = entry.path().stem().string();
                info.vendor = "Unknown";
                info.category = "Effect";
                info.numInputs = 2;
                info.numOutputs = 2;
                info.isSynth = false;
                plugins_.push_back(info);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[PluginScanner] Error scanning directory: " << e.what() << "\n";
    }
}

bool PluginScanner::isPluginFile(const std::string& path) {
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

#ifdef __linux__
    return ext == ".so" || ext == ".vst3";
#elif __APPLE__
    return ext == ".vst3" || ext == ".component";
#elif _WIN32
    return ext == ".dll" || ext == ".vst3";
#else
    return false;
#endif
}

// ============================================================================
// PluginHost Implementation
// ============================================================================

PluginHost::PluginHost()
    : nextPluginId_(1)
    , sampleRate_(48000.0f)
    , blockSize_(512)
{
}

PluginHost::~PluginHost() = default;

void PluginHost::registerBuiltInPlugins() {
    // Register built-in utility plugins
    int gainId = loadPlugin("builtin://gain");
    int panId = loadPlugin("builtin://pan");

    std::cout << "[PluginHost] Registered built-in plugins: Gain (ID: " << gainId
              << "), Pan (ID: " << panId << ")\n";
}

int PluginHost::loadPlugin(const std::string& path) {
    std::unique_ptr<Plugin> plugin;

    // Check for built-in plugins
    if (path == "builtin://gain") {
        plugin = std::make_unique<GainPlugin>();
    } else if (path == "builtin://pan") {
        plugin = std::make_unique<PanPlugin>();
    } else {
        // For real VST3 loading, would use VST3 SDK here
        // For now, just report that external plugins are not yet fully implemented
        std::cout << "[PluginHost] External plugin loading not yet implemented: " << path << "\n";
        std::cout << "[PluginHost] Hint: Use built-in plugins or wrap DSP effects\n";
        return -1;
    }

    if (plugin && plugin->initialize(sampleRate_, blockSize_)) {
        int id = nextPluginId_++;
        plugins_[id] = std::move(plugin);
        std::cout << "[PluginHost] Loaded plugin: " << plugins_[id]->getName()
                  << " (ID: " << id << ")\n";
        return id;
    }

    return -1;
}

void PluginHost::unloadPlugin(int pluginId) {
    auto it = plugins_.find(pluginId);
    if (it != plugins_.end()) {
        it->second->terminate();
        plugins_.erase(it);

        // Remove from chain
        pluginChain_.erase(
            std::remove(pluginChain_.begin(), pluginChain_.end(), pluginId),
            pluginChain_.end()
        );

        std::cout << "[PluginHost] Unloaded plugin ID: " << pluginId << "\n";
    }
}

Plugin* PluginHost::getPlugin(int pluginId) {
    auto it = plugins_.find(pluginId);
    return (it != plugins_.end()) ? it->second.get() : nullptr;
}

void PluginHost::addPluginToChain(int pluginId) {
    if (plugins_.find(pluginId) != plugins_.end()) {
        pluginChain_.push_back(pluginId);
        std::cout << "[PluginHost] Added plugin " << pluginId << " to chain\n";
    }
}

void PluginHost::removePluginFromChain(int pluginId) {
    pluginChain_.erase(
        std::remove(pluginChain_.begin(), pluginChain_.end(), pluginId),
        pluginChain_.end()
    );
}

void PluginHost::reorderChain(const std::vector<int>& newOrder) {
    pluginChain_ = newOrder;
}

void PluginHost::processPluginChain(Core::AudioBuffer& buffer) {
    for (int pluginId : pluginChain_) {
        auto it = plugins_.find(pluginId);
        if (it != plugins_.end()) {
            it->second->process(buffer);
        }
    }
}

void PluginHost::processPlugin(int pluginId, Core::AudioBuffer& buffer) {
    auto it = plugins_.find(pluginId);
    if (it != plugins_.end()) {
        it->second->process(buffer);
    }
}

void PluginHost::setSampleRate(float sampleRate) {
    sampleRate_ = sampleRate;
    // Reinitialize all loaded plugins
    for (auto& pair : plugins_) {
        pair.second->initialize(sampleRate_, blockSize_);
    }
}

void PluginHost::setBlockSize(int blockSize) {
    blockSize_ = blockSize;
}

// ============================================================================
// GainPlugin Implementation
// ============================================================================

GainPlugin::GainPlugin()
    : gainDB_(0.0f)
    , gainLinear_(1.0f)
{
}

bool GainPlugin::initialize(float /*sampleRate*/, int /*maxBlockSize*/) {
    return true;
}

void GainPlugin::terminate() {
}

void GainPlugin::process(Core::AudioBuffer& buffer) {
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch) {
        buffer.applyGain(ch, gainLinear_);
    }
}

void GainPlugin::processReplacing(float** inputs, float** outputs, int numSamples) {
    for (int ch = 0; ch < 2; ++ch) {
        for (int i = 0; i < numSamples; ++i) {
            outputs[ch][i] = inputs[ch][i] * gainLinear_;
        }
    }
}

PluginParameter GainPlugin::getParameterInfo(int index) const {
    if (index == 0) {
        PluginParameter param;
        param.id = "gain";
        param.name = "Gain";
        param.value = (gainDB_ + 24.0f) / 48.0f; // Normalize -24dB to +24dB -> 0.0 to 1.0
        param.defaultValue = 0.5f; // 0dB
        param.minValue = 0.0f;
        param.maxValue = 1.0f;
        param.unit = "dB";
        param.isAutomatable = true;
        return param;
    }
    return {};
}

void GainPlugin::setParameterValue(int index, float value) {
    if (index == 0) {
        gainDB_ = value * 48.0f - 24.0f; // Denormalize to -24dB to +24dB
        updateGainLinear();
    }
}

float GainPlugin::getParameterValue(int index) const {
    if (index == 0) {
        return (gainDB_ + 24.0f) / 48.0f;
    }
    return 0.0f;
}

void GainPlugin::updateGainLinear() {
    gainLinear_ = std::pow(10.0f, gainDB_ / 20.0f);
}

// ============================================================================
// PanPlugin Implementation
// ============================================================================

PanPlugin::PanPlugin()
    : pan_(0.0f)
    , leftGain_(1.0f)
    , rightGain_(1.0f)
{
}

bool PanPlugin::initialize(float /*sampleRate*/, int /*maxBlockSize*/) {
    return true;
}

void PanPlugin::terminate() {
}

void PanPlugin::process(Core::AudioBuffer& buffer) {
    if (buffer.getNumChannels() < 2) {
        return;
    }

    float* left = buffer.getWritePointer(0);
    float* right = buffer.getWritePointer(1);

    for (int i = 0; i < buffer.getNumSamples(); ++i) {
        float mono = (left[i] + right[i]) * 0.5f;
        left[i] = mono * leftGain_;
        right[i] = mono * rightGain_;
    }
}

void PanPlugin::processReplacing(float** inputs, float** outputs, int numSamples) {
    for (int i = 0; i < numSamples; ++i) {
        float mono = (inputs[0][i] + inputs[1][i]) * 0.5f;
        outputs[0][i] = mono * leftGain_;
        outputs[1][i] = mono * rightGain_;
    }
}

PluginParameter PanPlugin::getParameterInfo(int index) const {
    if (index == 0) {
        PluginParameter param;
        param.id = "pan";
        param.name = "Pan";
        param.value = (pan_ + 1.0f) / 2.0f; // Normalize -1.0 to 1.0 -> 0.0 to 1.0
        param.defaultValue = 0.5f; // Center
        param.minValue = 0.0f;
        param.maxValue = 1.0f;
        param.unit = "";
        param.isAutomatable = true;
        return param;
    }
    return {};
}

void PanPlugin::setParameterValue(int index, float value) {
    if (index == 0) {
        pan_ = value * 2.0f - 1.0f; // Denormalize to -1.0 to 1.0
        updatePanGains();
    }
}

float PanPlugin::getParameterValue(int index) const {
    if (index == 0) {
        return (pan_ + 1.0f) / 2.0f;
    }
    return 0.0f;
}

void PanPlugin::updatePanGains() {
    // Constant power panning
    const float piOver4 = M_PI / 4.0f;
    float angle = pan_ * piOver4; // -pi/4 to +pi/4

    leftGain_ = std::cos(angle);
    rightGain_ = std::sin(angle);
}

} // namespace Plugins
} // namespace MolinAntro
