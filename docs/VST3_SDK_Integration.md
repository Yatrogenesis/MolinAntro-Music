# VST3 SDK Integration Guide

## Overview

This document describes how to integrate the official Steinberg VST3 SDK with MolinAntro DAW. The plugin hosting infrastructure is already 100% ready - only VST3 SDK linking is needed.

## Prerequisites

### Download VST3 SDK

```bash
# Clone VST3 SDK
cd /opt
sudo git clone --recursive https://github.com/steinbergmedia/vst3sdk.git
cd vst3sdk
git checkout v3.7.7_build_19
```

### Build VST3 SDK

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
sudo cmake --install .
```

## Project Integration

### Update CMakeLists.txt

Add to root `CMakeLists.txt`:

```cmake
# VST3 SDK Support
option(BUILD_VST3_SUPPORT "Build with VST3 SDK support" OFF)

if(BUILD_VST3_SUPPORT)
    set(VST3_SDK_PATH "/opt/vst3sdk" CACHE PATH "Path to VST3 SDK")

    if(EXISTS "${VST3_SDK_PATH}")
        add_subdirectory(${VST3_SDK_PATH} ${CMAKE_BINARY_DIR}/vst3sdk)

        # VST3 SDK targets
        set(VST3_LIBS
            sdk
            base
            pluginterfaces
        )

        message(STATUS "VST3 SDK found: ${VST3_SDK_PATH}")
    else()
        message(WARNING "VST3 SDK not found at: ${VST3_SDK_PATH}")
    endif()
endif()
```

Update `src/plugins/CMakeLists.txt`:

```cmake
set(PLUGINS_SOURCES
    PluginHost.cpp
)

if(BUILD_VST3_SUPPORT)
    list(APPEND PLUGINS_SOURCES
        VST3Loader.cpp
        VST3Wrapper.cpp
    )
endif()

add_library(MolinAntro_Plugins STATIC ${PLUGINS_SOURCES})

target_include_directories(MolinAntro_Plugins
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(MolinAntro_Plugins
    PUBLIC
        MolinAntro_Core
)

if(BUILD_VST3_SUPPORT)
    target_link_libraries(MolinAntro_Plugins
        PUBLIC
            ${VST3_LIBS}
    )
    target_compile_definitions(MolinAntro_Plugins
        PUBLIC
            MOLINANTRO_VST3_SUPPORT=1
    )
endif()
```

## VST3 Loader Implementation

### Header: `include/plugins/VST3Loader.h`

```cpp
#pragma once

#ifdef MOLINANTRO_VST3_SUPPORT

#include "plugins/PluginHost.h"
#include <pluginterfaces/vst/ivstaudioprocessor.h>
#include <pluginterfaces/vst/ivstcomponent.h>
#include <pluginterfaces/vst/ivsteditcontroller.h>
#include <public.sdk/source/vst/hosting/module.h>

namespace MolinAntro {
namespace Plugins {

/**
 * VST3 Plugin Wrapper
 * Wraps Steinberg VST3 plugin into MolinAntro Plugin interface
 */
class VST3Plugin : public Plugin {
public:
    VST3Plugin();
    ~VST3Plugin() override;

    // Load VST3 from file
    bool loadFromFile(const std::string& path);

    // Plugin interface
    bool initialize(float sampleRate, int maxBlockSize) override;
    void terminate() override;
    void process(Core::AudioBuffer& buffer) override;
    void processReplacing(float** inputs, float** outputs, int numSamples) override;

    int getParameterCount() const override;
    PluginParameter getParameterInfo(int index) const override;
    void setParameterValue(int index, float value) override;
    float getParameterValue(int index) const override;

    std::string getName() const override;
    std::string getVendor() const override;
    std::string getVersion() const override;
    int getNumInputs() const override;
    int getNumOutputs() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;

    bool saveState(std::vector<uint8_t>& data) override;
    bool loadState(const std::vector<uint8_t>& data) override;

private:
    using namespace Steinberg;
    using namespace Steinberg::Vst;

    VST3::Hosting::Module::Ptr module_;
    IComponent* component_;
    IAudioProcessor* processor_;
    IEditController* controller_;

    ProcessSetup processSetup_;
    ProcessData processData_;

    std::vector<AudioBusBuffers> inputBuses_;
    std::vector<AudioBusBuffers> outputBuses_;

    std::string name_;
    std::string vendor_;
    int numInputs_;
    int numOutputs_;

    bool setupProcessing();
    bool activateBuses();
};

/**
 * VST3 Plugin Scanner with Real VST3 Support
 */
class VST3Scanner {
public:
    struct VST3PluginInfo {
        std::string path;
        std::string name;
        std::string vendor;
        std::string version;
        std::string category;
        std::string subCategories;
        int numInputs;
        int numOutputs;
        bool isSynth;
        bool hasEditor;
    };

    VST3Scanner();

    // Scan default VST3 locations
    void scanDefaultLocations();

    // Scan specific path
    void scanPath(const std::string& path);

    // Get all scanned plugins
    std::vector<VST3PluginInfo> getPlugins() const { return plugins_; }

    // Get plugins by category
    std::vector<VST3PluginInfo> getPluginsByCategory(const std::string& category) const;

private:
    std::vector<VST3PluginInfo> plugins_;

    void scanFile(const std::string& path);
    bool extractPluginInfo(const std::string& path, VST3PluginInfo& info);
};

/**
 * Enhanced Plugin Host with VST3 Support
 */
class VST3PluginHost : public PluginHost {
public:
    VST3PluginHost();
    ~VST3PluginHost() override;

    // Load VST3 plugin
    int loadVST3Plugin(const std::string& path);

    // Scan for plugins
    void scanForVST3Plugins();

    // Get available VST3 plugins
    std::vector<VST3Scanner::VST3PluginInfo> getAvailableVST3Plugins() const;

private:
    std::unique_ptr<VST3Scanner> scanner_;
};

} // namespace Plugins
} // namespace MolinAntro

#endif // MOLINANTRO_VST3_SUPPORT
```

### Implementation: `src/plugins/VST3Loader.cpp`

```cpp
#include "plugins/VST3Loader.h"

#ifdef MOLINANTRO_VST3_SUPPORT

#include <iostream>
#include <public.sdk/source/vst/hosting/hostclasses.h>
#include <public.sdk/source/vst/utility/stringconvert.h>

namespace MolinAntro {
namespace Plugins {

using namespace Steinberg;
using namespace Steinberg::Vst;

// ============================================================================
// VST3Plugin Implementation
// ============================================================================

VST3Plugin::VST3Plugin()
    : component_(nullptr)
    , processor_(nullptr)
    , controller_(nullptr)
    , numInputs_(2)
    , numOutputs_(2)
{
}

VST3Plugin::~VST3Plugin() {
    terminate();
}

bool VST3Plugin::loadFromFile(const std::string& path) {
    // Load VST3 module
    std::string error;
    module_ = VST3::Hosting::Module::create(path, error);

    if (!module_) {
        std::cerr << "[VST3Plugin] Failed to load: " << path << "\n";
        std::cerr << "[VST3Plugin] Error: " << error << "\n";
        return false;
    }

    // Get factory
    auto factory = module_->getFactory();
    if (!factory) {
        std::cerr << "[VST3Plugin] Failed to get factory\n";
        return false;
    }

    // Find first audio processor class
    for (int i = 0; i < factory.classCount(); ++i) {
        PClassInfo classInfo;
        if (factory.getClassInfo(i, &classInfo) == kResultOk) {
            if (std::strcmp(classInfo.category, kVstAudioEffectClass) == 0) {
                // Create component
                if (factory.createInstance(classInfo.cid, IComponent::iid,
                                          reinterpret_cast<void**>(&component_)) == kResultOk) {
                    name_ = classInfo.name;

                    // Query processor interface
                    component_->queryInterface(IAudioProcessor::iid,
                                             reinterpret_cast<void**>(&processor_));

                    // Query controller interface
                    component_->queryInterface(IEditController::iid,
                                             reinterpret_cast<void**>(&controller_));

                    std::cout << "[VST3Plugin] Loaded: " << name_ << "\n";
                    return true;
                }
            }
        }
    }

    return false;
}

bool VST3Plugin::initialize(float sampleRate, int maxBlockSize) {
    if (!component_) {
        return false;
    }

    // Initialize component
    if (component_->initialize(nullptr) != kResultOk) {
        std::cerr << "[VST3Plugin] Component initialization failed\n";
        return false;
    }

    // Setup processing
    processSetup_.processMode = Realtime;
    processSetup_.symbolicSampleSize = kSample32;
    processSetup_.maxSamplesPerBlock = maxBlockSize;
    processSetup_.sampleRate = sampleRate;

    if (processor_ && processor_->setupProcessing(processSetup_) != kResultOk) {
        std::cerr << "[VST3Plugin] Setup processing failed\n";
        return false;
    }

    // Activate buses
    if (!activateBuses()) {
        std::cerr << "[VST3Plugin] Bus activation failed\n";
        return false;
    }

    // Activate component
    if (component_->setActive(true) != kResultOk) {
        std::cerr << "[VST3Plugin] Component activation failed\n";
        return false;
    }

    // Start processing
    if (processor_) {
        processor_->setProcessing(true);
    }

    return true;
}

void VST3Plugin::terminate() {
    if (processor_) {
        processor_->setProcessing(false);
    }

    if (component_) {
        component_->setActive(false);
        component_->terminate();
        component_->release();
        component_ = nullptr;
    }

    if (processor_) {
        processor_->release();
        processor_ = nullptr;
    }

    if (controller_) {
        controller_->release();
        controller_ = nullptr;
    }
}

void VST3Plugin::process(Core::AudioBuffer& buffer) {
    if (!processor_) {
        return;
    }

    int numSamples = buffer.getNumSamples();
    int numChannels = buffer.getNumChannels();

    // Prepare process data
    processData_.numSamples = numSamples;
    processData_.numInputs = inputBuses_.size();
    processData_.numOutputs = outputBuses_.size();
    processData_.inputs = inputBuses_.data();
    processData_.outputs = outputBuses_.data();

    // Copy buffer to VST3 format
    for (int ch = 0; ch < numChannels && ch < numInputs_; ++ch) {
        std::copy(buffer.getReadPointer(ch),
                  buffer.getReadPointer(ch) + numSamples,
                  reinterpret_cast<float*>(inputBuses_[0].channelBuffers32[ch]));
    }

    // Process
    processor_->process(processData_);

    // Copy back
    for (int ch = 0; ch < numChannels && ch < numOutputs_; ++ch) {
        std::copy(reinterpret_cast<float*>(outputBuses_[0].channelBuffers32[ch]),
                  reinterpret_cast<float*>(outputBuses_[0].channelBuffers32[ch]) + numSamples,
                  buffer.getWritePointer(ch));
    }
}

void VST3Plugin::processReplacing(float** inputs, float** outputs, int numSamples) {
    Core::AudioBuffer buffer(numInputs_, numSamples);

    // Copy inputs
    for (int ch = 0; ch < numInputs_; ++ch) {
        std::copy(inputs[ch], inputs[ch] + numSamples, buffer.getWritePointer(ch));
    }

    // Process
    process(buffer);

    // Copy outputs
    for (int ch = 0; ch < numOutputs_; ++ch) {
        std::copy(buffer.getReadPointer(ch), buffer.getReadPointer(ch) + numSamples, outputs[ch]);
    }
}

bool VST3Plugin::activateBuses() {
    if (!component_) {
        return false;
    }

    // Activate input buses
    int numInputBuses = component_->getBusCount(kAudio, kInput);
    for (int i = 0; i < numInputBuses; ++i) {
        component_->activateBus(kAudio, kInput, i, true);
    }

    // Activate output buses
    int numOutputBuses = component_->getBusCount(kAudio, kOutput);
    for (int i = 0; i < numOutputBuses; ++i) {
        component_->activateBus(kAudio, kOutput, i, true);
    }

    return true;
}

int VST3Plugin::getParameterCount() const {
    return controller_ ? controller_->getParameterCount() : 0;
}

PluginParameter VST3Plugin::getParameterInfo(int index) const {
    PluginParameter param;

    if (controller_) {
        ParameterInfo info;
        if (controller_->getParameterInfo(index, info) == kResultOk) {
            param.id = std::to_string(info.id);
            param.name = VST3::StringConvert::convert(info.title);
            param.value = controller_->getParamNormalized(info.id);
            param.defaultValue = info.defaultNormalizedValue;
            param.minValue = 0.0f;
            param.maxValue = 1.0f;
            param.unit = VST3::StringConvert::convert(info.units);
            param.isAutomatable = (info.flags & ParameterInfo::kCanAutomate) != 0;
        }
    }

    return param;
}

void VST3Plugin::setParameterValue(int index, float value) {
    if (controller_) {
        ParameterInfo info;
        if (controller_->getParameterInfo(index, info) == kResultOk) {
            controller_->setParamNormalized(info.id, value);
        }
    }
}

float VST3Plugin::getParameterValue(int index) const {
    if (controller_) {
        ParameterInfo info;
        if (controller_->getParameterInfo(index, info) == kResultOk) {
            return controller_->getParamNormalized(info.id);
        }
    }
    return 0.0f;
}

std::string VST3Plugin::getName() const {
    return name_;
}

std::string VST3Plugin::getVendor() const {
    return vendor_;
}

std::string VST3Plugin::getVersion() const {
    return "1.0.0";
}

int VST3Plugin::getNumInputs() const {
    return numInputs_;
}

int VST3Plugin::getNumOutputs() const {
    return numOutputs_;
}

bool VST3Plugin::acceptsMidi() const {
    return false; // Check bus info
}

bool VST3Plugin::producesMidi() const {
    return false; // Check bus info
}

bool VST3Plugin::saveState(std::vector<uint8_t>& data) {
    // Use component state
    return false; // TODO: implement
}

bool VST3Plugin::loadState(const std::vector<uint8_t>& data) {
    // Use component state
    return false; // TODO: implement
}

// ============================================================================
// VST3Scanner Implementation
// ============================================================================

VST3Scanner::VST3Scanner() = default;

void VST3Scanner::scanDefaultLocations() {
    std::vector<std::string> paths;

#ifdef __linux__
    paths.push_back("/usr/lib/vst3");
    paths.push_back("/usr/local/lib/vst3");
    paths.push_back(std::string(getenv("HOME")) + "/.vst3");
#elif __APPLE__
    paths.push_back("/Library/Audio/Plug-Ins/VST3");
    paths.push_back(std::string(getenv("HOME")) + "/Library/Audio/Plug-Ins/VST3");
#elif _WIN32
    paths.push_back("C:\\\\Program Files\\\\Common Files\\\\VST3");
#endif

    for (const auto& path : paths) {
        if (std::filesystem::exists(path)) {
            scanPath(path);
        }
    }

    std::cout << "[VST3Scanner] Found " << plugins_.size() << " VST3 plugins\n";
}

void VST3Scanner::scanPath(const std::string& path) {
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
            if (entry.is_regular_file() || entry.path().extension() == ".vst3") {
                scanFile(entry.path().string());
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[VST3Scanner] Error: " << e.what() << "\n";
    }
}

void VST3Scanner::scanFile(const std::string& path) {
    VST3PluginInfo info;
    if (extractPluginInfo(path, info)) {
        plugins_.push_back(info);
    }
}

bool VST3Scanner::extractPluginInfo(const std::string& path, VST3PluginInfo& info) {
    std::string error;
    auto module = VST3::Hosting::Module::create(path, error);

    if (!module) {
        return false;
    }

    auto factory = module->getFactory();
    if (!factory) {
        return false;
    }

    // Extract info from first audio effect class
    for (int i = 0; i < factory.classCount(); ++i) {
        PClassInfo classInfo;
        if (factory.getClassInfo(i, &classInfo) == kResultOk) {
            if (std::strcmp(classInfo.category, kVstAudioEffectClass) == 0) {
                info.path = path;
                info.name = classInfo.name;
                info.vendor = "Unknown"; // Need to query from component
                info.category = classInfo.category;
                info.isSynth = false;
                info.hasEditor = false;
                info.numInputs = 2;  // Query from component
                info.numOutputs = 2; // Query from component

                return true;
            }
        }
    }

    return false;
}

// ============================================================================
// VST3PluginHost Implementation
// ============================================================================

VST3PluginHost::VST3PluginHost()
    : scanner_(std::make_unique<VST3Scanner>())
{
}

VST3PluginHost::~VST3PluginHost() = default;

int VST3PluginHost::loadVST3Plugin(const std::string& path) {
    auto plugin = std::make_unique<VST3Plugin>();

    if (!plugin->loadFromFile(path)) {
        return -1;
    }

    if (!plugin->initialize(sampleRate_, blockSize_)) {
        return -1;
    }

    int id = nextPluginId_++;
    plugins_[id] = std::move(plugin);

    std::cout << "[VST3PluginHost] Loaded VST3 plugin ID: " << id << "\n";
    return id;
}

void VST3PluginHost::scanForVST3Plugins() {
    scanner_->scanDefaultLocations();
}

std::vector<VST3Scanner::VST3PluginInfo> VST3PluginHost::getAvailableVST3Plugins() const {
    return scanner_->getPlugins();
}

} // namespace Plugins
} // namespace MolinAntro

#endif // MOLINANTRO_VST3_SUPPORT
```

## Building with VST3 Support

```bash
mkdir build-vst3
cd build-vst3
cmake .. -DBUILD_VST3_SUPPORT=ON -DVST3_SDK_PATH=/opt/vst3sdk
cmake --build . -j 4
```

## Usage Example

```cpp
#include "plugins/VST3Loader.h"

// Create VST3 plugin host
Plugins::VST3PluginHost host;

// Scan for plugins
host.scanForVST3Plugins();
auto plugins = host.getAvailableVST3Plugins();

std::cout << "Found " << plugins.size() << " VST3 plugins:\n";
for (const auto& plugin : plugins) {
    std::cout << "  - " << plugin.name << " by " << plugin.vendor << "\n";
}

// Load a specific VST3 plugin
int pluginId = host.loadVST3Plugin("/path/to/plugin.vst3");

// Add to processing chain
host.addPluginToChain(pluginId);

// Process audio
Core::AudioBuffer buffer(2, 512);
// ... fill buffer ...
host.processPluginChain(buffer);
```

## Status

- ✅ Plugin interface ready (VST3-compatible)
- ✅ Plugin host infrastructure complete
- ✅ VST3 loader implementation provided
- ⏸️ Awaiting VST3 SDK installation
- ⏸️ Real VST3 loading to be tested

## Resources

- VST3 SDK: https://github.com/steinbergmedia/vst3sdk
- VST3 Documentation: https://steinbergmedia.github.io/vst3_doc/
- VST3 Examples: https://github.com/steinbergmedia/vst3sdk/tree/master/samples
