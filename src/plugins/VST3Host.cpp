/**
 * @file VST3Host.cpp
 * @brief VST3 Plugin Hosting Implementation
 *
 * Full VST3 hosting using Steinberg VST3 SDK.
 * Implements plugin loading, parameter control, audio processing, and editor hosting.
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "plugins/PluginFormats.h"
#include <iostream>
#include <filesystem>
#include <cstring>
#include <vector>
#include <map>
#include <mutex>

#ifdef HAVE_VST3_SDK
// VST3 SDK includes
#include "pluginterfaces/base/funknown.h"
#include "pluginterfaces/base/ipluginbase.h"
#include "pluginterfaces/vst/ivstcomponent.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstevents.h"
#include "pluginterfaces/vst/ivstunits.h"
#include "pluginterfaces/vst/ivsthostapplication.h"
#include "pluginterfaces/vst/ivstmidicontrollers.h"
#include "pluginterfaces/gui/iplugview.h"
#include "public.sdk/source/vst/hosting/module.h"
#include "public.sdk/source/vst/hosting/hostclasses.h"
#include "public.sdk/source/vst/hosting/plugprovider.h"
#include "base/source/fstring.h"

using namespace Steinberg;
using namespace Steinberg::Vst;
#endif

namespace MolinAntro {
namespace Plugins {

// ============================================================================
// VST3 Internal Implementation
// ============================================================================

#ifdef HAVE_VST3_SDK

/**
 * @brief Host application context for VST3 plugins
 */
class VST3HostContext : public Steinberg::Vst::IHostApplication,
                        public Steinberg::Vst::IComponentHandler {
public:
    VST3HostContext() : refCount_(1) {}

    // IUnknown
    Steinberg::tresult PLUGIN_API queryInterface(const Steinberg::TUID iid, void** obj) override {
        QUERY_INTERFACE(iid, obj, Steinberg::FUnknown::iid, IHostApplication)
        QUERY_INTERFACE(iid, obj, IHostApplication::iid, IHostApplication)
        QUERY_INTERFACE(iid, obj, IComponentHandler::iid, IComponentHandler)
        *obj = nullptr;
        return Steinberg::kNoInterface;
    }

    Steinberg::uint32 PLUGIN_API addRef() override { return ++refCount_; }
    Steinberg::uint32 PLUGIN_API release() override {
        if (--refCount_ == 0) {
            delete this;
            return 0;
        }
        return refCount_;
    }

    // IHostApplication
    Steinberg::tresult PLUGIN_API getName(Steinberg::Vst::String128 name) override {
        Steinberg::String str("MolinAntro DAW");
        str.copyTo(name, 128);
        return Steinberg::kResultOk;
    }

    Steinberg::tresult PLUGIN_API createInstance(Steinberg::TUID /*cid*/, Steinberg::TUID /*iid*/,
                                                  void** /*obj*/) override {
        return Steinberg::kNotImplemented;
    }

    // IComponentHandler
    Steinberg::tresult PLUGIN_API beginEdit(Steinberg::Vst::ParamID /*id*/) override {
        return Steinberg::kResultOk;
    }

    Steinberg::tresult PLUGIN_API performEdit(Steinberg::Vst::ParamID id,
                                               Steinberg::Vst::ParamValue value) override {
        std::lock_guard<std::mutex> lock(paramMutex_);
        pendingChanges_[id] = value;
        return Steinberg::kResultOk;
    }

    Steinberg::tresult PLUGIN_API endEdit(Steinberg::Vst::ParamID /*id*/) override {
        return Steinberg::kResultOk;
    }

    Steinberg::tresult PLUGIN_API restartComponent(Steinberg::int32 /*flags*/) override {
        return Steinberg::kResultOk;
    }

    std::map<Steinberg::Vst::ParamID, Steinberg::Vst::ParamValue> getPendingChanges() {
        std::lock_guard<std::mutex> lock(paramMutex_);
        auto changes = std::move(pendingChanges_);
        pendingChanges_.clear();
        return changes;
    }

private:
    std::atomic<Steinberg::int32> refCount_;
    std::mutex paramMutex_;
    std::map<Steinberg::Vst::ParamID, Steinberg::Vst::ParamValue> pendingChanges_;
};

/**
 * @brief Parameter change queue for VST3 processing
 */
class VST3ParameterChanges : public Steinberg::Vst::IParameterChanges {
public:
    VST3ParameterChanges() : refCount_(1) {}

    Steinberg::uint32 PLUGIN_API addRef() override { return ++refCount_; }
    Steinberg::uint32 PLUGIN_API release() override {
        if (--refCount_ == 0) { delete this; return 0; }
        return refCount_;
    }

    Steinberg::tresult PLUGIN_API queryInterface(const Steinberg::TUID iid, void** obj) override {
        QUERY_INTERFACE(iid, obj, Steinberg::FUnknown::iid, IParameterChanges)
        QUERY_INTERFACE(iid, obj, IParameterChanges::iid, IParameterChanges)
        *obj = nullptr;
        return Steinberg::kNoInterface;
    }

    Steinberg::int32 PLUGIN_API getParameterCount() override {
        return static_cast<Steinberg::int32>(queues_.size());
    }

    Steinberg::Vst::IParamValueQueue* PLUGIN_API getParameterData(Steinberg::int32 index) override {
        if (index >= 0 && index < static_cast<Steinberg::int32>(queues_.size())) {
            return queues_[index].get();
        }
        return nullptr;
    }

    Steinberg::Vst::IParamValueQueue* PLUGIN_API addParameterData(
        const Steinberg::Vst::ParamID& id, Steinberg::int32& index) override {
        // Simplified - would need full implementation
        index = -1;
        return nullptr;
    }

    void clear() { queues_.clear(); }

private:
    std::atomic<Steinberg::int32> refCount_;
    std::vector<std::unique_ptr<Steinberg::Vst::IParamValueQueue>> queues_;
};

#endif // HAVE_VST3_SDK

// ============================================================================
// VST3Plugin Implementation
// ============================================================================

struct VST3Plugin::Impl {
#ifdef HAVE_VST3_SDK
    VST3::Hosting::Module::Ptr module;
    IPtr<IComponent> component;
    IPtr<IAudioProcessor> processor;
    IPtr<IEditController> controller;
    IPtr<IPlugView> view;
    VST3HostContext* hostContext = nullptr;

    // Processing buffers
    Steinberg::Vst::ProcessData processData;
    Steinberg::Vst::AudioBusBuffers inputBusBuffers;
    Steinberg::Vst::AudioBusBuffers outputBusBuffers;
    std::vector<float*> inputChannelPtrs;
    std::vector<float*> outputChannelPtrs;
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
#endif

    PluginInfo info;
    std::vector<PluginParameter> parameters;

    bool initialized = false;
    bool processing = false;
    bool bypassed = false;

    int sampleRate = 48000;
    int maxBlockSize = 512;

    int latencySamples = 0;
    int tailSamples = 0;
};

VST3Plugin::VST3Plugin() : impl_(std::make_unique<Impl>()) {
    impl_->info.format = PluginFormat::VST3;
}

VST3Plugin::~VST3Plugin() {
    terminate();
}

bool VST3Plugin::load(const std::string& path, VST3Plugin& plugin) {
#ifdef HAVE_VST3_SDK
    std::string error;
    plugin.impl_->module = VST3::Hosting::Module::create(path, error);

    if (!plugin.impl_->module) {
        std::cerr << "[VST3] Failed to load module: " << error << "\n";
        return false;
    }

    auto factory = plugin.impl_->module->getFactory();
    if (!factory.get()) {
        std::cerr << "[VST3] No factory in module\n";
        return false;
    }

    // Get plugin info
    auto classInfo = factory.classInfos();
    for (const auto& ci : classInfo) {
        if (ci.category() == kVstAudioEffectClass) {
            plugin.impl_->info.id = ci.ID().toString();
            plugin.impl_->info.name = ci.name();
            plugin.impl_->info.vendor = ci.vendor();
            plugin.impl_->info.version = ci.version();
            plugin.impl_->info.filePath = path;

            // Create component
            plugin.impl_->component = factory.createInstance<IComponent>(ci.ID());
            if (!plugin.impl_->component) {
                continue;
            }

            // Create host context
            plugin.impl_->hostContext = new VST3HostContext();

            // Initialize component
            if (plugin.impl_->component->initialize(plugin.impl_->hostContext) != kResultOk) {
                std::cerr << "[VST3] Component initialization failed\n";
                return false;
            }

            // Query audio processor
            plugin.impl_->processor = FUnknownPtr<IAudioProcessor>(plugin.impl_->component);
            if (!plugin.impl_->processor) {
                std::cerr << "[VST3] No audio processor interface\n";
                return false;
            }

            // Query edit controller
            TUID controllerCID;
            if (plugin.impl_->component->getControllerClassId(controllerCID) == kResultOk) {
                plugin.impl_->controller = factory.createInstance<IEditController>(controllerCID);
                if (plugin.impl_->controller) {
                    plugin.impl_->controller->initialize(plugin.impl_->hostContext);
                    plugin.impl_->controller->setComponentHandler(plugin.impl_->hostContext);

                    // Get parameters
                    int numParams = plugin.impl_->controller->getParameterCount();
                    for (int i = 0; i < numParams; ++i) {
                        ParameterInfo pinfo;
                        if (plugin.impl_->controller->getParameterInfo(i, pinfo) == kResultOk) {
                            PluginParameter param;
                            param.id = pinfo.id;
                            param.name = VST3::StringConvert::convert(pinfo.title);
                            param.unit = VST3::StringConvert::convert(pinfo.units);
                            param.defaultValue = pinfo.defaultNormalizedValue;
                            param.value = pinfo.defaultNormalizedValue;
                            param.minValue = 0.0f;
                            param.maxValue = 1.0f;
                            param.numSteps = pinfo.stepCount;
                            param.isAutomatable = (pinfo.flags & ParameterInfo::kCanAutomate) != 0;
                            param.isBypass = (pinfo.flags & ParameterInfo::kIsBypass) != 0;
                            plugin.impl_->parameters.push_back(param);
                        }
                    }
                }
            }

            // Get bus info
            int numInputBuses = plugin.impl_->component->getBusCount(kAudio, kInput);
            int numOutputBuses = plugin.impl_->component->getBusCount(kAudio, kOutput);

            plugin.impl_->info.numAudioInputs = 0;
            plugin.impl_->info.numAudioOutputs = 0;

            for (int i = 0; i < numInputBuses; ++i) {
                BusInfo busInfo;
                if (plugin.impl_->component->getBusInfo(kAudio, kInput, i, busInfo) == kResultOk) {
                    plugin.impl_->info.numAudioInputs += busInfo.channelCount;
                }
            }

            for (int i = 0; i < numOutputBuses; ++i) {
                BusInfo busInfo;
                if (plugin.impl_->component->getBusInfo(kAudio, kOutput, i, busInfo) == kResultOk) {
                    plugin.impl_->info.numAudioOutputs += busInfo.channelCount;
                }
            }

            // Check for editor
            if (plugin.impl_->controller) {
                plugin.impl_->view = plugin.impl_->controller->createView(ViewType::kEditor);
                plugin.impl_->info.hasEditor = (plugin.impl_->view != nullptr);
            }

            std::cout << "[VST3] Loaded: " << plugin.impl_->info.name
                      << " by " << plugin.impl_->info.vendor
                      << " (Inputs: " << plugin.impl_->info.numAudioInputs
                      << ", Outputs: " << plugin.impl_->info.numAudioOutputs
                      << ", Params: " << plugin.impl_->parameters.size() << ")\n";

            return true;
        }
    }

    std::cerr << "[VST3] No audio effect class found\n";
    return false;

#else
    // Fallback when VST3 SDK is not available
    std::cout << "[VST3] VST3 SDK not compiled - creating stub for: " << path << "\n";

    plugin.impl_->info.filePath = path;
    plugin.impl_->info.name = std::filesystem::path(path).stem().string();
    plugin.impl_->info.vendor = "Unknown";
    plugin.impl_->info.version = "1.0.0";
    plugin.impl_->info.numAudioInputs = 2;
    plugin.impl_->info.numAudioOutputs = 2;
    plugin.impl_->info.hasEditor = false;

    // Add some generic parameters for demonstration
    for (int i = 0; i < 4; ++i) {
        PluginParameter param;
        param.id = i;
        param.name = "Parameter " + std::to_string(i + 1);
        param.value = 0.5f;
        param.defaultValue = 0.5f;
        param.minValue = 0.0f;
        param.maxValue = 1.0f;
        param.isAutomatable = true;
        plugin.impl_->parameters.push_back(param);
    }

    return true;
#endif
}

PluginInfo VST3Plugin::getInfo() const {
    return impl_->info;
}

bool VST3Plugin::initialize(int sampleRate, int maxBlockSize) {
#ifdef HAVE_VST3_SDK
    if (!impl_->processor) {
        std::cerr << "[VST3] No processor to initialize\n";
        return false;
    }

    impl_->sampleRate = sampleRate;
    impl_->maxBlockSize = maxBlockSize;

    // Setup processing
    ProcessSetup setup;
    setup.processMode = kRealtime;
    setup.symbolicSampleSize = kSample32;
    setup.maxSamplesPerBlock = maxBlockSize;
    setup.sampleRate = static_cast<double>(sampleRate);

    if (impl_->processor->setupProcessing(setup) != kResultOk) {
        std::cerr << "[VST3] Setup processing failed\n";
        return false;
    }

    // Activate buses
    int numInputBuses = impl_->component->getBusCount(kAudio, kInput);
    int numOutputBuses = impl_->component->getBusCount(kAudio, kOutput);

    for (int i = 0; i < numInputBuses; ++i) {
        impl_->component->activateBus(kAudio, kInput, i, true);
    }
    for (int i = 0; i < numOutputBuses; ++i) {
        impl_->component->activateBus(kAudio, kOutput, i, true);
    }

    // Allocate processing buffers
    int numInputChannels = impl_->info.numAudioInputs;
    int numOutputChannels = impl_->info.numAudioOutputs;

    impl_->inputBuffer.resize(numInputChannels * maxBlockSize);
    impl_->outputBuffer.resize(numOutputChannels * maxBlockSize);

    impl_->inputChannelPtrs.resize(numInputChannels);
    impl_->outputChannelPtrs.resize(numOutputChannels);

    for (int i = 0; i < numInputChannels; ++i) {
        impl_->inputChannelPtrs[i] = impl_->inputBuffer.data() + i * maxBlockSize;
    }
    for (int i = 0; i < numOutputChannels; ++i) {
        impl_->outputChannelPtrs[i] = impl_->outputBuffer.data() + i * maxBlockSize;
    }

    // Setup bus buffers
    impl_->inputBusBuffers.numChannels = numInputChannels;
    impl_->inputBusBuffers.channelBuffers32 = impl_->inputChannelPtrs.data();
    impl_->inputBusBuffers.silenceFlags = 0;

    impl_->outputBusBuffers.numChannels = numOutputChannels;
    impl_->outputBusBuffers.channelBuffers32 = impl_->outputChannelPtrs.data();
    impl_->outputBusBuffers.silenceFlags = 0;

    // Get latency
    impl_->latencySamples = impl_->processor->getLatencySamples();
    impl_->tailSamples = impl_->processor->getTailSamples();

    impl_->info.latencySamples = impl_->latencySamples;

    impl_->initialized = true;

    std::cout << "[VST3] Initialized: " << impl_->info.name
              << " @ " << sampleRate << " Hz, block size " << maxBlockSize
              << ", latency " << impl_->latencySamples << " samples\n";

    return true;

#else
    impl_->sampleRate = sampleRate;
    impl_->maxBlockSize = maxBlockSize;
    impl_->initialized = true;
    return true;
#endif
}

void VST3Plugin::terminate() {
#ifdef HAVE_VST3_SDK
    if (impl_->view) {
        impl_->view->removed();
        impl_->view = nullptr;
    }

    if (impl_->controller) {
        impl_->controller->terminate();
        impl_->controller = nullptr;
    }

    if (impl_->component) {
        impl_->component->terminate();
        impl_->component = nullptr;
    }

    if (impl_->hostContext) {
        impl_->hostContext->release();
        impl_->hostContext = nullptr;
    }

    impl_->module = nullptr;
#endif

    impl_->initialized = false;
    impl_->processing = false;
}

bool VST3Plugin::isInitialized() const {
    return impl_->initialized;
}

void VST3Plugin::process(Core::AudioBuffer& buffer,
                         std::vector<MIDI::MIDIMessage>& midiIn,
                         std::vector<MIDI::MIDIMessage>& midiOut) {
    if (!impl_->initialized || impl_->bypassed) {
        return;
    }

#ifdef HAVE_VST3_SDK
    if (!impl_->processor) {
        return;
    }

    int numSamples = buffer.getNumSamples();
    int numChannels = buffer.getNumChannels();

    // Copy input
    for (int ch = 0; ch < std::min(numChannels, static_cast<int>(impl_->inputChannelPtrs.size())); ++ch) {
        const float* src = buffer.getReadPointer(ch);
        std::copy(src, src + numSamples, impl_->inputChannelPtrs[ch]);
    }

    // Setup process data
    ProcessData processData;
    processData.processMode = kRealtime;
    processData.symbolicSampleSize = kSample32;
    processData.numSamples = numSamples;
    processData.numInputs = 1;
    processData.numOutputs = 1;
    processData.inputs = &impl_->inputBusBuffers;
    processData.outputs = &impl_->outputBusBuffers;

    // TODO: Handle MIDI events through IEventList

    // Process
    impl_->processor->process(processData);

    // Copy output
    for (int ch = 0; ch < std::min(numChannels, static_cast<int>(impl_->outputChannelPtrs.size())); ++ch) {
        float* dst = buffer.getWritePointer(ch);
        std::copy(impl_->outputChannelPtrs[ch], impl_->outputChannelPtrs[ch] + numSamples, dst);
    }

#else
    // Passthrough when VST3 SDK not available
    (void)midiIn;
    (void)midiOut;
#endif
}

void VST3Plugin::setProcessing(bool active) {
#ifdef HAVE_VST3_SDK
    if (impl_->processor) {
        impl_->processor->setProcessing(active ? Steinberg::kResultTrue : Steinberg::kResultFalse);
    }
#endif
    impl_->processing = active;
}

bool VST3Plugin::isProcessing() const {
    return impl_->processing;
}

int VST3Plugin::getNumParameters() const {
    return static_cast<int>(impl_->parameters.size());
}

PluginParameter VST3Plugin::getParameter(int index) const {
    if (index >= 0 && index < static_cast<int>(impl_->parameters.size())) {
        return impl_->parameters[index];
    }
    return {};
}

float VST3Plugin::getParameterValue(int index) const {
#ifdef HAVE_VST3_SDK
    if (impl_->controller && index >= 0 && index < static_cast<int>(impl_->parameters.size())) {
        return static_cast<float>(impl_->controller->getParamNormalized(impl_->parameters[index].id));
    }
#endif
    if (index >= 0 && index < static_cast<int>(impl_->parameters.size())) {
        return impl_->parameters[index].value;
    }
    return 0.0f;
}

void VST3Plugin::setParameterValue(int index, float value) {
#ifdef HAVE_VST3_SDK
    if (impl_->controller && index >= 0 && index < static_cast<int>(impl_->parameters.size())) {
        impl_->controller->setParamNormalized(impl_->parameters[index].id, value);
    }
#endif
    if (index >= 0 && index < static_cast<int>(impl_->parameters.size())) {
        impl_->parameters[index].value = value;
    }
}

std::string VST3Plugin::getParameterDisplay(int index) const {
#ifdef HAVE_VST3_SDK
    if (impl_->controller && index >= 0 && index < static_cast<int>(impl_->parameters.size())) {
        Steinberg::Vst::String128 display;
        ParamID id = impl_->parameters[index].id;
        ParamValue value = impl_->controller->getParamNormalized(id);
        if (impl_->controller->getParamStringByValue(id, value, display) == kResultOk) {
            return VST3::StringConvert::convert(display);
        }
    }
#endif
    return std::to_string(getParameterValue(index));
}

PluginState VST3Plugin::getState() const {
    PluginState state;
    state.pluginId = impl_->info.id;
    state.bypassState = impl_->bypassed;

    for (int i = 0; i < static_cast<int>(impl_->parameters.size()); ++i) {
        state.parameters[i] = getParameterValue(i);
    }

    // TODO: Get component state chunk using IComponent::getState()

    return state;
}

bool VST3Plugin::setState(const PluginState& state) {
    impl_->bypassed = state.bypassState;

    for (const auto& [index, value] : state.parameters) {
        setParameterValue(index, value);
    }

    // TODO: Set component state chunk using IComponent::setState()

    return true;
}

std::vector<PluginPreset> VST3Plugin::getPresets() const {
    std::vector<PluginPreset> presets;

#ifdef HAVE_VST3_SDK
    // TODO: Implement preset enumeration using IUnitInfo
#endif

    return presets;
}

bool VST3Plugin::loadPreset(const PluginPreset& preset) {
    for (const auto& [id, value] : preset.parameters) {
        setParameterValue(id, value);
    }
    return true;
}

PluginPreset VST3Plugin::savePreset(const std::string& name) const {
    PluginPreset preset;
    preset.name = name;
    preset.author = "User";

    for (int i = 0; i < static_cast<int>(impl_->parameters.size()); ++i) {
        preset.parameters[i] = getParameterValue(i);
    }

    return preset;
}

bool VST3Plugin::hasEditor() const {
    return impl_->info.hasEditor;
}

void* VST3Plugin::createEditor(void* parentWindow) {
#ifdef HAVE_VST3_SDK
    if (impl_->controller && !impl_->view) {
        impl_->view = impl_->controller->createView(ViewType::kEditor);
    }

    if (impl_->view) {
        if (impl_->view->attached(parentWindow, kPlatformTypeHWND) == kResultOk) {
            return impl_->view.get();
        }
    }
#endif
    return nullptr;
}

void VST3Plugin::destroyEditor() {
#ifdef HAVE_VST3_SDK
    if (impl_->view) {
        impl_->view->removed();
        impl_->view = nullptr;
    }
#endif
}

void VST3Plugin::getEditorSize(int& width, int& height) const {
#ifdef HAVE_VST3_SDK
    if (impl_->view) {
        ViewRect rect;
        if (impl_->view->getSize(&rect) == kResultOk) {
            width = rect.getWidth();
            height = rect.getHeight();
            return;
        }
    }
#endif
    width = 400;
    height = 300;
}

void VST3Plugin::resizeEditor(int width, int height) {
#ifdef HAVE_VST3_SDK
    if (impl_->view) {
        ViewRect rect(0, 0, width, height);
        impl_->view->onSize(&rect);
    }
#endif
}

void VST3Plugin::setBypass(bool bypass) {
    impl_->bypassed = bypass;

#ifdef HAVE_VST3_SDK
    // Find bypass parameter and set it
    for (const auto& param : impl_->parameters) {
        if (param.isBypass) {
            setParameterValue(param.id, bypass ? 1.0f : 0.0f);
            break;
        }
    }
#endif
}

bool VST3Plugin::isBypassed() const {
    return impl_->bypassed;
}

int VST3Plugin::getLatency() const {
    return impl_->latencySamples;
}

int VST3Plugin::getTailLength() const {
    return impl_->tailSamples;
}

// ============================================================================
// PluginManager Implementation
// ============================================================================

struct PluginManager::Impl {
    std::map<PluginFormat, std::vector<std::string>> searchPaths;
    std::vector<PluginInfo> plugins;
    std::vector<std::string> blacklist;

    std::map<std::string, std::vector<PluginPreset>> presets;
};

PluginManager::PluginManager() : impl_(std::make_unique<Impl>()) {
    // Add default search paths
#ifdef _WIN32
    impl_->searchPaths[PluginFormat::VST3].push_back("C:\\Program Files\\Common Files\\VST3");
    impl_->searchPaths[PluginFormat::VST3].push_back("C:\\Program Files (x86)\\Common Files\\VST3");
#elif __APPLE__
    impl_->searchPaths[PluginFormat::VST3].push_back("/Library/Audio/Plug-Ins/VST3");
    impl_->searchPaths[PluginFormat::AU].push_back("/Library/Audio/Plug-Ins/Components");
#else
    impl_->searchPaths[PluginFormat::VST3].push_back("/usr/lib/vst3");
    impl_->searchPaths[PluginFormat::VST3].push_back("/usr/local/lib/vst3");
    impl_->searchPaths[PluginFormat::LV2].push_back("/usr/lib/lv2");
#endif
}

PluginManager::~PluginManager() = default;

void PluginManager::addSearchPath(const std::string& path, PluginFormat format) {
    impl_->searchPaths[format].push_back(path);
}

void PluginManager::removeSearchPath(const std::string& path) {
    for (auto& [format, paths] : impl_->searchPaths) {
        paths.erase(std::remove(paths.begin(), paths.end(), path), paths.end());
    }
}

std::vector<std::string> PluginManager::getSearchPaths(PluginFormat format) const {
    auto it = impl_->searchPaths.find(format);
    if (it != impl_->searchPaths.end()) {
        return it->second;
    }
    return {};
}

void PluginManager::scanPlugins(PluginFormat format) {
    scanning_ = true;
    scanProgress_ = 0.0f;

    auto paths = getSearchPaths(format);
    int totalPaths = static_cast<int>(paths.size());
    int currentPath = 0;

    for (const auto& searchPath : paths) {
        if (!std::filesystem::exists(searchPath)) {
            continue;
        }

        for (const auto& entry : std::filesystem::recursive_directory_iterator(searchPath)) {
            if (!entry.is_regular_file() && !entry.is_directory()) {
                continue;
            }

            std::string ext = entry.path().extension().string();
            std::string path = entry.path().string();

            bool isPlugin = false;
            if (format == PluginFormat::VST3 && (ext == ".vst3" || entry.path().extension() == ".vst3")) {
                isPlugin = true;
            }

            if (isPlugin && !isBlacklisted(path)) {
                currentScanPlugin_ = entry.path().filename().string();

                if (onScan_) {
                    onScan_(scanProgress_, currentScanPlugin_);
                }

                // Try to load and get info
                VST3Plugin testPlugin;
                if (VST3Plugin::load(path, testPlugin)) {
                    PluginInfo info = testPlugin.getInfo();
                    info.format = format;
                    impl_->plugins.push_back(info);

                    if (onPluginFound_) {
                        onPluginFound_(info);
                    }
                }
            }
        }

        ++currentPath;
        scanProgress_ = static_cast<float>(currentPath) / static_cast<float>(totalPaths);
    }

    scanning_ = false;
    scanProgress_ = 1.0f;
    currentScanPlugin_.clear();

    std::cout << "[PluginManager] Scan complete. Found " << impl_->plugins.size() << " plugins.\n";
}

void PluginManager::scanAllFormats() {
    scanPlugins(PluginFormat::VST3);
#ifdef __APPLE__
    scanPlugins(PluginFormat::AU);
#endif
#ifdef __linux__
    scanPlugins(PluginFormat::LV2);
#endif
    scanPlugins(PluginFormat::CLAP);
}

void PluginManager::rescanPlugin(const std::string& pluginId) {
    // Remove existing entry
    impl_->plugins.erase(
        std::remove_if(impl_->plugins.begin(), impl_->plugins.end(),
            [&pluginId](const PluginInfo& info) { return info.id == pluginId; }),
        impl_->plugins.end()
    );

    // Rescan would require knowing the path - simplified for now
}

std::vector<PluginInfo> PluginManager::getPlugins() const {
    return impl_->plugins;
}

std::vector<PluginInfo> PluginManager::getPluginsByFormat(PluginFormat format) const {
    std::vector<PluginInfo> result;
    for (const auto& plugin : impl_->plugins) {
        if (plugin.format == format) {
            result.push_back(plugin);
        }
    }
    return result;
}

std::vector<PluginInfo> PluginManager::getPluginsByCategory(PluginCategory category) const {
    std::vector<PluginInfo> result;
    for (const auto& plugin : impl_->plugins) {
        if (plugin.category == category) {
            result.push_back(plugin);
        }
    }
    return result;
}

std::vector<PluginInfo> PluginManager::searchPlugins(const std::string& query) const {
    std::vector<PluginInfo> result;
    std::string lowerQuery = query;
    std::transform(lowerQuery.begin(), lowerQuery.end(), lowerQuery.begin(), ::tolower);

    for (const auto& plugin : impl_->plugins) {
        std::string lowerName = plugin.name;
        std::transform(lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);

        std::string lowerVendor = plugin.vendor;
        std::transform(lowerVendor.begin(), lowerVendor.end(), lowerVendor.begin(), ::tolower);

        if (lowerName.find(lowerQuery) != std::string::npos ||
            lowerVendor.find(lowerQuery) != std::string::npos) {
            result.push_back(plugin);
        }
    }
    return result;
}

PluginInfo PluginManager::getPluginInfo(const std::string& pluginId) const {
    for (const auto& plugin : impl_->plugins) {
        if (plugin.id == pluginId) {
            return plugin;
        }
    }
    return {};
}

std::unique_ptr<IPluginInstance> PluginManager::createInstance(const std::string& pluginId) {
    auto info = getPluginInfo(pluginId);
    if (info.filePath.empty()) {
        return nullptr;
    }

    switch (info.format) {
        case PluginFormat::VST3: {
            auto plugin = std::make_unique<VST3Plugin>();
            if (VST3Plugin::load(info.filePath, *plugin)) {
                return plugin;
            }
            break;
        }
        // Other formats would be handled here
        default:
            break;
    }

    return nullptr;
}

bool PluginManager::unloadPlugin(const std::string& pluginId) {
    impl_->plugins.erase(
        std::remove_if(impl_->plugins.begin(), impl_->plugins.end(),
            [&pluginId](const PluginInfo& info) { return info.id == pluginId; }),
        impl_->plugins.end()
    );
    return true;
}

void PluginManager::blacklistPlugin(const std::string& pluginId) {
    impl_->blacklist.push_back(pluginId);
}

void PluginManager::removeFromBlacklist(const std::string& pluginId) {
    impl_->blacklist.erase(
        std::remove(impl_->blacklist.begin(), impl_->blacklist.end(), pluginId),
        impl_->blacklist.end()
    );
}

bool PluginManager::isBlacklisted(const std::string& pluginId) const {
    return std::find(impl_->blacklist.begin(), impl_->blacklist.end(), pluginId) != impl_->blacklist.end();
}

std::vector<std::string> PluginManager::getBlacklist() const {
    return impl_->blacklist;
}

void PluginManager::setFavorite(const std::string& pluginId, bool favorite) {
    for (auto& plugin : impl_->plugins) {
        if (plugin.id == pluginId) {
            plugin.isFavorite = favorite;
            break;
        }
    }
}

void PluginManager::setRating(const std::string& pluginId, int rating) {
    for (auto& plugin : impl_->plugins) {
        if (plugin.id == pluginId) {
            plugin.rating = std::clamp(rating, 0, 5);
            break;
        }
    }
}

std::vector<PluginPreset> PluginManager::getPresetsForPlugin(const std::string& pluginId) const {
    auto it = impl_->presets.find(pluginId);
    if (it != impl_->presets.end()) {
        return it->second;
    }
    return {};
}

void PluginManager::savePreset(const std::string& pluginId, const PluginPreset& preset) {
    impl_->presets[pluginId].push_back(preset);
}

void PluginManager::deletePreset(const std::string& pluginId, const std::string& presetName) {
    auto it = impl_->presets.find(pluginId);
    if (it != impl_->presets.end()) {
        it->second.erase(
            std::remove_if(it->second.begin(), it->second.end(),
                [&presetName](const PluginPreset& p) { return p.name == presetName; }),
            it->second.end()
        );
    }
}

void PluginManager::scanFactoryPresets(const std::string& /*pluginId*/) {
    // TODO: Scan factory presets from plugin directory
}

void PluginManager::saveDatabase(const std::string& /*path*/) {
    // TODO: Save plugin database to JSON/XML
}

void PluginManager::loadDatabase(const std::string& /*path*/) {
    // TODO: Load plugin database from JSON/XML
}

// ============================================================================
// PluginChain Implementation
// ============================================================================

PluginChain::PluginChain() = default;
PluginChain::~PluginChain() = default;

int PluginChain::addPlugin(std::unique_ptr<IPluginInstance> plugin) {
    int index = static_cast<int>(plugins_.size());
    plugins_.push_back(std::move(plugin));
    return index;
}

void PluginChain::removePlugin(int index) {
    if (index >= 0 && index < static_cast<int>(plugins_.size())) {
        plugins_.erase(plugins_.begin() + index);
    }
}

void PluginChain::movePlugin(int fromIndex, int toIndex) {
    if (fromIndex < 0 || fromIndex >= static_cast<int>(plugins_.size()) ||
        toIndex < 0 || toIndex >= static_cast<int>(plugins_.size()) ||
        fromIndex == toIndex) {
        return;
    }

    auto plugin = std::move(plugins_[fromIndex]);
    plugins_.erase(plugins_.begin() + fromIndex);
    plugins_.insert(plugins_.begin() + toIndex, std::move(plugin));
}

void PluginChain::clearChain() {
    plugins_.clear();
}

IPluginInstance* PluginChain::getPlugin(int index) {
    if (index >= 0 && index < static_cast<int>(plugins_.size())) {
        return plugins_[index].get();
    }
    return nullptr;
}

void PluginChain::process(Core::AudioBuffer& buffer,
                          std::vector<MIDI::MIDIMessage>& midiIn,
                          std::vector<MIDI::MIDIMessage>& midiOut) {
    if (!active_) {
        return;
    }

    for (auto& plugin : plugins_) {
        if (plugin && !plugin->isBypassed()) {
            plugin->process(buffer, midiIn, midiOut);
        }
    }
}

void PluginChain::setPluginBypass(int index, bool bypass) {
    if (index >= 0 && index < static_cast<int>(plugins_.size()) && plugins_[index]) {
        plugins_[index]->setBypass(bypass);
    }
}

bool PluginChain::isPluginBypassed(int index) const {
    if (index >= 0 && index < static_cast<int>(plugins_.size()) && plugins_[index]) {
        return plugins_[index]->isBypassed();
    }
    return false;
}

PluginChain::ChainState PluginChain::getState() const {
    ChainState state;
    for (const auto& plugin : plugins_) {
        if (plugin) {
            state.pluginStates.push_back(plugin->getState());
            state.bypassStates.push_back(plugin->isBypassed());
        }
    }
    return state;
}

bool PluginChain::setState(const ChainState& state) {
    for (size_t i = 0; i < plugins_.size() && i < state.pluginStates.size(); ++i) {
        if (plugins_[i]) {
            plugins_[i]->setState(state.pluginStates[i]);
            if (i < state.bypassStates.size()) {
                plugins_[i]->setBypass(state.bypassStates[i]);
            }
        }
    }
    return true;
}

int PluginChain::getTotalLatency() const {
    int totalLatency = 0;
    for (const auto& plugin : plugins_) {
        if (plugin && !plugin->isBypassed()) {
            totalLatency += plugin->getLatency();
        }
    }
    return totalLatency;
}

// ============================================================================
// SendReturn Implementation
// ============================================================================

SendReturn::SendReturn() = default;
SendReturn::~SendReturn() = default;

void SendReturn::addSend(const Send& send) {
    sends_.push_back(send);
}

void SendReturn::removeSend(int sourceTrack, int returnTrack) {
    sends_.erase(
        std::remove_if(sends_.begin(), sends_.end(),
            [sourceTrack, returnTrack](const Send& s) {
                return s.sourceTrackIndex == sourceTrack && s.returnTrackIndex == returnTrack;
            }),
        sends_.end()
    );
}

void SendReturn::setSendAmount(int sourceTrack, int returnTrack, float amount) {
    for (auto& send : sends_) {
        if (send.sourceTrackIndex == sourceTrack && send.returnTrackIndex == returnTrack) {
            send.amount = std::clamp(amount, 0.0f, 1.0f);
            break;
        }
    }
}

float SendReturn::getSendAmount(int sourceTrack, int returnTrack) const {
    for (const auto& send : sends_) {
        if (send.sourceTrackIndex == sourceTrack && send.returnTrackIndex == returnTrack) {
            return send.amount;
        }
    }
    return 0.0f;
}

std::vector<SendReturn::Send> SendReturn::getSendsForTrack(int sourceTrack) const {
    std::vector<Send> result;
    for (const auto& send : sends_) {
        if (send.sourceTrackIndex == sourceTrack) {
            result.push_back(send);
        }
    }
    return result;
}

std::vector<SendReturn::Send> SendReturn::getReturnsForTrack(int returnTrack) const {
    std::vector<Send> result;
    for (const auto& send : sends_) {
        if (send.returnTrackIndex == returnTrack) {
            result.push_back(send);
        }
    }
    return result;
}

void SendReturn::processSends(const std::vector<Core::AudioBuffer*>& trackBuffers,
                              std::vector<Core::AudioBuffer>& returnBuffers) {
    // Clear return buffers
    for (auto& returnBuffer : returnBuffers) {
        returnBuffer.clear();
    }

    // Process sends
    for (const auto& send : sends_) {
        if (send.sourceTrackIndex >= 0 &&
            send.sourceTrackIndex < static_cast<int>(trackBuffers.size()) &&
            send.returnTrackIndex >= 0 &&
            send.returnTrackIndex < static_cast<int>(returnBuffers.size()) &&
            send.amount > 0.0f) {

            auto* source = trackBuffers[send.sourceTrackIndex];
            auto& destination = returnBuffers[send.returnTrackIndex];

            if (source && destination.getNumSamples() == source->getNumSamples()) {
                int numChannels = std::min(source->getNumChannels(), destination.getNumChannels());
                int numSamples = source->getNumSamples();

                for (int ch = 0; ch < numChannels; ++ch) {
                    const float* src = source->getReadPointer(ch);
                    float* dst = destination.getWritePointer(ch);

                    for (int i = 0; i < numSamples; ++i) {
                        dst[i] += src[i] * send.amount;
                    }
                }
            }
        }
    }
}

} // namespace Plugins
} // namespace MolinAntro
