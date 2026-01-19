/**
 * @file DemucsONNX.cpp
 * @brief SOTA Stem Separation using Demucs/HTDemucs ONNX models
 *
 * Implementation of state-of-the-art source separation.
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#include "dsp/DemucsONNX.h"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Conditional ONNX Runtime include
#ifdef ENABLE_ONNX_RUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace MolinAntro {
namespace DSP {

// ============================================================================
// ONNX Runtime Components (internal structure)
// ============================================================================

struct DemucsONNX::OrtComponents {
#ifdef ENABLE_ONNX_RUNTIME
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Model metadata
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
#endif

    bool isValid = false;
    std::string modelInfo;
    std::string gpuInfo;
    bool gpuAvailable = false;
};

// ============================================================================
// DemucsONNX Implementation
// ============================================================================

DemucsONNX::DemucsONNX()
    : ort_(std::make_unique<OrtComponents>())
{
    initializeONNX();
}

DemucsONNX::~DemucsONNX() = default;

DemucsONNX::DemucsONNX(DemucsONNX&&) noexcept = default;
DemucsONNX& DemucsONNX::operator=(DemucsONNX&&) noexcept = default;

void DemucsONNX::initializeONNX() {
#ifdef ENABLE_ONNX_RUNTIME
    try {
        ort_->env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DemucsONNX");
        ort_->sessionOptions = std::make_unique<Ort::SessionOptions>();

        // Set optimization level
        ort_->sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Check for GPU support
        std::vector<std::string> providers = Ort::GetAvailableProviders();
        for (const auto& provider : providers) {
            if (provider == "CUDAExecutionProvider" ||
                provider == "DmlExecutionProvider" ||
                provider == "CoreMLExecutionProvider") {
                ort_->gpuAvailable = true;
                ort_->gpuInfo = provider;
                break;
            }
        }

        std::cout << "[DemucsONNX] ONNX Runtime initialized\n";
        std::cout << "[DemucsONNX] GPU available: " << (ort_->gpuAvailable ? ort_->gpuInfo : "No") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "[DemucsONNX] ONNX Runtime initialization failed: " << e.what() << "\n";
    }
#else
    std::cout << "[DemucsONNX] ONNX Runtime not enabled at compile time\n";
    std::cout << "[DemucsONNX] Will use fallback DSP algorithms\n";
#endif
}

void DemucsONNX::setupExecutionProvider() {
#ifdef ENABLE_ONNX_RUNTIME
    if (config_.useGPU && ort_->gpuAvailable) {
        try {
#ifdef USE_CUDA
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id = config_.gpuDeviceId;
            ort_->sessionOptions->AppendExecutionProvider_CUDA(cudaOptions);
            std::cout << "[DemucsONNX] Using CUDA execution provider\n";
#elif defined(USE_DIRECTML)
            ort_->sessionOptions->AppendExecutionProvider_DML(config_.gpuDeviceId);
            std::cout << "[DemucsONNX] Using DirectML execution provider\n";
#elif defined(__APPLE__)
            ort_->sessionOptions->AppendExecutionProvider_CoreML(0);
            std::cout << "[DemucsONNX] Using CoreML execution provider\n";
#endif
        } catch (const std::exception& e) {
            std::cerr << "[DemucsONNX] GPU provider failed, falling back to CPU: " << e.what() << "\n";
        }
    }

    // CPU settings
    ort_->sessionOptions->SetIntraOpNumThreads(config_.numThreads);
    ort_->sessionOptions->SetInterOpNumThreads(config_.numThreads);
#endif
}

bool DemucsONNX::initialize(const DemucsConfig& config) {
    config_ = config;

    if (!config.modelPath.empty()) {
        return loadModel(config.modelPath, config.modelType);
    }

    // Create STFT window
    createWindow(config_.winLength);

    initialized_ = true;
    return true;
}

bool DemucsONNX::loadModel(const std::string& modelPath, DemucsModelType modelType) {
#ifdef ENABLE_ONNX_RUNTIME
    try {
        // Check if file exists
        std::ifstream file(modelPath);
        if (!file.good()) {
            std::cerr << "[DemucsONNX] Model file not found: " << modelPath << "\n";
            return false;
        }
        file.close();

        config_.modelPath = modelPath;
        config_.modelType = modelType;

        // Setup execution provider based on config
        setupExecutionProvider();

        // Create session
#ifdef _WIN32
        std::wstring wpath(modelPath.begin(), modelPath.end());
        ort_->session = std::make_unique<Ort::Session>(*ort_->env, wpath.c_str(), *ort_->sessionOptions);
#else
        ort_->session = std::make_unique<Ort::Session>(*ort_->env, modelPath.c_str(), *ort_->sessionOptions);
#endif

        // Get model info
        Ort::AllocatorWithDefaultOptions allocator;

        // Input info
        size_t numInputs = ort_->session->GetInputCount();
        ort_->inputNames.resize(numInputs);
        ort_->inputShapes.resize(numInputs);

        for (size_t i = 0; i < numInputs; ++i) {
            auto inputName = ort_->session->GetInputNameAllocated(i, allocator);
            ort_->inputNames[i] = strdup(inputName.get());

            auto typeInfo = ort_->session->GetInputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            ort_->inputShapes[i] = tensorInfo.GetShape();
        }

        // Output info
        size_t numOutputs = ort_->session->GetOutputCount();
        ort_->outputNames.resize(numOutputs);
        ort_->outputShapes.resize(numOutputs);

        for (size_t i = 0; i < numOutputs; ++i) {
            auto outputName = ort_->session->GetOutputNameAllocated(i, allocator);
            ort_->outputNames[i] = strdup(outputName.get());

            auto typeInfo = ort_->session->GetOutputTypeInfo(i);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            ort_->outputShapes[i] = tensorInfo.GetShape();
        }

        ort_->isValid = true;

        // Build model info string
        ort_->modelInfo = "Demucs ONNX Model\n";
        ort_->modelInfo += "  Path: " + modelPath + "\n";
        ort_->modelInfo += "  Inputs: " + std::to_string(numInputs) + "\n";
        ort_->modelInfo += "  Outputs: " + std::to_string(numOutputs) + " (stems)\n";

        std::cout << "[DemucsONNX] Model loaded successfully\n";
        std::cout << ort_->modelInfo;

        initialized_ = true;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "[DemucsONNX] ONNX error: " << e.what() << "\n";
        return false;
    } catch (const std::exception& e) {
        std::cerr << "[DemucsONNX] Error loading model: " << e.what() << "\n";
        return false;
    }
#else
    std::cerr << "[DemucsONNX] ONNX Runtime not available\n";
    return false;
#endif
}

bool DemucsONNX::isReady() const {
    return initialized_ && ort_->isValid;
}

std::string DemucsONNX::getModelInfo() const {
    return ort_->modelInfo;
}

bool DemucsONNX::isGPUAvailable() const {
    return ort_->gpuAvailable;
}

void DemucsONNX::setUseGPU(bool useGPU) {
    config_.useGPU = useGPU;
}

std::string DemucsONNX::getGPUInfo() const {
    return ort_->gpuInfo;
}

void DemucsONNX::setConfig(const DemucsConfig& config) {
    config_ = config;
}

void DemucsONNX::reset() {
    overlapBuffer_.clear();
    overlapSamples_ = 0;
}

// ============================================================================
// Audio Preprocessing
// ============================================================================

std::vector<float> DemucsONNX::preprocessAudio(const Core::AudioBuffer& input) {
    int numSamples = input.getNumSamples();
    int numChannels = input.getNumChannels();

    // Convert to interleaved stereo
    std::vector<float> audio(numSamples * 2);

    if (numChannels >= 2) {
        // Stereo input
        const float* left = input.getReadPointer(0);
        const float* right = input.getReadPointer(1);
        for (int i = 0; i < numSamples; ++i) {
            audio[i * 2] = left[i];
            audio[i * 2 + 1] = right[i];
        }
    } else {
        // Mono input - duplicate to stereo
        const float* mono = input.getReadPointer(0);
        for (int i = 0; i < numSamples; ++i) {
            audio[i * 2] = mono[i];
            audio[i * 2 + 1] = mono[i];
        }
    }

    // Resample if needed
    // TODO: Implement proper resampling (libsamplerate or custom)

    // Normalize
    normalizeAudio(audio);

    return audio;
}

void DemucsONNX::normalizeAudio(std::vector<float>& audio) {
    // Find peak
    float peak = 0.0f;
    for (float sample : audio) {
        peak = std::max(peak, std::abs(sample));
    }

    // Normalize to -1dB headroom
    if (peak > 0.001f) {
        float gain = 0.891f / peak;  // -1dB
        for (float& sample : audio) {
            sample *= gain;
        }
    }
}

void DemucsONNX::resampleIfNeeded(std::vector<float>& audio, int sourceSR, int targetSR) {
    if (sourceSR == targetSR) return;

    // Simple linear interpolation resampling
    // For production, use libsamplerate or similar
    double ratio = static_cast<double>(targetSR) / sourceSR;
    int newSize = static_cast<int>(audio.size() * ratio);
    std::vector<float> resampled(newSize);

    for (int i = 0; i < newSize; ++i) {
        double srcPos = i / ratio;
        int idx = static_cast<int>(srcPos);
        double frac = srcPos - idx;

        if (idx + 1 < static_cast<int>(audio.size())) {
            resampled[i] = static_cast<float>(audio[idx] * (1.0 - frac) + audio[idx + 1] * frac);
        } else {
            resampled[i] = audio[idx];
        }
    }

    audio = std::move(resampled);
}

// ============================================================================
// STFT Operations
// ============================================================================

void DemucsONNX::createWindow(int size) {
    window_.resize(size);
    // Hann window
    for (int i = 0; i < size; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
}

void DemucsONNX::computeSTFT(const std::vector<float>& audio,
                             std::vector<std::complex<float>>& spectrum) {
    int numFrames = (audio.size() - config_.winLength) / config_.hopLength + 1;
    int numBins = config_.nfft / 2 + 1;

    spectrum.resize(numFrames * numBins);

    // Simple DFT (for production, use FFTW or similar)
    std::vector<float> frame(config_.nfft, 0.0f);
    std::vector<std::complex<float>> fftOut(config_.nfft);

    for (int f = 0; f < numFrames; ++f) {
        int offset = f * config_.hopLength;

        // Apply window
        for (int i = 0; i < config_.winLength && offset + i < static_cast<int>(audio.size()); ++i) {
            frame[i] = audio[offset + i] * window_[i];
        }

        // Zero-pad
        for (int i = config_.winLength; i < config_.nfft; ++i) {
            frame[i] = 0.0f;
        }

        // DFT
        for (int k = 0; k < numBins; ++k) {
            std::complex<float> sum(0.0f, 0.0f);
            for (int n = 0; n < config_.nfft; ++n) {
                float angle = -2.0f * M_PI * k * n / config_.nfft;
                sum += frame[n] * std::complex<float>(std::cos(angle), std::sin(angle));
            }
            spectrum[f * numBins + k] = sum;
        }
    }
}

void DemucsONNX::computeISTFT(const std::vector<std::complex<float>>& spectrum,
                              std::vector<float>& audio) {
    int numBins = config_.nfft / 2 + 1;
    int numFrames = spectrum.size() / numBins;
    int outputLength = (numFrames - 1) * config_.hopLength + config_.winLength;

    audio.resize(outputLength, 0.0f);
    std::vector<float> windowSum(outputLength, 0.0f);

    std::vector<float> frame(config_.nfft);

    for (int f = 0; f < numFrames; ++f) {
        // IDFT
        for (int n = 0; n < config_.nfft; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < numBins; ++k) {
                float angle = 2.0f * M_PI * k * n / config_.nfft;
                auto& bin = spectrum[f * numBins + k];
                sum += bin.real() * std::cos(angle) - bin.imag() * std::sin(angle);

                // Conjugate symmetry for real signal
                if (k > 0 && k < numBins - 1) {
                    sum += bin.real() * std::cos(angle) + bin.imag() * std::sin(angle);
                }
            }
            frame[n] = sum / config_.nfft;
        }

        // Overlap-add with window
        int offset = f * config_.hopLength;
        for (int i = 0; i < config_.winLength && offset + i < outputLength; ++i) {
            audio[offset + i] += frame[i] * window_[i];
            windowSum[offset + i] += window_[i] * window_[i];
        }
    }

    // Normalize by window sum
    for (int i = 0; i < outputLength; ++i) {
        if (windowSum[i] > 1e-8f) {
            audio[i] /= windowSum[i];
        }
    }
}

// ============================================================================
// Neural Inference
// ============================================================================

std::vector<std::vector<float>> DemucsONNX::runInference(const std::vector<float>& input) {
#ifdef ENABLE_ONNX_RUNTIME
    if (!ort_->isValid || !ort_->session) {
        std::cerr << "[DemucsONNX] Model not loaded\n";
        return {};
    }

    try {
        // Prepare input tensor
        // Shape: [batch, channels, samples]
        int numSamples = input.size() / 2;  // Stereo
        std::vector<int64_t> inputShape = {1, 2, numSamples};

        // Deinterleave to channels-first format
        std::vector<float> inputData(input.size());
        for (int i = 0; i < numSamples; ++i) {
            inputData[i] = input[i * 2];                    // Left channel
            inputData[numSamples + i] = input[i * 2 + 1];   // Right channel
        }

        // Create input tensor
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            ort_->memoryInfo,
            inputData.data(),
            inputData.size(),
            inputShape.data(),
            inputShape.size()
        );

        // Run inference
        auto outputTensors = ort_->session->Run(
            Ort::RunOptions{nullptr},
            ort_->inputNames.data(),
            &inputTensor,
            1,
            ort_->outputNames.data(),
            ort_->outputNames.size()
        );

        // Process output
        // Expected shape: [batch, stems, channels, samples]
        std::vector<std::vector<float>> stems(config_.numStems);

        for (size_t s = 0; s < outputTensors.size() && s < static_cast<size_t>(config_.numStems); ++s) {
            auto& tensor = outputTensors[s];
            auto* data = tensor.GetTensorData<float>();
            auto shape = tensor.GetTensorTypeAndShapeInfo().GetShape();

            int stemSamples = shape.back();
            stems[s].resize(stemSamples * 2);  // Stereo interleaved

            // Convert from channels-first to interleaved
            for (int i = 0; i < stemSamples; ++i) {
                stems[s][i * 2] = data[i];                      // Left
                stems[s][i * 2 + 1] = data[stemSamples + i];    // Right
            }
        }

        return stems;

    } catch (const Ort::Exception& e) {
        std::cerr << "[DemucsONNX] Inference error: " << e.what() << "\n";
        return {};
    }
#else
    return {};
#endif
}

std::vector<std::vector<float>> DemucsONNX::runHybridInference(
    const std::vector<float>& timeDomain,
    const std::vector<std::complex<float>>& freqDomain)
{
    // HTDemucs uses both time and frequency domain
    // For now, just use time domain
    return runInference(timeDomain);
}

// ============================================================================
// Mask Estimation (for spectrogram-based models like Spleeter/Open-Unmix)
// ============================================================================

std::vector<std::vector<float>> DemucsONNX::estimateMasks(
    const std::vector<std::complex<float>>& mixture)
{
    int numBins = config_.nfft / 2 + 1;
    int numFrames = mixture.size() / numBins;

    std::vector<std::vector<float>> masks(config_.numStems);
    for (auto& mask : masks) {
        mask.resize(mixture.size(), 0.25f);  // Default uniform
    }

#ifdef ENABLE_ONNX_RUNTIME
    if (ort_->isValid) {
        // Run mask estimation model
        // ... (model-specific implementation)
    }
#endif

    return masks;
}

void DemucsONNX::applySoftmask(std::vector<std::complex<float>>& spectrum,
                                const std::vector<float>& mask,
                                float temperature) {
    for (size_t i = 0; i < spectrum.size() && i < mask.size(); ++i) {
        float m = std::exp(mask[i] / temperature);
        spectrum[i] *= m;
    }
}

// ============================================================================
// Main Separation Function
// ============================================================================

SeparatedStems DemucsONNX::separate(const Core::AudioBuffer& input) {
    return separate(input, nullptr);
}

SeparatedStems DemucsONNX::separate(const Core::AudioBuffer& input, ProgressCallback callback) {
    auto startTime = std::chrono::high_resolution_clock::now();

    SeparatedStems result;

    if (callback) callback(0.0f, "Preprocessing audio...");

    // Preprocess
    std::vector<float> audio = preprocessAudio(input);

    if (callback) callback(0.1f, "Running neural network...");

    std::vector<std::vector<float>> stems;

#ifdef ENABLE_ONNX_RUNTIME
    if (ort_->isValid) {
        // Use neural network
        if (config_.modelType == DemucsModelType::HTDemucs ||
            config_.modelType == DemucsModelType::Demucs_v3) {
            // Hybrid model: compute STFT
            std::vector<std::complex<float>> spectrum;
            computeSTFT(audio, spectrum);
            stems = runHybridInference(audio, spectrum);
        } else {
            // Time-domain model
            stems = runInference(audio);
        }

        result.usedGPU = config_.useGPU && ort_->gpuAvailable;
    }
#endif

    // Fallback to DSP-based separation if no model or failed
    if (stems.empty()) {
        if (callback) callback(0.3f, "Using DSP fallback...");
        stems = separateWithDSP(audio);
    }

    if (callback) callback(0.7f, "Postprocessing stems...");

    // Postprocess
    postprocessStems(stems);

    // Convert to AudioBuffers
    int numSamples = input.getNumSamples();

    if (stems.size() > 0) {
        result.vocals = vectorToBuffer(stems[0], 2);
        result.confidence[0] = calculateConfidence(stems[0], audio);
    }
    if (stems.size() > 1) {
        result.drums = vectorToBuffer(stems[1], 2);
        result.confidence[1] = calculateConfidence(stems[1], audio);
    }
    if (stems.size() > 2) {
        result.bass = vectorToBuffer(stems[2], 2);
        result.confidence[2] = calculateConfidence(stems[2], audio);
    }
    if (stems.size() > 3) {
        result.other = vectorToBuffer(stems[3], 2);
        result.confidence[3] = calculateConfidence(stems[3], audio);
    }
    if (stems.size() > 4) {
        result.piano = vectorToBuffer(stems[4], 2);
        result.confidence[4] = calculateConfidence(stems[4], audio);
    }
    if (stems.size() > 5) {
        result.guitar = vectorToBuffer(stems[5], 2);
        result.confidence[5] = calculateConfidence(stems[5], audio);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    result.processingTimeMs = static_cast<float>(duration.count());

    if (callback) callback(1.0f, "Separation complete!");

    std::cout << "[DemucsONNX] Separation completed in " << result.processingTimeMs << " ms\n";
    std::cout << "[DemucsONNX] Used GPU: " << (result.usedGPU ? "Yes" : "No") << "\n";

    return result;
}

// DSP Fallback (when ONNX model not available)
std::vector<std::vector<float>> DemucsONNX::separateWithDSP(const std::vector<float>& audio) {
    // This is the fallback using classical DSP
    // Uses HPSS (Harmonic-Percussive Source Separation) + frequency bands

    std::vector<std::vector<float>> stems(4);
    int numSamples = audio.size() / 2;

    // Initialize output stems
    for (auto& stem : stems) {
        stem.resize(audio.size(), 0.0f);
    }

    // Compute STFT
    std::vector<std::complex<float>> spectrum;
    computeSTFT(audio, spectrum);

    int numBins = config_.nfft / 2 + 1;
    int numFrames = spectrum.size() / numBins;

    // Frequency boundaries (in bins)
    float binWidth = static_cast<float>(config_.sampleRate) / config_.nfft;
    int bassMaxBin = static_cast<int>(250.0f / binWidth);
    int vocalsMinBin = static_cast<int>(200.0f / binWidth);
    int vocalsMaxBin = static_cast<int>(4000.0f / binWidth);

    // Compute HPSS-style masks
    // Harmonic = median filter along time (vertical smoothing)
    // Percussive = median filter along frequency (horizontal smoothing)

    std::vector<std::vector<float>> magnitudes(numFrames, std::vector<float>(numBins));
    std::vector<std::vector<float>> harmonicMask(numFrames, std::vector<float>(numBins));
    std::vector<std::vector<float>> percussiveMask(numFrames, std::vector<float>(numBins));

    // Extract magnitudes
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            magnitudes[f][b] = std::abs(spectrum[f * numBins + b]);
        }
    }

    // Compute harmonic mask (median along time - 7 frames)
    int medianSize = 7;
    std::vector<float> medianBuffer(medianSize);

    for (int b = 0; b < numBins; ++b) {
        for (int f = 0; f < numFrames; ++f) {
            int count = 0;
            for (int k = -medianSize/2; k <= medianSize/2; ++k) {
                int ff = std::clamp(f + k, 0, numFrames - 1);
                medianBuffer[count++] = magnitudes[ff][b];
            }
            std::sort(medianBuffer.begin(), medianBuffer.begin() + count);
            harmonicMask[f][b] = medianBuffer[count/2];
        }
    }

    // Compute percussive mask (median along frequency - 7 bins)
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            int count = 0;
            for (int k = -medianSize/2; k <= medianSize/2; ++k) {
                int bb = std::clamp(b + k, 0, numBins - 1);
                medianBuffer[count++] = magnitudes[f][bb];
            }
            std::sort(medianBuffer.begin(), medianBuffer.begin() + count);
            percussiveMask[f][b] = medianBuffer[count/2];
        }
    }

    // Create soft masks using Wiener filtering
    // M_h = H^2 / (H^2 + P^2), M_p = P^2 / (H^2 + P^2)
    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            float h2 = harmonicMask[f][b] * harmonicMask[f][b];
            float p2 = percussiveMask[f][b] * percussiveMask[f][b];
            float denom = h2 + p2 + 1e-10f;
            harmonicMask[f][b] = h2 / denom;
            percussiveMask[f][b] = p2 / denom;
        }
    }

    // Apply masks to create stems
    std::vector<std::complex<float>> vocalSpec(spectrum.size());
    std::vector<std::complex<float>> drumSpec(spectrum.size());
    std::vector<std::complex<float>> bassSpec(spectrum.size());
    std::vector<std::complex<float>> otherSpec(spectrum.size());

    for (int f = 0; f < numFrames; ++f) {
        for (int b = 0; b < numBins; ++b) {
            auto& orig = spectrum[f * numBins + b];

            // Bass: low frequency + harmonic
            if (b <= bassMaxBin) {
                bassSpec[f * numBins + b] = orig * harmonicMask[f][b] * 0.8f;
            }

            // Drums: percussive component
            drumSpec[f * numBins + b] = orig * percussiveMask[f][b];

            // Vocals: mid-frequency harmonic content
            if (b >= vocalsMinBin && b <= vocalsMaxBin) {
                vocalSpec[f * numBins + b] = orig * harmonicMask[f][b] * 0.7f;
            }

            // Other: residual
            float usedMask = 0.0f;
            if (b <= bassMaxBin) usedMask += harmonicMask[f][b] * 0.8f;
            usedMask += percussiveMask[f][b];
            if (b >= vocalsMinBin && b <= vocalsMaxBin) usedMask += harmonicMask[f][b] * 0.7f;

            otherSpec[f * numBins + b] = orig * std::max(0.0f, 1.0f - usedMask);
        }
    }

    // ISTFT to time domain
    std::vector<float> vocals, drums, bass, other;
    computeISTFT(vocalSpec, vocals);
    computeISTFT(drumSpec, drums);
    computeISTFT(bassSpec, bass);
    computeISTFT(otherSpec, other);

    // Convert to stereo interleaved
    auto toStereo = [](const std::vector<float>& mono, std::vector<float>& stereo) {
        stereo.resize(mono.size() * 2);
        for (size_t i = 0; i < mono.size(); ++i) {
            stereo[i * 2] = mono[i];
            stereo[i * 2 + 1] = mono[i];
        }
    };

    toStereo(vocals, stems[0]);
    toStereo(drums, stems[1]);
    toStereo(bass, stems[2]);
    toStereo(other, stems[3]);

    return stems;
}

float DemucsONNX::calculateConfidence(const std::vector<float>& stem, const std::vector<float>& mixture) {
    // Simple energy ratio as confidence
    float stemEnergy = 0.0f;
    float mixEnergy = 0.0f;

    for (size_t i = 0; i < std::min(stem.size(), mixture.size()); ++i) {
        stemEnergy += stem[i] * stem[i];
        mixEnergy += mixture[i] * mixture[i];
    }

    if (mixEnergy < 1e-10f) return 0.0f;
    return std::min(1.0f, stemEnergy / mixEnergy);
}

std::unique_ptr<Core::AudioBuffer> DemucsONNX::extractStem(const Core::AudioBuffer& input, StemType stem) {
    auto result = separate(input);

    switch (stem) {
        case StemType::Vocals: return std::move(result.vocals);
        case StemType::Drums:  return std::move(result.drums);
        case StemType::Bass:   return std::move(result.bass);
        case StemType::Other:  return std::move(result.other);
        case StemType::Piano:  return std::move(result.piano);
        case StemType::Guitar: return std::move(result.guitar);
    }

    return nullptr;
}

void DemucsONNX::postprocessStems(std::vector<std::vector<float>>& stems) {
    // Apply gentle limiting to prevent clipping
    const float threshold = 0.95f;
    const float knee = 0.05f;

    for (auto& stem : stems) {
        for (float& sample : stem) {
            float absVal = std::abs(sample);
            if (absVal > threshold) {
                // Soft knee limiter
                float overshoot = absVal - threshold;
                float gain = threshold + knee * (1.0f - std::exp(-overshoot / knee));
                sample = (sample > 0 ? 1.0f : -1.0f) * std::min(gain, 0.99f);
            }
        }
    }
}

std::unique_ptr<Core::AudioBuffer> DemucsONNX::vectorToBuffer(const std::vector<float>& data, int channels) {
    int numSamples = data.size() / channels;
    auto buffer = std::make_unique<Core::AudioBuffer>(channels, numSamples);

    for (int ch = 0; ch < channels; ++ch) {
        float* out = buffer->getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i) {
            out[i] = data[i * channels + ch];
        }
    }

    return buffer;
}

// ============================================================================
// Real-time Streaming
// ============================================================================

void DemucsONNX::prepareRealtime(int sampleRate, int blockSize) {
    config_.sampleRate = sampleRate;
    config_.segmentLength = blockSize * 8;  // Process 8 blocks at a time
    config_.overlap = blockSize * 2;

    overlapBuffer_.resize(config_.overlap * 2, 0.0f);  // Stereo
    overlapSamples_ = 0;

    createWindow(config_.winLength);
}

void DemucsONNX::processBlock(const Core::AudioBuffer& input,
                               std::array<Core::AudioBuffer*, 4>& outputs,
                               int blockSize) {
    // Accumulate input
    // When we have enough samples, process and output with overlap-add

    // Simplified real-time processing
    auto result = separate(input);

    if (result.vocals && outputs[0]) outputs[0]->copyFrom(*result.vocals, 0, 0);
    if (result.drums && outputs[1])  outputs[1]->copyFrom(*result.drums, 0, 0);
    if (result.bass && outputs[2])   outputs[2]->copyFrom(*result.bass, 0, 0);
    if (result.other && outputs[3])  outputs[3]->copyFrom(*result.other, 0, 0);
}

// ============================================================================
// Utility Functions
// ============================================================================

float DemucsONNX::calculateSDR(const std::vector<float>& reference,
                                const std::vector<float>& estimate) {
    // Signal-to-Distortion Ratio
    float signalPower = 0.0f;
    float noisePower = 0.0f;

    for (size_t i = 0; i < std::min(reference.size(), estimate.size()); ++i) {
        signalPower += reference[i] * reference[i];
        float noise = reference[i] - estimate[i];
        noisePower += noise * noise;
    }

    if (noisePower < 1e-10f) return 100.0f;  // Perfect separation
    return 10.0f * std::log10(signalPower / noisePower);
}

// ============================================================================
// DemucsFactory Implementation
// ============================================================================

std::unique_ptr<DemucsONNX> DemucsFactory::createHighQuality(const std::string& modelPath) {
    auto instance = std::make_unique<DemucsONNX>();

    DemucsConfig config;
    config.modelPath = modelPath;
    config.modelType = DemucsModelType::HTDemucs;
    config.useGPU = true;
    config.numStems = 4;
    config.segmentLength = 44100 * 10;  // 10 seconds for best quality
    config.overlap = 44100 * 2;         // 2 second overlap

    instance->initialize(config);
    return instance;
}

std::unique_ptr<DemucsONNX> DemucsFactory::createFast(const std::string& modelPath) {
    auto instance = std::make_unique<DemucsONNX>();

    DemucsConfig config;
    config.modelPath = modelPath;
    config.modelType = DemucsModelType::Demucs_v2;
    config.useGPU = true;
    config.numStems = 4;
    config.segmentLength = 44100 * 5;   // 5 seconds
    config.overlap = 44100;             // 1 second overlap

    instance->initialize(config);
    return instance;
}

std::unique_ptr<DemucsONNX> DemucsFactory::createRealtime(const std::string& modelPath, int sampleRate) {
    auto instance = std::make_unique<DemucsONNX>();

    DemucsConfig config;
    config.modelPath = modelPath;
    config.modelType = DemucsModelType::Demucs_v2;
    config.sampleRate = sampleRate;
    config.useGPU = true;
    config.numStems = 4;
    config.segmentLength = sampleRate * 2;  // 2 seconds
    config.overlap = sampleRate / 2;        // 0.5 second overlap

    instance->initialize(config);
    instance->prepareRealtime(sampleRate, 512);
    return instance;
}

std::unique_ptr<DemucsONNX> DemucsFactory::create5Stem(const std::string& modelPath) {
    auto instance = std::make_unique<DemucsONNX>();

    DemucsConfig config;
    config.modelPath = modelPath;
    config.modelType = DemucsModelType::HTDemucs;
    config.useGPU = true;
    config.numStems = 5;  // Vocals, Drums, Bass, Piano, Other
    config.segmentLength = 44100 * 10;

    instance->initialize(config);
    return instance;
}

std::vector<std::string> DemucsFactory::listAvailableModels() {
    return {
        "htdemucs",          // Hybrid Transformer (best quality)
        "htdemucs_ft",       // Fine-tuned version
        "htdemucs_6s",       // 6-stem (with guitar)
        "mdx_extra",         // MDX-Net variant
        "demucs_v3",         // Hybrid time-frequency
        "demucs_v2",         // Time-domain only (faster)
        "spleeter_2stems",   // Vocals/accompaniment
        "spleeter_4stems",   // Vocals/drums/bass/other
        "spleeter_5stems",   // +piano
        "open_unmix"         // Open-Unmix
    };
}

std::string DemucsFactory::getRecommendedModel(bool realtime, int numStems) {
    if (realtime) {
        return numStems <= 4 ? "demucs_v2" : "spleeter_5stems";
    } else {
        return numStems <= 4 ? "htdemucs_ft" : "htdemucs_6s";
    }
}

bool DemucsFactory::downloadModel(const std::string& modelName, const std::string& outputPath) {
    // TODO: Implement model download from HuggingFace
    std::cout << "[DemucsFactory] Model download not yet implemented\n";
    std::cout << "[DemucsFactory] Please download manually from:\n";
    std::cout << "  https://huggingface.co/facebook/demucs\n";
    return false;
}

// ============================================================================
// DemucsBenchmark Implementation
// ============================================================================

DemucsBenchmark::BenchmarkResult DemucsBenchmark::benchmark(
    DemucsONNX& separator,
    const Core::AudioBuffer& testAudio,
    const SeparatedStems* groundTruth)
{
    BenchmarkResult result = {};

    auto start = std::chrono::high_resolution_clock::now();
    auto stems = separator.separate(testAudio);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    result.processingTimeMs = static_cast<float>(duration.count());

    // Calculate real-time factor
    float audioDuration = static_cast<float>(testAudio.getNumSamples()) / 44100.0f;
    result.realtimeFactor = audioDuration * 1000.0f / result.processingTimeMs;

    result.gpuMemoryMB = 0.0f;  // TODO: Query GPU memory usage
    result.cpuUsagePercent = 0.0f;  // TODO: Measure CPU usage

    // Calculate SDR if ground truth provided
    if (groundTruth) {
        // TODO: Implement SDR calculation against ground truth
    }

    std::cout << "[DemucsBenchmark] Processing time: " << result.processingTimeMs << " ms\n";
    std::cout << "[DemucsBenchmark] Real-time factor: " << result.realtimeFactor << "x\n";

    return result;
}

void DemucsBenchmark::compareWithReference(
    DemucsONNX& separator,
    const std::string& testFile,
    const std::string& referenceDir)
{
    // TODO: Load test file, separate, compare with reference stems
    std::cout << "[DemucsBenchmark] Reference comparison not yet implemented\n";
}

} // namespace DSP
} // namespace MolinAntro
