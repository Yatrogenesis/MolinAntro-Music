# MolinAntro DAW

## Professional Digital Audio Workstation

**Status**: 🔒 Private Development | **Version**: 3.0.0-ACME | **License**: Proprietary

---

## Overview

MolinAntro DAW is a next-generation digital audio workstation designed to surpass industry leaders while providing specialized capabilities for:

- 🎵 **Commercial Music Production**
- 🎬 **Post-Production & Film Audio**
- 🔬 **Forensic Audio Analysis**
- 🛡️ **Military-Grade Signal Processing**
- 🎓 **Academic Research**

---

## Key Features

### Audio Engine
- **Ultra-low latency**: 32-64 samples @ 48 kHz
- **High resolution**: Up to 64-bit float, 384 kHz (768 kHz experimental)
- **Multi-format support**: VST2/3, AU, AAX, CLAP, LV2
- **GPU acceleration**: CUDA/Metal/OpenCL

### 🧠 Real AI Features (ACME Edition)

MolinAntro DAW ACME Edition integrates **Microsoft ONNX Runtime** for professional-grade neural inference.

### Requirements
- **Models**: You must download the required `.onnx` model weights to `models/`.
  - `hubert_base.onnx` (Content Encoder)
  - `final_rvc.onnx` (RVC Generator)
  - `mastering_v1.onnx` (Neural Mastering)
- **Use the Script**: Run `scripts/download_models.sh` to setup initial placeholders or download real weights.

### DSP Logic
- **Voice Cloning**: Uses RVC (Retrieval-based Voice Conversion) V2 architecture.
  - Fallback: High-quality Phase Vocoder / Granular Pitch Shifting (if models missing).
- **Mastering**: Hybrid Neural + Analog-Modeled DSP.
  - 64-bit precision Biquad IIR Filters.
  - Lookahead Limiter for transparent loudness maximization.

### Advanced Capabilities
- 📊 **Spectral Editing** - Frequency-domain visual editing
- 🔍 **Forensic Analysis** - Watermark detection, ENF analysis, authentication
- 🔐 **Military-Grade Security** - AES-256 encryption, secure boot

### Workflow Features
- **Session View** (Ableton-style clip launching)
- **Modular Rack** (Reason-style virtual cables)
- **Advanced Piano Roll** (FL Studio-inspired)
- **Spectral Waveform Editor** (Adobe Audition-level)

---

## System Requirements

### Minimum
- **CPU**: Intel Core i5 (8th gen) / AMD Ryzen 5 3600
- **RAM**: 16 GB DDR4
- **Storage**: 20 GB SSD
- **OS**: Windows 10 (64-bit), macOS 12.0+, Ubuntu 22.04+

### Recommended
- **CPU**: Intel Core i9 / AMD Ryzen 9 / Apple M2/M3 Pro
- **RAM**: 32 GB DDR5
- **Storage**: 100 GB NVMe SSD
- **GPU**: NVIDIA RTX 3060 / AMD RX 6700 XT / Apple Silicon GPU

---

## Build Instructions

### Prerequisites
- CMake 3.20+
- C++20 compatible compiler (Clang 15+, GCC 11+, MSVC 19.30+)
- ONNX Runtime 1.16+
- PortAudio, JUCE, Qt6

### Building from Source

The project includes a comprehensive build script located at `scripts/build.sh`.

```bash
# Make the script executable
chmod +x scripts/build.sh

# Build the project (Release mode by default)
./scripts/build.sh

# Run the project after building
./build/release/MolinAntro
```

---

## Documentation

- 📘 **[Technical Design Document](./TDD-MolinAntro-DAW.md)** - Complete technical specifications
- 📗 **User Manual** *(Coming Soon)*
- 📙 **Plugin Developer SDK** *(Coming Soon)*
- 📕 **API Reference** *(Coming Soon)*

---

## Development Status

### Phase 1: Foundation (Completed)
- [x] Core audio engine
- [x] Basic UI framework
- [x] VST3 hosting
- [x] MIDI sequencing
- [x] Transport controls

### Phase 2: Professional Features (Completed)
- [x] Advanced mixer
- [x] Effects suite
- [x] Sampler & Synthesizer
- [x] Cloud sync
- [x] Beta release

### Phase 3: Advanced & Specialized (Current)
- [x] AI features (Voice Cloning, Mastering)
- [/] Forensic module
- [x] GPU acceleration
- [ ] v3.0 Release Candidate

---

## Technology Stack

- **Languages**: C++20 (70%), Python (15%), JavaScript/TypeScript (10%), Rust (5%)
- **Frameworks**: JUCE, Qt6, PortAudio
- **DSP Libraries**: FFTW3, RubberBand, libsamplerate
- **ML Libraries**: ONNX Runtime, TensorFlow Lite
- **Build System**: CMake 3.20+, VCPKG

---

## License

**Proprietary Software** - All rights reserved © 2026 MolinAntro Technologies

This software is confidential and proprietary. Unauthorized copying, distribution, or modification is strictly prohibited.

---

## Contact

- **Company**: MolinAntro Technologies
- **Website**: *(Coming Soon)*
- **Support**: *(Coming Soon)*
- **Sales**: *(Coming Soon)*

---

## Security Notice

⚠️ **This repository contains proprietary information**

- Do not share credentials or access
- Report security vulnerabilities immediately
- All code reviews must be conducted securely
- Export restrictions may apply for cryptographic modules

---

**Last Updated**: 2026-01-12
