# MolinAntro DAW - Quick Start Guide

## Welcome to MolinAntro DAW v1.0.0

This guide will help you get started with MolinAntro DAW, your professional digital audio workstation.

---

## Table of Contents

1. [Installation](#installation)
2. [First Launch](#first-launch)
3. [Basic Concepts](#basic-concepts)
4. [Your First Session](#your-first-session)
5. [Transport Controls](#transport-controls)
6. [Working with Audio Files](#working-with-audio-files)
7. [Keyboard Shortcuts](#keyboard-shortcuts)
8. [Getting Help](#getting-help)

---

## Installation

### Linux (Ubuntu/Debian)

```bash
# Download the .deb package
wget https://molinantro.com/downloads/molinantro-daw_1.0.0_amd64.deb

# Install
sudo dpkg -i molinantro-daw_1.0.0_amd64.deb
sudo apt-get install -f  # Install dependencies

# Launch
MolinAntroDaw
```

### macOS

```bash
# Download the .dmg package
# Double-click to mount
# Drag MolinAntro DAW to Applications folder

# Launch from Applications or terminal
/Applications/MolinAntroDaw.app/Contents/MacOS/MolinAntroDaw
```

### Windows

```powershell
# Download the installer
# Run MolinAntroDaw-Setup-1.0.0.exe
# Follow installation wizard

# Launch from Start Menu or Desktop shortcut
```

### Build from Source

```bash
git clone https://github.com/molinantro/molinantro-daw.git
cd molinantro-daw
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j 4
sudo cmake --install .
```

---

## First Launch

When you first launch MolinAntro DAW, you'll see:

```
╔═══════════════════════════════════════════════════════╗
║         MolinAntro DAW - Console Interface           ║
║              Professional Audio Workstation           ║
╚═══════════════════════════════════════════════════════╝

Available commands:
  play      - Start playback
  stop      - Stop playback
  pause     - Pause playback
  record    - Start recording
  bpm <n>   - Set tempo (BPM)
  status    - Show current status
  help      - Show this help
  exit/quit - Exit application

MAW>
```

---

## Basic Concepts

### Audio Engine

The **Audio Engine** is the heart of MolinAntro DAW:
- Processes audio in real-time
- Supports sample rates from 8kHz to 384kHz
- Configurable buffer sizes (32-4096 samples)
- Ultra-low latency (down to 0.67ms)

### Transport

The **Transport** controls playback and recording:
- **Play**: Start playback from current position
- **Stop**: Stop and reset to beginning
- **Pause**: Pause at current position
- **Record**: Start recording input

### BPM (Tempo)

- Default: 120 BPM
- Range: 20-999 BPM
- Can be changed during playback

### Time Signature

- Default: 4/4
- Supported: Any time signature from 1/1 to 16/16

---

## Your First Session

### Step 1: Check Status

```
MAW> status
```

You'll see:

```
╔═══ STATUS ═══════════════════════════════════════╗
║ Engine State: Stopped                            ║
║ Sample Rate:  48000 Hz                           ║
║ Buffer Size:    512 samples                      ║
║ CPU Usage:      0.0 %                            ║
║                                                   ║
║ BPM:          120.00                             ║
║ Time Sig:     4/4                                ║
║ Position:     Bar   1 | Beat 1                   ║
║ Time:           0.00 sec                         ║
╚═══════════════════════════════════════════════════╝
```

### Step 2: Set Your Tempo

```
MAW> bpm 140
♪ BPM set to 140
```

### Step 3: Start Playback

```
MAW> play
▶ Playing...
```

### Step 4: Stop Playback

```
MAW> stop
■ Stopped
```

---

## Transport Controls

### Play

Starts playback from the current position.

```
MAW> play
```

**State**: Transport will be in **Playing** mode.

### Stop

Stops playback and resets the position to the beginning.

```
MAW> stop
```

**State**: Transport will be in **Stopped** mode, position reset to 0:0:0.

### Pause

Pauses playback at the current position without resetting.

```
MAW> pause
```

**State**: Transport will be in **Paused** mode.

### Record

Starts recording mode (playback + recording enabled).

```
MAW> record
```

**State**: Transport will be in **Recording** mode.

---

## Working with Audio Files

### Supported Formats

- **WAV** (8, 16, 24, 32-bit)
- **AIFF** (coming soon)
- **FLAC** (coming soon)
- **MP3** (coming soon)

### Loading Audio Files

Currently done via the C++ API:

```cpp
#include "dsp/AudioFile.h"

MolinAntro::DSP::AudioFile audioFile;
if (audioFile.load("path/to/audio.wav")) {
    std::cout << "Loaded: " << audioFile.getInfo().durationSeconds << " seconds\n";
}
```

### Saving Audio Files

```cpp
MolinAntro::Core::AudioBuffer buffer(2, 48000);
// ... fill buffer with audio data ...

MolinAntro::DSP::AudioFile audioFile;
audioFile.save("output.wav", buffer, 48000, 24);
```

---

## Keyboard Shortcuts

### Transport

- **Space**: Play/Pause toggle
- **Enter**: Stop
- **R**: Record
- **0**: Reset to beginning

### Navigation

- **Arrow Up/Down**: Adjust BPM
- **Page Up/Down**: Jump bars
- **Home**: Go to start
- **End**: Go to end

### System

- **Ctrl+Q**: Quit
- **Ctrl+S**: Save session
- **Ctrl+O**: Open session
- **F1**: Help

> **Note**: GUI keyboard shortcuts coming in v1.1.0

---

## Performance Tips

### Optimizing Latency

1. **Reduce buffer size** (32-64 samples for lowest latency)
2. **Use ASIO drivers** (Windows)
3. **Disable power saving** modes
4. **Close unnecessary applications**

### Reducing CPU Usage

1. **Freeze tracks** with heavy processing
2. **Increase buffer size** (512-1024 samples)
3. **Render effects** to audio
4. **Use lighter plugins**

### System Requirements

**Minimum**:
- CPU: Intel Core i5 (8th gen) / AMD Ryzen 5
- RAM: 8 GB
- Storage: 10 GB SSD

**Recommended**:
- CPU: Intel Core i9 / AMD Ryzen 9
- RAM: 32 GB
- Storage: 50 GB NVMe SSD

---

## Getting Help

### Documentation

- **User Manual**: `docs/user/USER_MANUAL.md`
- **API Reference**: `docs/api/API_REFERENCE.md`
- **Developer Guide**: `docs/developer/DEVELOPER_GUIDE.md`

### Online Resources

- **Website**: https://molinantro.com
- **Support**: support@molinantro.com
- **Forum**: https://forum.molinantro.com
- **GitHub**: https://github.com/molinantro/molinantro-daw

### Command Line Help

```bash
MolinAntroDaw --help
```

### In-Application Help

```
MAW> help
```

---

## What's Next?

Now that you're familiar with the basics:

1. **Explore advanced features** in the User Manual
2. **Learn about plugins** in the Plugin Guide
3. **Optimize your workflow** with keyboard shortcuts
4. **Join the community** on our forum

---

**Happy Music Making!** 🎵

---

*MolinAntro DAW v1.0.0 | © 2025 MolinAntro Technologies*
