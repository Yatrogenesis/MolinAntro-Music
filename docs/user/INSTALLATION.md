# MolinAntro DAW - Installation Guide

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core i5 (8th gen) / AMD Ryzen 5 3600 |
| **RAM** | 8 GB DDR4 |
| **Storage** | 10 GB SSD |
| **GPU** | DirectX 11 / OpenGL 4.5 compatible |
| **OS** | Windows 10 (64-bit), macOS 11.0+, Ubuntu 20.04+ |
| **Audio** | ASIO/CoreAudio compatible interface |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Core i9 / AMD Ryzen 9 / Apple M2 Pro |
| **RAM** | 32 GB DDR4/DDR5 |
| **Storage** | 50 GB NVMe SSD |
| **GPU** | NVIDIA RTX 3060 / AMD RX 6700 XT |
| **Audio** | Professional-grade interface (64-128 samples) |

---

## Installation

### Linux (Ubuntu/Debian)

#### Method 1: Package Manager (Recommended)

```bash
# Download .deb package
wget https://molinantro.com/downloads/molinantro-daw_1.0.0_amd64.deb

# Install
sudo dpkg -i molinantro-daw_1.0.0_amd64.deb

# Fix dependencies
sudo apt-get install -f

# Verify installation
MolinAntroDaw --version
```

#### Method 2: Build from Source

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    libasound2-dev \
    libjack-jackd2-dev \
    libx11-dev \
    libxext-dev \
    libfreetype6-dev

# Clone repository
git clone https://github.com/molinantro/molinantro-daw.git
cd molinantro-daw

# Build
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Install
sudo cmake --install .

# Verify
MolinAntroDaw --version
```

### macOS

#### Method 1: DMG Installer (Recommended)

```bash
# Download DMG
curl -O https://molinantro.com/downloads/MolinAntroDaw-1.0.0.dmg

# Mount and install
open MolinAntroDaw-1.0.0.dmg
# Drag to Applications folder

# Launch
open /Applications/MolinAntroDaw.app
```

#### Method 2: Homebrew

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Add tap
brew tap molinantro/daw

# Install
brew install molinantro-daw

# Launch
MolinAntroDaw
```

### Windows

#### Method 1: Installer (Recommended)

1. Download `MolinAntroDaw-Setup-1.0.0.exe`
2. Run the installer
3. Follow the installation wizard
4. Choose installation directory (default: `C:\Program Files\MolinAntro DAW`)
5. Select components:
   - ☑ Core Application
   - ☑ ASIO Drivers
   - ☑ VST3 Support
   - ☐ Developer Tools (optional)
6. Click "Install"
7. Launch from Start Menu or Desktop shortcut

#### Method 2: Portable Version

```powershell
# Download portable ZIP
Invoke-WebRequest -Uri https://molinantro.com/downloads/MolinAntroDaw-1.0.0-Portable.zip -OutFile MolinAntroDaw.zip

# Extract
Expand-Archive -Path MolinAntroDaw.zip -DestinationPath C:\MolinAntro

# Run
C:\MolinAntro\MolinAntroDaw.exe
```

---

## Post-Installation

### Audio Configuration

#### Linux (JACK)

```bash
# Install JACK
sudo apt-get install jackd2

# Start JACK server
jackd -d alsa -r 48000 -p 512

# Configure MolinAntro to use JACK
MolinAntroDaw --audio-driver jack
```

#### macOS (CoreAudio)

CoreAudio is configured automatically. No additional setup required.

#### Windows (ASIO)

1. Install your audio interface's ASIO driver
2. Launch MolinAntro DAW
3. Go to Settings → Audio
4. Select your ASIO driver from the dropdown

### Plugin Directories

MolinAntro DAW scans the following directories for plugins:

#### Linux

- VST3: `~/.vst3`
- LV2: `~/.lv2`

#### macOS

- VST3: `~/Library/Audio/Plug-Ins/VST3`
- AU: `~/Library/Audio/Plug-Ins/Components`

#### Windows

- VST3: `C:\Program Files\Common Files\VST3`
- VST2: `C:\Program Files\VstPlugins`

### License Activation

```bash
# Online activation
MolinAntroDaw --activate <license-key>

# Offline activation (generate challenge file)
MolinAntroDaw --challenge > challenge.txt

# Import response file
MolinAntroDaw --response response.txt
```

---

## Troubleshooting

### Linux: Audio Device Not Found

```bash
# List available devices
aplay -l

# Check permissions
sudo usermod -a -G audio $USER

# Restart audio services
systemctl --user restart pulseaudio
```

### macOS: Application Can't Be Opened

```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine /Applications/MolinAntroDaw.app
```

### Windows: ASIO Driver Not Detected

1. Reinstall audio interface drivers
2. Run as Administrator
3. Check Windows Audio settings

### Build Errors

```bash
# Clear CMake cache
rm -rf build
mkdir build && cd build

# Reconfigure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Rebuild
cmake --build . --clean-first
```

---

## Uninstallation

### Linux

```bash
# If installed via package
sudo apt-get remove molinantro-daw

# If built from source
sudo rm -rf /usr/local/bin/MolinAntroDaw
sudo rm -rf /usr/local/lib/libMolinAntro*
```

### macOS

```bash
# Remove application
rm -rf /Applications/MolinAntroDaw.app

# Remove user data (optional)
rm -rf ~/Library/Application\ Support/MolinAntro
```

### Windows

1. Open "Add or Remove Programs"
2. Find "MolinAntro DAW"
3. Click "Uninstall"
4. Follow the wizard

---

## Updates

### Automatic Updates

MolinAntro DAW checks for updates automatically. When an update is available:

1. Click "Update Available" notification
2. Review release notes
3. Click "Install Update"
4. Application will restart

### Manual Updates

```bash
# Check current version
MolinAntroDaw --version

# Download latest version
# Install using same method as original installation
```

---

**Need Help?**

- Support: support@molinantro.com
- Forum: https://forum.molinantro.com
- Documentation: https://docs.molinantro.com

---

*MolinAntro DAW v1.0.0 | © 2025 MolinAntro Technologies*
