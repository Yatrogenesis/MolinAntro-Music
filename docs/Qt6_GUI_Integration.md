# Qt6 GUI Integration Guide

## Overview

This document describes how to integrate Qt6 GUI with MolinAntro DAW. The audio engine and DSP components are already 100% ready - only UI layer needs to be added.

## Prerequisites

### Install Qt6

#### Ubuntu/Debian
```bash
sudo apt-get install qt6-base-dev qt6-multimedia-dev qt6-charts-dev
```

#### macOS
```bash
brew install qt@6
```

#### Windows
Download from: https://www.qt.io/download-qt-installer

## Project Structure

```
MolinAntro-Music/
├── src/
│   └── gui/                    # New Qt6 GUI module
│       ├── MainWindow.cpp
│       ├── TransportControls.cpp
│       ├── MixerPanel.cpp
│       ├── PluginChainView.cpp
│       ├── WaveformDisplay.cpp
│       └── SpectrumAnalyzer.cpp
├── include/
│   └── gui/                    # Qt6 headers
│       └── MainWindow.h
└── CMakeLists.txt              # Update with Qt6 support
```

## CMakeLists.txt Integration

Add to root `CMakeLists.txt`:

```cmake
# Qt6 Support (optional)
option(BUILD_QT6_GUI "Build Qt6 GUI" OFF)

if(BUILD_QT6_GUI)
    find_package(Qt6 REQUIRED COMPONENTS
        Core
        Widgets
        Multimedia
        Charts
    )

    add_subdirectory(src/gui)
endif()
```

Create `src/gui/CMakeLists.txt`:

```cmake
set(GUI_SOURCES
    MainWindow.cpp
    TransportControls.cpp
    MixerPanel.cpp
    PluginChainView.cpp
    WaveformDisplay.cpp
    SpectrumAnalyzer.cpp
)

set(GUI_HEADERS
    ../../../include/gui/MainWindow.h
    ../../../include/gui/TransportControls.h
    ../../../include/gui/MixerPanel.h
)

qt6_wrap_cpp(GUI_MOC ${GUI_HEADERS})

add_library(MolinAntro_GUI STATIC ${GUI_SOURCES} ${GUI_MOC})

target_link_libraries(MolinAntro_GUI
    PUBLIC
        MolinAntro_Core
        MolinAntro_DSP
        MolinAntro_MIDI
        MolinAntro_Instruments
        MolinAntro_Plugins
        Qt6::Core
        Qt6::Widgets
        Qt6::Multimedia
        Qt6::Charts
)
```

## Main Window Implementation

### Header: `include/gui/MainWindow.h`

```cpp
#pragma once

#include <QMainWindow>
#include <QTimer>
#include "core/AudioEngine.h"
#include "core/Transport.h"
#include "plugins/PluginHost.h"

namespace Ui {
    class MainWindow;
}

namespace MolinAntro {
namespace GUI {

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_playButton_clicked();
    void on_stopButton_clicked();
    void on_recordButton_clicked();
    void updateUI();

private:
    Ui::MainWindow *ui;

    Core::AudioEngine* engine_;
    Core::Transport* transport_;
    Plugins::PluginHost* pluginHost_;

    QTimer* updateTimer_;

    void setupAudio();
    void setupConnections();
};

} // namespace GUI
} // namespace MolinAntro
```

### Implementation: `src/gui/MainWindow.cpp`

```cpp
#include "gui/MainWindow.h"
#include "ui_mainwindow.h"

namespace MolinAntro {
namespace GUI {

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , engine_(new Core::AudioEngine())
    , transport_(new Core::Transport())
    , pluginHost_(new Plugins::PluginHost())
    , updateTimer_(new QTimer(this))
{
    ui->setupUi(this);
    setupAudio();
    setupConnections();

    // Update UI at 30 FPS
    updateTimer_->setInterval(33);
    connect(updateTimer_, &QTimer::timeout, this, &MainWindow::updateUI);
    updateTimer_->start();
}

MainWindow::~MainWindow() {
    delete engine_;
    delete transport_;
    delete pluginHost_;
    delete ui;
}

void MainWindow::setupAudio() {
    engine_->initialize(48000, 512);
    pluginHost_->setSampleRate(48000);
    pluginHost_->setBlockSize(512);
}

void MainWindow::setupConnections() {
    connect(ui->playButton, &QPushButton::clicked,
            this, &MainWindow::on_playButton_clicked);
    connect(ui->stopButton, &QPushButton::clicked,
            this, &MainWindow::on_stopButton_clicked);
    connect(ui->recordButton, &QPushButton::clicked,
            this, &MainWindow::on_recordButton_clicked);
}

void MainWindow::on_playButton_clicked() {
    if (transport_->getState() == Core::Transport::State::Playing) {
        transport_->pause();
    } else {
        transport_->play();
        engine_->start();
    }
}

void MainWindow::on_stopButton_clicked() {
    transport_->stop();
    engine_->stop();
}

void MainWindow::on_recordButton_clicked() {
    if (transport_->getState() == Core::Transport::State::Recording) {
        transport_->stop();
    } else {
        transport_->record();
        engine_->start();
    }
}

void MainWindow::updateUI() {
    // Update transport display
    auto timeInfo = transport_->getTimeInfo();
    ui->timeLabel->setText(QString("%1:%2:%3")
        .arg(timeInfo.bar)
        .arg(timeInfo.beat)
        .arg(timeInfo.tick));

    // Update CPU usage
    float cpuUsage = engine_->getCPUUsage();
    ui->cpuMeter->setValue(static_cast<int>(cpuUsage * 100));
}

} // namespace GUI
} // namespace MolinAntro
```

## UI Components

### 1. Transport Controls

```cpp
class TransportControls : public QWidget {
    Q_OBJECT
public:
    TransportControls(QWidget* parent = nullptr);

signals:
    void playClicked();
    void stopClicked();
    void recordClicked();
    void loopToggled(bool enabled);

private:
    QPushButton* playButton_;
    QPushButton* stopButton_;
    QPushButton* recordButton_;
    QPushButton* loopButton_;
    QLabel* positionLabel_;
};
```

### 2. Mixer Panel

```cpp
class MixerPanel : public QWidget {
    Q_OBJECT
public:
    MixerPanel(QWidget* parent = nullptr);

    void addChannel(const QString& name);
    void updateLevel(int channel, float leftPeak, float rightPeak);

signals:
    void volumeChanged(int channel, float volume);
    void panChanged(int channel, float pan);
    void muteToggled(int channel, bool muted);
    void soloToggled(int channel, bool solo);

private:
    struct ChannelStrip {
        QSlider* volumeSlider;
        QDial* panDial;
        QPushButton* muteButton;
        QPushButton* soloButton;
        QProgressBar* levelMeterL;
        QProgressBar* levelMeterR;
    };

    QVector<ChannelStrip> channels_;
};
```

### 3. Waveform Display

```cpp
class WaveformDisplay : public QWidget {
    Q_OBJECT
public:
    WaveformDisplay(QWidget* parent = nullptr);

    void setAudioBuffer(const Core::AudioBuffer& buffer);
    void setZoom(float zoom);
    void setPlayheadPosition(double seconds);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

signals:
    void seekRequested(double seconds);

private:
    std::vector<float> waveformData_;
    float zoom_;
    double playheadPosition_;
};
```

### 4. Spectrum Analyzer

```cpp
class SpectrumAnalyzer : public QWidget {
    Q_OBJECT
public:
    SpectrumAnalyzer(QWidget* parent = nullptr);

    void updateSpectrum(const std::vector<float>& magnitudes);

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<float> spectrum_;
    int numBins_;
};
```

## Qt Designer UI File

Create `mainwindow.ui`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<ui version="4.0">
 <class>MainWindow</class>
 <widget class="QMainWindow" name="MainWindow">
  <property name="geometry">
   <rect>
    <x>0</x>
    <y>0</y>
    <width>1280</width>
    <height>720</height>
   </rect>
  </property>
  <property name="windowTitle">
   <string>MolinAntro DAW v2.0</string>
  </property>
  <widget class="QWidget" name="centralwidget">
   <layout class="QVBoxLayout">
    <item>
     <widget class="TransportControls" name="transportControls"/>
    </item>
    <item>
     <widget class="WaveformDisplay" name="waveformDisplay"/>
    </item>
    <item>
     <widget class="MixerPanel" name="mixerPanel"/>
    </item>
   </layout>
  </widget>
  <widget class="QMenuBar" name="menubar"/>
  <widget class="QStatusBar" name="statusbar"/>
 </widget>
 <resources/>
 <connections/>
</ui>
```

## Building with Qt6

```bash
mkdir build-qt6
cd build-qt6
cmake .. -DBUILD_QT6_GUI=ON
cmake --build . -j 4
```

## Executable

```bash
./bin/MolinAntroDaw-qt6
```

## Integration with Existing Audio Engine

The Qt6 GUI connects seamlessly with existing audio engine:

```cpp
// In MainWindow::setupAudio()
engine_->setAudioCallback([this](float** inputs, float** outputs, int numSamples) {
    // Process through plugin chain
    Core::AudioBuffer buffer(2, numSamples);

    // Copy inputs
    for (int ch = 0; ch < 2; ++ch) {
        std::copy(inputs[ch], inputs[ch] + numSamples,
                  buffer.getWritePointer(ch));
    }

    // Process effects
    pluginHost_->processPluginChain(buffer);

    // Copy to outputs
    for (int ch = 0; ch < 2; ++ch) {
        std::copy(buffer.getReadPointer(ch),
                  buffer.getReadPointer(ch) + numSamples,
                  outputs[ch]);
    }
});
```

## Advanced Features

### Real-time Spectrum Display

```cpp
void MainWindow::setupSpectrum() {
    // Use existing SpectralProcessor
    auto processor = new DSP::SpectralProcessor();
    processor->setFFTSize(2048);
    processor->setSampleRate(48000);

    // Update in audio callback
    engine_->setAudioCallback([processor](float** in, float** out, int n) {
        Core::AudioBuffer buf(2, n);
        // ... copy input ...
        processor->analyze(buf);

        // Get spectrum for display
        auto& frames = processor->getFrames();
        if (!frames.empty()) {
            emit spectrumReady(frames.back().magnitudes);
        }
    });
}
```

## Status

- ✅ All audio components ready
- ✅ Build system prepared
- ⏸️ Awaiting Qt6 installation
- ⏸️ UI components to be implemented

## Resources

- Qt6 Documentation: https://doc.qt.io/qt-6/
- Qt Creator IDE: https://www.qt.io/product/development-tools
- Qt Examples: https://doc.qt.io/qt-6/qtexamples.html
