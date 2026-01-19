#pragma once

/**
 * @file Qt6MainWindow.h
 * @brief Professional Qt6 GUI for MolinAntro DAW
 *
 * Design Principles:
 * - Minimum button size: 44x32px (touch-friendly)
 * - Label padding: 8px minimum
 * - Font size: 13px minimum for readability
 * - Proper scroll bars on all lists/views
 * - Responsive docking system
 * - HiDPI support
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#ifdef BUILD_QT6_GUI

#include <QMainWindow>
#include <QDockWidget>
#include <QToolBar>
#include <QStatusBar>
#include <QMenuBar>
#include <QSplitter>
#include <QStackedWidget>
#include <QTimer>
#include <QSettings>
#include <memory>

// Forward declarations
namespace MolinAntro {
namespace Core {
    class AudioEngine;
    class Transport;
}
namespace DSP {
    class SpectralProcessor;
}
namespace Plugins {
    class PluginHost;
}
}

namespace MolinAntro {
namespace GUI {

// Forward declarations
class TransportPanel;
class MixerPanel;
class ArrangementPanel;
class BrowserPanel;
class PianoRollPanel;
class SessionPanel;
class PluginEditorPanel;
class WaveformWidget;
class SpectrumWidget;
class MeterWidget;

/**
 * @brief UI Style Constants - Ensuring readable, accessible interface
 */
namespace Style {
    // Minimum sizes for accessibility
    constexpr int MIN_BUTTON_WIDTH = 44;
    constexpr int MIN_BUTTON_HEIGHT = 32;
    constexpr int MIN_ICON_SIZE = 24;
    constexpr int MIN_FONT_SIZE = 13;
    constexpr int LABEL_PADDING = 8;
    constexpr int WIDGET_SPACING = 4;
    constexpr int PANEL_MARGIN = 8;

    // Comfortable sizes
    constexpr int FADER_WIDTH = 36;
    constexpr int FADER_HEIGHT = 120;
    constexpr int KNOB_SIZE = 48;
    constexpr int METER_WIDTH = 12;
    constexpr int TOOLBAR_HEIGHT = 48;
    constexpr int STATUSBAR_HEIGHT = 28;

    // Scrollbar
    constexpr int SCROLLBAR_WIDTH = 14;

    // Track lane heights
    constexpr int TRACK_MIN_HEIGHT = 60;
    constexpr int TRACK_DEFAULT_HEIGHT = 80;
    constexpr int TRACK_MAX_HEIGHT = 200;

    // Channel strip
    constexpr int CHANNEL_WIDTH = 80;
    constexpr int CHANNEL_MIN_WIDTH = 60;
}

/**
 * @brief Main DAW Window with professional docking system
 */
class Qt6MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit Qt6MainWindow(QWidget* parent = nullptr);
    ~Qt6MainWindow() override;

    // Initialization
    bool initialize();

    // Panel access
    TransportPanel* getTransport() const { return transportPanel_; }
    MixerPanel* getMixer() const { return mixerPanel_; }
    ArrangementPanel* getArrangement() const { return arrangementPanel_; }
    BrowserPanel* getBrowser() const { return browserPanel_; }
    PianoRollPanel* getPianoRoll() const { return pianoRollPanel_; }
    SessionPanel* getSession() const { return sessionPanel_; }

    // View modes
    enum class ViewMode {
        Arrangement,    // Timeline view
        Session,        // Clip launcher (Ableton-style)
        Mixer,          // Full mixer view
        Edit            // Detailed editing
    };

    void setViewMode(ViewMode mode);
    ViewMode getViewMode() const { return currentViewMode_; }

public slots:
    // File operations
    void newProject();
    void openProject();
    void saveProject();
    void saveProjectAs();
    void exportAudio();

    // Edit operations
    void undo();
    void redo();
    void cut();
    void copy();
    void paste();
    void deleteSelection();
    void selectAll();

    // View operations
    void toggleBrowser();
    void toggleMixer();
    void togglePianoRoll();
    void toggleSession();
    void toggleArrangement();
    void toggleFullscreen();
    void zoomIn();
    void zoomOut();
    void zoomToFit();

    // Transport
    void play();
    void stop();
    void record();
    void toggleLoop();

    // Panels
    void showPreferences();
    void showPluginManager();
    void showAudioSettings();
    void showMIDISettings();
    void showAbout();

signals:
    void projectChanged();
    void viewModeChanged(ViewMode mode);
    void playbackStarted();
    void playbackStopped();

protected:
    void closeEvent(QCloseEvent* event) override;
    void resizeEvent(QResizeEvent* event) override;
    void keyPressEvent(QKeyEvent* event) override;

private slots:
    void updateUI();
    void updateMeters();
    void onTransportStateChanged();
    void onProjectModified();

private:
    // Setup methods
    void setupMenuBar();
    void setupToolBar();
    void setupStatusBar();
    void setupDockWidgets();
    void setupCentralWidget();
    void setupConnections();
    void setupShortcuts();
    void applyTheme();

    // Layout persistence
    void saveLayout();
    void restoreLayout();

    // Audio engine integration
    void setupAudioEngine();
    void startAudio();
    void stopAudio();

    // Panels (owned by dock widgets)
    TransportPanel* transportPanel_ = nullptr;
    MixerPanel* mixerPanel_ = nullptr;
    ArrangementPanel* arrangementPanel_ = nullptr;
    BrowserPanel* browserPanel_ = nullptr;
    PianoRollPanel* pianoRollPanel_ = nullptr;
    SessionPanel* sessionPanel_ = nullptr;

    // Dock widgets
    QDockWidget* browserDock_ = nullptr;
    QDockWidget* mixerDock_ = nullptr;
    QDockWidget* pianoRollDock_ = nullptr;

    // Central area
    QSplitter* mainSplitter_ = nullptr;
    QStackedWidget* viewStack_ = nullptr;

    // Toolbars
    QToolBar* mainToolBar_ = nullptr;
    QToolBar* transportToolBar_ = nullptr;

    // Engine components
    std::unique_ptr<Core::AudioEngine> audioEngine_;
    std::unique_ptr<Core::Transport> transport_;
    std::unique_ptr<Plugins::PluginHost> pluginHost_;
    std::unique_ptr<DSP::SpectralProcessor> spectralProcessor_;

    // State
    ViewMode currentViewMode_ = ViewMode::Arrangement;
    QString currentProjectPath_;
    bool projectModified_ = false;

    // Timers
    QTimer* uiUpdateTimer_ = nullptr;
    QTimer* meterUpdateTimer_ = nullptr;

    // Settings
    QSettings settings_;
};

} // namespace GUI
} // namespace MolinAntro

#endif // BUILD_QT6_GUI
