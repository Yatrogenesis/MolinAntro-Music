/**
 * @file Qt6MainWindow.cpp
 * @brief Professional Qt6 GUI implementation for MolinAntro DAW
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#ifdef BUILD_QT6_GUI

#include "gui/Qt6MainWindow.h"
#include "gui/Qt6Components.h"
#include "gui/Qt6Styles.h"

#include <QApplication>
#include <QScreen>
#include <QFileDialog>
#include <QMessageBox>
#include <QCloseEvent>
#include <QKeyEvent>
#include <QAction>
#include <QMenu>
#include <QShortcut>
#include <QLabel>

namespace MolinAntro {
namespace GUI {

Qt6MainWindow::Qt6MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , settings_("MolinAntro", "DAW")
{
    setWindowTitle("MolinAntro DAW");
    setMinimumSize(1280, 720);

    // Get screen DPI for proper scaling
    qreal dpr = screen()->devicePixelRatio();
    if (dpr > 1.0) {
        // HiDPI scaling
        setMinimumSize(static_cast<int>(1280 * dpr), static_cast<int>(720 * dpr));
    }
}

Qt6MainWindow::~Qt6MainWindow() {
    saveLayout();
    stopAudio();
}

bool Qt6MainWindow::initialize() {
    // Apply theme
    applyTheme();

    // Setup UI components in order
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
    setupCentralWidget();
    setupDockWidgets();
    setupConnections();
    setupShortcuts();

    // Setup audio engine
    setupAudioEngine();

    // Restore layout from settings
    restoreLayout();

    // Start UI update timers
    uiUpdateTimer_ = new QTimer(this);
    connect(uiUpdateTimer_, &QTimer::timeout, this, &Qt6MainWindow::updateUI);
    uiUpdateTimer_->start(33);  // ~30 FPS

    meterUpdateTimer_ = new QTimer(this);
    connect(meterUpdateTimer_, &QTimer::timeout, this, &Qt6MainWindow::updateMeters);
    meterUpdateTimer_->start(16);  // ~60 FPS for smooth meters

    return true;
}

void Qt6MainWindow::applyTheme() {
    auto& theme = ThemeManager::instance();
    theme.setTheme(ThemeManager::Theme::Dark);

    // Apply palette
    QApplication::setPalette(theme.getPalette());

    // Apply stylesheet
    setStyleSheet(theme.getStyleSheet());
}

void Qt6MainWindow::setupMenuBar() {
    QMenuBar* menuBar = new QMenuBar(this);
    menuBar->setMinimumHeight(Style::TOOLBAR_HEIGHT);

    // File menu
    QMenu* fileMenu = menuBar->addMenu(tr("&File"));

    QAction* newAction = fileMenu->addAction(tr("&New Project"));
    newAction->setShortcut(QKeySequence::New);
    connect(newAction, &QAction::triggered, this, &Qt6MainWindow::newProject);

    QAction* openAction = fileMenu->addAction(tr("&Open Project..."));
    openAction->setShortcut(QKeySequence::Open);
    connect(openAction, &QAction::triggered, this, &Qt6MainWindow::openProject);

    fileMenu->addSeparator();

    QAction* saveAction = fileMenu->addAction(tr("&Save"));
    saveAction->setShortcut(QKeySequence::Save);
    connect(saveAction, &QAction::triggered, this, &Qt6MainWindow::saveProject);

    QAction* saveAsAction = fileMenu->addAction(tr("Save &As..."));
    saveAsAction->setShortcut(QKeySequence::SaveAs);
    connect(saveAsAction, &QAction::triggered, this, &Qt6MainWindow::saveProjectAs);

    fileMenu->addSeparator();

    QAction* exportAction = fileMenu->addAction(tr("&Export Audio..."));
    exportAction->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_E));
    connect(exportAction, &QAction::triggered, this, &Qt6MainWindow::exportAudio);

    fileMenu->addSeparator();

    QAction* quitAction = fileMenu->addAction(tr("&Quit"));
    quitAction->setShortcut(QKeySequence::Quit);
    connect(quitAction, &QAction::triggered, this, &QMainWindow::close);

    // Edit menu
    QMenu* editMenu = menuBar->addMenu(tr("&Edit"));

    QAction* undoAction = editMenu->addAction(tr("&Undo"));
    undoAction->setShortcut(QKeySequence::Undo);
    connect(undoAction, &QAction::triggered, this, &Qt6MainWindow::undo);

    QAction* redoAction = editMenu->addAction(tr("&Redo"));
    redoAction->setShortcut(QKeySequence::Redo);
    connect(redoAction, &QAction::triggered, this, &Qt6MainWindow::redo);

    editMenu->addSeparator();

    QAction* cutAction = editMenu->addAction(tr("Cu&t"));
    cutAction->setShortcut(QKeySequence::Cut);
    connect(cutAction, &QAction::triggered, this, &Qt6MainWindow::cut);

    QAction* copyAction = editMenu->addAction(tr("&Copy"));
    copyAction->setShortcut(QKeySequence::Copy);
    connect(copyAction, &QAction::triggered, this, &Qt6MainWindow::copy);

    QAction* pasteAction = editMenu->addAction(tr("&Paste"));
    pasteAction->setShortcut(QKeySequence::Paste);
    connect(pasteAction, &QAction::triggered, this, &Qt6MainWindow::paste);

    QAction* deleteAction = editMenu->addAction(tr("&Delete"));
    deleteAction->setShortcut(QKeySequence::Delete);
    connect(deleteAction, &QAction::triggered, this, &Qt6MainWindow::deleteSelection);

    editMenu->addSeparator();

    QAction* selectAllAction = editMenu->addAction(tr("Select &All"));
    selectAllAction->setShortcut(QKeySequence::SelectAll);
    connect(selectAllAction, &QAction::triggered, this, &Qt6MainWindow::selectAll);

    // View menu
    QMenu* viewMenu = menuBar->addMenu(tr("&View"));

    QAction* browserAction = viewMenu->addAction(tr("&Browser"));
    browserAction->setShortcut(QKeySequence(Qt::Key_B));
    browserAction->setCheckable(true);
    browserAction->setChecked(true);
    connect(browserAction, &QAction::triggered, this, &Qt6MainWindow::toggleBrowser);

    QAction* mixerAction = viewMenu->addAction(tr("&Mixer"));
    mixerAction->setShortcut(QKeySequence(Qt::Key_M));
    mixerAction->setCheckable(true);
    mixerAction->setChecked(true);
    connect(mixerAction, &QAction::triggered, this, &Qt6MainWindow::toggleMixer);

    QAction* pianoRollAction = viewMenu->addAction(tr("&Piano Roll"));
    pianoRollAction->setShortcut(QKeySequence(Qt::Key_P));
    pianoRollAction->setCheckable(true);
    connect(pianoRollAction, &QAction::triggered, this, &Qt6MainWindow::togglePianoRoll);

    viewMenu->addSeparator();

    QAction* arrangementAction = viewMenu->addAction(tr("&Arrangement View"));
    arrangementAction->setShortcut(QKeySequence(Qt::Key_1));
    connect(arrangementAction, &QAction::triggered, this, &Qt6MainWindow::toggleArrangement);

    QAction* sessionAction = viewMenu->addAction(tr("&Session View"));
    sessionAction->setShortcut(QKeySequence(Qt::Key_2));
    connect(sessionAction, &QAction::triggered, this, &Qt6MainWindow::toggleSession);

    viewMenu->addSeparator();

    QAction* zoomInAction = viewMenu->addAction(tr("Zoom &In"));
    zoomInAction->setShortcut(QKeySequence::ZoomIn);
    connect(zoomInAction, &QAction::triggered, this, &Qt6MainWindow::zoomIn);

    QAction* zoomOutAction = viewMenu->addAction(tr("Zoom &Out"));
    zoomOutAction->setShortcut(QKeySequence::ZoomOut);
    connect(zoomOutAction, &QAction::triggered, this, &Qt6MainWindow::zoomOut);

    QAction* zoomFitAction = viewMenu->addAction(tr("Zoom to &Fit"));
    zoomFitAction->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_0));
    connect(zoomFitAction, &QAction::triggered, this, &Qt6MainWindow::zoomToFit);

    viewMenu->addSeparator();

    QAction* fullscreenAction = viewMenu->addAction(tr("&Fullscreen"));
    fullscreenAction->setShortcut(QKeySequence::FullScreen);
    fullscreenAction->setCheckable(true);
    connect(fullscreenAction, &QAction::triggered, this, &Qt6MainWindow::toggleFullscreen);

    // Transport menu
    QMenu* transportMenu = menuBar->addMenu(tr("&Transport"));

    QAction* playAction = transportMenu->addAction(tr("&Play/Pause"));
    playAction->setShortcut(QKeySequence(Qt::Key_Space));
    connect(playAction, &QAction::triggered, this, &Qt6MainWindow::play);

    QAction* stopAction = transportMenu->addAction(tr("&Stop"));
    stopAction->setShortcut(QKeySequence(Qt::Key_Return));
    connect(stopAction, &QAction::triggered, this, &Qt6MainWindow::stop);

    QAction* recordAction = transportMenu->addAction(tr("&Record"));
    recordAction->setShortcut(QKeySequence(Qt::Key_R));
    connect(recordAction, &QAction::triggered, this, &Qt6MainWindow::record);

    transportMenu->addSeparator();

    QAction* loopAction = transportMenu->addAction(tr("&Loop"));
    loopAction->setShortcut(QKeySequence(Qt::Key_L));
    loopAction->setCheckable(true);
    connect(loopAction, &QAction::triggered, this, &Qt6MainWindow::toggleLoop);

    // Options menu
    QMenu* optionsMenu = menuBar->addMenu(tr("&Options"));

    QAction* prefsAction = optionsMenu->addAction(tr("&Preferences..."));
    prefsAction->setShortcut(QKeySequence::Preferences);
    connect(prefsAction, &QAction::triggered, this, &Qt6MainWindow::showPreferences);

    QAction* audioAction = optionsMenu->addAction(tr("&Audio Settings..."));
    connect(audioAction, &QAction::triggered, this, &Qt6MainWindow::showAudioSettings);

    QAction* midiAction = optionsMenu->addAction(tr("&MIDI Settings..."));
    connect(midiAction, &QAction::triggered, this, &Qt6MainWindow::showMIDISettings);

    optionsMenu->addSeparator();

    QAction* pluginsAction = optionsMenu->addAction(tr("&Plugin Manager..."));
    connect(pluginsAction, &QAction::triggered, this, &Qt6MainWindow::showPluginManager);

    // Help menu
    QMenu* helpMenu = menuBar->addMenu(tr("&Help"));

    QAction* aboutAction = helpMenu->addAction(tr("&About MolinAntro DAW..."));
    connect(aboutAction, &QAction::triggered, this, &Qt6MainWindow::showAbout);

    setMenuBar(menuBar);
}

void Qt6MainWindow::setupToolBar() {
    // Main toolbar
    mainToolBar_ = new QToolBar(tr("Main"), this);
    mainToolBar_->setMovable(false);
    mainToolBar_->setFloatable(false);
    mainToolBar_->setIconSize(QSize(Style::MIN_ICON_SIZE, Style::MIN_ICON_SIZE));
    mainToolBar_->setMinimumHeight(Style::TOOLBAR_HEIGHT);

    // File operations
    QAction* newBtn = mainToolBar_->addAction(tr("New"));
    newBtn->setToolTip(tr("New Project (Ctrl+N)"));
    connect(newBtn, &QAction::triggered, this, &Qt6MainWindow::newProject);

    QAction* openBtn = mainToolBar_->addAction(tr("Open"));
    openBtn->setToolTip(tr("Open Project (Ctrl+O)"));
    connect(openBtn, &QAction::triggered, this, &Qt6MainWindow::openProject);

    QAction* saveBtn = mainToolBar_->addAction(tr("Save"));
    saveBtn->setToolTip(tr("Save Project (Ctrl+S)"));
    connect(saveBtn, &QAction::triggered, this, &Qt6MainWindow::saveProject);

    mainToolBar_->addSeparator();

    // Edit operations
    QAction* undoBtn = mainToolBar_->addAction(tr("Undo"));
    undoBtn->setToolTip(tr("Undo (Ctrl+Z)"));
    connect(undoBtn, &QAction::triggered, this, &Qt6MainWindow::undo);

    QAction* redoBtn = mainToolBar_->addAction(tr("Redo"));
    redoBtn->setToolTip(tr("Redo (Ctrl+Y)"));
    connect(redoBtn, &QAction::triggered, this, &Qt6MainWindow::redo);

    addToolBar(Qt::TopToolBarArea, mainToolBar_);

    // Transport toolbar (separate, centered)
    transportToolBar_ = new QToolBar(tr("Transport"), this);
    transportToolBar_->setMovable(false);
    transportToolBar_->setFloatable(false);
    transportToolBar_->setMinimumHeight(Style::TOOLBAR_HEIGHT);

    // Add spacer to center transport controls
    QWidget* spacer1 = new QWidget();
    spacer1->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    transportToolBar_->addWidget(spacer1);

    // Transport panel widget
    transportPanel_ = new TransportPanel(this);
    transportToolBar_->addWidget(transportPanel_);

    QWidget* spacer2 = new QWidget();
    spacer2->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Preferred);
    transportToolBar_->addWidget(spacer2);

    addToolBar(Qt::TopToolBarArea, transportToolBar_);
}

void Qt6MainWindow::setupStatusBar() {
    QStatusBar* status = new QStatusBar(this);
    status->setMinimumHeight(Style::STATUSBAR_HEIGHT);

    // CPU usage
    QLabel* cpuLabel = new QLabel(tr("CPU: 0%"), this);
    cpuLabel->setMinimumWidth(80);
    status->addWidget(cpuLabel);

    // Sample rate
    QLabel* srLabel = new QLabel(tr("44100 Hz"), this);
    srLabel->setMinimumWidth(80);
    status->addWidget(srLabel);

    // Buffer size
    QLabel* bufferLabel = new QLabel(tr("512 smp"), this);
    bufferLabel->setMinimumWidth(80);
    status->addWidget(bufferLabel);

    // Latency
    QLabel* latencyLabel = new QLabel(tr("11.6 ms"), this);
    latencyLabel->setMinimumWidth(80);
    status->addWidget(latencyLabel);

    // Project status (right side)
    QLabel* projectLabel = new QLabel(tr("No project"), this);
    status->addPermanentWidget(projectLabel);

    setStatusBar(status);
}

void Qt6MainWindow::setupCentralWidget() {
    // Main splitter for central area
    mainSplitter_ = new QSplitter(Qt::Horizontal, this);

    // Stacked widget for different views
    viewStack_ = new QStackedWidget(this);

    // Arrangement view
    arrangementPanel_ = new ArrangementPanel(this);
    viewStack_->addWidget(arrangementPanel_);

    // Session view
    sessionPanel_ = new SessionPanel(this);
    viewStack_->addWidget(sessionPanel_);

    // Start with arrangement view
    viewStack_->setCurrentIndex(0);

    mainSplitter_->addWidget(viewStack_);

    setCentralWidget(mainSplitter_);
}

void Qt6MainWindow::setupDockWidgets() {
    // Browser dock (left side)
    browserDock_ = new QDockWidget(tr("Browser"), this);
    browserDock_->setAllowedAreas(Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea);
    browserDock_->setMinimumWidth(200);
    browserDock_->setFeatures(QDockWidget::DockWidgetClosable |
                              QDockWidget::DockWidgetMovable |
                              QDockWidget::DockWidgetFloatable);

    browserPanel_ = new BrowserPanel(this);
    browserDock_->setWidget(browserPanel_);
    addDockWidget(Qt::LeftDockWidgetArea, browserDock_);

    // Mixer dock (bottom)
    mixerDock_ = new QDockWidget(tr("Mixer"), this);
    mixerDock_->setAllowedAreas(Qt::BottomDockWidgetArea);
    mixerDock_->setMinimumHeight(200);
    mixerDock_->setFeatures(QDockWidget::DockWidgetClosable |
                            QDockWidget::DockWidgetMovable |
                            QDockWidget::DockWidgetFloatable);

    mixerPanel_ = new MixerPanel(this);
    mixerDock_->setWidget(mixerPanel_);
    addDockWidget(Qt::BottomDockWidgetArea, mixerDock_);

    // Piano Roll dock (bottom, tabbed with mixer)
    pianoRollDock_ = new QDockWidget(tr("Piano Roll"), this);
    pianoRollDock_->setAllowedAreas(Qt::BottomDockWidgetArea);
    pianoRollDock_->setMinimumHeight(200);
    pianoRollDock_->setFeatures(QDockWidget::DockWidgetClosable |
                                 QDockWidget::DockWidgetMovable |
                                 QDockWidget::DockWidgetFloatable);

    pianoRollPanel_ = new PianoRollPanel(this);
    pianoRollDock_->setWidget(pianoRollPanel_);
    addDockWidget(Qt::BottomDockWidgetArea, pianoRollDock_);

    // Tab the bottom docks
    tabifyDockWidget(mixerDock_, pianoRollDock_);
    mixerDock_->raise();  // Show mixer by default

    // Hide piano roll initially
    pianoRollDock_->hide();
}

void Qt6MainWindow::setupConnections() {
    // Transport panel connections
    connect(transportPanel_, &TransportPanel::playClicked, this, &Qt6MainWindow::play);
    connect(transportPanel_, &TransportPanel::stopClicked, this, &Qt6MainWindow::stop);
    connect(transportPanel_, &TransportPanel::recordClicked, this, &Qt6MainWindow::record);
    connect(transportPanel_, &TransportPanel::loopToggled, this, [this](bool enabled) {
        // Handle loop toggle
    });

    // Mixer connections
    connect(mixerPanel_, &MixerPanel::channelVolumeChanged, this, [this](int index, float dB) {
        // Handle volume change
    });
    connect(mixerPanel_, &MixerPanel::channelSelected, this, [this](int index) {
        // Handle channel selection
    });

    // Browser connections
    connect(browserPanel_, &BrowserPanel::itemDoubleClicked, this, [this](const QString& path) {
        // Handle file load
    });

    // Arrangement connections
    connect(arrangementPanel_, &ArrangementPanel::positionChanged, this, [this](double beats) {
        transportPanel_->setPosition(beats);
    });
}

void Qt6MainWindow::setupShortcuts() {
    // Additional keyboard shortcuts not covered by menus
    auto* tabKey = new QShortcut(QKeySequence(Qt::Key_Tab), this);
    connect(tabKey, &QShortcut::activated, this, [this]() {
        // Cycle through views
        if (currentViewMode_ == ViewMode::Arrangement) {
            setViewMode(ViewMode::Session);
        } else {
            setViewMode(ViewMode::Arrangement);
        }
    });
}

void Qt6MainWindow::setupAudioEngine() {
    // Initialize audio engine (placeholder)
    // audioEngine_ = std::make_unique<Core::AudioEngine>();
    // transport_ = std::make_unique<Core::Transport>();
}

void Qt6MainWindow::startAudio() {
    // Start audio engine
}

void Qt6MainWindow::stopAudio() {
    // Stop audio engine
}

void Qt6MainWindow::saveLayout() {
    settings_.setValue("geometry", saveGeometry());
    settings_.setValue("windowState", saveState());
    settings_.setValue("viewMode", static_cast<int>(currentViewMode_));
}

void Qt6MainWindow::restoreLayout() {
    if (settings_.contains("geometry")) {
        restoreGeometry(settings_.value("geometry").toByteArray());
    }
    if (settings_.contains("windowState")) {
        restoreState(settings_.value("windowState").toByteArray());
    }
    if (settings_.contains("viewMode")) {
        setViewMode(static_cast<ViewMode>(settings_.value("viewMode").toInt()));
    }
}

void Qt6MainWindow::setViewMode(ViewMode mode) {
    currentViewMode_ = mode;

    switch (mode) {
        case ViewMode::Arrangement:
            viewStack_->setCurrentWidget(arrangementPanel_);
            break;
        case ViewMode::Session:
            viewStack_->setCurrentWidget(sessionPanel_);
            break;
        case ViewMode::Mixer:
            mixerDock_->show();
            mixerDock_->raise();
            break;
        case ViewMode::Edit:
            pianoRollDock_->show();
            pianoRollDock_->raise();
            break;
    }

    emit viewModeChanged(mode);
}

// ============================================================================
// Slots
// ============================================================================

void Qt6MainWindow::updateUI() {
    // Update UI elements periodically
}

void Qt6MainWindow::updateMeters() {
    // Update meter displays at high frequency
    if (mixerPanel_) {
        // Get levels from audio engine and update
        // std::vector<std::pair<float, float>> levels;
        // mixerPanel_->updateMeters(levels);
    }
}

void Qt6MainWindow::onTransportStateChanged() {
    // Update transport display
}

void Qt6MainWindow::onProjectModified() {
    projectModified_ = true;
    QString title = windowTitle();
    if (!title.endsWith("*")) {
        setWindowTitle(title + "*");
    }
}

// ============================================================================
// File operations
// ============================================================================

void Qt6MainWindow::newProject() {
    if (projectModified_) {
        auto result = QMessageBox::question(this, tr("Save Project"),
            tr("Do you want to save the current project?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);

        if (result == QMessageBox::Save) {
            saveProject();
        } else if (result == QMessageBox::Cancel) {
            return;
        }
    }

    // Create new project
    currentProjectPath_.clear();
    projectModified_ = false;
    setWindowTitle(tr("MolinAntro DAW - Untitled"));
    emit projectChanged();
}

void Qt6MainWindow::openProject() {
    QString path = QFileDialog::getOpenFileName(this,
        tr("Open Project"),
        QString(),
        tr("MolinAntro Projects (*.mdaw);;All Files (*)"));

    if (path.isEmpty()) {
        return;
    }

    // Load project
    currentProjectPath_ = path;
    projectModified_ = false;
    setWindowTitle(tr("MolinAntro DAW - %1").arg(QFileInfo(path).fileName()));
    emit projectChanged();
}

void Qt6MainWindow::saveProject() {
    if (currentProjectPath_.isEmpty()) {
        saveProjectAs();
        return;
    }

    // Save project
    projectModified_ = false;
    QString title = windowTitle();
    if (title.endsWith("*")) {
        setWindowTitle(title.left(title.length() - 1));
    }
}

void Qt6MainWindow::saveProjectAs() {
    QString path = QFileDialog::getSaveFileName(this,
        tr("Save Project As"),
        QString(),
        tr("MolinAntro Projects (*.mdaw);;All Files (*)"));

    if (path.isEmpty()) {
        return;
    }

    currentProjectPath_ = path;
    saveProject();
    setWindowTitle(tr("MolinAntro DAW - %1").arg(QFileInfo(path).fileName()));
}

void Qt6MainWindow::exportAudio() {
    QString path = QFileDialog::getSaveFileName(this,
        tr("Export Audio"),
        QString(),
        tr("WAV Files (*.wav);;AIFF Files (*.aiff);;FLAC Files (*.flac)"));

    if (path.isEmpty()) {
        return;
    }

    // Export audio
}

// ============================================================================
// Edit operations
// ============================================================================

void Qt6MainWindow::undo() {
    // Undo
}

void Qt6MainWindow::redo() {
    // Redo
}

void Qt6MainWindow::cut() {
    // Cut
}

void Qt6MainWindow::copy() {
    // Copy
}

void Qt6MainWindow::paste() {
    // Paste
}

void Qt6MainWindow::deleteSelection() {
    // Delete
}

void Qt6MainWindow::selectAll() {
    // Select all
}

// ============================================================================
// View operations
// ============================================================================

void Qt6MainWindow::toggleBrowser() {
    browserDock_->setVisible(!browserDock_->isVisible());
}

void Qt6MainWindow::toggleMixer() {
    mixerDock_->setVisible(!mixerDock_->isVisible());
    if (mixerDock_->isVisible()) {
        mixerDock_->raise();
    }
}

void Qt6MainWindow::togglePianoRoll() {
    pianoRollDock_->setVisible(!pianoRollDock_->isVisible());
    if (pianoRollDock_->isVisible()) {
        pianoRollDock_->raise();
    }
}

void Qt6MainWindow::toggleSession() {
    setViewMode(ViewMode::Session);
}

void Qt6MainWindow::toggleArrangement() {
    setViewMode(ViewMode::Arrangement);
}

void Qt6MainWindow::toggleFullscreen() {
    if (isFullScreen()) {
        showNormal();
    } else {
        showFullScreen();
    }
}

void Qt6MainWindow::zoomIn() {
    if (arrangementPanel_) {
        double zoom = arrangementPanel_->getZoom() * 1.2;
        arrangementPanel_->setZoom(zoom);
    }
}

void Qt6MainWindow::zoomOut() {
    if (arrangementPanel_) {
        double zoom = arrangementPanel_->getZoom() / 1.2;
        arrangementPanel_->setZoom(zoom);
    }
}

void Qt6MainWindow::zoomToFit() {
    // Zoom to fit all content
}

// ============================================================================
// Transport
// ============================================================================

void Qt6MainWindow::play() {
    if (transportPanel_->getState() == TransportPanel::State::Playing) {
        transportPanel_->setState(TransportPanel::State::Paused);
        emit playbackStopped();
    } else {
        transportPanel_->setState(TransportPanel::State::Playing);
        emit playbackStarted();
    }
}

void Qt6MainWindow::stop() {
    transportPanel_->setState(TransportPanel::State::Stopped);
    transportPanel_->setPosition(0.0);
    emit playbackStopped();
}

void Qt6MainWindow::record() {
    if (transportPanel_->getState() == TransportPanel::State::Recording) {
        transportPanel_->setState(TransportPanel::State::Stopped);
    } else {
        transportPanel_->setState(TransportPanel::State::Recording);
    }
}

void Qt6MainWindow::toggleLoop() {
    // Toggle loop
}

// ============================================================================
// Dialogs
// ============================================================================

void Qt6MainWindow::showPreferences() {
    // Show preferences dialog
}

void Qt6MainWindow::showPluginManager() {
    // Show plugin manager
}

void Qt6MainWindow::showAudioSettings() {
    // Show audio settings
}

void Qt6MainWindow::showMIDISettings() {
    // Show MIDI settings
}

void Qt6MainWindow::showAbout() {
    QMessageBox::about(this, tr("About MolinAntro DAW"),
        tr("<h2>MolinAntro DAW</h2>"
           "<p>Version 3.0.0-ACME Professional Edition</p>"
           "<p>Professional audio workstation for commercial, forensic, "
           "and military applications.</p>"
           "<p>&copy; 2026 MolinAntro Technologies</p>"
           "<p>Author: Francisco Molina-Burgos</p>"
           "<p>Avermex Research Division<br>"
           "Mérida, Yucatán, México</p>"));
}

// ============================================================================
// Event handlers
// ============================================================================

void Qt6MainWindow::closeEvent(QCloseEvent* event) {
    if (projectModified_) {
        auto result = QMessageBox::question(this, tr("Save Project"),
            tr("Do you want to save the current project before closing?"),
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);

        if (result == QMessageBox::Save) {
            saveProject();
        } else if (result == QMessageBox::Cancel) {
            event->ignore();
            return;
        }
    }

    saveLayout();
    event->accept();
}

void Qt6MainWindow::resizeEvent(QResizeEvent* event) {
    QMainWindow::resizeEvent(event);
}

void Qt6MainWindow::keyPressEvent(QKeyEvent* event) {
    // Handle global key events
    QMainWindow::keyPressEvent(event);
}

} // namespace GUI
} // namespace MolinAntro

#endif // BUILD_QT6_GUI
