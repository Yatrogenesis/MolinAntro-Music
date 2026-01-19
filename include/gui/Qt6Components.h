#pragma once

/**
 * @file Qt6Components.h
 * @brief Professional Qt6 UI Components for MolinAntro DAW
 *
 * Component Design Guidelines:
 * - All buttons: minimum 44x32px
 * - All labels: 8px padding, 13px+ font
 * - All lists: visible scrollbars (14px width)
 * - All inputs: clear focus indicators
 * - HiDPI aware: use devicePixelRatio
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#ifdef BUILD_QT6_GUI

#include <QWidget>
#include <QPushButton>
#include <QSlider>
#include <QDial>
#include <QLabel>
#include <QScrollArea>
#include <QListWidget>
#include <QTreeWidget>
#include <QProgressBar>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLineEdit>
#include <QGroupBox>
#include <QFrame>
#include <QGridLayout>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QStyleOption>
#include <QMouseEvent>
#include <QWheelEvent>
#include <vector>

namespace MolinAntro {
namespace GUI {

// ============================================================================
// TRANSPORT PANEL
// ============================================================================

/**
 * @brief Professional transport controls with large, accessible buttons
 */
class TransportPanel : public QWidget {
    Q_OBJECT

public:
    explicit TransportPanel(QWidget* parent = nullptr);

    enum class State { Stopped, Playing, Paused, Recording };

    void setState(State state);
    State getState() const { return state_; }

    void setPosition(double beats);
    void setTempo(double bpm);
    void setTimeSignature(int num, int denom);
    void setLoopEnabled(bool enabled);
    void setMetronomeEnabled(bool enabled);

    // Time display mode
    enum class TimeMode { BarsBeats, MinSec, Samples, SMPTE };
    void setTimeMode(TimeMode mode);

signals:
    void playClicked();
    void stopClicked();
    void recordClicked();
    void rewindClicked();
    void forwardClicked();
    void loopToggled(bool enabled);
    void metronomeToggled(bool enabled);
    void tempoChanged(double bpm);
    void positionClicked();  // For seeking

private:
    void setupUI();
    void updateDisplay();

    State state_ = State::Stopped;
    TimeMode timeMode_ = TimeMode::BarsBeats;
    double position_ = 0.0;
    double tempo_ = 120.0;
    int timeNum_ = 4;
    int timeDenom_ = 4;
    bool loopEnabled_ = false;
    bool metronomeEnabled_ = true;

    // Buttons (minimum 44x32px)
    QPushButton* playButton_ = nullptr;
    QPushButton* stopButton_ = nullptr;
    QPushButton* recordButton_ = nullptr;
    QPushButton* rewindButton_ = nullptr;
    QPushButton* forwardButton_ = nullptr;
    QPushButton* loopButton_ = nullptr;
    QPushButton* metronomeButton_ = nullptr;

    // Display
    QLabel* positionLabel_ = nullptr;
    QLabel* tempoLabel_ = nullptr;
    QDoubleSpinBox* tempoSpin_ = nullptr;
};

// ============================================================================
// MIXER PANEL
// ============================================================================

/**
 * @brief Single channel strip for mixer
 */
class ChannelStrip : public QFrame {
    Q_OBJECT

public:
    explicit ChannelStrip(int index, QWidget* parent = nullptr);

    void setName(const QString& name);
    void setVolume(float dB);
    void setPan(float pan);
    void setMute(bool mute);
    void setSolo(bool solo);
    void setArm(bool arm);
    void setColor(const QColor& color);
    void updateMeter(float leftPeak, float rightPeak);

    int getIndex() const { return index_; }

signals:
    void volumeChanged(int index, float dB);
    void panChanged(int index, float pan);
    void muteToggled(int index, bool mute);
    void soloToggled(int index, bool solo);
    void armToggled(int index, bool arm);
    void selected(int index);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;

private:
    void setupUI();

    int index_;
    QString name_;
    float volume_ = 0.0f;
    float pan_ = 0.0f;
    bool mute_ = false;
    bool solo_ = false;
    bool arm_ = false;
    QColor color_;

    float meterL_ = 0.0f;
    float meterR_ = 0.0f;

    // UI elements
    QLabel* nameLabel_ = nullptr;
    QSlider* volumeFader_ = nullptr;
    QDial* panKnob_ = nullptr;
    QPushButton* muteButton_ = nullptr;
    QPushButton* soloButton_ = nullptr;
    QPushButton* armButton_ = nullptr;
    QLabel* volumeValue_ = nullptr;
};

/**
 * @brief Full mixer panel with scroll support
 */
class MixerPanel : public QWidget {
    Q_OBJECT

public:
    explicit MixerPanel(QWidget* parent = nullptr);

    void setChannelCount(int count);
    int getChannelCount() const { return channels_.size(); }

    ChannelStrip* getChannel(int index);
    void updateMeters(const std::vector<std::pair<float, float>>& levels);

signals:
    void channelVolumeChanged(int index, float dB);
    void channelPanChanged(int index, float pan);
    void channelMuteToggled(int index, bool mute);
    void channelSoloToggled(int index, bool solo);
    void channelSelected(int index);

private:
    void setupUI();

    std::vector<ChannelStrip*> channels_;
    QScrollArea* scrollArea_ = nullptr;
    QWidget* channelsContainer_ = nullptr;
    QHBoxLayout* channelsLayout_ = nullptr;

    // Master section
    ChannelStrip* masterChannel_ = nullptr;
};

// ============================================================================
// BROWSER PANEL
// ============================================================================

/**
 * @brief File/preset browser with proper scrolling
 */
class BrowserPanel : public QWidget {
    Q_OBJECT

public:
    explicit BrowserPanel(QWidget* parent = nullptr);

    enum class Category {
        Samples,
        Presets,
        Plugins,
        Projects,
        Favorites,
        Recent
    };

    void setCategory(Category category);
    void setRootPath(const QString& path);
    void refresh();

signals:
    void itemSelected(const QString& path);
    void itemDoubleClicked(const QString& path);
    void itemDragged(const QString& path);
    void previewRequested(const QString& path);

private slots:
    void onCategoryChanged(int index);
    void onSearchTextChanged(const QString& text);
    void onItemClicked(QTreeWidgetItem* item, int column);
    void onItemDoubleClicked(QTreeWidgetItem* item, int column);

private:
    void setupUI();
    void populateTree();

    Category currentCategory_ = Category::Samples;
    QString rootPath_;
    QString searchText_;

    // UI
    QComboBox* categoryCombo_ = nullptr;
    QLineEdit* searchBox_ = nullptr;
    QTreeWidget* fileTree_ = nullptr;  // Has built-in scrollbars
    QPushButton* refreshButton_ = nullptr;
    QPushButton* favoriteButton_ = nullptr;
};

// ============================================================================
// ARRANGEMENT PANEL
// ============================================================================

/**
 * @brief Timeline arrangement view
 */
class ArrangementPanel : public QWidget {
    Q_OBJECT

public:
    explicit ArrangementPanel(QWidget* parent = nullptr);

    void setZoom(double pixelsPerBeat);
    double getZoom() const { return pixelsPerBeat_; }
    void setPosition(double beats);
    void setTrackCount(int count);

signals:
    void zoomChanged(double pixelsPerBeat);
    void positionChanged(double beats);
    void trackAdded();
    void trackRemoved(int index);
    void regionMoved(int track, int region, double newStart);

protected:
    void paintEvent(QPaintEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;

private:
    void setupUI();
    void drawTimeline(QPainter& painter);
    void drawTracks(QPainter& painter);
    void drawPlayhead(QPainter& painter);

    double pixelsPerBeat_ = 20.0;
    double position_ = 0.0;
    double scrollOffset_ = 0.0;
    int trackCount_ = 8;

    QScrollArea* scrollArea_ = nullptr;
    QWidget* trackHeaders_ = nullptr;
    QWidget* timelineRuler_ = nullptr;
};

// ============================================================================
// PIANO ROLL PANEL
// ============================================================================

/**
 * @brief MIDI piano roll editor
 */
class PianoRollPanel : public QWidget {
    Q_OBJECT

public:
    explicit PianoRollPanel(QWidget* parent = nullptr);

    enum class Tool { Select, Draw, Erase, Velocity };

    void setTool(Tool tool);
    Tool getTool() const { return currentTool_; }

    void setZoom(double horizontal, double vertical);
    void setGridSize(double beats);  // 0.25 = 16th notes
    void setSnapToGrid(bool snap);

signals:
    void noteAdded(int pitch, double start, double length, int velocity);
    void noteRemoved(int index);
    void noteMoved(int index, int newPitch, double newStart);
    void velocityChanged(int index, int newVelocity);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void setupUI();
    void drawPianoKeys(QPainter& painter);
    void drawGrid(QPainter& painter);
    void drawNotes(QPainter& painter);

    Tool currentTool_ = Tool::Draw;
    double hZoom_ = 20.0;
    double vZoom_ = 12.0;
    double gridSize_ = 0.25;
    bool snapToGrid_ = true;

    QScrollArea* scrollArea_ = nullptr;
    QWidget* pianoKeys_ = nullptr;
};

// ============================================================================
// SESSION PANEL (Ableton-style clip launcher)
// ============================================================================

/**
 * @brief Session/clip launcher view
 */
class SessionPanel : public QWidget {
    Q_OBJECT

public:
    explicit SessionPanel(QWidget* parent = nullptr);

    void setGridSize(int tracks, int scenes);

    enum class ClipState { Empty, Stopped, Playing, Recording, Queued };
    void setClipState(int track, int scene, ClipState state);
    void setClipName(int track, int scene, const QString& name);
    void setClipColor(int track, int scene, const QColor& color);

signals:
    void clipLaunched(int track, int scene);
    void clipStopped(int track, int scene);
    void clipRecorded(int track, int scene);
    void sceneLaunched(int scene);
    void sceneStoppped(int scene);

private:
    void setupUI();

    struct ClipCell {
        QPushButton* button = nullptr;
        ClipState state = ClipState::Empty;
        QString name;
        QColor color;
    };

    std::vector<std::vector<ClipCell>> grid_;
    int trackCount_ = 8;
    int sceneCount_ = 8;

    QScrollArea* scrollArea_ = nullptr;
    QGridLayout* gridLayout_ = nullptr;
};

// ============================================================================
// CUSTOM WIDGETS
// ============================================================================

/**
 * @brief Professional rotary knob widget
 */
class KnobWidget : public QWidget {
    Q_OBJECT

public:
    explicit KnobWidget(QWidget* parent = nullptr);

    void setValue(float value);  // 0.0 to 1.0
    float getValue() const { return value_; }
    void setRange(float min, float max);
    void setLabel(const QString& label);
    void setBipolar(bool bipolar);

    QSize sizeHint() const override { return QSize(48, 64); }
    QSize minimumSizeHint() const override { return QSize(48, 64); }

signals:
    void valueChanged(float value);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    float value_ = 0.5f;
    float minValue_ = 0.0f;
    float maxValue_ = 1.0f;
    QString label_;
    bool bipolar_ = false;
    QPoint dragStart_;
    float dragStartValue_ = 0.0f;
};

/**
 * @brief VU/Peak meter widget
 */
class MeterWidget : public QWidget {
    Q_OBJECT

public:
    explicit MeterWidget(Qt::Orientation orientation = Qt::Vertical,
                         QWidget* parent = nullptr);

    void setLevel(float dB);
    void setPeak(float dB);
    void resetPeak();
    void setRange(float min, float max);  // dB
    void setStereo(bool stereo);

    QSize sizeHint() const override;
    QSize minimumSizeHint() const override;

protected:
    void paintEvent(QPaintEvent* event) override;
    void timerEvent(QTimerEvent* event) override;

private:
    Qt::Orientation orientation_;
    float level_ = -60.0f;
    float peakLevel_ = -60.0f;
    float targetLevel_ = -60.0f;
    float minDB_ = -60.0f;
    float maxDB_ = 6.0f;
    bool stereo_ = false;
    float levelL_ = -60.0f;
    float levelR_ = -60.0f;

    int timerId_ = 0;
};

/**
 * @brief Waveform display widget
 */
class WaveformWidget : public QWidget {
    Q_OBJECT

public:
    explicit WaveformWidget(QWidget* parent = nullptr);

    void setAudioData(const std::vector<float>& data, int sampleRate);
    void setZoom(double samplesPerPixel);
    void setOffset(int sampleOffset);
    void setPlayheadPosition(double seconds);
    void setSelection(double startSec, double endSec);

signals:
    void positionClicked(double seconds);
    void selectionChanged(double startSec, double endSec);

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    void mouseMoveEvent(QMouseEvent* event) override;
    void mouseReleaseEvent(QMouseEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void generateWaveformCache();

    std::vector<float> audioData_;
    std::vector<std::pair<float, float>> waveformCache_;  // min/max per pixel
    int sampleRate_ = 44100;
    double samplesPerPixel_ = 100.0;
    int sampleOffset_ = 0;
    double playheadPos_ = 0.0;
    double selectionStart_ = -1.0;
    double selectionEnd_ = -1.0;
    bool isDragging_ = false;
};

/**
 * @brief Spectrum analyzer display
 */
class SpectrumWidget : public QWidget {
    Q_OBJECT

public:
    explicit SpectrumWidget(QWidget* parent = nullptr);

    void setMagnitudes(const std::vector<float>& mags);
    void setFrequencyRange(float minHz, float maxHz);
    void setdBRange(float minDB, float maxDB);
    void setStyle(int style);  // 0=line, 1=filled, 2=bars

protected:
    void paintEvent(QPaintEvent* event) override;

private:
    std::vector<float> magnitudes_;
    std::vector<float> smoothedMags_;
    float minFreq_ = 20.0f;
    float maxFreq_ = 20000.0f;
    float minDB_ = -90.0f;
    float maxDB_ = 0.0f;
    int style_ = 1;
};

} // namespace GUI
} // namespace MolinAntro

#endif // BUILD_QT6_GUI
