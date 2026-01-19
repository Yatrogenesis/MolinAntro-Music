/**
 * @file Qt6Components.cpp
 * @brief Professional Qt6 UI Components implementation for MolinAntro DAW
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#ifdef BUILD_QT6_GUI

#include "gui/Qt6Components.h"
#include "gui/Qt6MainWindow.h"
#include "gui/Qt6Styles.h"

#include <QPainter>
#include <QPainterPath>
#include <QStyleOption>
#include <QFileSystemModel>
#include <cmath>

namespace MolinAntro {
namespace GUI {

// ============================================================================
// TRANSPORT PANEL
// ============================================================================

TransportPanel::TransportPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    updateDisplay();
}

void TransportPanel::setupUI() {
    auto* layout = new QHBoxLayout(this);
    layout->setSpacing(Style::WIDGET_SPACING);
    layout->setContentsMargins(Style::PANEL_MARGIN, 4, Style::PANEL_MARGIN, 4);

    // Transport buttons (minimum 44x32px as per guidelines)
    rewindButton_ = new QPushButton(QString::fromUtf8("\u23EE"), this);  // ⏮
    rewindButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    rewindButton_->setToolTip(tr("Rewind to Start"));
    connect(rewindButton_, &QPushButton::clicked, this, &TransportPanel::rewindClicked);
    layout->addWidget(rewindButton_);

    stopButton_ = new QPushButton(QString::fromUtf8("\u23F9"), this);  // ⏹
    stopButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    stopButton_->setToolTip(tr("Stop (Enter)"));
    connect(stopButton_, &QPushButton::clicked, this, &TransportPanel::stopClicked);
    layout->addWidget(stopButton_);

    playButton_ = new QPushButton(QString::fromUtf8("\u25B6"), this);  // ▶
    playButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH + 8, Style::MIN_BUTTON_HEIGHT);
    playButton_->setToolTip(tr("Play/Pause (Space)"));
    playButton_->setCheckable(true);
    connect(playButton_, &QPushButton::clicked, this, &TransportPanel::playClicked);
    layout->addWidget(playButton_);

    recordButton_ = new QPushButton(QString::fromUtf8("\u23FA"), this);  // ⏺
    recordButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    recordButton_->setToolTip(tr("Record (R)"));
    recordButton_->setCheckable(true);
    recordButton_->setStyleSheet(QString("QPushButton:checked { background-color: %1; }")
        .arg(StyleUtils::colorToRgba(Colors::RecordRed)));
    connect(recordButton_, &QPushButton::clicked, this, &TransportPanel::recordClicked);
    layout->addWidget(recordButton_);

    forwardButton_ = new QPushButton(QString::fromUtf8("\u23ED"), this);  // ⏭
    forwardButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    forwardButton_->setToolTip(tr("Forward"));
    connect(forwardButton_, &QPushButton::clicked, this, &TransportPanel::forwardClicked);
    layout->addWidget(forwardButton_);

    layout->addSpacing(16);

    // Loop button
    loopButton_ = new QPushButton(QString::fromUtf8("\u21BB"), this);  // ↻
    loopButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    loopButton_->setToolTip(tr("Loop (L)"));
    loopButton_->setCheckable(true);
    connect(loopButton_, &QPushButton::toggled, this, &TransportPanel::loopToggled);
    layout->addWidget(loopButton_);

    // Metronome button
    metronomeButton_ = new QPushButton(QString::fromUtf8("\u266A"), this);  // ♪
    metronomeButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    metronomeButton_->setToolTip(tr("Metronome"));
    metronomeButton_->setCheckable(true);
    metronomeButton_->setChecked(true);
    connect(metronomeButton_, &QPushButton::toggled, this, &TransportPanel::metronomeToggled);
    layout->addWidget(metronomeButton_);

    layout->addSpacing(16);

    // Position display (large, readable)
    positionLabel_ = new QLabel("001.01.000", this);
    positionLabel_->setMinimumWidth(120);
    positionLabel_->setAlignment(Qt::AlignCenter);
    positionLabel_->setStyleSheet(QString(
        "QLabel {"
        "  background-color: %1;"
        "  color: %2;"
        "  font-family: '%3';"
        "  font-size: %4px;"
        "  font-weight: bold;"
        "  padding: %5px %6px;"
        "  border-radius: 4px;"
        "}")
        .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
        .arg(StyleUtils::colorToRgba(Colors::AccentGreen))
        .arg(Fonts::getMonoFamily())
        .arg(Fonts::SizeLarge)
        .arg(Style::LABEL_PADDING / 2)
        .arg(Style::LABEL_PADDING));
    layout->addWidget(positionLabel_);

    layout->addSpacing(16);

    // Tempo controls
    tempoLabel_ = new QLabel(tr("BPM:"), this);
    tempoLabel_->setMinimumWidth(40);
    layout->addWidget(tempoLabel_);

    tempoSpin_ = new QDoubleSpinBox(this);
    tempoSpin_->setRange(20.0, 300.0);
    tempoSpin_->setValue(120.0);
    tempoSpin_->setSingleStep(1.0);
    tempoSpin_->setDecimals(1);
    tempoSpin_->setMinimumWidth(80);
    tempoSpin_->setMinimumHeight(Style::MIN_BUTTON_HEIGHT);
    connect(tempoSpin_, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
            this, &TransportPanel::tempoChanged);
    layout->addWidget(tempoSpin_);

    setLayout(layout);
}

void TransportPanel::setState(State state) {
    state_ = state;
    updateDisplay();
}

void TransportPanel::setPosition(double beats) {
    position_ = beats;
    updateDisplay();
}

void TransportPanel::setTempo(double bpm) {
    tempo_ = bpm;
    tempoSpin_->setValue(bpm);
}

void TransportPanel::setTimeSignature(int num, int denom) {
    timeNum_ = num;
    timeDenom_ = denom;
    updateDisplay();
}

void TransportPanel::setLoopEnabled(bool enabled) {
    loopEnabled_ = enabled;
    loopButton_->setChecked(enabled);
}

void TransportPanel::setMetronomeEnabled(bool enabled) {
    metronomeEnabled_ = enabled;
    metronomeButton_->setChecked(enabled);
}

void TransportPanel::setTimeMode(TimeMode mode) {
    timeMode_ = mode;
    updateDisplay();
}

void TransportPanel::updateDisplay() {
    // Update play button state
    bool isPlaying = (state_ == State::Playing || state_ == State::Recording);
    playButton_->setChecked(isPlaying);
    playButton_->setStyleSheet(isPlaying ?
        QString("QPushButton { background-color: %1; }")
            .arg(StyleUtils::colorToRgba(Colors::PlayGreen)) : "");

    // Update position display based on mode
    QString posText;
    switch (timeMode_) {
        case TimeMode::BarsBeats: {
            int bars = static_cast<int>(position_ / timeNum_) + 1;
            double beatInBar = std::fmod(position_, static_cast<double>(timeNum_));
            int beat = static_cast<int>(beatInBar) + 1;
            int ticks = static_cast<int>((beatInBar - static_cast<int>(beatInBar)) * 1000);
            posText = QString("%1.%2.%3")
                .arg(bars, 3, 10, QChar('0'))
                .arg(beat, 2, 10, QChar('0'))
                .arg(ticks, 3, 10, QChar('0'));
            break;
        }
        case TimeMode::MinSec: {
            double seconds = (position_ / tempo_) * 60.0;
            int mins = static_cast<int>(seconds / 60);
            int secs = static_cast<int>(std::fmod(seconds, 60.0));
            int ms = static_cast<int>((seconds - static_cast<int>(seconds)) * 1000);
            posText = QString("%1:%2.%3")
                .arg(mins, 2, 10, QChar('0'))
                .arg(secs, 2, 10, QChar('0'))
                .arg(ms, 3, 10, QChar('0'));
            break;
        }
        case TimeMode::Samples: {
            int samples = static_cast<int>((position_ / tempo_) * 60.0 * 44100);
            posText = QString::number(samples);
            break;
        }
        case TimeMode::SMPTE: {
            double seconds = (position_ / tempo_) * 60.0;
            int hours = static_cast<int>(seconds / 3600);
            int mins = static_cast<int>(std::fmod(seconds / 60, 60.0));
            int secs = static_cast<int>(std::fmod(seconds, 60.0));
            int frames = static_cast<int>((seconds - static_cast<int>(seconds)) * 30);
            posText = QString("%1:%2:%3:%4")
                .arg(hours, 2, 10, QChar('0'))
                .arg(mins, 2, 10, QChar('0'))
                .arg(secs, 2, 10, QChar('0'))
                .arg(frames, 2, 10, QChar('0'));
            break;
        }
    }
    positionLabel_->setText(posText);
}

// ============================================================================
// CHANNEL STRIP
// ============================================================================

ChannelStrip::ChannelStrip(int index, QWidget* parent)
    : QFrame(parent)
    , index_(index)
    , color_(Colors::TrackColors[index % Colors::TrackColorCount])
{
    setProperty("class", "ChannelStrip");
    setFrameStyle(QFrame::StyledPanel);
    setupUI();
}

void ChannelStrip::setupUI() {
    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(4);
    layout->setContentsMargins(4, 8, 4, 8);

    // Channel name (editable label area)
    nameLabel_ = new QLabel(QString("Track %1").arg(index_ + 1), this);
    nameLabel_->setAlignment(Qt::AlignCenter);
    nameLabel_->setMinimumHeight(24);
    nameLabel_->setStyleSheet(QString(
        "QLabel {"
        "  background-color: %1;"
        "  color: %2;"
        "  padding: 4px;"
        "  border-radius: 3px;"
        "  font-size: %3px;"
        "  font-weight: 500;"
        "}")
        .arg(StyleUtils::colorToRgba(color_))
        .arg(StyleUtils::colorToRgba(Colors::TextHighlight))
        .arg(Fonts::SizeSmall));
    layout->addWidget(nameLabel_);

    layout->addSpacing(4);

    // Pan knob
    panKnob_ = new QDial(this);
    panKnob_->setRange(-100, 100);
    panKnob_->setValue(0);
    panKnob_->setNotchesVisible(true);
    panKnob_->setMinimumSize(40, 40);
    panKnob_->setMaximumSize(48, 48);
    panKnob_->setToolTip(tr("Pan"));
    connect(panKnob_, &QDial::valueChanged, this, [this](int value) {
        pan_ = value / 100.0f;
        emit panChanged(index_, pan_);
    });
    layout->addWidget(panKnob_, 0, Qt::AlignCenter);

    // Volume fader
    volumeFader_ = new QSlider(Qt::Vertical, this);
    volumeFader_->setRange(-60, 6);
    volumeFader_->setValue(0);
    volumeFader_->setMinimumHeight(Style::FADER_HEIGHT);
    volumeFader_->setMinimumWidth(Style::FADER_WIDTH);
    volumeFader_->setToolTip(tr("Volume (dB)"));
    connect(volumeFader_, &QSlider::valueChanged, this, [this](int value) {
        volume_ = static_cast<float>(value);
        volumeValue_->setText(QString("%1 dB").arg(value));
        emit volumeChanged(index_, volume_);
    });
    layout->addWidget(volumeFader_, 1, Qt::AlignCenter);

    // Volume value display
    volumeValue_ = new QLabel("0 dB", this);
    volumeValue_->setAlignment(Qt::AlignCenter);
    volumeValue_->setMinimumWidth(50);
    volumeValue_->setStyleSheet(QString("font-size: %1px;").arg(Fonts::SizeSmall));
    layout->addWidget(volumeValue_);

    layout->addSpacing(4);

    // Button row (M S R)
    auto* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(2);

    muteButton_ = new QPushButton("M", this);
    muteButton_->setCheckable(true);
    muteButton_->setMinimumSize(24, 24);
    muteButton_->setMaximumSize(28, 28);
    muteButton_->setToolTip(tr("Mute"));
    muteButton_->setStyleSheet(
        "QPushButton:checked { background-color: " +
        StyleUtils::colorToRgba(Colors::AccentOrange) + "; }");
    connect(muteButton_, &QPushButton::toggled, this, [this](bool checked) {
        mute_ = checked;
        emit muteToggled(index_, mute_);
    });
    buttonLayout->addWidget(muteButton_);

    soloButton_ = new QPushButton("S", this);
    soloButton_->setCheckable(true);
    soloButton_->setMinimumSize(24, 24);
    soloButton_->setMaximumSize(28, 28);
    soloButton_->setToolTip(tr("Solo"));
    soloButton_->setStyleSheet(
        "QPushButton:checked { background-color: " +
        StyleUtils::colorToRgba(Colors::AccentYellow) + "; }");
    connect(soloButton_, &QPushButton::toggled, this, [this](bool checked) {
        solo_ = checked;
        emit soloToggled(index_, solo_);
    });
    buttonLayout->addWidget(soloButton_);

    armButton_ = new QPushButton("R", this);
    armButton_->setCheckable(true);
    armButton_->setMinimumSize(24, 24);
    armButton_->setMaximumSize(28, 28);
    armButton_->setToolTip(tr("Record Arm"));
    armButton_->setStyleSheet(
        "QPushButton:checked { background-color: " +
        StyleUtils::colorToRgba(Colors::RecordRed) + "; }");
    connect(armButton_, &QPushButton::toggled, this, [this](bool checked) {
        arm_ = checked;
        emit armToggled(index_, arm_);
    });
    buttonLayout->addWidget(armButton_);

    layout->addLayout(buttonLayout);

    setMinimumWidth(Style::CHANNEL_MIN_WIDTH);
    setMaximumWidth(Style::CHANNEL_WIDTH + 10);
}

void ChannelStrip::setName(const QString& name) {
    name_ = name;
    nameLabel_->setText(name);
}

void ChannelStrip::setVolume(float dB) {
    volume_ = dB;
    volumeFader_->setValue(static_cast<int>(dB));
    volumeValue_->setText(QString("%1 dB").arg(static_cast<int>(dB)));
}

void ChannelStrip::setPan(float pan) {
    pan_ = pan;
    panKnob_->setValue(static_cast<int>(pan * 100));
}

void ChannelStrip::setMute(bool mute) {
    mute_ = mute;
    muteButton_->setChecked(mute);
}

void ChannelStrip::setSolo(bool solo) {
    solo_ = solo;
    soloButton_->setChecked(solo);
}

void ChannelStrip::setArm(bool arm) {
    arm_ = arm;
    armButton_->setChecked(arm);
}

void ChannelStrip::setColor(const QColor& color) {
    color_ = color;
    nameLabel_->setStyleSheet(QString(
        "QLabel {"
        "  background-color: %1;"
        "  color: %2;"
        "  padding: 4px;"
        "  border-radius: 3px;"
        "}")
        .arg(StyleUtils::colorToRgba(color_))
        .arg(StyleUtils::colorToRgba(Colors::TextHighlight)));
}

void ChannelStrip::updateMeter(float leftPeak, float rightPeak) {
    meterL_ = leftPeak;
    meterR_ = rightPeak;
    update();
}

void ChannelStrip::paintEvent(QPaintEvent* event) {
    QFrame::paintEvent(event);

    // Draw meters alongside fader
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Meter position (to the right of fader)
    QRect faderGeom = volumeFader_->geometry();
    int meterWidth = Style::METER_WIDTH;
    int meterX = faderGeom.right() + 4;
    int meterHeight = faderGeom.height();
    int meterY = faderGeom.y();

    // Background
    painter.fillRect(meterX, meterY, meterWidth, meterHeight, Colors::MeterBackground);

    // Calculate meter levels (dB to linear)
    auto dbToHeight = [meterHeight](float dB) -> int {
        float normalized = (dB + 60.0f) / 66.0f;  // -60 to +6 dB range
        return static_cast<int>(std::max(0.0f, std::min(1.0f, normalized)) * meterHeight);
    };

    int heightL = dbToHeight(meterL_);
    int heightR = dbToHeight(meterR_);

    // Draw meter bars
    int halfWidth = (meterWidth - 2) / 2;

    // Left channel
    QLinearGradient gradL(meterX, meterY + meterHeight, meterX, meterY);
    gradL.setColorAt(0.0, Colors::MeterGreen);
    gradL.setColorAt(0.7, Colors::MeterGreen);
    gradL.setColorAt(0.85, Colors::MeterYellow);
    gradL.setColorAt(1.0, Colors::MeterRed);
    painter.fillRect(meterX, meterY + meterHeight - heightL, halfWidth, heightL, gradL);

    // Right channel
    QLinearGradient gradR(meterX + halfWidth + 2, meterY + meterHeight,
                          meterX + halfWidth + 2, meterY);
    gradR.setColorAt(0.0, Colors::MeterGreen);
    gradR.setColorAt(0.7, Colors::MeterGreen);
    gradR.setColorAt(0.85, Colors::MeterYellow);
    gradR.setColorAt(1.0, Colors::MeterRed);
    painter.fillRect(meterX + halfWidth + 2, meterY + meterHeight - heightR,
                     halfWidth, heightR, gradR);
}

void ChannelStrip::mousePressEvent(QMouseEvent* event) {
    QFrame::mousePressEvent(event);
    emit selected(index_);
}

// ============================================================================
// MIXER PANEL
// ============================================================================

MixerPanel::MixerPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    setChannelCount(8);  // Default 8 channels
}

void MixerPanel::setupUI() {
    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    layout->setSpacing(0);

    // Scroll area for channels (with visible scrollbar)
    scrollArea_ = new QScrollArea(this);
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    scrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scrollArea_->setMinimumHeight(280);

    // Container for channels
    channelsContainer_ = new QWidget(scrollArea_);
    channelsLayout_ = new QHBoxLayout(channelsContainer_);
    channelsLayout_->setSpacing(2);
    channelsLayout_->setContentsMargins(4, 4, 4, 4);
    channelsLayout_->addStretch();

    scrollArea_->setWidget(channelsContainer_);
    layout->addWidget(scrollArea_, 1);

    // Separator
    QFrame* sep = new QFrame(this);
    sep->setFrameShape(QFrame::VLine);
    sep->setStyleSheet(QString("background-color: %1;")
        .arg(StyleUtils::colorToRgba(Colors::Border)));
    layout->addWidget(sep);

    // Master channel (always visible)
    masterChannel_ = new ChannelStrip(-1, this);
    masterChannel_->setName(tr("Master"));
    masterChannel_->setColor(Colors::TextSecondary);
    masterChannel_->setMinimumWidth(Style::CHANNEL_WIDTH);
    layout->addWidget(masterChannel_);
}

void MixerPanel::setChannelCount(int count) {
    // Remove existing channels
    for (auto* channel : channels_) {
        channelsLayout_->removeWidget(channel);
        delete channel;
    }
    channels_.clear();

    // Remove stretch
    QLayoutItem* stretch = channelsLayout_->takeAt(channelsLayout_->count() - 1);
    delete stretch;

    // Create new channels
    for (int i = 0; i < count; ++i) {
        auto* channel = new ChannelStrip(i, channelsContainer_);
        channels_.push_back(channel);
        channelsLayout_->addWidget(channel);

        // Connect signals
        connect(channel, &ChannelStrip::volumeChanged, this, &MixerPanel::channelVolumeChanged);
        connect(channel, &ChannelStrip::panChanged, this, &MixerPanel::channelPanChanged);
        connect(channel, &ChannelStrip::muteToggled, this, &MixerPanel::channelMuteToggled);
        connect(channel, &ChannelStrip::soloToggled, this, &MixerPanel::channelSoloToggled);
        connect(channel, &ChannelStrip::selected, this, &MixerPanel::channelSelected);
    }

    // Re-add stretch
    channelsLayout_->addStretch();
}

ChannelStrip* MixerPanel::getChannel(int index) {
    if (index >= 0 && index < static_cast<int>(channels_.size())) {
        return channels_[index];
    }
    return nullptr;
}

void MixerPanel::updateMeters(const std::vector<std::pair<float, float>>& levels) {
    for (size_t i = 0; i < std::min(levels.size(), channels_.size()); ++i) {
        channels_[i]->updateMeter(levels[i].first, levels[i].second);
    }
}

// ============================================================================
// BROWSER PANEL
// ============================================================================

BrowserPanel::BrowserPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
}

void BrowserPanel::setupUI() {
    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(Style::WIDGET_SPACING);
    layout->setContentsMargins(Style::PANEL_MARGIN, Style::PANEL_MARGIN,
                               Style::PANEL_MARGIN, Style::PANEL_MARGIN);

    // Category selector
    categoryCombo_ = new QComboBox(this);
    categoryCombo_->addItem(tr("Samples"), static_cast<int>(Category::Samples));
    categoryCombo_->addItem(tr("Presets"), static_cast<int>(Category::Presets));
    categoryCombo_->addItem(tr("Plugins"), static_cast<int>(Category::Plugins));
    categoryCombo_->addItem(tr("Projects"), static_cast<int>(Category::Projects));
    categoryCombo_->addItem(tr("Favorites"), static_cast<int>(Category::Favorites));
    categoryCombo_->addItem(tr("Recent"), static_cast<int>(Category::Recent));
    categoryCombo_->setMinimumHeight(Style::MIN_BUTTON_HEIGHT);
    connect(categoryCombo_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &BrowserPanel::onCategoryChanged);
    layout->addWidget(categoryCombo_);

    // Search box
    searchBox_ = new QLineEdit(this);
    searchBox_->setPlaceholderText(tr("Search..."));
    searchBox_->setMinimumHeight(Style::MIN_BUTTON_HEIGHT);
    searchBox_->setClearButtonEnabled(true);
    connect(searchBox_, &QLineEdit::textChanged, this, &BrowserPanel::onSearchTextChanged);
    layout->addWidget(searchBox_);

    // File tree (with visible scrollbars)
    fileTree_ = new QTreeWidget(this);
    fileTree_->setHeaderHidden(true);
    fileTree_->setRootIsDecorated(true);
    fileTree_->setAnimated(true);
    fileTree_->setIndentation(16);
    fileTree_->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    fileTree_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    connect(fileTree_, &QTreeWidget::itemClicked, this, &BrowserPanel::onItemClicked);
    connect(fileTree_, &QTreeWidget::itemDoubleClicked, this, &BrowserPanel::onItemDoubleClicked);
    layout->addWidget(fileTree_, 1);

    // Button row
    auto* buttonLayout = new QHBoxLayout();
    buttonLayout->setSpacing(4);

    refreshButton_ = new QPushButton(tr("Refresh"), this);
    refreshButton_->setMinimumSize(Style::MIN_BUTTON_WIDTH, Style::MIN_BUTTON_HEIGHT);
    connect(refreshButton_, &QPushButton::clicked, this, &BrowserPanel::refresh);
    buttonLayout->addWidget(refreshButton_);

    favoriteButton_ = new QPushButton(QString::fromUtf8("\u2605"), this);  // ★
    favoriteButton_->setMinimumSize(Style::MIN_BUTTON_HEIGHT, Style::MIN_BUTTON_HEIGHT);
    favoriteButton_->setMaximumWidth(Style::MIN_BUTTON_HEIGHT);
    favoriteButton_->setToolTip(tr("Add to Favorites"));
    buttonLayout->addWidget(favoriteButton_);

    buttonLayout->addStretch();
    layout->addLayout(buttonLayout);
}

void BrowserPanel::setCategory(Category category) {
    currentCategory_ = category;
    categoryCombo_->setCurrentIndex(static_cast<int>(category));
    populateTree();
}

void BrowserPanel::setRootPath(const QString& path) {
    rootPath_ = path;
    populateTree();
}

void BrowserPanel::refresh() {
    populateTree();
}

void BrowserPanel::onCategoryChanged(int index) {
    currentCategory_ = static_cast<Category>(categoryCombo_->itemData(index).toInt());
    populateTree();
}

void BrowserPanel::onSearchTextChanged(const QString& text) {
    searchText_ = text;
    // Filter tree items
    for (int i = 0; i < fileTree_->topLevelItemCount(); ++i) {
        QTreeWidgetItem* item = fileTree_->topLevelItem(i);
        bool matches = text.isEmpty() ||
                       item->text(0).contains(text, Qt::CaseInsensitive);
        item->setHidden(!matches);
    }
}

void BrowserPanel::onItemClicked(QTreeWidgetItem* item, int) {
    QString path = item->data(0, Qt::UserRole).toString();
    emit itemSelected(path);
}

void BrowserPanel::onItemDoubleClicked(QTreeWidgetItem* item, int) {
    QString path = item->data(0, Qt::UserRole).toString();
    emit itemDoubleClicked(path);
}

void BrowserPanel::populateTree() {
    fileTree_->clear();

    // Placeholder items for demonstration
    QStringList items;
    switch (currentCategory_) {
        case Category::Samples:
            items << "Drums" << "Bass" << "Synths" << "FX" << "Vocals";
            break;
        case Category::Presets:
            items << "Leads" << "Pads" << "Bass" << "Drums" << "FX";
            break;
        case Category::Plugins:
            items << "Instruments" << "Effects" << "MIDI Effects";
            break;
        case Category::Projects:
            items << "Recent Projects" << "Templates";
            break;
        case Category::Favorites:
            items << "Favorite Samples" << "Favorite Presets";
            break;
        case Category::Recent:
            items << "Today" << "This Week" << "This Month";
            break;
    }

    for (const QString& item : items) {
        QTreeWidgetItem* treeItem = new QTreeWidgetItem(fileTree_);
        treeItem->setText(0, item);
        treeItem->setData(0, Qt::UserRole, rootPath_ + "/" + item);
    }
}

// ============================================================================
// ARRANGEMENT PANEL
// ============================================================================

ArrangementPanel::ArrangementPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
}

void ArrangementPanel::setupUI() {
    // Will be populated with timeline, tracks, etc.
    setMinimumSize(400, 200);
    setStyleSheet(QString("background-color: %1;")
        .arg(StyleUtils::colorToRgba(Colors::BackgroundDark)));
}

void ArrangementPanel::setZoom(double pixelsPerBeat) {
    pixelsPerBeat_ = std::max(5.0, std::min(200.0, pixelsPerBeat));
    emit zoomChanged(pixelsPerBeat_);
    update();
}

void ArrangementPanel::setPosition(double beats) {
    position_ = beats;
    emit positionChanged(position_);
    update();
}

void ArrangementPanel::setTrackCount(int count) {
    trackCount_ = count;
    update();
}

void ArrangementPanel::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Draw background
    painter.fillRect(rect(), Colors::BackgroundDark);

    // Draw grid
    drawTimeline(painter);
    drawTracks(painter);
    drawPlayhead(painter);
}

void ArrangementPanel::drawTimeline(QPainter& painter) {
    int rulerHeight = 24;

    // Ruler background
    painter.fillRect(0, 0, width(), rulerHeight, Colors::Panel);

    // Draw beat lines and numbers
    painter.setPen(Colors::TextSecondary);
    painter.setFont(QFont(Fonts::getFontFamily(), Fonts::SizeSmall));

    double beatsVisible = width() / pixelsPerBeat_;
    int startBeat = static_cast<int>(scrollOffset_ / pixelsPerBeat_);

    for (int beat = startBeat; beat < startBeat + static_cast<int>(beatsVisible) + 2; ++beat) {
        double x = beat * pixelsPerBeat_ - scrollOffset_;
        if (x < 0 || x > width()) continue;

        // Major lines (every 4 beats)
        if (beat % 4 == 0) {
            painter.setPen(Colors::GridMajor);
            painter.drawLine(QPointF(x, rulerHeight), QPointF(x, height()));
            painter.setPen(Colors::TextPrimary);
            painter.drawText(QPointF(x + 4, rulerHeight - 6), QString::number(beat / 4 + 1));
        } else {
            // Minor lines
            painter.setPen(Colors::GridMinor);
            painter.drawLine(QPointF(x, rulerHeight), QPointF(x, height()));
        }
    }

    // Bottom border
    painter.setPen(Colors::Border);
    painter.drawLine(0, rulerHeight, width(), rulerHeight);
}

void ArrangementPanel::drawTracks(QPainter& painter) {
    int rulerHeight = 24;
    int trackHeight = Style::TRACK_DEFAULT_HEIGHT;

    for (int i = 0; i < trackCount_; ++i) {
        int y = rulerHeight + i * trackHeight;

        // Track background (alternating)
        QColor bgColor = (i % 2 == 0) ? Colors::BackgroundDark : Colors::Panel;
        painter.fillRect(0, y, width(), trackHeight, bgColor);

        // Track separator
        painter.setPen(Colors::Separator);
        painter.drawLine(0, y + trackHeight - 1, width(), y + trackHeight - 1);
    }
}

void ArrangementPanel::drawPlayhead(QPainter& painter) {
    int rulerHeight = 24;
    double x = position_ * pixelsPerBeat_ - scrollOffset_;

    if (x >= 0 && x <= width()) {
        painter.setPen(QPen(Colors::Playhead, 2));
        painter.drawLine(QPointF(x, 0), QPointF(x, height()));

        // Playhead triangle
        QPainterPath path;
        path.moveTo(x - 6, 0);
        path.lineTo(x + 6, 0);
        path.lineTo(x, rulerHeight / 2);
        path.closeSubpath();
        painter.fillPath(path, Colors::Playhead);
    }
}

void ArrangementPanel::wheelEvent(QWheelEvent* event) {
    if (event->modifiers() & Qt::ControlModifier) {
        // Zoom
        double factor = event->angleDelta().y() > 0 ? 1.2 : 1.0 / 1.2;
        setZoom(pixelsPerBeat_ * factor);
    } else {
        // Scroll
        scrollOffset_ -= event->angleDelta().x() + event->angleDelta().y();
        scrollOffset_ = std::max(0.0, scrollOffset_);
        update();
    }
    event->accept();
}

void ArrangementPanel::mousePressEvent(QMouseEvent* event) {
    double beats = (event->position().x() + scrollOffset_) / pixelsPerBeat_;
    setPosition(beats);
}

void ArrangementPanel::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        double beats = (event->position().x() + scrollOffset_) / pixelsPerBeat_;
        setPosition(beats);
    }
}

void ArrangementPanel::mouseReleaseEvent(QMouseEvent*) {
    // Handle release
}

// ============================================================================
// PIANO ROLL PANEL
// ============================================================================

PianoRollPanel::PianoRollPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
}

void PianoRollPanel::setupUI() {
    setMinimumSize(400, 200);
    setStyleSheet(QString("background-color: %1;")
        .arg(StyleUtils::colorToRgba(Colors::BackgroundDark)));
}

void PianoRollPanel::setTool(Tool tool) {
    currentTool_ = tool;
}

void PianoRollPanel::setZoom(double horizontal, double vertical) {
    hZoom_ = horizontal;
    vZoom_ = vertical;
    update();
}

void PianoRollPanel::setGridSize(double beats) {
    gridSize_ = beats;
    update();
}

void PianoRollPanel::setSnapToGrid(bool snap) {
    snapToGrid_ = snap;
}

void PianoRollPanel::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    drawPianoKeys(painter);
    drawGrid(painter);
    drawNotes(painter);
}

void PianoRollPanel::drawPianoKeys(QPainter& painter) {
    int keyWidth = 40;

    // Draw 128 MIDI notes
    for (int note = 0; note < 128; ++note) {
        int y = static_cast<int>((127 - note) * vZoom_);
        bool isBlack = false;
        int noteInOctave = note % 12;
        if (noteInOctave == 1 || noteInOctave == 3 ||
            noteInOctave == 6 || noteInOctave == 8 || noteInOctave == 10) {
            isBlack = true;
        }

        QColor color = isBlack ? Colors::BackgroundDark : Colors::TextPrimary;
        painter.fillRect(0, y, keyWidth, static_cast<int>(vZoom_), color);

        // Key border
        painter.setPen(Colors::Border);
        painter.drawLine(0, y, keyWidth, y);

        // Note name for C notes
        if (noteInOctave == 0) {
            painter.setPen(isBlack ? Colors::TextPrimary : Colors::BackgroundDark);
            painter.setFont(QFont(Fonts::getFontFamily(), Fonts::SizeSmall - 2));
            painter.drawText(2, y + static_cast<int>(vZoom_) - 2,
                           QString("C%1").arg(note / 12 - 1));
        }
    }

    // Right border
    painter.setPen(Colors::Border);
    painter.drawLine(keyWidth, 0, keyWidth, height());
}

void PianoRollPanel::drawGrid(QPainter& painter) {
    int keyWidth = 40;

    // Vertical grid (time)
    double beatsVisible = (width() - keyWidth) / hZoom_;

    for (double beat = 0; beat < beatsVisible; beat += gridSize_) {
        int x = keyWidth + static_cast<int>(beat * hZoom_);
        bool isMajor = std::fmod(beat, 1.0) < 0.001;

        painter.setPen(isMajor ? Colors::GridMajor : Colors::GridMinor);
        painter.drawLine(x, 0, x, height());
    }

    // Horizontal grid (pitch) - only for octave lines
    for (int note = 0; note < 128; note += 12) {
        int y = static_cast<int>((127 - note) * vZoom_);
        painter.setPen(Colors::GridMajor);
        painter.drawLine(keyWidth, y, width(), y);
    }
}

void PianoRollPanel::drawNotes(QPainter& painter) {
    // Draw MIDI notes (placeholder - would come from MIDI clip data)
    // Notes are drawn as rounded rectangles
}

void PianoRollPanel::mousePressEvent(QMouseEvent* event) {
    // Handle note creation/selection based on current tool
    QWidget::mousePressEvent(event);
}

void PianoRollPanel::mouseMoveEvent(QMouseEvent* event) {
    QWidget::mouseMoveEvent(event);
}

void PianoRollPanel::mouseReleaseEvent(QMouseEvent* event) {
    QWidget::mouseReleaseEvent(event);
}

void PianoRollPanel::wheelEvent(QWheelEvent* event) {
    if (event->modifiers() & Qt::ControlModifier) {
        // Zoom
        double factor = event->angleDelta().y() > 0 ? 1.1 : 1.0 / 1.1;
        if (event->modifiers() & Qt::ShiftModifier) {
            vZoom_ *= factor;
        } else {
            hZoom_ *= factor;
        }
        update();
    }
    event->accept();
}

// ============================================================================
// SESSION PANEL
// ============================================================================

SessionPanel::SessionPanel(QWidget* parent) : QWidget(parent) {
    setupUI();
    setGridSize(8, 8);
}

void SessionPanel::setupUI() {
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);

    scrollArea_ = new QScrollArea(this);
    scrollArea_->setWidgetResizable(true);
    scrollArea_->setHorizontalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    scrollArea_->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);

    QWidget* container = new QWidget(scrollArea_);
    gridLayout_ = new QGridLayout(container);
    gridLayout_->setSpacing(2);
    gridLayout_->setContentsMargins(4, 4, 4, 4);

    scrollArea_->setWidget(container);
    layout->addWidget(scrollArea_);
}

void SessionPanel::setGridSize(int tracks, int scenes) {
    // Clear existing grid
    QLayoutItem* item;
    while ((item = gridLayout_->takeAt(0)) != nullptr) {
        delete item->widget();
        delete item;
    }
    grid_.clear();

    trackCount_ = tracks;
    sceneCount_ = scenes;

    // Track headers
    for (int t = 0; t < tracks; ++t) {
        QLabel* header = new QLabel(QString("Track %1").arg(t + 1));
        header->setAlignment(Qt::AlignCenter);
        header->setMinimumHeight(24);
        header->setStyleSheet(QString("background-color: %1; color: %2; font-weight: bold;")
            .arg(StyleUtils::colorToRgba(Colors::TrackColors[t % Colors::TrackColorCount]))
            .arg(StyleUtils::colorToRgba(Colors::TextHighlight)));
        gridLayout_->addWidget(header, 0, t + 1);
    }

    // Scene launch buttons
    for (int s = 0; s < scenes; ++s) {
        QPushButton* sceneBtn = new QPushButton(QString::fromUtf8("\u25B6"));
        sceneBtn->setMinimumSize(32, Style::MIN_BUTTON_HEIGHT);
        sceneBtn->setMaximumWidth(32);
        sceneBtn->setToolTip(tr("Launch Scene %1").arg(s + 1));
        connect(sceneBtn, &QPushButton::clicked, this, [this, s]() {
            emit sceneLaunched(s);
        });
        gridLayout_->addWidget(sceneBtn, s + 1, 0);
    }

    // Clip grid
    grid_.resize(tracks);
    for (int t = 0; t < tracks; ++t) {
        grid_[t].resize(scenes);
        for (int s = 0; s < scenes; ++s) {
            QPushButton* clipBtn = new QPushButton();
            clipBtn->setMinimumSize(80, Style::MIN_BUTTON_HEIGHT + 8);
            clipBtn->setStyleSheet(QString(
                "QPushButton { background-color: %1; border: 1px solid %2; border-radius: 4px; }"
                "QPushButton:hover { border-color: %3; }")
                .arg(StyleUtils::colorToRgba(Colors::Panel))
                .arg(StyleUtils::colorToRgba(Colors::Border))
                .arg(StyleUtils::colorToRgba(Colors::AccentBlue)));

            connect(clipBtn, &QPushButton::clicked, this, [this, t, s]() {
                emit clipLaunched(t, s);
            });

            gridLayout_->addWidget(clipBtn, s + 1, t + 1);

            grid_[t][s].button = clipBtn;
            grid_[t][s].state = ClipState::Empty;
        }
    }

    // Stop all button
    QPushButton* stopAll = new QPushButton(QString::fromUtf8("\u23F9"));
    stopAll->setMinimumSize(32, Style::MIN_BUTTON_HEIGHT);
    stopAll->setToolTip(tr("Stop All Clips"));
    gridLayout_->addWidget(stopAll, scenes + 1, 0);
}

void SessionPanel::setClipState(int track, int scene, ClipState state) {
    if (track < 0 || track >= trackCount_ || scene < 0 || scene >= sceneCount_) {
        return;
    }

    grid_[track][scene].state = state;

    QColor bgColor;
    QString icon;
    switch (state) {
        case ClipState::Empty:
            bgColor = Colors::Panel;
            icon = "";
            break;
        case ClipState::Stopped:
            bgColor = Colors::TrackColors[track % Colors::TrackColorCount].darker(150);
            icon = QString::fromUtf8("\u25A0");  // ■
            break;
        case ClipState::Playing:
            bgColor = Colors::PlayGreen;
            icon = QString::fromUtf8("\u25B6");  // ▶
            break;
        case ClipState::Recording:
            bgColor = Colors::RecordRed;
            icon = QString::fromUtf8("\u23FA");  // ⏺
            break;
        case ClipState::Queued:
            bgColor = Colors::AccentOrange;
            icon = QString::fromUtf8("\u25B7");  // ▷
            break;
    }

    grid_[track][scene].button->setText(grid_[track][scene].name.isEmpty() ?
                                         icon : grid_[track][scene].name);
    grid_[track][scene].button->setStyleSheet(QString(
        "QPushButton { background-color: %1; border: 1px solid %2; border-radius: 4px; }"
        "QPushButton:hover { border-color: %3; }")
        .arg(StyleUtils::colorToRgba(bgColor))
        .arg(StyleUtils::colorToRgba(Colors::Border))
        .arg(StyleUtils::colorToRgba(Colors::AccentBlue)));
}

void SessionPanel::setClipName(int track, int scene, const QString& name) {
    if (track < 0 || track >= trackCount_ || scene < 0 || scene >= sceneCount_) {
        return;
    }
    grid_[track][scene].name = name;
    if (!name.isEmpty()) {
        grid_[track][scene].button->setText(name);
    }
}

void SessionPanel::setClipColor(int track, int scene, const QColor& color) {
    if (track < 0 || track >= trackCount_ || scene < 0 || scene >= sceneCount_) {
        return;
    }
    grid_[track][scene].color = color;
}

// ============================================================================
// KNOB WIDGET
// ============================================================================

KnobWidget::KnobWidget(QWidget* parent) : QWidget(parent) {
    setMinimumSize(Style::KNOB_SIZE, Style::KNOB_SIZE + 16);
}

void KnobWidget::setValue(float value) {
    value_ = std::max(0.0f, std::min(1.0f, value));
    emit valueChanged(value_);
    update();
}

void KnobWidget::setRange(float min, float max) {
    minValue_ = min;
    maxValue_ = max;
}

void KnobWidget::setLabel(const QString& label) {
    label_ = label;
    update();
}

void KnobWidget::setBipolar(bool bipolar) {
    bipolar_ = bipolar;
    update();
}

void KnobWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    int size = std::min(width(), height() - 16);
    int x = (width() - size) / 2;
    int y = 0;
    QRectF knobRect(x + 4, y + 4, size - 8, size - 8);

    // Outer ring
    painter.setPen(QPen(Colors::Border, 2));
    painter.setBrush(Colors::BackgroundDark);
    painter.drawEllipse(knobRect);

    // Arc (value indicator)
    double startAngle = 225;  // degrees
    double endAngle = -45;    // degrees
    double range = startAngle - endAngle;
    double valueAngle;

    if (bipolar_) {
        valueAngle = startAngle - (value_ * range);
        double centerAngle = startAngle - (range / 2);

        painter.setPen(QPen(Colors::AccentBlue, 3));
        if (value_ > 0.5f) {
            painter.drawArc(knobRect.adjusted(2, 2, -2, -2),
                           static_cast<int>(centerAngle * 16),
                           static_cast<int>((centerAngle - valueAngle) * 16));
        } else {
            painter.drawArc(knobRect.adjusted(2, 2, -2, -2),
                           static_cast<int>(valueAngle * 16),
                           static_cast<int>((centerAngle - valueAngle) * 16));
        }
    } else {
        valueAngle = startAngle - (value_ * range);
        painter.setPen(QPen(Colors::AccentBlue, 3));
        painter.drawArc(knobRect.adjusted(2, 2, -2, -2),
                       static_cast<int>(startAngle * 16),
                       static_cast<int>((startAngle - valueAngle) * 16));
    }

    // Pointer
    double pointerAngle = (startAngle - (value_ * range)) * M_PI / 180.0;
    QPointF center = knobRect.center();
    double radius = knobRect.width() / 2 - 8;
    QPointF end(center.x() + radius * std::cos(pointerAngle),
                center.y() - radius * std::sin(pointerAngle));

    painter.setPen(QPen(Colors::TextPrimary, 2));
    painter.drawLine(center, end);

    // Center dot
    painter.setBrush(Colors::TextPrimary);
    painter.setPen(Qt::NoPen);
    painter.drawEllipse(center, 3, 3);

    // Label
    if (!label_.isEmpty()) {
        painter.setPen(Colors::TextSecondary);
        painter.setFont(QFont(Fonts::getFontFamily(), Fonts::SizeSmall - 1));
        painter.drawText(QRect(0, size, width(), 16), Qt::AlignCenter, label_);
    }
}

void KnobWidget::mousePressEvent(QMouseEvent* event) {
    dragStart_ = event->pos();
    dragStartValue_ = value_;
}

void KnobWidget::mouseMoveEvent(QMouseEvent* event) {
    if (event->buttons() & Qt::LeftButton) {
        int dy = dragStart_.y() - event->pos().y();
        float delta = dy / 200.0f;
        setValue(dragStartValue_ + delta);
    }
}

void KnobWidget::wheelEvent(QWheelEvent* event) {
    float delta = event->angleDelta().y() / 1200.0f;
    setValue(value_ + delta);
    event->accept();
}

// ============================================================================
// METER WIDGET
// ============================================================================

MeterWidget::MeterWidget(Qt::Orientation orientation, QWidget* parent)
    : QWidget(parent)
    , orientation_(orientation)
{
    timerId_ = startTimer(30);  // ~33 FPS for smooth decay
}

void MeterWidget::setLevel(float dB) {
    targetLevel_ = std::max(minDB_, std::min(maxDB_, dB));
    if (targetLevel_ > level_) {
        level_ = targetLevel_;  // Instant attack
    }
}

void MeterWidget::setPeak(float dB) {
    if (dB > peakLevel_) {
        peakLevel_ = dB;
    }
}

void MeterWidget::resetPeak() {
    peakLevel_ = minDB_;
}

void MeterWidget::setRange(float min, float max) {
    minDB_ = min;
    maxDB_ = max;
}

void MeterWidget::setStereo(bool stereo) {
    stereo_ = stereo;
    update();
}

QSize MeterWidget::sizeHint() const {
    if (orientation_ == Qt::Vertical) {
        return QSize(stereo_ ? Style::METER_WIDTH * 2 + 2 : Style::METER_WIDTH, 100);
    } else {
        return QSize(100, stereo_ ? Style::METER_WIDTH * 2 + 2 : Style::METER_WIDTH);
    }
}

QSize MeterWidget::minimumSizeHint() const {
    return sizeHint();
}

void MeterWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Background
    painter.fillRect(rect(), Colors::MeterBackground);

    // Calculate normalized level
    float range = maxDB_ - minDB_;
    float normalizedLevel = (level_ - minDB_) / range;
    float normalizedPeak = (peakLevel_ - minDB_) / range;

    // Gradient
    QLinearGradient gradient;
    if (orientation_ == Qt::Vertical) {
        gradient = QLinearGradient(0, height(), 0, 0);
    } else {
        gradient = QLinearGradient(0, 0, width(), 0);
    }
    gradient.setColorAt(0.0, Colors::MeterGreen);
    gradient.setColorAt(0.6, Colors::MeterGreen);
    gradient.setColorAt(0.85, Colors::MeterYellow);
    gradient.setColorAt(1.0, Colors::MeterRed);

    if (orientation_ == Qt::Vertical) {
        int meterHeight = static_cast<int>(normalizedLevel * height());
        painter.fillRect(0, height() - meterHeight, width(), meterHeight, gradient);

        // Peak indicator
        int peakY = height() - static_cast<int>(normalizedPeak * height());
        painter.setPen(QPen(Colors::TextHighlight, 2));
        painter.drawLine(0, peakY, width(), peakY);
    } else {
        int meterWidth = static_cast<int>(normalizedLevel * width());
        painter.fillRect(0, 0, meterWidth, height(), gradient);

        // Peak indicator
        int peakX = static_cast<int>(normalizedPeak * width());
        painter.setPen(QPen(Colors::TextHighlight, 2));
        painter.drawLine(peakX, 0, peakX, height());
    }
}

void MeterWidget::timerEvent(QTimerEvent*) {
    // Decay
    const float decayRate = 0.95f;
    if (level_ > targetLevel_) {
        level_ = level_ * decayRate + targetLevel_ * (1.0f - decayRate);
        if (level_ < targetLevel_ + 0.1f) {
            level_ = targetLevel_;
        }
    }

    // Peak hold and decay (slower)
    static int peakHoldCounter = 0;
    if (peakHoldCounter++ > 30) {  // ~1 second hold
        peakLevel_ *= 0.98f;
        if (peakLevel_ < minDB_) {
            peakLevel_ = minDB_;
        }
    }

    update();
}

// ============================================================================
// WAVEFORM WIDGET
// ============================================================================

WaveformWidget::WaveformWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(60);
}

void WaveformWidget::setAudioData(const std::vector<float>& data, int sampleRate) {
    audioData_ = data;
    sampleRate_ = sampleRate;
    generateWaveformCache();
    update();
}

void WaveformWidget::setZoom(double samplesPerPixel) {
    samplesPerPixel_ = samplesPerPixel;
    generateWaveformCache();
    update();
}

void WaveformWidget::setOffset(int sampleOffset) {
    sampleOffset_ = sampleOffset;
    update();
}

void WaveformWidget::setPlayheadPosition(double seconds) {
    playheadPos_ = seconds;
    update();
}

void WaveformWidget::setSelection(double startSec, double endSec) {
    selectionStart_ = startSec;
    selectionEnd_ = endSec;
    update();
}

void WaveformWidget::generateWaveformCache() {
    waveformCache_.clear();

    if (audioData_.empty()) return;

    int pixels = width();
    waveformCache_.resize(pixels);

    for (int px = 0; px < pixels; ++px) {
        int startSample = sampleOffset_ + static_cast<int>(px * samplesPerPixel_);
        int endSample = startSample + static_cast<int>(samplesPerPixel_);

        float minVal = 0.0f;
        float maxVal = 0.0f;

        for (int s = startSample; s < endSample && s < static_cast<int>(audioData_.size()); ++s) {
            if (s >= 0) {
                minVal = std::min(minVal, audioData_[s]);
                maxVal = std::max(maxVal, audioData_[s]);
            }
        }

        waveformCache_[px] = {minVal, maxVal};
    }
}

void WaveformWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Background
    painter.fillRect(rect(), Colors::BackgroundDark);

    // Selection
    if (selectionStart_ >= 0 && selectionEnd_ > selectionStart_) {
        int startX = static_cast<int>((selectionStart_ * sampleRate_ - sampleOffset_) / samplesPerPixel_);
        int endX = static_cast<int>((selectionEnd_ * sampleRate_ - sampleOffset_) / samplesPerPixel_);
        painter.fillRect(startX, 0, endX - startX, height(), Colors::Selection);
    }

    // Center line
    int centerY = height() / 2;
    painter.setPen(Colors::GridMinor);
    painter.drawLine(0, centerY, width(), centerY);

    // Waveform
    if (!waveformCache_.empty()) {
        painter.setPen(Qt::NoPen);
        painter.setBrush(Colors::WaveformFill);

        QPainterPath path;
        path.moveTo(0, centerY);

        // Top half
        for (size_t px = 0; px < waveformCache_.size(); ++px) {
            int y = centerY - static_cast<int>(waveformCache_[px].second * centerY);
            path.lineTo(px, y);
        }

        // Bottom half (reverse)
        for (int px = static_cast<int>(waveformCache_.size()) - 1; px >= 0; --px) {
            int y = centerY - static_cast<int>(waveformCache_[px].first * centerY);
            path.lineTo(px, y);
        }

        path.closeSubpath();
        painter.drawPath(path);

        // Outline
        painter.setPen(Colors::WaveformStroke);
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);
    }

    // Playhead
    int playheadX = static_cast<int>((playheadPos_ * sampleRate_ - sampleOffset_) / samplesPerPixel_);
    if (playheadX >= 0 && playheadX < width()) {
        painter.setPen(QPen(Colors::Playhead, 2));
        painter.drawLine(playheadX, 0, playheadX, height());
    }
}

void WaveformWidget::mousePressEvent(QMouseEvent* event) {
    isDragging_ = true;
    double seconds = (event->pos().x() * samplesPerPixel_ + sampleOffset_) / sampleRate_;
    selectionStart_ = seconds;
    selectionEnd_ = seconds;
    emit positionClicked(seconds);
    update();
}

void WaveformWidget::mouseMoveEvent(QMouseEvent* event) {
    if (isDragging_) {
        double seconds = (event->pos().x() * samplesPerPixel_ + sampleOffset_) / sampleRate_;
        selectionEnd_ = seconds;
        update();
    }
}

void WaveformWidget::mouseReleaseEvent(QMouseEvent*) {
    if (isDragging_) {
        isDragging_ = false;
        if (selectionEnd_ < selectionStart_) {
            std::swap(selectionStart_, selectionEnd_);
        }
        if (selectionEnd_ - selectionStart_ > 0.01) {
            emit selectionChanged(selectionStart_, selectionEnd_);
        }
    }
}

void WaveformWidget::wheelEvent(QWheelEvent* event) {
    double factor = event->angleDelta().y() > 0 ? 0.9 : 1.1;
    samplesPerPixel_ *= factor;
    samplesPerPixel_ = std::max(1.0, std::min(10000.0, samplesPerPixel_));
    generateWaveformCache();
    update();
    event->accept();
}

// ============================================================================
// SPECTRUM WIDGET
// ============================================================================

SpectrumWidget::SpectrumWidget(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(100);
}

void SpectrumWidget::setMagnitudes(const std::vector<float>& mags) {
    // Smooth the magnitudes
    if (smoothedMags_.size() != mags.size()) {
        smoothedMags_ = mags;
    } else {
        for (size_t i = 0; i < mags.size(); ++i) {
            if (mags[i] > smoothedMags_[i]) {
                smoothedMags_[i] = mags[i];  // Instant attack
            } else {
                smoothedMags_[i] = smoothedMags_[i] * 0.9f + mags[i] * 0.1f;  // Slow decay
            }
        }
    }
    magnitudes_ = mags;
    update();
}

void SpectrumWidget::setFrequencyRange(float minHz, float maxHz) {
    minFreq_ = minHz;
    maxFreq_ = maxHz;
}

void SpectrumWidget::setdBRange(float minDB, float maxDB) {
    minDB_ = minDB;
    maxDB_ = maxDB;
}

void SpectrumWidget::setStyle(int style) {
    style_ = style;
    update();
}

void SpectrumWidget::paintEvent(QPaintEvent*) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // Background
    painter.fillRect(rect(), Colors::BackgroundDark);

    if (smoothedMags_.empty()) return;

    // Draw frequency grid
    painter.setPen(Colors::GridMinor);
    std::vector<float> freqs = {100, 1000, 10000};
    for (float freq : freqs) {
        float logFreq = std::log10(freq / minFreq_) / std::log10(maxFreq_ / minFreq_);
        int x = static_cast<int>(logFreq * width());
        painter.drawLine(x, 0, x, height());

        painter.setPen(Colors::TextSecondary);
        painter.setFont(QFont(Fonts::getFontFamily(), Fonts::SizeSmall - 2));
        QString label = freq >= 1000 ? QString("%1k").arg(freq / 1000) : QString::number(freq);
        painter.drawText(x + 2, height() - 2, label);
        painter.setPen(Colors::GridMinor);
    }

    // Draw dB grid
    for (float db = minDB_; db <= maxDB_; db += 10) {
        float normalized = (db - minDB_) / (maxDB_ - minDB_);
        int y = height() - static_cast<int>(normalized * height());
        painter.drawLine(0, y, width(), y);
    }

    // Draw spectrum
    float dbRange = maxDB_ - minDB_;

    if (style_ == 0 || style_ == 1) {
        // Line or filled
        QPainterPath path;
        path.moveTo(0, height());

        for (size_t i = 0; i < smoothedMags_.size(); ++i) {
            float freq = minFreq_ * std::pow(maxFreq_ / minFreq_,
                                             static_cast<float>(i) / smoothedMags_.size());
            float logFreq = std::log10(freq / minFreq_) / std::log10(maxFreq_ / minFreq_);
            int x = static_cast<int>(logFreq * width());

            float db = 20.0f * std::log10(std::max(0.00001f, smoothedMags_[i]));
            float normalized = (db - minDB_) / dbRange;
            int y = height() - static_cast<int>(std::max(0.0f, std::min(1.0f, normalized)) * height());

            if (i == 0) {
                path.moveTo(x, y);
            } else {
                path.lineTo(x, y);
            }
        }

        if (style_ == 1) {
            // Filled
            path.lineTo(width(), height());
            path.lineTo(0, height());
            path.closeSubpath();

            QLinearGradient gradient(0, 0, 0, height());
            gradient.setColorAt(0.0, Colors::AccentBlue);
            gradient.setColorAt(1.0, Colors::AccentBlue.darker(200));
            painter.fillPath(path, gradient);
        }

        painter.setPen(QPen(Colors::AccentBlue, 1.5));
        painter.setBrush(Qt::NoBrush);
        painter.drawPath(path);

    } else if (style_ == 2) {
        // Bars
        int numBars = 32;
        int barWidth = width() / numBars - 1;

        for (int bar = 0; bar < numBars; ++bar) {
            int startBin = bar * smoothedMags_.size() / numBars;
            int endBin = (bar + 1) * smoothedMags_.size() / numBars;

            float maxMag = 0;
            for (int bin = startBin; bin < endBin; ++bin) {
                maxMag = std::max(maxMag, smoothedMags_[bin]);
            }

            float db = 20.0f * std::log10(std::max(0.00001f, maxMag));
            float normalized = (db - minDB_) / dbRange;
            int barHeight = static_cast<int>(std::max(0.0f, std::min(1.0f, normalized)) * height());

            int x = bar * (barWidth + 1);

            QLinearGradient gradient(x, height(), x, height() - barHeight);
            gradient.setColorAt(0.0, Colors::MeterGreen);
            gradient.setColorAt(0.7, Colors::MeterYellow);
            gradient.setColorAt(1.0, Colors::MeterRed);

            painter.fillRect(x, height() - barHeight, barWidth, barHeight, gradient);
        }
    }
}

} // namespace GUI
} // namespace MolinAntro

#endif // BUILD_QT6_GUI
