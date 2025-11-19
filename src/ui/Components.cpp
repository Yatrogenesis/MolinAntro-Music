// Components.cpp - Professional UI Components Implementation
// MolinAntro DAW ACME Edition v3.0.0

#include "ui/Components.h"
#include "ui/Theme.h"
#include <cmath>
#include <algorithm>

namespace MolinAntro {
namespace UI {

// ============================================================================
// Component Base Implementation
// ============================================================================

Component::Component() {
}

void Component::update(float /*deltaTime*/) {
    // Override in subclasses
}

void Component::onMouseDown(float /*x*/, float /*y*/) {}
void Component::onMouseUp(float /*x*/, float /*y*/) {}
void Component::onMouseMove(float /*x*/, float /*y*/) {}
void Component::onMouseDrag(float /*x*/, float /*y*/) {}
void Component::onMouseWheel(float /*delta*/) {}
void Component::onKeyPress(int /*keyCode*/) {}

void Component::addChild(std::shared_ptr<Component> child) {
    children_.push_back(child);
}

void Component::removeChild(std::shared_ptr<Component> child) {
    children_.erase(
        std::remove(children_.begin(), children_.end(), child),
        children_.end()
    );
}

// ============================================================================
// Knob Implementation
// ============================================================================

Knob::Knob(const std::string& label)
    : label_(label) {
    setAccessibilityLabel(label);
    setAccessibilityHint("Rotary control. Drag to adjust value");
}

void Knob::setValue(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    if (value_ != value) {
        value_ = value;
        if (valueChangeCallback_) {
            valueChangeCallback_(value);
        }
    }
}

void Knob::setRange(float min, float max) {
    minValue_ = min;
    maxValue_ = max;
}

void Knob::paint(/* GraphicsContext& g */) {
    // Get theme
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    // Calculate knob position
    float centerX = bounds_.x + bounds_.width * 0.5f;
    float centerY = bounds_.y + bounds_.height * 0.5f;
    float radius = std::min(bounds_.width, bounds_.height) * 0.4f;

    // Draw background circle
    // g.setColor(theme->colors().surface);
    // g.fillEllipse(centerX, centerY, radius);

    // Draw value arc
    float startAngle = 0.75f * 3.14159f;  // 135 degrees
    float endAngle = 2.25f * 3.14159f;    // 405 degrees
    float valueAngle = startAngle + (endAngle - startAngle) * value_;

    // g.setColor(theme->colors().primary);
    // g.drawArc(centerX, centerY, radius, startAngle, valueAngle, 3.0f);

    // Draw indicator line
    float indicatorX = centerX + std::cos(valueAngle) * radius * 0.7f;
    float indicatorY = centerY + std::sin(valueAngle) * radius * 0.7f;
    // g.setColor(theme->colors().textPrimary);
    // g.drawLine(centerX, centerY, indicatorX, indicatorY, 2.0f);

    // Draw label
    // g.setFont(theme->typography().fontFamily, theme->typography().fontSize);
    // g.setColor(theme->colors().textSecondary);
    // g.drawText(label_, bounds_.x, bounds_.y + bounds_.height + 5, bounds_.width, 20);

    // Draw value
    if (valueFormatter_) {
        std::string valueText = valueFormatter_(value_);
        // g.drawText(valueText, bounds_.x, bounds_.y - 20, bounds_.width, 20);
    }
}

void Knob::onMouseDown(float x, float y) {
    dragStartY_ = y;
    dragStartValue_ = value_;
    focused_ = true;
}

void Knob::onMouseDrag(float /*x*/, float y) {
    float delta = (dragStartY_ - y) * sensitivity_;
    float newValue = std::clamp(dragStartValue_ + delta, 0.0f, 1.0f);

    // Snap to default on middle click or with modifier
    if (snapToDefault_ && std::abs(newValue - defaultValue_) < 0.02f) {
        newValue = defaultValue_;
    }

    setValue(newValue);
}

void Knob::onMouseWheel(float delta) {
    float newValue = value_ + delta * 0.01f;
    setValue(newValue);
}

// ============================================================================
// Fader Implementation
// ============================================================================

Fader::Fader(Orientation orientation)
    : orientation_(orientation) {
    setAccessibilityLabel("Fader");
    setAccessibilityHint("Linear fader control. Drag to adjust value");
}

void Fader::setValue(float value) {
    value = std::clamp(value, minValue_, maxValue_);
    if (value_ != value) {
        value_ = value;
        if (valueChangeCallback_) {
            valueChangeCallback_(value);
        }
    }
}

void Fader::setRange(float min, float max) {
    minValue_ = min;
    maxValue_ = max;
}

void Fader::paint(/* GraphicsContext& g */) {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    // Calculate fader position
    float normalizedValue = (value_ - minValue_) / (maxValue_ - minValue_);

    if (orientation_ == Orientation::Vertical) {
        float trackX = bounds_.x + bounds_.width * 0.5f - 2.0f;
        float trackHeight = bounds_.height - 20.0f;
        float handleY = bounds_.y + trackHeight * (1.0f - normalizedValue);

        // Draw track
        // g.setColor(theme->colors().border);
        // g.fillRect(trackX, bounds_.y, 4, trackHeight);

        // Draw filled portion
        // g.setColor(theme->colors().primary);
        // g.fillRect(trackX, handleY, 4, trackHeight * normalizedValue);

        // Draw handle
        // g.setColor(theme->colors().textPrimary);
        // g.fillRect(trackX - 6, handleY - 5, 16, 10);
    } else {
        float trackY = bounds_.y + bounds_.height * 0.5f - 2.0f;
        float trackWidth = bounds_.width - 20.0f;
        float handleX = bounds_.x + trackWidth * normalizedValue;

        // Draw track and handle
        // Similar to vertical but horizontal
    }

    // Draw value text
    if (showValue_) {
        char valueText[32];
        std::snprintf(valueText, sizeof(valueText), "%.1f %s",
                     value_, unit_.c_str());
        // g.drawText(valueText, ...);
    }
}

void Fader::onMouseDown(float x, float y) {
    // Calculate value from click position
    if (orientation_ == Orientation::Vertical) {
        float normalizedY = (y - bounds_.y) / bounds_.height;
        setValue(minValue_ + (1.0f - normalizedY) * (maxValue_ - minValue_));
    } else {
        float normalizedX = (x - bounds_.x) / bounds_.width;
        setValue(minValue_ + normalizedX * (maxValue_ - minValue_));
    }
}

void Fader::onMouseDrag(float x, float y) {
    onMouseDown(x, y);  // Same behavior
}

// ============================================================================
// Meter Implementation
// ============================================================================

Meter::Meter(Type type)
    : type_(type)
    , orientation_(Fader::Orientation::Vertical) {
    setAccessibilityLabel("Level Meter");
    setAccessibilityHint("Audio level indicator");
}

void Meter::setLevel(float level) {
    targetLevel_ = level;
}

void Meter::setPeak(float peak) {
    peakLevel_ = std::max(peakLevel_, peak);
}

void Meter::resetPeak() {
    peakLevel_ = minLevel_;
}

void Meter::setRange(float min, float max) {
    minLevel_ = min;
    maxLevel_ = max;
}

void Meter::setChannels(int channels) {
    channels_ = channels;
}

void Meter::setAttackTime(float ms) {
    attackCoeff_ = std::exp(-1.0f / (ms * 0.001f * 60.0f));
}

void Meter::setReleaseTime(float ms) {
    releaseCoeff_ = std::exp(-1.0f / (ms * 0.001f * 60.0f));
}

Color Meter::getLevelColor(float level) const {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return Color(0.5f, 0.5f, 0.5f);

    // Color zones: green < -18dB, yellow < -6dB, red >= -6dB
    if (level < -18.0f) {
        return theme->colors().meterGreen;
    } else if (level < -6.0f) {
        // Interpolate between green and yellow
        float t = (level + 18.0f) / 12.0f;
        return theme->colors().meterGreen.interpolate(
            theme->colors().meterYellow, t);
    } else {
        // Interpolate between yellow and red
        float t = std::min((level + 6.0f) / 6.0f, 1.0f);
        return theme->colors().meterYellow.interpolate(
            theme->colors().meterRed, t);
    }
}

void Meter::update(float deltaTime) {
    // Ballistics simulation
    if (currentLevel_ < targetLevel_) {
        // Fast attack
        float coeff = attackCoeff_;
        currentLevel_ = targetLevel_ + (currentLevel_ - targetLevel_) * coeff;
    } else {
        // Slow release
        float coeff = releaseCoeff_;
        currentLevel_ = targetLevel_ + (currentLevel_ - targetLevel_) * coeff;
    }

    // Peak hold with decay
    if (peakLevel_ > currentLevel_) {
        peakLevel_ -= deltaTime * 10.0f;  // 10 dB/sec decay
    }
}

void Meter::paint(/* GraphicsContext& g */) {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    float normalizedLevel = (currentLevel_ - minLevel_) / (maxLevel_ - minLevel_);
    normalizedLevel = std::clamp(normalizedLevel, 0.0f, 1.0f);

    if (orientation_ == Fader::Orientation::Vertical) {
        float meterHeight = bounds_.height * normalizedLevel;

        // Draw segments if segmented mode
        if (segmented_) {
            int numSegments = 20;
            float segmentHeight = bounds_.height / numSegments;
            float gap = 1.0f;

            for (int i = 0; i < numSegments; ++i) {
                float segmentY = bounds_.y + bounds_.height - (i + 1) * segmentHeight;
                float segmentLevel = minLevel_ + (maxLevel_ - minLevel_) *
                                   (float)i / numSegments;

                if (segmentLevel <= currentLevel_) {
                    Color color = getLevelColor(segmentLevel);
                    // g.setColor(color);
                    // g.fillRect(bounds_.x, segmentY, bounds_.width, segmentHeight - gap);
                }
            }
        } else {
            // Continuous gradient
            // g.setColor(getLevelColor(currentLevel_));
            // g.fillRect(bounds_.x, bounds_.y + bounds_.height - meterHeight,
            //           bounds_.width, meterHeight);
        }

        // Draw peak indicator
        if (showPeak_ && peakLevel_ > minLevel_) {
            float peakY = bounds_.y + bounds_.height *
                         (1.0f - (peakLevel_ - minLevel_) / (maxLevel_ - minLevel_));
            // g.setColor(Color(1.0f, 1.0f, 1.0f));
            // g.fillRect(bounds_.x, peakY, bounds_.width, 2);
        }
    }
}

// ============================================================================
// WaveformDisplay Implementation
// ============================================================================

WaveformDisplay::WaveformDisplay() {
    setAccessibilityLabel("Waveform Display");
    setAccessibilityHint("Audio waveform visualization. Scroll to zoom, drag to select");

    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (theme) {
        waveformColor_ = theme->colors().waveform;
        backgroundColor_ = theme->colors().background;
    }
}

void WaveformDisplay::setAudioBuffer(const Core::AudioBuffer& buffer) {
    generateWaveformData(buffer);
    if (showRMS_) {
        generateRMSData(buffer);
    }
}

void WaveformDisplay::setZoom(float zoom) {
    zoom_ = std::clamp(zoom, 0.1f, 100.0f);
}

void WaveformDisplay::setOffset(float offset) {
    offset_ = std::clamp(offset, 0.0f, 1.0f);
}

void WaveformDisplay::setVerticalZoom(float zoom) {
    verticalZoom_ = std::clamp(zoom, 0.1f, 10.0f);
}

void WaveformDisplay::generateWaveformData(const Core::AudioBuffer& buffer) {
    int numSamples = buffer.getNumSamples();
    if (numSamples == 0) return;

    // Downsample for display (e.g., max 2000 points)
    int displayWidth = std::min(2000, numSamples);
    waveformData_.resize(displayWidth * 2);  // min/max pairs

    const float* samples = buffer.getReadPointer(channel_);
    int samplesPerPixel = std::max(1, numSamples / displayWidth);

    for (int i = 0; i < displayWidth; ++i) {
        float minVal = 1.0f;
        float maxVal = -1.0f;

        for (int j = 0; j < samplesPerPixel; ++j) {
            int sampleIndex = i * samplesPerPixel + j;
            if (sampleIndex < numSamples) {
                float sample = samples[sampleIndex];
                minVal = std::min(minVal, sample);
                maxVal = std::max(maxVal, sample);
            }
        }

        waveformData_[i * 2] = minVal;
        waveformData_[i * 2 + 1] = maxVal;
    }
}

void WaveformDisplay::generateRMSData(const Core::AudioBuffer& buffer) {
    // Calculate RMS in windows
    int windowSize = 1024;
    int numWindows = buffer.getNumSamples() / windowSize;
    rmsData_.resize(numWindows);

    const float* samples = buffer.getReadPointer(channel_);

    for (int i = 0; i < numWindows; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < windowSize; ++j) {
            float sample = samples[i * windowSize + j];
            sum += sample * sample;
        }
        rmsData_[i] = std::sqrt(sum / windowSize);
    }
}

void WaveformDisplay::paint(/* GraphicsContext& g */) {
    // Draw background
    // g.setColor(backgroundColor_);
    // g.fillRect(bounds_.x, bounds_.y, bounds_.width, bounds_.height);

    // Draw grid if enabled
    if (showGrid_) {
        // Draw time grid, amplitude grid, etc.
    }

    // Draw waveform
    float centerY = bounds_.y + bounds_.height * 0.5f;
    float amplitude = bounds_.height * 0.5f * verticalZoom_;

    // g.setColor(waveformColor_);
    for (size_t i = 0; i < waveformData_.size() / 2; ++i) {
        float x = bounds_.x + (float)i / waveformData_.size() * 2 * bounds_.width * zoom_;
        float yMin = centerY - waveformData_[i * 2] * amplitude;
        float yMax = centerY - waveformData_[i * 2 + 1] * amplitude;

        // g.drawLine(x, yMin, x, yMax);
    }

    // Draw selection
    if (selectionStart_ != selectionEnd_) {
        float x1 = bounds_.x + selectionStart_ * bounds_.width;
        float x2 = bounds_.x + selectionEnd_ * bounds_.width;
        // g.setColor(Color(0.3f, 0.6f, 0.9f, 0.3f));
        // g.fillRect(x1, bounds_.y, x2 - x1, bounds_.height);
    }
}

void WaveformDisplay::onMouseDown(float x, float /*y*/) {
    selectionStart_ = (x - bounds_.x) / bounds_.width;
    selectionEnd_ = selectionStart_;
}

void WaveformDisplay::onMouseDrag(float x, float /*y*/) {
    selectionEnd_ = (x - bounds_.x) / bounds_.width;
    if (selectionCallback_) {
        selectionCallback_(selectionStart_, selectionEnd_);
    }
}

void WaveformDisplay::onMouseWheel(float delta) {
    setZoom(zoom_ * (1.0f + delta * 0.1f));
}

// ============================================================================
// SpectrumAnalyzer Implementation
// ============================================================================

SpectrumAnalyzer::SpectrumAnalyzer() {
    setAccessibilityLabel("Spectrum Analyzer");
    setAccessibilityHint("Frequency spectrum visualization");

    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (theme) {
        spectrumColor_ = theme->colors().spectrum;
    }

    magnitudes_.resize(fftSize_ / 2);
    smoothedMagnitudes_.resize(fftSize_ / 2);
    peakMagnitudes_.resize(fftSize_ / 2);
}

void SpectrumAnalyzer::analyzeAudio(const Core::AudioBuffer& buffer) {
    if (!frozen_) {
        performFFT(buffer);
        smoothSpectrum();
    }
}

void SpectrumAnalyzer::setFFTSize(int size) {
    fftSize_ = size;
    magnitudes_.resize(size / 2);
    smoothedMagnitudes_.resize(size / 2);
    peakMagnitudes_.resize(size / 2);
}

void SpectrumAnalyzer::setFrequencyRange(float min, float max) {
    minFreq_ = min;
    maxFreq_ = max;
}

void SpectrumAnalyzer::setdBRange(float min, float max) {
    mindB_ = min;
    maxdB_ = max;
}

void SpectrumAnalyzer::performFFT(const Core::AudioBuffer& buffer) {
    // Simplified FFT - in production, use FFTW or similar
    // For now, just placeholder magnitudes
    const float* samples = buffer.getReadPointer(channel_);
    int numSamples = std::min(fftSize_, buffer.getNumSamples());

    for (int i = 0; i < fftSize_ / 2; ++i) {
        // Simplified magnitude calculation
        if (i < numSamples) {
            magnitudes_[i] = std::abs(samples[i]);
        }
    }
}

void SpectrumAnalyzer::smoothSpectrum(float smoothing) {
    for (size_t i = 0; i < magnitudes_.size(); ++i) {
        smoothedMagnitudes_[i] = smoothing * smoothedMagnitudes_[i] +
                                (1.0f - smoothing) * magnitudes_[i];

        if (showPeaks_) {
            peakMagnitudes_[i] = std::max(peakMagnitudes_[i], smoothedMagnitudes_[i]);
        }
    }
}

void SpectrumAnalyzer::update(float deltaTime) {
    // Decay peaks
    if (showPeaks_) {
        for (auto& peak : peakMagnitudes_) {
            peak *= 0.99f;  // Slow decay
        }
    }
}

void SpectrumAnalyzer::paint(/* GraphicsContext& g */) {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    // Draw background
    // g.setColor(theme->colors().background);
    // g.fillRect(bounds_.x, bounds_.y, bounds_.width, bounds_.height);

    // Draw grid if enabled
    if (showGrid_) {
        // Draw frequency grid (20Hz, 100Hz, 1kHz, 10kHz, etc.)
    }

    // Draw spectrum
    // g.setColor(spectrumColor_);

    for (size_t i = 0; i < smoothedMagnitudes_.size(); ++i) {
        float mag = smoothedMagnitudes_[i];
        float dB = 20.0f * std::log10(mag + 1e-10f);

        // Normalize to display range
        float normalized = (dB - mindB_) / (maxdB_ - mindB_);
        normalized = std::clamp(normalized, 0.0f, 1.0f);

        float x = bounds_.x + (float)i / smoothedMagnitudes_.size() * bounds_.width;
        float height = normalized * bounds_.height;
        float y = bounds_.y + bounds_.height - height;

        // Draw bar or line based on style
        if (style_ == 2) {  // Bars
            // g.fillRect(x, y, bounds_.width / smoothedMagnitudes_.size(), height);
        } else if (style_ == 1) {  // Filled
            // g.fillRect(x, y, 1, height);
        } else {  // Line
            // g.drawLine(x, y, x, bounds_.y + bounds_.height);
        }
    }
}

// ============================================================================
// Button Implementation
// ============================================================================

Button::Button(const std::string& text, Style style)
    : text_(text), style_(style) {
    setAccessibilityLabel(text);
    setAccessibilityHint("Button");
}

void Button::paint(/* GraphicsContext& g */) {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    // Determine colors based on state
    Color bgColor = theme->colors().surface;
    Color textColor = theme->colors().textPrimary;

    if (!enabled_) {
        bgColor = bgColor.darker(0.2f);
        textColor = theme->colors().textDisabled;
    } else if (hovered_) {
        bgColor = theme->colors().hover;
    }

    if (toggled_ && style_ == Style::Toggle) {
        bgColor = theme->colors().primary;
        textColor = theme->colors().textOnPrimary;
    }

    // Draw background with rounded corners
    // g.setColor(bgColor);
    // g.fillRoundedRect(bounds_.x, bounds_.y, bounds_.width, bounds_.height, cornerRadius_);

    // Draw border
    // g.setColor(theme->colors().border);
    // g.drawRoundedRect(bounds_.x, bounds_.y, bounds_.width, bounds_.height, cornerRadius_);

    // Draw text or icon
    // g.setColor(textColor);
    if (!icon_.empty()) {
        // Draw icon
    } else {
        // g.drawText(text_, bounds_.x, bounds_.y, bounds_.width, bounds_.height);
    }
}

void Button::onMouseDown(float /*x*/, float /*y*/) {
    if (!enabled_) return;

    if (style_ == Style::Toggle) {
        toggled_ = !toggled_;
    }
}

void Button::onMouseUp(float /*x*/, float /*y*/) {
    if (!enabled_) return;

    if (clickCallback_) {
        clickCallback_();
    }
}

// ============================================================================
// ComboBox Implementation
// ============================================================================

ComboBox::ComboBox() {
    setAccessibilityLabel("Combo Box");
    setAccessibilityHint("Dropdown menu. Click to show options");
}

void ComboBox::addItem(const std::string& text, int id) {
    if (id == -1) {
        id = static_cast<int>(items_.size());
    }
    items_.push_back({text, id});
}

void ComboBox::clear() {
    items_.clear();
    selectedId_ = -1;
}

void ComboBox::setSelectedId(int id) {
    selectedId_ = id;
    if (selectionCallback_) {
        selectionCallback_(id, getSelectedText());
    }
}

std::string ComboBox::getSelectedText() const {
    for (const auto& item : items_) {
        if (item.id == selectedId_) {
            return item.text;
        }
    }
    return "";
}

void ComboBox::paint(/* GraphicsContext& g */) {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    // Draw main box
    // g.setColor(theme->colors().surface);
    // g.fillRect(bounds_.x, bounds_.y, bounds_.width, bounds_.height);
    // g.setColor(theme->colors().border);
    // g.drawRect(bounds_.x, bounds_.y, bounds_.width, bounds_.height);

    // Draw selected text
    // g.setColor(theme->colors().textPrimary);
    // g.drawText(getSelectedText(), bounds_.x + 5, bounds_.y, bounds_.width - 25, bounds_.height);

    // Draw dropdown arrow
    // g.drawText("▼", bounds_.x + bounds_.width - 20, bounds_.y, 20, bounds_.height);

    // Draw dropdown menu if open
    if (dropdownOpen_) {
        float itemHeight = 25.0f;
        float menuHeight = items_.size() * itemHeight;

        // g.setColor(theme->colors().surface);
        // g.fillRect(bounds_.x, bounds_.y + bounds_.height, bounds_.width, menuHeight);

        for (size_t i = 0; i < items_.size(); ++i) {
            float itemY = bounds_.y + bounds_.height + i * itemHeight;

            // g.setColor(items_[i].id == selectedId_ ?
            //           theme->colors().primary : theme->colors().surface);
            // g.fillRect(bounds_.x, itemY, bounds_.width, itemHeight);

            // g.setColor(items_[i].id == selectedId_ ?
            //           theme->colors().textOnPrimary : theme->colors().textPrimary);
            // g.drawText(items_[i].text, bounds_.x + 5, itemY, bounds_.width - 10, itemHeight);
        }
    }
}

void ComboBox::onMouseDown(float /*x*/, float y) {
    dropdownOpen_ = !dropdownOpen_;

    if (dropdownOpen_) {
        // Check if clicked on an item
        float itemHeight = 25.0f;
        int itemIndex = static_cast<int>((y - bounds_.y - bounds_.height) / itemHeight);
        if (itemIndex >= 0 && itemIndex < static_cast<int>(items_.size())) {
            setSelectedId(items_[itemIndex].id);
            dropdownOpen_ = false;
        }
    }
}

// ============================================================================
// Label Implementation
// ============================================================================

Label::Label(const std::string& text) : text_(text) {
    setAccessibilityLabel("Label");
    setFocusable(false);

    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (theme) {
        color_ = theme->colors().textPrimary;
        fontSize_ = theme->typography().fontSize;
    }
}

void Label::paint(/* GraphicsContext& g */) {
    // g.setColor(color_);
    // g.setFont(fontSize_, bold_ ? "Bold" : "Regular");

    float textX = bounds_.x;
    if (alignment_ == Alignment::Center) {
        // Center align
        textX = bounds_.x + bounds_.width * 0.5f;
    } else if (alignment_ == Alignment::Right) {
        // Right align
        textX = bounds_.x + bounds_.width;
    }

    // g.drawText(text_, textX, bounds_.y, alignment_);
}

// ============================================================================
// TransportControls Implementation
// ============================================================================

TransportControls::TransportControls() {
    setAccessibilityLabel("Transport Controls");

    // Initialize buttons
    buttons_.resize(5);

    // Play, Pause, Stop, Record, Loop
    buttons_[0].icon = "▶";  // Play
    buttons_[1].icon = "⏸";  // Pause
    buttons_[2].icon = "⏹";  // Stop
    buttons_[3].icon = "⏺";  // Record
    buttons_[4].icon = "⟲";  // Loop
}

void TransportControls::setState(State state) {
    if (state_ != state) {
        state_ = state;
        if (stateCallback_) {
            stateCallback_(state);
        }
    }
}

void TransportControls::paint(/* GraphicsContext& g */) {
    auto* theme = ThemeManager::getInstance().getCurrentTheme();
    if (!theme) return;

    float buttonWidth = 40.0f;
    float spacing = 5.0f;

    for (size_t i = 0; i < buttons_.size(); ++i) {
        buttons_[i].bounds.x = bounds_.x + i * (buttonWidth + spacing);
        buttons_[i].bounds.y = bounds_.y;
        buttons_[i].bounds.width = buttonWidth;
        buttons_[i].bounds.height = bounds_.height;

        // Highlight active button
        Color bgColor = theme->colors().surface;
        if (i == 0 && state_ == State::Playing) {
            bgColor = theme->colors().success;
        } else if (i == 3 && state_ == State::Recording) {
            bgColor = theme->colors().error;
        }

        // g.setColor(bgColor);
        // g.fillRect(buttons_[i].bounds.x, buttons_[i].bounds.y,
        //           buttons_[i].bounds.width, buttons_[i].bounds.height);

        // g.setColor(theme->colors().textPrimary);
        // g.drawText(buttons_[i].icon, ...);
    }
}

void TransportControls::onMouseDown(float x, float /*y*/) {
    for (size_t i = 0; i < buttons_.size(); ++i) {
        if (buttons_[i].bounds.contains(x, bounds_.y + bounds_.height / 2)) {
            switch (i) {
                case 0: setState(State::Playing); break;
                case 1: setState(State::Paused); break;
                case 2: setState(State::Stopped); break;
                case 3: setState(State::Recording); break;
                case 4: /* Toggle loop */ break;
            }
            break;
        }
    }
}

} // namespace UI
} // namespace MolinAntro
