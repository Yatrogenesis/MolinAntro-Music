#pragma once

#include "ui/Theme.h"
#include "core/AudioBuffer.h"
#include <functional>
#include <vector>
#include <memory>
#include <string>

namespace MolinAntro {
namespace UI {

/**
 * @brief Base class for all UI components with accessibility
 */
class Component {
public:
    Component();
    virtual ~Component() = default;

    // Geometry
    struct Bounds {
        float x{0.0f};
        float y{0.0f};
        float width{100.0f};
        float height{100.0f};

        bool contains(float px, float py) const {
            return px >= x && px < (x + width) && py >= y && py < (y + height);
        }
    };

    void setBounds(const Bounds& bounds) { bounds_ = bounds; }
    const Bounds& getBounds() const { return bounds_; }

    // Visibility
    void setVisible(bool visible) { visible_ = visible; }
    bool isVisible() const { return visible_; }

    // Enable/Disable
    void setEnabled(bool enabled) { enabled_ = enabled; }
    bool isEnabled() const { return enabled_; }

    // Focus (accessibility)
    void setFocusable(bool focusable) { focusable_ = focusable; }
    bool isFocusable() const { return focusable_; }
    void setFocused(bool focused) { focused_ = focused; }
    bool isFocused() const { return focused_; }

    // Accessibility
    void setAccessibilityLabel(const std::string& label) { accessibilityLabel_ = label; }
    const std::string& getAccessibilityLabel() const { return accessibilityLabel_; }
    void setAccessibilityHint(const std::string& hint) { accessibilityHint_ = hint; }
    const std::string& getAccessibilityHint() const { return accessibilityHint_; }

    // Rendering
    virtual void paint(/* GraphicsContext& g */) = 0;
    virtual void update(float deltaTime);

    // Input
    virtual void onMouseDown(float x, float y);
    virtual void onMouseUp(float x, float y);
    virtual void onMouseMove(float x, float y);
    virtual void onMouseDrag(float x, float y);
    virtual void onMouseWheel(float delta);
    virtual void onKeyPress(int keyCode);

    // Parent/Child hierarchy
    void addChild(std::shared_ptr<Component> child);
    void removeChild(std::shared_ptr<Component> child);
    const std::vector<std::shared_ptr<Component>>& getChildren() const { return children_; }

protected:
    Bounds bounds_;
    bool visible_{true};
    bool enabled_{true};
    bool focusable_{true};
    bool focused_{false};
    bool hovered_{false};
    std::string accessibilityLabel_;
    std::string accessibilityHint_;
    std::vector<std::shared_ptr<Component>> children_;
};

/**
 * @brief Professional rotary knob control
 */
class Knob : public Component {
public:
    Knob(const std::string& label = "Knob");

    // Value management
    void setValue(float value);  ///< Set value (0-1)
    float getValue() const { return value_; }
    void setDefaultValue(float defaultValue) { defaultValue_ = defaultValue; }

    // Range
    void setRange(float min, float max);
    float getMinValue() const { return minValue_; }
    float getMaxValue() const { return maxValue_; }

    // Display
    void setLabel(const std::string& label) { label_ = label; }
    const std::string& getLabel() const { return label_; }
    void setUnit(const std::string& unit) { unit_ = unit; }
    void setValueFormatter(std::function<std::string(float)> formatter) {
        valueFormatter_ = formatter;
    }

    // Behavior
    void setSensitivity(float sensitivity) { sensitivity_ = sensitivity; }
    void setSnapToDefault(bool snap) { snapToDefault_ = snap; }
    void setBipolar(bool bipolar) { bipolar_ = bipolar; }  ///< -1 to 1 instead of 0 to 1

    // Callbacks
    using ValueChangeCallback = std::function<void(float)>;
    void onValueChange(ValueChangeCallback callback) { valueChangeCallback_ = callback; }

    void paint(/* GraphicsContext& g */) override;
    void onMouseDown(float x, float y) override;
    void onMouseDrag(float x, float y) override;
    void onMouseWheel(float delta) override;

private:
    std::string label_;
    std::string unit_;
    float value_{0.5f};
    float defaultValue_{0.5f};
    float minValue_{0.0f};
    float maxValue_{1.0f};
    float sensitivity_{0.01f};
    bool snapToDefault_{true};
    bool bipolar_{false};
    float dragStartY_{0.0f};
    float dragStartValue_{0.0f};
    std::function<std::string(float)> valueFormatter_;
    ValueChangeCallback valueChangeCallback_;
};

/**
 * @brief Professional fader/slider control
 */
class Fader : public Component {
public:
    enum class Orientation {
        Vertical,
        Horizontal
    };

    Fader(Orientation orientation = Orientation::Vertical);

    // Value
    void setValue(float value);
    float getValue() const { return value_; }
    void setRange(float min, float max);

    // Display
    void setLabel(const std::string& label) { label_ = label; }
    void setUnit(const std::string& unit) { unit_ = unit; }
    void setShowValue(bool show) { showValue_ = show; }
    void setScale(float scale) { scale_ = scale; }  ///< dB scale, linear, etc.

    // Behavior
    void setOrientation(Orientation orientation) { orientation_ = orientation; }
    void setFineControl(bool fine) { fineControl_ = fine; }

    // Callbacks
    using ValueChangeCallback = std::function<void(float)>;
    void onValueChange(ValueChangeCallback callback) { valueChangeCallback_ = callback; }

    void paint(/* GraphicsContext& g */) override;
    void onMouseDown(float x, float y) override;
    void onMouseDrag(float x, float y) override;

private:
    Orientation orientation_;
    std::string label_;
    std::string unit_;
    float value_{0.0f};
    float minValue_{-60.0f};
    float maxValue_{6.0f};
    float scale_{1.0f};
    bool showValue_{true};
    bool fineControl_{false};
    ValueChangeCallback valueChangeCallback_;
};

/**
 * @brief Professional VU/Peak meter
 */
class Meter : public Component {
public:
    enum class Type {
        VU,         ///< VU meter (slow ballistics)
        Peak,       ///< Peak meter (fast attack)
        RMS,        ///< RMS level
        LUFS        ///< LUFS loudness
    };

    Meter(Type type = Type::Peak);

    // Level
    void setLevel(float level);  ///< Set level in dB
    float getLevel() const { return currentLevel_; }
    void setPeak(float peak);
    void resetPeak();

    // Configuration
    void setType(Type type) { type_ = type; }
    void setRange(float min, float max);  ///< dB range
    void setChannels(int channels);
    void setStereo(bool stereo) { stereo_ = stereo; }

    // Ballistics
    void setAttackTime(float ms);
    void setReleaseTime(float ms);

    // Appearance
    void setOrientation(Fader::Orientation orientation) { orientation_ = orientation; }
    void setSegmented(bool segmented) { segmented_ = segmented; }
    void setShowPeak(bool show) { showPeak_ = show; }
    void setShowValue(bool show) { showValue_ = show; }

    void paint(/* GraphicsContext& g */) override;
    void update(float deltaTime) override;

private:
    Type type_;
    Fader::Orientation orientation_;
    float currentLevel_{-60.0f};
    float peakLevel_{-60.0f};
    float targetLevel_{-60.0f};
    float minLevel_{-60.0f};
    float maxLevel_{6.0f};
    float attackCoeff_{0.99f};
    float releaseCoeff_{0.999f};
    int channels_{1};
    bool stereo_{false};
    bool segmented_{true};
    bool showPeak_{true};
    bool showValue_{true};

    Color getLevelColor(float level) const;
};

/**
 * @brief Professional waveform display
 */
class WaveformDisplay : public Component {
public:
    WaveformDisplay();

    // Audio data
    void setAudioBuffer(const Core::AudioBuffer& buffer);
    void setChannel(int channel) { channel_ = channel; }

    // View
    void setZoom(float zoom);        ///< Horizontal zoom
    void setOffset(float offset);    ///< Horizontal scroll offset
    void setVerticalZoom(float zoom);

    // Appearance
    void setWaveformColor(const Color& color) { waveformColor_ = color; }
    void setBackgroundColor(const Color& color) { backgroundColor_ = color; }
    void setShowGrid(bool show) { showGrid_ = show; }
    void setShowRMS(bool show) { showRMS_ = show; }

    // Interaction
    using SelectionCallback = std::function<void(float start, float end)>;
    void onSelection(SelectionCallback callback) { selectionCallback_ = callback; }

    void paint(/* GraphicsContext& g */) override;
    void onMouseDown(float x, float y) override;
    void onMouseDrag(float x, float y) override;
    void onMouseWheel(float delta) override;

private:
    std::vector<float> waveformData_;
    std::vector<float> rmsData_;
    int channel_{0};
    float zoom_{1.0f};
    float offset_{0.0f};
    float verticalZoom_{1.0f};
    float selectionStart_{0.0f};
    float selectionEnd_{0.0f};
    Color waveformColor_;
    Color backgroundColor_;
    bool showGrid_{true};
    bool showRMS_{false};
    SelectionCallback selectionCallback_;

    void generateWaveformData(const Core::AudioBuffer& buffer);
    void generateRMSData(const Core::AudioBuffer& buffer);
};

/**
 * @brief Professional spectrum analyzer
 */
class SpectrumAnalyzer : public Component {
public:
    SpectrumAnalyzer();

    // Analysis
    void analyzeAudio(const Core::AudioBuffer& buffer);
    void setFFTSize(int size);
    void setChannel(int channel) { channel_ = channel; }

    // Display
    void setFrequencyRange(float min, float max);
    void setdBRange(float min, float max);
    void setShowGrid(bool show) { showGrid_ = show; }
    void setShowPeaks(bool show) { showPeaks_ = show; }
    void setFrozen(bool frozen) { frozen_ = frozen; }

    // Appearance
    void setSpectrumColor(const Color& color) { spectrumColor_ = color; }
    void setStyle(int style) { style_ = style; }  ///< 0=line, 1=filled, 2=bars

    void paint(/* GraphicsContext& g */) override;
    void update(float deltaTime) override;

private:
    std::vector<float> magnitudes_;
    std::vector<float> smoothedMagnitudes_;
    std::vector<float> peakMagnitudes_;
    int fftSize_{2048};
    int channel_{0};
    float minFreq_{20.0f};
    float maxFreq_{20000.0f};
    float mindB_{-90.0f};
    float maxdB_{0.0f};
    bool showGrid_{true};
    bool showPeaks_{false};
    bool frozen_{false};
    int style_{1};
    Color spectrumColor_;

    void performFFT(const Core::AudioBuffer& buffer);
    void smoothSpectrum(float smoothing = 0.8f);
};

/**
 * @brief Transport control panel
 */
class TransportControls : public Component {
public:
    TransportControls();

    enum class State {
        Stopped,
        Playing,
        Recording,
        Paused
    };

    void setState(State state);
    State getState() const { return state_; }

    // Callbacks
    using StateChangeCallback = std::function<void(State)>;
    void onStateChange(StateChangeCallback callback) { stateCallback_ = callback; }

    void paint(/* GraphicsContext& g */) override;
    void onMouseDown(float x, float y) override;

private:
    State state_{State::Stopped};
    StateChangeCallback stateCallback_;

    struct Button {
        Bounds bounds;
        std::string icon;
        bool hovered{false};
    };

    std::vector<Button> buttons_;
};

/**
 * @brief Professional button with multiple styles
 */
class Button : public Component {
public:
    enum class Style {
        Normal,
        Toggle,
        Radio,
        Icon
    };

    Button(const std::string& text = "", Style style = Style::Normal);

    // Content
    void setText(const std::string& text) { text_ = text; }
    const std::string& getText() const { return text_; }
    void setIcon(const std::string& icon) { icon_ = icon; }

    // State
    void setToggled(bool toggled) { toggled_ = toggled; }
    bool isToggled() const { return toggled_; }

    // Appearance
    void setStyle(Style style) { style_ = style; }
    void setCornerRadius(float radius) { cornerRadius_ = radius; }

    // Callbacks
    using ClickCallback = std::function<void()>;
    void onClick(ClickCallback callback) { clickCallback_ = callback; }

    void paint(/* GraphicsContext& g */) override;
    void onMouseDown(float x, float y) override;
    void onMouseUp(float x, float y) override;

private:
    std::string text_;
    std::string icon_;
    Style style_;
    bool toggled_{false};
    float cornerRadius_{4.0f};
    ClickCallback clickCallback_;
};

/**
 * @brief Professional combobox/dropdown
 */
class ComboBox : public Component {
public:
    ComboBox();

    // Items
    void addItem(const std::string& text, int id = -1);
    void clear();
    void setSelectedId(int id);
    int getSelectedId() const { return selectedId_; }
    std::string getSelectedText() const;

    // Callbacks
    using SelectionCallback = std::function<void(int id, const std::string& text)>;
    void onSelectionChange(SelectionCallback callback) { selectionCallback_ = callback; }

    void paint(/* GraphicsContext& g */) override;
    void onMouseDown(float x, float y) override;

private:
    struct Item {
        std::string text;
        int id;
    };

    std::vector<Item> items_;
    int selectedId_{-1};
    bool dropdownOpen_{false};
    SelectionCallback selectionCallback_;
};

/**
 * @brief Label with rich text support
 */
class Label : public Component {
public:
    Label(const std::string& text = "");

    void setText(const std::string& text) { text_ = text; }
    const std::string& getText() const { return text_; }

    enum class Alignment {
        Left,
        Center,
        Right
    };

    void setAlignment(Alignment alignment) { alignment_ = alignment; }
    void setFontSize(float size) { fontSize_ = size; }
    void setColor(const Color& color) { color_ = color; }
    void setBold(bool bold) { bold_ = bold; }

    void paint(/* GraphicsContext& g */) override;

private:
    std::string text_;
    Alignment alignment_{Alignment::Left};
    float fontSize_{14.0f};
    Color color_;
    bool bold_{false};
};

} // namespace UI
} // namespace MolinAntro
