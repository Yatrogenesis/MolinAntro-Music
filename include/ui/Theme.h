#pragma once

#include <string>
#include <map>
#include <memory>
#include <functional>
#include <vector>

namespace MolinAntro {
namespace UI {

/**
 * @brief Color representation with accessibility support
 */
struct Color {
    float r{0.0f};  ///< Red (0-1)
    float g{0.0f};  ///< Green (0-1)
    float b{0.0f};  ///< Blue (0-1)
    float a{1.0f};  ///< Alpha (0-1)

    Color() = default;
    Color(float red, float green, float blue, float alpha = 1.0f)
        : r(red), g(green), b(blue), a(alpha) {}

    /// RGB constructor (0-255)
    static Color fromRGB(int red, int green, int blue, int alpha = 255);

    /// Hex color constructor
    static Color fromHex(const std::string& hex);

    /// Convert to accessible color for colorblind users
    Color toAccessible(const std::string& colorblindType) const;

    /// Get contrast ratio with another color (WCAG 2.1)
    float contrastRatio(const Color& other) const;

    /// Lighten/darken
    Color lighter(float amount = 0.2f) const;
    Color darker(float amount = 0.2f) const;

    /// Interpolate between colors
    Color interpolate(const Color& other, float t) const;
};

/**
 * @brief Typography settings with dyslexia support
 */
struct Typography {
    std::string fontFamily{"Inter"};           ///< Font family
    float fontSize{14.0f};                     ///< Base font size (pt)
    float lineHeight{1.5f};                    ///< Line height multiplier
    float letterSpacing{0.0f};                 ///< Letter spacing (em)
    bool dyslexiaFriendly{false};             ///< Use dyslexia-friendly font

    // Font weights
    enum class Weight {
        Thin = 100,
        Light = 300,
        Regular = 400,
        Medium = 500,
        SemiBold = 600,
        Bold = 700,
        ExtraBold = 800
    };

    // Dyslexia-friendly fonts: OpenDyslexic, Comic Sans, Arial
    std::string getDyslexiaFont() const;
    float getDyslexiaSpacing() const;  ///< Returns optimal spacing for dyslexia
};

/**
 * @brief Spacing and sizing system
 */
struct Spacing {
    float xs{4.0f};
    float sm{8.0f};
    float md{16.0f};
    float lg{24.0f};
    float xl{32.0f};
    float xxl{48.0f};
};

/**
 * @brief Animation timing and easing
 */
struct Animation {
    enum class Easing {
        Linear,
        EaseIn,
        EaseOut,
        EaseInOut,
        Bounce,
        Elastic,
        Back
    };

    float duration{0.2f};           ///< Animation duration (seconds)
    Easing easing{Easing::EaseOut}; ///< Easing function
    bool useGPU{true};              ///< Use GPU acceleration

    // Preset animations
    static Animation fast();
    static Animation normal();
    static Animation slow();
    static Animation smooth();
};

/**
 * @brief Accessibility settings
 */
struct Accessibility {
    enum class ColorblindMode {
        None,
        Protanopia,    ///< Red-blind
        Deuteranopia,  ///< Green-blind
        Tritanopia,    ///< Blue-blind
        Achromatopsia  ///< Total colorblindness
    };

    bool highContrast{false};
    bool reducedMotion{false};
    bool screenReaderMode{false};
    bool largeText{false};
    bool dyslexiaMode{false};
    ColorblindMode colorblindMode{ColorblindMode::None};
    float textScale{1.0f};  ///< Text scaling multiplier

    // WCAG 2.1 compliance level
    enum class WCAGLevel {
        A,
        AA,
        AAA
    };
    WCAGLevel targetLevel{WCAGLevel::AAA};
};

/**
 * @brief Complete theme definition
 */
class Theme {
public:
    enum class Mode {
        Light,
        Dark,
        Auto,           ///< Follow system preference
        HighContrast,
        Custom
    };

    Theme(const std::string& name, Mode mode);
    ~Theme() = default;

    // Theme metadata
    std::string getName() const { return name_; }
    Mode getMode() const { return mode_; }
    std::string getDescription() const { return description_; }
    void setDescription(const std::string& desc) { description_ = desc; }

    // Color palette
    struct Colors {
        // Primary colors
        Color primary{0.2f, 0.5f, 0.9f};        ///< Primary accent
        Color secondary{0.5f, 0.3f, 0.8f};      ///< Secondary accent
        Color success{0.2f, 0.8f, 0.4f};        ///< Success/positive
        Color warning{0.9f, 0.7f, 0.2f};        ///< Warning
        Color error{0.9f, 0.3f, 0.2f};          ///< Error/negative
        Color info{0.3f, 0.7f, 0.9f};           ///< Informational

        // Backgrounds
        Color background{0.15f, 0.15f, 0.15f};  ///< Main background
        Color surface{0.2f, 0.2f, 0.2f};        ///< Surface/panel
        Color surfaceVariant{0.25f, 0.25f, 0.25f}; ///< Variant surface
        Color overlay{0.0f, 0.0f, 0.0f, 0.5f};  ///< Overlay/modal

        // Text
        Color textPrimary{0.95f, 0.95f, 0.95f}; ///< Primary text
        Color textSecondary{0.7f, 0.7f, 0.7f};  ///< Secondary text
        Color textDisabled{0.5f, 0.5f, 0.5f};   ///< Disabled text
        Color textOnPrimary{1.0f, 1.0f, 1.0f};  ///< Text on primary color

        // UI Elements
        Color border{0.3f, 0.3f, 0.3f};         ///< Borders
        Color divider{0.25f, 0.25f, 0.25f};     ///< Dividers
        Color hover{0.3f, 0.3f, 0.3f};          ///< Hover state
        Color active{0.35f, 0.35f, 0.35f};      ///< Active/pressed state
        Color focus{0.2f, 0.5f, 0.9f, 0.3f};    ///< Focus ring

        // DAW-specific colors
        Color waveform{0.3f, 0.7f, 0.9f};       ///< Waveform display
        Color midiNote{0.5f, 0.8f, 0.3f};       ///< MIDI note
        Color marker{0.9f, 0.5f, 0.2f};         ///< Timeline marker
        Color playhead{0.9f, 0.2f, 0.2f};       ///< Playhead/cursor
        Color selection{0.3f, 0.6f, 0.9f, 0.3f}; ///< Selection
        Color grid{0.25f, 0.25f, 0.25f};        ///< Grid lines

        // Meters and visualizers
        Color meterGreen{0.2f, 0.9f, 0.3f};     ///< Level meter (safe)
        Color meterYellow{0.9f, 0.8f, 0.2f};    ///< Level meter (caution)
        Color meterRed{0.9f, 0.2f, 0.2f};       ///< Level meter (clip)
        Color spectrum{0.2f, 0.6f, 0.9f};       ///< Spectrum analyzer
    };

    Colors& colors() { return colors_; }
    const Colors& colors() const { return colors_; }

    Typography& typography() { return typography_; }
    const Typography& typography() const { return typography_; }

    Spacing& spacing() { return spacing_; }
    const Spacing& spacing() const { return spacing_; }

    Animation& animation() { return animation_; }
    const Animation& animation() const { return animation_; }

    Accessibility& accessibility() { return accessibility_; }
    const Accessibility& accessibility() const { return accessibility_; }

    // Apply accessibility transformations
    void applyAccessibility();

    // Get color with accessibility applied
    Color getAccessibleColor(const Color& color) const;

    // Validate WCAG compliance
    bool isWCAGCompliant() const;
    std::vector<std::string> getWCAGViolations() const;

    // Export/Import
    std::string toJSON() const;
    static std::unique_ptr<Theme> fromJSON(const std::string& json);

private:
    std::string name_;
    Mode mode_;
    std::string description_;
    Colors colors_;
    Typography typography_;
    Spacing spacing_;
    Animation animation_;
    Accessibility accessibility_;
};

/**
 * @brief Theme manager - handles multiple themes and preferences
 */
class ThemeManager {
public:
    static ThemeManager& getInstance();

    // Theme registration
    void registerTheme(std::unique_ptr<Theme> theme);
    void unregisterTheme(const std::string& name);

    // Theme selection
    void setCurrentTheme(const std::string& name);
    Theme* getCurrentTheme() const { return currentTheme_.get(); }
    const std::string& getCurrentThemeName() const { return currentThemeName_; }

    // Get available themes
    std::vector<std::string> getAvailableThemes() const;
    Theme* getTheme(const std::string& name) const;

    // System theme detection
    void setFollowSystemTheme(bool follow);
    bool isFollowingSystemTheme() const { return followSystemTheme_; }
    Theme::Mode detectSystemTheme() const;

    // Theme change callback
    using ThemeChangeCallback = std::function<void(Theme*)>;
    void onThemeChange(ThemeChangeCallback callback);

    // Built-in themes
    static std::unique_ptr<Theme> createDarkTheme();
    static std::unique_ptr<Theme> createLightTheme();
    static std::unique_ptr<Theme> createHighContrastTheme();
    static std::unique_ptr<Theme> createAbletomTheme();      ///< Ableton-inspired
    static std::unique_ptr<Theme> createLogicProTheme();     ///< Logic Pro-inspired
    static std::unique_ptr<Theme> createFLStudioTheme();     ///< FL Studio-inspired
    static std::unique_ptr<Theme> createStudioOneTheme();    ///< Studio One-inspired

    // Persistence
    void loadPreferences();
    void savePreferences();
    std::string getPreferencesPath() const;

private:
    ThemeManager();
    ~ThemeManager() = default;
    ThemeManager(const ThemeManager&) = delete;
    ThemeManager& operator=(const ThemeManager&) = delete;

    void registerBuiltInThemes();
    void notifyThemeChange();

    std::map<std::string, std::unique_ptr<Theme>> themes_;
    std::unique_ptr<Theme> currentTheme_;
    std::string currentThemeName_;
    bool followSystemTheme_{false};
    std::vector<ThemeChangeCallback> callbacks_;
};

} // namespace UI
} // namespace MolinAntro
