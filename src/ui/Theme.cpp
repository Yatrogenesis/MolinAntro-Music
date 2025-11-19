// Theme.cpp - Professional Theme System with Accessibility
// MolinAntro DAW ACME Edition v3.0.0

#include "ui/Theme.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <fstream>

namespace MolinAntro {
namespace UI {

// ============================================================================
// Color Implementation
// ============================================================================

Color Color::fromRGB(int red, int green, int blue, int alpha) {
    return Color(
        std::clamp(red, 0, 255) / 255.0f,
        std::clamp(green, 0, 255) / 255.0f,
        std::clamp(blue, 0, 255) / 255.0f,
        std::clamp(alpha, 0, 255) / 255.0f
    );
}

Color Color::fromHex(const std::string& hex) {
    std::string h = hex;
    if (h[0] == '#') h = h.substr(1);

    unsigned int value;
    std::stringstream ss;
    ss << std::hex << h;
    ss >> value;

    if (h.length() == 6) {
        return fromRGB((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF);
    } else if (h.length() == 8) {
        return fromRGB((value >> 24) & 0xFF, (value >> 16) & 0xFF,
                      (value >> 8) & 0xFF, value & 0xFF);
    }

    return Color();
}

Color Color::toAccessible(const std::string& colorblindType) const {
    // Simulate colorblindness using Vienot, Brettel, and Mollon transformation
    Color result = *this;

    if (colorblindType == "protanopia") {
        // Red-blind: remove red component influence
        result.r = 0.567f * r + 0.433f * g;
        result.g = 0.558f * r + 0.442f * g;
        result.b = 0.242f * g + 0.758f * b;
    } else if (colorblindType == "deuteranopia") {
        // Green-blind: remove green component influence
        result.r = 0.625f * r + 0.375f * g;
        result.g = 0.7f * r + 0.3f * g;
        result.b = 0.3f * g + 0.7f * b;
    } else if (colorblindType == "tritanopia") {
        // Blue-blind: remove blue component influence
        result.r = 0.95f * r + 0.05f * g;
        result.g = 0.433f * g + 0.567f * b;
        result.b = 0.475f * g + 0.525f * b;
    } else if (colorblindType == "achromatopsia") {
        // Total colorblindness: convert to grayscale
        float gray = 0.299f * r + 0.587f * g + 0.114f * b;
        result.r = result.g = result.b = gray;
    }

    return result;
}

float Color::contrastRatio(const Color& other) const {
    // Calculate relative luminance (WCAG 2.1 formula)
    auto luminance = [](const Color& c) -> float {
        auto adjust = [](float channel) -> float {
            return (channel <= 0.03928f) ? channel / 12.92f
                                         : std::pow((channel + 0.055f) / 1.055f, 2.4f);
        };
        return 0.2126f * adjust(c.r) + 0.7152f * adjust(c.g) + 0.0722f * adjust(c.b);
    };

    float l1 = luminance(*this);
    float l2 = luminance(other);

    float lighter = std::max(l1, l2);
    float darker = std::min(l1, l2);

    return (lighter + 0.05f) / (darker + 0.05f);
}

Color Color::lighter(float amount) const {
    return Color(
        std::min(r + amount, 1.0f),
        std::min(g + amount, 1.0f),
        std::min(b + amount, 1.0f),
        a
    );
}

Color Color::darker(float amount) const {
    return Color(
        std::max(r - amount, 0.0f),
        std::max(g - amount, 0.0f),
        std::max(b - amount, 0.0f),
        a
    );
}

Color Color::interpolate(const Color& other, float t) const {
    t = std::clamp(t, 0.0f, 1.0f);
    return Color(
        r + (other.r - r) * t,
        g + (other.g - g) * t,
        b + (other.b - b) * t,
        a + (other.a - a) * t
    );
}

// ============================================================================
// Typography Implementation
// ============================================================================

std::string Typography::getDyslexiaFont() const {
    if (dyslexiaFriendly) {
        return "OpenDyslexic";  // Falls back to Comic Sans or Arial
    }
    return fontFamily;
}

float Typography::getDyslexiaSpacing() const {
    if (dyslexiaFriendly) {
        return 0.12f;  // 12% extra letter spacing
    }
    return letterSpacing;
}

// ============================================================================
// Animation Implementation
// ============================================================================

Animation Animation::fast() {
    Animation anim;
    anim.duration = 0.1f;
    anim.easing = Easing::EaseOut;
    return anim;
}

Animation Animation::normal() {
    Animation anim;
    anim.duration = 0.2f;
    anim.easing = Easing::EaseInOut;
    return anim;
}

Animation Animation::slow() {
    Animation anim;
    anim.duration = 0.4f;
    anim.easing = Easing::EaseInOut;
    return anim;
}

Animation Animation::smooth() {
    Animation anim;
    anim.duration = 0.3f;
    anim.easing = Easing::EaseOut;
    anim.useGPU = true;
    return anim;
}

// ============================================================================
// Theme Implementation
// ============================================================================

Theme::Theme(const std::string& name, Mode mode)
    : name_(name), mode_(mode) {
}

void Theme::applyAccessibility() {
    if (accessibility_.highContrast) {
        // Increase contrast for all colors
        colors_.background = colors_.background.darker(0.1f);
        colors_.textPrimary = Color(1.0f, 1.0f, 1.0f);
        colors_.border = colors_.border.lighter(0.2f);
    }

    if (accessibility_.colorblindMode != Accessibility::ColorblindMode::None) {
        // Apply colorblind simulation to all colors
        std::string mode;
        switch (accessibility_.colorblindMode) {
            case Accessibility::ColorblindMode::Protanopia:
                mode = "protanopia"; break;
            case Accessibility::ColorblindMode::Deuteranopia:
                mode = "deuteranopia"; break;
            case Accessibility::ColorblindMode::Tritanopia:
                mode = "tritanopia"; break;
            case Accessibility::ColorblindMode::Achromatopsia:
                mode = "achromatopsia"; break;
            default: break;
        }

        if (!mode.empty()) {
            colors_.primary = colors_.primary.toAccessible(mode);
            colors_.secondary = colors_.secondary.toAccessible(mode);
            colors_.success = colors_.success.toAccessible(mode);
            colors_.warning = colors_.warning.toAccessible(mode);
            colors_.error = colors_.error.toAccessible(mode);
        }
    }

    if (accessibility_.dyslexiaMode) {
        typography_.dyslexiaFriendly = true;
        typography_.lineHeight = 1.8f;
        typography_.letterSpacing = 0.12f;
    }

    if (accessibility_.largeText) {
        typography_.fontSize *= accessibility_.textScale;
    }

    if (accessibility_.reducedMotion) {
        animation_.duration = 0.0f;  // Disable animations
    }
}

Color Theme::getAccessibleColor(const Color& color) const {
    if (accessibility_.colorblindMode == Accessibility::ColorblindMode::None) {
        return color;
    }

    std::string mode;
    switch (accessibility_.colorblindMode) {
        case Accessibility::ColorblindMode::Protanopia:
            mode = "protanopia"; break;
        case Accessibility::ColorblindMode::Deuteranopia:
            mode = "deuteranopia"; break;
        case Accessibility::ColorblindMode::Tritanopia:
            mode = "tritanopia"; break;
        case Accessibility::ColorblindMode::Achromatopsia:
            mode = "achromatopsia"; break;
        default: return color;
    }

    return color.toAccessible(mode);
}

bool Theme::isWCAGCompliant() const {
    // Check text/background contrast ratios
    float minRatio = 4.5f;  // AA
    if (accessibility_.targetLevel == Accessibility::WCAGLevel::AAA) {
        minRatio = 7.0f;
    }

    float textBgRatio = colors_.textPrimary.contrastRatio(colors_.background);
    float primaryTextRatio = colors_.textOnPrimary.contrastRatio(colors_.primary);

    return textBgRatio >= minRatio && primaryTextRatio >= minRatio;
}

std::vector<std::string> Theme::getWCAGViolations() const {
    std::vector<std::string> violations;

    float minRatio = (accessibility_.targetLevel == Accessibility::WCAGLevel::AAA) ? 7.0f : 4.5f;

    auto checkContrast = [&](const Color& fg, const Color& bg, const std::string& name) {
        float ratio = fg.contrastRatio(bg);
        if (ratio < minRatio) {
            std::stringstream ss;
            ss << name << ": " << std::fixed << std::setprecision(2)
               << ratio << " (required: " << minRatio << ")";
            violations.push_back(ss.str());
        }
    };

    checkContrast(colors_.textPrimary, colors_.background, "Text/Background");
    checkContrast(colors_.textOnPrimary, colors_.primary, "Text on Primary");
    checkContrast(colors_.error, colors_.background, "Error/Background");
    checkContrast(colors_.success, colors_.background, "Success/Background");

    return violations;
}

std::string Theme::toJSON() const {
    // Simple JSON export (in production, use a proper JSON library)
    std::stringstream ss;
    ss << "{\n";
    ss << "  \"name\": \"" << name_ << "\",\n";
    ss << "  \"mode\": " << static_cast<int>(mode_) << ",\n";
    ss << "  \"description\": \"" << description_ << "\"\n";
    ss << "}\n";
    return ss.str();
}

// ============================================================================
// Built-in Themes
// ============================================================================

std::unique_ptr<Theme> ThemeManager::createDarkTheme() {
    auto theme = std::make_unique<Theme>("Dark", Theme::Mode::Dark);
    theme->setDescription("Professional dark theme optimized for long sessions");

    auto& c = theme->colors();

    // Backgrounds - Deep dark grays
    c.background = Color(0.12f, 0.12f, 0.12f);
    c.surface = Color(0.18f, 0.18f, 0.18f);
    c.surfaceVariant = Color(0.22f, 0.22f, 0.22f);

    // Primary - Electric blue
    c.primary = Color::fromHex("#2196F3");
    c.secondary = Color::fromHex("#9C27B0");

    // Feedback colors
    c.success = Color::fromHex("#4CAF50");
    c.warning = Color::fromHex("#FF9800");
    c.error = Color::fromHex("#F44336");
    c.info = Color::fromHex("#00BCD4");

    // Text - High contrast
    c.textPrimary = Color(0.95f, 0.95f, 0.95f);
    c.textSecondary = Color(0.7f, 0.7f, 0.7f);
    c.textDisabled = Color(0.5f, 0.5f, 0.5f);

    // DAW colors
    c.waveform = Color::fromHex("#42A5F5");
    c.midiNote = Color::fromHex("#66BB6A");
    c.playhead = Color::fromHex("#EF5350");
    c.selection = Color::fromHex("#2196F3");
    c.selection.a = 0.3f;

    // Meters
    c.meterGreen = Color::fromHex("#66BB6A");
    c.meterYellow = Color::fromHex("#FFEB3B");
    c.meterRed = Color::fromHex("#EF5350");

    return theme;
}

std::unique_ptr<Theme> ThemeManager::createLightTheme() {
    auto theme = std::make_unique<Theme>("Light", Theme::Mode::Light);
    theme->setDescription("Clean light theme for bright environments");

    auto& c = theme->colors();

    // Backgrounds - Light grays
    c.background = Color(0.98f, 0.98f, 0.98f);
    c.surface = Color(1.0f, 1.0f, 1.0f);
    c.surfaceVariant = Color(0.96f, 0.96f, 0.96f);

    // Primary - Vibrant blue
    c.primary = Color::fromHex("#1976D2");
    c.secondary = Color::fromHex("#7B1FA2");

    // Feedback colors
    c.success = Color::fromHex("#388E3C");
    c.warning = Color::fromHex("#F57C00");
    c.error = Color::fromHex("#D32F2F");
    c.info = Color::fromHex("#0097A7");

    // Text - Dark for contrast
    c.textPrimary = Color(0.13f, 0.13f, 0.13f);
    c.textSecondary = Color(0.38f, 0.38f, 0.38f);
    c.textDisabled = Color(0.62f, 0.62f, 0.62f);
    c.textOnPrimary = Color(1.0f, 1.0f, 1.0f);

    // DAW colors
    c.waveform = Color::fromHex("#1976D2");
    c.midiNote = Color::fromHex("#388E3C");
    c.playhead = Color::fromHex("#D32F2F");
    c.selection = Color::fromHex("#1976D2");
    c.selection.a = 0.2f;

    return theme;
}

std::unique_ptr<Theme> ThemeManager::createHighContrastTheme() {
    auto theme = std::make_unique<Theme>("High Contrast", Theme::Mode::HighContrast);
    theme->setDescription("Maximum contrast for accessibility");

    auto& c = theme->colors();

    // Pure black and white
    c.background = Color(0.0f, 0.0f, 0.0f);
    c.surface = Color(0.05f, 0.05f, 0.05f);
    c.surfaceVariant = Color(0.1f, 0.1f, 0.1f);

    // Bright, saturated colors
    c.primary = Color(0.0f, 0.7f, 1.0f);
    c.secondary = Color(1.0f, 0.3f, 1.0f);
    c.success = Color(0.0f, 1.0f, 0.0f);
    c.warning = Color(1.0f, 0.9f, 0.0f);
    c.error = Color(1.0f, 0.0f, 0.0f);

    // Pure white text
    c.textPrimary = Color(1.0f, 1.0f, 1.0f);
    c.textSecondary = Color(0.9f, 0.9f, 0.9f);
    c.border = Color(1.0f, 1.0f, 1.0f);

    // Enable accessibility
    theme->accessibility().highContrast = true;
    theme->accessibility().targetLevel = Accessibility::WCAGLevel::AAA;

    return theme;
}

std::unique_ptr<Theme> ThemeManager::createAbletomTheme() {
    auto theme = std::make_unique<Theme>("Ableton", Theme::Mode::Dark);
    theme->setDescription("Inspired by Ableton Live's interface");

    auto& c = theme->colors();

    // Ableton's characteristic dark blue-gray
    c.background = Color::fromHex("#1E1E1E");
    c.surface = Color::fromHex("#2A2A2A");
    c.surfaceVariant = Color::fromHex("#353535");

    // Ableton's orange accent
    c.primary = Color::fromHex("#FF764D");
    c.secondary = Color::fromHex("#5FB4FF");

    // Status colors
    c.success = Color::fromHex("#8AFF80");
    c.warning = Color::fromHex("#FFB84D");
    c.error = Color::fromHex("#FF6B6B");

    // Ableton-style text
    c.textPrimary = Color(0.9f, 0.9f, 0.9f);
    c.textSecondary = Color(0.65f, 0.65f, 0.65f);

    // Clip colors
    c.waveform = Color::fromHex("#00A9FF");
    c.midiNote = Color::fromHex("#8AFF80");
    c.playhead = Color::fromHex("#FF764D");

    return theme;
}

std::unique_ptr<Theme> ThemeManager::createLogicProTheme() {
    auto theme = std::make_unique<Theme>("Logic Pro", Theme::Mode::Dark);
    theme->setDescription("Inspired by Logic Pro X");

    auto& c = theme->colors();

    // Logic's dark gray
    c.background = Color::fromHex("#2D2D2D");
    c.surface = Color::fromHex("#3A3A3A");
    c.surfaceVariant = Color::fromHex("#474747");

    // Logic's blue accent
    c.primary = Color::fromHex("#0074D9");
    c.secondary = Color::fromHex("#B10DC9");

    c.success = Color::fromHex("#2ECC40");
    c.warning = Color::fromHex("#FF851B");
    c.error = Color::fromHex("#FF4136");

    c.textPrimary = Color(0.92f, 0.92f, 0.92f);
    c.waveform = Color::fromHex("#0074D9");
    c.midiNote = Color::fromHex("#7FDBFF");

    return theme;
}

std::unique_ptr<Theme> ThemeManager::createFLStudioTheme() {
    auto theme = std::make_unique<Theme>("FL Studio", Theme::Mode::Dark);
    theme->setDescription("Inspired by FL Studio's vibrant interface");

    auto& c = theme->colors();

    // FL Studio's dark background
    c.background = Color::fromHex("#1E1E1E");
    c.surface = Color::fromHex("#2B2B2B");
    c.surfaceVariant = Color::fromHex("#383838");

    // FL Studio's orange
    c.primary = Color::fromHex("#FF8C00");
    c.secondary = Color::fromHex("#FF1744");

    c.success = Color::fromHex("#00E676");
    c.warning = Color::fromHex("#FFAB00");
    c.error = Color::fromHex("#FF1744");

    c.waveform = Color::fromHex("#00E5FF");
    c.midiNote = Color::fromHex("#76FF03");
    c.playhead = Color::fromHex("#FF1744");

    return theme;
}

std::unique_ptr<Theme> ThemeManager::createStudioOneTheme() {
    auto theme = std::make_unique<Theme>("Studio One", Theme::Mode::Dark);
    theme->setDescription("Inspired by PreSonus Studio One");

    auto& c = theme->colors();

    // Studio One's blue-gray
    c.background = Color::fromHex("#24292E");
    c.surface = Color::fromHex("#2F363D");
    c.surfaceVariant = Color::fromHex("#3A4249");

    // Studio One's blue
    c.primary = Color::fromHex("#0366D6");
    c.secondary = Color::fromHex("#6F42C1");

    c.success = Color::fromHex("#28A745");
    c.warning = Color::fromHex("#FFD33D");
    c.error = Color::fromHex("#D73A49");

    c.waveform = Color::fromHex("#58A6FF");
    c.midiNote = Color::fromHex("#56D364");

    return theme;
}

// ============================================================================
// ThemeManager Implementation
// ============================================================================

ThemeManager::ThemeManager() {
    registerBuiltInThemes();
    setCurrentTheme("Dark");
}

ThemeManager& ThemeManager::getInstance() {
    static ThemeManager instance;
    return instance;
}

void ThemeManager::registerTheme(std::unique_ptr<Theme> theme) {
    std::string name = theme->getName();
    themes_[name] = std::move(theme);
}

void ThemeManager::unregisterTheme(const std::string& name) {
    themes_.erase(name);
}

void ThemeManager::setCurrentTheme(const std::string& name) {
    auto it = themes_.find(name);
    if (it != themes_.end()) {
        currentTheme_ = std::make_unique<Theme>(*it->second);
        currentThemeName_ = name;
        currentTheme_->applyAccessibility();
        notifyThemeChange();
        savePreferences();
    }
}

std::vector<std::string> ThemeManager::getAvailableThemes() const {
    std::vector<std::string> names;
    for (const auto& pair : themes_) {
        names.push_back(pair.first);
    }
    return names;
}

Theme* ThemeManager::getTheme(const std::string& name) const {
    auto it = themes_.find(name);
    return (it != themes_.end()) ? it->second.get() : nullptr;
}

void ThemeManager::setFollowSystemTheme(bool follow) {
    followSystemTheme_ = follow;
    if (follow) {
        Theme::Mode systemMode = detectSystemTheme();
        std::string themeName = (systemMode == Theme::Mode::Dark) ? "Dark" : "Light";
        setCurrentTheme(themeName);
    }
}

Theme::Mode ThemeManager::detectSystemTheme() const {
    // On Linux, check GNOME/KDE settings
    // On macOS, check NSAppearance
    // On Windows, check registry
    // For now, default to dark
    return Theme::Mode::Dark;
}

void ThemeManager::onThemeChange(ThemeChangeCallback callback) {
    callbacks_.push_back(callback);
}

void ThemeManager::registerBuiltInThemes() {
    registerTheme(createDarkTheme());
    registerTheme(createLightTheme());
    registerTheme(createHighContrastTheme());
    registerTheme(createAbletomTheme());
    registerTheme(createLogicProTheme());
    registerTheme(createFLStudioTheme());
    registerTheme(createStudioOneTheme());
}

void ThemeManager::notifyThemeChange() {
    for (auto& callback : callbacks_) {
        callback(currentTheme_.get());
    }
}

void ThemeManager::loadPreferences() {
    // TODO: Load from JSON file
    std::string path = getPreferencesPath();
    std::ifstream file(path);
    if (file.is_open()) {
        // Parse JSON and apply settings
        file.close();
    }
}

void ThemeManager::savePreferences() {
    // TODO: Save to JSON file
    std::string path = getPreferencesPath();
    std::ofstream file(path);
    if (file.is_open()) {
        if (currentTheme_) {
            file << currentTheme_->toJSON();
        }
        file.close();
    }
}

std::string ThemeManager::getPreferencesPath() const {
    // Platform-specific preferences path
    #ifdef _WIN32
        return std::string(getenv("APPDATA")) + "/MolinAntro/theme.json";
    #elif __APPLE__
        return std::string(getenv("HOME")) + "/Library/Application Support/MolinAntro/theme.json";
    #else
        return std::string(getenv("HOME")) + "/.config/molinantro/theme.json";
    #endif
}

} // namespace UI
} // namespace MolinAntro
