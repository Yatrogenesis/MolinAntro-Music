#pragma once

/**
 * @file Qt6Styles.h
 * @brief Theme and styling system for MolinAntro DAW
 *
 * Professional dark theme with proper contrast ratios
 * for accessibility (WCAG 2.1 AA compliance target)
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#ifdef BUILD_QT6_GUI

#include <QString>
#include <QColor>
#include <QFont>
#include <QPalette>

namespace MolinAntro {
namespace GUI {

/**
 * @brief Color palette for professional DAW appearance
 */
namespace Colors {
    // Background colors (dark theme)
    const QColor Background        = QColor(30, 30, 32);
    const QColor BackgroundLight   = QColor(42, 42, 46);
    const QColor BackgroundDark    = QColor(22, 22, 24);
    const QColor Panel             = QColor(38, 38, 42);
    const QColor PanelHighlight    = QColor(50, 50, 56);

    // Text colors (high contrast)
    const QColor TextPrimary       = QColor(240, 240, 242);
    const QColor TextSecondary     = QColor(180, 180, 186);
    const QColor TextDisabled      = QColor(100, 100, 106);
    const QColor TextHighlight     = QColor(255, 255, 255);

    // Accent colors
    const QColor AccentBlue        = QColor(66, 133, 244);
    const QColor AccentGreen       = QColor(52, 199, 89);
    const QColor AccentRed         = QColor(255, 69, 58);
    const QColor AccentOrange      = QColor(255, 159, 10);
    const QColor AccentPurple      = QColor(175, 82, 222);
    const QColor AccentYellow      = QColor(255, 214, 10);

    // Transport colors
    const QColor PlayGreen         = QColor(48, 209, 88);
    const QColor StopGray          = QColor(142, 142, 147);
    const QColor RecordRed         = QColor(255, 59, 48);
    const QColor PausedYellow      = QColor(255, 204, 0);

    // Meter colors
    const QColor MeterGreen        = QColor(52, 199, 89);
    const QColor MeterYellow       = QColor(255, 204, 0);
    const QColor MeterRed          = QColor(255, 59, 48);
    const QColor MeterBackground   = QColor(26, 26, 28);

    // Track colors (for channel strips)
    const QColor TrackColors[] = {
        QColor(255, 107, 107),   // Coral
        QColor(255, 179, 71),    // Orange
        QColor(255, 230, 109),   // Yellow
        QColor(134, 227, 206),   // Teal
        QColor(126, 214, 223),   // Cyan
        QColor(123, 182, 255),   // Blue
        QColor(178, 153, 255),   // Purple
        QColor(255, 153, 204),   // Pink
    };
    const int TrackColorCount = 8;

    // Border and separator
    const QColor Border            = QColor(60, 60, 66);
    const QColor BorderLight       = QColor(80, 80, 88);
    const QColor Separator         = QColor(48, 48, 52);

    // Focus indicator (accessibility)
    const QColor Focus             = QColor(66, 133, 244);
    const QColor FocusRing         = QColor(66, 133, 244, 180);

    // Selection
    const QColor Selection         = QColor(66, 133, 244, 100);
    const QColor SelectionBorder   = QColor(66, 133, 244);

    // Waveform colors
    const QColor WaveformFill      = QColor(66, 133, 244, 180);
    const QColor WaveformStroke    = QColor(100, 160, 255);
    const QColor Playhead          = QColor(255, 255, 255);
    const QColor LoopRegion        = QColor(255, 204, 0, 60);

    // Grid
    const QColor GridMajor         = QColor(60, 60, 66);
    const QColor GridMinor         = QColor(42, 42, 46);
}

/**
 * @brief Font definitions
 */
namespace Fonts {
    // Font sizes (in pixels, scaled by DPI)
    constexpr int SizeSmall        = 11;
    constexpr int SizeNormal       = 13;
    constexpr int SizeMedium       = 15;
    constexpr int SizeLarge        = 18;
    constexpr int SizeHeading      = 22;
    constexpr int SizeDisplay      = 28;

    // Font weights
    constexpr int WeightLight      = 300;
    constexpr int WeightNormal     = 400;
    constexpr int WeightMedium     = 500;
    constexpr int WeightBold       = 700;

    // Font families (cross-platform)
    inline QString getFontFamily() {
#ifdef Q_OS_WIN
        return "Segoe UI";
#elif defined(Q_OS_MAC)
        return "SF Pro Display";
#else
        return "Ubuntu";
#endif
    }

    inline QString getMonoFamily() {
#ifdef Q_OS_WIN
        return "Cascadia Mono";
#elif defined(Q_OS_MAC)
        return "SF Mono";
#else
        return "Ubuntu Mono";
#endif
    }
}

/**
 * @brief Theme manager for applying styles
 */
class ThemeManager {
public:
    enum class Theme {
        Dark,
        Light,
        HighContrast
    };

    static ThemeManager& instance();

    void setTheme(Theme theme);
    Theme getTheme() const { return currentTheme_; }

    // Get complete stylesheet for application
    QString getStyleSheet() const;

    // Get palette for theme
    QPalette getPalette() const;

    // Specific component styles
    QString getButtonStyle() const;
    QString getSliderStyle() const;
    QString getScrollBarStyle() const;
    QString getComboBoxStyle() const;
    QString getSpinBoxStyle() const;
    QString getLineEditStyle() const;
    QString getListWidgetStyle() const;
    QString getTreeWidgetStyle() const;
    QString getDockWidgetStyle() const;
    QString getMenuBarStyle() const;
    QString getToolBarStyle() const;
    QString getStatusBarStyle() const;
    QString getGroupBoxStyle() const;
    QString getTabWidgetStyle() const;

    // Custom DAW component styles
    QString getTransportButtonStyle(bool isActive = false) const;
    QString getFaderStyle() const;
    QString getKnobStyle() const;
    QString getMeterStyle() const;
    QString getChannelStripStyle() const;

private:
    ThemeManager() = default;
    Theme currentTheme_ = Theme::Dark;

    // Prevent copying
    ThemeManager(const ThemeManager&) = delete;
    ThemeManager& operator=(const ThemeManager&) = delete;
};

/**
 * @brief Utility functions for styling
 */
namespace StyleUtils {
    // Convert color to CSS rgba string
    QString colorToRgba(const QColor& color);

    // Get scaled size for HiDPI
    int scaledSize(int baseSize, qreal dpr);

    // Get focus ring CSS
    QString getFocusRingStyle(const QColor& color = Colors::Focus);

    // Get gradient CSS for buttons
    QString getButtonGradient(const QColor& base, bool pressed = false);

    // Generate meter gradient
    QString getMeterGradient(bool horizontal = false);
}

} // namespace GUI
} // namespace MolinAntro

#endif // BUILD_QT6_GUI
