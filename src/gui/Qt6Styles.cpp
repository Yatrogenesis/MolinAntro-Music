/**
 * @file Qt6Styles.cpp
 * @brief Theme and styling implementation for MolinAntro DAW
 *
 * @author Francisco Molina-Burgos
 * @date January 2026
 */

#ifdef BUILD_QT6_GUI

#include "gui/Qt6Styles.h"
#include "gui/Qt6MainWindow.h"
#include <QApplication>
#include <QScreen>

namespace MolinAntro {
namespace GUI {

ThemeManager& ThemeManager::instance() {
    static ThemeManager instance;
    return instance;
}

void ThemeManager::setTheme(Theme theme) {
    currentTheme_ = theme;
}

QString ThemeManager::getStyleSheet() const {
    QString style;
    style += getButtonStyle();
    style += getSliderStyle();
    style += getScrollBarStyle();
    style += getComboBoxStyle();
    style += getSpinBoxStyle();
    style += getLineEditStyle();
    style += getListWidgetStyle();
    style += getTreeWidgetStyle();
    style += getDockWidgetStyle();
    style += getMenuBarStyle();
    style += getToolBarStyle();
    style += getStatusBarStyle();
    style += getGroupBoxStyle();
    style += getTabWidgetStyle();

    // Base application style
    style += QString(R"(
        QWidget {
            background-color: %1;
            color: %2;
            font-family: '%3';
            font-size: %4px;
        }

        QMainWindow {
            background-color: %5;
        }

        QMainWindow::separator {
            background-color: %6;
            width: 2px;
            height: 2px;
        }

        QMainWindow::separator:hover {
            background-color: %7;
        }

        QSplitter::handle {
            background-color: %6;
        }

        QSplitter::handle:hover {
            background-color: %7;
        }

        QSplitter::handle:horizontal {
            width: 4px;
        }

        QSplitter::handle:vertical {
            height: 4px;
        }

        QFrame {
            border: none;
        }

        QFrame[frameShape="4"] {
            background-color: %6;
            max-height: 1px;
        }

        QFrame[frameShape="5"] {
            background-color: %6;
            max-width: 1px;
        }

        QToolTip {
            background-color: %8;
            color: %2;
            border: 1px solid %6;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: %9px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Background))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(Fonts::getFontFamily())
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(Fonts::SizeSmall);

    return style;
}

QPalette ThemeManager::getPalette() const {
    QPalette palette;

    palette.setColor(QPalette::Window, Colors::Background);
    palette.setColor(QPalette::WindowText, Colors::TextPrimary);
    palette.setColor(QPalette::Base, Colors::BackgroundLight);
    palette.setColor(QPalette::AlternateBase, Colors::Panel);
    palette.setColor(QPalette::Text, Colors::TextPrimary);
    palette.setColor(QPalette::Button, Colors::Panel);
    palette.setColor(QPalette::ButtonText, Colors::TextPrimary);
    palette.setColor(QPalette::BrightText, Colors::TextHighlight);
    palette.setColor(QPalette::Highlight, Colors::AccentBlue);
    palette.setColor(QPalette::HighlightedText, Colors::TextHighlight);
    palette.setColor(QPalette::Disabled, QPalette::Text, Colors::TextDisabled);
    palette.setColor(QPalette::Disabled, QPalette::ButtonText, Colors::TextDisabled);
    palette.setColor(QPalette::Link, Colors::AccentBlue);
    palette.setColor(QPalette::LinkVisited, Colors::AccentPurple);

    return palette;
}

QString ThemeManager::getButtonStyle() const {
    return QString(R"(
        QPushButton {
            background-color: %1;
            color: %2;
            border: 1px solid %3;
            border-radius: 4px;
            padding: 6px 16px;
            min-width: %4px;
            min-height: %5px;
            font-size: %6px;
            font-weight: 500;
        }

        QPushButton:hover {
            background-color: %7;
            border-color: %8;
        }

        QPushButton:pressed {
            background-color: %9;
        }

        QPushButton:focus {
            border: 2px solid %10;
            outline: none;
        }

        QPushButton:disabled {
            background-color: %11;
            color: %12;
            border-color: %11;
        }

        QPushButton:checked {
            background-color: %8;
            border-color: %8;
        }

        QPushButton:checked:hover {
            background-color: %10;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Style::MIN_BUTTON_WIDTH)
    .arg(Style::MIN_BUTTON_HEIGHT)
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::Focus))
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextDisabled));
}

QString ThemeManager::getSliderStyle() const {
    return QString(R"(
        QSlider {
            min-height: 24px;
        }

        QSlider::groove:horizontal {
            height: 6px;
            background: %1;
            border-radius: 3px;
            margin: 0 8px;
        }

        QSlider::groove:vertical {
            width: 6px;
            background: %1;
            border-radius: 3px;
            margin: 8px 0;
        }

        QSlider::handle:horizontal {
            background: %2;
            width: 18px;
            height: 18px;
            margin: -6px -8px;
            border-radius: 9px;
            border: 2px solid %3;
        }

        QSlider::handle:vertical {
            background: %2;
            width: 18px;
            height: 18px;
            margin: -8px -6px;
            border-radius: 9px;
            border: 2px solid %3;
        }

        QSlider::handle:horizontal:hover,
        QSlider::handle:vertical:hover {
            background: %4;
            border-color: %4;
        }

        QSlider::handle:horizontal:pressed,
        QSlider::handle:vertical:pressed {
            background: %5;
        }

        QSlider::sub-page:horizontal {
            background: %4;
            border-radius: 3px;
            margin: 0 8px;
        }

        QSlider::add-page:vertical {
            background: %4;
            border-radius: 3px;
            margin: 8px 0;
        }

        QSlider:focus {
            outline: none;
        }

        QSlider::handle:focus {
            border: 2px solid %6;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue.darker(120)))
    .arg(StyleUtils::colorToRgba(Colors::Focus));
}

QString ThemeManager::getScrollBarStyle() const {
    return QString(R"(
        QScrollBar:vertical {
            background: %1;
            width: %2px;
            margin: 0;
            border: none;
        }

        QScrollBar::handle:vertical {
            background: %3;
            min-height: 32px;
            border-radius: %4px;
            margin: 2px;
        }

        QScrollBar::handle:vertical:hover {
            background: %5;
        }

        QScrollBar::handle:vertical:pressed {
            background: %6;
        }

        QScrollBar::add-line:vertical,
        QScrollBar::sub-line:vertical {
            height: 0;
            background: none;
        }

        QScrollBar::add-page:vertical,
        QScrollBar::sub-page:vertical {
            background: none;
        }

        QScrollBar:horizontal {
            background: %1;
            height: %2px;
            margin: 0;
            border: none;
        }

        QScrollBar::handle:horizontal {
            background: %3;
            min-width: 32px;
            border-radius: %4px;
            margin: 2px;
        }

        QScrollBar::handle:horizontal:hover {
            background: %5;
        }

        QScrollBar::handle:horizontal:pressed {
            background: %6;
        }

        QScrollBar::add-line:horizontal,
        QScrollBar::sub-line:horizontal {
            width: 0;
            background: none;
        }

        QScrollBar::add-page:horizontal,
        QScrollBar::sub-page:horizontal {
            background: none;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(Style::SCROLLBAR_WIDTH)
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg((Style::SCROLLBAR_WIDTH - 4) / 2)
    .arg(StyleUtils::colorToRgba(Colors::BorderLight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue));
}

QString ThemeManager::getComboBoxStyle() const {
    return QString(R"(
        QComboBox {
            background-color: %1;
            color: %2;
            border: 1px solid %3;
            border-radius: 4px;
            padding: 6px 12px;
            padding-right: 28px;
            min-height: %4px;
            font-size: %5px;
        }

        QComboBox:hover {
            border-color: %6;
        }

        QComboBox:focus {
            border: 2px solid %7;
        }

        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: center right;
            width: 24px;
            border: none;
            background: transparent;
        }

        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid %2;
            margin-right: 8px;
        }

        QComboBox QAbstractItemView {
            background-color: %8;
            border: 1px solid %3;
            border-radius: 4px;
            selection-background-color: %6;
            selection-color: %9;
            padding: 4px;
        }

        QComboBox QAbstractItemView::item {
            padding: 8px 12px;
            min-height: 24px;
        }

        QComboBox QAbstractItemView::item:hover {
            background-color: %10;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Style::MIN_BUTTON_HEIGHT)
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::Focus))
    .arg(StyleUtils::colorToRgba(Colors::BackgroundLight))
    .arg(StyleUtils::colorToRgba(Colors::TextHighlight))
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight));
}

QString ThemeManager::getSpinBoxStyle() const {
    return QString(R"(
        QSpinBox, QDoubleSpinBox {
            background-color: %1;
            color: %2;
            border: 1px solid %3;
            border-radius: 4px;
            padding: 6px 8px;
            padding-right: 24px;
            min-height: %4px;
            font-size: %5px;
        }

        QSpinBox:hover, QDoubleSpinBox:hover {
            border-color: %6;
        }

        QSpinBox:focus, QDoubleSpinBox:focus {
            border: 2px solid %7;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            background: transparent;
            border: none;
            width: 20px;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-position: top right;
        }

        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-position: bottom right;
        }

        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-bottom: 5px solid %2;
        }

        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid %2;
        }

        QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover,
        QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {
            border-bottom-color: %6;
            border-top-color: %6;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Style::MIN_BUTTON_HEIGHT)
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::Focus));
}

QString ThemeManager::getLineEditStyle() const {
    return QString(R"(
        QLineEdit {
            background-color: %1;
            color: %2;
            border: 1px solid %3;
            border-radius: 4px;
            padding: 8px 12px;
            min-height: %4px;
            font-size: %5px;
            selection-background-color: %6;
            selection-color: %7;
        }

        QLineEdit:hover {
            border-color: %8;
        }

        QLineEdit:focus {
            border: 2px solid %9;
        }

        QLineEdit:disabled {
            background-color: %10;
            color: %11;
        }

        QLineEdit::placeholder {
            color: %12;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Style::MIN_BUTTON_HEIGHT - 8)
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::TextHighlight))
    .arg(StyleUtils::colorToRgba(Colors::BorderLight))
    .arg(StyleUtils::colorToRgba(Colors::Focus))
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextDisabled))
    .arg(StyleUtils::colorToRgba(Colors::TextSecondary));
}

QString ThemeManager::getListWidgetStyle() const {
    return QString(R"(
        QListWidget, QListView {
            background-color: %1;
            color: %2;
            border: 1px solid %3;
            border-radius: 4px;
            padding: 4px;
            outline: none;
        }

        QListWidget::item, QListView::item {
            padding: 8px 12px;
            min-height: 28px;
            border-radius: 4px;
            margin: 1px 0;
        }

        QListWidget::item:hover, QListView::item:hover {
            background-color: %4;
        }

        QListWidget::item:selected, QListView::item:selected {
            background-color: %5;
            color: %6;
        }

        QListWidget::item:selected:hover, QListView::item:selected:hover {
            background-color: %7;
        }

        QListWidget:focus, QListView:focus {
            border: 2px solid %8;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::TextHighlight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue.lighter(110)))
    .arg(StyleUtils::colorToRgba(Colors::Focus));
}

QString ThemeManager::getTreeWidgetStyle() const {
    return QString(R"(
        QTreeWidget, QTreeView {
            background-color: %1;
            color: %2;
            border: 1px solid %3;
            border-radius: 4px;
            padding: 4px;
            outline: none;
            alternate-background-color: %4;
        }

        QTreeWidget::item, QTreeView::item {
            padding: 6px 8px;
            min-height: 26px;
        }

        QTreeWidget::item:hover, QTreeView::item:hover {
            background-color: %5;
        }

        QTreeWidget::item:selected, QTreeView::item:selected {
            background-color: %6;
            color: %7;
        }

        QTreeWidget::branch {
            background: transparent;
        }

        QTreeWidget::branch:has-children:!has-siblings:closed,
        QTreeWidget::branch:closed:has-children:has-siblings {
            border-image: none;
            image: none;
        }

        QTreeWidget::branch:open:has-children:!has-siblings,
        QTreeWidget::branch:open:has-children:has-siblings {
            border-image: none;
            image: none;
        }

        QTreeWidget:focus, QTreeView:focus {
            border: 2px solid %8;
        }

        QHeaderView::section {
            background-color: %9;
            color: %2;
            padding: 8px 12px;
            border: none;
            border-right: 1px solid %3;
            font-weight: 500;
        }

        QHeaderView::section:hover {
            background-color: %5;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::TextHighlight))
    .arg(StyleUtils::colorToRgba(Colors::Focus))
    .arg(StyleUtils::colorToRgba(Colors::Panel));
}

QString ThemeManager::getDockWidgetStyle() const {
    return QString(R"(
        QDockWidget {
            color: %1;
            titlebar-close-icon: none;
            titlebar-normal-icon: none;
        }

        QDockWidget::title {
            background-color: %2;
            padding: 8px 12px;
            font-size: %3px;
            font-weight: 500;
            border-bottom: 1px solid %4;
        }

        QDockWidget::close-button,
        QDockWidget::float-button {
            background: transparent;
            border: none;
            padding: 4px;
        }

        QDockWidget::close-button:hover,
        QDockWidget::float-button:hover {
            background-color: %5;
            border-radius: 4px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight));
}

QString ThemeManager::getMenuBarStyle() const {
    return QString(R"(
        QMenuBar {
            background-color: %1;
            color: %2;
            padding: 4px 8px;
            spacing: 2px;
            font-size: %3px;
        }

        QMenuBar::item {
            background: transparent;
            padding: 8px 12px;
            border-radius: 4px;
        }

        QMenuBar::item:selected {
            background-color: %4;
        }

        QMenuBar::item:pressed {
            background-color: %5;
        }

        QMenu {
            background-color: %6;
            color: %2;
            border: 1px solid %7;
            border-radius: 8px;
            padding: 8px 4px;
        }

        QMenu::item {
            padding: 8px 32px 8px 24px;
            margin: 2px 4px;
            border-radius: 4px;
        }

        QMenu::item:selected {
            background-color: %5;
        }

        QMenu::separator {
            height: 1px;
            background-color: %7;
            margin: 6px 12px;
        }

        QMenu::indicator {
            width: 16px;
            height: 16px;
            margin-left: 8px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::Border));
}

QString ThemeManager::getToolBarStyle() const {
    return QString(R"(
        QToolBar {
            background-color: %1;
            border: none;
            border-bottom: 1px solid %2;
            padding: 4px 8px;
            spacing: 4px;
        }

        QToolBar::separator {
            background-color: %2;
            width: 1px;
            margin: 8px 4px;
        }

        QToolButton {
            background: transparent;
            border: none;
            border-radius: 4px;
            padding: 8px;
            min-width: %3px;
            min-height: %4px;
        }

        QToolButton:hover {
            background-color: %5;
        }

        QToolButton:pressed {
            background-color: %6;
        }

        QToolButton:checked {
            background-color: %7;
        }

        QToolButton:focus {
            outline: 2px solid %8;
            outline-offset: 2px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Style::MIN_BUTTON_WIDTH)
    .arg(Style::MIN_BUTTON_HEIGHT)
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue.darker(120)))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue))
    .arg(StyleUtils::colorToRgba(Colors::Focus));
}

QString ThemeManager::getStatusBarStyle() const {
    return QString(R"(
        QStatusBar {
            background-color: %1;
            color: %2;
            border-top: 1px solid %3;
            padding: 4px 12px;
            font-size: %4px;
            min-height: %5px;
        }

        QStatusBar::item {
            border: none;
        }

        QStatusBar QLabel {
            padding: 0 8px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextSecondary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Fonts::SizeSmall)
    .arg(Style::STATUSBAR_HEIGHT);
}

QString ThemeManager::getGroupBoxStyle() const {
    return QString(R"(
        QGroupBox {
            background-color: %1;
            border: 1px solid %2;
            border-radius: 6px;
            margin-top: 16px;
            padding: 16px 12px 12px 12px;
            font-weight: 500;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            padding: 0 8px;
            color: %3;
            background-color: %1;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary));
}

QString ThemeManager::getTabWidgetStyle() const {
    return QString(R"(
        QTabWidget::pane {
            background-color: %1;
            border: 1px solid %2;
            border-radius: 4px;
            margin-top: -1px;
        }

        QTabBar::tab {
            background-color: %3;
            color: %4;
            padding: 10px 20px;
            margin-right: 2px;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
            font-size: %5px;
            min-width: 80px;
        }

        QTabBar::tab:selected {
            background-color: %1;
            border: 1px solid %2;
            border-bottom: none;
        }

        QTabBar::tab:!selected {
            margin-top: 2px;
        }

        QTabBar::tab:hover:!selected {
            background-color: %6;
        }

        QTabBar::tab:focus {
            outline: 2px solid %7;
            outline-offset: -2px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::BackgroundDark))
    .arg(StyleUtils::colorToRgba(Colors::TextSecondary))
    .arg(Fonts::SizeNormal)
    .arg(StyleUtils::colorToRgba(Colors::PanelHighlight))
    .arg(StyleUtils::colorToRgba(Colors::Focus));
}

QString ThemeManager::getTransportButtonStyle(bool isActive) const {
    QColor bgColor = isActive ? Colors::AccentBlue : Colors::Panel;
    QColor hoverColor = isActive ? Colors::AccentBlue.lighter(110) : Colors::PanelHighlight;

    return QString(R"(
        QPushButton {
            background-color: %1;
            color: %2;
            border: none;
            border-radius: 6px;
            min-width: 48px;
            min-height: 40px;
            font-size: 18px;
            font-weight: bold;
        }

        QPushButton:hover {
            background-color: %3;
        }

        QPushButton:pressed {
            background-color: %4;
        }

        QPushButton:focus {
            outline: 2px solid %5;
            outline-offset: 2px;
        }
    )")
    .arg(StyleUtils::colorToRgba(bgColor))
    .arg(StyleUtils::colorToRgba(Colors::TextHighlight))
    .arg(StyleUtils::colorToRgba(hoverColor))
    .arg(StyleUtils::colorToRgba(bgColor.darker(120)))
    .arg(StyleUtils::colorToRgba(Colors::Focus));
}

QString ThemeManager::getFaderStyle() const {
    return QString(R"(
        QSlider::groove:vertical {
            width: %1px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 %2, stop:0.7 %3, stop:1 %4);
            border-radius: 3px;
        }

        QSlider::handle:vertical {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 %5, stop:0.5 %6, stop:1 %5);
            height: 24px;
            margin: 0 -8px;
            border-radius: 4px;
            border: 1px solid %7;
        }

        QSlider::handle:vertical:hover {
            border-color: %8;
        }

        QSlider::sub-page:vertical {
            background: transparent;
        }
    )")
    .arg(Style::FADER_WIDTH - 12)
    .arg(StyleUtils::colorToRgba(Colors::MeterGreen))
    .arg(StyleUtils::colorToRgba(Colors::MeterYellow))
    .arg(StyleUtils::colorToRgba(Colors::MeterRed))
    .arg(StyleUtils::colorToRgba(Colors::TextPrimary))
    .arg(StyleUtils::colorToRgba(Colors::TextSecondary))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue));
}

QString ThemeManager::getKnobStyle() const {
    // Knobs are custom-painted, this is for the container
    return QString(R"(
        QWidget[class="KnobWidget"] {
            background: transparent;
            min-width: %1px;
            min-height: %2px;
        }
    )")
    .arg(Style::KNOB_SIZE)
    .arg(Style::KNOB_SIZE + 16);
}

QString ThemeManager::getMeterStyle() const {
    return QString(R"(
        QProgressBar {
            background-color: %1;
            border: none;
            border-radius: 2px;
        }

        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:1, x2:0, y2:0,
                stop:0 %2, stop:0.7 %3, stop:1 %4);
            border-radius: 2px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::MeterBackground))
    .arg(StyleUtils::colorToRgba(Colors::MeterGreen))
    .arg(StyleUtils::colorToRgba(Colors::MeterYellow))
    .arg(StyleUtils::colorToRgba(Colors::MeterRed));
}

QString ThemeManager::getChannelStripStyle() const {
    return QString(R"(
        QFrame[class="ChannelStrip"] {
            background-color: %1;
            border: 1px solid %2;
            border-radius: 6px;
            min-width: %3px;
            max-width: %4px;
        }

        QFrame[class="ChannelStrip"]:hover {
            border-color: %5;
        }

        QFrame[class="ChannelStrip"][selected="true"] {
            border-color: %6;
            border-width: 2px;
        }
    )")
    .arg(StyleUtils::colorToRgba(Colors::Panel))
    .arg(StyleUtils::colorToRgba(Colors::Border))
    .arg(Style::CHANNEL_MIN_WIDTH)
    .arg(Style::CHANNEL_WIDTH + 20)
    .arg(StyleUtils::colorToRgba(Colors::BorderLight))
    .arg(StyleUtils::colorToRgba(Colors::AccentBlue));
}

// ============================================================================
// StyleUtils implementation
// ============================================================================

namespace StyleUtils {

QString colorToRgba(const QColor& color) {
    return QString("rgba(%1, %2, %3, %4)")
        .arg(color.red())
        .arg(color.green())
        .arg(color.blue())
        .arg(color.alpha());
}

int scaledSize(int baseSize, qreal dpr) {
    return static_cast<int>(baseSize * dpr);
}

QString getFocusRingStyle(const QColor& color) {
    return QString("outline: 2px solid %1; outline-offset: 2px;")
        .arg(colorToRgba(color));
}

QString getButtonGradient(const QColor& base, bool pressed) {
    QColor light = base.lighter(pressed ? 90 : 110);
    QColor dark = base.darker(pressed ? 110 : 90);

    return QString("qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 %1, stop:1 %2)")
        .arg(colorToRgba(light))
        .arg(colorToRgba(dark));
}

QString getMeterGradient(bool horizontal) {
    QString orient = horizontal ? "x1:0, y1:0, x2:1, y2:0" : "x1:0, y1:1, x2:0, y2:0";

    return QString("qlineargradient(%1, stop:0 %2, stop:0.6 %3, stop:0.85 %4, stop:1 %5)")
        .arg(orient)
        .arg(colorToRgba(Colors::MeterGreen))
        .arg(colorToRgba(Colors::MeterGreen))
        .arg(colorToRgba(Colors::MeterYellow))
        .arg(colorToRgba(Colors::MeterRed));
}

} // namespace StyleUtils

} // namespace GUI
} // namespace MolinAntro

#endif // BUILD_QT6_GUI
