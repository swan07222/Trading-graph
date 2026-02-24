"""Modern UI tokens and shared styles for Trading Graph.

This module centralizes colors, typography, spacing, and reusable style helpers
to keep the desktop UI visually consistent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class ModernColors:
    """Professional midnight palette with bright, high-contrast accents."""

    # Core surfaces
    BG_CANVAS = "#040a15"
    BG_PRIMARY = "#0a1426"
    BG_SECONDARY = "#111f37"
    BG_TERTIARY = "#172a48"
    BG_ELEVATED = "#1f3760"

    # Brand and semantic accents
    ACCENT_PRIMARY = "#3ad8ff"
    ACCENT_SECONDARY = "#67e1bf"
    ACCENT_SUCCESS = "#36d3a4"
    ACCENT_WARNING = "#f1c96d"
    ACCENT_DANGER = "#ff6c80"
    ACCENT_INFO = "#83abff"

    # Text
    TEXT_PRIMARY = "#ecf3ff"
    TEXT_SECONDARY = "#b7c8e8"
    TEXT_MUTED = "#7f96c0"
    TEXT_STRONG = "#f7fbff"

    # Borders
    BORDER_SUBTLE = "#24406b"
    BORDER_DEFAULT = "#31588f"
    BORDER_FOCUS = "#69deff"

    # Signal colors
    SIGNAL_BUY = ACCENT_SUCCESS
    SIGNAL_BUY_BG = "rgba(54, 211, 164, 0.17)"
    SIGNAL_SELL = ACCENT_DANGER
    SIGNAL_SELL_BG = "rgba(255, 108, 128, 0.17)"
    SIGNAL_HOLD = ACCENT_WARNING
    SIGNAL_HOLD_BG = "rgba(241, 201, 109, 0.15)"

    # Gradients
    GRADIENT_PRIMARY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #1ec7ef, stop:1 #3ad8ff)"
    )
    GRADIENT_BUY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #1fbf8f, stop:1 #36d3a4)"
    )
    GRADIENT_SELL = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #f25c72, stop:1 #ff6c80)"
    )
    GRADIENT_WARNING = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #e8b24f, stop:1 #f1c96d)"
    )
    GRADIENT_SUBTLE = (
        "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        " stop:0 #192b4a, stop:1 #0f1d37)"
    )


class ModernFonts:
    """Typography tokens for desktop UI."""

    FAMILY_PRIMARY = "IBM Plex Sans"
    FAMILY_DISPLAY = "Sora SemiBold"
    FAMILY_MONOSPACE = "JetBrains Mono"

    PRIMARY_CANDIDATES = (
        "IBM Plex Sans",
        "Sora",
        "Aptos",
        "Segoe UI Variable Text",
        "Bahnschrift",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "DejaVu Sans",
    )
    DISPLAY_CANDIDATES = (
        "Sora SemiBold",
        "Sora",
        "IBM Plex Sans Medium",
        "Aptos Display",
        "Bahnschrift SemiBold",
        "Segoe UI Semibold",
        "Microsoft YaHei UI",
        "DejaVu Sans",
    )
    MONOSPACE_CANDIDATES = (
        "JetBrains Mono",
        "Cascadia Code",
        "Cascadia Mono",
        "Consolas",
        "DejaVu Sans Mono",
    )

    SIZE_XS = 10
    SIZE_SM = 12
    SIZE_BASE = 13
    SIZE_LG = 15
    SIZE_XL = 18
    SIZE_XXL = 24
    SIZE_HERO = 34

    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700


class ModernSpacing:
    """Spacing scale."""

    XS = 4
    SM = 8
    BASE = 12
    LG = 16
    XL = 20
    XXL = 24
    XXXL = 32


@lru_cache(maxsize=1)
def _font_family_set() -> set[str]:
    """Best-effort query of installed font families."""
    try:
        from PyQt6.QtGui import QFontDatabase

        return {str(name) for name in QFontDatabase.families()}
    except Exception:
        return set()


def _pick_font_family(
    candidates: tuple[str, ...],
    fallback: str,
) -> str:
    available = _font_family_set()
    if not available:
        return str(fallback)
    for name in candidates:
        if str(name) in available:
            return str(name)
    return str(fallback)


@lru_cache(maxsize=1)
def get_primary_font_family() -> str:
    """Get the primary font family for UI text."""
    return _pick_font_family(
        ModernFonts.PRIMARY_CANDIDATES,
        ModernFonts.FAMILY_PRIMARY,
    )


@lru_cache(maxsize=1)
def get_display_font_family() -> str:
    """Get the display font family for headings and prominent text."""
    return _pick_font_family(
        ModernFonts.DISPLAY_CANDIDATES,
        get_primary_font_family(),
    )


@lru_cache(maxsize=1)
def get_monospace_font_family() -> str:
    """Get the monospace font family for code and numeric data."""
    return _pick_font_family(
        ModernFonts.MONOSPACE_CANDIDATES,
        ModernFonts.FAMILY_MONOSPACE,
    )


def get_main_window_style() -> str:
    """Get the stylesheet for the main application window."""
    return f"""
        QMainWindow {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 {ModernColors.BG_PRIMARY},
                stop:0.42 #091629,
                stop:1 {ModernColors.BG_CANVAS}
            );
        }}
        QWidget#AppRoot {{
            background: qradialgradient(
                cx:0.12, cy:0.06, radius:1.2, fx:0.14, fy:0.08,
                stop:0 #1d3d69,
                stop:0.25 #102440,
                stop:0.55 #0b1830,
                stop:1 #050d1d
            );
        }}
    """


def get_central_widget_style() -> str:
    primary_font = get_primary_font_family()
    return f"""
        QWidget {{
            color: {ModernColors.TEXT_PRIMARY};
            font-family: "{primary_font}";
            font-size: {ModernFonts.SIZE_BASE}px;
        }}
        QLabel {{
            background: transparent;
        }}
        QWidget:disabled {{
            color: {ModernColors.TEXT_MUTED};
        }}
    """


def get_card_style() -> str:
    return f"""
        QFrame#cardFrame,
        QFrame#statFrame,
        QFrame#newsGauge,
        QFrame#metricCard,
        QFrame#statusCard {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a3053,
                stop:0.55 #132744,
                stop:1 #10203a
            );
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 16px;
        }}
        QFrame#actionStrip,
        QFrame#chartActionStrip {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #152b4a,
                stop:1 #11233e
            );
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 13px;
            padding: 4px;
        }}
        QFrame#cardFrame:hover,
        QFrame#statFrame:hover,
        QFrame#metricCard:hover,
        QFrame#newsGauge:hover,
        QFrame#chartActionStrip:hover,
        QFrame#actionStrip:hover {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
    """


def get_group_box_style() -> str:
    return f"""
        QGroupBox {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #152a48,
                stop:1 #101f38
            );
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 14px;
            margin-top: 11px;
            padding: 15px 11px 11px 11px;
            font-size: {ModernFonts.SIZE_SM}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            color: {ModernColors.TEXT_SECONDARY};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: -1px;
            padding: 2px 9px;
            background: #10203a;
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 7px;
            color: {ModernColors.ACCENT_INFO};
        }}
    """


def get_button_style(primary: bool = False, danger: bool = False) -> str:
    if primary:
        bg = ModernColors.GRADIENT_PRIMARY
        border = ModernColors.BORDER_FOCUS
        hover_bg = (
            "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #33d4f6, stop:1 #58e4ff)"
        )
    elif danger:
        bg = ModernColors.GRADIENT_SELL
        border = "#ff8595"
        hover_bg = (
            "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #ff6a80, stop:1 #ff8d9d)"
        )
    else:
        bg = (
            "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            " stop:0 #223a63, stop:1 #1a2f54)"
        )
        border = ModernColors.BORDER_DEFAULT
        hover_bg = (
            "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
            " stop:0 #284875, stop:1 #1f3a62)"
        )

    return f"""
        QPushButton {{
            background: {bg};
            color: {ModernColors.TEXT_STRONG};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 6px 12px;
            min-height: 32px;
            font-size: {ModernFonts.SIZE_SM}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QPushButton:hover {{
            background: {hover_bg};
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        QPushButton:pressed {{
            background: #132849;
        }}
        QPushButton:disabled {{
            background: #112238;
            border-color: #243a5b;
            color: {ModernColors.TEXT_MUTED};
        }}
    """


def get_table_style() -> str:
    return f"""
        QTableWidget, QTableView, QListWidget {{
            background: #0f1d34;
            alternate-background-color: #0c1830;
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
            gridline-color: #1f3353;
            selection-background-color: #254f80;
            selection-color: {ModernColors.TEXT_STRONG};
            color: {ModernColors.TEXT_PRIMARY};
            font-size: {ModernFonts.SIZE_SM}px;
            outline: none;
        }}
        QTableWidget::item, QTableView::item, QListWidget::item {{
            padding: 8px 10px;
            border: none;
        }}
        QTableWidget::item:selected, QTableView::item:selected, QListWidget::item:selected {{
            background: #2a578a;
            color: {ModernColors.TEXT_STRONG};
        }}
        QTableWidget::item:hover, QTableView::item:hover, QListWidget::item:hover {{
            background: #1c3961;
        }}
        QHeaderView::section {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #1a3358,
                stop:1 #142845
            );
            color: {ModernColors.TEXT_SECONDARY};
            border: none;
            border-right: 1px solid #274066;
            border-bottom: 1px solid #274066;
            padding: 8px 10px;
            font-size: {ModernFonts.SIZE_XS}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QTableCornerButton::section {{
            background: #173154;
            border: none;
            border-right: 1px solid #274066;
            border-bottom: 1px solid #274066;
        }}
    """


def get_progress_bar_style(color: str = "primary") -> str:
    gradients = {
        "primary": ModernColors.GRADIENT_PRIMARY,
        "success": ModernColors.GRADIENT_BUY,
        "danger": ModernColors.GRADIENT_SELL,
        "warning": ModernColors.GRADIENT_WARNING,
        "accent": ModernColors.GRADIENT_PRIMARY,
    }
    gradient = gradients.get(str(color).lower(), gradients["primary"])
    return f"""
        QProgressBar {{
            background: #112139;
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 8px;
            min-height: 17px;
            color: {ModernColors.TEXT_PRIMARY};
            text-align: center;
            font-size: {ModernFonts.SIZE_XS}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QProgressBar::chunk {{
            background: {gradient};
            border-radius: 7px;
        }}
    """


def get_label_style(
    size: str = "base",
    weight: str = "normal",
    color: str = "primary",
) -> str:
    primary_font = get_primary_font_family()
    sizes = {
        "xs": ModernFonts.SIZE_XS,
        "sm": ModernFonts.SIZE_SM,
        "base": ModernFonts.SIZE_BASE,
        "lg": ModernFonts.SIZE_LG,
        "xl": ModernFonts.SIZE_XL,
        "xxl": ModernFonts.SIZE_XXL,
        "hero": ModernFonts.SIZE_HERO,
    }
    weights = {
        "normal": ModernFonts.WEIGHT_NORMAL,
        "medium": ModernFonts.WEIGHT_MEDIUM,
        "semibold": ModernFonts.WEIGHT_SEMIBOLD,
        "bold": ModernFonts.WEIGHT_BOLD,
    }
    colors_map = {
        "primary": ModernColors.TEXT_PRIMARY,
        "secondary": ModernColors.TEXT_SECONDARY,
        "muted": ModernColors.TEXT_MUTED,
        "success": ModernColors.ACCENT_SUCCESS,
        "warning": ModernColors.ACCENT_WARNING,
        "danger": ModernColors.ACCENT_DANGER,
        "info": ModernColors.ACCENT_INFO,
    }
    size_px = sizes.get(size, ModernFonts.SIZE_BASE)
    font_weight = weights.get(weight, ModernFonts.WEIGHT_NORMAL)
    text_color = colors_map.get(color, ModernColors.TEXT_PRIMARY)
    return (
        "QLabel {"
        f" color: {text_color};"
        f" font-size: {size_px}px;"
        f" font-weight: {font_weight};"
        f' font-family: "{primary_font}";'
        "}"
    )


def get_input_style() -> str:
    mono_font = get_monospace_font_family()
    return f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
            background: #152946;
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 9px;
            padding: 6px 9px;
            color: {ModernColors.TEXT_PRIMARY};
            selection-background-color: #2f689d;
            selection-color: {ModernColors.TEXT_STRONG};
            min-height: 28px;
        }}
        QTextEdit, QPlainTextEdit {{
            min-height: 0;
            font-family: "{mono_font}";
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
            border-color: {ModernColors.BORDER_FOCUS};
            background: #1b3559;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none;
            width: 0px;
            height: 0px;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {ModernColors.TEXT_SECONDARY};
            margin-right: 6px;
        }}
        QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{
            border: none;
            background: transparent;
            width: 16px;
        }}
        QComboBox QAbstractItemView {{
            background: #132845;
            border: 1px solid {ModernColors.BORDER_DEFAULT};
            selection-background-color: #2b5585;
            color: {ModernColors.TEXT_PRIMARY};
        }}
    """


def get_tab_widget_style() -> str:
    return f"""
        QTabWidget::pane {{
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
            background: #101f36;
            top: -1px;
        }}
        QTabBar::tab {{
            background: #142744;
            color: {ModernColors.TEXT_MUTED};
            border: 1px solid transparent;
            border-top-left-radius: 9px;
            border-top-right-radius: 9px;
            padding: 7px 13px;
            margin-right: 5px;
            min-width: 78px;
            font-size: {ModernFonts.SIZE_SM}px;
            font-weight: {ModernFonts.WEIGHT_MEDIUM};
        }}
        QTabBar::tab:selected {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #244b78,
                stop:1 #1a3658
            );
            color: {ModernColors.TEXT_PRIMARY};
            border-color: {ModernColors.BORDER_DEFAULT};
        }}
        QTabBar::tab:hover:!selected {{
            background: #1b3558;
            color: {ModernColors.TEXT_SECONDARY};
        }}
    """


def get_selection_control_style() -> str:
    return f"""
        QCheckBox, QRadioButton {{
            spacing: 7px;
            color: {ModernColors.TEXT_SECONDARY};
            font-size: {ModernFonts.SIZE_SM}px;
            font-weight: {ModernFonts.WEIGHT_MEDIUM};
        }}
        QCheckBox::indicator {{
            width: 16px;
            height: 16px;
            border-radius: 5px;
            border: 1px solid {ModernColors.BORDER_DEFAULT};
            background: #142642;
        }}
        QCheckBox::indicator:hover {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        QCheckBox::indicator:checked {{
            background: {ModernColors.GRADIENT_PRIMARY};
            border: 1px solid {ModernColors.BORDER_FOCUS};
        }}
        QRadioButton::indicator {{
            width: 15px;
            height: 15px;
            border-radius: 7px;
            border: 1px solid {ModernColors.BORDER_DEFAULT};
            background: #142642;
        }}
        QRadioButton::indicator:checked {{
            background: {ModernColors.GRADIENT_PRIMARY};
            border-color: {ModernColors.BORDER_FOCUS};
        }}
    """


def get_scroll_area_style() -> str:
    return """
        QScrollArea {
            background: transparent;
            border: none;
        }
        QScrollBar:vertical {
            background: #0f2038;
            width: 11px;
            margin: 2px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical {
            background: #375983;
            min-height: 30px;
            border-radius: 5px;
        }
        QScrollBar::handle:vertical:hover {
            background: #4e79ad;
        }
        QScrollBar:horizontal {
            background: #0f2038;
            height: 11px;
            margin: 2px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal {
            background: #375983;
            min-width: 28px;
            border-radius: 5px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #4e79ad;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
            width: 0;
            height: 0;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: none;
        }
    """


def get_signal_panel_style(signal: str = "hold") -> str:
    display_font = get_display_font_family()
    color_map = {
        "buy": (ModernColors.SIGNAL_BUY, ModernColors.SIGNAL_BUY_BG),
        "sell": (ModernColors.SIGNAL_SELL, ModernColors.SIGNAL_SELL_BG),
        "hold": (ModernColors.SIGNAL_HOLD, ModernColors.SIGNAL_HOLD_BG),
    }
    color, bg = color_map.get(str(signal).lower(), color_map["hold"])
    return f"""
        QFrame#signalPanelFrame {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 {bg},
                stop:1 #101f38
            );
            border: 2px solid {color};
            border-radius: 14px;
        }}
        QLabel#signalLabel {{
            color: {color};
            font-size: {ModernFonts.SIZE_HERO}px;
            font-weight: {ModernFonts.WEIGHT_BOLD};
            font-family: "{display_font}";
        }}
    """


def get_status_indicator_style(status: str = "healthy") -> str:
    color_map = {
        "healthy": ModernColors.ACCENT_SUCCESS,
        "degraded": ModernColors.ACCENT_WARNING,
        "error": ModernColors.ACCENT_DANGER,
        "unknown": ModernColors.TEXT_MUTED,
    }
    color = color_map.get(str(status).lower(), color_map["unknown"])
    return (
        "QLabel {"
        f" color: {color};"
        f" border: 1px solid {color};"
        " border-radius: 8px;"
        " padding: 3px 8px;"
        f" font-weight: {ModernFonts.WEIGHT_SEMIBOLD};"
        "}"
    )


def get_connection_status_style(connected: bool) -> str:
    if connected:
        color = ModernColors.ACCENT_SUCCESS
    else:
        color = ModernColors.ACCENT_DANGER
    return (
        f"color: {color};"
        f"font-size: {ModernFonts.SIZE_SM}px;"
        f"font-weight: {ModernFonts.WEIGHT_BOLD};"
    )


def get_connection_button_style(connected: bool) -> str:
    if connected:
        gradient = ModernColors.GRADIENT_SELL
        border = "#ff8595"
    else:
        gradient = ModernColors.GRADIENT_BUY
        border = "#4be1af"
    return f"""
        QPushButton {{
            background: {gradient};
            color: {ModernColors.TEXT_STRONG};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 8px 12px;
            min-height: 36px;
            font-size: {ModernFonts.SIZE_BASE}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QPushButton:hover {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        QPushButton:pressed {{
            background: #123154;
        }}
    """


def get_status_badge_style(kind: str) -> str:
    kind_norm = str(kind or "").strip().lower()
    if kind_norm in {"auto", "active", "connected", "running", "success"}:
        color = ModernColors.ACCENT_SUCCESS
        bg = "rgba(54, 211, 164, 0.16)"
        border = "rgba(54, 211, 164, 0.40)"
    elif kind_norm in {"semi-auto", "warning", "hold"}:
        color = ModernColors.ACCENT_WARNING
        bg = "rgba(241, 201, 109, 0.16)"
        border = "rgba(241, 201, 109, 0.36)"
    elif kind_norm in {"error", "danger", "disconnected", "paused"}:
        color = ModernColors.ACCENT_DANGER
        bg = "rgba(255, 108, 128, 0.16)"
        border = "rgba(255, 108, 128, 0.40)"
    else:
        color = ModernColors.ACCENT_INFO
        bg = "rgba(131, 171, 255, 0.16)"
        border = "rgba(131, 171, 255, 0.36)"

    return (
        f"color: {color};"
        f"font-size: {ModernFonts.SIZE_SM}px;"
        f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        "padding: 2px 10px;"
        f"background-color: {bg};"
        f"border: 1px solid {border};"
        "border-radius: 9px;"
    )


def get_app_stylesheet() -> str:
    return "\n".join(
        [
            get_main_window_style(),
            get_central_widget_style(),
            get_card_style(),
            get_group_box_style(),
            get_button_style(),
            get_table_style(),
            get_input_style(),
            get_tab_widget_style(),
            get_selection_control_style(),
            get_scroll_area_style(),
            f"""
            QWidget#leftPanel, QWidget#centerPanel, QWidget#rightPanel {{
                background: transparent;
            }}
            QMenuBar {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #10223d,
                    stop:1 #0c1a31
                );
                border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_PRIMARY};
                padding: 3px 8px;
            }}
            QMenuBar::item {{
                padding: 6px 12px;
                margin: 2px 3px;
                border-radius: 8px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background: #203f67;
            }}
            QMenu {{
                background: #112340;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 10px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 12px;
                border-radius: 7px;
            }}
            QMenu::item:selected {{
                background: #254b77;
            }}
            QToolBar {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f2140,
                    stop:1 #0c1a32
                );
                border: none;
                border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
                spacing: 10px;
                padding: 8px 10px;
            }}
            QToolBar QLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                padding: 0 2px;
                background: transparent;
            }}
            QToolButton {{
                background: #1b3559;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 9px;
                padding: 5px 10px;
                min-height: 28px;
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QToolButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
                background: #234872;
            }}
            QToolButton:checked {{
                background: #2b5687;
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QSplitter::handle {{
                background: #1a3356;
            }}
            QSplitter::handle:hover {{
                background: #2d507d;
            }}
            QStatusBar {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0f2038,
                    stop:1 #0c172b
                );
                border-top: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QStatusBar::item {{
                border: none;
            }}
            QToolTip {{
                background: #1b3356;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 7px;
                padding: 6px 8px;
            }}
            QMessageBox {{
                background: #10213a;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
            }}
            QMessageBox QLabel {{
                background: transparent;
                color: {ModernColors.TEXT_PRIMARY};
                min-width: 260px;
            }}
            QMessageBox QAbstractButton {{
                background: #1c3458;
                color: {ModernColors.TEXT_STRONG};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 9px;
                min-height: 30px;
                min-width: 74px;
                padding: 5px 10px;
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QMessageBox QAbstractButton:hover {{
                background: #285182;
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#primaryActionButton {{
                background: {ModernColors.GRADIENT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_FOCUS};
                color: {ModernColors.TEXT_STRONG};
                min-height: 32px;
                min-width: 90px;
                border-radius: 10px;
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
                padding: 6px 12px;
            }}
            QPushButton#primaryActionButton:hover {{
                border-color: #9be9ff;
            }}
            QPushButton#secondaryActionButton {{
                background: #1a3256;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_PRIMARY};
                min-height: 30px;
                min-width: 84px;
                border-radius: 10px;
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                padding: 5px 10px;
            }}
            QPushButton#secondaryActionButton:hover {{
                background: #214169;
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#dangerActionButton {{
                background: {ModernColors.GRADIENT_SELL};
                border: 1px solid #ff8fa0;
                color: {ModernColors.TEXT_STRONG};
                min-height: 30px;
                min-width: 84px;
                border-radius: 10px;
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                padding: 5px 10px;
            }}
            QPushButton#dangerActionButton:hover {{
                border-color: #ffb5c0;
            }}
            QLabel#metaLabel {{
                color: {ModernColors.TEXT_MUTED};
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QLabel#subtleLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            QLabel#chartLatestLabel {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_XS}px;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                background: #11213c;
                border-radius: 7px;
                padding: 4px 8px;
            }}
            QLabel#monitorLabel {{
                color: {ModernColors.TEXT_MUTED};
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            QLabel#connectionStatus {{
                color: {ModernColors.ACCENT_DANGER};
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QFrame#actionStrip {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #152b4a,
                    stop:1 #10223c
                );
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 12px;
            }}
            QFrame#chartActionStrip {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #122743,
                    stop:1 #0f213a
                );
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 10px;
            }}
            QPushButton#buyButton {{
                background: {ModernColors.GRADIENT_BUY};
                border: 1px solid #44dcaa;
                min-height: 38px;
                font-size: {ModernFonts.SIZE_LG}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QPushButton#buyButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#sellButton {{
                background: {ModernColors.GRADIENT_SELL};
                border: 1px solid #ff8a99;
                min-height: 38px;
                font-size: {ModernFonts.SIZE_LG}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QPushButton#sellButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#chartToolButton {{
                min-height: 26px;
                border-radius: 9px;
                padding: 4px 9px;
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QPushButton#smallGhostButton {{
                background: #152c4b;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 9px;
                min-height: 26px;
                padding: 4px 8px;
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QPushButton#smallGhostButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
                background: #1e3d63;
            }}
            QCheckBox#overlayToggle {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QGroupBox#chartPrimaryGroup {{
                border-color: #2e5f96;
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #153056,
                    stop:1 #0f223d
                );
            }}
            QGroupBox#analysisDetailsGroup,
            QGroupBox#systemLogGroup {{
                border-color: #2a507f;
            }}
            QLabel#sentimentModeLabel {{
                color: {ModernColors.ACCENT_INFO};
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            """,
        ]
    )


def get_dialog_style() -> str:
    """Dialog stylesheet aligned with the main application theme."""
    mono_font = get_monospace_font_family()
    return "\n".join(
        [
            get_central_widget_style(),
            get_card_style(),
            get_group_box_style(),
            get_button_style(),
            get_table_style(),
            get_input_style(),
            get_tab_widget_style(),
            get_selection_control_style(),
            get_scroll_area_style(),
            f"""
            QDialog {{
                background: qradialgradient(
                    cx:0.12, cy:0.08, radius:1.3, fx:0.16, fy:0.10,
                    stop:0 #18355b,
                    stop:0.32 #112744,
                    stop:0.65 #0d1d35,
                    stop:1 #091427
                );
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QMessageBox {{
                background: #0f1f36;
                color: {ModernColors.TEXT_PRIMARY};
            }}
            QMessageBox QLabel {{
                background: transparent;
                color: {ModernColors.TEXT_PRIMARY};
                min-width: 260px;
            }}
            QMessageBox QAbstractButton {{
                background: #1c3458;
                color: {ModernColors.TEXT_STRONG};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 9px;
                min-height: 30px;
                min-width: 74px;
                padding: 5px 10px;
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QMessageBox QAbstractButton:hover {{
                background: #285182;
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QLabel#dialogTitle {{
                color: {ModernColors.ACCENT_INFO};
                font-size: {ModernFonts.SIZE_XL}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
                padding: 2px 0 2px 0;
            }}
            QLabel#dialogSubtitle {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_SM}px;
                padding: 0 0 8px 0;
            }}
            QLabel#dialogStatus {{
                color: {ModernColors.TEXT_PRIMARY};
                font-size: {ModernFonts.SIZE_BASE}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QLabel#dialogMetricValue {{
                color: {ModernColors.ACCENT_INFO};
                font-size: {ModernFonts.SIZE_LG}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QFrame#dialogSection {{
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #132947,
                    stop:1 #101f37
                );
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 12px;
            }}
            QListWidget#dialogStockList {{
                background: #0f1d34;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 10px;
                color: {ModernColors.TEXT_PRIMARY};
                padding: 2px;
            }}
            QListWidget#dialogStockList::item {{
                padding: 6px 10px;
                border-radius: 7px;
                margin: 1px 2px;
            }}
            QListWidget#dialogStockList::item:selected {{
                background: #2d5b90;
                color: {ModernColors.TEXT_STRONG};
            }}
            QDialogButtonBox QPushButton {{
                min-height: 30px;
                min-width: 84px;
            }}
            QPushButton#primaryActionButton {{
                background: {ModernColors.GRADIENT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_FOCUS};
                color: {ModernColors.TEXT_STRONG};
                min-height: 32px;
                min-width: 90px;
                border-radius: 10px;
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
                padding: 6px 12px;
            }}
            QPushButton#primaryActionButton:hover {{
                border-color: #9be9ff;
            }}
            QPushButton#secondaryActionButton {{
                background: #1a3256;
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_PRIMARY};
                min-height: 30px;
                min-width: 84px;
                border-radius: 10px;
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                padding: 5px 10px;
            }}
            QPushButton#secondaryActionButton:hover {{
                background: #214169;
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#dangerActionButton {{
                background: {ModernColors.GRADIENT_SELL};
                border: 1px solid #ff8fa0;
                color: {ModernColors.TEXT_STRONG};
                min-height: 30px;
                min-width: 84px;
                border-radius: 10px;
                font-size: {ModernFonts.SIZE_XS}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                padding: 5px 10px;
            }}
            QPushButton#dangerActionButton:hover {{
                border-color: #ffb5c0;
            }}
            QLabel#dialogHint {{
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            QTextEdit#dialogLog {{
                font-family: "{mono_font}";
                font-size: {ModernFonts.SIZE_SM}px;
            }}
            """,
        ]
    )


def apply_modern_theme(app: Any) -> None:
    """Apply the shared modern stylesheet to a QApplication."""
    app.setStyleSheet(get_app_stylesheet())


def get_signal_color(signal: str) -> str:
    mapping = {
        "STRONG_BUY": ModernColors.ACCENT_SUCCESS,
        "BUY": ModernColors.ACCENT_SUCCESS,
        "HOLD": ModernColors.ACCENT_WARNING,
        "SELL": ModernColors.ACCENT_DANGER,
        "STRONG_SELL": ModernColors.ACCENT_DANGER,
    }
    return mapping.get(str(signal).upper(), ModernColors.TEXT_SECONDARY)


def get_signal_bg(signal: str) -> str:
    mapping = {
        "STRONG_BUY": "rgba(54, 211, 164, 0.16)",
        "BUY": "rgba(54, 211, 164, 0.12)",
        "HOLD": "rgba(241, 201, 109, 0.12)",
        "SELL": "rgba(255, 108, 128, 0.12)",
        "STRONG_SELL": "rgba(255, 108, 128, 0.16)",
    }
    return mapping.get(str(signal).upper(), "transparent")
