"""
Modern UI tokens and shared styles for Trading Graph.

This module centralizes colors, typography, spacing, and reusable style helpers
to keep the desktop UI visually consistent.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class ModernColors:
    """Professional midnight palette with bright, high-contrast accents."""

    # Core surfaces
    BG_CANVAS = "#050a13"
    BG_PRIMARY = "#0b1324"
    BG_SECONDARY = "#121d33"
    BG_TERTIARY = "#1a2a46"
    BG_ELEVATED = "#22375b"

    # Brand and semantic accents
    ACCENT_PRIMARY = "#3fd7ff"
    ACCENT_SECONDARY = "#5ce0b8"
    ACCENT_SUCCESS = "#38d39f"
    ACCENT_WARNING = "#f3c969"
    ACCENT_DANGER = "#ff6f7f"
    ACCENT_INFO = "#7aa9ff"

    # Text
    TEXT_PRIMARY = "#eaf1ff"
    TEXT_SECONDARY = "#b4c4e3"
    TEXT_MUTED = "#7990ba"
    TEXT_STRONG = "#f8fbff"

    # Borders
    BORDER_SUBTLE = "#243a5f"
    BORDER_DEFAULT = "#325485"
    BORDER_FOCUS = "#52defd"

    # Signal colors
    SIGNAL_BUY = ACCENT_SUCCESS
    SIGNAL_BUY_BG = "rgba(56, 211, 159, 0.16)"
    SIGNAL_SELL = ACCENT_DANGER
    SIGNAL_SELL_BG = "rgba(255, 111, 127, 0.16)"
    SIGNAL_HOLD = ACCENT_WARNING
    SIGNAL_HOLD_BG = "rgba(243, 201, 105, 0.14)"

    # Gradients
    GRADIENT_PRIMARY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #25c8ea, stop:1 #3fd7ff)"
    )
    GRADIENT_BUY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #22b786, stop:1 #38d39f)"
    )
    GRADIENT_SELL = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #f35d72, stop:1 #ff6f7f)"
    )
    GRADIENT_WARNING = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #eebd50, stop:1 #f3c969)"
    )
    GRADIENT_SUBTLE = (
        "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        " stop:0 #17253f, stop:1 #0d1830)"
    )


class ModernFonts:
    """Typography tokens for desktop UI."""

    FAMILY_PRIMARY = "Bahnschrift"
    FAMILY_DISPLAY = "Bahnschrift"
    FAMILY_MONOSPACE = "Cascadia Code"

    PRIMARY_CANDIDATES = (
        "Bahnschrift",
        "Segoe UI",
        "Trebuchet MS",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Arial",
        "DejaVu Sans",
    )
    DISPLAY_CANDIDATES = (
        "Bahnschrift SemiBold",
        "Bahnschrift",
        "Segoe UI Semibold",
        "Segoe UI",
        "Microsoft YaHei UI",
        "Microsoft YaHei",
        "Arial",
        "DejaVu Sans",
    )
    MONOSPACE_CANDIDATES = (
        "Cascadia Code",
        "Cascadia Mono",
        "JetBrains Mono",
        "Consolas",
        "Lucida Console",
        "Courier New",
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
    return _pick_font_family(
        ModernFonts.PRIMARY_CANDIDATES,
        ModernFonts.FAMILY_PRIMARY,
    )


@lru_cache(maxsize=1)
def get_display_font_family() -> str:
    return _pick_font_family(
        ModernFonts.DISPLAY_CANDIDATES,
        get_primary_font_family(),
    )


@lru_cache(maxsize=1)
def get_monospace_font_family() -> str:
    return _pick_font_family(
        ModernFonts.MONOSPACE_CANDIDATES,
        ModernFonts.FAMILY_MONOSPACE,
    )


def get_main_window_style() -> str:
    return f"""
        QMainWindow {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 {ModernColors.BG_PRIMARY},
                stop:1 {ModernColors.BG_CANVAS}
            );
        }}
        QWidget#AppRoot {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 #0c1528,
                stop:0.55 #091325,
                stop:1 #060d1b
            );
        }}
    """


def get_central_widget_style() -> str:
    primary_font = get_primary_font_family()
    return f"""
        QWidget {{
            color: {ModernColors.TEXT_PRIMARY};
            background-color: transparent;
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
                stop:0 #182946,
                stop:1 #0f1d35
            );
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 14px;
        }}
        QFrame#actionStrip {{
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:0,
                stop:0 #15243f,
                stop:1 #11203a
            );
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
            padding: 4px;
        }}
        QFrame#cardFrame:hover,
        QFrame#statFrame:hover,
        QFrame#metricCard:hover,
        QFrame#actionStrip:hover {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
    """


def get_group_box_style() -> str:
    return f"""
        QGroupBox {{
            background-color: {ModernColors.BG_SECONDARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 14px;
            margin-top: 12px;
            padding: 18px 12px 12px 12px;
            font-size: {ModernFonts.SIZE_BASE}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            color: {ModernColors.TEXT_SECONDARY};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: 0px;
            padding: 1px 8px;
            background-color: #0f1c33;
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 7px;
            color: {ModernColors.ACCENT_INFO};
        }}
    """


def get_button_style(primary: bool = False, danger: bool = False) -> str:
    if primary:
        bg = ModernColors.GRADIENT_PRIMARY
        border = ModernColors.BORDER_FOCUS
        hover_bg = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2cd0f2, stop:1 #5be3ff)"
    elif danger:
        bg = ModernColors.GRADIENT_SELL
        border = "#ff8492"
        hover_bg = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ff6479, stop:1 #ff8594)"
    else:
        bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #203454, stop:1 #172a46)"
        border = ModernColors.BORDER_DEFAULT
        hover_bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #274166, stop:1 #1c3151)"

    return f"""
        QPushButton {{
            background: {bg};
            color: {ModernColors.TEXT_STRONG};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 8px 16px;
            min-height: 36px;
            font-size: {ModernFonts.SIZE_BASE}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QPushButton:hover {{
            background: {hover_bg};
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        QPushButton:pressed {{
            background: #11213c;
        }}
        QPushButton:disabled {{
            background: #13233f;
            border-color: #243756;
            color: {ModernColors.TEXT_MUTED};
        }}
    """


def get_table_style() -> str:
    return f"""
        QTableWidget, QTableView, QListWidget {{
            background-color: {ModernColors.BG_SECONDARY};
            alternate-background-color: #0f192d;
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
            gridline-color: {ModernColors.BORDER_SUBTLE};
            selection-background-color: #244b78;
            selection-color: {ModernColors.TEXT_STRONG};
            color: {ModernColors.TEXT_PRIMARY};
            font-size: {ModernFonts.SIZE_SM}px;
            outline: none;
        }}
        QTableWidget::item, QTableView::item, QListWidget::item {{
            padding: 8px 10px;
            border: none;
        }}
        QTableWidget::item:hover, QTableView::item:hover, QListWidget::item:hover {{
            background-color: #1d3658;
        }}
        QHeaderView::section {{
            background-color: #162845;
            color: {ModernColors.TEXT_SECONDARY};
            border: none;
            border-right: 1px solid {ModernColors.BORDER_SUBTLE};
            border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
            padding: 8px 10px;
            font-size: {ModernFonts.SIZE_XS}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QTableCornerButton::section {{
            background-color: #162845;
            border: none;
            border-right: 1px solid {ModernColors.BORDER_SUBTLE};
            border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
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
            background: {ModernColors.BG_TERTIARY};
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
            background-color: {ModernColors.BG_TERTIARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 9px;
            padding: 7px 10px;
            color: {ModernColors.TEXT_PRIMARY};
            selection-background-color: #2f689d;
            selection-color: {ModernColors.TEXT_STRONG};
            min-height: 30px;
        }}
        QTextEdit, QPlainTextEdit {{
            min-height: 0;
            font-family: "{mono_font}";
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
            border-color: {ModernColors.BORDER_FOCUS};
            background-color: #203353;
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox QAbstractItemView {{
            background-color: #162947;
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
            background-color: #101d34;
            top: -1px;
        }}
        QTabBar::tab {{
            background: #142440;
            color: {ModernColors.TEXT_MUTED};
            border: 1px solid transparent;
            border-top-left-radius: 9px;
            border-top-right-radius: 9px;
            padding: 9px 16px;
            margin-right: 5px;
            min-width: 84px;
            font-size: {ModernFonts.SIZE_BASE}px;
            font-weight: {ModernFonts.WEIGHT_MEDIUM};
        }}
        QTabBar::tab:selected {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 #1f3f67,
                stop:1 #183150
            );
            color: {ModernColors.TEXT_PRIMARY};
            border-color: {ModernColors.BORDER_DEFAULT};
        }}
        QTabBar::tab:hover:!selected {{
            background: #1b3151;
            color: {ModernColors.TEXT_SECONDARY};
        }}
    """


def get_scroll_area_style() -> str:
    return f"""
        QScrollArea {{
            background: transparent;
            border: none;
        }}
        QScrollBar:vertical {{
            background: #111f37;
            width: 11px;
            margin: 2px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical {{
            background: #33527f;
            min-height: 30px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: #4d78ad;
        }}
        QScrollBar:horizontal {{
            background: #111f37;
            height: 11px;
            margin: 2px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal {{
            background: #33527f;
            min-width: 28px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: #4d78ad;
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            border: none;
            background: none;
            width: 0;
            height: 0;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
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
            background-color: {bg};
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
        border = "#ff8a97"
    else:
        gradient = ModernColors.GRADIENT_BUY
        border = "#43dfa9"
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
            background: #122443;
        }}
    """


def get_status_badge_style(kind: str) -> str:
    kind_norm = str(kind or "").strip().lower()
    if kind_norm in {"auto", "active", "connected", "running", "success"}:
        color = ModernColors.ACCENT_SUCCESS
        bg = "rgba(56, 211, 159, 0.15)"
        border = "rgba(56, 211, 159, 0.38)"
    elif kind_norm in {"semi-auto", "warning", "hold"}:
        color = ModernColors.ACCENT_WARNING
        bg = "rgba(243, 201, 105, 0.15)"
        border = "rgba(243, 201, 105, 0.34)"
    elif kind_norm in {"error", "danger", "disconnected", "paused"}:
        color = ModernColors.ACCENT_DANGER
        bg = "rgba(255, 111, 127, 0.15)"
        border = "rgba(255, 111, 127, 0.36)"
    else:
        color = ModernColors.ACCENT_INFO
        bg = "rgba(122, 169, 255, 0.15)"
        border = "rgba(122, 169, 255, 0.34)"

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
            get_scroll_area_style(),
            f"""
            QWidget#leftPanel, QWidget#centerPanel, QWidget#rightPanel {{
                background: transparent;
            }}
            QMenuBar {{
                background: #0d182d;
                border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_PRIMARY};
                padding: 3px 8px;
            }}
            QMenuBar::item {{
                padding: 6px 12px;
                margin: 2px 3px;
                border-radius: 7px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background: #1a3254;
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
                background: #224067;
            }}
            QToolBar {{
                background: #0d192f;
                border: none;
                border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
                spacing: 10px;
                padding: 7px 10px;
            }}
            QToolButton {{
                background: #1b3152;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 9px;
                padding: 6px 12px;
                min-height: 30px;
                font-size: {ModernFonts.SIZE_BASE}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QToolButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
                background: #22406a;
            }}
            QSplitter::handle {{
                background: #1a2f4e;
            }}
            QSplitter::handle:hover {{
                background: #2b4d79;
            }}
            QStatusBar {{
                background: #0d182d;
                border-top: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QStatusBar::item {{
                border: none;
            }}
            QToolTip {{
                background: #1b3153;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 7px;
                padding: 6px 8px;
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
                color: {ModernColors.TEXT_MUTED};
                font-size: {ModernFonts.SIZE_XS}px;
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
                    stop:0 #15243f,
                    stop:1 #11203a
                );
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 12px;
            }}
            QPushButton#buyButton {{
                background: {ModernColors.GRADIENT_BUY};
                border: 1px solid #43dfa9;
                min-height: 44px;
                font-size: {ModernFonts.SIZE_XL}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QPushButton#buyButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#sellButton {{
                background: {ModernColors.GRADIENT_SELL};
                border: 1px solid #ff8a97;
                min-height: 44px;
                font-size: {ModernFonts.SIZE_XL}px;
                font-weight: {ModernFonts.WEIGHT_BOLD};
            }}
            QPushButton#sellButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
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
        "STRONG_BUY": "rgba(52, 211, 153, 0.15)",
        "BUY": "rgba(52, 211, 153, 0.11)",
        "HOLD": "rgba(246, 192, 75, 0.11)",
        "SELL": "rgba(248, 113, 113, 0.11)",
        "STRONG_SELL": "rgba(248, 113, 113, 0.15)",
    }
    return mapping.get(str(signal).upper(), "transparent")
