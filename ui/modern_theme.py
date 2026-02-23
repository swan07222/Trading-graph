"""
Modern UI tokens and shared styles for Trading Graph.

This module centralizes colors, typography, spacing, and reusable style helpers
to keep the desktop UI visually consistent.
"""

from __future__ import annotations

from typing import Any


class ModernColors:
    """Professional dark-blue palette with high readability."""

    # Core surfaces
    BG_CANVAS = "#050914"
    BG_PRIMARY = "#0a1020"
    BG_SECONDARY = "#111a2d"
    BG_TERTIARY = "#1b2840"
    BG_ELEVATED = "#223353"

    # Brand and semantic accents
    ACCENT_PRIMARY = "#3ec1e7"
    ACCENT_SECONDARY = "#56d6ba"
    ACCENT_SUCCESS = "#34d399"
    ACCENT_WARNING = "#f6c04b"
    ACCENT_DANGER = "#f87171"
    ACCENT_INFO = "#60a5fa"

    # Text
    TEXT_PRIMARY = "#e7eefc"
    TEXT_SECONDARY = "#a9b8d3"
    TEXT_MUTED = "#6f83a8"
    TEXT_STRONG = "#f5f9ff"

    # Borders
    BORDER_SUBTLE = "#243756"
    BORDER_DEFAULT = "#2f466c"
    BORDER_FOCUS = "#46c8ec"

    # Signal colors
    SIGNAL_BUY = ACCENT_SUCCESS
    SIGNAL_BUY_BG = "rgba(52, 211, 153, 0.12)"
    SIGNAL_SELL = ACCENT_DANGER
    SIGNAL_SELL_BG = "rgba(248, 113, 113, 0.12)"
    SIGNAL_HOLD = ACCENT_WARNING
    SIGNAL_HOLD_BG = "rgba(246, 192, 75, 0.12)"

    # Gradients
    GRADIENT_PRIMARY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #2ba8d1, stop:1 #3ec1e7)"
    )
    GRADIENT_BUY = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #21b88e, stop:1 #34d399)"
    )
    GRADIENT_SELL = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #f05d68, stop:1 #f87171)"
    )
    GRADIENT_WARNING = (
        "qlineargradient(x1:0, y1:0, x2:1, y2:0,"
        " stop:0 #f5b640, stop:1 #f6c04b)"
    )
    GRADIENT_SUBTLE = (
        "qlineargradient(x1:0, y1:0, x2:0, y2:1,"
        " stop:0 #142038, stop:1 #0d1528)"
    )


class ModernFonts:
    """Typography tokens for desktop UI."""

    FAMILY_PRIMARY = "Segoe UI Variable Text"
    FAMILY_DISPLAY = "Segoe UI Variable Display"
    FAMILY_MONOSPACE = "JetBrains Mono"

    SIZE_XS = 10
    SIZE_SM = 11
    SIZE_BASE = 12
    SIZE_LG = 14
    SIZE_XL = 16
    SIZE_XXL = 22
    SIZE_HERO = 30

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


def get_main_window_style() -> str:
    return f"""
        QMainWindow {{
            background: qlineargradient(
                x1:0, y1:0, x2:0, y2:1,
                stop:0 {ModernColors.BG_PRIMARY},
                stop:1 {ModernColors.BG_CANVAS}
            );
        }}
    """


def get_central_widget_style() -> str:
    return f"""
        QWidget {{
            color: {ModernColors.TEXT_PRIMARY};
            background-color: {ModernColors.BG_PRIMARY};
            font-family: {ModernFonts.FAMILY_PRIMARY};
            font-size: {ModernFonts.SIZE_BASE}px;
        }}
        QWidget:disabled {{
            color: {ModernColors.TEXT_MUTED};
        }}
    """


def get_card_style() -> str:
    return f"""
        QFrame#cardFrame,
        QFrame#statFrame,
        QFrame#actionStrip,
        QFrame#newsGauge,
        QFrame#metricCard,
        QFrame#statusCard {{
            background: {ModernColors.GRADIENT_SUBTLE};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
        }}
        QFrame#cardFrame:hover,
        QFrame#statFrame:hover,
        QFrame#metricCard:hover {{
            border-color: {ModernColors.BORDER_DEFAULT};
        }}
    """


def get_group_box_style() -> str:
    return f"""
        QGroupBox {{
            background-color: {ModernColors.BG_SECONDARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
            margin-top: 14px;
            padding-top: 12px;
            font-size: {ModernFonts.SIZE_SM}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            color: {ModernColors.TEXT_SECONDARY};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 12px;
            top: -2px;
            padding: 0 8px;
            color: {ModernColors.ACCENT_PRIMARY};
        }}
    """


def get_button_style(primary: bool = False, danger: bool = False) -> str:
    if primary:
        bg = ModernColors.GRADIENT_PRIMARY
        border = ModernColors.BORDER_FOCUS
    elif danger:
        bg = ModernColors.GRADIENT_SELL
        border = "#f26c76"
    else:
        bg = "qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #1a2b47, stop:1 #14233d)"
        border = ModernColors.BORDER_DEFAULT

    return f"""
        QPushButton {{
            background: {bg};
            color: {ModernColors.TEXT_STRONG};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 9px 16px;
            min-height: 34px;
            font-size: {ModernFonts.SIZE_BASE}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QPushButton:hover {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        QPushButton:pressed {{
            background: #10203a;
        }}
        QPushButton:disabled {{
            background: #17233b;
            border-color: #243756;
            color: {ModernColors.TEXT_MUTED};
        }}
    """


def get_table_style() -> str:
    return f"""
        QTableWidget, QTableView, QListWidget {{
            background-color: {ModernColors.BG_SECONDARY};
            alternate-background-color: {ModernColors.BG_PRIMARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 10px;
            gridline-color: {ModernColors.BORDER_SUBTLE};
            selection-background-color: #1f3d63;
            selection-color: {ModernColors.TEXT_STRONG};
            color: {ModernColors.TEXT_PRIMARY};
            font-size: {ModernFonts.SIZE_SM}px;
            outline: none;
        }}
        QTableWidget::item, QTableView::item, QListWidget::item {{
            padding: 7px 10px;
            border: none;
        }}
        QTableWidget::item:hover, QTableView::item:hover, QListWidget::item:hover {{
            background-color: #1a2e4a;
        }}
        QHeaderView::section {{
            background-color: #15243c;
            color: {ModernColors.TEXT_SECONDARY};
            border: none;
            border-right: 1px solid {ModernColors.BORDER_SUBTLE};
            border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
            padding: 8px 10px;
            font-size: {ModernFonts.SIZE_XS}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
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
            border-radius: 7px;
            min-height: 16px;
            color: {ModernColors.TEXT_PRIMARY};
            text-align: center;
            font-size: {ModernFonts.SIZE_XS}px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
        }}
        QProgressBar::chunk {{
            background: {gradient};
            border-radius: 6px;
        }}
    """


def get_label_style(
    size: str = "base",
    weight: str = "normal",
    color: str = "primary",
) -> str:
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
        f" font-family: {ModernFonts.FAMILY_PRIMARY};"
        "}"
    )


def get_input_style() -> str:
    return f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
            background-color: {ModernColors.BG_TERTIARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 8px;
            padding: 6px 10px;
            color: {ModernColors.TEXT_PRIMARY};
            selection-background-color: #2c6b9d;
            selection-color: {ModernColors.TEXT_STRONG};
            min-height: 28px;
        }}
        QTextEdit, QPlainTextEdit {{
            min-height: 0;
            font-family: {ModernFonts.FAMILY_MONOSPACE};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextEdit:focus {{
            border-color: {ModernColors.BORDER_FOCUS};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {ModernColors.TEXT_SECONDARY};
            margin-right: 8px;
        }}
    """


def get_tab_widget_style() -> str:
    return f"""
        QTabWidget::pane {{
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 10px;
            background-color: {ModernColors.BG_SECONDARY};
            top: -1px;
        }}
        QTabBar::tab {{
            background: #15243c;
            color: {ModernColors.TEXT_MUTED};
            border: 1px solid transparent;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 8px 14px;
            margin-right: 4px;
            min-width: 84px;
            font-size: {ModernFonts.SIZE_SM}px;
            font-weight: {ModernFonts.WEIGHT_MEDIUM};
        }}
        QTabBar::tab:selected {{
            background: #1d3150;
            color: {ModernColors.TEXT_PRIMARY};
            border-color: {ModernColors.BORDER_DEFAULT};
        }}
        QTabBar::tab:hover:!selected {{
            background: #1b2b46;
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
            background: {ModernColors.BG_SECONDARY};
            width: 12px;
            margin: 2px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background: {ModernColors.BORDER_DEFAULT};
            min-height: 28px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {ModernColors.ACCENT_INFO};
        }}
        QScrollBar:horizontal {{
            background: {ModernColors.BG_SECONDARY};
            height: 12px;
            margin: 2px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal {{
            background: {ModernColors.BORDER_DEFAULT};
            min-width: 28px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {ModernColors.ACCENT_INFO};
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
            font-family: {ModernFonts.FAMILY_DISPLAY};
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
        border = "#ef6f7a"
    else:
        gradient = ModernColors.GRADIENT_BUY
        border = "#2ec695"
    return f"""
        QPushButton {{
            background: {gradient};
            color: {ModernColors.TEXT_STRONG};
            border: 1px solid {border};
            border-radius: 10px;
            padding: 9px 12px;
            min-height: 34px;
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
    elif kind_norm in {"semi-auto", "warning", "hold"}:
        color = ModernColors.ACCENT_WARNING
    elif kind_norm in {"error", "danger", "disconnected", "paused"}:
        color = ModernColors.ACCENT_DANGER
    else:
        color = ModernColors.ACCENT_INFO

    return (
        f"color: {color};"
        f"font-size: {ModernFonts.SIZE_SM}px;"
        f"font-weight: {ModernFonts.WEIGHT_BOLD};"
        "padding: 0 8px;"
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
            QMenuBar {{
                background: #0f1a2e;
                border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_PRIMARY};
                padding: 2px 8px;
            }}
            QMenuBar::item {{
                padding: 6px 12px;
                margin: 2px 3px;
                border-radius: 6px;
                background: transparent;
            }}
            QMenuBar::item:selected {{
                background: #1a2d49;
            }}
            QMenu {{
                background: #11203a;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 8px;
                padding: 6px;
            }}
            QMenu::item {{
                padding: 8px 12px;
                border-radius: 6px;
            }}
            QMenu::item:selected {{
                background: #1e3659;
            }}
            QToolBar {{
                background: #0f1a2e;
                border: none;
                border-bottom: 1px solid {ModernColors.BORDER_SUBTLE};
                spacing: 8px;
                padding: 6px 10px;
            }}
            QToolButton {{
                background: #182944;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 8px;
                padding: 6px 12px;
                min-height: 28px;
                font-size: {ModernFonts.SIZE_SM}px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            }}
            QToolButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
                background: #1e3455;
            }}
            QSplitter::handle {{
                background: #1b2b46;
            }}
            QSplitter::handle:hover {{
                background: #2a4165;
            }}
            QStatusBar {{
                background: #0f1a2e;
                border-top: 1px solid {ModernColors.BORDER_SUBTLE};
                color: {ModernColors.TEXT_SECONDARY};
                font-size: {ModernFonts.SIZE_XS}px;
            }}
            QStatusBar::item {{
                border: none;
            }}
            QToolTip {{
                background: #1b2b46;
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_DEFAULT};
                border-radius: 6px;
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
                background: transparent;
                border: none;
            }}
            QPushButton#buyButton {{
                background: {ModernColors.GRADIENT_BUY};
                border: 1px solid #2ec695;
            }}
            QPushButton#buyButton:hover {{
                border-color: {ModernColors.BORDER_FOCUS};
            }}
            QPushButton#sellButton {{
                background: {ModernColors.GRADIENT_SELL};
                border: 1px solid #ef6f7a;
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

