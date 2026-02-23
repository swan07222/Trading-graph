"""
Modern Professional UI Theme for Trading Graph

Professional dark theme with modern design principles:
- Consistent color palette
- Smooth gradients
- Professional typography
- High contrast for readability
- Modern spacing and layout
"""

# =============================================================================
# COLOR PALETTE - Professional Dark Theme
# =============================================================================

class ModernColors:
    """Modern professional color palette."""
    
    # Primary Background Colors
    BG_PRIMARY = "#0f172a"      # Deep navy background
    BG_SECONDARY = "#1e293b"    # Card/panel background
    BG_TERTIARY = "#334155"     # Hover/active states
    
    # Accent Colors
    ACCENT_PRIMARY = "#3b82f6"  # Primary blue
    ACCENT_SECONDARY = "#8b5cf6"  # Purple
    ACCENT_SUCCESS = "#10b981"  # Green (profit/up)
    ACCENT_WARNING = "#f59e0b"  # Amber (warning)
    ACCENT_DANGER = "#ef4444"   # Red (loss/down)
    ACCENT_INFO = "#06b6d4"     # Cyan (info)
    
    # Text Colors
    TEXT_PRIMARY = "#f8fafc"    # Primary text
    TEXT_SECONDARY = "#94a3b8"  # Secondary text
    TEXT_MUTED = "#64748b"      # Muted text
    
    # Border & Divider
    BORDER_SUBTLE = "#334155"
    BORDER_DEFAULT = "#475569"
    
    # Signal Colors
    SIGNAL_BUY = "#10b981"
    SIGNAL_BUY_BG = "rgba(16, 185, 129, 0.15)"
    SIGNAL_SELL = "#ef4444"
    SIGNAL_SELL_BG = "rgba(239, 68, 68, 0.15)"
    SIGNAL_HOLD = "#f59e0b"
    SIGNAL_HOLD_BG = "rgba(245, 158, 11, 0.15)"
    
    # Gradients
    GRADIENT_BUY = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669)"
    GRADIENT_SELL = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #ef4444, stop:1 #dc2626)"
    GRADIENT_HOLD = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f59e0b, stop:1 #d97706)"
    GRADIENT_PRIMARY = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3b82f6, stop:1 #2563eb)"
    GRADIENT_ACCENT = "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #8b5cf6, stop:1 #7c3aed)"


# =============================================================================
# TYPOGRAPHY
# =============================================================================

class ModernFonts:
    """Modern typography settings."""
    
    FAMILY_PRIMARY = "Segoe UI"
    FAMILY_MONOSPACE = "Consolas"
    
    SIZE_XS = 9
    SIZE_SM = 11
    SIZE_BASE = 13
    SIZE_LG = 15
    SIZE_XL = 18
    SIZE_XXL = 24
    SIZE_HERO = 36
    
    WEIGHT_NORMAL = 400
    WEIGHT_MEDIUM = 500
    WEIGHT_SEMIBOLD = 600
    WEIGHT_BOLD = 700


# =============================================================================
# SPACING & LAYOUT
# =============================================================================

class ModernSpacing:
    """Modern spacing scale."""
    
    XS = 4
    SM = 8
    BASE = 12
    LG = 16
    XL = 20
    XXL = 24
    XXXL = 32


# =============================================================================
# STYLESHEETS
# =============================================================================

def get_main_window_style() -> str:
    """Main window stylesheet."""
    return f"""
        QMainWindow {{
            background-color: {ModernColors.BG_PRIMARY};
        }}
        QStatusBar {{
            background-color: {ModernColors.BG_SECONDARY};
            color: {ModernColors.TEXT_SECONDARY};
            border-top: 1px solid {ModernColors.BORDER_SUBTLE};
            padding: 4px;
        }}
    """


def get_central_widget_style() -> str:
    """Central widget stylesheet."""
    return f"""
        QWidget {{
            background-color: {ModernColors.BG_PRIMARY};
            color: {ModernColors.TEXT_PRIMARY};
            font-family: {ModernFonts.FAMILY_PRIMARY};
            font-size: {ModernFonts.SIZE_BASE}px;
        }}
    """


def get_card_style() -> str:
    """Card/panel container style."""
    return f"""
        QFrame#cardFrame {{
            background-color: {ModernColors.BG_SECONDARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 12px;
        }}
        QFrame#cardFrame:hover {{
            border-color: {ModernColors.BORDER_DEFAULT};
        }}
    """


def get_group_box_style() -> str:
    """Group box style."""
    return f"""
        QGroupBox {{
            background-color: {ModernColors.BG_SECONDARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 10px;
            margin-top: 16px;
            padding-top: 16px;
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            font-size: {ModernFonts.SIZE_LG}px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 16px;
            padding: 0 12px;
            color: {ModernColors.TEXT_PRIMARY};
        }}
    """


def get_button_style(primary: bool = False, danger: bool = False) -> str:
    """Button styles."""
    if primary:
        return f"""
            QPushButton {{
                background: {ModernColors.GRADIENT_PRIMARY};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                font-size: {ModernFonts.SIZE_BASE}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2563eb, stop:1 #1d4ed8);
            }}
            QPushButton:pressed {{
                background: {ModernColors.ACCENT_PRIMARY};
            }}
            QPushButton:disabled {{
                background: {ModernColors.BG_TERTIARY};
                color: {ModernColors.TEXT_MUTED};
            }}
        """
    elif danger:
        return f"""
            QPushButton {{
                background: {ModernColors.GRADIENT_SELL};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
                font-size: {ModernFonts.SIZE_BASE}px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #dc2626, stop:1 #b91c1c);
            }}
            QPushButton:pressed {{
                background: {ModernColors.ACCENT_DANGER};
            }}
        """
    else:
        return f"""
            QPushButton {{
                background-color: {ModernColors.BG_TERTIARY};
                color: {ModernColors.TEXT_PRIMARY};
                border: 1px solid {ModernColors.BORDER_SUBTLE};
                border-radius: 8px;
                padding: 10px 20px;
                font-weight: {ModernFonts.WEIGHT_MEDIUM};
                font-size: {ModernFonts.SIZE_BASE}px;
            }}
            QPushButton:hover {{
                background-color: {ModernColors.BG_SECONDARY};
                border-color: {ModernColors.BORDER_DEFAULT};
            }}
            QPushButton:pressed {{
                background-color: {ModernColors.BG_PRIMARY};
            }}
        """


def get_table_style() -> str:
    """Table widget style."""
    return f"""
        QTableWidget {{
            background-color: {ModernColors.BG_SECONDARY};
            alternate-background-color: {ModernColors.BG_PRIMARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 8px;
            gridline-color: {ModernColors.BORDER_SUBTLE};
            selection-background-color: {ModernColors.BG_TERTIARY};
            color: {ModernColors.TEXT_PRIMARY};
            font-size: {ModernFonts.SIZE_BASE}px;
        }}
        QTableWidget::item {{
            padding: 8px 12px;
            border: none;
        }}
        QTableWidget::item:selected {{
            background-color: {ModernColors.BG_TERTIARY};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        QHeaderView::section {{
            background-color: {ModernColors.BG_TERTIARY};
            color: {ModernColors.TEXT_SECONDARY};
            padding: 10px 12px;
            border: none;
            border-bottom: 2px solid {ModernColors.BORDER_DEFAULT};
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            font-size: {ModernFonts.SIZE_SM}px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        QTableWidget::item:hover {{
            background-color: {ModernColors.BG_TERTIARY};
        }}
    """


def get_progress_bar_style(color: str = "primary") -> str:
    """Progress bar styles."""
    colors = {
        "primary": ModernColors.GRADIENT_PRIMARY,
        "success": ModernColors.GRADIENT_BUY,
        "danger": ModernColors.GRADIENT_SELL,
        "warning": ModernColors.GRADIENT_HOLD,
        "accent": ModernColors.GRADIENT_ACCENT,
    }
    gradient = colors.get(color, colors["primary"])
    
    return f"""
        QProgressBar {{
            background-color: {ModernColors.BG_TERTIARY};
            border: none;
            border-radius: 8px;
            text-align: center;
            color: {ModernColors.TEXT_PRIMARY};
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            font-size: {ModernFonts.SIZE_SM}px;
            height: 20px;
        }}
        QProgressBar::chunk {{
            background: {gradient};
            border-radius: 7px;
        }}
    """


def get_label_style(size: str = "base", weight: str = "normal", color: str = "primary") -> str:
    """Label styles."""
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
    
    return f"""
        QLabel {{
            color: {colors_map.get(color, ModernColors.TEXT_PRIMARY)};
            font-size: {sizes.get(size, ModernFonts.SIZE_BASE)}px;
            font-weight: {weights.get(weight, ModernFonts.WEIGHT_NORMAL)};
            font-family: {ModernFonts.FAMILY_PRIMARY};
        }}
    """


def get_input_style() -> str:
    """Input field styles."""
    return f"""
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background-color: {ModernColors.BG_PRIMARY};
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 6px;
            padding: 8px 12px;
            color: {ModernColors.TEXT_PRIMARY};
            font-size: {ModernFonts.SIZE_BASE}px;
            font-family: {ModernFonts.FAMILY_PRIMARY};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {ModernColors.ACCENT_PRIMARY};
        }}
        QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border-color: {ModernColors.BORDER_DEFAULT};
        }}
        QComboBox::drop-down {{
            border: none;
            width: 24px;
        }}
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid {ModernColors.TEXT_SECONDARY};
            margin-right: 8px;
        }}
    """


def get_tab_widget_style() -> str:
    """Tab widget style."""
    return f"""
        QTabWidget::pane {{
            border: 1px solid {ModernColors.BORDER_SUBTLE};
            border-radius: 8px;
            background-color: {ModernColors.BG_SECONDARY};
        }}
        QTabBar::tab {{
            background-color: {ModernColors.BG_PRIMARY};
            color: {ModernColors.TEXT_SECONDARY};
            padding: 12px 24px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: {ModernFonts.WEIGHT_MEDIUM};
        }}
        QTabBar::tab:selected {{
            background-color: {ModernColors.BG_SECONDARY};
            color: {ModernColors.TEXT_PRIMARY};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {ModernColors.BG_TERTIARY};
        }}
    """


def get_scroll_area_style() -> str:
    """Scroll area style."""
    return f"""
        QScrollArea {{
            background-color: transparent;
            border: none;
        }}
        QScrollBar:vertical {{
            background-color: {ModernColors.BG_PRIMARY};
            width: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        QScrollBar::handle:vertical {{
            background-color: {ModernColors.BG_TERTIARY};
            border-radius: 6px;
            min-height: 30px;
        }}
        QScrollBar::handle:vertical:hover {{
            background-color: {ModernColors.BORDER_DEFAULT};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar:horizontal {{
            background-color: {ModernColors.BG_PRIMARY};
            height: 12px;
            border-radius: 6px;
            margin: 0;
        }}
        QScrollBar::handle:horizontal {{
            background-color: {ModernColors.BG_TERTIARY};
            border-radius: 6px;
            min-width: 30px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background-color: {ModernColors.BORDER_DEFAULT};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0;
        }}
    """


def get_signal_panel_style(signal: str = "hold") -> str:
    """Signal panel with dynamic coloring."""
    colors = {
        "buy": (ModernColors.SIGNAL_BUY, ModernColors.SIGNAL_BUY_BG),
        "sell": (ModernColors.SIGNAL_SELL, ModernColors.SIGNAL_SELL_BG),
        "hold": (ModernColors.SIGNAL_HOLD, ModernColors.SIGNAL_HOLD_BG),
    }
    color, bg = colors.get(signal.lower(), colors["hold"])
    
    return f"""
        QFrame {{
            background-color: {bg};
            border: 2px solid {color};
            border-radius: 16px;
        }}
        QLabel#signalLabel {{
            color: {color};
            font-size: {ModernFonts.SIZE_HERO}px;
            font-weight: {ModernFonts.WEIGHT_BOLD};
        }}
    """


def get_status_indicator_style(status: str = "healthy") -> str:
    """Status indicator styles."""
    colors = {
        "healthy": ModernColors.ACCENT_SUCCESS,
        "degraded": ModernColors.ACCENT_WARNING,
        "error": ModernColors.ACCENT_DANGER,
        "unknown": ModernColors.TEXT_MUTED,
    }
    color = colors.get(status.lower(), colors["unknown"])
    
    return f"""
        QLabel {{
            color: {color};
            font-weight: {ModernFonts.WEIGHT_SEMIBOLD};
            padding: 4px 12px;
            background-color: {color}20;
            border-radius: 6px;
            border: 1px solid {color};
        }}
    """


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def apply_modern_theme(app) -> None:
    """Apply modern theme to entire application."""
    app.setStyleSheet(f"""
        {get_main_window_style()}
        {get_central_widget_style()}
        {get_card_style()}
        {get_group_box_style()}
        {get_table_style()}
        {get_input_style()}
        {get_tab_widget_style()}
        {get_scroll_area_style()}
    """)


def get_signal_color(signal: str) -> str:
    """Get color for signal type."""
    signal_colors = {
        "STRONG_BUY": ModernColors.ACCENT_SUCCESS,
        "BUY": ModernColors.ACCENT_SUCCESS,
        "HOLD": ModernColors.ACCENT_WARNING,
        "SELL": ModernColors.ACCENT_DANGER,
        "STRONG_SELL": ModernColors.ACCENT_DANGER,
    }
    return signal_colors.get(str(signal).upper(), ModernColors.TEXT_SECONDARY)


def get_signal_bg(signal: str) -> str:
    """Get background color for signal type."""
    signal_bgs = {
        "STRONG_BUY": "rgba(16, 185, 129, 0.2)",
        "BUY": "rgba(16, 185, 129, 0.15)",
        "HOLD": "rgba(245, 158, 11, 0.15)",
        "SELL": "rgba(239, 68, 68, 0.15)",
        "STRONG_SELL": "rgba(239, 68, 68, 0.2)",
    }
    return signal_bgs.get(str(signal).upper(), "transparent")
