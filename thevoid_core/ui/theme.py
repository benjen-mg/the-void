"""A Material Design 3-inspired visual theme for the Void.

Qt Widgets has no native M3 support, so this approximates it with a
QSS stylesheet built from M3's own published "Baseline" light colour
roles (https://m3.material.io/styles/color/roles) - the same palette
Material 3's reference implementation ships with, not an invented
one - applied to the widget types this application actually uses:
dock widgets, tabs, tables, buttons, menus, and form controls.

Genuine M3 also specifies elevation via layered shadows and motion via
animated state transitions; QSS can't express either, so this settles
for M3's flatter surface-tint approach to elevation (a slightly
different background tone per surface level) and skips animation
entirely - a reasonable, honestly-scoped approximation rather than a
pixel-perfect implementation of a spec Qt Widgets was never built for.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    """M3 Baseline light colour roles actually used by this theme."""

    primary: str = "#6750A4"
    on_primary: str = "#FFFFFF"
    primary_container: str = "#EADDFF"
    on_primary_container: str = "#21005E"

    secondary_container: str = "#E8DEF8"
    on_secondary_container: str = "#1D192B"

    surface: str = "#FFFBFE"
    surface_variant: str = "#E7E0EC"
    surface_dim: str = "#F3EDF7"
    on_surface: str = "#1C1B1F"
    on_surface_variant: str = "#49454E"

    outline: str = "#79747E"
    outline_variant: str = "#CAC4D0"

    error: str = "#B3261E"


PALETTE = Palette()


def build_stylesheet(palette: Palette = PALETTE) -> str:
    """The application-wide QSS string. A function rather than a module
    constant so a future dark palette can reuse it - build_stylesheet
    (Palette(surface="#1C1B1F", ...)) - without duplicating every rule.
    """
    p = palette

    return f"""
        QMainWindow, QWidget {{
            background: {p.surface};
            color: {p.on_surface};
            font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
            font-size: 13px;
        }}

        /* Plain labels must blend into whatever container they sit in
           (a group box, a tab pane) rather than painting their own
           opaque surface colour on top of it - without this, every
           label inside a tinted card shows as a mismatched pale patch
           behind the text. */
        QLabel {{
            background: transparent;
        }}

        /* --- Dock widgets: flat surfaces, no heavy 3D frame --- */
        QDockWidget {{
            titlebar-close-icon: none;
            border: none;
        }}
        QDockWidget::title {{
            background: {p.surface_variant};
            color: {p.on_surface_variant};
            padding: 8px 12px;
            font-weight: 600;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }}

        /* --- Tabs: M3 "pill" selected-tab treatment --- */
        QTabWidget::pane {{
            border: 1px solid {p.outline_variant};
            border-radius: 8px;
            background: {p.surface};
            top: -1px;
        }}
        QTabBar::tab {{
            background: transparent;
            color: {p.on_surface_variant};
            padding: 7px 16px;
            margin: 4px 2px 0 2px;
            border-radius: 16px;
        }}
        QTabBar::tab:selected {{
            background: {p.secondary_container};
            color: {p.on_secondary_container};
            font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{
            background: {p.surface_variant};
        }}

        /* --- Toolbar buttons: outlined when off, filled primary when
           checked - the same "which one is active" treatment as the
           selected-tab styling above, since an unstyled QToolButton's
           default checked state is too subtle a change to notice at a
           glance (confirmed to be the actual complaint: no visible
           difference between the Select and View cursor-mode buttons
           until this was added). --- */
        QToolBar {{
            background: {p.surface};
            border: none;
            spacing: 4px;
            padding: 4px;
        }}
        QToolButton {{
            background: transparent;
            color: {p.on_surface_variant};
            border: 1px solid {p.outline_variant};
            border-radius: 8px;
            padding: 6px 14px;
        }}
        QToolButton:hover {{
            background: {p.surface_variant};
        }}
        QToolButton:checked {{
            background: {p.primary};
            color: {p.on_primary};
            border: 1px solid {p.primary};
            font-weight: 600;
        }}
        QToolButton:checked:hover {{
            background: {p.on_primary_container};
        }}

        /* --- Buttons: filled primary, M3 "full" (pill) shape --- */
        QPushButton {{
            background: {p.primary};
            color: {p.on_primary};
            border: none;
            border-radius: 18px;
            padding: 8px 20px;
            font-weight: 600;
        }}
        QPushButton:hover {{
            background: {p.on_primary_container};
        }}
        QPushButton:disabled {{
            background: {p.surface_variant};
            color: {p.outline};
        }}

        /* --- Form controls: rounded, outlined --- */
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {p.surface};
            border: 1px solid {p.outline};
            border-radius: 8px;
            padding: 4px 8px;
            selection-background-color: {p.primary_container};
        }}
        QLineEdit:focus, QComboBox:focus,
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {p.primary};
        }}

        /* --- Group boxes: card-like containers --- */
        QGroupBox {{
            background: {p.surface_dim};
            border: 1px solid {p.outline_variant};
            border-radius: 12px;
            margin-top: 14px;
            padding: 12px 8px 8px 8px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 6px;
            color: {p.on_surface_variant};
        }}

        /* --- Tables and trees --- */
        QTableWidget, QTreeWidget {{
            background: {p.surface};
            alternate-background-color: {p.surface_dim};
            gridline-color: {p.outline_variant};
            border: 1px solid {p.outline_variant};
            border-radius: 8px;
        }}
        QHeaderView::section {{
            background: {p.surface_variant};
            color: {p.on_surface_variant};
            padding: 6px;
            border: none;
            font-weight: 600;
        }}
        QTableWidget::item:selected, QTreeWidget::item:selected {{
            background: {p.primary_container};
            color: {p.on_primary_container};
        }}

        /* --- Menus --- */
        QMenuBar {{
            background: {p.surface};
            border-bottom: 1px solid {p.outline_variant};
        }}
        QMenuBar::item:selected {{
            background: {p.secondary_container};
            border-radius: 6px;
        }}
        QMenu {{
            background: {p.surface};
            border: 1px solid {p.outline_variant};
            border-radius: 8px;
            padding: 4px;
        }}
        QMenu::item {{
            padding: 6px 24px;
            border-radius: 6px;
        }}
        QMenu::item:selected {{
            background: {p.secondary_container};
            color: {p.on_secondary_container};
        }}

        /* --- Status bar --- */
        QStatusBar {{
            background: {p.surface_variant};
            color: {p.on_surface_variant};
        }}

        QToolTip {{
            background: {p.on_surface};
            color: {p.surface};
            border: none;
            border-radius: 6px;
            padding: 6px 10px;
        }}
    """


def apply_theme(app, palette: Palette = PALETTE) -> None:
    """Apply the theme to a QApplication instance."""
    app.setStyleSheet(build_stylesheet(palette))
