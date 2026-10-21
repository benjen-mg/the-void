"""Theme module: the QSS gets applied without raising, and the
important structural rules (no opaque label backgrounds hiding inside
tinted containers) are present.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.ui.theme import Palette, apply_theme, build_stylesheet  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_build_stylesheet_returns_non_empty_css():
    css = build_stylesheet()

    assert isinstance(css, str)
    assert len(css) > 100


def test_apply_theme_sets_the_application_stylesheet(qt_app):
    apply_theme(qt_app)

    assert qt_app.styleSheet() != ""


def test_labels_are_transparent_so_they_blend_into_tinted_containers():
    """Regression guard for a real rendered bug: without this rule,
    every QLabel painted its own opaque surface colour on top of
    whatever card or tab pane contained it, showing as a mismatched
    pale rectangle behind the text - caught by rendering and looking
    at it, not just reading the stylesheet.
    """
    css = build_stylesheet()

    assert "QLabel" in css
    assert "background: transparent" in css


def test_custom_palette_colours_appear_in_the_stylesheet():
    palette = Palette(primary="#123456")

    css = build_stylesheet(palette)

    assert "#123456" in css


def test_default_palette_matches_m3_baseline_primary():
    assert Palette().primary == "#6750A4"
