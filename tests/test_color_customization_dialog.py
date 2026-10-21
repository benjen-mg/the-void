"""ColorCustomizationDialog: per-element color changes applied
immediately, with reset support, through the real Qt widget."""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import QApplication, QColorDialog  # noqa: E402

from thevoid_core.data.element_colors import ElementColorProvider  # noqa: E402
from thevoid_core.ui.widgets.color_customization_dialog import (  # noqa: E402
    ColorCustomizationDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_one_row_per_element(qt_app):
    provider = ElementColorProvider()
    dialog = ColorCustomizationDialog(provider, {"Zn", "O", "C"})

    assert set(dialog._rows.keys()) == {"Zn", "O", "C"}


def test_choosing_a_color_applies_it_immediately(qt_app, monkeypatch):
    provider = ElementColorProvider()
    dialog = ColorCustomizationDialog(provider, {"C"})

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(255, 0, 0)),
    )

    dialog._on_change_clicked("C")

    assert provider.get_color("C") == pytest.approx((1.0, 0.0, 0.0))


def test_choosing_a_color_emits_colors_changed(qt_app, monkeypatch):
    provider = ElementColorProvider()
    dialog = ColorCustomizationDialog(provider, {"C"})

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(255, 0, 0)),
    )

    received = []
    dialog.colors_changed.connect(lambda: received.append(True))

    dialog._on_change_clicked("C")

    assert received == [True]


def test_cancelling_the_color_picker_does_not_change_anything(
    qt_app, monkeypatch
):
    provider = ElementColorProvider()
    original = provider.get_color("C")
    dialog = ColorCustomizationDialog(provider, {"C"})

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor()),  # invalid = cancelled
    )

    received = []
    dialog.colors_changed.connect(lambda: received.append(True))

    dialog._on_change_clicked("C")

    assert provider.get_color("C") == original
    assert received == []


def test_reset_button_reverts_a_single_element(qt_app):
    provider = ElementColorProvider()
    original = provider.get_color("C")
    provider.set_color("C", (1.0, 0.0, 0.0))
    dialog = ColorCustomizationDialog(provider, {"C", "O"})

    dialog._on_reset_clicked("C")

    assert provider.get_color("C") == original
    assert not provider.is_overridden("C")


def test_reset_button_does_not_affect_other_elements(qt_app):
    provider = ElementColorProvider()
    provider.set_color("C", (1.0, 0.0, 0.0))
    provider.set_color("O", (0.0, 1.0, 0.0))
    dialog = ColorCustomizationDialog(provider, {"C", "O"})

    dialog._on_reset_clicked("C")

    assert provider.get_color("O") == (0.0, 1.0, 0.0)


def test_reset_all_clears_every_override(qt_app):
    provider = ElementColorProvider()
    provider.set_color("C", (1.0, 0.0, 0.0))
    provider.set_color("O", (0.0, 1.0, 0.0))
    dialog = ColorCustomizationDialog(provider, {"C", "O"})

    dialog._on_reset_all_clicked()

    assert not provider.is_overridden("C")
    assert not provider.is_overridden("O")


def test_reset_all_emits_colors_changed(qt_app):
    provider = ElementColorProvider()
    provider.set_color("C", (1.0, 0.0, 0.0))
    dialog = ColorCustomizationDialog(provider, {"C"})

    received = []
    dialog.colors_changed.connect(lambda: received.append(True))

    dialog._on_reset_all_clicked()

    assert received == [True]


def test_close_button_accepts_the_dialog(qt_app):
    provider = ElementColorProvider()
    dialog = ColorCustomizationDialog(provider, {"C"})

    received = []
    dialog.accepted.connect(lambda: received.append(True))

    # The close button is the last widget added; find it by walking
    # the dialog's own layout rather than assuming an attribute name,
    # since it isn't kept as a named attribute the way reset_all_button
    # is.
    for child in dialog.findChildren(type(dialog.reset_all_button)):
        if child.text() == "Close":
            child.click()
            break

    assert received == [True]
