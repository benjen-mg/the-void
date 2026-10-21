"""End-to-end atom color customization, through the real MainWindow's
actual menu-triggered path.
"""

import os

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtGui import QColor  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QColorDialog,
    QMessageBox,
)

from thevoid_core.ui.widgets.color_customization_dialog import (  # noqa: E402
    ColorCustomizationDialog,
)
from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

# An openly licensed Zn MOF (the tests recolour its Zn).
SAMPLE_CIF = str(ZN_DIA)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def sample_window(qt_app):
    window = MainWindow()

    payload = window._load_and_analyse(SAMPLE_CIF)
    window._on_structure_ready(payload)

    yield window

    window.close()


def test_changing_a_color_through_the_real_dialog_updates_the_provider(
    sample_window, monkeypatch
):
    window = sample_window

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(255, 0, 0)),
    )

    def fake_exec(self):
        self._on_change_clicked("Zn")
        return 0

    monkeypatch.setattr(ColorCustomizationDialog, "exec", fake_exec)

    window._on_customize_colors_menu_action()

    assert window.viewer.color_provider.get_color("Zn") == pytest.approx(
        (1.0, 0.0, 0.0)
    )


def test_changing_a_color_updates_the_legend_panel(
    sample_window, monkeypatch
):
    window = sample_window

    monkeypatch.setattr(
        QColorDialog,
        "getColor",
        staticmethod(lambda *a, **k: QColor(0, 255, 0)),
    )

    def fake_exec(self):
        self._on_change_clicked("Zn")
        return 0

    monkeypatch.setattr(ColorCustomizationDialog, "exec", fake_exec)

    window._on_customize_colors_menu_action()

    assert len(window.element_legend._rows) > 0


def test_no_structure_loaded_shows_a_message(qt_app, monkeypatch):
    window = MainWindow()

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: shown.append(a)),
    )

    window._on_customize_colors_menu_action()

    assert len(shown) == 1
    window.close()
