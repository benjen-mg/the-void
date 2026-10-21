"""PoreAnalysisPanel: controls and result display, through the real Qt
widget.
"""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.pore_analysis import PoreAnalysisResult  # noqa: E402
from thevoid_core.ui.widgets.pore_analysis_panel import (  # noqa: E402
    DEFAULT_PROBE_RADIUS,
    PLACEHOLDER,
    PoreAnalysisPanel,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _make_result(**overrides):
    defaults = dict(
        probe_radius=1.2,
        grid_shape=(30, 30, 30),
        lcd=12.5,
        pld=6.3,
        void_fraction_percent=45.2,
        accessible_volume=500.0,
        non_accessible_volume=25.0,
        cell_volume=1200.0,
        accessible_dimensionality=3,
        accessible_points=np.zeros((10, 3)),
    )
    defaults.update(overrides)
    return PoreAnalysisResult(**defaults)


def test_starts_with_placeholders_and_disabled_visualization_toggle(qt_app):
    panel = PoreAnalysisPanel()

    assert panel.lcd_value.text() == PLACEHOLDER
    assert panel.pld_value.text() == PLACEHOLDER
    assert not panel.show_visualization_checkbox.isEnabled()


def test_default_probe_radius_matches_the_documented_default(qt_app):
    panel = PoreAnalysisPanel()

    assert panel.probe_radius() == DEFAULT_PROBE_RADIUS


def test_set_result_populates_every_field(qt_app):
    panel = PoreAnalysisPanel()

    panel.set_result(_make_result())

    assert panel.lcd_value.text() == "12.500 \u00c5"
    assert panel.pld_value.text() == "6.300 \u00c5"
    assert panel.void_fraction_value.text() == "45.20 %"
    assert panel.accessible_volume_value.text() == "500.00 \u00c5\u00b3"
    assert panel.non_accessible_volume_value.text() == "25.00 \u00c5\u00b3"
    assert "3D" in panel.dimensionality_value.text()


def test_set_result_enables_the_visualization_toggle(qt_app):
    panel = PoreAnalysisPanel()

    panel.set_result(_make_result())

    assert panel.show_visualization_checkbox.isEnabled()


def test_no_percolating_pathway_is_reported_clearly(qt_app):
    panel = PoreAnalysisPanel()

    panel.set_result(_make_result(pld=None, accessible_dimensionality=0))

    assert "none" in panel.pld_value.text().lower()
    assert "isolated" in panel.dimensionality_value.text().lower()


def test_empty_structure_lcd_is_reported_as_unbounded(qt_app):
    panel = PoreAnalysisPanel()

    panel.set_result(_make_result(lcd=float("inf")))

    assert "unbounded" in panel.lcd_value.text().lower()


def test_run_button_emits_the_current_probe_radius(qt_app):
    panel = PoreAnalysisPanel()
    panel.probe_radius_box.setValue(2.5)

    received = []
    panel.run_requested.connect(received.append)

    panel.run_button.click()

    assert received == [2.5]


def test_visualization_checkbox_emits_its_new_state(qt_app):
    panel = PoreAnalysisPanel()
    panel.set_result(_make_result())  # enables the checkbox

    received = []
    panel.visualization_toggled.connect(received.append)

    panel.show_visualization_checkbox.setChecked(False)

    assert received == [False]


def test_clear_resets_everything(qt_app):
    panel = PoreAnalysisPanel()
    panel.set_result(_make_result())

    panel.clear()

    assert panel.lcd_value.text() == PLACEHOLDER
    assert panel.pld_value.text() == PLACEHOLDER
    assert not panel.show_visualization_checkbox.isEnabled()
    assert panel.show_visualization_checkbox.isChecked()


def test_setting_a_second_result_replaces_the_first(qt_app):
    panel = PoreAnalysisPanel()

    panel.set_result(_make_result(lcd=10.0))
    panel.set_result(_make_result(lcd=20.0))

    assert panel.lcd_value.text() == "20.000 \u00c5"
