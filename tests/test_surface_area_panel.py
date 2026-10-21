"""SurfaceAreaPanel: probe selection, controls, and result display,
through the real Qt widget."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.surface_area import (  # noqa: E402
    PROBE_RADII,
    SurfaceAreaResult,
)
from thevoid_core.ui.widgets.surface_area_panel import (  # noqa: E402
    PLACEHOLDER,
    SurfaceAreaPanel,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _make_result(**overrides):
    defaults = dict(
        probe_name="N2",
        probe_radius=1.82,
        n_samples_per_atom=500,
        accessible_surface_area=200.0,
        geometric_surface_area=150.0,
        asa_per_mass=1500.0,
        asa_per_volume=2000.0,
        total_accessible_area=200.0,
        internal_accessible_area=180.0,
        cell_volume=1000.0,
        cell_mass_grams=1e-22,
        sampled_points=np.zeros((10, 3)),
        accessible_mask=np.ones(10, dtype=bool),
    )
    defaults.update(overrides)
    return SurfaceAreaResult(**defaults)


def test_starts_with_placeholders_and_disabled_toggles(qt_app):
    panel = SurfaceAreaPanel()

    assert panel.asa_value.text() == PLACEHOLDER
    assert not panel.show_points_checkbox.isEnabled()
    assert not panel.show_mesh_checkbox.isEnabled()


def test_probe_combo_lists_all_required_probes_plus_custom(qt_app):
    """CH4 joined the shared probe_radii table alongside guest
    accessibility analysis, which surface area analysis picks up
    naturally rather than needing its own separate list.
    """
    panel = SurfaceAreaPanel()

    items = [
        panel.probe_combo.itemText(i)
        for i in range(panel.probe_combo.count())
    ]

    assert set(items) == {"N2", "CO2", "H2", "H2O", "CH4", "Custom"}
    assert items[-1] == "Custom"


def test_default_probe_radius_matches_the_selected_named_probe(qt_app):
    panel = SurfaceAreaPanel()
    panel.probe_combo.setCurrentText("CO2")

    assert panel.probe_radius() == PROBE_RADII["CO2"]


def test_custom_probe_box_is_disabled_until_custom_is_selected(qt_app):
    panel = SurfaceAreaPanel()

    assert not panel.custom_probe_radius_box.isEnabled()

    panel.probe_combo.setCurrentText("Custom")

    assert panel.custom_probe_radius_box.isEnabled()


def test_custom_probe_spin_arrows_hidden_except_in_custom_mode(qt_app):
    """A named probe (N2, CO2, ...) shows a fixed, non-editable
    radius - the spin arrows must not be visible in that state, since
    a visible-but-disabled arrow looks like a broken control rather
    than a value that simply isn't editable here. Only Custom, where
    the value genuinely is user-editable, shows them.
    """
    from PySide6.QtWidgets import QAbstractSpinBox

    panel = SurfaceAreaPanel()

    assert (
        panel.custom_probe_radius_box.buttonSymbols()
        == QAbstractSpinBox.ButtonSymbols.NoButtons
    )

    panel.probe_combo.setCurrentText("Custom")

    assert (
        panel.custom_probe_radius_box.buttonSymbols()
        == QAbstractSpinBox.ButtonSymbols.UpDownArrows
    )

    panel.probe_combo.setCurrentText("N2")

    assert (
        panel.custom_probe_radius_box.buttonSymbols()
        == QAbstractSpinBox.ButtonSymbols.NoButtons
    )


def test_custom_probe_radius_is_used_when_custom_is_selected(qt_app):
    panel = SurfaceAreaPanel()
    panel.probe_combo.setCurrentText("Custom")
    panel.custom_probe_radius_box.setValue(3.3)

    assert panel.probe_radius() == 3.3


def test_run_button_emits_probe_radius_and_name(qt_app):
    panel = SurfaceAreaPanel()
    panel.probe_combo.setCurrentText("H2O")

    received = []
    panel.run_requested.connect(lambda r, n: received.append((r, n)))

    panel.run_button.click()

    assert received == [(PROBE_RADII["H2O"], "H2O")]


def test_set_result_populates_every_field(qt_app):
    panel = SurfaceAreaPanel()

    panel.set_result(_make_result())

    assert panel.asa_value.text() == "200.00 \u00c5\u00b2"
    assert panel.geometric_value.text() == "150.00 \u00c5\u00b2"
    assert panel.asa_per_mass_value.text() == "1500.00 m\u00b2/g"
    assert panel.asa_per_volume_value.text() == "2000.00 m\u00b2/cm\u00b3"
    assert panel.total_accessible_value.text() == "200.00 \u00c5\u00b2"
    assert panel.internal_accessible_value.text() == "180.00 \u00c5\u00b2"


def test_set_result_enables_both_visualization_toggles(qt_app):
    panel = SurfaceAreaPanel()

    panel.set_result(_make_result())

    assert panel.show_points_checkbox.isEnabled()
    assert panel.show_mesh_checkbox.isEnabled()


def test_points_toggle_emits_its_new_state(qt_app):
    panel = SurfaceAreaPanel()
    panel.set_result(_make_result())

    received = []
    panel.points_visualization_toggled.connect(received.append)

    panel.show_points_checkbox.setChecked(False)

    assert received == [False]


def test_mesh_toggle_emits_its_new_state(qt_app):
    panel = SurfaceAreaPanel()
    panel.set_result(_make_result())

    received = []
    panel.mesh_visualization_toggled.connect(received.append)

    panel.show_mesh_checkbox.setChecked(True)

    assert received == [True]


def test_clear_resets_everything(qt_app):
    panel = SurfaceAreaPanel()
    panel.set_result(_make_result())

    panel.clear()

    assert panel.asa_value.text() == PLACEHOLDER
    assert not panel.show_points_checkbox.isEnabled()
    assert not panel.show_mesh_checkbox.isEnabled()
    assert panel.show_points_checkbox.isChecked()
    assert not panel.show_mesh_checkbox.isChecked()
