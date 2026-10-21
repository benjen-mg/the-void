"""GuestAccessibilityPanel: guest selection, controls, and result
display, through the real Qt widget."""

import os

import numpy as np
import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.data.probe_radii import PROBE_RADII  # noqa: E402
from thevoid_core.services.guest_accessibility import (  # noqa: E402
    AccessibilityStatus,
    Bottleneck,
    GuestAccessibilityResult,
)
from thevoid_core.ui.widgets.guest_accessibility_panel import (  # noqa: E402
    GUEST_NAMES,
    GuestAccessibilityPanel,
    PLACEHOLDER,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _make_result(**overrides):
    defaults = dict(
        guest_name="N2",
        guest_diameter=3.64,
        status=AccessibilityStatus.ACCESSIBLE,
        minimum_window_diameter=4.0,
        maximum_guest_diameter=4.0,
        largest_cavity_diameter=10.0,
        largest_accessible_cavity=9.0,
        bottlenecks=[
            Bottleneck(grid_index=(1, 1, 1), position=np.zeros(3), radius=2.0)
        ],
        suggested_probe_radius=2.0,
        grid_shape=(10, 10, 10),
        accessible_points=np.zeros((5, 3)),
        blocked_points=np.zeros((3, 3)),
    )
    defaults.update(overrides)
    return GuestAccessibilityResult(**defaults)


def test_starts_with_placeholders_and_disabled_toggles(qt_app):
    panel = GuestAccessibilityPanel()

    assert panel.status_value.text() == PLACEHOLDER
    assert not panel.show_accessible_checkbox.isEnabled()
    assert not panel.show_blocked_checkbox.isEnabled()
    assert not panel.show_bottlenecks_checkbox.isEnabled()


def test_guest_combo_lists_all_required_guests_plus_custom(qt_app):
    panel = GuestAccessibilityPanel()

    items = [
        panel.guest_combo.itemText(i)
        for i in range(panel.guest_combo.count())
    ]

    assert items == GUEST_NAMES
    assert set(items) == {"CO2", "CH4", "N2", "H2", "H2O", "Custom"}


def test_default_guest_diameter_is_twice_the_probe_radius(qt_app):
    panel = GuestAccessibilityPanel()
    panel.guest_combo.setCurrentText("CH4")

    assert panel.guest_diameter() == pytest.approx(2 * PROBE_RADII["CH4"])


def test_custom_diameter_box_enabled_only_for_custom(qt_app):
    panel = GuestAccessibilityPanel()

    assert not panel.custom_diameter_box.isEnabled()

    panel.guest_combo.setCurrentText("Custom")

    assert panel.custom_diameter_box.isEnabled()


def test_custom_diameter_spin_arrows_hidden_except_in_custom_mode(qt_app):
    """A named guest (N2, CO2, ...) shows a fixed, non-editable
    diameter - the spin arrows must not be visible in that state, since
    a visible-but-disabled arrow looks like a broken control rather
    than a value that simply isn't editable here. Only Custom, where
    the value genuinely is user-editable, shows them.
    """
    from PySide6.QtWidgets import QAbstractSpinBox

    panel = GuestAccessibilityPanel()

    assert (
        panel.custom_diameter_box.buttonSymbols()
        == QAbstractSpinBox.ButtonSymbols.NoButtons
    )

    panel.guest_combo.setCurrentText("Custom")

    assert (
        panel.custom_diameter_box.buttonSymbols()
        == QAbstractSpinBox.ButtonSymbols.UpDownArrows
    )

    panel.guest_combo.setCurrentText("CO2")

    assert (
        panel.custom_diameter_box.buttonSymbols()
        == QAbstractSpinBox.ButtonSymbols.NoButtons
    )


def test_custom_diameter_is_used_directly_not_doubled(qt_app):
    panel = GuestAccessibilityPanel()
    panel.guest_combo.setCurrentText("Custom")
    panel.custom_diameter_box.setValue(5.5)

    assert panel.guest_diameter() == 5.5


def test_run_button_emits_diameter_and_name(qt_app):
    panel = GuestAccessibilityPanel()
    panel.guest_combo.setCurrentText("H2O")

    received = []
    panel.run_requested.connect(lambda d, n: received.append((d, n)))

    panel.run_button.click()

    assert received == [(2 * PROBE_RADII["H2O"], "H2O")]


def test_set_result_populates_every_field(qt_app):
    panel = GuestAccessibilityPanel()

    panel.set_result(_make_result())

    assert panel.status_value.text() == "Accessible"
    assert panel.limiting_pore_size_value.text() == "4.000 \u00c5"
    assert panel.largest_accessible_cavity_value.text() == "9.000 \u00c5"
    assert panel.suggested_probe_radius_value.text() == "2.000 \u00c5"


def test_set_result_enables_all_three_visualization_toggles(qt_app):
    panel = GuestAccessibilityPanel()

    panel.set_result(_make_result())

    assert panel.show_accessible_checkbox.isEnabled()
    assert panel.show_blocked_checkbox.isEnabled()
    assert panel.show_bottlenecks_checkbox.isEnabled()


def test_no_percolating_pathway_is_reported_clearly(qt_app):
    panel = GuestAccessibilityPanel()

    panel.set_result(
        _make_result(
            status=AccessibilityStatus.PARTIALLY_ACCESSIBLE,
            minimum_window_diameter=None,
            maximum_guest_diameter=None,
            suggested_probe_radius=None,
        )
    )

    assert "none" in panel.limiting_pore_size_value.text().lower()
    assert panel.suggested_probe_radius_value.text() == PLACEHOLDER


def test_status_label_shows_a_distinct_color_per_status(qt_app):
    panel = GuestAccessibilityPanel()

    panel.set_result(_make_result(status=AccessibilityStatus.INACCESSIBLE))
    inaccessible_style = panel.status_value.styleSheet()

    panel.set_result(_make_result(status=AccessibilityStatus.ACCESSIBLE))
    accessible_style = panel.status_value.styleSheet()

    assert inaccessible_style != accessible_style


def test_each_visualization_toggle_emits_independently(qt_app):
    panel = GuestAccessibilityPanel()
    panel.set_result(_make_result())

    accessible_received = []
    blocked_received = []
    bottleneck_received = []
    panel.accessible_visualization_toggled.connect(accessible_received.append)
    panel.blocked_visualization_toggled.connect(blocked_received.append)
    panel.bottleneck_visualization_toggled.connect(bottleneck_received.append)

    panel.show_accessible_checkbox.setChecked(False)
    panel.show_blocked_checkbox.setChecked(True)
    panel.show_bottlenecks_checkbox.setChecked(True)

    assert accessible_received == [False]
    assert blocked_received == [True]
    assert bottleneck_received == [True]


def test_clear_resets_everything(qt_app):
    panel = GuestAccessibilityPanel()
    panel.set_result(_make_result())

    panel.clear()

    assert panel.status_value.text() == PLACEHOLDER
    assert not panel.show_accessible_checkbox.isEnabled()
    assert not panel.show_blocked_checkbox.isEnabled()
    assert not panel.show_bottlenecks_checkbox.isEnabled()
    assert panel.show_accessible_checkbox.isChecked()
    assert not panel.show_blocked_checkbox.isChecked()
    assert not panel.show_bottlenecks_checkbox.isChecked()
