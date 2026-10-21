"""End-to-end guest accessibility analysis, through the real
MainWindow rather than the service function or the panel in isolation.
"""

import os

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from thevoid_core.services.guest_accessibility import (  # noqa: E402
    AccessibilityStatus,
)
from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

OPEN_CELL_CIF = """\
data_test
_cell_length_a 15.0
_cell_length_b 15.0
_cell_length_c 15.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Zn1 Zn 0.5 0.5 0.5 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def guest_window(qt_app, tmp_path):
    cif_path = tmp_path / "open_cell.cif"
    cif_path.write_text(OPEN_CELL_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def _run_synchronously(window, guest_diameter=None, guest_name=None):
    if guest_diameter is None:
        guest_diameter = window.guest_accessibility_panel.guest_diameter()
    if guest_name is None:
        guest_name = (
            window.guest_accessibility_panel.guest_combo.currentText()
        )

    result = window._run_guest_accessibility_analysis(
        window.current_structure,
        guest_diameter,
        guest_name,
        window.viewer.radius_provider,
    )
    window._on_guest_accessibility_analysis_ready(result)

    return result


def test_guest_accessibility_populates_the_panel(guest_window):
    window = guest_window

    result = _run_synchronously(window)

    panel = window.guest_accessibility_panel
    assert panel.status_value.text() != "\u2013"
    assert (
        f"{result.largest_accessible_cavity:.3f}"
        in panel.largest_accessible_cavity_value.text()
    )


def test_guest_accessibility_switches_to_its_tab(guest_window):
    window = guest_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)

    _run_synchronously(window)

    assert (
        window.inspector_tabs.currentWidget()
        is window.guest_accessibility_panel
    )


def test_a_wide_open_cell_is_accessible_to_every_named_guest(guest_window):
    """A single small atom in a large, mostly-open cell should be
    ACCESSIBLE to every one of the required guest molecules - none of
    them are anywhere near large enough to be blocked by one atom.
    """
    window = guest_window

    for guest_name in ("CO2", "CH4", "N2", "H2", "H2O"):
        window.guest_accessibility_panel.guest_combo.setCurrentText(
            guest_name
        )
        result = _run_synchronously(window)

        assert result.status == AccessibilityStatus.ACCESSIBLE, guest_name


def test_visualization_shows_accessible_channels_by_default(guest_window):
    window = guest_window

    assert window.viewer._guest_accessible_actor is None

    _run_synchronously(window)

    assert window.viewer._guest_accessible_actor is not None
    assert window.viewer._guest_accessible_actor.GetVisibility() == 1


def test_blocked_and_bottleneck_visualizations_are_computed_but_hidden_by_default(
    guest_window,
):
    """Both are still built (so toggling them on later is instant, no
    re-analysis needed) but start hidden, matching the panel's own
    default-unchecked state for these two toggles.
    """
    window = guest_window

    _run_synchronously(window)

    assert window.viewer._guest_blocked_actor is not None
    assert window.viewer._guest_blocked_actor.GetVisibility() == 0
    assert window.viewer._guest_bottleneck_actor is not None
    assert window.viewer._guest_bottleneck_actor.GetVisibility() == 0


def test_toggling_blocked_regions_on_makes_it_visible(guest_window):
    window = guest_window

    _run_synchronously(window)

    window.guest_accessibility_panel.show_blocked_checkbox.setChecked(True)

    assert window.viewer._guest_blocked_actor.GetVisibility() == 1


def test_menu_action_switches_tab_and_starts_analysis(guest_window):
    window = guest_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)
    window._on_guest_accessibility_menu_action()

    assert (
        window.inspector_tabs.currentWidget()
        is window.guest_accessibility_panel
    )
    assert window._busy is True


def test_custom_guest_diameter_is_used_end_to_end(guest_window):
    window = guest_window
    window.guest_accessibility_panel.guest_combo.setCurrentText("Custom")
    window.guest_accessibility_panel.custom_diameter_box.setValue(2.5)

    result = _run_synchronously(window)

    assert result.guest_diameter == 2.5
    assert result.guest_name == "Custom"


def test_loading_a_new_structure_clears_the_panel(guest_window, tmp_path):
    window = guest_window

    _run_synchronously(window)
    assert window.guest_accessibility_panel.status_value.text() != "\u2013"

    other_cif = tmp_path / "other.cif"
    other_cif.write_text(
        """\
data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.5 0.5 0.5 1.0
"""
    )

    payload = window._load_and_analyse(str(other_cif))
    window._on_structure_ready(payload)

    assert window.guest_accessibility_panel.status_value.text() == "\u2013"
