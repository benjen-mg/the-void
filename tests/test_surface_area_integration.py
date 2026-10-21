"""End-to-end accessible surface area analysis, through the real
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

from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

OPEN_CELL_CIF = """\
data_test
_cell_length_a 12.0
_cell_length_b 12.0
_cell_length_c 12.0
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
def surface_window(qt_app, tmp_path):
    cif_path = tmp_path / "open_cell.cif"
    cif_path.write_text(OPEN_CELL_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def _run_synchronously(window, probe_radius=None, probe_name=None):
    if probe_radius is None:
        probe_radius = window.surface_area_panel.probe_radius()
    if probe_name is None:
        probe_name = window.surface_area_panel.probe_combo.currentText()

    result = window._run_surface_area_analysis(
        window.current_structure,
        probe_radius,
        probe_name,
        window.viewer.radius_provider,
    )
    window._on_surface_area_analysis_ready(result)

    return result


def test_surface_area_populates_the_panel(surface_window):
    window = surface_window

    result = _run_synchronously(window)

    assert (
        f"{result.accessible_surface_area:.2f}"
        in window.surface_area_panel.asa_value.text()
    )
    assert (
        f"{result.asa_per_mass:.2f}"
        in window.surface_area_panel.asa_per_mass_value.text()
    )


def test_surface_area_switches_to_its_tab(surface_window):
    window = surface_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)

    _run_synchronously(window)

    assert (
        window.inspector_tabs.currentWidget() is window.surface_area_panel
    )


def test_surface_area_shows_points_in_the_3d_view(surface_window):
    window = surface_window

    assert window.viewer._surface_points_actor is None

    _run_synchronously(window)

    assert window.viewer._surface_points_actor is not None


def test_a_fully_open_cell_has_all_internal_accessible_area(surface_window):
    """The same regression check as the service-level test, now through
    the real application: a single atom in an open cell should have its
    entire accessible surface classified as bordering the fully
    percolating network, not split roughly in half by the boundary-case
    artifact this project found and fixed while building this feature.
    """
    window = surface_window

    result = _run_synchronously(window)

    assert result.internal_accessible_area == pytest.approx(
        result.total_accessible_area, rel=0.05
    )


def test_mesh_overlay_is_built_lazily_on_first_toggle(surface_window):
    window = surface_window

    _run_synchronously(window)
    assert window.viewer._surface_mesh_actor is None

    window.surface_area_panel.show_mesh_checkbox.setChecked(True)

    assert window.viewer._surface_mesh_actor is not None


def test_unchecking_points_hides_without_clearing(surface_window):
    window = surface_window

    _run_synchronously(window)
    actor = window.viewer._surface_points_actor

    window.surface_area_panel.show_points_checkbox.setChecked(False)

    assert window.viewer._surface_points_actor is actor
    assert window.viewer._surface_points_actor.GetVisibility() == 0


def test_menu_action_switches_tab_and_starts_analysis(surface_window):
    window = surface_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)
    window._on_surface_area_menu_action()

    assert (
        window.inspector_tabs.currentWidget() is window.surface_area_panel
    )
    assert window._busy is True


def test_custom_probe_is_used_end_to_end(surface_window):
    window = surface_window
    window.surface_area_panel.probe_combo.setCurrentText("Custom")
    window.surface_area_panel.custom_probe_radius_box.setValue(2.0)

    result = _run_synchronously(window)

    assert result.probe_radius == 2.0
    assert result.probe_name == "Custom"


def test_loading_a_new_structure_clears_the_surface_area_panel(
    surface_window, tmp_path
):
    window = surface_window

    _run_synchronously(window)
    assert "\u2013" not in window.surface_area_panel.asa_value.text()

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

    assert window.surface_area_panel.asa_value.text() == "\u2013"
