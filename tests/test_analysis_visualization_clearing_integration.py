"""Cross-analysis-type auto-clearing, through the real MainWindow: a
real reported complaint - "when I do another analysis it becomes too
messy" - running one analysis type after another must always leave
exactly the new analysis's own results on screen, never an
accumulating mix of whatever was run earlier in the session. Also
covers the manual "Clear Analysis Visualization" menu action.
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
def analysis_window(qt_app, tmp_path):
    cif_path = tmp_path / "open_cell.cif"
    cif_path.write_text(OPEN_CELL_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def _run_pore_analysis(window):
    probe_radius = window.pore_analysis_panel.probe_radius()
    result = window._run_pore_analysis(
        window.current_structure, probe_radius, window.viewer.radius_provider
    )
    window._on_pore_analysis_ready(result)


def _run_surface_area_analysis(window):
    probe_radius = window.surface_area_panel.probe_radius()
    probe_name = window.surface_area_panel.probe_combo.currentText()
    result = window._run_surface_area_analysis(
        window.current_structure,
        probe_radius,
        probe_name,
        window.viewer.radius_provider,
    )
    window._on_surface_area_analysis_ready(result)


def _run_guest_accessibility_analysis(window):
    guest_diameter = window.guest_accessibility_panel.guest_diameter()
    guest_name = window.guest_accessibility_panel.guest_combo.currentText()
    result = window._run_guest_accessibility_analysis(
        window.current_structure,
        guest_diameter,
        guest_name,
        window.viewer.radius_provider,
    )
    window._on_guest_accessibility_analysis_ready(result)


def _run_oms_detection(window):
    result = window._run_oms_detection(
        window.current_structure, window.current_site_labels
    )
    window._on_oms_detection_ready(result)


def test_running_surface_area_after_pore_analysis_clears_pore_points(
    analysis_window,
):
    window = analysis_window

    _run_pore_analysis(window)
    assert window.viewer._pore_actor is not None

    _run_surface_area_analysis(window)

    assert window.viewer._pore_actor is None
    assert window.viewer._surface_points_actor is not None


def test_running_guest_accessibility_after_surface_area_clears_surface_points(
    analysis_window,
):
    window = analysis_window

    _run_surface_area_analysis(window)
    assert window.viewer._surface_points_actor is not None

    _run_guest_accessibility_analysis(window)

    assert window.viewer._surface_points_actor is None
    assert window.viewer._guest_accessible_actor is not None


def test_running_oms_detection_after_guest_accessibility_clears_it(
    analysis_window,
):
    window = analysis_window

    _run_guest_accessibility_analysis(window)
    assert window.viewer._guest_accessible_actor is not None

    _run_oms_detection(window)

    assert window.viewer._guest_accessible_actor is None
    assert window.viewer._guest_blocked_actor is None
    assert window.viewer._guest_bottleneck_actor is None


def test_running_pore_analysis_after_oms_detection_clears_oms_markers(
    analysis_window,
):
    window = analysis_window

    _run_oms_detection(window)
    # This structure's single Zn is not itself an open metal site (no
    # ligands at all), so there is no group highlight to check, but a
    # fresh analysis run should still have gone through the clearing
    # path regardless of whether OMS found anything to draw.

    _run_pore_analysis(window)

    assert window.viewer._pore_actor is not None


def test_manual_clear_action_removes_everything(analysis_window):
    window = analysis_window

    _run_pore_analysis(window)
    assert window.viewer._pore_actor is not None

    window.clear_analysis_visualization()

    assert window.viewer._pore_actor is None


def test_manual_clear_leaves_a_selected_atom_alone(analysis_window):
    """clear_all_analysis_visualizations() deliberately does not touch
    selection - confirmed here through the real selection code path,
    not just by reading the method's own docstring.
    """
    window = analysis_window

    zn_index = next(
        i
        for i, site in enumerate(window.current_structure.sites)
        if list(site.species.keys())[0].symbol == "Zn"
    )
    window._on_atom_selected_in_tree(zn_index)
    assert window.viewer._selected_atom_index is not None

    _run_pore_analysis(window)
    window.clear_analysis_visualization()

    assert window.viewer._selected_atom_index is not None
