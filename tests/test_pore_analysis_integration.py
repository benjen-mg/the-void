"""End-to-end pore analysis, through the real MainWindow rather than
the service function or the panel in isolation - confirms the menu
action, the panel's run button, the background-worker result path, the
3D visualization, and the visibility toggle all agree once wired
together.
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

#: A single Zn at the centre of a 12 A cubic cell - large open cell,
#: heavily porous, and small enough that the default grid spacing
#: finishes quickly in a test.
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
def pore_window(qt_app, tmp_path):
    cif_path = tmp_path / "open_cell.cif"
    cif_path.write_text(OPEN_CELL_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def _run_synchronously(window, probe_radius=None):
    """Exercises the exact same worker function and result callback the
    real (asynchronous) run_requested/menu-action path uses, without
    actually spinning up a QThreadPool worker - the same pattern used
    throughout this project's other analysis integration tests.
    """
    if probe_radius is None:
        probe_radius = window.pore_analysis_panel.probe_radius()

    result = window._run_pore_analysis(
        window.current_structure, probe_radius, window.viewer.radius_provider
    )
    window._on_pore_analysis_ready(result)

    return result


def test_pore_analysis_populates_the_panel(pore_window):
    window = pore_window

    result = _run_synchronously(window)

    assert f"{result.lcd:.3f}" in window.pore_analysis_panel.lcd_value.text()
    assert window.pore_analysis_panel.void_fraction_value.text() == (
        f"{result.void_fraction_percent:.2f} %"
    )


def test_pore_analysis_switches_to_the_pores_tab(pore_window):
    window = pore_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)

    _run_synchronously(window)

    assert (
        window.inspector_tabs.currentWidget() is window.pore_analysis_panel
    )


def test_pore_analysis_shows_pore_points_in_the_3d_view(pore_window):
    window = pore_window

    assert window.viewer._pore_actor is None

    _run_synchronously(window)

    assert window.viewer._pore_actor is not None
    assert len(window.viewer._last_pore_points) > 0


def test_a_mostly_open_cell_reports_high_void_fraction_and_3d_network(
    pore_window,
):
    """Sanity check on the real pipeline end to end, not just that
    values are present: a large cell with one small atom must report
    high void fraction and a 3D-interconnected accessible network,
    matching the standalone analyze_pores tests for the same kind of
    structure.
    """
    window = pore_window

    result = _run_synchronously(window)

    assert result.void_fraction_percent > 70.0
    assert result.accessible_dimensionality == 3
    assert result.pld is not None


def test_unchecking_visualization_hides_without_clearing(pore_window):
    window = pore_window

    _run_synchronously(window)
    actor = window.viewer._pore_actor

    window.pore_analysis_panel.show_visualization_checkbox.setChecked(False)

    assert window.viewer._pore_actor is actor  # still the same actor
    assert window.viewer._pore_actor.GetVisibility() == 0

    window.pore_analysis_panel.show_visualization_checkbox.setChecked(True)
    assert window.viewer._pore_actor.GetVisibility() == 1


def test_menu_action_switches_tab_and_runs_with_the_panels_probe_radius(
    pore_window,
):
    window = pore_window
    window.pore_analysis_panel.probe_radius_box.setValue(1.5)

    window.inspector_tabs.setCurrentWidget(window.info_panel)
    window._on_pore_menu_action()

    assert (
        window.inspector_tabs.currentWidget() is window.pore_analysis_panel
    )
    # The busy state was entered (a real async worker was started) -
    # confirm it did not error out synchronously and left the UI in a
    # sensible state rather than asserting on the eventual async result,
    # which the other tests already cover via the synchronous path.
    assert window._busy is True


def test_loading_a_new_structure_clears_the_pore_panel(pore_window, tmp_path):
    window = pore_window

    _run_synchronously(window)
    assert "\u2013" not in window.pore_analysis_panel.lcd_value.text()

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

    assert window.pore_analysis_panel.lcd_value.text() == "\u2013"
