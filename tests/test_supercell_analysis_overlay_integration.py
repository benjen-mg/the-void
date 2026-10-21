"""Supercell-aware analysis overlay redisplay, through the real
MainWindow: a real reported complaint - overlays "only show for the
first cell" when the supercell is enlarged (e.g. to 3x3x3) rather than
extending to match the rest of the displayed atoms. Also covers a
supercell change happening *after* an analysis has already been run -
the overlay must extend to match, not just work correctly if the
analysis happens to run after the supercell was already set.
"""

import os

import numpy as np
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


def test_overlay_is_confined_to_one_cell_at_1x1x1(analysis_window):
    window = analysis_window
    assert window.supercell == (1, 1, 1)

    _run_pore_analysis(window)

    points = window.viewer._last_pore_points
    assert points is not None and len(points) > 0
    # Every point must lie within one 10 A cell (with a little slack
    # for floating point).
    assert np.all(points <= 10.0 + 1e-6)


def test_changing_supercell_after_analysis_extends_the_overlay(
    analysis_window,
):
    """The exact reported behaviour: run an analysis first, *then*
    enlarge the supercell - the overlay must extend to match, not stay
    confined to the original cell just because the analysis ran before
    the supercell was changed.
    """
    window = analysis_window

    _run_pore_analysis(window)
    points_before = window.viewer._last_pore_points
    count_before = len(points_before)

    window.supercell = (3, 3, 3)
    window._rebuild_view()

    points_after = window.viewer._last_pore_points

    assert points_after is not None
    assert len(points_after) == count_before * 27
    # At least some points must now lie well beyond the original cell,
    # in the extended supercell region.
    assert np.any(points_after > 10.0 + 1e-6)


def test_changing_supercell_before_analysis_also_tiles_correctly(
    analysis_window,
):
    window = analysis_window

    window.supercell = (2, 1, 1)
    window._rebuild_view()

    _run_pore_analysis(window)

    points = window.viewer._last_pore_points
    assert np.any(points > 10.0 + 1e-6)
    assert np.any(points <= 10.0 + 1e-6)


def test_reverting_to_1x1x1_shrinks_the_overlay_back(analysis_window):
    window = analysis_window

    _run_pore_analysis(window)
    window.supercell = (2, 2, 2)
    window._rebuild_view()
    assert len(window.viewer._last_pore_points) > 0

    window.supercell = (1, 1, 1)
    window._rebuild_view()

    points = window.viewer._last_pore_points
    assert np.all(points <= 10.0 + 1e-6)


def test_clearing_then_changing_supercell_does_not_bring_the_overlay_back(
    analysis_window,
):
    window = analysis_window

    _run_pore_analysis(window)
    window.clear_analysis_visualization()
    assert window.viewer._pore_actor is None

    window.supercell = (3, 3, 3)
    window._rebuild_view()

    assert window.viewer._pore_actor is None


def test_an_unrelated_display_change_also_redisplays_the_overlay(
    analysis_window,
):
    """Not just supercell changes - any rebuild (a display style
    change, toggling hydrogens, ...) goes through the same
    render_model() reset, so the same redisplay must happen for those
    too.
    """
    window = analysis_window

    _run_pore_analysis(window)

    window.show_hydrogens = not window.show_hydrogens
    window._rebuild_view()

    assert window.viewer._pore_actor is not None
    assert len(window.viewer._last_pore_points) > 0


def test_loading_a_new_structure_prevents_stale_redisplay(
    analysis_window, tmp_path
):
    window = analysis_window

    _run_pore_analysis(window)
    assert window.current_analysis_visualization == "pore"

    other_cif = tmp_path / "other.cif"
    other_cif.write_text(
        """\
data_test
_cell_length_a 8.0
_cell_length_b 8.0
_cell_length_c 8.0
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

    assert window.current_analysis_visualization is None
    assert window.current_pore_result is None


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


def test_surface_area_accessible_mask_stays_aligned_after_tiling(
    analysis_window,
):
    """The trickiest part of this feature: accessible_mask must stay
    correctly aligned with sampled_points after tiling, not just have
    the right total length - every one of a given original point's 8
    tiled copies (2x2x2) must carry that same original point's
    accessible/blocked classification, confirmed by checking against
    the untiled result directly rather than only checking counts.
    """
    window = analysis_window

    probe_radius = window.surface_area_panel.probe_radius()
    probe_name = window.surface_area_panel.probe_combo.currentText()
    original_result = window._run_surface_area_analysis(
        window.current_structure,
        probe_radius,
        probe_name,
        window.viewer.radius_provider,
    )

    window.supercell = (2, 2, 2)
    window._rebuild_view()
    window._on_surface_area_analysis_ready(original_result)

    tiled_points = window.viewer._last_surface_points
    tiled_mask = window.viewer._last_surface_accessible_mask

    n_original = len(original_result.sampled_points)
    assert len(tiled_points) == n_original * 8
    assert len(tiled_mask) == n_original * 8

    # Reshape assuming the documented ordering (each original point's
    # 8 images consecutive) and confirm every image matches its
    # original point's own accessible/blocked classification.
    reshaped_mask = tiled_mask.reshape(n_original, 8)
    for i in range(n_original):
        assert np.all(reshaped_mask[i] == original_result.accessible_mask[i])
