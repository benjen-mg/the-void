"""End-to-end Open Metal Site detection, through the real MainWindow
rather than the service function or the panel in isolation.

Uses the same square-pyramidal Cu structure (4 equatorial + 1 axial O,
deliberately missing the opposite axial position) already hand-
verified at the service level in test_oms_detection_end_to_end.py:
CN=5, asymmetry exactly 0.200, missing direction exactly (0, 0, -1) -
confirmed here to come out identically through the real application's
full load -> analyze -> display pipeline, not just the bare service
function.
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

_D = 2.0 / 15.0
SQUARE_PYRAMIDAL_CU_CIF = f"""\
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
Cu1 Cu 0.5 0.5 0.5 1.0
O1 O {0.5 + _D} 0.5 0.5 1.0
O2 O {0.5 - _D} 0.5 0.5 1.0
O3 O 0.5 {0.5 + _D} 0.5 1.0
O4 O 0.5 {0.5 - _D} 0.5 1.0
O5 O 0.5 0.5 {0.5 + _D} 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def oms_window(qt_app, tmp_path):
    cif_path = tmp_path / "square_pyramidal_cu.cif"
    cif_path.write_text(SQUARE_PYRAMIDAL_CU_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def _run_synchronously(window):
    result = window._run_oms_detection(
        window.current_structure, window.current_site_labels
    )
    window._on_oms_detection_ready(result)
    return result


def test_square_pyramidal_cu_is_detected_end_to_end(oms_window):
    window = oms_window

    result = _run_synchronously(window)

    assert result.oms_count == 1
    site = result.open_sites[0]
    assert site.metal_symbol == "Cu"
    assert site.coordination_number == 5
    assert site.asymmetry_score == pytest.approx(0.2, abs=1e-6)
    assert site.missing_positions[0].direction == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-6
    )


def test_oms_detection_populates_the_panel(oms_window):
    window = oms_window

    _run_synchronously(window)

    assert "1 open metal site" in window.oms_panel.summary_label.text()
    assert window.oms_panel.site_table.rowCount() == 1
    assert window.oms_panel.site_table.item(0, 0).text() == "Cu1"
    assert window.oms_panel.site_table.item(0, 3).text() == "Open"


def test_oms_detection_switches_to_its_tab(oms_window):
    window = oms_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)

    _run_synchronously(window)

    assert window.inspector_tabs.currentWidget() is window.oms_panel


def test_all_oms_are_highlighted_by_default(oms_window):
    window = oms_window

    assert window.viewer._group_highlight_actor is None

    _run_synchronously(window)

    assert window.viewer._group_highlight_actor is not None
    assert window.viewer._oms_marker_actor is not None


def test_unchecking_highlight_all_clears_the_highlight(oms_window):
    window = oms_window

    _run_synchronously(window)
    assert window.viewer._group_highlight_actor is not None

    window.oms_panel.highlight_all_checkbox.setChecked(False)

    assert window.viewer._group_highlight_actor is None
    assert window.viewer._oms_marker_actor is None


def test_inspection_mode_shows_local_environment_for_selected_row(
    oms_window,
):
    window = oms_window

    _run_synchronously(window)

    window.oms_panel.site_table.selectRow(0)

    assert window.oms_panel.detail_metal_value.text() == "Cu"
    assert window.oms_panel.detail_cn_value.text() == "5"
    assert window.oms_panel.neighbor_table.rowCount() == 5
    assert window.viewer._group_highlight_actor is not None
    assert window.viewer._oms_marker_actor is not None


def test_menu_action_switches_tab_and_starts_detection(oms_window):
    window = oms_window

    window.inspector_tabs.setCurrentWidget(window.info_panel)
    window._on_oms_menu_action()

    assert window.inspector_tabs.currentWidget() is window.oms_panel
    assert window._busy is True


def test_loading_a_new_structure_clears_the_panel(oms_window, tmp_path):
    window = oms_window

    _run_synchronously(window)
    assert window.oms_panel.site_table.rowCount() == 1

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

    assert window.oms_panel.site_table.rowCount() == 0
    assert window.current_oms_result is None


def test_highlight_all_covers_every_supercell_copy(oms_window, monkeypatch):
    """Round 49: in a 2x2x2 supercell, "highlight all open metal sites"
    must outline every drawn copy of the open Cu site - all eight -
    not just the reference-cell one while the markers tile around it.
    """
    window = oms_window

    _run_synchronously(window)

    captured = {}
    original = window.viewer.highlight_group

    def spy(atom_indices, color):
        captured["indices"] = list(atom_indices)
        return original(atom_indices, color)

    monkeypatch.setattr(window.viewer, "highlight_group", spy)

    window.supercell = (2, 2, 2)
    window._rebuild_view()

    model = window.current_render_model
    cu_copies = [
        i for i, atom in enumerate(model.atoms) if atom.symbol == "Cu"
    ]

    assert len(cu_copies) == 8
    assert sorted(captured["indices"]) == sorted(cu_copies)


def test_selecting_one_site_still_highlights_a_single_copy(oms_window):
    """Inspection of one row keeps the old single-representative rule
    - the local coordination sphere of one site, not every copy.
    """
    window = oms_window
    _run_synchronously(window)

    window.supercell = (2, 2, 2)
    window._rebuild_view()

    indices = window._render_atom_indices_for_sites([0])
    assert len(indices) == 1
