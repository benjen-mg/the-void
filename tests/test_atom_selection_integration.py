"""End-to-end atom selection sync, through the real MainWindow rather
than individual widgets in isolation - confirms the tree, the viewer,
and the Selected Atom panel actually agree with each other once wired
together, not just that each piece is individually correct.

The one thing this cannot exercise is a literal mouse click landing on
an atom in a live render window - this sandbox has no GPU, and
CrystalViewer.enable_selection() fails gracefully in exactly the same
already-tested way CrystalViewer.enable_measurement() does under the
same constraint (see test_viewer_measurement.py). Every step past the
initial pick - resolving to an atom, updating the panel, syncing the
tree, syncing back - is real production code exercised through the
real MainWindow.
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

CIF = """\
data_test
_cell_length_a 6.0
_cell_length_b 6.0
_cell_length_c 6.0
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
C1 C 0.10 0.10 0.10 1.0
O1 O 0.20 0.10 0.10 1.0
O2 O 0.30 0.30 0.30 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    # A real, blocking QMessageBox in a script with no running event
    # loop would hang the test suite; there is nothing here that should
    # trigger one, but silence it defensively the way other integration
    # checks in this project already do.
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def loaded_window(qt_app, tmp_path):
    cif_path = tmp_path / "test.cif"
    cif_path.write_text(CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def test_structure_loads_with_expected_labels(loaded_window):
    assert loaded_window.current_site_labels == ["C1", "O1", "O2"]
    assert loaded_window.tree.tree.topLevelItemCount() == 3


def test_selecting_in_3d_updates_the_panel(loaded_window):
    window = loaded_window

    atom = window.viewer.select_atom_by_site_index(1)  # O1
    window._on_atom_selected_in_3d(atom)

    assert window.selected_atom_panel.label_value.text() == "O1"
    assert window.selected_atom_panel.element_value.text() == "O"
    assert window.selected_atom_panel.occupancy_value.text() == "1.000"
    assert (
        window.selected_atom_panel.fractional_value.text()
        == "(0.20000, 0.10000, 0.10000)"
    )
    # Cartesian coordinates for a 6 A cubic cell: 0.20 * 6 = 1.2, etc.
    assert (
        window.selected_atom_panel.cartesian_value.text()
        == "(1.2000, 0.6000, 0.6000)"
    )


def test_selecting_in_3d_syncs_the_tree(loaded_window):
    window = loaded_window

    atom = window.viewer.select_atom_by_site_index(1)  # O1
    window._on_atom_selected_in_3d(atom)

    assert window.tree.tree.currentItem().text(0) == "O1"


def test_selecting_in_the_tree_updates_the_panel(loaded_window):
    window = loaded_window

    window.tree.atom_selected.emit(2)  # O2

    assert window.selected_atom_panel.label_value.text() == "O2"
    assert (
        window.selected_atom_panel.fractional_value.text()
        == "(0.30000, 0.30000, 0.30000)"
    )


def test_selecting_in_the_tree_syncs_the_viewer(loaded_window):
    window = loaded_window

    window.tree.atom_selected.emit(2)  # O2

    selected_index = window.viewer._selected_atom_index
    assert selected_index is not None

    selected_atom = window.viewer._current_model.atoms[selected_index]
    assert selected_atom.site_index == 2


def test_only_one_atom_selected_at_a_time_across_the_whole_app(
    loaded_window,
):
    window = loaded_window

    atom = window.viewer.select_atom_by_site_index(0)
    window._on_atom_selected_in_3d(atom)
    first_actor = window.viewer._selection_actor
    total_after_first = len(window.viewer.plotter.renderer.actors)

    atom = window.viewer.select_atom_by_site_index(1)
    window._on_atom_selected_in_3d(atom)
    second_actor = window.viewer._selection_actor
    total_after_second = len(window.viewer.plotter.renderer.actors)

    # A new selection replaces the highlight actor outright - the
    # previous one is removed, not left behind alongside a new one -
    # so there is exactly one selection highlight in the scene at any
    # time, and the total actor count does not grow between selections.
    assert first_actor is not None
    assert second_actor is not None
    assert second_actor is not first_actor
    assert total_after_second == total_after_first


def test_loading_a_new_structure_clears_the_selected_atom_panel(
    loaded_window, tmp_path
):
    window = loaded_window

    atom = window.viewer.select_atom_by_site_index(0)
    window._on_atom_selected_in_3d(atom)
    assert window.selected_atom_panel.label_value.text() != "\u2013"

    other_cif = tmp_path / "other.cif"
    other_cif.write_text(
        CIF.replace("C1 C 0.10 0.10 0.10 1.0\n", "")
    )

    payload = window._load_and_analyse(str(other_cif))
    window._on_structure_ready(payload)

    assert window.selected_atom_panel.label_value.text() == "\u2013"


