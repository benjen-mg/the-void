"""End-to-end coordination environment analysis, through the real
MainWindow rather than the service function in isolation - confirms
atom selection, CrystalNN analysis, panel population, and coordination
sphere highlighting all agree once wired together.

Uses a rock-salt NaCl-like structure: Na octahedrally coordinated by 6
Cl at exactly a/2, an unambiguous textbook answer to check the real
application against rather than trusting the pipeline blind.
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

ROCK_SALT_CIF = """\
data_test
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
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
Na1 Na 0.0 0.0 0.0 1.0
Na2 Na 0.5 0.5 0.0 1.0
Na3 Na 0.5 0.0 0.5 1.0
Na4 Na 0.0 0.5 0.5 1.0
Cl1 Cl 0.5 0.5 0.5 1.0
Cl2 Cl 0.0 0.0 0.5 1.0
Cl3 Cl 0.0 0.5 0.0 1.0
Cl4 Cl 0.5 0.0 0.0 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def rock_salt_window(qt_app, tmp_path):
    cif_path = tmp_path / "nacl.cif"
    cif_path.write_text(ROCK_SALT_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def _first_site_index(window, element):
    return next(
        i
        for i, site in enumerate(window.current_structure.sites)
        if list(site.species.keys())[0].symbol == element
    )


def test_selecting_an_atom_sets_the_tracked_site_index(rock_salt_window):
    window = rock_salt_window

    na_index = _first_site_index(window, "Na")
    window._on_atom_selected_in_tree(na_index)

    assert window.current_selected_site_index == na_index


def test_coordination_analysis_with_no_selection_shows_a_message(
    rock_salt_window, monkeypatch
):
    window = rock_salt_window
    window.current_selected_site_index = None

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: shown.append(a)),
    )

    window.analyze_coordination_environment()

    assert len(shown) == 1


def test_sodium_is_six_coordinate_by_chloride(rock_salt_window):
    window = rock_salt_window

    na_index = _first_site_index(window, "Na")
    window._on_atom_selected_in_tree(na_index)

    env = window._run_coordination_analysis(
        window.current_structure,
        window.current_selected_site_index,
        window.current_site_labels,
    )

    assert env.coordination_number == 6
    assert all(neighbor.symbol == "Cl" for neighbor in env.neighbors)
    assert env.average_bond_length == pytest.approx(5.64 / 2, abs=0.01)
    assert env.min_bond_length == pytest.approx(5.64 / 2, abs=0.01)
    assert env.max_bond_length == pytest.approx(5.64 / 2, abs=0.01)


def test_coordination_environment_populates_the_panel(rock_salt_window):
    window = rock_salt_window

    na_index = _first_site_index(window, "Na")
    window._on_atom_selected_in_tree(na_index)

    env = window._run_coordination_analysis(
        window.current_structure,
        window.current_selected_site_index,
        window.current_site_labels,
    )
    window._on_coordination_environment_ready(env)

    assert window.coordination_panel.coordination_number_value.text() == "6"
    assert window.coordination_panel.neighbor_table.rowCount() == 6
    assert window.coordination_panel.average_value.text() == "2.820 \u00c5"


def test_coordination_environment_highlights_the_sphere(rock_salt_window):
    window = rock_salt_window

    na_index = _first_site_index(window, "Na")
    window._on_atom_selected_in_tree(na_index)

    assert window.viewer._group_highlight_actor is None

    env = window._run_coordination_analysis(
        window.current_structure,
        window.current_selected_site_index,
        window.current_site_labels,
    )
    window._on_coordination_environment_ready(env)

    assert window.viewer._group_highlight_actor is not None


def test_loading_a_new_structure_clears_coordination_state(
    rock_salt_window, tmp_path
):
    window = rock_salt_window

    na_index = _first_site_index(window, "Na")
    window._on_atom_selected_in_tree(na_index)
    env = window._run_coordination_analysis(
        window.current_structure,
        window.current_selected_site_index,
        window.current_site_labels,
    )
    window._on_coordination_environment_ready(env)
    assert window.coordination_panel.neighbor_table.rowCount() == 6

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

    assert window.current_selected_site_index is None
    assert window.coordination_panel.neighbor_table.rowCount() == 0
