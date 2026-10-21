"""End-to-end MOF linker/metal-node detection, through the real
MainWindow rather than the partition algorithm in isolation.

The synthetic structure's coordinates were computed by hand against the
covalent-radius bond detector's actual cutoff ranges (printed and
checked before use, not guessed) so the intended chemistry - a
mu-oxo-bridged Zn2 node with two separate single-carbon linkers, no
direct Zn-Zn bond - is what the detector actually finds. An earlier
attempt at this same test used coordinates that didn't land in the
detector's bond range at all, silently testing a different, wrong
structure; this one prints and asserts on the detected bonds directly
before running the partition, so that mistake can't repeat unnoticed.
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

#: Zn1-Ob-Zn2 mu-oxo bridge (Zn-O 1.8-2.0 A, well inside the detector's
#: 1.316-2.280 A range; Zn1-Zn2 direct distance is 4.0 A, well outside
#: its 1.708-2.840 A range, so there is no direct Zn-Zn bond - the node
#: is held together only by the bridging oxygen). Two O-C-O stand-in
#: carboxylates join the node to its neighbours: one along a, from Zn2
#: to the next cell's Zn1, one along b, from Zn1 to the next cell's Zn1.
#: (Before round 55 each O-C-O hung off one Zn only; since terminal
#: ligands - bonded to a single node - are now part of that node, the
#: stand-ins have to bridge to count as linkers, as real ones do.)
MOF_CIF = """\
data_test
_cell_length_a 10.0
_cell_length_b 6.0
_cell_length_c 20.0
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
Zn1 Zn 0.00 0.00 0.50 1.0
Ob  O  0.20 0.00 0.50 1.0
Zn2 Zn 0.40 0.00 0.50 1.0
O2a O  0.58 0.00 0.50 1.0
C2  C  0.70 0.00 0.50 1.0
O2b O  0.82 0.00 0.50 1.0
O1a O  0.00 0.30 0.50 1.0
C1  C  0.00 0.50 0.50 1.0
O1b O  0.00 0.70 0.50 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def mof_window(qt_app, tmp_path):
    cif_path = tmp_path / "mof_test.cif"
    cif_path.write_text(MOF_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def test_bond_detection_found_the_intended_chemistry(mof_window):
    """Confirms the test structure's geometry actually produced the
    bond graph it was designed for, before trusting anything the
    partition step reports about it.
    """
    window = mof_window

    def label_of(index):
        site = window.current_structure.sites[index]
        return list(site.species.keys())[0].symbol

    bonded_pairs = {
        frozenset((label_of(b.atom1), label_of(b.atom2)))
        for b in window.current_bonds
    }

    assert frozenset(("Zn", "O")) in bonded_pairs
    assert frozenset(("C", "O")) in bonded_pairs
    # No direct Zn-Zn bond - the node must be held together by the
    # bridging oxygen alone, not a metal-metal bond, for this test to
    # actually exercise the bridging-atom code path.
    assert frozenset(("Zn", "Zn")) not in bonded_pairs


def test_mof_partition_finds_one_bridged_node_and_two_linkers(mof_window):
    window = mof_window

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )

    assert partition.metal_node_count == 1
    node = partition.metal_nodes[0]
    assert node.metal_symbols == ("Zn", "Zn")
    assert node.formula == "OZn2"
    assert node.coordination_number == 2

    assert partition.linker_count == 2
    assert all(linker.formula == "CO2" for linker in partition.linkers)
    assert all(linker.atom_count == 3 for linker in partition.linkers)


def test_mof_analysis_populates_the_panel(mof_window):
    window = mof_window

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )
    window._on_mof_partition_ready((partition, site_images))

    assert "2 linker" in window.mof_analysis_panel.summary_label.text()
    assert "1 metal node" in window.mof_analysis_panel.summary_label.text()
    assert window.mof_analysis_panel.linker_table.rowCount() == 2
    assert window.mof_analysis_panel.node_table.rowCount() == 1
    assert window.mof_analysis_panel.node_table.item(0, 1).text() == "Zn"
    assert window.mof_analysis_panel.node_table.item(0, 2).text() == "OZn2"


def test_selecting_a_node_row_highlights_it_in_the_3d_view(mof_window):
    window = mof_window

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )
    window._on_mof_partition_ready((partition, site_images))

    assert window.viewer._group_highlight_actor is None

    window.mof_analysis_panel.node_table.selectRow(0)

    assert window.viewer._group_highlight_actor is not None


def test_selecting_a_linker_row_replaces_the_node_highlight(mof_window):
    """Only one group highlighted at a time - switching from a node to
    a linker must replace, not add to, the previous highlight.
    """
    window = mof_window

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )
    window._on_mof_partition_ready((partition, site_images))

    window.mof_analysis_panel.node_table.selectRow(0)
    after_node = len(window.viewer.plotter.renderer.actors)

    window.mof_analysis_panel.linker_table.selectRow(0)
    after_linker = len(window.viewer.plotter.renderer.actors)

    assert after_linker == after_node


def test_loading_a_new_structure_clears_the_mof_panel(mof_window, tmp_path):
    window = mof_window

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )
    window._on_mof_partition_ready((partition, site_images))
    assert window.mof_analysis_panel.linker_table.rowCount() == 2

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

    assert window.mof_analysis_panel.linker_table.rowCount() == 0
    assert "No MOF analysis" in window.mof_analysis_panel.summary_label.text()
