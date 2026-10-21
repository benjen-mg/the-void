"""Round 49: torsion (dihedral) measurement - the service function,
the viewer's four-pick mode, the Measurements panel row, and the menu.
"""

import os

import pytest

from thevoid_core.services.measurement import dihedral


# --- dihedral() -----------------------------------------------------------

B = (0.0, 0.0, 0.0)
C = (0.0, 0.0, 1.5)
A = (1.0, 0.0, 0.0)


@pytest.mark.parametrize(
    "d, expected",
    [
        ((1.0, 0.0, 1.5), 0.0),      # eclipsed / syn-periplanar
        ((-1.0, 0.0, 1.5), 180.0),   # anti-periplanar
        ((0.0, 1.0, 1.5), 90.0),
        ((0.0, -1.0, 1.5), -90.0),
        ((0.5, 3 ** 0.5 / 2, 1.5), 60.0),   # gauche+
        ((0.5, -(3 ** 0.5) / 2, 1.5), -60.0),  # gauche-
    ],
)
def test_reference_torsions(d, expected):
    assert dihedral(A, B, C, d) == pytest.approx(expected, abs=1e-9)


def test_sign_follows_the_iupac_convention():
    """Looking along B->C, A (front) turning clockwise to eclipse D
    (back) is positive. With B->C along +z and the viewer looking down
    +z, +x to +y is a clockwise turn - so D at +y is +90, matching the
    standard |b2| b1.(b2 x b3) / (b1 x b2).(b2 x b3) formula.
    """
    import math

    import numpy as np

    a, b, c, d = map(np.array, (A, B, C, (0.0, 1.0, 1.5)))
    b1, b2, b3 = b - a, c - b, d - c
    reference = math.degrees(
        math.atan2(
            np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3)),
            np.dot(np.cross(b1, b2), np.cross(b2, b3)),
        )
    )

    assert dihedral(A, B, C, (0.0, 1.0, 1.5)) == pytest.approx(reference)
    assert reference == pytest.approx(90.0)


def test_reversing_the_chain_keeps_the_same_torsion():
    """D-C-B-A describes the same torsion as A-B-C-D."""
    d = (0.3, 0.8, 1.9)

    assert dihedral(A, B, C, d) == pytest.approx(dihedral(d, C, B, A))


def test_mirror_image_flips_the_sign():
    d = (0.3, 0.8, 1.9)
    mirrored = lambda p: (p[0], -p[1], p[2])  # noqa: E731

    assert dihedral(A, B, C, d) == pytest.approx(
        -dihedral(mirrored(A), mirrored(B), mirrored(C), mirrored(d))
    )


def test_range_is_minus_180_exclusive_to_180_inclusive():
    for tiny in (1e-15, 0.0, -1e-15):
        value = dihedral(A, B, C, (-1.0, tiny, 1.5))
        assert -180.0 < value <= 180.0
        assert abs(value) == pytest.approx(180.0)


def test_collinear_atoms_raise():
    with pytest.raises(ValueError):
        dihedral((0, 0, -1), B, C, (1, 0, 2))  # A-B-C straight


# --- viewer four-pick mode ------------------------------------------------

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from thevoid_core.ui.widgets.viewer_widget import CrystalViewer  # noqa: E402


class FakeAtom:
    def __init__(self, symbol, position, site_index):
        self.symbol = symbol
        self.position = tuple(position)
        self.site_index = site_index


class FakeModel:
    def __init__(self, atoms):
        self.atoms = atoms


@pytest.fixture
def viewer():
    pv.OFF_SCREEN = True

    v = CrystalViewer.__new__(CrystalViewer)
    v.picking_enabled = True
    v.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    v._measurement_mode = None
    v._measurement_picks = []
    v._measurement_actor_names = []
    v.on_measurement = None
    v.on_measurement_failed = None
    v.resolve_distance = None
    v._lattice_matrix = None
    # A gauche(+60 degree) butane-like chain, bonds 1.5 A apart.
    v._current_model = FakeModel(
        [
            FakeAtom("C", A, 0),
            FakeAtom("C", B, 1),
            FakeAtom("C", C, 2),
            FakeAtom("C", (0.5, 3 ** 0.5 / 2, 1.5), 3),
            FakeAtom("C", (0.0, 0.0, 3.0), 4),  # collinear with B-C
        ]
    )
    yield v
    v.plotter.close()


def test_torsion_needs_four_picks_and_reports_the_dihedral(viewer):
    results = []
    viewer.on_measurement = results.append
    viewer._measurement_mode = "torsion"

    for atom in viewer._current_model.atoms[:3]:
        viewer._on_measurement_pick(atom.position)
        assert results == []

    viewer._on_measurement_pick(viewer._current_model.atoms[3].position)

    assert len(results) == 1
    result = results[0]
    assert result.kind == "torsion"
    assert result.value == pytest.approx(60.0, abs=1e-6)
    assert result.atom_indices == (0, 1, 2, 3)
    assert result.site_indices == (0, 1, 2, 3)
    assert result.label == "60.0°"
    # Three connecting segments, a label, and four pick markers.
    assert len(viewer._measurement_actor_names) == 4 + 3 + 1


def test_clicking_the_same_atom_twice_is_ignored(viewer):
    viewer._measurement_mode = "torsion"
    atoms = viewer._current_model.atoms

    viewer._on_measurement_pick(atoms[0].position)
    viewer._on_measurement_pick(atoms[0].position)

    assert viewer._measurement_picks == [0]


def test_collinear_torsion_reports_failure_and_restarts(viewer):
    results, failures = [], []
    viewer.on_measurement = results.append
    viewer.on_measurement_failed = failures.append
    viewer._measurement_mode = "torsion"
    atoms = viewer._current_model.atoms

    # A, B, C, then a D sitting on the B-C line extended: B-C-D is
    # straight, so the torsion is undefined.
    for index in (0, 1, 2, 4):
        viewer._on_measurement_pick(atoms[index].position)

    assert results == []
    assert len(failures) == 1
    assert "collinear" in failures[0]
    assert viewer._measurement_picks == []


def test_torsion_is_a_known_measurement_kind(viewer):
    assert viewer.MEASUREMENT_PICKS_NEEDED["torsion"] == 4

    # An unknown kind is rejected before any picking is touched.
    with pytest.raises(ValueError):
        viewer.enable_measurement("volume")


# --- panel ---------------------------------------------------------------

def test_panel_torsion_row_and_visual_column_order():
    from PySide6.QtWidgets import QApplication

    from thevoid_core.ui.widgets.measurements_panel import (
        NOT_APPLICABLE,
        VALUE_COLUMN,
        MeasurementsPanel,
    )
    from thevoid_core.ui.widgets.table_tools import table_rows

    QApplication.instance() or QApplication([])
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="C1", element_a="C", label_b="O1", element_b="O",
        distance_angstrom=1.234,
    )
    panel.add_torsion(
        label_a="C1", element_a="C", label_b="C2", element_b="C",
        label_c="C3", element_c="C", label_d="C4", element_d="C",
        torsion_degrees=-62.25,
    )

    table = panel.table
    assert VALUE_COLUMN == 7
    assert table.item(1, 0).text() == "Torsion"
    assert table.item(1, VALUE_COLUMN).text() == "-62.2°"
    assert table.item(1, 8).text() == "C4"
    assert table.item(0, 8).text() == NOT_APPLICABLE

    header, distance_row, torsion_row = table_rows(table, selected_only=False)
    assert header[-1] == "Value"
    assert header[-3:-1] == ["Atom D", "Element D"]
    assert torsion_row[-1] == "-62.2°"
    assert distance_row[-1] == "1.234 Å"


def test_measure_menu_offers_torsion():
    from PySide6.QtWidgets import QApplication, QMessageBox

    from thevoid_core.ui.windows.main_window import MainWindow

    QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    window = MainWindow()
    try:
        modes = [a.data() for a in window.measure_group.actions()]
        assert modes == [None, "distance", "angle", "torsion"]
    finally:
        window.close()


def test_agrees_with_pymatgen_on_a_real_structure():
    """Cross-check against pymatgen's own Molecule/Structure dihedral on
    a bundled open Zn MOF's first sites - an
    independent implementation of the same quantity.
    """
    pytest.importorskip("pymatgen")
    from pymatgen.core import Structure

    from tests.sample_structures import ZN_DIA

    structure = Structure.from_file(str(ZN_DIA))

    checked = 0
    for i in range(0, min(len(structure) - 3, 40), 3):
        idx = (i, i + 1, i + 2, i + 3)
        points = [structure[j].coords for j in idx]
        try:
            ours = dihedral(*points)
        except ValueError:
            continue
        theirs = structure.get_dihedral(*idx)
        assert ours == pytest.approx(theirs, abs=1e-6)
        checked += 1

    assert checked > 3
