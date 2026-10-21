"""End-to-end distance and angle measurement, through the real
MainWindow rather than individual pieces in isolation - in particular,
confirms the nearest-image (minimum periodic image) convention actually
changes the reported distance/angle and the drawn line, not just that
the resolver function is correct in isolation.

Every periodic test case here was hand-verified with a short numpy
script before being written into a test, rather than trusting an
intuition about which geometry would actually exercise periodicity -
two earlier attempts at an angle test case turned out to be
geometrically coincidental (both the naive and the resolved position
were perpendicular to the other leg, so the reported angle came out
identical either way despite periodic resolution genuinely having
run). The case used below - O1-H1-H2 in a 10 A cubic cell - gives a
hand-verified ~6.3 degrees unresolved versus ~135 degrees resolved, an
unambiguous difference.

The one thing this cannot exercise is a literal mouse click landing on
an atom in a live render window - this sandbox has no GPU, and
CrystalViewer.enable_measurement() fails gracefully in exactly the same
already-tested way (see test_viewer_measurement.py). Every fixture here
bypasses only that one registration step, by setting the viewer's
measurement mode directly to whatever a working enable_measurement()
call would have left it as - everything downstream of that (picking,
resolving the nearest image, drawing the line, populating the
Measurements panel) is real production code.

pymatgen also reorders sites internally (alphabetically by element, in
this build) rather than preserving CIF file order, discovered while
writing these tests - site_index cannot be assumed to match the order
atoms were listed in the CIF, so sites below are identified by their
actual element and fractional position via _find_site_index() rather
than by an assumed index.
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

#: C1 and O1 sit near opposite faces of a 10 A cubic cell: their
#: same-image straight-line distance is ~9.0 A, but the true
#: nearest-image distance (O1's image one cell over) is ~1.0 A - the
#: whole reason this feature exists.
PERIODIC_CIF = """\
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
C1 C 0.05 0.50 0.50 1.0
O1 O 0.95 0.50 0.50 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def periodic_window(qt_app, tmp_path):
    cif_path = tmp_path / "periodic.cif"
    cif_path.write_text(PERIODIC_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    # Bypass only the real-VTK-picking-registration step (see module
    # docstring) - not what is under test here.
    window.viewer._measurement_mode = "distance"
    window.measurement_mode = "distance"

    yield window

    window.close()


def _pick_by_site_index(window, site_index):
    atom = next(
        a
        for a in window.viewer._current_model.atoms
        if a.site_index == site_index
    )
    window.viewer._on_measurement_pick(atom.position)
    return atom


def _finish_and_get_result(window, second_site_index):
    results = []
    window.viewer.on_measurement = results.append
    _pick_by_site_index(window, second_site_index)
    return results[0]


def test_distance_measurement_uses_nearest_image_not_same_cell(
    periodic_window,
):
    window = periodic_window
    results = []
    window.viewer.on_measurement = results.append

    _pick_by_site_index(window, 0)  # C1
    _pick_by_site_index(window, 1)  # O1

    assert len(results) == 1
    # ~1.0 A (nearest image), not ~9.0 A (same-cell straight line).
    assert results[0].value == pytest.approx(1.0, abs=0.01)


def test_distance_measurement_reports_correct_labels_and_elements(
    periodic_window,
):
    window = periodic_window
    results = []
    window.viewer.on_measurement = results.append

    _pick_by_site_index(window, 0)
    _pick_by_site_index(window, 1)

    window._on_measurement_result(results[0])

    row = [
        window.measurements_panel.table.item(0, c).text()
        for c in range(8)
    ]
    assert row == [
        "Distance", "C1", "C", "O1", "O", "\u2013", "\u2013", "1.000 \u00c5",
    ]


def test_distance_measurement_draws_the_resolved_short_line(
    periodic_window,
):
    """The drawn line must reflect the resolved (short) distance, not
    the naive long one - otherwise the line and the reported number
    would visually disagree with each other.
    """
    window = periodic_window
    _pick_by_site_index(window, 0)
    _pick_by_site_index(window, 1)

    found_short_line = False

    for actor in window.viewer.plotter.renderer.actors.values():
        mapper = getattr(actor, "mapper", None)
        dataset = getattr(mapper, "dataset", None) if mapper else None

        if dataset is None or len(dataset.points) < 2:
            continue

        xs = [p[0] for p in dataset.points]

        if 0.9 < (max(xs) - min(xs)) < 1.1:
            found_short_line = True
            break

    assert found_short_line


def test_multiple_distance_measurements_accumulate_in_the_panel(
    periodic_window,
):
    window = periodic_window

    _pick_by_site_index(window, 0)
    result_1 = _finish_and_get_result(window, 1)
    window._on_measurement_result(result_1)

    _pick_by_site_index(window, 1)
    result_2 = _finish_and_get_result(window, 0)
    window._on_measurement_result(result_2)

    assert window.measurements_panel.table.rowCount() == 2


def test_clear_measurements_empties_the_panel(periodic_window):
    window = periodic_window

    _pick_by_site_index(window, 0)
    result = _finish_and_get_result(window, 1)
    window._on_measurement_result(result)
    assert window.measurements_panel.table.rowCount() == 1

    window._on_clear_measurements()

    assert window.measurements_panel.table.rowCount() == 0


def test_clear_measurements_also_clears_the_3d_scene(periodic_window):
    window = periodic_window

    _pick_by_site_index(window, 0)
    result = _finish_and_get_result(window, 1)
    window._on_measurement_result(result)

    assert window.viewer._measurement_actor_names

    window._on_clear_measurements()

    assert window.viewer._measurement_actor_names == []


# --- Angle measurement, including periodic boundary support ---

#: O1-H1 is an ordinary 1 A bond needing no periodicity. H2's
#: reference-cell position is ~9.05 A from O1 (not a real bond, and at
#: a shallow ~6.3 degree angle from the O1-H1 direction) but its
#: nearest periodic image is only ~1.41 A away, at a very different
#: ~135 degree angle - hand-verified before use, not just asserted.
ANGLE_PERIODIC_CIF = """\
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
O1 O 0.05 0.50 0.50 1.0
H1 H 0.15 0.50 0.50 1.0
H2 H 0.95 0.60 0.50 1.0
"""


@pytest.fixture
def angle_periodic_window(qt_app, tmp_path):
    cif_path = tmp_path / "angle_periodic.cif"
    cif_path.write_text(ANGLE_PERIODIC_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    window.viewer._measurement_mode = "angle"
    window.measurement_mode = "angle"

    yield window

    window.close()


def _find_site_index(window, element, approx_frac):
    """pymatgen reorders sites internally (alphabetically by element,
    in this build) - site_index cannot be assumed to match CIF file
    order, so sites are identified by their actual element and
    fractional position instead.
    """
    for index, site in enumerate(window.current_structure.sites):
        species = list(site.species.keys())
        symbol = species[0].symbol if species else "?"

        if symbol != element:
            continue

        fx, fy, fz = site.frac_coords
        ax, ay, az = approx_frac

        if abs(fx - ax) < 0.01 and abs(fy - ay) < 0.01 and abs(fz - az) < 0.01:
            return index

    raise AssertionError(f"no matching site for {element} near {approx_frac}")


def _pick_atom_at(window, site_index):
    atom = next(
        a
        for a in window.viewer._current_model.atoms
        if a.site_index == site_index
    )
    window.viewer._on_measurement_pick(atom.position)
    return atom


def test_angle_measurement_uses_nearest_image_not_same_cell(
    angle_periodic_window,
):
    window = angle_periodic_window

    o1 = _find_site_index(window, "O", (0.05, 0.5, 0.5))
    h1 = _find_site_index(window, "H", (0.15, 0.5, 0.5))
    h2 = _find_site_index(window, "H", (0.95, 0.6, 0.5))

    results = []
    window.viewer.on_measurement = results.append

    _pick_atom_at(window, h1)  # A
    _pick_atom_at(window, o1)  # vertex B
    _pick_atom_at(window, h2)  # C

    assert len(results) == 1
    # ~135 degrees (nearest image), not ~6.3 degrees (same-cell,
    # non-bonded straight line) - hand-verified above.
    assert results[0].value == pytest.approx(135.0, abs=0.5)


def test_angle_measurement_reports_correct_labels_and_elements(
    angle_periodic_window,
):
    window = angle_periodic_window

    o1 = _find_site_index(window, "O", (0.05, 0.5, 0.5))
    h1 = _find_site_index(window, "H", (0.15, 0.5, 0.5))
    h2 = _find_site_index(window, "H", (0.95, 0.6, 0.5))

    results = []
    window.viewer.on_measurement = results.append

    _pick_atom_at(window, h1)
    _pick_atom_at(window, o1)
    _pick_atom_at(window, h2)

    window._on_measurement_result(results[0])

    row = [
        window.measurements_panel.table.item(0, c).text() for c in range(8)
    ]

    assert row[0] == "Angle"
    assert row[1] == "H1"  # atom A
    assert row[2] == "H"  # element A
    assert row[3] == "O1"  # the vertex, as atom B
    assert row[4] == "O"  # element B
    assert row[5] == "H2"  # atom C - in its own column, not the value text
    assert row[6] == "H"  # element C
    assert row[7] == "135.0\u00b0"  # the value, on its own


def test_clear_measurements_removes_both_distance_and_angle_rows(
    angle_periodic_window,
):
    """Only one clearing action exists - it must remove everything,
    distances and angles alike, in one step.
    """
    window = angle_periodic_window

    o1 = _find_site_index(window, "O", (0.05, 0.5, 0.5))
    h1 = _find_site_index(window, "H", (0.15, 0.5, 0.5))
    h2 = _find_site_index(window, "H", (0.95, 0.6, 0.5))

    angle_results = []
    window.viewer.on_measurement = angle_results.append
    _pick_atom_at(window, h1)
    _pick_atom_at(window, o1)
    _pick_atom_at(window, h2)
    window._on_measurement_result(angle_results[0])

    window.viewer._measurement_mode = "distance"
    window.viewer._measurement_picks = []
    distance_results = []
    window.viewer.on_measurement = distance_results.append
    _pick_atom_at(window, o1)
    _pick_atom_at(window, h1)
    window._on_measurement_result(distance_results[0])

    assert window.measurements_panel.table.rowCount() == 2

    window._on_clear_measurements()

    assert window.measurements_panel.table.rowCount() == 0
    assert window.viewer._measurement_actor_names == []
