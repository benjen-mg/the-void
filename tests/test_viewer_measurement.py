"""CrystalViewer measurement mode, exercised end to end except for the
literal mouse click: _on_measurement_pick is called directly with a
picked point, the same argument pyvista's enable_point_picking would
hand it (confirmed from pyvista's own source: "the picked point is
input as the first parameter to this callable"). Everything downstream
of that - snapping to the nearest atom, accumulating picks, computing
the result, drawing markers/lines/labels, calling on_measurement, and
resetting for the next measurement - is real production code running
against a real off-screen VTK plotter.

What this cannot verify: that a live mouse click in an actual window
actually invokes the callback pyvista's picking machinery is wired to.
That needs a real GPU/display, which this sandbox does not have.
"""

import os

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from thevoid_core.services.measurement import MeasurementResult  # noqa: E402
from thevoid_core.ui.widgets.viewer_widget import CrystalViewer  # noqa: E402


class FakeAtom:
    def __init__(self, symbol, position, site_index=0):
        self.symbol = symbol
        self.position = tuple(position)
        self.site_index = site_index


class FakeModel:
    def __init__(self, atoms):
        self.atoms = atoms


@pytest.fixture
def viewer_with_atoms():
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.picking_enabled = True
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer.plotter.add_axes = lambda *a, **k: None  # not needed for this
    viewer._measurement_mode = None
    viewer._measurement_picks = []
    viewer._measurement_actor_names = []
    viewer.on_measurement = None
    viewer.resolve_distance = None
    viewer._lattice_matrix = None

    # Three atoms forming a right angle at the origin, easy to check by
    # hand: O-C distance 1.5, O-N distance 2.0, C-O-N angle 90 degrees.
    viewer._current_model = FakeModel(
        [
            FakeAtom("C", (1.5, 0.0, 0.0), site_index=0),
            FakeAtom("O", (0.0, 0.0, 0.0), site_index=1),
            FakeAtom("N", (0.0, 2.0, 0.0), site_index=2),
        ]
    )

    yield viewer

    viewer.plotter.close()


def test_distance_measurement_end_to_end(viewer_with_atoms):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer._measurement_mode = "distance"

    # Click near atom 0 (C), then near atom 1 (O) - slightly off-centre,
    # as a real click on a sphere's surface would be.
    viewer._on_measurement_pick((1.4, 0.05, 0.0))
    assert len(results) == 0  # only one atom picked so far

    viewer._on_measurement_pick((0.05, -0.05, 0.0))
    assert len(results) == 1

    result = results[0]
    assert isinstance(result, MeasurementResult)
    assert result.kind == "distance"
    assert result.value == pytest.approx(1.5, abs=0.01)
    assert result.atom_indices == (0, 1)


def test_angle_measurement_end_to_end(viewer_with_atoms):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer._measurement_mode = "angle"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))   # C
    viewer._on_measurement_pick((0.0, 0.0, 0.0))   # O - vertex
    viewer._on_measurement_pick((0.0, 2.0, 0.0))   # N

    assert len(results) == 1

    result = results[0]
    assert result.kind == "angle"
    assert result.value == pytest.approx(90.0, abs=0.1)
    assert result.atom_indices == (0, 1, 2)  # vertex is the middle index


def test_a_click_far_from_any_atom_is_ignored(viewer_with_atoms):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))       # hits C
    viewer._on_measurement_pick((50.0, 50.0, 50.0))    # hits nothing

    # The miss must not count as a pick - still waiting on the second
    # atom, not silently paired with something 80+ A away.
    assert len(viewer._measurement_picks) == 1
    assert results == []


def test_measurement_resets_after_completing_so_another_can_start(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))
    viewer._on_measurement_pick((0.0, 0.0, 0.0))
    assert len(results) == 1

    # Immediately start a second measurement without re-enabling.
    viewer._on_measurement_pick((0.0, 0.0, 0.0))   # O
    viewer._on_measurement_pick((0.0, 2.0, 0.0))   # N
    assert len(results) == 2

    assert results[1].value == pytest.approx(2.0, abs=0.01)


def test_measurement_draws_visible_actors(viewer_with_atoms):
    viewer = viewer_with_atoms
    viewer._measurement_mode = "distance"

    before = len(viewer.plotter.renderer.actors)

    viewer._on_measurement_pick((1.5, 0.0, 0.0))
    after_first_pick = len(viewer.plotter.renderer.actors)
    assert after_first_pick > before  # marker for the first pick

    viewer._on_measurement_pick((0.0, 0.0, 0.0))
    after_second_pick = len(viewer.plotter.renderer.actors)
    # Second marker, a connecting line, and a text label.
    assert after_second_pick > after_first_pick


def test_clear_measurements_removes_drawn_actors(viewer_with_atoms):
    viewer = viewer_with_atoms
    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))
    viewer._on_measurement_pick((0.0, 0.0, 0.0))

    assert viewer._measurement_actor_names  # something was drawn

    viewer.clear_measurements()

    assert viewer._measurement_actor_names == []


def test_disable_measurement_resets_state(viewer_with_atoms):
    viewer = viewer_with_atoms
    viewer._measurement_mode = "distance"
    viewer._on_measurement_pick((1.5, 0.0, 0.0))

    viewer.disable_measurement()

    assert viewer._measurement_mode is None
    assert viewer._measurement_picks == []


def test_enable_measurement_rejects_an_invalid_kind(viewer_with_atoms):
    with pytest.raises(ValueError):
        viewer_with_atoms.enable_measurement("volume")


def test_switching_directly_between_distance_and_angle_both_succeed(
    viewer_with_atoms,
):
    """Regression test for a real reported bug: Distance worked, but
    switching straight to Angle afterwards - without going back to Off
    first, which is exactly what clicking a different Measure menu item
    does - failed with "Measurement unavailable". Calling
    enable_point_picking() a second time on top of an already-active
    picking session is a real failure mode on some VTK/pyvista builds;
    enable_measurement() must tear down any prior session first rather
    than assuming it's starting from a clean state.
    """
    viewer = viewer_with_atoms

    assert viewer.enable_measurement("distance") is True
    assert viewer._measurement_mode == "distance"

    # The exact reported sequence: switch directly to Angle, no Off
    # in between.
    assert viewer.enable_measurement("angle") is True
    assert viewer._measurement_mode == "angle"

    # And back again, for good measure - this must keep working no
    # matter how many times it's toggled.
    assert viewer.enable_measurement("distance") is True
    assert viewer._measurement_mode == "distance"


def test_enable_measurement_returns_false_when_picking_is_unavailable(
    viewer_with_atoms, monkeypatch
):
    """Confirmed directly (see the comment in enable_measurement):
    enable_point_picking() raises AttributeError rather than returning
    an error when the VTK interactor isn't ready - the exact situation
    this sandbox hits with no GPU. The viewer must turn that into a
    clean False rather than letting it propagate as an unhandled
    exception up through a Qt slot.
    """
    viewer = viewer_with_atoms

    def _raise(*args, **kwargs):
        raise AttributeError("'NoneType' object has no attribute 'picker'")

    monkeypatch.setattr(viewer.plotter, "enable_point_picking", _raise)

    result = viewer.enable_measurement("distance")

    assert result is False
    assert viewer._measurement_mode is None


def test_picking_with_no_model_loaded_does_nothing(viewer_with_atoms):
    viewer = viewer_with_atoms
    viewer._current_model = None
    viewer._measurement_mode = "distance"

    # Must not raise even though there's nothing to pick.
    viewer._on_measurement_pick((0.0, 0.0, 0.0))

    assert viewer._measurement_picks == []


def test_distance_result_carries_site_indices(viewer_with_atoms):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer._measurement_mode = "distance"
    viewer._on_measurement_pick((1.5, 0.0, 0.0))  # C, site_index 0
    viewer._on_measurement_pick((0.0, 0.0, 0.0))  # O, site_index 1

    assert results[0].site_indices == (0, 1)


def test_angle_result_carries_site_indices_with_vertex_in_the_middle(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer._measurement_mode = "angle"
    viewer._on_measurement_pick((1.5, 0.0, 0.0))  # C, site 0
    viewer._on_measurement_pick((0.0, 0.0, 0.0))  # O, site 1 (vertex)
    viewer._on_measurement_pick((0.0, 2.0, 0.0))  # N, site 2

    assert results[0].site_indices == (0, 1, 2)


def test_resolve_distance_hook_overrides_the_reported_value(
    viewer_with_atoms,
):
    """The core of nearest-image support: when the host window supplies
    a resolver, its answer wins over the naive straight-line distance
    between the two literally-clicked points.
    """
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    def fake_resolver(atom_a, atom_b, lattice_matrix):
        # Pretend the nearest periodic image of O is actually much
        # closer than its as-clicked position suggests.
        return (0.3, 0.0, 0.0), 1.2

    viewer.resolve_distance = fake_resolver
    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))  # C
    viewer._on_measurement_pick((0.0, 0.0, 0.0))  # O

    assert results[0].value == pytest.approx(1.2)


def test_resolve_distance_hook_is_consulted_for_both_legs_of_an_angle(
    viewer_with_atoms,
):
    """Periodic boundary support for angles: each outer atom (A and C)
    is independently resolved against the vertex (B) as anchor, so the
    resolver must be called twice - once per leg - with the vertex as
    the anchor both times, never the vertex resolved against itself.
    """
    viewer = viewer_with_atoms
    calls = []

    def fake_resolver(anchor_atom, target_atom, lattice_matrix):
        calls.append((anchor_atom.symbol, target_atom.symbol))
        return None  # fall back to the plain position; just recording calls

    viewer.resolve_distance = fake_resolver
    viewer._measurement_mode = "angle"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))  # C (atom A of the angle)
    viewer._on_measurement_pick((0.0, 0.0, 0.0))  # O (vertex)
    viewer._on_measurement_pick((0.0, 2.0, 0.0))  # N (atom C of the angle)

    # Vertex (O) is the anchor for both calls; C and N are each
    # resolved against it in turn. Order doesn't matter, only that both
    # legs were actually resolved and the vertex was never the target.
    assert set(calls) == {("O", "C"), ("O", "N")}


def test_resolve_distance_hook_can_change_the_reported_angle(
    viewer_with_atoms,
):
    """If periodic resolution moves one leg, the angle must be
    recomputed from the resolved positions, not silently kept at the
    naive as-clicked value.
    """
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    def fake_resolver(anchor_atom, target_atom, lattice_matrix):
        if target_atom.symbol == "N":
            # Naive: N at (0, 2, 0), direction from vertex is +y,
            # giving a 90 degree angle with C along +x. Resolved:
            # move N onto the same +x ray as C entirely, which must
            # collapse the angle to 0 degrees - unambiguously
            # different from the naive 90, so this proves the
            # resolved position is what's actually used.
            return (1.5, 0.0, 0.0), 1.5
        return None

    viewer.resolve_distance = fake_resolver
    viewer._measurement_mode = "angle"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))  # C
    viewer._on_measurement_pick((0.0, 0.0, 0.0))  # O (vertex)
    viewer._on_measurement_pick((0.0, 2.0, 0.0))  # N, resolved onto C's ray

    assert results[0].value == pytest.approx(0.0, abs=0.01)


def test_resolve_distance_hook_receives_the_lattice_matrix(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    received = []

    def fake_resolver(atom_a, atom_b, lattice_matrix):
        received.append(lattice_matrix)
        return None  # fall back, but still confirm what was passed

    viewer.resolve_distance = fake_resolver
    viewer._lattice_matrix = [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))
    viewer._on_measurement_pick((0.0, 0.0, 0.0))

    assert received == [[[10, 0, 0], [0, 10, 0], [0, 0, 10]]]


def test_resolve_distance_returning_none_falls_back_to_plain_distance(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    viewer.resolve_distance = lambda a, b, lattice: None
    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))  # C
    viewer._on_measurement_pick((0.0, 0.0, 0.0))  # O

    # C at (1.5,0,0), O at (0,0,0): straight-line distance 1.5.
    assert results[0].value == pytest.approx(1.5)


def test_resolve_distance_hook_raising_does_not_break_the_measurement(
    viewer_with_atoms,
):
    """A bad resolver must degrade to the plain distance, not crash the
    whole measurement.
    """
    viewer = viewer_with_atoms
    results = []
    viewer.on_measurement = results.append

    def broken_resolver(atom_a, atom_b, lattice_matrix):
        raise RuntimeError("boom")

    viewer.resolve_distance = broken_resolver
    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))
    viewer._on_measurement_pick((0.0, 0.0, 0.0))

    assert results[0].value == pytest.approx(1.5)


def test_resolved_distance_line_uses_the_resolver_position_not_the_click(
    viewer_with_atoms,
):
    """The drawn line's second endpoint must move to match the
    resolver's answer - otherwise the line and the reported number
    would visually disagree with each other.
    """
    viewer = viewer_with_atoms

    resolved_position = (7.0, 8.0, 9.0)

    viewer.resolve_distance = lambda a, b, lattice: (resolved_position, 5.0)
    viewer._measurement_mode = "distance"

    viewer._on_measurement_pick((1.5, 0.0, 0.0))
    viewer._on_measurement_pick((0.0, 0.0, 0.0))

    # The line actor's points must include the resolved position, not
    # the atom's literal clicked position (0, 0, 0).
    # Rather than guess actor naming details, inspect all line meshes'
    # bounds for the resolved coordinate directly.
    found = False
    for actor in viewer.plotter.renderer.actors.values():
        mapper = getattr(actor, "mapper", None)
        if mapper is None:
            continue
        points = mapper.dataset.points
        if len(points) and any(
            tuple(p) == pytest.approx(resolved_position) for p in points
        ):
            found = True
            break

    assert found
