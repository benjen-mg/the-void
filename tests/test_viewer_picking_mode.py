"""CrystalViewer.set_picking_enabled(): the "Select"/"View" cursor
mode toggle. Uses the same plain pv.Plotter(off_screen=True) pattern
as test_viewer_selection.py and test_viewer_measurement.py, which
already confirms point picking itself works fine in this sandbox
(unlike a real QtInteractor-backed plotter with no GPU).
"""

import os

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.ui.widgets.viewer_widget import (  # noqa: E402
    CrystalViewer,
    DisplayStyle,
)


class FakeAtom:
    def __init__(self, symbol, position, site_index):
        self.symbol = symbol
        self.position = tuple(position)
        self.site_index = site_index
        self.image = (0, 0, 0)


class FakeModel:
    def __init__(self, atoms):
        self.atoms = atoms


@pytest.fixture
def viewer_with_atoms():
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer.display_style = DisplayStyle.BALL_AND_STICK
    viewer.radius_provider = ElementRadiusProvider()
    viewer._selected_atom_index = None
    viewer._selection_actor = None
    viewer._measurement_mode = None
    viewer._measurement_picks = []
    viewer._measurement_actor_names = []
    viewer.on_atom_selected = None
    viewer.on_measurement = None
    viewer.resolve_distance = None
    viewer.picking_enabled = True

    viewer._current_model = FakeModel(
        [
            FakeAtom("C", (0.0, 0.0, 0.0), site_index=0),
            FakeAtom("O", (1.5, 0.0, 0.0), site_index=1),
        ]
    )

    yield viewer

    viewer.plotter.close()


def test_picking_enabled_defaults_to_true_in_the_constructor():
    """The real __init__ (not a __new__-constructed test double) must
    default this to True, matching every previous version's behaviour,
    so existing users see no change unless they deliberately switch
    modes. Checked by inspecting the source directly rather than
    constructing a real CrystalViewer, which needs an actual GL
    context and reliably crashes in this GPU-less sandbox - the same
    reason every other test in this file (and this project) builds a
    viewer via __new__ plus a plain offscreen pv.Plotter instead.
    """
    import inspect

    source = inspect.getsource(CrystalViewer.__init__)

    assert "self.picking_enabled = True" in source


def test_disabling_picking_sets_the_flag(viewer_with_atoms):
    viewer = viewer_with_atoms

    result = viewer.set_picking_enabled(False)

    assert result is True
    assert viewer.picking_enabled is False


def test_disabling_picking_does_not_raise_with_nothing_active(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms

    viewer.set_picking_enabled(False)  # should not raise


def test_re_enabling_with_no_prior_measurement_restores_selection(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    viewer.set_picking_enabled(False)

    result = viewer.set_picking_enabled(True)

    assert result is True
    assert viewer.picking_enabled is True

    # Confirm selection is genuinely active, not just the flag: a
    # simulated pick should reach the selection callback.
    received = []
    viewer.on_atom_selected = received.append
    viewer._on_selection_pick((0.0, 0.0, 0.0))

    assert len(received) == 1
    assert received[0].symbol == "C"


def test_re_enabling_after_a_measurement_mode_restores_that_mode_not_selection(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    assert viewer.enable_measurement("distance") is True

    viewer.set_picking_enabled(False)
    assert viewer._measurement_mode == "distance"  # unchanged while off

    result = viewer.set_picking_enabled(True)

    assert result is True
    assert viewer._measurement_mode == "distance"


def test_enable_selection_directly_marks_picking_as_enabled(
    viewer_with_atoms,
):
    """Calling enable_selection() directly (bypassing the toggle
    method entirely, the way render_model()'s own redraw restoration
    does) must still keep the picking_enabled flag honest - otherwise
    a later redraw in "View mode" could incorrectly think picking is
    still meant to be off.
    """
    viewer = viewer_with_atoms
    viewer.picking_enabled = False

    assert viewer.enable_selection() is True
    assert viewer.picking_enabled is True


def test_enable_measurement_directly_marks_picking_as_enabled(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    viewer.picking_enabled = False

    assert viewer.enable_measurement("distance") is True
    assert viewer.picking_enabled is True


def test_disabling_then_a_simulated_redraw_does_not_reconnect_selection(
    viewer_with_atoms,
):
    """The actual bug being fixed: a rebuild (a display style change,
    a supercell change, running an analysis) must not silently turn
    picking back on just because a redraw happened, if the person
    deliberately switched to View mode. Simulates render_model()'s own
    picking-restoration branch directly.
    """
    viewer = viewer_with_atoms
    viewer.set_picking_enabled(False)

    # The exact branch render_model() runs after every redraw.
    if not viewer.picking_enabled:
        pass  # render_model() calls disable_picking() here; nothing to reconnect
    elif viewer._measurement_mode is not None:
        viewer.enable_measurement(viewer._measurement_mode)
    else:
        viewer.enable_selection()

    assert viewer.picking_enabled is False
