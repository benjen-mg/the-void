"""Regression test: all three Display Style menu options silently
rendered as Ball-and-Stick.

Root cause: QAction.setData(DisplayStyle.X) followed by action.data()
returns a plain str, not the DisplayStyle enum instance - Qt's data
marshalling silently unwraps a str-subclassed Enum. render_model() used
'is' comparisons against DisplayStyle members, which then always
evaluated False for a value that came back from Qt, so every style
selection fell through to the else branch (Ball-and-Stick) regardless
of what was actually selected in the menu.

These tests exercise the real PySide6 QAction round-trip rather than
just constructing a RenderModel directly, since that's exactly the step
a mock or a hand-built DisplayStyle value would skip past without
catching this.
"""

import os

import pytest

pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtGui import QAction  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.data.element_colors import ElementColorProvider  # noqa: E402
from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services.bonds.bond_model import Bond  # noqa: E402
from thevoid_core.services.structure.render_model import (  # noqa: E402
    CompletionMode,
    RenderModelBuilder,
)
from thevoid_core.ui.widgets.viewer_widget import (  # noqa: E402
    CrystalViewer,
    DisplayStyle,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class FakeSpecies:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeSite:
    def __init__(self, symbol, coords):
        self.species = {FakeSpecies(symbol): 1.0}
        self.coords = coords
        self.frac_coords = [c / 10.0 for c in coords]


class FakeLattice:
    def __init__(self, matrix):
        self.matrix = matrix


class FakeStructure:
    def __init__(self, matrix, sites):
        self.lattice = FakeLattice(matrix)
        self.sites = sites

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, index):
        return self.sites[index]


CUBIC_10A = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def test_qaction_data_roundtrip_loses_the_enum_type(qt_app):
    """Pins the actual Qt behaviour this bug depended on. If a future
    PySide6 version changes this, the other tests below would still
    pass (DisplayStyle(...) handles both cases), but this documents
    exactly what was happening.
    """
    action = QAction("test")
    action.setData(DisplayStyle.SPACE_FILLING)

    retrieved = action.data()

    assert retrieved == DisplayStyle.SPACE_FILLING  # value survives
    assert retrieved is not DisplayStyle.SPACE_FILLING  # identity doesn't


def _bare_viewer():
    """A CrystalViewer with a real off-screen plotter but skipping Qt
    widget construction (no GPU/X server here to drive QtInteractor).

    Sets every attribute render_model() reads that __init__ would
    normally set, so this stays correct as the viewer grows new state -
    one place to update rather than three ad hoc constructions.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.color_provider = ElementColorProvider()
    viewer.radius_provider = ElementRadiusProvider()
    viewer.show_unit_cell = False
    viewer.display_style = DisplayStyle.BALL_AND_STICK
    viewer.opacity = 1.0
    viewer._measurement_mode = None
    viewer._measurement_picks = []
    viewer._measurement_actor_names = []
    viewer._current_model = None
    viewer.on_measurement = None
    viewer.picking_enabled = True
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))

    return viewer


def _actor_count_for_style(qt_app, model, style):
    """Exactly what MainWindow._on_display_style_changed does: round
    the style through a real QAction, then through DisplayStyle(...).
    """
    action = QAction("style")
    action.setData(style)

    resolved = DisplayStyle(action.data())

    viewer = _bare_viewer()
    viewer.display_style = resolved
    viewer.render_model(model, lattice_matrix=None, reset_camera=True)

    count = len(viewer.plotter.renderer.actors)
    viewer.plotter.close()

    return count


@pytest.fixture
def three_atom_model():
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [4.0, 5.0, 5.0]),
            FakeSite("O", [5.2, 5.0, 5.0]),
            FakeSite("Na", [8.0, 8.0, 8.0]),  # isolated, no bonds
        ],
    )
    bonds = [Bond(atom1=0, atom2=1, image=(0, 0, 0), length=1.2)]

    return RenderModelBuilder().build(
        structure, bonds, mode=CompletionMode.CELL_ONLY
    )


def test_ball_and_stick_through_qt_has_more_actors_than_either_other_style(
    qt_app, three_atom_model
):
    """Ball-and-Stick draws both atoms and bonds as separate meshes;
    the other two each draw only one or the other, so it alone should
    have the most actors for this fixture.
    """
    counts = {
        style: _actor_count_for_style(qt_app, three_atom_model, style)
        for style in (
            DisplayStyle.BALL_AND_STICK,
            DisplayStyle.SPACE_FILLING,
            DisplayStyle.WIREFRAME,
        )
    }

    assert counts[DisplayStyle.BALL_AND_STICK] > counts[DisplayStyle.SPACE_FILLING]
    assert counts[DisplayStyle.BALL_AND_STICK] > counts[DisplayStyle.WIREFRAME]


def test_space_filling_and_wireframe_draw_different_geometry_through_qt(
    qt_app, three_atom_model
):
    """Space-Filling and Wireframe happen to produce the same actor
    *count* for this fixture (3 atom meshes vs. 2 bond meshes + 1
    isolated-atom mesh) - actor count alone doesn't distinguish them.
    Their total point counts must still differ: Space-Filling glyphs
    full van-der-Waals spheres onto every atom, Wireframe glyphs thin
    bond cylinders plus one small sphere for the single isolated atom,
    and those are very different amounts of geometry.
    """

    def total_points(style):
        action = QAction("style")
        action.setData(style)
        resolved = DisplayStyle(action.data())

        viewer = _bare_viewer()
        viewer.display_style = resolved
        viewer.render_model(three_atom_model, lattice_matrix=None)

        points = sum(
            actor.mapper.dataset.n_points
            for actor in viewer.plotter.renderer.actors.values()
            if getattr(actor, "mapper", None) is not None
        )
        viewer.plotter.close()

        return points

    space_filling_points = total_points(DisplayStyle.SPACE_FILLING)
    wireframe_points = total_points(DisplayStyle.WIREFRAME)

    assert space_filling_points != wireframe_points


def test_space_filling_through_qt_draws_no_bonds(qt_app, three_atom_model):
    # Ball-and-stick: 3 element atom meshes + 2 element bond meshes = 5.
    # Space-filling: 3 element atom meshes, no bonds = 3.
    ball_and_stick = _actor_count_for_style(
        qt_app, three_atom_model, DisplayStyle.BALL_AND_STICK
    )
    space_filling = _actor_count_for_style(
        qt_app, three_atom_model, DisplayStyle.SPACE_FILLING
    )

    assert space_filling < ball_and_stick


def test_wireframe_through_qt_gives_the_isolated_atom_a_sphere(
    qt_app, three_atom_model
):
    """Wireframe draws no atom spheres except for unbonded atoms - the
    lone Na here must still produce visible geometry, not zero actors.
    """
    count = _actor_count_for_style(
        qt_app, three_atom_model, DisplayStyle.WIREFRAME
    )

    assert count > 0


def test_render_model_handles_a_raw_plain_string_style(
    qt_app, three_atom_model
):
    """The precise historical bug, reproduced directly: main_window's
    handler used to do `self.viewer.display_style = action.data()` with
    no DisplayStyle(...) conversion. Since Qt's data round-trip returns
    a plain str (pinned above), display_style ended up holding the
    right *value* but not a real DisplayStyle instance. render_model()
    compared it with 'is' against DisplayStyle members, which is False
    for a plain str even when the value matches, so every style silently
    rendered as Ball-and-Stick.

    This sets display_style to a raw plain string directly - the exact
    value action.data() actually returns, bypassing the DisplayStyle(...)
    boundary fix entirely - to confirm render_model() itself is robust
    regardless of whether a caller remembers to convert. Both fixes
    (the boundary conversion in the handler, and == instead of is here)
    matter: this test only exercises the second one.
    """
    plain_string_style = "space_filling"
    assert type(plain_string_style) is str  # not DisplayStyle - the point

    viewer = _bare_viewer()
    viewer.display_style = plain_string_style

    viewer.render_model(three_atom_model, lattice_matrix=None)

    space_filling_count = len(viewer.plotter.renderer.actors)
    viewer.plotter.close()

    # Compare against the genuine enum value to confirm they match -
    # if render_model silently fell back to Ball-and-Stick for the raw
    # string, this would differ (Ball-and-Stick has 5 actors here vs.
    # Space-Filling's 3).
    ball_and_stick_count = _actor_count_for_style(
        qt_app, three_atom_model, DisplayStyle.BALL_AND_STICK
    )

    assert space_filling_count != ball_and_stick_count
