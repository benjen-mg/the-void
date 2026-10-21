"""CrystalViewer.highlight_group: the single merged outline used for
MOF linkers, metal nodes, and (eventually) coordination spheres.
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
    viewer.display_style = DisplayStyle.BALL_AND_STICK
    viewer.radius_provider = ElementRadiusProvider()
    viewer._group_highlight_actor = None

    viewer._current_model = FakeModel(
        [
            FakeAtom("Zn", (0.0, 0.0, 0.0), site_index=0),
            FakeAtom("O", (1.5, 0.0, 0.0), site_index=1),
            FakeAtom("Zn", (3.0, 0.0, 0.0), site_index=2),
            FakeAtom("C", (10.0, 10.0, 10.0), site_index=3),
        ]
    )

    yield viewer

    viewer.plotter.close()


def test_highlight_group_adds_an_actor(viewer_with_atoms):
    viewer = viewer_with_atoms
    before = len(viewer.plotter.renderer.actors)

    viewer.highlight_group([0, 1, 2], "#ff8800")

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._group_highlight_actor is not None


def test_highlighting_a_second_group_replaces_the_first(viewer_with_atoms):
    viewer = viewer_with_atoms

    viewer.highlight_group([0, 1, 2], "#ff8800")
    after_first = len(viewer.plotter.renderer.actors)

    viewer.highlight_group([3], "#00aaff")
    after_second = len(viewer.plotter.renderer.actors)

    assert after_second == after_first


def test_clear_group_highlight_removes_the_actor(viewer_with_atoms):
    viewer = viewer_with_atoms

    viewer.highlight_group([0, 1, 2], "#ff8800")
    before = len(viewer.plotter.renderer.actors)

    viewer.clear_group_highlight()

    assert viewer._group_highlight_actor is None
    assert len(viewer.plotter.renderer.actors) == before - 1


def test_highlight_group_with_empty_list_does_nothing_and_does_not_raise(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    before = len(viewer.plotter.renderer.actors)

    viewer.highlight_group([], "#ff8800")

    assert viewer._group_highlight_actor is None
    assert len(viewer.plotter.renderer.actors) == before


def test_highlight_group_with_no_current_model_does_not_raise(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    viewer._current_model = None

    viewer.highlight_group([0, 1], "#ff8800")  # should not raise

    assert viewer._group_highlight_actor is None


def test_highlight_group_ignores_out_of_range_indices(viewer_with_atoms):
    viewer = viewer_with_atoms

    # Index 99 doesn't exist; the valid ones should still be highlighted.
    viewer.highlight_group([0, 99], "#ff8800")

    assert viewer._group_highlight_actor is not None


def test_highlight_group_all_out_of_range_does_nothing(viewer_with_atoms):
    viewer = viewer_with_atoms

    viewer.highlight_group([99, 100], "#ff8800")

    assert viewer._group_highlight_actor is None


def test_render_model_clears_the_group_highlight(viewer_with_atoms):
    """A rebuild (new structure, display style change, ...) must not
    leave a stale group highlight pointing at atoms that may no longer
    exist at the same indices.
    """
    viewer = viewer_with_atoms
    viewer.color_provider = None  # not needed since model.atoms is empty
    viewer.show_unit_cell = False

    viewer.highlight_group([0, 1], "#ff8800")
    assert viewer._group_highlight_actor is not None

    viewer.render_model(FakeModel([]), lattice_matrix=None, reset_camera=False)

    assert viewer._group_highlight_actor is None
