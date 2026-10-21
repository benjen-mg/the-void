"""CrystalViewer.clear_all_analysis_visualizations(): the single
"clear everything analysis-related" method backing both the manual
reset action and the automatic clear-before-showing-new-results
behaviour every analysis feature now uses.
"""

import os

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np  # noqa: E402

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
def viewer_with_everything_shown():
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer.display_style = DisplayStyle.BALL_AND_STICK
    viewer.radius_provider = ElementRadiusProvider()
    viewer.picking_enabled = True
    viewer._group_highlight_actor = None
    viewer._pore_actor = None
    viewer._last_pore_points = None
    viewer._surface_points_actor = None
    viewer._surface_mesh_actor = None
    viewer._last_surface_points = None
    viewer._last_surface_accessible_mask = None
    viewer._guest_accessible_actor = None
    viewer._guest_blocked_actor = None
    viewer._guest_bottleneck_actor = None
    viewer._last_guest_accessible_points = None
    viewer._last_guest_blocked_points = None
    viewer._last_guest_bottleneck_points = None
    viewer._oms_marker_actor = None
    viewer._last_oms_marker_points = None
    viewer._selection_actor = None
    viewer._selected_atom_index = None

    viewer._current_model = FakeModel(
        [FakeAtom("Zn", (0.0, 0.0, 0.0)), FakeAtom("O", (2.0, 0.0, 0.0))]
    )

    points = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])

    # A real, densely-sampled cluster for the mesh call specifically -
    # show_surface_mesh now groups points into connected surface
    # patches and requires a reasonable minimum per patch, so a
    # handful of duplicated points (enough for the old, simpler
    # implementation) no longer produces a patch large enough to
    # reconstruct anything from.
    rng = np.random.default_rng(0)
    sphere_direction = rng.normal(size=(30, 3))
    sphere_direction /= np.linalg.norm(sphere_direction, axis=1, keepdims=True)
    mesh_points = sphere_direction * 2.0

    # Show one of every kind of analysis overlay.
    viewer.highlight_group([0, 1], viewer.OMS_HIGHLIGHT_COLOR)
    viewer.show_pore_points(points)
    viewer.show_surface_points(points, np.array([True, False]))
    viewer.show_surface_mesh(mesh_points)
    viewer.show_guest_accessible_channels(points)
    viewer.show_guest_blocked_regions(points)
    viewer.show_guest_bottlenecks(points)
    viewer.show_oms_adsorption_markers(points)

    yield viewer

    viewer.plotter.close()


def test_clears_every_analysis_actor(viewer_with_everything_shown):
    viewer = viewer_with_everything_shown

    viewer.clear_all_analysis_visualizations()

    assert viewer._group_highlight_actor is None
    assert viewer._pore_actor is None
    assert viewer._surface_points_actor is None
    assert viewer._surface_mesh_actor is None
    assert viewer._guest_accessible_actor is None
    assert viewer._guest_blocked_actor is None
    assert viewer._guest_bottleneck_actor is None
    assert viewer._oms_marker_actor is None


def test_removes_every_analysis_actor_from_the_scene(
    viewer_with_everything_shown,
):
    viewer = viewer_with_everything_shown
    before = len(viewer.plotter.renderer.actors)
    assert before >= 8  # one per overlay type shown in the fixture

    viewer.clear_all_analysis_visualizations()

    # Every actor added by the fixture's eight show_* calls should be
    # gone - only whatever (if anything) existed beforehand remains.
    after = len(viewer.plotter.renderer.actors)
    assert after <= before - 8


def test_does_not_raise_when_nothing_is_shown():
    pv.OFF_SCREEN = True
    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(100, 100))
    viewer._group_highlight_actor = None
    viewer._pore_actor = None
    viewer._surface_points_actor = None
    viewer._surface_mesh_actor = None
    viewer._guest_accessible_actor = None
    viewer._guest_blocked_actor = None
    viewer._guest_bottleneck_actor = None
    viewer._oms_marker_actor = None

    viewer.clear_all_analysis_visualizations()  # should not raise

    viewer.plotter.close()


def test_calling_twice_in_a_row_does_not_raise(viewer_with_everything_shown):
    viewer = viewer_with_everything_shown

    viewer.clear_all_analysis_visualizations()
    viewer.clear_all_analysis_visualizations()  # should not raise
