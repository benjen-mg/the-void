"""CrystalViewer guest accessibility visualization: accessible
channels, blocked regions, and bottleneck markers, each independently
toggleable."""

import os

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np  # noqa: E402

from thevoid_core.ui.widgets.viewer_widget import CrystalViewer  # noqa: E402


@pytest.fixture
def bare_viewer():
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.picking_enabled = True
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer._guest_accessible_actor = None
    viewer._guest_blocked_actor = None
    viewer._guest_bottleneck_actor = None
    viewer._last_guest_accessible_points = None
    viewer._last_guest_blocked_points = None
    viewer._last_guest_bottleneck_points = None

    yield viewer

    viewer.plotter.close()


def _points(n=100, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)) * 3.0


def test_show_accessible_channels_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_guest_accessible_channels(_points())

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._guest_accessible_actor is not None


def test_show_blocked_regions_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_guest_blocked_regions(_points())

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._guest_blocked_actor is not None


def test_show_bottlenecks_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_guest_bottlenecks(_points(n=5))

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._guest_bottleneck_actor is not None


def test_all_three_visualizations_are_independent(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_guest_accessible_channels(_points())
    viewer.show_guest_blocked_regions(_points())
    viewer.show_guest_bottlenecks(_points(n=5))

    assert len(viewer.plotter.renderer.actors) == before + 3


def test_showing_accessible_channels_twice_replaces_not_adds(bare_viewer):
    viewer = bare_viewer

    viewer.show_guest_accessible_channels(_points())
    after_first = len(viewer.plotter.renderer.actors)

    viewer.show_guest_accessible_channels(_points())
    after_second = len(viewer.plotter.renderer.actors)

    assert after_second == after_first


def test_empty_and_none_points_do_not_raise(bare_viewer):
    viewer = bare_viewer

    viewer.show_guest_accessible_channels(np.zeros((0, 3)))
    assert viewer._guest_accessible_actor is None

    viewer.show_guest_blocked_regions(None)
    assert viewer._guest_blocked_actor is None

    viewer.show_guest_bottlenecks(np.zeros((0, 3)))
    assert viewer._guest_bottleneck_actor is None


def test_clear_removes_all_three_actors(bare_viewer):
    viewer = bare_viewer

    viewer.show_guest_accessible_channels(_points())
    viewer.show_guest_blocked_regions(_points())
    viewer.show_guest_bottlenecks(_points(n=5))
    before = len(viewer.plotter.renderer.actors)

    viewer.clear_guest_accessibility_visualization()

    assert viewer._guest_accessible_actor is None
    assert viewer._guest_blocked_actor is None
    assert viewer._guest_bottleneck_actor is None
    assert len(viewer.plotter.renderer.actors) == before - 3


def test_visibility_toggles_are_independent(bare_viewer):
    viewer = bare_viewer

    viewer.show_guest_accessible_channels(_points())
    viewer.show_guest_blocked_regions(_points())
    viewer.show_guest_bottlenecks(_points(n=5))

    viewer.set_guest_accessible_visibility(False)

    assert viewer._guest_accessible_actor.GetVisibility() == 0
    assert viewer._guest_blocked_actor.GetVisibility() == 1
    assert viewer._guest_bottleneck_actor.GetVisibility() == 1

    viewer.set_guest_blocked_visibility(False)
    assert viewer._guest_blocked_actor.GetVisibility() == 0

    viewer.set_guest_bottleneck_visibility(False)
    assert viewer._guest_bottleneck_actor.GetVisibility() == 0

    viewer.set_guest_accessible_visibility(True)
    assert viewer._guest_accessible_actor.GetVisibility() == 1


def test_visibility_toggle_with_nothing_shown_does_not_raise(bare_viewer):
    bare_viewer.set_guest_accessible_visibility(True)
    bare_viewer.set_guest_blocked_visibility(False)
    bare_viewer.set_guest_bottleneck_visibility(True)
