"""CrystalViewer pore-space visualization: show/clear/toggle."""

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
    viewer._pore_actor = None
    viewer._last_pore_points = None
    viewer._last_pore_color = CrystalViewer.PORE_POINT_COLOR

    yield viewer

    viewer.plotter.close()


def test_show_pore_points_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    viewer.show_pore_points(points)

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._pore_actor is not None


def test_showing_pore_points_twice_replaces_not_adds(bare_viewer):
    viewer = bare_viewer

    viewer.show_pore_points(np.array([[0.0, 0.0, 0.0]]))
    after_first = len(viewer.plotter.renderer.actors)

    viewer.show_pore_points(np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    after_second = len(viewer.plotter.renderer.actors)

    assert after_second == after_first


def test_clear_pore_visualization_removes_the_actor(bare_viewer):
    viewer = bare_viewer

    viewer.show_pore_points(np.array([[0.0, 0.0, 0.0]]))
    before = len(viewer.plotter.renderer.actors)

    viewer.clear_pore_visualization()

    assert viewer._pore_actor is None
    assert len(viewer.plotter.renderer.actors) == before - 1


def test_empty_points_array_does_not_raise_and_adds_nothing(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_pore_points(np.zeros((0, 3)))

    assert viewer._pore_actor is None
    assert len(viewer.plotter.renderer.actors) == before


def test_none_points_does_not_raise(bare_viewer):
    viewer = bare_viewer

    viewer.show_pore_points(None)  # should not raise

    assert viewer._pore_actor is None


def test_set_pore_visibility_does_not_remove_the_actor(bare_viewer):
    """Toggling off must hide, not discard - toggling back on afterwards
    should not require re-running analysis.
    """
    viewer = bare_viewer

    viewer.show_pore_points(np.array([[0.0, 0.0, 0.0]]))
    total_with_pore = len(viewer.plotter.renderer.actors)

    viewer.set_pore_visibility(False)

    # The actor itself is still present (just hidden), so the total
    # actor count in the scene does not change.
    assert len(viewer.plotter.renderer.actors) == total_with_pore
    assert viewer._pore_actor is not None
    assert viewer._pore_actor.GetVisibility() == 0

    viewer.set_pore_visibility(True)
    assert viewer._pore_actor.GetVisibility() == 1


def test_set_pore_visibility_with_nothing_shown_does_not_raise(bare_viewer):
    bare_viewer.set_pore_visibility(True)  # should not raise
    bare_viewer.set_pore_visibility(False)  # should not raise
