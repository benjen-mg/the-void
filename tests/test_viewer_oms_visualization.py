"""CrystalViewer OMS adsorption-marker visualization."""

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
    viewer._oms_marker_actor = None
    viewer._last_oms_marker_points = None

    yield viewer

    viewer.plotter.close()


def test_show_markers_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_oms_adsorption_markers(np.array([[0.0, 0.0, 1.0]]))

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._oms_marker_actor is not None


def test_showing_markers_twice_replaces_not_adds(bare_viewer):
    viewer = bare_viewer

    viewer.show_oms_adsorption_markers(np.array([[0.0, 0.0, 1.0]]))
    after_first = len(viewer.plotter.renderer.actors)

    viewer.show_oms_adsorption_markers(np.array([[1.0, 1.0, 1.0]]))
    after_second = len(viewer.plotter.renderer.actors)

    assert after_second == after_first


def test_empty_and_none_points_do_not_raise(bare_viewer):
    viewer = bare_viewer

    viewer.show_oms_adsorption_markers(np.zeros((0, 3)))
    assert viewer._oms_marker_actor is None

    viewer.show_oms_adsorption_markers(None)
    assert viewer._oms_marker_actor is None


def test_clear_removes_the_actor(bare_viewer):
    viewer = bare_viewer

    viewer.show_oms_adsorption_markers(np.array([[0.0, 0.0, 1.0]]))
    before = len(viewer.plotter.renderer.actors)

    viewer.clear_oms_marker_visualization()

    assert viewer._oms_marker_actor is None
    assert len(viewer.plotter.renderer.actors) == before - 1


def test_set_visibility_toggles_without_removing(bare_viewer):
    viewer = bare_viewer

    viewer.show_oms_adsorption_markers(np.array([[0.0, 0.0, 1.0]]))
    actor = viewer._oms_marker_actor

    viewer.set_oms_marker_visibility(False)
    assert viewer._oms_marker_actor is actor
    assert viewer._oms_marker_actor.GetVisibility() == 0

    viewer.set_oms_marker_visibility(True)
    assert viewer._oms_marker_actor.GetVisibility() == 1


def test_visibility_toggle_with_nothing_shown_does_not_raise(bare_viewer):
    bare_viewer.set_oms_marker_visibility(True)
