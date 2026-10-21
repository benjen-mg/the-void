"""Regression test for View > Reset View doing nothing.

plotter.reset_camera() only re-fits zoom/pan to the camera's *current*
orientation - it does not change which direction the camera faces. After
the user rotates the scene with the mouse, calling it alone leaves the
rotated orientation in place, which can look exactly like the button did
nothing. CrystalViewer.reset_view() must restore the original
orientation, not just re-fit bounds from wherever the camera currently
points.
"""

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

import os  # noqa: E402

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from thevoid_core.ui.widgets.viewer_widget import CrystalViewer  # noqa: E402


@pytest.fixture
def bare_viewer():
    """A CrystalViewer with a real off-screen plotter, bypassing the Qt
    widget construction this sandbox can't drive (no GPU/X server).
    """
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.picking_enabled = True
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer.plotter.add_mesh(pv.Sphere())
    viewer.plotter.camera_position = CrystalViewer.DEFAULT_CAMERA_POSITION
    viewer.plotter.reset_camera()

    yield viewer

    viewer.plotter.close()


def test_reset_view_restores_orientation_after_rotation(bare_viewer):
    original_position = bare_viewer.plotter.camera.position

    bare_viewer.plotter.camera.Azimuth(120)
    bare_viewer.plotter.camera.Elevation(-40)
    rotated_position = bare_viewer.plotter.camera.position

    assert rotated_position != original_position  # sanity: rotation worked

    bare_viewer.reset_view()

    assert bare_viewer.plotter.camera.position == pytest.approx(
        original_position
    )


def test_plain_reset_camera_alone_would_not_have_caught_this(bare_viewer):
    """Documents why reset_view() can't just call plotter.reset_camera():
    that alone does not restore orientation, which is the bug this file
    guards against regressing to.
    """
    bare_viewer.plotter.camera.Azimuth(120)
    rotated_position = bare_viewer.plotter.camera.position

    bare_viewer.plotter.reset_camera()  # the old, buggy implementation

    # Same direction as after the rotation (reset_camera re-fits
    # distance along whatever direction the camera already faces, it
    # doesn't change the direction) - allow for the tiny floating-point
    # drift reset_camera's bounds refit introduces.
    assert bare_viewer.plotter.camera.position == pytest.approx(
        rotated_position
    )


def test_reset_view_also_refits_zoom_after_the_user_zooms_in(bare_viewer):
    """VTK's Zoom() narrows the field of view (view_angle), not camera
    distance - reset_view must undo that too, not just orientation.
    """
    default_angle = bare_viewer.plotter.camera.view_angle

    bare_viewer.plotter.camera.Zoom(50)
    zoomed_angle = bare_viewer.plotter.camera.view_angle
    assert zoomed_angle != default_angle  # sanity: zoom worked

    bare_viewer.reset_view()

    assert bare_viewer.plotter.camera.view_angle == pytest.approx(
        default_angle
    )


def test_reset_view_is_idempotent(bare_viewer):
    bare_viewer.plotter.camera.Azimuth(75)

    bare_viewer.reset_view()
    first = bare_viewer.plotter.camera.position

    bare_viewer.reset_view()
    second = bare_viewer.plotter.camera.position

    assert first == pytest.approx(second)
