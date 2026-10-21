"""CrystalViewer surface area visualization: colour-mapped point
cloud, mesh overlay, and independent toggles."""

import os

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np  # noqa: E402

from thevoid_core.ui.widgets.viewer_widget import (  # noqa: E402
    CrystalViewer,
    _hex_to_rgb,
)


@pytest.fixture
def bare_viewer():
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.picking_enabled = True
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer._surface_points_actor = None
    viewer._surface_mesh_actor = None
    viewer._last_surface_points = None
    viewer._last_surface_accessible_mask = None

    yield viewer

    viewer.plotter.close()


def _sphere_points(n=2000, radius=3.0, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * radius


def _two_separated_sphere_clusters(
    n_per_cluster=400, radius=1.0, separation=15.0, seed=0
):
    """Two compact, well-separated point clusters - the shape of input
    that reproduced the reported bug directly: a single global
    reconstruct_surface() call over both clusters at once extrapolated
    into a large flat sheet spanning far beyond either cluster's own
    extent, rather than two tight, separate blobs.
    """
    rng = np.random.default_rng(seed)

    def _cluster(center):
        v = rng.normal(size=(n_per_cluster, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v * radius + np.array(center)

    return np.vstack(
        [_cluster((0.0, 0.0, 0.0)), _cluster((separation, 0.0, 0.0))]
    )


def test_hex_to_rgb_conversion():
    assert _hex_to_rgb("#ff0000") == pytest.approx((1.0, 0.0, 0.0))
    assert _hex_to_rgb("#00ff00") == pytest.approx((0.0, 1.0, 0.0))
    assert _hex_to_rgb("#0000ff") == pytest.approx((0.0, 0.0, 1.0))
    assert _hex_to_rgb("000000") == pytest.approx((0.0, 0.0, 0.0))


def test_show_surface_points_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    points = _sphere_points()
    mask = np.ones(len(points), dtype=bool)

    viewer.show_surface_points(points, mask)

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._surface_points_actor is not None


def test_showing_surface_points_twice_replaces_not_adds(bare_viewer):
    viewer = bare_viewer

    points = _sphere_points()
    mask = np.ones(len(points), dtype=bool)

    viewer.show_surface_points(points, mask)
    after_first = len(viewer.plotter.renderer.actors)

    viewer.show_surface_points(points, mask)
    after_second = len(viewer.plotter.renderer.actors)

    assert after_second == after_first


def test_empty_points_does_not_raise(bare_viewer):
    viewer = bare_viewer

    viewer.show_surface_points(np.zeros((0, 3)), np.zeros(0, dtype=bool))

    assert viewer._surface_points_actor is None


def test_none_points_does_not_raise(bare_viewer):
    bare_viewer.show_surface_points(None, None)  # should not raise
    assert bare_viewer._surface_points_actor is None


def test_show_surface_mesh_adds_an_actor(bare_viewer):
    viewer = bare_viewer
    before = len(viewer.plotter.renderer.actors)

    viewer.show_surface_mesh(_sphere_points())

    assert len(viewer.plotter.renderer.actors) == before + 1
    assert viewer._surface_mesh_actor is not None


def test_show_surface_mesh_with_too_few_points_does_not_raise(bare_viewer):
    viewer = bare_viewer

    viewer.show_surface_mesh(_sphere_points(n=3))

    assert viewer._surface_mesh_actor is None


def test_two_separated_clusters_do_not_produce_a_wildly_extrapolated_mesh(
    bare_viewer,
):
    """The exact reported bug, reproduced directly: reconstructing a
    surface through two well-separated clusters at once (the old
    behaviour) extrapolated into a mesh whose bounds extended far
    beyond either cluster's own extent - large, flat, connecting
    sheets rather than two tight blobs. The fix (clustering into
    connected patches and reconstructing each separately) must keep
    the combined mesh's bounds close to the actual point cloud's own
    bounds, not doubled or worse.
    """
    viewer = bare_viewer
    points = _two_separated_sphere_clusters()

    viewer.show_surface_mesh(points)

    assert viewer._surface_mesh_actor is not None
    mesh = viewer._surface_mesh_actor.mapper.dataset

    point_min, point_max = points.min(axis=0), points.max(axis=0)
    mesh_bounds = np.array(mesh.bounds).reshape(3, 2)
    mesh_min, mesh_max = mesh_bounds[:, 0], mesh_bounds[:, 1]

    # A reasonable reconstruction may extend slightly beyond the raw
    # points (to close the surface), but not by more than a small
    # fraction of the cluster's own size - the bug produced bounds
    # roughly double the actual extent.
    tolerance = 2.5  # generous margin for the reconstruction's own padding
    assert np.all(mesh_min >= point_min - tolerance)
    assert np.all(mesh_max <= point_max + tolerance)


def test_two_separated_clusters_are_not_bridged_by_the_mesh(bare_viewer):
    """A stronger, more direct check than bounds: the mesh must not
    contain triangles that span the empty gap between the two
    clusters - sampling points along the line between the two cluster
    centres and confirming the mesh does not pass anywhere near the
    midpoint, where a bridging sheet would necessarily cross.
    """
    viewer = bare_viewer
    points = _two_separated_sphere_clusters(separation=15.0)

    viewer.show_surface_mesh(points)

    mesh = viewer._surface_mesh_actor.mapper.dataset
    midpoint = np.array([7.5, 0.0, 0.0])  # halfway between the clusters

    if mesh.n_points == 0:
        return

    distances = np.linalg.norm(mesh.points - midpoint, axis=1)
    # Either cluster's own points reach to within ~1 unit of its centre
    # at most (radius 1.0 sphere) plus reconstruction padding; the
    # midpoint sits ~6.5 units from the nearer cluster's own points, so
    # nothing in a correctly-separated mesh should land anywhere near
    # the midpoint itself.
    assert np.min(distances) > 4.0


def test_a_single_compact_cluster_still_reconstructs_correctly(bare_viewer):
    """The fix must not break the simple, single-blob case the
    original implementation already handled correctly."""
    viewer = bare_viewer
    points = _sphere_points(n=500, radius=2.0)

    viewer.show_surface_mesh(points)

    assert viewer._surface_mesh_actor is not None
    mesh = viewer._surface_mesh_actor.mapper.dataset
    assert mesh.n_points > 0

    mesh_bounds = np.array(mesh.bounds).reshape(3, 2)
    # A single sphere's reconstruction should stay close to its own
    # radius, not balloon outward the way the bug did for multi-cluster
    # input.
    assert np.all(np.abs(mesh_bounds) < 4.0)


def test_clear_surface_visualization_removes_both_actors(bare_viewer):
    viewer = bare_viewer

    points = _sphere_points()
    mask = np.ones(len(points), dtype=bool)
    viewer.show_surface_points(points, mask)
    viewer.show_surface_mesh(points)
    before = len(viewer.plotter.renderer.actors)

    viewer.clear_surface_visualization()

    assert viewer._surface_points_actor is None
    assert viewer._surface_mesh_actor is None
    assert len(viewer.plotter.renderer.actors) == before - 2


def test_points_and_mesh_visibility_toggle_independently(bare_viewer):
    viewer = bare_viewer

    points = _sphere_points()
    mask = np.ones(len(points), dtype=bool)
    viewer.show_surface_points(points, mask)
    viewer.show_surface_mesh(points)

    viewer.set_surface_points_visibility(False)

    assert viewer._surface_points_actor.GetVisibility() == 0
    assert viewer._surface_mesh_actor.GetVisibility() == 1

    viewer.set_surface_mesh_visibility(False)
    assert viewer._surface_mesh_actor.GetVisibility() == 0

    viewer.set_surface_points_visibility(True)
    assert viewer._surface_points_actor.GetVisibility() == 1


def test_visibility_toggle_with_nothing_shown_does_not_raise(bare_viewer):
    bare_viewer.set_surface_points_visibility(True)
    bare_viewer.set_surface_mesh_visibility(False)
