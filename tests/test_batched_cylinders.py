"""CrystalViewer._batched_cylinders: the vectorized bond-geometry
builder that replaced constructing one pv.Cylinder() per half-bond.

That per-object approach measured at roughly 1.5ms per call regardless
of size - about 24 seconds for the ~17,000 half-bonds a 3x3x3 supercell
of a moderately dense molecular crystal produces, long enough to look
like a hang. These tests pin the geometry this replacement must
produce: same radius regardless of length (glyph()'s vector-scale mode
was tried first and rejected - it scales isotropically, so a 5x longer
bond came out 5x fatter too), correct endpoints, and sane behaviour on
edge cases like a bond aligned with the arbitrary reference axis used
internally.
"""

import pytest

pytest.importorskip("numpy")
pv = pytest.importorskip("pyvista")

import numpy as np  # noqa: E402

from thevoid_core.ui.widgets.viewer_widget import CrystalViewer  # noqa: E402


def build(starts, ends, radius=0.075, resolution=12):
    return CrystalViewer._batched_cylinders(
        np.asarray(starts, dtype=float),
        np.asarray(ends, dtype=float),
        radius=radius,
        resolution=resolution,
    )


def test_empty_input_returns_an_empty_mesh():
    mesh = build([], [])

    assert mesh.n_points == 0
    assert mesh.n_cells == 0


def test_point_and_cell_counts_match_the_batch_size():
    starts = [[0, 0, 0], [5, 0, 0], [10, 0, 0]]
    ends = [[1, 0, 0], [6, 0, 0], [11, 0, 0]]

    mesh = build(starts, ends, resolution=12)

    # 2 rings x resolution points per cylinder; resolution quad faces
    # per cylinder.
    assert mesh.n_points == 3 * 2 * 12
    assert mesh.n_cells == 3 * 12


def test_radius_is_independent_of_length():
    """The property glyph()'s isotropic vector-scale mode couldn't give:
    a short and a long cylinder must have the identical cross-section.
    """
    starts = np.array([[0, 0, 0], [100, 0, 0]], dtype=float)
    ends = np.array([[1, 0, 0], [130, 0, 0]], dtype=float)  # lengths 1, 30

    mesh = build(starts, ends, radius=0.075, resolution=16)

    points = mesh.points
    near_short = points[np.linalg.norm(points - [0.5, 0, 0], axis=1) < 2]
    near_long = points[np.linalg.norm(points - [115, 0, 0], axis=1) < 20]

    short_radius = np.linalg.norm(near_short[:, 1:3], axis=1).max()
    long_radius = np.linalg.norm(near_long[:, 1:3], axis=1).max()

    assert short_radius == pytest.approx(0.075, abs=1e-9)
    assert long_radius == pytest.approx(0.075, abs=1e-9)


def test_endpoints_match_the_requested_start_and_end():
    start = np.array([1.0, 2.0, 3.0])
    end = np.array([4.0, 2.0, 3.0])

    mesh = build([start], [end], radius=0.05, resolution=12)

    assert mesh.points[:, 0].min() == pytest.approx(start[0])
    assert mesh.points[:, 0].max() == pytest.approx(end[0])


def test_a_cylinder_along_the_internal_reference_axis_is_still_valid():
    """The builder picks +Z as a reference to construct a perpendicular
    frame, falling back to +X only when the bond direction is itself
    nearly parallel to +Z. A bond running along +Z is exactly the case
    that fallback exists for - it must not produce degenerate geometry.
    """
    mesh = build([[0, 0, 0]], [[0, 0, 5]], radius=0.1, resolution=12)

    radii = np.linalg.norm(mesh.points[:, 0:2], axis=1)

    assert np.allclose(radii, 0.1, atol=1e-9)
    assert mesh.points[:, 2].min() == pytest.approx(0.0)
    assert mesh.points[:, 2].max() == pytest.approx(5.0)


def test_many_random_orientations_all_keep_constant_radius():
    """Broad sweep: whatever direction a bond points, its cross-section
    radius must come out the same.
    """
    rng = np.random.default_rng(7)

    starts = rng.uniform(-5, 5, size=(50, 3))
    directions = rng.normal(size=(50, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    lengths = rng.uniform(0.3, 4.0, size=(50, 1))
    ends = starts + directions * lengths

    mesh = build(starts, ends, radius=0.06, resolution=10)

    # For each cylinder, every ring point should sit exactly `radius`
    # from the centreline (start + t * direction for the nearest t).
    for i in range(50):
        base = i * 2 * 10
        ring_points = mesh.points[base : base + 20]

        direction = directions[i]

        for point in ring_points:
            relative = point - starts[i]
            along = np.dot(relative, direction)
            radial = relative - along * direction
            assert np.linalg.norm(radial) == pytest.approx(0.06, abs=1e-6)


def test_faces_are_valid_quads_referencing_real_points():
    mesh = build([[0, 0, 0]], [[1, 0, 0]], radius=0.05, resolution=8)

    faces = mesh.faces.reshape(-1, 5)

    assert (faces[:, 0] == 4).all()
    assert faces[:, 1:].max() < mesh.n_points
    assert faces[:, 1:].min() >= 0


def test_normals_are_computed():
    """smooth_shading in add_mesh needs point normals to interpolate."""
    mesh = build([[0, 0, 0]], [[2, 0, 0]], radius=0.08, resolution=12)

    assert "Normals" in mesh.point_data
