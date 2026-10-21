"""tile_points_for_supercell(): replicates analysis overlay points to
match the 3D view's own atom-tiling convention exactly."""

import numpy as np

from thevoid_core.ui.widgets.viewer_widget import tile_points_for_supercell


def test_1x1x1_returns_points_unchanged():
    points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lattice = np.eye(3) * 10.0

    result = tile_points_for_supercell(points, lattice, (1, 1, 1))

    assert np.array_equal(result, points)


def test_2x1x1_adds_exactly_one_image_shifted_by_the_a_vector():
    points = np.array([[1.0, 2.0, 3.0]])
    lattice = np.eye(3) * 10.0

    result = tile_points_for_supercell(points, lattice, (2, 1, 1))

    expected = {(1.0, 2.0, 3.0), (11.0, 2.0, 3.0)}
    assert {tuple(row) for row in result} == expected


def test_2x2x2_produces_eight_copies_per_point():
    points = np.array([[0.0, 0.0, 0.0]])
    lattice = np.eye(3) * 5.0

    result = tile_points_for_supercell(points, lattice, (2, 2, 2))

    assert len(result) == 8


def test_tiling_is_corner_anchored_at_the_origin_cell_not_centred():
    """Matches RenderModelBuilder's own convention: images run
    0..n-1 in each direction, not e.g. -1..+1 for a 3x range. A point
    at the origin, tiled 3x3x3, must never appear at a negative
    lattice-vector offset.
    """
    points = np.array([[0.0, 0.0, 0.0]])
    lattice = np.eye(3) * 10.0

    result = tile_points_for_supercell(points, lattice, (3, 3, 3))

    assert np.all(result >= -1e-9)
    assert np.max(result) == 20.0  # (n-1)*10 for n=3
    assert len(result) == 27


def test_tiling_works_for_a_non_orthogonal_lattice():
    points = np.array([[0.0, 0.0, 0.0]])
    lattice = np.array([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.0, 0.0, 5.0]])

    result = tile_points_for_supercell(points, lattice, (2, 2, 1))

    expected = {
        (0.0, 0.0, 0.0),
        (5.0, 0.0, 0.0),
        (1.0, 5.0, 0.0),
        (6.0, 5.0, 0.0),
    }
    assert {tuple(np.round(row, 6)) for row in result} == expected


def test_multiple_points_each_get_their_own_full_set_of_images():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lattice = np.eye(3) * 10.0

    result = tile_points_for_supercell(points, lattice, (2, 1, 1))

    assert len(result) == 4  # 2 points x 2 images each


def test_empty_points_array_returns_empty_without_raising():
    lattice = np.eye(3) * 10.0

    result = tile_points_for_supercell(
        np.zeros((0, 3)), lattice, (3, 3, 3)
    )

    assert result.shape == (0, 3)


def test_1x1x1_with_empty_points_returns_empty():
    lattice = np.eye(3) * 10.0

    result = tile_points_for_supercell(
        np.zeros((0, 3)), lattice, (1, 1, 1)
    )

    assert result.shape == (0, 3)
