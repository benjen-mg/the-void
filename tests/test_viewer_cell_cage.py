"""Geometry check for the supercell wireframe cage.

Runs the same corner/edge construction the viewer uses, without needing
a live render window, and checks the point and edge counts scale the way
a grid of cages should: 8 points and 12 edges per cell, no sharing.
"""

import numpy as np

CUBIC_5A = np.eye(3) * 5.0


def build_cage(lattice_matrix, supercell):
    """Mirrors CrystalViewer._render_unit_cell's geometry construction."""
    a, b, c = np.asarray(lattice_matrix, dtype=float)
    nx, ny, nz = supercell

    corner_offsets = np.array(
        [
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ],
        dtype=float,
    )
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]

    all_points = []
    all_line_cells = []
    cell_index = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                origin = i * a + j * b + k * c
                corners = origin + corner_offsets @ np.array([a, b, c])
                all_points.append(corners)

                base = cell_index * 8
                for p, q in edges:
                    all_line_cells += [2, base + p, base + q]

                cell_index += 1

    return np.vstack(all_points), np.array(all_line_cells, dtype=int)


def test_single_cell_has_eight_corners_and_twelve_edges():
    points, lines = build_cage(CUBIC_5A, (1, 1, 1))

    assert points.shape == (8, 3)
    assert len(lines) == 12 * 3  # [count, i, j] per edge


def test_2x2x2_has_eight_cages():
    points, lines = build_cage(CUBIC_5A, (2, 2, 2))

    assert points.shape == (8 * 8, 3)
    assert len(lines) == 8 * 12 * 3


def test_3x3x3_has_twenty_seven_cages():
    points, lines = build_cage(CUBIC_5A, (3, 3, 3))

    assert points.shape == (27 * 8, 3)
    assert len(lines) == 27 * 12 * 3


def test_adjacent_cages_share_a_geometric_corner_but_not_an_index():
    """No vertex welding: point (1,0,0)'s cage duplicates the (0,0,0)
    cage's +x face rather than indexing into it. Simpler and cheap at
    these sizes, but worth pinning so it isn't assumed away later.
    """
    points, _ = build_cage(CUBIC_5A, (2, 1, 1))

    # Cell 0's +x corners (local indices 1,4,5,7, offset x=1) sit at
    # x=5; cell 1's -x corners (local indices 0,2,3,6, offset x=0) also
    # sit at x=5, since cell 1's origin is itself at x=5.
    cell0_plus_x = points[[1, 4, 5, 7]]
    cell1_minus_x = points[8 + np.array([0, 2, 3, 6])]

    assert np.allclose(sorted(cell0_plus_x[:, 0]), sorted(cell1_minus_x[:, 0]))


def test_origin_of_each_cell_is_the_correct_lattice_translation():
    points, _ = build_cage(CUBIC_5A, (2, 2, 1))

    # Cell order is i-major, then j, then k (matches the nested loops),
    # so cell index 1 is (i=0, j=1, k=0): origin at (0, 5, 0).
    second_cell_origin = points[8]  # first corner of the second cage

    assert np.allclose(second_cell_origin, [0.0, 5.0, 0.0])
