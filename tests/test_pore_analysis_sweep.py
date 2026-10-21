"""sweep_connectivity(): PLD and probe-radius accessibility on small,
fully hand-constructed radius grids where the expected answer can be
worked out by hand, rather than real crystal geometry.

Grid indexing matches build_grid_edges: row-major flattening of
(nx, ny, nz), 6-connectivity with periodic wraparound on every axis.
Test grids deliberately use size >= 2 on every axis - a size-1 axis
creates a self-loop edge on every node (a real, correct consequence of
a single-layer periodic grid trivially "repeating" along that axis, not
a bug), which would be a confusing artifact to reason about by hand in
a test specifically checking percolation logic.
"""

import numpy as np

from thevoid_core.services.pore_analysis import (
    build_grid_edges,
    sweep_connectivity,
)


def _flat_index(shape, i, j, k):
    nx, ny, nz = shape
    return i * ny * nz + j * nz + k


def test_build_grid_edges_count_matches_expectation():
    """A 6-connected periodic grid has exactly 3 edges per node (one
    per axis, counting each undirected edge once).
    """
    shape = (3, 3, 3)

    edges = build_grid_edges(shape)

    assert len(edges) == 3 * 3 * 3 * 3


def test_build_grid_edges_wraps_with_the_correct_offset():
    shape = (3, 2, 2)

    edges = build_grid_edges(shape)

    # The edge from the last x-layer back to the first must carry
    # offset (1, 0, 0), and no other axis.
    wrap_edges = [
        (a, b, offset) for a, b, offset in edges if offset != (0, 0, 0)
    ]

    assert all(offset[1] == 0 and offset[2] == 0 for _, _, offset in wrap_edges if offset[0] != 0)


def test_a_uniform_ring_percolates_with_pld_matching_its_radius():
    """Every node the same radius, full periodic connectivity in every
    direction: the whole grid is one interconnected network, and PLD
    is exactly twice that uniform radius - there is no narrower
    bottleneck anywhere for it to be limited by.
    """
    shape = (3, 3, 3)
    radius_values = np.full(3 * 3 * 3, 2.0)

    result = sweep_connectivity(radius_values, shape, probe_radius=1.0)

    assert result.pld_radius == 2.0
    assert result.accessible.all()


def test_an_isolated_line_percolates_in_1d_only():
    """A single line of wide-open nodes along x (which wraps, since the
    grid is periodic), surrounded entirely by nodes too narrow for the
    probe: the line forms its own 1D ring, isolated from everything
    else. Only those three nodes should be accessible, and the network
    they form is exactly 1-dimensional.
    """
    shape = (3, 3, 3)
    radius_values = np.zeros(3 * 3 * 3)

    line_indices = [_flat_index(shape, i, 0, 0) for i in range(3)]
    for index in line_indices:
        radius_values[index] = 2.0

    result = sweep_connectivity(radius_values, shape, probe_radius=1.0)

    assert result.pld_radius == 2.0

    accessible_indices = set(np.nonzero(result.accessible)[0].tolist())
    assert accessible_indices == set(line_indices)

    dimensionalities = set(result.dimensionality_by_root.values())
    assert dimensionalities == {1}


def test_a_sealed_pocket_is_not_accessible_even_though_locally_wide():
    """A single node with plenty of local room, but with every
    neighbour too narrow for the probe to pass through to reach it:
    geometrically a cavity, but not one anything could actually enter
    or leave - exactly the "isolated void" case this module exists to
    separate from a real, usable pore.
    """
    shape = (3, 3, 3)
    radius_values = np.zeros(3 * 3 * 3)

    pocket_index = _flat_index(shape, 1, 1, 1)
    radius_values[pocket_index] = 5.0  # locally huge...

    result = sweep_connectivity(radius_values, shape, probe_radius=1.0)

    # ...but every edge touching it also touches a radius-0 neighbour,
    # so its edge capacities are all 0 - below probe_radius - and it
    # never gets unioned with anything.
    assert result.accessible[pocket_index] == False  # noqa: E712
    assert result.pld_radius is None


def test_two_independent_channels_give_two_dimensionality_one_entries():
    """Two separate 1D lines, each wrapping around x independently,
    with no connection between them: two distinct accessible networks,
    each its own 1D channel.
    """
    shape = (3, 3, 3)
    radius_values = np.zeros(3 * 3 * 3)

    line_a = [_flat_index(shape, i, 0, 0) for i in range(3)]
    line_b = [_flat_index(shape, i, 2, 2) for i in range(3)]

    for index in line_a + line_b:
        radius_values[index] = 1.5

    result = sweep_connectivity(radius_values, shape, probe_radius=1.0)

    accessible_indices = set(np.nonzero(result.accessible)[0].tolist())
    assert accessible_indices == set(line_a) | set(line_b)

    assert len(result.dimensionality_by_root) == 2
    assert set(result.dimensionality_by_root.values()) == {1}


def test_pld_is_found_even_when_narrower_than_the_requested_probe_radius():
    """The regression this specifically guards against: PLD is a
    property of the material alone and must be found correctly even
    when the requested probe_radius is larger than the true bottleneck
    - an earlier, flawed version of this function stopped its sweep
    early at probe_radius and could miss a narrower true PLD entirely.
    """
    shape = (3, 3, 3)
    radius_values = np.zeros(3 * 3 * 3)

    line_indices = [_flat_index(shape, i, 0, 0) for i in range(3)]
    for index in line_indices:
        radius_values[index] = 1.0  # narrow: true PLD radius is 1.0

    # Ask for accessibility at a much larger probe than the line
    # supports - nothing should be reported accessible, but PLD must
    # still correctly reflect the line's own bottleneck.
    result = sweep_connectivity(radius_values, shape, probe_radius=3.0)

    assert result.pld_radius == 1.0
    assert not result.accessible.any()


def test_probe_radius_larger_than_everything_finds_nothing_accessible():
    shape = (2, 2, 2)
    radius_values = np.full(2 * 2 * 2, 1.0)

    result = sweep_connectivity(radius_values, shape, probe_radius=5.0)

    assert not result.accessible.any()


def test_all_zero_radius_grid_has_no_pld_and_nothing_accessible():
    shape = (2, 2, 2)
    radius_values = np.zeros(2 * 2 * 2)

    result = sweep_connectivity(radius_values, shape, probe_radius=0.5)

    assert result.pld_radius is None
    assert not result.accessible.any()
