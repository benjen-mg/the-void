"""_OffsetUnionFind: the mechanism that tells an isolated pocket apart
from a pore network that repeats itself across a periodic boundary, and
tells a 1D channel apart from a 2D sheet or a 3D interconnected network.

Every test here is a small, fully hand-traceable graph - deliberately
not real crystal geometry - so the connectivity logic itself is
verified independently of the (separately tested) geometry that turns
atoms into a distance field.
"""

from thevoid_core.services.pore_analysis import (
    _OffsetUnionFind,
    dimensionality_from_vectors,
)


def test_two_nodes_joined_with_zero_offset_do_not_percolate():
    """A single ordinary edge within one cell - no periodic boundary
    involved at all - must not be mistaken for percolation.
    """
    uf = _OffsetUnionFind.for_n_nodes(2)

    uf.union(0, 1, (0, 0, 0))

    root, _ = uf.find(0)
    assert uf.percolation_vectors_for_root(root) == []


def test_a_simple_cycle_with_zero_net_offset_does_not_percolate():
    """0-1-2-0, all steps within the same cell (offsets summing to
    zero around the loop): a closed loop that never crosses a boundary,
    however it's connected internally, is still just one ordinary
    region - not a sign of periodicity.
    """
    uf = _OffsetUnionFind.for_n_nodes(3)

    uf.union(0, 1, (1, 0, 0))
    uf.union(1, 2, (-1, 1, 0))
    uf.union(2, 0, (0, -1, 0))  # closes the loop: net (0,0,0)

    root, _ = uf.find(0)
    assert uf.percolation_vectors_for_root(root) == []


def test_a_ring_that_wraps_around_one_axis_percolates_in_1d():
    """A chain of grid points 0-1-2-...-(n-1)-0, where the LAST step
    back to node 0 crosses one periodic boundary (offset (1,0,0)) -
    exactly a 1D channel that tiles along x, the simplest real case
    this exists to detect.
    """
    n = 5
    uf = _OffsetUnionFind.for_n_nodes(n)

    for i in range(n - 1):
        uf.union(i, i + 1, (0, 0, 0))

    uf.union(n - 1, 0, (1, 0, 0))  # the wraparound step

    root, _ = uf.find(0)
    vectors = uf.percolation_vectors_for_root(root)

    assert vectors == [(1, 0, 0)]
    assert dimensionality_from_vectors(vectors) == 1


def test_two_independent_wraparounds_give_2d_dimensionality():
    """A small sheet: nodes arranged so the network wraps around both
    x and y independently - a 2D pore layer, not a 1D channel.
    """
    #  0 - 1
    #  |   |
    #  2 - 3
    # with 1->0 wrapping in x, and 2->0 wrapping in y.
    uf = _OffsetUnionFind.for_n_nodes(4)

    uf.union(0, 1, (0, 0, 0))
    uf.union(0, 2, (0, 0, 0))
    uf.union(1, 3, (0, 0, 0))
    uf.union(2, 3, (0, 0, 0))
    uf.union(1, 0, (1, 0, 0))  # wraps in x
    uf.union(2, 0, (0, 1, 0))  # wraps in y

    root, _ = uf.find(0)
    vectors = uf.percolation_vectors_for_root(root)

    assert dimensionality_from_vectors(vectors) == 2


def test_three_independent_wraparounds_give_3d_dimensionality():
    uf = _OffsetUnionFind.for_n_nodes(1)
    # A single node connected to itself via three independent lattice
    # translations - the degenerate but valid way to express "this
    # network repeats in all three directions" with the fewest nodes.
    uf.union(0, 0, (1, 0, 0))
    uf.union(0, 0, (0, 1, 0))
    uf.union(0, 0, (0, 0, 1))

    root, _ = uf.find(0)
    vectors = uf.percolation_vectors_for_root(root)

    assert dimensionality_from_vectors(vectors) == 3


def test_disconnected_nodes_are_in_different_sets():
    uf = _OffsetUnionFind.for_n_nodes(4)

    uf.union(0, 1, (0, 0, 0))
    uf.union(2, 3, (0, 0, 0))

    root_01, _ = uf.find(0)
    root_23, _ = uf.find(2)

    assert root_01 != root_23


def test_an_isolated_pocket_has_no_percolation_vectors():
    """A small closed pocket (no periodic wraparound edge at all) must
    report zero percolation vectors and dimensionality 0 - the
    "isolated void" case the whole module exists to distinguish from a
    real, usable pore.
    """
    uf = _OffsetUnionFind.for_n_nodes(3)

    uf.union(0, 1, (0, 0, 0))
    uf.union(1, 2, (0, 0, 0))

    root, _ = uf.find(0)
    vectors = uf.percolation_vectors_for_root(root)

    assert vectors == []
    assert dimensionality_from_vectors(vectors) == 0


def test_find_reports_the_correct_cumulative_offset():
    """find() must correctly accumulate offsets along a chain, not just
    detect percolation - the dimensionality/PLD machinery both depend
    on offsets being individually correct, not just "some cycle was
    found somewhere".
    """
    uf = _OffsetUnionFind.for_n_nodes(3)

    uf.union(0, 1, (1, 0, 0))
    uf.union(1, 2, (1, 0, 0))

    _, offset_of_2 = uf.find(2)

    assert offset_of_2 == (2, 0, 0)


def test_redundant_edge_matching_the_existing_offset_is_not_a_percolation():
    """Adding an edge a second time (or an equivalent alternate path)
    that agrees with the already-known offset between two nodes must
    not be misreported as a new percolation vector.
    """
    uf = _OffsetUnionFind.for_n_nodes(3)

    uf.union(0, 1, (1, 0, 0))
    uf.union(1, 2, (0, 0, 0))
    # An alternate path 0->2 with an offset consistent with the
    # existing tree (0->1 is +1 in x, 1->2 is +0, so 0->2 should be
    # +1 in x - exactly what's asserted here, not a new drift).
    uf.union(0, 2, (1, 0, 0))

    root, _ = uf.find(0)
    assert uf.percolation_vectors_for_root(root) == []


def test_path_compression_does_not_corrupt_offsets():
    """A longer chain, deliberately long enough to exercise path
    compression more than once, must still report the correct
    cumulative offset at the end - a bug here would silently corrupt
    every distance calculation built on top of it.
    """
    n = 20
    uf = _OffsetUnionFind.for_n_nodes(n)

    for i in range(n - 1):
        uf.union(i, i + 1, (1, 0, 0))

    # Trigger path compression with finds interleaved among the unions.
    for i in range(n):
        uf.find(i)

    _, offset_of_last = uf.find(n - 1)
    assert offset_of_last == (n - 1, 0, 0)
