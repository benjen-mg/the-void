"""classify_accessibility(), largest_accessible_cavity(), and
identify_bottlenecks(): pure functions, checked against hand-worked
cases rather than real crystal geometry, the same testing strategy
used throughout this project's pore- and surface-analysis modules.
"""

import numpy as np

from thevoid_core.services.guest_accessibility import (
    AccessibilityStatus,
    Bottleneck,
    classify_accessibility,
    identify_bottlenecks,
    largest_accessible_cavity,
)


# --- classify_accessibility ---


def test_guest_smaller_than_pld_is_accessible():
    assert (
        classify_accessibility(lcd=10.0, pld=4.0, guest_diameter=3.0)
        == AccessibilityStatus.ACCESSIBLE
    )


def test_guest_exactly_equal_to_pld_is_accessible():
    """Boundary case: exactly fits through the narrowest window."""
    assert (
        classify_accessibility(lcd=10.0, pld=4.0, guest_diameter=4.0)
        == AccessibilityStatus.ACCESSIBLE
    )


def test_guest_between_pld_and_lcd_is_partially_accessible():
    assert (
        classify_accessibility(lcd=10.0, pld=4.0, guest_diameter=6.0)
        == AccessibilityStatus.PARTIALLY_ACCESSIBLE
    )


def test_guest_exactly_equal_to_lcd_is_partially_accessible():
    """Boundary case: exactly fits the single largest cavity, but not
    through the (narrower) window to get there freely.
    """
    assert (
        classify_accessibility(lcd=10.0, pld=4.0, guest_diameter=10.0)
        == AccessibilityStatus.PARTIALLY_ACCESSIBLE
    )


def test_guest_larger_than_lcd_is_inaccessible():
    assert (
        classify_accessibility(lcd=10.0, pld=4.0, guest_diameter=12.0)
        == AccessibilityStatus.INACCESSIBLE
    )


def test_no_percolating_pathway_at_all_falls_through_to_lcd_check():
    """pld=None (nothing percolates for this guest size) - a guest that
    still fits the largest cavity is partially accessible, not
    accessible (there being no window to be accessible through), and a
    guest too big even for that is inaccessible.
    """
    assert (
        classify_accessibility(lcd=8.0, pld=None, guest_diameter=5.0)
        == AccessibilityStatus.PARTIALLY_ACCESSIBLE
    )
    assert (
        classify_accessibility(lcd=8.0, pld=None, guest_diameter=9.0)
        == AccessibilityStatus.INACCESSIBLE
    )


def test_fully_solid_structure_is_inaccessible_to_everything():
    assert (
        classify_accessibility(lcd=0.0, pld=None, guest_diameter=0.5)
        == AccessibilityStatus.INACCESSIBLE
    )


# --- largest_accessible_cavity ---


def test_largest_accessible_cavity_ignores_isolated_pockets():
    """A huge isolated pocket (radius 5.0, not accessible) must not be
    reported as the "largest accessible cavity" just because it is the
    largest cavity anywhere - only the smaller, but actually reachable,
    cavity (radius 2.0) counts.
    """
    distance_field = np.array([5.0, 2.0, 1.0])
    accessible_mask = np.array([False, True, True])

    result = largest_accessible_cavity(distance_field, accessible_mask)

    assert result == 4.0  # 2 * 2.0, not 2 * 5.0


def test_largest_accessible_cavity_with_nothing_accessible():
    distance_field = np.array([5.0, 2.0, 1.0])
    accessible_mask = np.array([False, False, False])

    assert largest_accessible_cavity(distance_field, accessible_mask) == 0.0


def test_largest_accessible_cavity_with_everything_accessible():
    distance_field = np.array([1.0, 3.0, 2.0])
    accessible_mask = np.array([True, True, True])

    assert largest_accessible_cavity(distance_field, accessible_mask) == 6.0


# --- identify_bottlenecks ---


def _linear_chain_grid_shape():
    # A 5x2x2 grid is enough room for a clear 1D chain with periodic
    # wraparound in x, matching build_grid_edges' own indexing.
    return (5, 2, 2)


def test_a_single_narrow_point_in_a_wide_channel_is_the_bottleneck():
    """A channel that's uniformly wide except for one narrow point:
    that one point - and nothing else - should be identified.
    """
    shape = _linear_chain_grid_shape()
    n = shape[0] * shape[1] * shape[2]

    # Every node "wide" (radius 3.0) except node index 2, which is
    # narrow (radius 1.0) - a single constriction along the chain.
    distance_field = np.full(n, 3.0)
    distance_field[2] = 1.0

    accessible_mask = np.ones(n, dtype=bool)
    grid_positions = np.zeros((n, 3))  # positions irrelevant to this check

    bottlenecks = identify_bottlenecks(
        distance_field, shape, accessible_mask, grid_positions
    )

    narrow_indices = {b.grid_index for b in bottlenecks}

    # Node 2 in row-major (nx, ny, nz) = (5,2,2) indexing is grid index
    # (0, 1, 0): 2 // (2*2) = 0, (2 // 2) % 2 = 1, 2 % 2 = 0.
    assert (0, 1, 0) in narrow_indices


def test_a_uniformly_wide_network_has_no_bottlenecks():
    """Every point equally wide: no point is a *strict* local minimum
    anywhere, so nothing should be flagged - a flat, uniform channel
    isn't a "bottleneck" anywhere in particular.
    """
    shape = _linear_chain_grid_shape()
    n = shape[0] * shape[1] * shape[2]

    distance_field = np.full(n, 2.0)
    accessible_mask = np.ones(n, dtype=bool)
    grid_positions = np.zeros((n, 3))

    bottlenecks = identify_bottlenecks(
        distance_field, shape, accessible_mask, grid_positions
    )

    assert bottlenecks == []


def test_non_accessible_points_are_never_reported_as_bottlenecks():
    shape = _linear_chain_grid_shape()
    n = shape[0] * shape[1] * shape[2]

    distance_field = np.full(n, 3.0)
    distance_field[2] = 0.1  # would be a bottleneck if it were accessible

    accessible_mask = np.zeros(n, dtype=bool)  # nothing accessible at all
    grid_positions = np.zeros((n, 3))

    bottlenecks = identify_bottlenecks(
        distance_field, shape, accessible_mask, grid_positions
    )

    assert bottlenecks == []


def test_bottleneck_radius_and_position_are_reported_correctly():
    shape = _linear_chain_grid_shape()
    n = shape[0] * shape[1] * shape[2]

    distance_field = np.full(n, 3.0)
    distance_field[0] = 0.75

    accessible_mask = np.ones(n, dtype=bool)
    grid_positions = np.arange(n * 3, dtype=float).reshape(n, 3)

    bottlenecks = identify_bottlenecks(
        distance_field, shape, accessible_mask, grid_positions
    )

    matching = [b for b in bottlenecks if b.grid_index == (0, 0, 0)]
    assert len(matching) == 1
    assert matching[0].radius == 0.75
    assert np.array_equal(matching[0].position, grid_positions[0])


def test_bottleneck_is_a_frozen_dataclass_with_expected_fields():
    b = Bottleneck(grid_index=(1, 2, 3), position=np.zeros(3), radius=1.5)

    assert b.grid_index == (1, 2, 3)
    assert b.radius == 1.5
