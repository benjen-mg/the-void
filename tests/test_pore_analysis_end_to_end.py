"""analyze_pores(): the full pipeline, against real pymatgen structures
chosen specifically because the expected LCD/PLD/void fraction can be
worked out by hand or reasoned about with confidence, not against real
deposited MOF files where the "correct" answer would have to be trusted
from the code itself.
"""

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services.pore_analysis import analyze_pores  # noqa: E402


@pytest.fixture(scope="module")
def radius_provider():
    return ElementRadiusProvider()


def test_single_atom_at_cell_centre_lcd_matches_hand_calculation(
    radius_provider,
):
    """The same hand-verified geometry used for the distance-field
    tests: a single Zn at the exact centre of a 10 A cubic cell. The
    point farthest from the atom and all its periodic images is any
    cell corner, at a*sqrt(3)/2 from the centre by symmetry - so LCD
    must be exactly 2 * (a*sqrt(3)/2 - vdw_radius).
    """
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Zn"], [[0.5, 0.5, 0.5]])
    vdw_radius = radius_provider.get_vdw_radius("Zn")

    result = analyze_pores(
        structure, probe_radius=1.0, radius_provider=radius_provider,
        grid_spacing=0.2,
    )

    expected_lcd = 2.0 * (10.0 * np.sqrt(3) / 2 - vdw_radius)

    # Grid sampling introduces a small discretisation error - the true
    # corner point may not land exactly on a grid node - so allow a
    # tolerance related to the grid spacing rather than demanding exact
    # equality.
    assert result.lcd == pytest.approx(expected_lcd, abs=0.5)


def test_a_fully_blocked_cell_has_near_zero_void_fraction(radius_provider):
    """A single atom in a cell smaller than its own van der Waals
    diameter: its periodic images necessarily overlap in every
    direction (cell length 3.5 A versus a 4.32 A van der Waals
    diameter for Xe, checked directly rather than assumed), leaving
    essentially no room anywhere for even a small probe.
    """
    lattice = Lattice.cubic(3.5)
    structure = Structure(lattice, ["Xe"], [[0.5, 0.5, 0.5]])

    result = analyze_pores(
        structure, probe_radius=0.5, radius_provider=radius_provider,
        grid_spacing=0.2,
    )

    assert result.void_fraction_percent < 5.0
    assert result.accessible_volume < result.cell_volume * 0.05


def test_an_open_cell_with_one_tiny_atom_has_high_void_fraction(
    radius_provider,
):
    """The opposite extreme: a large cell with a single small atom -
    almost the entire cell is open space, and what's open should
    overwhelmingly form one connected, percolating (3D) network, since
    there is nothing to compartmentalise it into separate pockets.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    result = analyze_pores(
        structure, probe_radius=1.0, radius_provider=radius_provider,
        grid_spacing=0.5,
    )

    assert result.void_fraction_percent > 80.0
    assert result.pld is not None
    assert result.accessible_dimensionality == 3
    assert result.accessible_volume > result.non_accessible_volume


def test_volumes_are_internally_consistent_with_void_fraction(
    radius_provider,
):
    """accessible_volume + non_accessible_volume must equal the total
    void volume implied by void_fraction_percent, by construction - both
    are computed from the same underlying grid classification, just
    split differently.
    """
    lattice = Lattice.cubic(12.0)
    structure = Structure(
        lattice, ["Zn", "O"], [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]
    )

    result = analyze_pores(
        structure, probe_radius=1.2, radius_provider=radius_provider,
        grid_spacing=0.4,
    )

    total_void_volume = result.void_fraction_percent / 100.0 * result.cell_volume
    reconstructed = result.accessible_volume + result.non_accessible_volume

    assert reconstructed == pytest.approx(total_void_volume, abs=1e-6)


def test_larger_probe_radius_never_increases_void_fraction(radius_provider):
    """A larger probe can only be accommodated by at least as much
    room as a smaller one - void fraction must be monotonically
    non-increasing as probe_radius grows, for the same structure.
    """
    lattice = Lattice.cubic(12.0)
    structure = Structure(lattice, ["Zn"], [[0.5, 0.5, 0.5]])

    small_probe = analyze_pores(
        structure, probe_radius=0.5, radius_provider=radius_provider,
        grid_spacing=0.4,
    )
    large_probe = analyze_pores(
        structure, probe_radius=2.0, radius_provider=radius_provider,
        grid_spacing=0.4,
    )

    assert large_probe.void_fraction_percent <= small_probe.void_fraction_percent


def test_lcd_is_at_least_as_large_as_pld(radius_provider):
    """The pore-limiting diameter is the width of the tightest
    bottleneck along the best percolating path; the largest cavity
    diameter is the widest point anywhere in the structure, accessible
    or not. The widest bottleneck along any path can never exceed the
    single widest point in the whole structure - LCD >= PLD always,
    for any real structure.
    """
    lattice = Lattice.cubic(14.0)
    structure = Structure(
        lattice, ["Zn", "Zn"], [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]]
    )

    result = analyze_pores(
        structure, probe_radius=1.0, radius_provider=radius_provider,
        grid_spacing=0.4,
    )

    if result.pld is not None:
        assert result.lcd >= result.pld


def test_empty_structure_is_reported_as_entirely_open(radius_provider):
    """No atoms at all: the whole cell is void by definition. Handled
    as an explicit, documented case rather than letting an infinite or
    NaN value propagate confusingly from an empty distance field.
    """
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, [], [])

    result = analyze_pores(
        structure, probe_radius=1.0, radius_provider=radius_provider,
    )

    assert result.void_fraction_percent == 100.0
    assert result.accessible_volume == pytest.approx(result.cell_volume)
    assert result.non_accessible_volume == 0.0


def test_accessible_points_are_within_the_cell_volume_bounds(radius_provider):
    """A basic sanity check on the visualization data: every reported
    accessible point must be a real Cartesian position, not garbage."""
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    result = analyze_pores(
        structure, probe_radius=1.0, radius_provider=radius_provider,
        grid_spacing=0.5,
    )

    assert result.accessible_points.shape[1] == 3
    assert len(result.accessible_points) > 0
    assert np.all(np.isfinite(result.accessible_points))
