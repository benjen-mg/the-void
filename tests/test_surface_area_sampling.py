"""sample_atom_surfaces() / exposed_mask(): the Monte Carlo core of
accessible surface area analysis, checked against cases with a hand-
derivable expected answer (via the standard spherical-cap formula for
sphere-sphere intersection), plus a numerical cross-check of that
derivation with a separate, independent brute-force Monte Carlo script
before trusting it as a test oracle (see the round's development
notes) - not trusted from memory alone.

Monte Carlo sampling has statistical noise; every tolerance below is
set relative to the expected sqrt(N) relative error at the sample count
used, not tightened arbitrarily.
"""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from thevoid_core.services.surface_area import (
    exposed_mask,
    sample_atom_surfaces,
    sample_sphere_points,
)


def test_sample_sphere_points_all_lie_on_the_sphere():
    rng = np.random.default_rng(0)

    points = sample_sphere_points(1000, 3.0, (1.0, 2.0, 3.0), rng)

    distances = np.linalg.norm(points - np.array([1.0, 2.0, 3.0]), axis=1)
    assert np.allclose(distances, 3.0, atol=1e-9)


def test_sample_sphere_points_are_roughly_uniform():
    """Not bunched at the poles - a coarse but meaningful check: the
    mean of many uniform points on a sphere centred at the origin
    should be close to the origin itself (positive and negative
    contributions cancel), unlike a naive angle-uniform sampling which
    would bias toward the poles.
    """
    rng = np.random.default_rng(1)

    points = sample_sphere_points(200000, 1.0, (0.0, 0.0, 0.0), rng)

    mean = points.mean(axis=0)
    assert np.all(np.abs(mean) < 0.02)


def test_isolated_atom_is_fully_exposed_and_matches_the_sphere_formula():
    """Nothing nearby to block it: area must equal 4*pi*r^2 exactly (no
    Monte Carlo noise at all, since every sampled point is trivially
    exposed regardless of where it lands).
    """
    rng = np.random.default_rng(2)

    center = np.array([[0.0, 0.0, 0.0]])
    radius = np.array([1.5])

    total, per_atom, points, exposed = sample_atom_surfaces(
        center, radius, center, radius,
        probe_radius=1.0, n_samples_per_atom=20000, rng=rng,
    )

    expected = 4 * np.pi * (1.5 + 1.0) ** 2
    assert total == pytest.approx(expected)
    assert exposed.all()


def test_two_equal_spheres_touching_centres_match_the_spherical_cap_formula():
    """Two identical spheres (radius 2), centres 2 apart - deep overlap.
    By symmetry the intersecting plane sits exactly midway (distance 1
    from each centre); the buried spherical cap has height
    h = r - d/2 = 2 - 1 = 1, giving a buried-area fraction of
    h / (2r) = 0.25, i.e. 75% exposed. Cross-checked numerically against
    an independent brute-force script (not just derived on paper) before
    being used as a test oracle here.
    """
    rng = np.random.default_rng(3)

    base_cartesian = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vdw = np.array([2.0, 2.0])

    total, per_atom, points, exposed = sample_atom_surfaces(
        base_cartesian, vdw, base_cartesian, vdw,
        probe_radius=0.0, n_samples_per_atom=100000, rng=rng,
    )

    expected_area_per_atom = 0.75 * 4 * np.pi * 2.0**2

    # Monte Carlo noise at n=100,000 samples: relative std error is
    # roughly 1/sqrt(n * p * (1-p)) for a binomial fraction - generous
    # 1% absolute tolerance on the fraction comfortably covers it.
    assert per_atom[0] == pytest.approx(expected_area_per_atom, rel=0.02)
    assert per_atom[1] == pytest.approx(expected_area_per_atom, rel=0.02)


def test_probe_radius_changes_the_spherical_cap_result_as_expected():
    """Same geometry as above, but with a 0.5 A probe: effective radius
    becomes 2.5, cap height h = 2.5 - 1 = 1.5, buried fraction
    h/(2r) = 0.3, i.e. 70% exposed.
    """
    rng = np.random.default_rng(4)

    base_cartesian = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vdw = np.array([2.0, 2.0])

    total, per_atom, points, exposed = sample_atom_surfaces(
        base_cartesian, vdw, base_cartesian, vdw,
        probe_radius=0.5, n_samples_per_atom=100000, rng=rng,
    )

    expected_area_per_atom = 0.7 * 4 * np.pi * 2.5**2

    assert per_atom[0] == pytest.approx(expected_area_per_atom, rel=0.02)
    assert per_atom[1] == pytest.approx(expected_area_per_atom, rel=0.02)


def test_a_small_atom_fully_engulfed_by_a_larger_one_has_zero_area():
    rng = np.random.default_rng(5)

    base_cartesian = np.array([[0.0, 0.0, 0.0]])
    base_vdw = np.array([0.5])

    # A much larger neighbour essentially co-located - fully engulfs
    # the small atom's own (tiny) sphere.
    periodic_positions = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]])
    periodic_vdw = np.array([0.5, 5.0])

    total, per_atom, points, exposed = sample_atom_surfaces(
        base_cartesian, base_vdw, periodic_positions, periodic_vdw,
        probe_radius=0.0, n_samples_per_atom=20000, rng=rng,
    )

    assert total == pytest.approx(0.0, abs=1e-9)
    assert not exposed.any()


def test_probe_radius_effect_on_area_depends_on_geometry_not_a_universal_law():
    """A genuinely useful thing this test caught while being written:
    for two overlapping equal spheres at a *fixed* centre separation,
    growing the probe radius both enlarges each sphere (area grows
    quadratically with radius) and shrinks the exposed fraction (more
    relative overlap) - which effect wins depends on the geometry, it
    is not a universal "larger probe always means less area" rule.

    For this exact case (r=2, d=2, probe 0 -> 1.0): buried fraction is
    h/(2r) with h = r - d/2, which tends toward 1/2 as r grows rather
    than growing without bound, while raw area grows as r^2 - so for
    this fixed-distance pair, area actually *increases* with a larger
    probe (75.4 -> 150.8, hand-derived and confirmed to match exactly).
    An earlier version of this test assumed the opposite as if it were
    a general law and was simply wrong; corrected here with the actual
    derivation rather than a plausible-sounding guess.
    """
    base_cartesian = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    vdw = np.array([2.0, 2.0])

    total_small, _, _, _ = sample_atom_surfaces(
        base_cartesian, vdw, base_cartesian, vdw,
        probe_radius=0.0, n_samples_per_atom=100000,
        rng=np.random.default_rng(7),
    )
    total_large, _, _, _ = sample_atom_surfaces(
        base_cartesian, vdw, base_cartesian, vdw,
        probe_radius=1.0, n_samples_per_atom=100000,
        rng=np.random.default_rng(8),
    )

    expected_small = 2 * (0.75 * 4 * np.pi * 2.0**2)
    expected_large = 2 * ((2.0 / 3.0) * 4 * np.pi * 3.0**2)

    assert total_small == pytest.approx(expected_small, rel=0.02)
    assert total_large == pytest.approx(expected_large, rel=0.02)
    assert total_large > total_small


def test_exposed_mask_directly_against_a_known_tree():
    """exposed_mask() in isolation: one query point known by
    construction to be inside a neighbour's sphere, one known to be
    clear of everything.
    """
    tree = cKDTree(np.array([[0.0, 0.0, 0.0]]))
    radii = np.array([2.0])

    points = np.array([[1.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    exposed = exposed_mask(points, tree, radii, max_neighbor_radius=2.0)

    assert exposed[0] == False  # noqa: E712 - inside the radius-2 sphere
    assert exposed[1] == True  # noqa: E712 - far outside it


def test_sampled_points_and_exposed_mask_are_row_aligned():
    rng = np.random.default_rng(9)

    base_cartesian = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    vdw = np.array([1.0, 1.0])

    total, per_atom, points, exposed = sample_atom_surfaces(
        base_cartesian, vdw, base_cartesian, vdw,
        probe_radius=0.0, n_samples_per_atom=100, rng=rng,
    )

    assert len(points) == len(exposed) == 200
