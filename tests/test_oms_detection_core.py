"""compute_coordination_asymmetry() and classify_site(): checked
against the exact reference coordination geometries independently
computed while choosing DEFAULT_ASYMMETRY_THRESHOLD (see that
constant's own docstring for the numbers) - not re-derived loosely
here, the same values.
"""

import numpy as np
import pytest

from thevoid_core.services.oms_detection import (
    DEFAULT_ASYMMETRY_THRESHOLD,
    OMSClassification,
    classify_site,
    compute_coordination_asymmetry,
)

# --- compute_coordination_asymmetry: symmetric (saturated) geometries ---


def test_perfect_tetrahedron_has_zero_asymmetry():
    vectors = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]

    _, magnitude = compute_coordination_asymmetry(vectors)

    assert magnitude == pytest.approx(0.0, abs=1e-9)


def test_perfect_octahedron_has_zero_asymmetry():
    vectors = [
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ]

    _, magnitude = compute_coordination_asymmetry(vectors)

    assert magnitude == pytest.approx(0.0, abs=1e-9)


def test_square_planar_has_zero_asymmetry():
    """4 ligands, genuinely saturated in their own right (a legitimate
    geometry for many metals) - must not be flagged just because it is
    "only" 4-coordinate.
    """
    vectors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]

    _, magnitude = compute_coordination_asymmetry(vectors)

    assert magnitude == pytest.approx(0.0, abs=1e-9)


# --- compute_coordination_asymmetry: single-vacancy geometries ---


def test_square_pyramidal_missing_axial_matches_hand_calculation():
    """5 of 6 octahedral positions, missing -z - the canonical real OMS
    case (desolvated Cu paddlewheels, M-MOF-74 nodes). Hand-derived:
    the 4 equatorial vectors cancel, leaving only the +z vector
    uncancelled, so the sum is (0,0,1), magnitude 1/5 = 0.2, missing
    direction the negative of that sum: (0,0,-1) - exactly the
    physically missing position.
    """
    vectors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]

    missing_direction, magnitude = compute_coordination_asymmetry(vectors)

    assert magnitude == pytest.approx(0.2, abs=1e-9)
    assert missing_direction == pytest.approx([0.0, 0.0, -1.0], abs=1e-9)


def test_t_shape_three_coordinate_has_larger_asymmetry():
    vectors = [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]

    _, magnitude = compute_coordination_asymmetry(vectors)

    assert magnitude == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_single_ligand_has_maximum_asymmetry():
    """One ligand only: nothing opposes it at all - the maximum
    possible asymmetry, magnitude exactly 1.0, missing direction
    exactly opposite the one ligand present.
    """
    vectors = [[0, 0, 1]]

    missing_direction, magnitude = compute_coordination_asymmetry(vectors)

    assert magnitude == pytest.approx(1.0)
    assert missing_direction == pytest.approx([0.0, 0.0, -1.0])


def test_empty_coordination_sphere_does_not_raise():
    missing_direction, magnitude = compute_coordination_asymmetry([])

    assert magnitude == 0.0
    assert np.array_equal(missing_direction, np.zeros(3))


def test_missing_direction_is_a_unit_vector():
    vectors = [[2, 0, 0], [0, 3, 0], [0, 0, 1]]  # non-unit inputs on purpose

    missing_direction, magnitude = compute_coordination_asymmetry(vectors)

    if magnitude > 1e-9:
        assert np.linalg.norm(missing_direction) == pytest.approx(1.0)


# --- classify_site ---


def test_saturated_site_at_typical_max_is_never_open():
    """CN already at the typical maximum: SATURATED regardless of any
    residual geometric asymmetry - distortion in a fully-coordinated
    site is not the same as a missing ligand.
    """
    status = classify_site(
        coordination_number=6,
        asymmetry_magnitude=0.9,  # deliberately huge
        typical_max_coordination=6,
    )

    assert status == OMSClassification.SATURATED


def test_low_coordination_with_high_asymmetry_is_open():
    status = classify_site(
        coordination_number=5,
        asymmetry_magnitude=0.2,  # the square-pyramidal reference value
        typical_max_coordination=6,
    )

    assert status == OMSClassification.OPEN


def test_low_coordination_but_symmetric_geometry_is_saturated():
    """4-coordinate but genuinely symmetric (e.g. square planar):
    fewer ligands than the typical maximum, but no real gap - must not
    be flagged.
    """
    status = classify_site(
        coordination_number=4,
        asymmetry_magnitude=0.0,
        typical_max_coordination=6,
    )

    assert status == OMSClassification.SATURATED


def test_asymmetry_exactly_at_threshold_is_not_open():
    """Strictly greater than the threshold, not equal - a boundary
    case worth pinning explicitly.
    """
    status = classify_site(
        coordination_number=4,
        asymmetry_magnitude=DEFAULT_ASYMMETRY_THRESHOLD,
        typical_max_coordination=6,
    )

    assert status == OMSClassification.SATURATED


def test_asymmetry_just_above_threshold_is_open():
    status = classify_site(
        coordination_number=4,
        asymmetry_magnitude=DEFAULT_ASYMMETRY_THRESHOLD + 0.001,
        typical_max_coordination=6,
    )

    assert status == OMSClassification.OPEN


def test_default_threshold_correctly_separates_every_reference_geometry():
    """The whole point of the chosen threshold value, checked directly
    rather than trusted from the docstring: every saturated reference
    geometry falls below it, every single-vacancy geometry falls above
    it.
    """
    saturated_magnitudes = [0.0, 0.0, 0.0]  # tetrahedral, octahedral, sq. planar
    open_magnitudes = [0.2, 1.0 / 3.0, 0.25]  # sq. pyramidal, T-shape, trig. bipy.

    assert all(m <= DEFAULT_ASYMMETRY_THRESHOLD for m in saturated_magnitudes)
    assert all(m > DEFAULT_ASYMMETRY_THRESHOLD for m in open_magnitudes)
