"""thevoid_core.services.measurement: pure geometry, no Qt or VTK
dependency. Distance and angle are ordinary vector math; nearest_atom_index
is the part that matters for picking - a click lands on a sphere's
surface, not its centre, and has to be snapped back to the right atom.
"""

import math

import pytest

from thevoid_core.services.measurement import (
    MeasurementResult,
    angle,
    distance,
    nearest_atom_index,
    nearest_periodic_image,
)


def test_distance_between_identical_points_is_zero():
    assert distance((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)) == 0.0


def test_distance_matches_a_known_3_4_5_triangle():
    assert distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_distance_is_symmetric():
    a, b = (1.0, 2.0, 3.0), (4.0, -1.0, 7.0)

    assert distance(a, b) == pytest.approx(distance(b, a))


def test_right_angle():
    # Vertex at origin, one arm along +x, one along +y.
    assert angle((1, 0, 0), (0, 0, 0), (0, 1, 0)) == pytest.approx(90.0)


def test_straight_angle():
    assert angle((-1, 0, 0), (0, 0, 0), (1, 0, 0)) == pytest.approx(180.0)


def test_zero_angle_between_coincident_directions():
    assert angle((1, 0, 0), (0, 0, 0), (2, 0, 0)) == pytest.approx(0.0)


def test_angle_is_symmetric_in_its_outer_arguments():
    a, v, c = (1.0, 0.3, -0.2), (0.0, 0.0, 0.0), (-0.4, 1.0, 0.1)

    assert angle(a, v, c) == pytest.approx(angle(c, v, a))


def test_water_like_angle():
    """A rough H-O-H geometry: ~104.5 degrees is the textbook value,
    this just checks the general vicinity and correct convention (the
    second argument is the vertex).
    """
    oxygen = (0.0, 0.0, 0.0)
    hydrogen_a = (0.96, 0.0, 0.0)
    hydrogen_b = (0.96 * math.cos(math.radians(104.5)), 0.96 * math.sin(math.radians(104.5)), 0.0)

    result = angle(hydrogen_a, oxygen, hydrogen_b)

    assert result == pytest.approx(104.5, abs=0.01)


def test_angle_raises_when_vertex_coincides_with_another_point():
    with pytest.raises(ValueError):
        angle((1, 0, 0), (1, 0, 0), (0, 1, 0))


def test_nearest_atom_finds_the_closest_position():
    positions = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]

    assert nearest_atom_index((4.8, 0.1, 0.0), positions) == 1


def test_nearest_atom_snaps_a_surface_point_back_to_its_centre():
    """The realistic case: picking returns a point on a glyphed sphere's
    surface, offset from the true centre by roughly the atom's radius.
    """
    positions = [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0)]

    surface_point = (0.3, 0.0, 0.0)  # 0.3 A off centre of atom 0

    assert nearest_atom_index(surface_point, positions) == 0


def test_nearest_atom_returns_none_for_empty_list():
    assert nearest_atom_index((0.0, 0.0, 0.0), []) is None


def test_nearest_atom_respects_max_distance():
    positions = [(0.0, 0.0, 0.0)]

    # Within tolerance: found.
    assert nearest_atom_index((0.3, 0.0, 0.0), positions, max_distance=0.5) == 0

    # A click that landed far from any atom (e.g. on a bond, or empty
    # space) should not silently snap to a distant atom.
    assert (
        nearest_atom_index((10.0, 0.0, 0.0), positions, max_distance=0.5)
        is None
    )


def test_measurement_result_distance_label():
    result = MeasurementResult(kind="distance", value=1.5432, atom_indices=(0, 1))

    assert result.label == "1.543 \u00c5"


def test_measurement_result_angle_label():
    result = MeasurementResult(kind="angle", value=104.47, atom_indices=(0, 1, 2))

    assert result.label == "104.5\u00b0"


CUBIC_10A = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def test_nearest_periodic_image_finds_a_closer_copy_across_a_cell_face():
    """Two sites 0.5 A apart through a cell boundary: the base position
    (drawn in the reference cell) is 9.5 A away in a straight line, but
    the periodic image one cell over is only 0.5 A away.
    """
    anchor = (0.25, 5.0, 5.0)
    base = (9.75, 5.0, 5.0)  # reference-cell position of the other site

    position, image, dist = nearest_periodic_image(anchor, base, CUBIC_10A)

    assert dist == pytest.approx(0.5, abs=1e-9)
    assert image == (-1, 0, 0)
    assert position[0] == pytest.approx(-0.25, abs=1e-9)


def test_nearest_periodic_image_matches_plain_distance_when_already_closest():
    """Two atoms already close together in the same cell: the reference
    image is already the nearest one, image offset (0, 0, 0).
    """
    anchor = (5.0, 5.0, 5.0)
    base = (6.2, 5.0, 5.0)

    position, image, dist = nearest_periodic_image(anchor, base, CUBIC_10A)

    assert image == (0, 0, 0)
    assert dist == pytest.approx(1.2, abs=1e-9)
    assert position == pytest.approx(base)


def test_nearest_periodic_image_searches_all_three_axes():
    """A site near a corner: the nearest image might require stepping
    in more than one lattice direction at once.
    """
    anchor = (0.3, 0.3, 0.3)
    base = (9.6, 9.6, 9.6)

    position, image, dist = nearest_periodic_image(anchor, base, CUBIC_10A)

    assert image == (-1, -1, -1)
    # Corner-to-corner through all three faces: (9.6-10, 9.6-10, 9.6-10)
    # = (-0.4,-0.4,-0.4) vs anchor (0.3,0.3,0.3): distance along each
    # axis is 0.7, total = sqrt(3 * 0.7^2).
    assert dist == pytest.approx((3 * 0.7**2) ** 0.5, abs=1e-6)


def test_nearest_periodic_image_agrees_with_plain_distance_for_the_chosen_image():
    anchor = (1.0, 2.0, 3.0)
    base = (9.0, 2.0, 3.0)

    position, image, dist = nearest_periodic_image(anchor, base, CUBIC_10A)

    assert dist == pytest.approx(distance(anchor, position))


def test_nearest_periodic_image_on_a_non_orthogonal_lattice():
    """A skewed (monoclinic-like) lattice: the search must use the
    actual lattice vectors, not just an axis-aligned box.
    """
    lattice = [[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 10.0]]

    anchor = (0.2, 0.2, 5.0)
    # Image (-1, -1, 0) translates base by -a - b = -(13, 9, 0), landing
    # at (13.3-13, 9.3-9, 5.0) = (0.3, 0.3, 5.0) - 0.1 A from the anchor
    # along both x and y.
    base = (13.3, 9.3, 5.0)

    position, image, dist = nearest_periodic_image(anchor, base, lattice)

    assert image == (-1, -1, 0)
    assert dist == pytest.approx((0.1**2 + 0.1**2) ** 0.5, abs=1e-9)


def test_nearest_periodic_image_zero_distance_when_anchor_equals_base():
    position, image, dist = nearest_periodic_image(
        (5.0, 5.0, 5.0), (5.0, 5.0, 5.0), CUBIC_10A
    )

    assert dist == pytest.approx(0.0, abs=1e-9)
    assert image == (0, 0, 0)


def test_measurement_result_carries_site_indices():
    result = MeasurementResult(
        kind="distance",
        value=1.5,
        atom_indices=(3, 7),
        site_indices=(1, 2),
    )

    assert result.site_indices == (1, 2)


def test_measurement_result_site_indices_default_to_none():
    result = MeasurementResult(kind="distance", value=1.5, atom_indices=(0, 1))

    assert result.site_indices is None
