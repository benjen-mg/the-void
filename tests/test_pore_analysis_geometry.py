"""build_periodic_atoms / compute_distance_field: the geometry layer
that turns real atom positions into the distance field the connectivity
engine consumes. Uses real pymatgen structures, since this is
specifically about periodic image handling and 3D distance geometry -
not meaningfully fakeable with plain doubles the way the pure graph
logic in test_pore_analysis_sweep.py is.
"""

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.pore_analysis import (  # noqa: E402
    build_periodic_atoms,
    compute_distance_field,
    major_occupancy_structure,
    perpendicular_widths,
)


def test_perpendicular_widths_of_a_cubic_cell_equal_its_side_length():
    lattice = np.eye(3) * 10.0

    widths = perpendicular_widths(lattice)

    assert widths == pytest.approx([10.0, 10.0, 10.0])


def test_perpendicular_widths_of_a_skewed_cell():
    """A cell leaning over in x as z increases: the perpendicular
    thickness along z is shorter than the raw c-vector length would
    suggest, since the vector itself is not perpendicular to the ab
    face. Hand-computed: a=(10,0,0), b=(0,10,0), c=(5,0,10) - volume is
    the same as the unskewed cube (base area 100 x true perpendicular
    height 10 = 1000, matching det). The perpendicular width along c
    must come out to exactly 10.0 despite |c| = sqrt(125) =~ 11.18,
    since only the component of c perpendicular to the ab-plane (which
    is the xy-plane here) matters.
    """
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [5.0, 0.0, 10.0]])

    widths = perpendicular_widths(lattice)

    assert widths[2] == pytest.approx(10.0)


def test_build_periodic_atoms_includes_enough_images_for_the_cutoff():
    """A single atom at the cell centre with a generous cutoff must
    produce images covering at least the requested range in every
    direction - checked by confirming a periodic copy exists roughly
    one full cell length away.
    """
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Zn"], [[0.5, 0.5, 0.5]])

    positions, radii = build_periodic_atoms(
        structure, np.array([1.4]), cutoff=12.0
    )

    # The base atom at (5,5,5) plus an image roughly (15,5,5) or
    # similar (one full lattice translation away) must be present.
    assert len(positions) > 1
    distances_from_origin_image = np.linalg.norm(
        positions - np.array([15.0, 5.0, 5.0]), axis=1
    )
    assert distances_from_origin_image.min() < 1e-6


def test_build_periodic_atoms_radii_are_tiled_consistently_with_positions():
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Zn", "O"], [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])

    positions, radii = build_periodic_atoms(
        structure, np.array([1.4, 0.6]), cutoff=5.0
    )

    # Every position claiming to be a Zn-radius atom must actually be
    # an image of the Zn site (frac x=0.2, i.e. cartesian x congruent
    # to 2.0 mod 10), not accidentally paired with the O site's position.
    zn_positions = positions[radii == 1.4]
    x_mod_10 = np.mod(zn_positions[:, 0], 10.0)
    assert np.allclose(x_mod_10, 2.0, atol=1e-6)


# --- compute_distance_field ---


def test_distance_field_for_a_single_atom_at_a_known_point():
    """One atom, one grid point, radius known: distance to surface must
    be exactly |point - atom| - radius, hand-computable.
    """
    atom_positions = np.array([[5.0, 5.0, 5.0]])
    atom_radii = np.array([1.0])
    grid_points = np.array([[0.0, 0.0, 0.0]])

    result = compute_distance_field(grid_points, atom_positions, atom_radii)

    expected = np.linalg.norm([5.0, 5.0, 5.0]) - 1.0
    assert result[0] == pytest.approx(expected)


def test_distance_field_picks_the_true_nearest_surface_not_nearest_centre():
    """Deliberately constructed so the nearest atom by raw centre
    distance is *not* the one with the nearest surface: a small atom
    close by, and a much larger atom slightly farther away whose
    surface actually reaches closer to the query point. A naive
    "nearest centre only" implementation would get this wrong.
    """
    grid_point = np.array([[0.0, 0.0, 0.0]])

    # Small atom, centre distance 3.0, radius 0.5 -> surface distance 2.5.
    # Large atom, centre distance 4.0, radius 3.0 -> surface distance 1.0.
    # The large, farther atom has the nearer surface.
    atom_positions = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    atom_radii = np.array([0.5, 3.0])

    result = compute_distance_field(grid_point, atom_positions, atom_radii)

    assert result[0] == pytest.approx(1.0)


def test_distance_field_with_no_atoms_is_infinite():
    grid_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    result = compute_distance_field(
        grid_points, np.zeros((0, 3)), np.zeros(0)
    )

    assert np.all(np.isinf(result))


def test_distance_field_negative_inside_an_atom():
    atom_positions = np.array([[0.0, 0.0, 0.0]])
    atom_radii = np.array([2.0])
    grid_points = np.array([[0.5, 0.0, 0.0]])  # well inside the 2.0 A radius

    result = compute_distance_field(grid_points, atom_positions, atom_radii)

    assert result[0] < 0


def test_distance_field_matches_hand_calculation_for_a_real_structure():
    """Single Zn at the exact centre of a cubic cell: the point
    farthest from the atom and all its periodic images is any cell
    corner, at distance a*sqrt(3)/2 from the centre by symmetry. This
    is the same hand-verified case used as the primary LCD sanity
    check for the full analyze_pores pipeline later; checked here
    directly against compute_distance_field alone, with periodic
    images included via build_periodic_atoms.
    """
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Zn"], [[0.5, 0.5, 0.5]])
    vdw_radius = 1.0

    positions, radii = build_periodic_atoms(
        structure, np.array([vdw_radius]), cutoff=20.0
    )

    corner = np.array([[0.0, 0.0, 0.0]])
    result = compute_distance_field(corner, positions, radii)

    expected_center_distance = 10.0 * np.sqrt(3) / 2
    expected = expected_center_distance - vdw_radius

    assert result[0] == pytest.approx(expected, abs=1e-6)


# --- major_occupancy_structure ---


def test_major_occupancy_structure_removes_minor_disorder_components():
    """Two sites at the same approximate position, split 0.25/0.75
    between two elements (a disorder pair): only the 0.75 component
    should survive a 0.5 threshold.
    """
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice,
        [{"Zn": 0.75}, {"Cu": 0.25}, "O"],
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]],
    )

    filtered = major_occupancy_structure(structure)

    assert len(filtered.sites) == 2
    symbols = sorted(
        list(site.species.keys())[0].symbol for site in filtered.sites
    )
    assert symbols == ["O", "Zn"]


def test_major_occupancy_structure_keeps_fully_occupied_sites_untouched():
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Zn", "O"], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])

    filtered = major_occupancy_structure(structure)

    assert len(filtered.sites) == 2
    assert filtered is structure  # no copy made when nothing needs removing


def test_major_occupancy_structure_respects_a_custom_threshold():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice, [{"Zn": 0.5}, {"Cu": 0.5}], [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
    )

    # At the default threshold (>= 0.5), both exactly-0.5 sites survive.
    assert len(major_occupancy_structure(structure).sites) == 2

    # A stricter threshold excludes them both.
    assert len(major_occupancy_structure(structure, min_occupancy=0.6).sites) == 0


def test_major_occupancy_structure_does_not_mutate_the_original():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice, [{"Zn": 0.25}, "O"], [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]
    )

    major_occupancy_structure(structure)

    assert len(structure.sites) == 2  # unchanged
