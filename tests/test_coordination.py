"""analyze_coordination_environment(): real pymatgen structures, since
CrystalNN needs a real coordination environment to analyze - no
meaningful way to fake this one out with plain test doubles.
"""

import pytest

pytest.importorskip("pymatgen")

import numpy as np  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.coordination import (  # noqa: E402
    CoordinationEnvironment,
    Neighbor,
    analyze_coordination_environment,
)


@pytest.fixture
def rock_salt_nacl():
    """Na octahedrally coordinated by 6 Cl at a/2 - a textbook case
    with an unambiguous, well-known answer to check against.
    """
    lattice = Lattice.cubic(5.64)

    return Structure(
        lattice,
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
        ],
    )


def test_rock_salt_sodium_is_six_coordinate(rock_salt_nacl):
    labels = ["Na1", "Na2", "Na3", "Na4", "Cl1", "Cl2", "Cl3", "Cl4"]

    env = analyze_coordination_environment(rock_salt_nacl, 0, labels)

    assert env.coordination_number == 6
    assert all(n.symbol == "Cl" for n in env.neighbors)


def test_rock_salt_bond_lengths_are_all_a_over_2(rock_salt_nacl):
    labels = ["Na1", "Na2", "Na3", "Na4", "Cl1", "Cl2", "Cl3", "Cl4"]

    env = analyze_coordination_environment(rock_salt_nacl, 0, labels)

    expected = 5.64 / 2

    assert env.average_bond_length == pytest.approx(expected, abs=0.01)
    assert env.min_bond_length == pytest.approx(expected, abs=0.01)
    assert env.max_bond_length == pytest.approx(expected, abs=0.01)


def test_neighbors_are_sorted_nearest_first():
    # A distorted environment: one Cl noticeably closer than the rest,
    # to confirm sorting actually happens rather than just being true
    # by coincidence in the symmetric rock-salt case.
    lattice = Lattice.cubic(8.0)
    structure = Structure(
        lattice,
        ["Na", "Cl", "Cl", "Cl"],
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.65], [0.7, 0.5, 0.5], [0.5, 0.8, 0.5]],
    )

    env = analyze_coordination_environment(structure, 0, ["Na1", "Cl1", "Cl2", "Cl3"])

    distances = [n.distance for n in env.neighbors]
    assert distances == sorted(distances)


def test_neighbor_labels_match_the_supplied_label_list():
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
        ],
    )
    labels = ["Sodium_A", "Sodium_B", "Sodium_C", "Sodium_D",
              "Chloride_A", "Chloride_B", "Chloride_C", "Chloride_D"]

    env = analyze_coordination_environment(structure, 0, labels)

    assert env.center_label == "Sodium_A"
    assert all(n.label.startswith("Chloride_") for n in env.neighbors)


def test_center_symbol_and_label_are_set():
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
        ],
    )

    env = analyze_coordination_environment(structure, 4, ["Na1", "Na2", "Na3", "Na4", "Cl1", "Cl2", "Cl3", "Cl4"])

    assert env.center_symbol == "Cl"
    assert env.center_label == "Cl1"


def test_missing_label_falls_back_to_a_placeholder():
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
        ],
    )

    env = analyze_coordination_environment(structure, 0, [])  # no labels supplied

    assert env.center_label == "?1"


# --- dataclass property tests (no pymatgen needed) ---


def test_coordination_number_is_neighbor_count():
    env = CoordinationEnvironment(
        center_site_index=0, center_label="M1", center_symbol="Zn",
        neighbors=(
            Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),
            Neighbor(2, "O", "O2", 2.1, (0, 0, 0)),
        ),
    )

    assert env.coordination_number == 2


def test_average_min_max_bond_length():
    env = CoordinationEnvironment(
        center_site_index=0, center_label="M1", center_symbol="Zn",
        neighbors=(
            Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),
            Neighbor(2, "O", "O2", 2.2, (0, 0, 0)),
            Neighbor(3, "O", "O3", 1.8, (0, 0, 0)),
        ),
    )

    assert env.average_bond_length == pytest.approx(2.0)
    assert env.min_bond_length == pytest.approx(1.8)
    assert env.max_bond_length == pytest.approx(2.2)


def test_empty_neighbors_do_not_raise():
    env = CoordinationEnvironment(
        center_site_index=0, center_label="M1", center_symbol="Zn",
        neighbors=(),
    )

    assert env.coordination_number == 0
    assert env.average_bond_length == 0.0
    assert env.min_bond_length == 0.0
    assert env.max_bond_length == 0.0


# --- special-attention metals ---


@pytest.mark.parametrize(
    "metal", ["Zn", "Cu", "Co", "Ni", "Fe", "Zr", "Ti", "Al"]
)
def test_special_attention_metal_in_an_octahedral_oxide_environment(metal):
    """Each of the eight requested metals, placed in the same simple
    rock-salt-like MO environment (6 oxygens at a/2), must be analyzed
    without raising and report a sensible non-zero coordination number.
    Not a claim that every real compound of these metals is
    octahedral - just confirming CrystalNN runs cleanly for each one
    in a well-defined, unambiguous test geometry.
    """
    lattice = Lattice.cubic(4.2)

    structure = Structure(
        lattice,
        [metal, "O", "O", "O", "O", "O", "O"],
        [
            [0, 0, 0],
            [0.5, 0, 0], [-0.5, 0, 0],
            [0, 0.5, 0], [0, -0.5, 0],
            [0, 0, 0.5], [0, 0, -0.5],
        ],
    )

    env = analyze_coordination_environment(
        structure, 0, [f"{metal}1"] + [f"O{i}" for i in range(1, 7)]
    )

    assert env.center_symbol == metal
    assert env.coordination_number > 0
    assert all(n.distance > 0 for n in env.neighbors)


def test_neighbor_direction_vectors_match_known_octahedral_geometry():
    """Rock-salt Na: 6 Cl neighbours at a/2 along +-x, +-y, +-z. Each
    neighbour's direction vector, normalised, should be very close to
    one of those six axis directions - and the vectors as a whole
    should sum to (approximately) zero, the defining property of a
    symmetric, saturated octahedral environment that OMS detection's
    asymmetry measure depends on.
    """
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice,
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"],
        [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
        ],
    )
    labels = ["Na1", "Na2", "Na3", "Na4", "Cl1", "Cl2", "Cl3", "Cl4"]

    env = analyze_coordination_environment(structure, 0, labels)

    assert all(n.direction is not None for n in env.neighbors)

    directions = np.array([n.direction for n in env.neighbors])
    unit_vectors = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    # Every unit vector should be very close to a permutation of
    # (+-1, 0, 0) - i.e. exactly one axis.
    for vector in unit_vectors:
        assert np.max(np.abs(vector)) == pytest.approx(1.0, abs=0.01)

    # Symmetric octahedral coordination: direction vectors cancel out.
    vector_sum = np.sum(unit_vectors, axis=0)
    assert np.linalg.norm(vector_sum) < 1e-6
