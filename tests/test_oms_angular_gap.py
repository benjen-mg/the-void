"""Round 50: the angular-gap criterion for open metal sites whose
vacancies come in symmetric pairs (and so cancel in the asymmetry
vector)."""

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.oms_detection import (  # noqa: E402
    DEFAULT_ANGULAR_GAP_THRESHOLD,
    OMSClassification,
    angular_gaps,
    classify_site,
    detect_open_metal_sites,
)

S3 = np.sqrt(3.0)

REFERENCE = {
    # geometry: (ligand directions, widest empty cone in degrees)
    "octahedron": (
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        np.degrees(np.arccos(1 / S3)),  # 54.74, toward a face centre
    ),
    "trigonal bipyramid": (
        [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-0.5, S3 / 2, 0],
         [-0.5, -S3 / 2, 0]],
        np.degrees(np.arctan(2.0)),  # 63.43
    ),
    "tetrahedron": (
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
        180 - np.degrees(np.arccos(-1 / 3)),  # 70.53, opposite a vertex
    ),
    "square pyramid": (
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]], 90.0,
    ),
    "square plane": (
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], 90.0,
    ),
    "trigonal plane": (
        [[1, 0, 0], [-0.5, S3 / 2, 0], [-0.5, -S3 / 2, 0]], 90.0,
    ),
}


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_widest_gap_of_reference_polyhedra(name):
    vectors, expected = REFERENCE[name]
    widest, _ = angular_gaps(vectors, threshold=DEFAULT_ANGULAR_GAP_THRESHOLD)

    assert widest == pytest.approx(expected, abs=1e-3)


def test_threshold_separates_saturated_from_open_geometries():
    saturated = ("octahedron", "trigonal bipyramid", "tetrahedron")
    open_ = ("square pyramid", "square plane", "trigonal plane")

    for name in saturated:
        assert REFERENCE[name][1] < DEFAULT_ANGULAR_GAP_THRESHOLD - 5
    for name in open_:
        assert REFERENCE[name][1] > DEFAULT_ANGULAR_GAP_THRESHOLD + 5


def test_square_plane_gaps_are_both_axial_positions():
    vectors, _ = REFERENCE["square plane"]
    _, directions = angular_gaps(vectors, threshold=80.0)

    found = sorted(tuple(np.round(d, 6)) for d, _ in directions)
    assert found == [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]


def test_classify_site_uses_the_gap_only_as_an_extra_route_to_open():
    # Symmetric vacancies: zero asymmetry, wide gap -> OPEN.
    assert classify_site(4, 0.0, 6, angular_gap=90.0) is OMSClassification.OPEN
    # Tetrahedral: zero asymmetry, 70.5 degree gap -> SATURATED.
    assert (
        classify_site(4, 0.0, 6, angular_gap=70.53)
        is OMSClassification.SATURATED
    )
    # At the reference maximum, nothing is open, gap or not.
    assert (
        classify_site(6, 0.0, 6, angular_gap=179.0)
        is OMSClassification.SATURATED
    )
    # Without a gap value, the historical behaviour.
    assert classify_site(4, 0.0, 6) is OMSClassification.SATURATED


def _square_planar(metal, extra=None):
    """Metal with 4 O at 2.0 A in the xy plane, in a 15 A box."""
    d = 2.0 / 15.0
    species = [metal, "O", "O", "O", "O"]
    coords = [
        [0.5, 0.5, 0.5],
        [0.5 + d, 0.5, 0.5], [0.5 - d, 0.5, 0.5],
        [0.5, 0.5 + d, 0.5], [0.5, 0.5 - d, 0.5],
    ]
    if extra:
        species.append(extra[0])
        coords.append(extra[1])
    return Structure(Lattice.cubic(15.0), species, coords)


@pytest.mark.parametrize("metal", ["Cu", "Ni"])
def test_square_planar_metal_with_both_axial_sites_empty_is_open(metal):
    """The metalloporphyrin case: asymmetry exactly 0, so before round 50
    this came out SATURATED."""
    structure = _square_planar(metal)
    labels = [f"{metal}1", "O1", "O2", "O3", "O4"]

    result = detect_open_metal_sites(structure, labels)

    site = result.sites[0]
    assert site.asymmetry_score == pytest.approx(0.0, abs=1e-9)
    assert site.angular_gap == pytest.approx(90.0, abs=1e-3)
    assert site.is_open
    directions = sorted(
        tuple(np.round(mp.direction, 4)) for mp in site.missing_positions
    )
    assert directions == [(0.0, 0.0, -1.0), (0.0, 0.0, 1.0)]


def test_paddlewheel_partner_blocks_its_axial_position():
    """A second metal 2.6 A along +z (the paddlewheel Cu-Cu contact)
    occupies that axial site whether or not CrystalNN counts it as a
    ligand: the only vacancy reported is -z."""
    structure = _square_planar("Cu", extra=("Cu", [0.5, 0.5, 0.5 + 2.6 / 15]))
    labels = ["Cu1", "O1", "O2", "O3", "O4", "Cu2"]

    result = detect_open_metal_sites(structure, labels)
    site = next(s for s in result.sites if s.site_index == 0)

    assert site.is_open
    assert len(site.missing_positions) == 1
    assert site.missing_positions[0].direction == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-3
    )


def test_tetrahedral_zinc_stays_saturated():
    b = 1.2 / 15.0
    structure = Structure(
        Lattice.cubic(15.0),
        ["Zn", "O", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.5 + b, 0.5 + b, 0.5 + b], [0.5 + b, 0.5 - b, 0.5 - b],
            [0.5 - b, 0.5 + b, 0.5 - b], [0.5 - b, 0.5 - b, 0.5 + b],
        ],
    )
    labels = ["Zn1", "O1", "O2", "O3", "O4"]

    site = detect_open_metal_sites(structure, labels).sites[0]

    assert site.coordination_number == 4
    assert site.angular_gap == pytest.approx(70.53, abs=0.05)
    assert not site.is_open


