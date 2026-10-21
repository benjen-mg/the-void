"""detect_open_metal_sites(): the full pipeline against real pymatgen
structures with known, hand-verifiable coordination geometry.
"""

import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.oms_detection import (  # noqa: E402
    OMSClassification,
    detect_open_metal_sites,
)


def test_saturated_octahedral_metal_is_not_flagged():
    """Zn octahedrally coordinated by 6 O (the same symmetric geometry
    already established for the rock-salt Na/Cl coordination test,
    with Zn substituted in as a supported OMS metal): fully saturated,
    zero asymmetry, must not be reported as open.
    """
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        ["Zn", "Zn", "Zn", "Zn", "O", "O", "O", "O"],
        [
            [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
            [0.5, 0.5, 0.5], [0, 0, 0.5], [0, 0.5, 0], [0.5, 0, 0],
        ],
    )
    labels = ["Zn1", "Zn2", "Zn3", "Zn4", "O1", "O2", "O3", "O4"]

    result = detect_open_metal_sites(structure, labels)

    assert result.oms_count == 0
    assert all(
        site.classification == OMSClassification.SATURATED
        for site in result.sites
    )
    assert all(site.coordination_number == 6 for site in result.sites)
    assert all(site.asymmetry_score == pytest.approx(0.0) for site in result.sites)


def test_square_pyramidal_cu_is_detected_as_open():
    """Cu with 5 O ligands - 4 equatorial, 1 axial - deliberately
    missing the opposite axial position, the canonical real open metal
    site geometry (desolvated Cu paddlewheels, M-MOF-74 nodes).
    Hand-verified: CN=5, asymmetry exactly 0.2 (matching the square-
    pyramidal reference case in test_oms_detection_core.py), missing
    direction exactly (0, 0, -1) - the physically missing position.
    """
    lattice = Lattice.cubic(15.0)
    d = 2.0 / 15.0

    structure = Structure(
        lattice,
        ["Cu", "O", "O", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.5 + d, 0.5, 0.5], [0.5 - d, 0.5, 0.5],
            [0.5, 0.5 + d, 0.5], [0.5, 0.5 - d, 0.5],
            [0.5, 0.5, 0.5 + d],
        ],
    )
    labels = ["Cu1", "O1", "O2", "O3", "O4", "O5"]

    result = detect_open_metal_sites(structure, labels)

    assert result.oms_count == 1
    site = result.open_sites[0]

    assert site.metal_symbol == "Cu"
    assert site.coordination_number == 5
    assert site.asymmetry_score == pytest.approx(0.2, abs=1e-6)
    assert len(site.missing_positions) == 1
    assert site.missing_positions[0].direction == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-6
    )


def test_unsupported_metal_is_not_analyzed_at_all():
    """Na is not in OMS_SUPPORTED_METALS - even though it is genuinely
    a metal and would otherwise show a clean coordination environment,
    it must not appear in the result at all.
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

    result = detect_open_metal_sites(structure, labels)

    assert result.sites == ()
    assert result.oms_count == 0
    # ...but it is named, so the result can say why nothing was checked.
    assert result.unsupported_metals == ("Na",)


def test_structure_with_no_metals_gives_an_empty_result():
    lattice = Lattice.cubic(10.0)
    structure = Structure(
        lattice, ["C", "H", "H", "H", "H"],
        [[0.5, 0.5, 0.5], [0.55, 0.5, 0.5], [0.45, 0.5, 0.5],
         [0.5, 0.55, 0.5], [0.5, 0.45, 0.5]],
    )
    labels = ["C1", "H1", "H2", "H3", "H4"]

    result = detect_open_metal_sites(structure, labels)

    assert result.sites == ()
    assert result.oms_count == 0
    assert result.unsupported_metals == ()


def test_every_supported_metal_runs_without_raising():
    """Each of the nine required metals, placed in the same simple
    six-oxygen octahedral test geometry (already used for coordination
    environment's own per-metal test), must be analyzed by OMS
    detection without raising - the point being CrystalNN and the
    asymmetry computation both handle every supported element cleanly,
    not a claim that every real compound of these metals is
    octahedral.
    """
    lattice = Lattice.cubic(4.2)

    for metal in ("Zn", "Cu", "Co", "Ni", "Fe", "Zr", "Ti", "Al", "Cr"):
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
        labels = [f"{metal}1"] + [f"O{i}" for i in range(1, 7)]

        result = detect_open_metal_sites(structure, labels)

        assert len(result.sites) == 1, metal
        assert result.sites[0].metal_symbol == metal


def test_open_sites_property_returns_only_open_ones():
    lattice = Lattice.cubic(15.0)
    d = 2.0 / 15.0

    structure = Structure(
        lattice,
        ["Cu", "O", "O", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.5 + d, 0.5, 0.5], [0.5 - d, 0.5, 0.5],
            [0.5, 0.5 + d, 0.5], [0.5, 0.5 - d, 0.5],
            [0.5, 0.5, 0.5 + d],
        ],
    )
    labels = ["Cu1", "O1", "O2", "O3", "O4", "O5"]

    result = detect_open_metal_sites(structure, labels)

    assert len(result.open_sites) == result.oms_count == 1


# --- Round 49: extended metal coverage ------------------------------------

ROUND_49_METALS = ("Mg", "Mn", "Cd", "V", "Sc", "Hf", "Ga")


def test_every_supported_metal_has_its_own_reference_coordination():
    """Every metal OMS detection accepts must have an explicit entry in
    TYPICAL_MAX_COORDINATION rather than silently falling back to the
    default of 6 - adding a metal to the supported set without deciding
    its reference maximum is exactly the mistake this guards against.
    """
    from thevoid_core.data.metal_classification import OMS_SUPPORTED_METALS
    from thevoid_core.services.oms_detection import TYPICAL_MAX_COORDINATION

    assert set(OMS_SUPPORTED_METALS) <= set(TYPICAL_MAX_COORDINATION)


@pytest.mark.parametrize("metal", ROUND_49_METALS)
def test_newly_supported_metal_is_analyzed(metal):
    """Before round 49 a Mn or Mg framework (MUF-16(Mn), Mg-MOF-74)
    reported zero metal centres analyzed. Each newly supported metal
    in a saturated octahedral environment must now appear, saturated.
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
    labels = [f"{metal}1"] + [f"O{i}" for i in range(1, 7)]

    result = detect_open_metal_sites(structure, labels)

    assert len(result.sites) == 1
    assert result.sites[0].metal_symbol == metal
    assert result.oms_count == 0


@pytest.mark.parametrize("metal", ("Mg", "Mn"))
def test_square_pyramidal_mof74_metals_are_detected_as_open(metal):
    """The desolvated M-MOF-74 node - five O ligands, the sixth
    (solvent) position vacated - is the canonical open metal site, and
    Mg/Mn are two of its best-known members. Same hand-verified
    geometry as the Cu case above: CN=5, asymmetry 0.2, gap at -z.
    """
    lattice = Lattice.cubic(15.0)
    d = 2.0 / 15.0

    structure = Structure(
        lattice,
        [metal, "O", "O", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.5 + d, 0.5, 0.5], [0.5 - d, 0.5, 0.5],
            [0.5, 0.5 + d, 0.5], [0.5, 0.5 - d, 0.5],
            [0.5, 0.5, 0.5 + d],
        ],
    )
    labels = [f"{metal}1", "O1", "O2", "O3", "O4", "O5"]

    result = detect_open_metal_sites(structure, labels)

    assert result.oms_count == 1
    site = result.open_sites[0]
    assert site.coordination_number == 5
    assert site.asymmetry_score == pytest.approx(0.2, abs=1e-6)
    assert site.missing_positions[0].direction == pytest.approx(
        (0.0, 0.0, -1.0), abs=1e-6
    )
