"""Tests for the covalent-radius bonding rule.

A fake neighbour list stands in for pymatgen so the cutoff logic can be
checked without the dependency.
"""

import pytest

pytest.importorskip("numpy")

from thevoid_core.services.bonds.covalent_bond_detector import (  # noqa: E402
    CovalentBondDetector,
)


class FakeSpecies:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeSite:
    def __init__(self, symbol):
        self.species = {FakeSpecies(symbol): 1.0}


class FakeStructure:
    """Serves a canned neighbour list, the way pymatgen would."""

    def __init__(self, symbols, contacts):
        self.sites = [FakeSite(s) for s in symbols]
        self._contacts = contacts

    def __len__(self):
        return len(self.sites)

    def get_neighbor_list(self, radius):
        self.requested_radius = radius

        centers = [c[0] for c in self._contacts]
        neighbors = [c[1] for c in self._contacts]
        images = [c[2] for c in self._contacts]
        distances = [c[3] for c in self._contacts]

        return centers, neighbors, images, distances


def test_contact_within_summed_radii_is_a_bond():
    # C 0.76 + O 0.66 + 0.4 tolerance = 1.82 A cutoff
    structure = FakeStructure(
        ["C", "O"],
        [
            (0, 1, (0, 0, 0), 1.43),
            (1, 0, (0, 0, 0), 1.43),
        ],
    )

    bonds = CovalentBondDetector().detect(structure)

    assert len(bonds) == 1
    assert bonds[0].length == pytest.approx(1.43)


def test_contact_beyond_the_cutoff_is_rejected():
    structure = FakeStructure(
        ["C", "O"],
        [(0, 1, (0, 0, 0), 2.90)],
    )

    assert CovalentBondDetector().detect(structure) == []


def test_both_directions_collapse_to_one_bond():
    structure = FakeStructure(
        ["C", "O"],
        [
            (0, 1, (1, 0, 0), 1.43),
            (1, 0, (-1, 0, 0), 1.43),
        ],
    )

    assert len(CovalentBondDetector().detect(structure)) == 1


def test_geminal_hydrogens_are_not_bonded():
    """The H...H contact on a CH2 is within range but is not a bond."""
    structure = FakeStructure(
        ["H", "H"],
        [(0, 1, (0, 0, 0), 0.95)],
    )

    assert CovalentBondDetector().detect(structure) == []


def test_overlapping_sites_are_ignored():
    """Disordered or duplicated sites must not bond to each other."""
    structure = FakeStructure(
        ["C", "C"],
        [(0, 1, (0, 0, 0), 0.05)],
    )

    assert CovalentBondDetector().detect(structure) == []


def test_search_radius_covers_the_largest_pair():
    structure = FakeStructure(["Cs", "C"], [])

    detector = CovalentBondDetector()
    detector.detect(structure)

    # 2 x Cs (2.44) + 0.4 tolerance
    assert structure.requested_radius == pytest.approx(5.28)


def test_tolerance_is_configurable():
    structure = FakeStructure(
        ["C", "O"],
        [(0, 1, (0, 0, 0), 1.90)],
    )

    assert CovalentBondDetector(tolerance=0.4).detect(structure) == []
    assert len(CovalentBondDetector(tolerance=0.6).detect(structure)) == 1


def test_disordered_oxygen_pair_is_not_bonded():
    """The 0.794 A O-O contact from the reported screenshot.

    O-O passes the 1.72 A covalent cutoff easily, so only a scaled lower
    bound catches it. 0.794 A is 60% of the summed radii; no real bond
    is that short.
    """
    structure = FakeStructure(
        ["O", "O"],
        [(0, 1, (0, 0, 0), 0.794)],
    )

    bonds, diagnostics = CovalentBondDetector().detect_with_diagnostics(
        structure
    )

    assert bonds == []
    assert diagnostics.short_contact_count == 1
    assert diagnostics.shortest_rejected == pytest.approx(0.794)


def test_real_short_bonds_still_pass_the_lower_bound():
    """The floor must not cut into genuine chemistry."""
    cases = [
        (["O", "H"], 0.96),   # hydroxyl
        (["C", "H"], 1.09),   # aliphatic C-H
        (["C", "O"], 1.21),   # carbonyl
        (["N", "O"], 1.24),   # nitrate
        (["C", "N"], 1.16),   # nitrile
    ]

    for symbols, distance in cases:
        structure = FakeStructure(symbols, [(0, 1, (0, 0, 0), distance)])

        bonds = CovalentBondDetector().detect(structure)

        assert len(bonds) == 1, f"{symbols} at {distance} A was rejected"


def test_partial_occupancy_is_counted():
    class PartialSite(FakeSite):
        def __init__(self, symbol, occupancy):
            self.species = {FakeSpecies(symbol): occupancy}

    structure = FakeStructure(["O", "O"], [(0, 1, (0, 0, 0), 0.794)])
    structure.sites = [PartialSite("O", 0.5), PartialSite("O", 0.5)]

    _, diagnostics = CovalentBondDetector().detect_with_diagnostics(structure)

    assert diagnostics.partial_occupancy_sites == 2
    assert diagnostics.suggests_disorder
    assert diagnostics.short_contacts[0].both_partial
