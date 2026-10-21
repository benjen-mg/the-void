"""compute_site_labels: pure, no Qt/pymatgen dependency needed to test
against fakes shaped like the real thing.
"""

from thevoid_core.services.site_labels import compute_site_labels


class FakeSpecies:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeSite:
    def __init__(self, symbol):
        self.species = {FakeSpecies(symbol): 1.0}


class FakeStructure:
    def __init__(self, symbols):
        self.sites = [FakeSite(s) for s in symbols]


def test_counters_are_per_element():
    structure = FakeStructure(["Si", "O", "Si", "O", "O"])

    assert compute_site_labels(structure) == [
        "Si1",
        "O1",
        "Si2",
        "O2",
        "O3",
    ]


def test_single_element_structure():
    structure = FakeStructure(["Na", "Na", "Na"])

    assert compute_site_labels(structure) == ["Na1", "Na2", "Na3"]


def test_empty_structure():
    structure = FakeStructure([])

    assert compute_site_labels(structure) == []


def test_labels_are_index_aligned_with_sites():
    structure = FakeStructure(["C", "H", "H", "H", "H"])

    labels = compute_site_labels(structure)

    assert len(labels) == len(structure.sites)
    assert labels[0] == "C1"
    assert labels[1:] == ["H1", "H2", "H3", "H4"]


def test_site_with_no_species_gets_a_placeholder_symbol():
    class EmptySpeciesSite:
        species = {}

    structure = FakeStructure([])
    structure.sites = [EmptySpeciesSite()]

    assert compute_site_labels(structure) == ["?1"]
