"""Exporting a disordered structure.

POSCAR and Quantum ESPRESSO cannot express fractional occupancy at
all. Fed a real disordered file, pymatgen's writers raise - POSCAR a
ValueError, PWInput an AttributeError ("attr 'specie' not found on
PeriodicSite"), the latter of which used to escape the export error
handler entirely and propagate out of a Qt slot.

These tests pin both halves of the fix: the service-level helper that
produces the ordered approximation, and the fact that every other
format still accepts a disordered structure directly rather than being
needlessly routed through that filter.
"""

import pytest
from pymatgen.core import Lattice, Structure

from thevoid_core.services.export import (
    FORMATS_REQUIRING_ORDERED_STRUCTURE,
    export_cif,
    export_cp2k,
    export_gaussian,
    export_poscar,
    export_quantum_espresso,
    export_xyz,
    ordered_structure_for_export,
)


@pytest.fixture
def ordered_structure():
    return Structure(
        Lattice.cubic(4.05),
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


@pytest.fixture
def disordered_structure():
    """One full-occupancy Na site plus four quarter-occupancy Cl
    alternatives - the shape of a real disordered site, four positions
    that are actually one atom.
    """
    return Structure(
        Lattice.cubic(5.64),
        [
            {"Na": 1.0},
            {"Cl": 0.25},
            {"Cl": 0.25},
            {"Cl": 0.25},
            {"Cl": 0.25},
        ],
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.52, 0.5, 0.5],
            [0.5, 0.52, 0.5],
            [0.5, 0.5, 0.52],
        ],
    )


def test_the_two_formats_that_cannot_hold_occupancy_are_the_listed_ones():
    assert FORMATS_REQUIRING_ORDERED_STRUCTURE == {
        "poscar",
        "quantum_espresso",
    }


def test_an_ordered_structure_is_returned_unchanged(ordered_structure):
    assert (
        ordered_structure_for_export(ordered_structure) is ordered_structure
    )


def test_minor_components_are_dropped_from_a_disordered_structure(
    disordered_structure,
):
    ordered = ordered_structure_for_export(disordered_structure)

    assert ordered.is_ordered
    assert len(ordered) == 1
    assert ordered[0].specie.symbol == "Na"


def test_the_original_structure_is_left_untouched(disordered_structure):
    ordered_structure_for_export(disordered_structure)

    assert len(disordered_structure) == 5
    assert not disordered_structure.is_ordered


@pytest.mark.parametrize(
    "export_function", [export_poscar, export_quantum_espresso]
)
def test_the_ordered_approximation_exports_without_raising(
    disordered_structure, export_function
):
    """The regression this whole module exists for: both of these
    raised on the disordered input, and the Quantum ESPRESSO one raised
    a type the caller did not catch.
    """
    ordered = ordered_structure_for_export(disordered_structure)

    content = export_function(ordered)

    assert content.strip()


@pytest.mark.parametrize(
    "export_function",
    [export_cif, export_xyz, export_gaussian, export_cp2k],
)
def test_the_other_formats_still_take_disorder_directly(
    disordered_structure, export_function
):
    """These four are deliberately not in
    FORMATS_REQUIRING_ORDERED_STRUCTURE: CIF records occupancies
    properly, and the other three write one line per site from its
    dominant species. Routing them through the filter would throw away
    atoms for no reason, so this guards against that being "tidied up"
    into applying everywhere.
    """
    content = export_function(disordered_structure)

    assert content.strip()


def test_cif_keeps_every_disordered_site(disordered_structure):
    content = export_cif(disordered_structure)

    assert content.count("Cl") >= 4
    assert "0.25" in content
