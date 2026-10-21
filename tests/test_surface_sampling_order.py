"""Round 63: surface area no longer depends on how a file lists its atoms.

Before, each atom's sample rotation came from one random stream taken
in atom order, so the same crystal with its atoms in another order gave
a different channel ASA: 0.08-3.5% on 12 CoRE structures at 500 samples
per atom (docs/shared-index-design.md, Appendix A). A shared index needs
two analyses of one determination to agree. The rotations are now drawn
in a canonical order of the atoms (surface_area.canonical_rotations),
so these tests hold the result fixed when a file lists the same atoms
in another order or is rewritten. Moving the origin or taking a
supercell still changes the samples, within sampling scatter, as it
always has: the channel/pocket grid is tied to the cell.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Structure  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services.surface_area import (  # noqa: E402
    analyze_surface_area,
    canonical_order,
    canonical_rotations,
)

FIXTURES = Path(__file__).parent / "fixtures"
from tests.sample_structures import ZN_DIA  # noqa: E402

CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"


def _asa(structure, seed=0, samples=300):
    result = analyze_surface_area(structure, probe_radius=1.86, probe_name="N2",
                                  radius_provider=ElementRadiusProvider(),
                                  n_samples_per_atom=samples, seed=seed)
    return result.channel_asa_per_mass, result.asa_per_mass


def _shuffled(structure, seed=1):
    order = np.random.default_rng(seed).permutation(len(structure))
    return Structure(structure.lattice, [structure[i].species for i in order],
                     [structure[i].frac_coords for i in order])


@pytest.fixture(scope="module", params=[ZN_DIA, CR_BDC], ids=["Zn-dia", "Cr-BDC"])
def structure(request):
    return Structure.from_file(request.param)


@pytest.fixture(scope="module")
def reference(structure):
    return _asa(structure)


def test_atom_order_does_not_matter(structure, reference):
    assert _asa(_shuffled(structure)) == pytest.approx(reference, rel=1e-9)


def test_rewriting_the_file_does_not_matter(structure, reference):
    rewritten = Structure.from_str(str(CifWriter(structure)), fmt="cif")
    assert _asa(rewritten) == pytest.approx(reference, rel=1e-9)


def test_the_seed_still_chooses_the_samples(structure, reference):
    """Another seed gives other rotations, so another estimate within
    sampling scatter; the seed is still the way to measure that
    scatter."""
    other = _asa(structure, seed=5)
    assert other != reference
    assert other[0] == pytest.approx(reference[0], rel=0.05)


def test_the_canonical_order_ignores_file_order_and_wrapping():
    frac = np.array([[0.5, 0.1, 0.1], [0.2, 0.9, 0.3], [0.2, 0.1, 0.3], [1.0, 0.25, 0.0]])
    labels = ["O", "C", "C", "O"]
    order = canonical_order(frac, labels)
    # C before O; within C by x, then y; 1.0 wraps to 0.0, before 0.5.
    assert list(order) == [2, 1, 3, 0]
    permutation = np.array([3, 1, 0, 2])
    again = canonical_order(frac[permutation], [labels[i] for i in permutation])
    assert list(permutation[again]) == list(order)


def test_rotations_follow_the_atoms():
    """Each atom keeps its rotation when the atoms are reordered, every
    rotation is proper, and atoms get independent ones."""
    structure = Structure.from_file(CR_BDC)
    labels = [site.species_string for site in structure]
    rotations = canonical_rotations(structure.frac_coords, labels, seed=0)
    for r in rotations:
        assert r @ r.T == pytest.approx(np.eye(3), abs=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0)
    assert len({tuple(np.round(r, 9).ravel()) for r in rotations}) == len(structure)
    permutation = np.random.default_rng(4).permutation(len(structure))
    moved = canonical_rotations(structure.frac_coords[permutation],
                                [labels[i] for i in permutation], seed=0)
    assert moved == pytest.approx(rotations[permutation], abs=1e-15)


def test_no_atoms_gives_no_rotations():
    assert canonical_rotations(np.zeros((0, 3)), []).shape == (0, 3, 3)
