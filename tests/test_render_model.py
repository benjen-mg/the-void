"""Regression tests for the dangling-bond defect.

The viewer used to draw spheres for the reference cell only, while
drawing bonds out to periodic-image neighbours. Any molecule crossing a
cell face rendered as a fragment with sticks trailing into empty space.
The invariant that fixes it: every bond in a RenderModel must have an
atom at both ends.
"""

import math

import pytest

pytest.importorskip("numpy")

from thevoid_core.services.bonds.bond_model import Bond  # noqa: E402
from thevoid_core.services.structure.render_model import (  # noqa: E402
    CompletionMode,
    RenderModelBuilder,
)


class FakeSpecies:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeSite:
    def __init__(self, symbol, coords, frac_coords=None):
        self.species = {FakeSpecies(symbol): 1.0}
        self.coords = coords
        # Every fixture in this file uses CUBIC_10A, so cartesian / 10
        # is the fractional coordinate unless a test overrides it.
        self.frac_coords = (
            frac_coords
            if frac_coords is not None
            else [c / 10.0 for c in coords]
        )


class FakeLattice:
    def __init__(self, matrix):
        self.matrix = matrix


class FakeStructure:
    def __init__(self, matrix, sites):
        self.lattice = FakeLattice(matrix)
        self.sites = sites

    def __len__(self):
        return len(self.sites)

    def __getitem__(self, index):
        return self.sites[index]


CUBIC_10A = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def straddling_dimer():
    """Two atoms 1 A apart, bonded through the x cell face."""
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [0.5, 5.0, 5.0]),
            FakeSite("O", [9.5, 5.0, 5.0]),
        ],
    )

    bonds = [Bond(atom1=0, atom2=1, image=(-1, 0, 0), length=1.0)]

    return structure, bonds


def every_bond_is_terminated(model) -> bool:
    return all(
        0 <= bond.atom_a < len(model.atoms)
        and 0 <= bond.atom_b < len(model.atoms)
        for bond in model.bonds
    )


def test_completion_materialises_the_image_atoms():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(
        structure,
        bonds,
        mode=CompletionMode.COMPLETE_MOLECULES,
    )

    # Both molecules are made whole: the in-cell pair plus one image
    # partner each.
    assert len(model.atoms) == 4
    assert len(model.bonds) == 2
    assert model.image_atom_count == 2
    assert every_bond_is_terminated(model)
    assert not model.fell_back


def test_completed_bond_has_the_right_geometry():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(structure, bonds)

    for bond in model.bonds:
        a = model.atoms[bond.atom_a].position
        b = model.atoms[bond.atom_b].position

        distance = math.dist(a, b)

        # 1 A through the face, not 9 A the long way round the cell.
        assert distance == pytest.approx(1.0, abs=1e-9)


def test_cell_only_mode_hides_bonds_it_cannot_terminate():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(
        structure,
        bonds,
        mode=CompletionMode.CELL_ONLY,
    )

    assert len(model.atoms) == 2
    assert model.bonds == []
    assert every_bond_is_terminated(model)


def test_hydrogens_can_be_suppressed():
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [5.0, 5.0, 5.0]),
            FakeSite("H", [6.1, 5.0, 5.0]),
        ],
    )
    bonds = [Bond(atom1=0, atom2=1, image=(0, 0, 0), length=1.1)]

    model = RenderModelBuilder().build(
        structure,
        bonds,
        show_hydrogens=False,
    )

    assert [atom.symbol for atom in model.atoms] == ["C"]
    assert model.bonds == []


def test_extended_structure_falls_back_instead_of_running_away():
    """A site bonded to its own translate percolates forever."""
    structure = FakeStructure(
        CUBIC_10A,
        [FakeSite("Na", [0.0, 0.0, 0.0])],
    )
    bonds = [Bond(atom1=0, atom2=0, image=(1, 0, 0), length=10.0)]

    builder = RenderModelBuilder()
    builder.MAX_ATOMS = 50

    model = builder.build(structure, bonds)

    assert model.fell_back
    assert len(model.atoms) <= 50
    assert every_bond_is_terminated(model)


def test_periodic_mof_like_framework_completes_without_falling_back():
    """A genuinely 3-periodic metal-organic framework - every linker
    connects to a metal, which connects to another linker, forever -
    used to be indistinguishable from the pathological "atom bonded to
    its own translate" case above: unbounded completion cannot
    terminate either way, so it fell back to a coordination shell,
    which truncates linkers wherever the shell boundary happens to cut
    through them. Confirmed directly against a real deposited MOF file
    before this fix (RenderModel.fell_back was True); this is the
    smallest synthetic case that reproduces the same mechanism: a
    repeating -Zn-N=N-Zn-N=N- chain with no discrete "molecule"
    boundary at all.
    """
    structure = FakeStructure(
        [[5.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        [
            FakeSite("Zn", [0.0, 5.0, 5.0]),
            FakeSite("N", [1.0, 5.0, 5.0]),
            FakeSite("N", [4.0, 5.0, 5.0]),
        ],
    )
    bonds = [
        Bond(atom1=0, atom2=1, image=(0, 0, 0), length=2.0),
        Bond(atom1=1, atom2=2, image=(0, 0, 0), length=1.2),
        Bond(atom1=2, atom2=0, image=(1, 0, 0), length=2.0),
    ]

    model = RenderModelBuilder().build(structure, bonds)

    assert model.fell_back is False
    assert every_bond_is_terminated(model)
    # Every drawn Zn must have at least one N neighbour actually
    # materialised - the point of the fix, not just "didn't crash".
    zn_indices = [
        i for i, atom in enumerate(model.atoms) if atom.symbol == "Zn"
    ]
    assert zn_indices
    for zn_index in zn_indices:
        assert any(
            bond.atom_a == zn_index or bond.atom_b == zn_index
            for bond in model.bonds
        )


def test_purely_metallic_percolation_still_falls_back_normally():
    """The metal-boundary attempt must not be trusted blindly: a
    purely metallic or ionic extended solid, where the percolating
    bonds are themselves metal-to-metal (not organic linkers bridging
    metal nodes), would degenerate under a naive "stop at every metal"
    rule to just the bare seeds and zero bonds - a worse result than
    the coordination-shell fallback, not a genuine completion. This is
    exactly the existing test_extended_structure_falls_back... case
    (Na is a metal), which must keep behaving exactly as before: still
    a normal fallback, not a degenerate empty-bonds "success".
    """
    structure = FakeStructure(
        CUBIC_10A,
        [FakeSite("Na", [0.0, 0.0, 0.0])],
    )
    bonds = [Bond(atom1=0, atom2=0, image=(1, 0, 0), length=10.0)]

    model = RenderModelBuilder().build(structure, bonds)

    assert model.fell_back is True
    assert len(model.bonds) > 0  # the 1-shell fallback, not zero bonds


def test_metal_atoms_own_bonds_are_still_fully_drawn():
    """The metal-boundary rule stops a metal from being a *launching
    point* for new territory - it must not stop the metal's own bonds
    to already-reachable atoms from being drawn. A single discrete
    coordination complex (not periodic at all) must render exactly as
    it always did.
    """
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("Zn", [5.0, 5.0, 5.0]),
            FakeSite("N", [6.5, 5.0, 5.0]),
            FakeSite("N", [3.5, 5.0, 5.0]),
            FakeSite("N", [5.0, 6.5, 5.0]),
            FakeSite("N", [5.0, 3.5, 5.0]),
        ],
    )
    bonds = [
        Bond(atom1=0, atom2=1, image=(0, 0, 0), length=1.5),
        Bond(atom1=0, atom2=2, image=(0, 0, 0), length=1.5),
        Bond(atom1=0, atom2=3, image=(0, 0, 0), length=1.5),
        Bond(atom1=0, atom2=4, image=(0, 0, 0), length=1.5),
    ]

    model = RenderModelBuilder().build(structure, bonds)

    assert model.fell_back is False
    assert len(model.atoms) == 5
    assert len(model.bonds) == 4
    assert every_bond_is_terminated(model)


def test_no_duplicate_bonds_between_the_same_pair():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(structure, bonds)

    keys = {
        tuple(sorted((bond.atom_a, bond.atom_b))) for bond in model.bonds
    }

    assert len(keys) == len(model.bonds)


def test_shell_fallback_also_respects_the_budget():
    """A dense cell where even one shell overruns must not crash."""
    sites = [FakeSite("C", [float(i), 0.0, 0.0]) for i in range(10)]
    structure = FakeStructure(CUBIC_10A, sites)

    bonds = [
        Bond(atom1=i, atom2=j, image=(1, 0, 0), length=1.5)
        for i in range(10)
        for j in range(10)
    ]

    builder = RenderModelBuilder()
    builder.MAX_ATOMS = 12

    model = builder.build(structure, bonds)

    assert model.fell_back
    assert every_bond_is_terminated(model)


def test_fragments_are_identified_and_grouped():
    """A framework chain plus two free nitrates."""
    sites = [
        FakeSite("Zn", [1.0, 1.0, 1.0]),
        FakeSite("N", [2.0, 1.0, 1.0]),
        FakeSite("N", [5.0, 5.0, 5.0]),
        FakeSite("O", [6.0, 5.0, 5.0]),
        FakeSite("N", [7.0, 7.0, 7.0]),
        FakeSite("O", [8.0, 7.0, 7.0]),
    ]
    structure = FakeStructure(CUBIC_10A, sites)

    bonds = [
        Bond(atom1=0, atom2=1, image=(0, 0, 0), length=2.0),
        Bond(atom1=2, atom2=3, image=(0, 0, 0), length=1.25),
        Bond(atom1=4, atom2=5, image=(0, 0, 0), length=1.25),
    ]

    model = RenderModelBuilder().build(
        structure,
        bonds,
        mode=CompletionMode.CELL_ONLY,
    )

    assert model.fragment_count == 3

    by_formula = {f.formula: f for f in model.fragments}

    assert by_formula["NO"].occurrences == 2
    assert by_formula["NZn"].occurrences == 1
    assert by_formula["NO"].is_isolated_ion_or_solvent


def test_lone_atom_is_its_own_fragment():
    """An unbonded counter-ion must still be counted, not dropped."""
    structure = FakeStructure(
        CUBIC_10A,
        [FakeSite("C", [1.0, 1.0, 1.0]), FakeSite("Na", [8.0, 8.0, 8.0])],
    )

    model = RenderModelBuilder().build(structure, [])

    assert model.fragment_count == 2


def test_minor_disorder_components_can_be_hidden():
    class PartialSite(FakeSite):
        def __init__(self, symbol, coords, occupancy):
            self.species = {FakeSpecies(symbol): occupancy}
            self.coords = coords

    structure = FakeStructure(
        CUBIC_10A,
        [
            PartialSite("O", [5.0, 5.0, 5.0], 1.0),
            PartialSite("O", [5.6, 5.0, 5.0], 0.4),
        ],
    )

    full = RenderModelBuilder().build(structure, [])
    trimmed = RenderModelBuilder().build(structure, [], min_occupancy=0.5)

    assert len(full.atoms) == 2
    assert len(trimmed.atoms) == 1
    assert trimmed.hidden_partial_sites == 1


def test_supercell_tiles_the_reference_content():
    """A 2x2x2 request on an unbonded lone atom gives 8 copies."""
    structure = FakeStructure(CUBIC_10A, [FakeSite("Na", [1.0, 1.0, 1.0])])

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        supercell=(2, 2, 2),
    )

    assert len(model.atoms) == 8
    assert model.supercell == (2, 2, 2)

    images = {atom.image for atom in model.atoms}
    assert images == {
        (i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)
    }


def test_supercell_does_not_duplicate_a_bond_spanning_two_tiles():
    """A dimer straddling the boundary inside a 2x1x1 supercell.

    The bond is seeded from both tile (0,0,0) and tile (1,0,0); it must
    still be drawn once, not twice.
    """
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [0.5, 5.0, 5.0]),
            FakeSite("O", [9.5, 5.0, 5.0]),
        ],
    )
    bonds = [Bond(atom1=0, atom2=1, image=(-1, 0, 0), length=1.0)]

    model = RenderModelBuilder().build(
        structure,
        bonds,
        supercell=(2, 1, 1),
    )

    keys = {tuple(sorted((b.atom_a, b.atom_b))) for b in model.bonds}
    assert len(keys) == len(model.bonds)


def test_outside_supercell_count_reflects_completion_only():
    """Atoms inside the requested tiling are not 'extra'."""
    structure = FakeStructure(CUBIC_10A, [FakeSite("Na", [1.0, 1.0, 1.0])])

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        supercell=(2, 2, 2),
    )

    assert model.outside_supercell_count == 0


def test_completion_can_still_extend_past_the_supercell_edge():
    """A molecule straddling the far edge of the tiling gets completed."""
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [9.5, 5.0, 5.0]),
            FakeSite("O", [0.5, 5.0, 5.0]),
        ],
    )
    # Bonded across the +x face of tile (1,0,0) into tile (2,0,0), which
    # lies outside a 2x1x1 request.
    bonds = [Bond(atom1=0, atom2=1, image=(1, 0, 0), length=1.0)]

    model = RenderModelBuilder().build(
        structure,
        bonds,
        supercell=(2, 1, 1),
    )

    assert model.outside_supercell_count > 0
    assert every_bond_is_terminated(model)


def test_default_supercell_is_a_single_cell():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(structure, bonds)

    assert model.supercell == (1, 1, 1)


def test_invalid_supercell_dims_are_rejected():
    structure = FakeStructure(CUBIC_10A, [FakeSite("Na", [1.0, 1.0, 1.0])])

    with pytest.raises(ValueError):
        RenderModelBuilder().build(structure, [], supercell=(0, 1, 1))


def test_bonded_atom_indices_covers_both_ends_of_every_bond():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(structure, bonds)

    assert model.bonded_atom_indices == set(range(len(model.atoms)))


def test_isolated_atom_indices_finds_the_unbonded_atom():
    """One bonded pair plus one lone ion."""
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [1.0, 1.0, 1.0]),
            FakeSite("O", [2.0, 1.0, 1.0]),
            FakeSite("Na", [8.0, 8.0, 8.0]),
        ],
    )
    bonds = [Bond(atom1=0, atom2=1, image=(0, 0, 0), length=1.3)]

    model = RenderModelBuilder().build(
        structure, bonds, mode=CompletionMode.CELL_ONLY
    )

    isolated_symbols = {model.atoms[i].symbol for i in model.isolated_atom_indices}

    assert isolated_symbols == {"Na"}


def test_isolated_atom_indices_empty_when_everything_is_bonded():
    structure, bonds = straddling_dimer()

    model = RenderModelBuilder().build(structure, bonds)

    assert model.isolated_atom_indices == set()


def test_isolated_atom_indices_is_everything_with_no_bonds_at_all():
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Ar", [1.0, 1.0, 1.0])]
    )

    model = RenderModelBuilder().build(structure, [])

    assert model.isolated_atom_indices == {0}


def test_face_atom_gets_one_replica_in_a_single_cell():
    """The classic single-cell case: an atom at frac x=0 gets a
    duplicate drawn at x=1, the far face of that same cell.
    """
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Na", [0.0, 5.0, 5.0])]
    )

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        show_boundary_replicas=True,
    )

    assert len(model.atoms) == 2
    assert model.boundary_replica_count == 1

    images = {atom.image for atom in model.atoms}
    assert images == {(0, 0, 0), (1, 0, 0)}


def test_edge_atom_gets_three_replicas():
    """An atom on two faces at once (an edge of the cell) needs
    replicas at each face plus the edge itself: three total, four
    atoms drawn.
    """
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Na", [0.0, 0.0, 5.0])]
    )

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        show_boundary_replicas=True,
    )

    assert model.boundary_replica_count == 3

    images = {atom.image for atom in model.atoms}
    assert images == {
        (0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
    }


def test_corner_atom_gets_seven_replicas():
    """A corner atom (on all three faces) needs all eight corners of
    the box represented: seven replicas plus the original.
    """
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Na", [0.0, 0.0, 0.0])]
    )

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        show_boundary_replicas=True,
    )

    assert model.boundary_replica_count == 7
    assert len(model.atoms) == 8

    images = {atom.image for atom in model.atoms}
    assert images == {
        (i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)
    }


def test_interior_atom_gets_no_replicas():
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Na", [5.0, 5.0, 5.0])]
    )

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        show_boundary_replicas=True,
    )

    assert model.boundary_replica_count == 0
    assert len(model.atoms) == 1


def test_off_by_default():
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Na", [0.0, 0.0, 0.0])]
    )

    model = RenderModelBuilder().build(
        structure, [], mode=CompletionMode.CELL_ONLY
    )

    assert model.boundary_replica_count == 0
    assert len(model.atoms) == 1


def test_interior_tiles_of_a_supercell_get_no_replica_only_the_outer_face():
    """In a 2x1x1 supercell, a face atom in the first tile is not
    replicated (its 'mirror' is already the second tile's own atom at
    the same physical position); only the second (last) tile's face
    atom gets a replica, at the true outer edge of the whole box.
    """
    structure = FakeStructure(
        CUBIC_10A, [FakeSite("Na", [0.0, 5.0, 5.0])]
    )

    model = RenderModelBuilder().build(
        structure,
        [],
        mode=CompletionMode.CELL_ONLY,
        supercell=(2, 1, 1),
        show_boundary_replicas=True,
    )

    # Two tiles (images x=0 and x=1) plus exactly one outer replica
    # at x=2.
    assert model.boundary_replica_count == 1

    images = {atom.image for atom in model.atoms}
    assert images == {(0, 0, 0), (1, 0, 0), (2, 0, 0)}


def test_replicas_have_no_bonds():
    """Replicas are decorative only - documented, tested directly."""
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [0.0, 5.0, 5.0]),
            FakeSite("O", [1.2, 5.0, 5.0]),
        ],
    )
    bonds = [Bond(atom1=0, atom2=1, image=(0, 0, 0), length=1.2)]

    model = RenderModelBuilder().build(
        structure,
        bonds,
        mode=CompletionMode.CELL_ONLY,
        show_boundary_replicas=True,
    )

    # One real bond; the boundary replica of the C atom (on the x=0
    # face) has no bond of its own.
    assert len(model.bonds) == 1
    assert model.boundary_replica_count == 1


def test_replica_does_not_duplicate_an_atom_completion_already_placed():
    """If molecule completion already pulled in the mirror image (a
    bond crosses that face), no second decorative copy is added on
    top of it.
    """
    structure = FakeStructure(
        CUBIC_10A,
        [
            FakeSite("C", [0.0, 5.0, 5.0]),
            FakeSite("O", [9.5, 5.0, 5.0]),
        ],
    )
    bonds = [Bond(atom1=0, atom2=1, image=(-1, 0, 0), length=1.2)]

    model = RenderModelBuilder().build(
        structure,
        bonds,
        mode=CompletionMode.COMPLETE_MOLECULES,
        show_boundary_replicas=True,
    )

    images = [
        atom.image for atom in model.atoms if atom.site_index == 0
    ]
    # C at (0,0,0) plus its completion-pulled image at (1,0,0) from
    # the bond - no third decorative duplicate on top of the second.
    assert sorted(images) == [(0, 0, 0), (1, 0, 0)]
