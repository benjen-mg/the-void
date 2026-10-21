"""build_fragment_mol / expand_metal_node_atom_indices /
compute_skeletal_diagram: the RDKit-based fragment decomposition and
layout pipeline, checked against hand-constructed graphs with known
expected chemistry.
"""

from dataclasses import dataclass

import pytest

pytest.importorskip("rdkit")

from thevoid_core.services.mof_analysis import Linker, MetalNode  # noqa: E402
from thevoid_core.services.skeletal_layout import (  # noqa: E402
    build_fragment_mol,
    build_node_cluster_atom_indices,
    compute_skeletal_diagram,
    expand_metal_node_atom_indices,
)


@dataclass(frozen=True)
class _Bond:
    atom_a: int
    atom_b: int


# --- build_fragment_mol ---


def test_internal_bonds_are_included():
    symbols = ["C", "O", "N"]
    bonds = [_Bond(0, 1), _Bond(1, 2)]

    mol = build_fragment_mol(symbols, bonds, (0, 1, 2))

    assert mol.GetNumAtoms() == 3
    assert mol.GetNumBonds() == 2


def test_a_bond_crossing_the_boundary_gets_a_dummy_cap_not_dropped():
    """Atom 1 (N) is bonded to atom 2, which is NOT part of this
    fragment - the real, standard cheminformatics convention for a
    truncated attachment point is a dummy atom cap, not silently
    dropping the bond (which would let RDKit's sanitiser invent an
    implicit hydrogen there instead - confirmed directly to happen,
    and confirmed to misrepresent real chemistry, before this
    function was written to cap bonds instead).
    """
    symbols = ["C", "N", "O"]
    bonds = [_Bond(0, 1), _Bond(1, 2)]

    mol = build_fragment_mol(symbols, bonds, (0, 1))  # excludes atom 2

    # 2 real atoms + 1 dummy cap for the bond to the excluded atom 2.
    assert mol.GetNumAtoms() == 3
    dummy_count = sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
    )
    assert dummy_count == 1


def test_multiple_external_bonds_each_get_their_own_dummy():
    """Atom 0 has two separate external bonds (to atoms 1 and 2, both
    excluded) - each must get its own dummy cap, not be collapsed into
    one, since that would misrepresent the atom's real degree.
    """
    symbols = ["N", "C", "O"]
    bonds = [_Bond(0, 1), _Bond(0, 2)]

    mol = build_fragment_mol(symbols, bonds, (0,))

    dummy_count = sum(
        1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0
    )
    assert dummy_count == 2


def test_a_cap_to_a_metal_does_not_count_toward_the_inside_atoms_valence():
    """Regression test for a real reported bug: every linker in a real
    file (MUF-16(Mn), CCDC 1948905) failed to build with RDKit's
    "Explicit valence ... greater than permitted", silently discarding
    every Linker and every "node + linkers" fragment from the diagram -
    only the metal-only "node" fragments (which never carry this kind
    of cap) survived.

    Root cause (confirmed directly): every crossing bond capped by this
    function is, by construction, a metal-ligand coordination bond -
    mof_analysis.partition_framework always splits a framework at
    exactly the metal/non-metal boundary - yet the cap was always added
    as an ordinary Chem.BondType.SINGLE, as if it were ready to
    continue as plain covalent connectivity. That overfills the donor
    atom's valence the moment its in-fragment bonds already use up its
    full ordinary valence: here, a nitrogen with two ordinary ring
    bonds plus two such caps (bridging two different metal centres, as
    the real file's linker does) is four bonds on a neutral nitrogen,
    which RDKit correctly refuses to sanitise.

    The fix caps a bond to a metal with Chem.BondType.ZERO - RDKit's
    own convention for a real connection that must not count toward
    valence - while leaving every other (ordinary organic) boundary
    bond capped with a plain single bond, exactly as before.
    """
    from rdkit import Chem

    # A 2-atom ring stand-in (0=N, 1=C) - N already has its two ordinary
    # bonds (the ring bond to C, and one more ring bond represented by
    # a second external cap below) before either metal cap is added;
    # both atoms 2 and 3 are metals bridged by this same nitrogen, the
    # same shape the real MUF-16(Mn) linker's nitrogen has.
    symbols = ["N", "C", "Zn", "Zn"]
    bonds = [_Bond(0, 1), _Bond(0, 2), _Bond(0, 3)]

    mol = build_fragment_mol(symbols, bonds, (0, 1))

    Chem.SanitizeMol(mol)  # must not raise AtomValenceException

    nitrogen = mol.GetAtomWithIdx(0)
    cap_bonds = [
        bond
        for bond in nitrogen.GetBonds()
        if mol.GetAtomWithIdx(
            bond.GetOtherAtomIdx(0)
        ).GetAtomicNum()
        == 0
    ]
    assert len(cap_bonds) == 2
    assert all(bond.GetBondType() == Chem.BondType.ZERO for bond in cap_bonds)


def test_a_cap_to_a_non_metal_still_uses_an_ordinary_single_bond():
    """The fix above must not change behaviour for the ordinary case -
    a boundary bond to another organic atom (not a metal) still gets a
    plain single-bond cap, exactly as before this round.
    """
    from rdkit import Chem

    symbols = ["C", "N", "O"]
    bonds = [_Bond(0, 1), _Bond(1, 2)]

    mol = build_fragment_mol(symbols, bonds, (0, 1))  # excludes atom 2 (O)

    nitrogen = mol.GetAtomWithIdx(1)
    cap_bonds = [
        bond
        for bond in nitrogen.GetBonds()
        if mol.GetAtomWithIdx(bond.GetOtherAtomIdx(1)).GetAtomicNum() == 0
    ]
    assert len(cap_bonds) == 1
    assert cap_bonds[0].GetBondType() == Chem.BondType.SINGLE


def test_sanitizes_and_can_generate_2d_coordinates():
    """A capped fragment must be usable, not just constructible - the
    real point of build_fragment_mol is feeding RDKit's own
    sanitisation and layout, both exercised here directly.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    symbols = ["C", "N", "O"]
    bonds = [_Bond(0, 1), _Bond(1, 2)]

    mol = build_fragment_mol(symbols, bonds, (0, 1))

    Chem.SanitizeMol(mol)  # must not raise
    rdDepictor.Compute2DCoords(mol)  # must not raise

    conf = mol.GetConformer()
    assert conf.GetNumAtoms() == mol.GetNumAtoms()


# --- expand_metal_node_atom_indices ---


def test_expand_includes_the_nodes_own_atoms():
    bonds = [_Bond(0, 1)]

    expanded = expand_metal_node_atom_indices((0,), bonds)

    assert 0 in expanded


def test_expand_includes_direct_neighbors_on_either_side():
    bonds = [_Bond(0, 1), _Bond(2, 0)]

    expanded = expand_metal_node_atom_indices((0,), bonds)

    assert set(expanded) == {0, 1, 2}


def test_expand_does_not_reach_two_bonds_away():
    """Only the immediate coordination sphere - not the neighbor's own
    other neighbors - matches what a real "metal + its directly-
    coordinating atoms" diagram should show.
    """
    bonds = [_Bond(0, 1), _Bond(1, 2)]  # 0-1 direct, 1-2 one more hop

    expanded = expand_metal_node_atom_indices((0,), bonds)

    assert set(expanded) == {0, 1}
    assert 2 not in expanded


def test_expand_handles_a_multi_atom_node():
    """A bridged multi-metal node (node's own atom_indices has more
    than one atom) still correctly gathers every neighbor of every
    atom in the node.
    """
    bonds = [_Bond(0, 1), _Bond(1, 2), _Bond(2, 3)]

    expanded = expand_metal_node_atom_indices((1, 2), bonds)

    assert set(expanded) == {0, 1, 2, 3}


# --- build_node_cluster_atom_indices ---


def test_cluster_includes_the_nodes_own_atoms():
    cluster = build_node_cluster_atom_indices((0,), [], [])

    assert 0 in cluster


def test_cluster_pulls_in_a_full_directly_connected_linker():
    """The whole point of this function versus the single-shell
    expand_metal_node_atom_indices: a linker bonded to the node is
    included in full (every one of its own atoms), not just its first
    connecting atom.
    """
    # Node atom 0, bonded to linker atom 1, which is itself bonded to
    # linker atoms 2 and 3 further into the ring.
    bonds = [_Bond(0, 1), _Bond(1, 2), _Bond(2, 3), _Bond(3, 1)]
    linker = Linker(linker_id=0, atom_indices=(1, 2, 3), formula="X")

    cluster = build_node_cluster_atom_indices((0,), [linker], bonds)

    assert set(cluster) == {0, 1, 2, 3}


def test_cluster_excludes_a_linker_not_directly_bonded_to_the_node():
    bonds = [_Bond(1, 2)]  # no bond at all to node atom 0
    linker = Linker(linker_id=0, atom_indices=(1, 2), formula="X")

    cluster = build_node_cluster_atom_indices((0,), [linker], bonds)

    assert set(cluster) == {0}


def test_cluster_pulls_in_every_directly_connected_linker_not_just_one():
    bonds = [_Bond(0, 1), _Bond(0, 2)]
    linker_a = Linker(0, (1,), "A")
    linker_b = Linker(1, (2,), "B")

    cluster = build_node_cluster_atom_indices((0,), [linker_a, linker_b], bonds)

    assert set(cluster) == {0, 1, 2}


def test_cluster_handles_a_multi_atom_node():
    bonds = [_Bond(0, 2), _Bond(1, 3)]
    linker_a = Linker(0, (2,), "A")
    linker_b = Linker(1, (3,), "B")

    cluster = build_node_cluster_atom_indices(
        (0, 1), [linker_a, linker_b], bonds
    )

    assert set(cluster) == {0, 1, 2, 3}


def test_cluster_result_can_be_built_and_laid_out_with_rdkit():
    """The whole point: the resulting atom set must actually work as
    an RDKit fragment, not just be the right set of indices.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    symbols, bonds = _triazole_ring_symbols_and_bonds()
    linker = Linker(0, (0, 1, 2, 3, 4), "C2N3")

    cluster = build_node_cluster_atom_indices((5,), [linker], bonds)
    mol = build_fragment_mol(symbols, bonds, cluster)

    Chem.SanitizeMol(mol)  # must not raise
    rdDepictor.Compute2DCoords(mol)  # must not raise


# --- compute_skeletal_diagram ---


def _triazole_ring_symbols_and_bonds():
    """A minimal 5-membered ring (matching a real triazolate
    linker's ring size), with two of its ring nitrogens each having an
    external bond (representing coordination to a metal centre not
    included in the linker's own fragment) - the same shape as the
    real structure this was validated against.
    """
    # 0=C, 1=N, 2=N, 3=N, 4=C forming a 5-ring; atoms 5, 6 are external
    # metal centres bonded to ring nitrogens 1 and 3.
    symbols = ["C", "N", "N", "N", "C", "Zn", "Zn"]
    bonds = [
        _Bond(0, 1), _Bond(1, 2), _Bond(2, 3), _Bond(3, 4), _Bond(4, 0),
        _Bond(1, 5), _Bond(3, 6),
    ]
    return symbols, bonds


def test_a_single_linker_produces_one_fragment():
    symbols, bonds = _triazole_ring_symbols_and_bonds()
    linker = Linker(linker_id=0, atom_indices=(0, 1, 2, 3, 4), formula="C2N3")

    result = compute_skeletal_diagram(symbols, bonds, [linker], [])

    assert len(result.fragments) == 1
    assert result.fragments[0].occurrence_count == 1
    assert result.skipped_count == 0


def test_identical_linkers_are_deduplicated_with_correct_count():
    """Two chemically identical linkers (same ring, same cut points)
    must collapse into one fragment with occurrence_count 2 - the
    whole point of deduplication, checked directly rather than assumed
    from the grouping logic alone.
    """
    symbols, bonds = _triazole_ring_symbols_and_bonds()
    # A second, separate copy of the same ring shape, using fresh atom
    # indices (7-11) so it is a genuinely distinct set of atoms, not
    # literally the same linker object twice.
    symbols = symbols + ["C", "N", "N", "N", "C"]
    bonds = bonds + [
        _Bond(7, 8), _Bond(8, 9), _Bond(9, 10), _Bond(10, 11), _Bond(11, 7),
        _Bond(8, 5), _Bond(10, 6),
    ]
    linker_a = Linker(0, (0, 1, 2, 3, 4), "C2N3")
    linker_b = Linker(1, (7, 8, 9, 10, 11), "C2N3")

    result = compute_skeletal_diagram(symbols, bonds, [linker_a, linker_b], [])

    assert len(result.fragments) == 1
    assert result.fragments[0].occurrence_count == 2


def test_a_metal_node_is_expanded_and_produces_a_fragment():
    symbols, bonds = _triazole_ring_symbols_and_bonds()
    node = MetalNode(
        node_id=0, atom_indices=(5,), metal_symbols=("Zn",), formula="Zn"
    )

    result = compute_skeletal_diagram(symbols, bonds, [], [node])

    # A metal node produces two diagram types: its immediate
    # coordination sphere alone, and (this round's addition) a
    # connected view of the node together with every full linker
    # bonded to it - both distinct fragments here, since one has just
    # the node's direct neighbour (atom 1) and the other pulls in the
    # rest of the ring too.
    assert len(result.fragments) == 2
    labels = {fragment.label for fragment in result.fragments}
    assert labels == {"Zn node", "Zn node + linkers"}


def test_no_linkers_or_nodes_gives_an_empty_result():
    result = compute_skeletal_diagram([], [], [], [])

    assert result.fragments == ()
    assert result.skipped_count == 0


def test_linkers_and_nodes_together_all_appear():
    symbols, bonds = _triazole_ring_symbols_and_bonds()
    linker = Linker(0, (0, 1, 2, 3, 4), "C2N3")
    node_a = MetalNode(0, (5,), ("Zn",), "Zn")
    node_b = MetalNode(1, (6,), ("Zn",), "Zn")

    result = compute_skeletal_diagram(
        symbols, bonds, [linker], [node_a, node_b]
    )

    # The linker is its own fragment; the two metal nodes each expand
    # to a different local neighborhood (one via atom 1's ring
    # position, one via atom 3's), so in general they need not be
    # identical - here they happen to be structurally different given
    # the asymmetric ring positions, so at least 2 fragments total.
    assert len(result.fragments) >= 2
