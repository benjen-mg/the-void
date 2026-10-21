"""partition_framework(): the core of MOF linker/metal-node detection.

Pure, no Qt/pymatgen/viewer dependency - is_metal is injected as a
plain callable so these tests don't need a real element database, just
a simple set membership check.
"""

from thevoid_core.services.mof_analysis import (
    connected_components,
    hill_formula,
    identify_bridging_atoms,
    partition_framework,
)


class FakeBond:
    def __init__(self, atom_a, atom_b):
        self.atom_a = atom_a
        self.atom_b = atom_b


def metal_set(*symbols):
    """Build an is_metal callable from an explicit set of symbols -
    keeps each test's chemistry assumptions visible in the test itself
    rather than depending on a real periodic table lookup.
    """
    metals = set(symbols)
    return lambda symbol: symbol in metals


IS_METAL = metal_set("Zn", "Cu", "Fe", "Co", "Ni", "Zr", "Ti", "Al")


# --- hill_formula ---


def test_hill_formula_puts_carbon_and_hydrogen_first():
    assert hill_formula({"O": 2, "C": 6, "H": 4}) == "C6H4O2"


def test_hill_formula_without_carbon_is_plain_alphabetical():
    assert hill_formula({"Zn": 1, "O": 4}) == "O4Zn"


def test_hill_formula_omits_count_of_one():
    assert hill_formula({"C": 1, "H": 1}) == "CH"


# --- connected_components ---


def test_connected_components_groups_linked_nodes():
    components = connected_components([0, 1, 2, 3], [(0, 1), (1, 2)])

    grouped = sorted(sorted(c) for c in components)
    assert grouped == [[0, 1, 2], [3]]


def test_connected_components_handles_no_edges():
    components = connected_components([0, 1, 2], [])

    assert sorted(sorted(c) for c in components) == [[0], [1], [2]]


def test_connected_components_handles_empty_input():
    assert connected_components([], []) == []


# --- identify_bridging_atoms ---


def test_bridging_atom_touches_two_different_metals():
    # O(1) bonded to Zn(0) and Zn(2): a mu-oxo bridge.
    symbols = ["Zn", "O", "Zn"]
    bonds = [FakeBond(0, 1), FakeBond(1, 2)]

    assert identify_bridging_atoms(symbols, bonds, IS_METAL) == {1}


def test_non_bridging_coordinating_atom_is_not_flagged():
    # O(1) bonded only to Zn(0), and to a carbon (2) - an ordinary
    # carboxylate oxygen coordinating a single metal, not a bridge.
    symbols = ["Zn", "O", "C"]
    bonds = [FakeBond(0, 1), FakeBond(1, 2)]

    assert identify_bridging_atoms(symbols, bonds, IS_METAL) == set()


def test_atom_bonded_to_the_same_metal_twice_is_not_a_bridge():
    """A chelating ligand atom bonded twice to the *same* metal (a
    five-membered chelate ring, say) is not bridging *different*
    metals, and must not be misclassified as one.
    """
    symbols = ["Zn", "O"]
    bonds = [FakeBond(0, 1), FakeBond(0, 1)]  # duplicate on purpose

    assert identify_bridging_atoms(symbols, bonds, IS_METAL) == set()


# --- partition_framework: single metal centre ---


def test_isolated_metal_ion_becomes_its_own_node():
    """Zn coordinated by two separate, non-bridging carboxylate
    fragments: one metal node, two distinct linkers.
    """
    # 0=Zn, 1=O, 2=C, 3=O  (linker A: O-C-O, coordinating via atom 1)
    # 4=O, 5=C, 6=O        (linker B: O-C-O, coordinating via atom 4)
    symbols = ["Zn", "O", "C", "O", "O", "C", "O"]
    bonds = [
        FakeBond(0, 1), FakeBond(1, 2), FakeBond(2, 3),
        FakeBond(0, 4), FakeBond(4, 5), FakeBond(5, 6),
    ]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert result.metal_node_count == 1
    assert result.metal_nodes[0].metal_symbols == ("Zn",)
    assert result.metal_nodes[0].atom_indices == (0,)

    assert result.linker_count == 2
    linker_formulas = sorted(linker.formula for linker in result.linkers)
    assert linker_formulas == ["CO2", "CO2"]


# --- partition_framework: bridged multi-metal node ---


def test_mu_oxo_bridged_dimer_becomes_one_node():
    # Zn(0)-O(1)-Zn(2), each Zn also coordinated by its own linker.
    symbols = ["Zn", "O", "Zn", "O", "C", "O", "O", "C", "O"]
    bonds = [
        FakeBond(0, 1), FakeBond(1, 2),           # the mu-oxo bridge
        FakeBond(0, 3), FakeBond(3, 4), FakeBond(4, 5),  # linker on Zn0
        FakeBond(2, 6), FakeBond(6, 7), FakeBond(7, 8),  # linker on Zn2
    ]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert result.metal_node_count == 1
    node = result.metal_nodes[0]
    assert node.metal_symbols == ("Zn", "Zn")
    assert node.coordination_number == 2
    assert set(node.atom_indices) == {0, 1, 2}  # both Zn and the bridge
    assert node.formula == "OZn2"

    assert result.linker_count == 2


def test_direct_metal_metal_bond_becomes_one_node():
    # Cu(0)-Cu(1) directly bonded (paddlewheel-style), no bridging atom.
    symbols = ["Cu", "Cu", "O", "C", "O"]
    bonds = [
        FakeBond(0, 1),
        FakeBond(0, 2), FakeBond(2, 3), FakeBond(3, 4),
    ]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert result.metal_node_count == 1
    assert result.metal_nodes[0].metal_symbols == ("Cu", "Cu")


def test_paddlewheel_without_a_bridge_or_direct_bond_splits_into_two_nodes():
    """The documented limitation: two metals connected only via
    non-bridging atoms (no shared bridge, no direct M-M bond) are not
    recognised as one node by this heuristic.
    """
    # Cu(0) and Cu(1), each coordinated by its own oxygen, no direct
    # Cu-Cu bond and no single oxygen touching both.
    symbols = ["Cu", "O", "Cu", "O"]
    bonds = [FakeBond(0, 1), FakeBond(2, 3)]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert result.metal_node_count == 2


# --- partition_framework: metalloids stay in the linker ---


def test_metalloid_in_a_linker_is_not_treated_as_a_metal():
    # A boron-containing linker coordinating Zn: B must stay part of
    # the linker fragment, not be pulled into the metal node.
    is_metal = metal_set("Zn")  # B deliberately excluded

    symbols = ["Zn", "O", "B", "O"]
    bonds = [FakeBond(0, 1), FakeBond(1, 2), FakeBond(2, 3)]

    result = partition_framework(symbols, bonds, is_metal)

    assert result.metal_node_count == 1
    assert result.metal_nodes[0].atom_indices == (0,)
    assert result.linker_count == 1
    assert "B" in result.linkers[0].formula


# --- partition_framework: linker IDs and counts ---


def test_linker_ids_are_assigned_sequentially():
    symbols = ["Zn", "O", "C", "O", "O", "C", "O", "O", "C", "O"]
    bonds = [
        FakeBond(0, 1), FakeBond(1, 2), FakeBond(2, 3),
        FakeBond(0, 4), FakeBond(4, 5), FakeBond(5, 6),
        FakeBond(0, 7), FakeBond(7, 8), FakeBond(8, 9),
    ]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert [linker.linker_id for linker in result.linkers] == [1, 2, 3]


def test_atom_count_property():
    symbols = ["Zn", "O", "C", "O"]
    bonds = [FakeBond(0, 1), FakeBond(1, 2), FakeBond(2, 3)]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert result.linkers[0].atom_count == 3
    assert result.metal_nodes[0].atom_count == 1


def test_structure_with_no_metal_atoms_has_no_metal_nodes():
    symbols = ["C", "H", "H", "H", "H"]
    bonds = [FakeBond(0, 1), FakeBond(0, 2), FakeBond(0, 3), FakeBond(0, 4)]

    result = partition_framework(symbols, bonds, IS_METAL)

    assert result.metal_node_count == 0
    assert result.linker_count == 1


def test_structure_with_no_bonds_at_all():
    symbols = ["Zn", "C"]

    result = partition_framework(symbols, [], IS_METAL)

    assert result.metal_node_count == 1
    assert result.linker_count == 1


def test_multiple_isolated_metal_ions_are_separate_nodes():
    symbols = ["Zn", "Zn", "Zn"]

    result = partition_framework(symbols, [], IS_METAL)

    assert result.metal_node_count == 3
    assert all(node.coordination_number == 1 for node in result.metal_nodes)


# --- build_periodic_graph_for_mof_analysis ---

import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.mof_analysis import (  # noqa: E402
    build_periodic_graph_for_mof_analysis,
)


class FakePeriodicBond:
    def __init__(self, atom1, atom2, image):
        self.atom1 = atom1
        self.atom2 = atom2
        self.image = image


def _structure(symbols, frac_coords, lattice_lengths=(10.0, 10.0, 10.0)):
    lattice = Lattice.orthorhombic(*lattice_lengths)
    return Structure(lattice, symbols, frac_coords)


def test_infinite_metal_ligand_chain_terminates():
    """The exact pathological case this exists for: a chain that
    repeats forever through periodic images, with no discrete
    'molecule' boundary at all. The old RenderModelBuilder-based
    approach could not terminate on this and fell back to an
    arbitrarily truncated coordination shell (confirmed directly
    against a real MOF file before writing this fix - see the
    docstring on build_periodic_graph_for_mof_analysis).
    """
    structure = _structure(
        ["Zn", "N", "N"],
        [[0.0, 0.5, 0.5], [0.2, 0.5, 0.5], [0.8, 0.5, 0.5]],
        lattice_lengths=(5.0, 10.0, 10.0),
    )
    bonds = [
        FakePeriodicBond(0, 1, (0, 0, 0)),
        FakePeriodicBond(1, 2, (0, 0, 0)),
        FakePeriodicBond(2, 0, (1, 0, 0)),  # extends the chain forever
    ]

    graph = build_periodic_graph_for_mof_analysis(
        structure, bonds, IS_METAL, max_atoms=200
    )

    assert graph.truncated is False
    assert graph.symbols == ["Zn", "N", "N", "Zn"]
    assert graph.site_images == [
        (0, (0, 0, 0)), (1, (0, 0, 0)), (2, (0, 0, 0)), (0, (1, 0, 0)),
    ]
    assert len(graph.bonds) == 3


def test_a_linker_straddling_a_cell_boundary_still_completes_fully():
    """An organic chain that crosses the periodic boundary (common for
    a linker positioned across a cell face) must still complete to its
    full extent - the metal-boundary rule only stops expansion *from*
    metal atoms, never from organic ones. What actually matters here -
    and what the original bug report was about - is that no carbon
    ends up isolated from its bonding partner; seeding every site
    independently can legitimately produce more than one overlapping
    periodic copy of the same two-carbon unit, which is not itself a
    problem as long as each copy is a complete, correctly bonded pair.
    """
    structure = _structure(
        ["Zn", "C", "C", "Zn"],
        [[0.05, 0.5, 0.5], [0.15, 0.5, 0.5], [0.95, 0.5, 0.5], [0.85, 0.5, 0.5]],
        lattice_lengths=(20.0, 10.0, 10.0),
    )
    bonds = [
        FakePeriodicBond(0, 1, (0, 0, 0)),   # Zn0 - C1, same cell
        FakePeriodicBond(1, 2, (-1, 0, 0)),  # C1 - C2's image one cell back
        FakePeriodicBond(2, 3, (0, 0, 0)),   # C2 - Zn3, same cell as C2 lives
    ]

    graph = build_periodic_graph_for_mof_analysis(
        structure, bonds, IS_METAL, max_atoms=200
    )

    assert graph.truncated is False

    result = partition_framework(graph.symbols, graph.bonds, IS_METAL)

    # The actual bug this guards against: no linker should be a single
    # isolated atom torn away from its bonding partner.
    assert all(linker.atom_count != 1 for linker in result.linkers)
    assert all(linker.formula == "C2" for linker in result.linkers)


def test_direct_metal_metal_bond_is_recorded_but_not_expanded_further():
    structure = _structure(["Cu", "Cu"], [[0.4, 0.5, 0.5], [0.6, 0.5, 0.5]])
    bonds = [FakePeriodicBond(0, 1, (0, 0, 0))]

    graph = build_periodic_graph_for_mof_analysis(
        structure, bonds, IS_METAL, max_atoms=200
    )

    assert graph.truncated is False
    assert graph.symbols == ["Cu", "Cu"]
    assert len(graph.bonds) == 1


def test_non_periodic_case_matches_plain_expectations():
    """A simple, non-repeating case behaves exactly as a flat bond
    graph would - the periodicity handling shouldn't change anything
    when there's no periodicity being exercised.
    """
    structure = _structure(
        ["Zn", "O", "C", "O"],
        [[0.5, 0.5, 0.5], [0.55, 0.5, 0.5], [0.6, 0.5, 0.5], [0.65, 0.5, 0.5]],
    )
    bonds = [
        FakePeriodicBond(0, 1, (0, 0, 0)),
        FakePeriodicBond(1, 2, (0, 0, 0)),
        FakePeriodicBond(2, 3, (0, 0, 0)),
    ]

    graph = build_periodic_graph_for_mof_analysis(
        structure, bonds, IS_METAL, max_atoms=200
    )

    assert graph.truncated is False
    assert len(graph.symbols) == 4
    assert len(graph.bonds) == 3


def test_truncated_flag_is_set_when_max_atoms_is_exceeded():
    structure = _structure(
        ["Zn", "N", "N"],
        [[0.0, 0.5, 0.5], [0.2, 0.5, 0.5], [0.8, 0.5, 0.5]],
        lattice_lengths=(5.0, 10.0, 10.0),
    )
    bonds = [
        FakePeriodicBond(0, 1, (0, 0, 0)),
        FakePeriodicBond(1, 2, (0, 0, 0)),
        FakePeriodicBond(2, 0, (1, 0, 0)),
    ]

    graph = build_periodic_graph_for_mof_analysis(
        structure, bonds, IS_METAL, max_atoms=2
    )

    assert graph.truncated is True


def test_graph_bonds_are_usable_directly_by_partition_framework():
    """PeriodicGraph.bonds must satisfy the same .atom_a/.atom_b shape
    partition_framework already expects - the whole point of keeping
    these as two separate pieces is that the existing, already-tested
    partition_framework doesn't need to change at all.
    """
    structure = _structure(
        ["Zn", "O", "C", "O", "Zn"],
        [
            [0.5, 0.5, 0.5], [0.55, 0.5, 0.5], [0.6, 0.5, 0.5],
            [0.65, 0.5, 0.5], [0.7, 0.5, 0.5],
        ],
    )
    bonds = [
        FakePeriodicBond(0, 1, (0, 0, 0)),
        FakePeriodicBond(1, 2, (0, 0, 0)),
        FakePeriodicBond(2, 3, (0, 0, 0)),
        FakePeriodicBond(3, 4, (0, 0, 0)),
    ]

    graph = build_periodic_graph_for_mof_analysis(
        structure, bonds, IS_METAL, max_atoms=200
    )

    result = partition_framework(graph.symbols, graph.bonds, IS_METAL)

    assert result.metal_node_count == 2
    assert result.linker_count == 1
    assert result.linkers[0].formula == "CO2"
