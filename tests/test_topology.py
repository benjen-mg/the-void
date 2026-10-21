"""Round 53: from a structure to its underlying net
(services/topology/deconstruct.py and analyze_topology).

Each structure here is built so that its net is known in advance:
linear linkers on a cubic lattice (pcu), two such nets interleaved
(2-fold pcu), a paddlewheel square grid (sql - the case the old
metal/non-metal split got wrong), a metal-oxide rod (which must be
reported, not forced into a net) and cristobalite (dia, as a T-atom
net). Where the geometry would make automatic bond
detection awkward, bonds are given explicitly: they are an input of the
deconstruction, not something it infers.
"""


import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.metal_classification import is_metal  # noqa: E402
from thevoid_core.services.bonds.bond_model import Bond  # noqa: E402
from thevoid_core.services.mof_analysis import (  # noqa: E402
    build_periodic_graph_for_mof_analysis,
    partition_framework,
)
from thevoid_core.services.topology import (  # noqa: E402
    analyze_topology,
)
from thevoid_core.services.topology.library import (  # noqa: E402
    ARCHIVE_ENVIRONMENT_VARIABLE,
    default_library,
)




def _linear_linker_pcu(offset=None, a=6.0):
    """Zn on a cubic lattice, joined along each axis by a C3 chain."""
    species, coords = [], []
    shifts = [np.zeros(3)] if offset is None else [np.zeros(3), np.array(offset)]
    for shift in shifts:
        species.append("Zn")
        coords.append(shift)
        for axis in range(3):
            for k in (1, 2, 3):
                point = shift.copy()
                point[axis] += 1.5 * k
                species.append("C")
                coords.append(point)
    frac = [np.mod(np.array(c) / a, 1.0) for c in coords]
    return Structure(Lattice.cubic(a), species, frac)


def _bonds(pairs):
    return [Bond(i, j, tuple(image), 1.5) for i, j, image in pairs]


def _scatter(n, a=10.0):
    """Distinct placeholder coordinates for a structure whose bonds are
    given explicitly."""
    rng = np.random.default_rng(0)
    return Structure(Lattice.cubic(a), ["H"] * n, rng.random((n, 3)))


# ----------------------------------------------------------------------
# MOF-like nets
# ----------------------------------------------------------------------


def test_linear_linkers_on_a_cubic_lattice_give_pcu():
    result = analyze_topology(_linear_linker_pcu())
    assert result.status == "identified"
    assert result.rcsr == "pcu"
    assert result.summary == "pcu 6-c"
    assert result.vertex_kinds == [
        {"kind": "sbu", "formula": "Zn", "connectivity": 6, "count": 1}
    ]
    assert result.edge_formulas == {"C3": 3}


def test_two_interleaved_nets_are_reported_as_interpenetrated():
    result = analyze_topology(_linear_linker_pcu(offset=(3.0, 3.0, 3.0)))
    assert result.rcsr == "pcu"
    assert result.interpenetration == 2
    assert result.summary == "pcu 6-c, 2-fold interpenetrated"


def _paddlewheel_grid(cu_separation=2.6):
    """A Cu2(O2C)4 paddlewheel square grid, bonds given explicitly.

    Sites: 0,1 Cu; for each of 4 carboxylates k: C, Oa (on Cu0),
    Ob (on Cu1); then linker atoms Lx and Ly joining carboxylates of
    neighbouring cells. The two Cu are NOT bonded to each other and no
    atom bridges them: they are held together only through O-C-O.
    """
    pairs = []
    index = 2
    carboxylate_c = []
    for k in range(4):
        c, oa, ob = index, index + 1, index + 2
        index += 3
        carboxylate_c.append(c)
        pairs += [(c, oa, (0, 0, 0)), (c, ob, (0, 0, 0)),
                  (0, oa, (0, 0, 0)), (1, ob, (0, 0, 0))]
    lx, ly = index, index + 1
    index += 2
    # +x carboxylate -> Lx -> -x carboxylate of the next cell, same for y
    pairs += [(carboxylate_c[0], lx, (0, 0, 0)), (lx, carboxylate_c[1], (1, 0, 0)),
              (carboxylate_c[2], ly, (0, 0, 0)), (ly, carboxylate_c[3], (0, 1, 0))]
    symbols = ["Cu", "Cu"] + ["C", "O", "O"] * 4 + ["C", "C"]
    structure = _scatter(index)
    for i, symbol in enumerate(symbols):
        structure.replace(i, symbol, structure[i].frac_coords)
    # Only the Cu-Cu distance matters to the cluster rule; the rest of
    # the geometry is placeholder (bonds are given explicitly).
    structure.replace(0, "Cu", [0.5, 0.5, 0.5])
    structure.replace(1, "Cu", [0.5, 0.5, 0.5 + cu_separation / 10.0])
    return structure, _bonds(pairs)


def test_a_paddlewheel_is_one_node():
    """The two Cu of a paddlewheel share no atom and no bond, so the old
    metal/non-metal partition splits them into two nodes (its docstring
    says so). Counting metal-bound carboxylate groups as part of the SBU
    makes it one 4-c node, and the grid comes out as sql."""
    structure, bonds = _paddlewheel_grid()

    result = analyze_topology(structure, bonds)
    assert result.rcsr == "sql"
    assert result.vertex_kinds == [
        {"kind": "sbu", "formula": "C4Cu2O8", "connectivity": 4, "count": 1}
    ]

    # Round 55: the MOF Analysis grouping follows the same rule, given
    # atom positions; without them it is the old split.
    graph = build_periodic_graph_for_mof_analysis(structure, bonds, is_metal)
    assert partition_framework(graph.symbols, graph.bonds, is_metal).metal_node_count >= 2
    grouped = partition_framework(graph.symbols, graph.bonds, is_metal, positions=graph.positions)
    assert "Cu2" in {node.formula for node in grouped.metal_nodes}


def test_a_carboxylate_joining_distant_metals_is_a_linker_not_a_cluster():
    """The same bonding with the two Cu 5.5 A apart - how far an oxalate
    holds its two metals - is two separate metal nodes joined by
    carboxylate linkers, not one cluster (round 55)."""
    structure, bonds = _paddlewheel_grid(cu_separation=5.5)
    result = analyze_topology(structure, bonds)
    kinds = {(k["kind"], k["formula"]) for k in result.vertex_kinds}
    assert ("sbu", "Cu") in kinds
    assert ("sbu", "C4Cu2O8") not in kinds


def test_an_infinite_sbu_is_split_at_its_metals():
    """Zn atoms joined through mu-O along a (a rod, as in MOF-74 or
    MIL-53), linked by C3 chains along b and c. Before round 70 this was
    reported as a rod SBU with no net."""
    pairs = [(0, 1, (0, 0, 0)), (1, 0, (1, 0, 0))]  # Zn-O-Zn' along a
    index = 2
    for axis_image in ((0, 1, 0), (0, 0, 1)):
        c1, c2 = index, index + 1
        index += 2
        pairs += [(0, c1, (0, 0, 0)), (c1, c2, (0, 0, 0)), (c2, 0, axis_image)]
    structure = _scatter(index)
    for i, symbol in enumerate(["Zn", "O", "C", "C", "C", "C"]):
        structure.replace(i, symbol, structure[i].frac_coords)

    # Round 70: the rod is split at its metals, as CrystalNets does. Each
    # Zn is a vertex, joined to the next along a through the O they
    # share, and to its images along b and c by the C chains: pcu.
    result = analyze_topology(structure, _bonds(pairs))
    assert result.status == "identified" and result.rcsr == "pcu"
    assert result.rod_sbus == [{"formula": "OZn", "dimensionality": 1, "metals": 1, "count": 1}]
    assert any("split at its metals" in note for note in result.notes)
    assert result.all_node["rcsr"] == "pcu"  # no carboxylates to split off


def test_terminal_ligands_are_dropped():
    """A Cl bonded to one Zn only is a terminal ligand, not part of the
    net."""
    structure = _linear_linker_pcu()
    structure.append("Cl", [0.9, 0.9, 0.9])
    bonds = _bonds_from_detector(structure)
    bonds = [b for b in bonds if structure[b.atom1].specie.symbol != "Cl"
             and structure[b.atom2].specie.symbol != "Cl"]
    bonds.append(Bond(0, len(structure) - 1, (0, 0, 0), 2.2))

    result = analyze_topology(structure, bonds)
    assert result.rcsr == "pcu"
    assert result.dropped_formulas == {"Cl": 1}


def _bonds_from_detector(structure):
    from thevoid_core.services.bonds.covalent_bond_detector import (
        CovalentBondDetector,
    )

    return CovalentBondDetector().detect(structure)


# ----------------------------------------------------------------------
# Zeolites and metal-free frameworks
# ----------------------------------------------------------------------


def _cristobalite(a=7.16):
    fcc = [np.array(p) for p in ((0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0))]
    bonds_out = [np.array(v) for v in ((.25, .25, .25), (.25, -.25, -.25),
                                       (-.25, .25, -.25), (-.25, -.25, .25))]
    silicon = fcc + [p + (.25, .25, .25) for p in fcc]
    oxygen = {tuple(np.round(np.mod(p + v / 2, 1), 6)) for p in fcc for v in bonds_out}
    return Structure(Lattice.cubic(a), ["Si"] * 8 + ["O"] * len(oxygen),
                     [np.mod(p, 1) for p in silicon] + [list(o) for o in oxygen])


def test_silica_is_reduced_to_its_t_atom_net():
    """Idealised beta-cristobalite: SiO4 tetrahedra sharing corners on a
    diamond lattice. The T-atom net is dia."""
    result = analyze_topology(_cristobalite())
    assert result.representation == "T-atom net"
    assert result.rcsr == "dia"
    assert result.edge_formulas == {"O": 16}


def test_a_metal_free_organic_framework_is_not_guessed_at():
    graphite_like = Structure(
        Lattice.hexagonal(2.46, 6.7), ["C", "C"], [[0, 0, 0], [1 / 3, 2 / 3, 0]]
    )
    result = analyze_topology(graphite_like)
    assert result.status == "no_metal"
    assert "COF" in result.message


# ----------------------------------------------------------------------
# The real fixture, records and failure paths
# ----------------------------------------------------------------------


def test_analysis_never_raises_for_a_structure_it_cannot_handle():
    even_split = Structure(Lattice.cubic(5.0), [{"Na": 0.5, "K": 0.5}], [[0, 0, 0]])
    result = analyze_topology(even_split)
    assert result.status == "failed"
    assert result.message


def test_a_full_archive_can_replace_the_bundled_library(tmp_path, monkeypatch):
    archive = tmp_path / "mini.arc"
    archive.write_text(
        "key      3 1 1 -1 0 0 1 1 0 -1 0 1 1 0 0 -1\nid       pcu\nend\n"
    )
    library = default_library(archive)
    assert library.names() == ["pcu"]
    assert "mini.arc" in library.description
    assert library.nets[0].description.startswith("primitive cubic")

    monkeypatch.setenv(ARCHIVE_ENVIRONMENT_VARIABLE, str(archive))
    assert default_library().names() == ["pcu"]

    missing = default_library(tmp_path / "nope.arc")
    assert len(missing) > 50  # fell back to the bundled set


def test_the_bundled_library_covers_the_common_nets():
    names = set(default_library().names())
    for name in ("pcu", "dia", "sql", "hcb", "srs", "tbo", "fcu", "sod",
                 "nbo", "pts", "acs", "qom", "lta", "rho", "kgm", "hxl"):
        assert name in names
