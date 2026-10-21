"""From a bonded crystal structure to its underlying net (round 53).

Single-node deconstruction, the convention behind most RCSR codes
quoted for MOFs (HKUST-1 tbo, MOF-5 pcu, UiO-66 fcu, ZIF-8 sod, MOF-505
nbo):

1. Nodes and linkers. The "node side" of the bond graph is every metal
   atom, every non-metal atom bonded to two or more metal atoms (mu-O,
   mu-OH: the rule mof_analysis.identify_bridging_atoms already uses),
   and every carboxylate group that holds a metal cluster together
   (its oxygens bound to one metal, or to metals within 4.5 A of each
   other; round 55). The carboxylate rule is what the plain
   metal/non-metal split was missing: a Cu2(O2C)4 paddlewheel has no atom bridging its two
   metals, so it used to come out as two single-metal nodes (see
   mof_analysis.partition_framework's docstring). With the O-C-O groups
   counted as node, it is one cluster, as it is chemically. Everything
   else is linker side. Connected pieces of each side, found with
   periodic offsets, are the SBUs and the linkers.
2. The component graph. Each bond between an SBU atom and a linker
   atom joins those two components, with the exact lattice translation
   between them - the same image bookkeeping pore analysis uses for
   percolation, so a linker bonded to two images of one SBU counts
   twice, as it should.
3. Simplification. Vertices with fewer than two connections (terminal
   ligands, capping groups) are removed; two-connected vertices (a
   ditopic linker, a two-connected metal) become edges; repeated edges
   between the same two vertices merge into one. What remains - SBUs
   and linkers with three or more connections - is the net.

Rod SBUs (round 70). An SBU that is itself infinite (the rods of
MIL-53 and MOF-74, inorganic layers) has no single point to place a
vertex at. It is split the way CrystalNets splits it, which is the
convention behind CoRE MOF 2024's `topology` column:
- each metal atom of the rod is a vertex;
- two metals are joined along the rod when they share an atom (a mu-O,
  mu-OH or mu-F, or one oxygen of a carboxylate bridging both);
- inorganic bridges (atoms bonded only to metals and H) are used up by
  those joins; everything organic in the rod, such as carboxylate
  groups, goes back to its linker.
A carboxylate that only spans two metals syn-syn (O-C-O) does not join
them. MIL-53 and MIL-47 come out bpq, as in CoRE. The "all-node" net
(CoRE's `topology_all_node`) additionally makes every carboxylate group
of a linker its own vertex: MIL-53 is then rna.

What this does not do, and says so instead of guessing:
- Metal-free frameworks are only reduced when they are zeolite-like
  (tetrahedral atoms joined through oxygen), to their T-atom net. A COF
  or other organic framework is reported as not deconstructed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from thevoid_core.services.mof_analysis import hill_formula
from thevoid_core.services.topology.net import PeriodicNet

#: Tetrahedral framework atoms of zeolites and zeotypes.
T_ATOMS = frozenset({"Si", "Al", "P", "Ge", "Ga", "B", "Be", "Zn"})
#: What else a zeolite-like framework may contain.
ZEOLITE_OTHER = frozenset({"O", "H", "F"})


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _neg(a):
    return (-a[0], -a[1], -a[2])


ZERO = (0, 0, 0)


@dataclass
class Component:
    """One SBU or linker: atoms (site indices) with their offset from the
    component's own reference cell, and its periodic dimensionality."""

    kind: str  # "sbu" or "linker" or "atom"
    atoms: list
    offsets: dict
    formula: str
    dimensionality: int


@dataclass
class VertexKind:
    kind: str
    formula: str
    connectivity: int
    count: int


@dataclass
class Deconstruction:
    #: "ok", "rod_sbu", "periodic_linker", "no_metal", "no_net"
    status: str
    net: PeriodicNet | None = None
    #: For each net vertex: (kind, formula).
    vertex_labels: list = field(default_factory=list)
    #: Components that became edges or were dropped, by formula.
    edge_formulas: dict = field(default_factory=dict)
    dropped_formulas: dict = field(default_factory=dict)
    merged_edges: int = 0
    message: str = ""
    representation: str = "single-node"
    #: The offending SBU when status == "rod_sbu".
    rod: dict | None = None
    #: Round 70: the infinite SBUs that were split at their metals,
    #: [{"formula", "dimensionality", "metals", "count"}]: formula and
    #: metals per repeat, count of such SBUs in the cell.
    rods: list = field(default_factory=list)

    def vertex_kinds(self, degrees) -> list[VertexKind]:
        counts: dict[tuple, int] = defaultdict(int)
        for (kind, formula), degree in zip(self.vertex_labels, degrees):
            counts[(kind, formula, degree)] += 1
        return [
            VertexKind(kind, formula, degree, count)
            for (kind, formula, degree), count in sorted(counts.items())
        ]


def _symbol(site) -> str:
    species = list(site.species.keys())
    return species[0].symbol if species else "X"


def _adjacency(n, bonds):
    adjacency = [[] for _ in range(n)]
    for bond in bonds:
        image = tuple(int(v) for v in bond.image)
        adjacency[bond.atom1].append((bond.atom2, image))
        adjacency[bond.atom2].append((bond.atom1, _neg(image)))
    return adjacency


def _components(atoms, adjacency, allowed, kind, symbols):
    """Connected components of the subgraph on `allowed`, each with
    per-atom offsets from a spanning tree and its dimensionality."""
    from thevoid_core.services.pore_analysis import dimensionality_from_vectors

    seen = set()
    result = []
    owner = {}

    for start in atoms:
        if start in seen:
            continue
        offsets = {start: ZERO}
        seen.add(start)
        stack = [start]
        cycles = []

        while stack:
            u = stack.pop()
            for w, image in adjacency[u]:
                if w not in allowed:
                    continue
                expected = _add(offsets[u], image)
                if w not in offsets:
                    offsets[w] = expected
                    seen.add(w)
                    stack.append(w)
                elif expected != offsets[w]:
                    cycles.append(_sub(expected, offsets[w]))

        members = sorted(offsets)
        counts = defaultdict(int)
        for index in members:
            counts[symbols[index]] += 1

        component = Component(
            kind=kind,
            atoms=members,
            offsets=offsets,
            formula=hill_formula(counts),
            dimensionality=dimensionality_from_vectors(cycles),
        )
        for index in members:
            owner[index] = len(result)
        result.append(component)

    return result, owner


def node_side_atoms(symbols, adjacency, is_metal, position=None) -> set[int]:
    """Metals, atoms bridging two or more metal atoms, and carboxylate
    groups that hold a metal cluster together (mof_analysis.
    carboxylate_groups / bridges_one_cluster): chelating one metal, or
    bridging metals within CLUSTER_BRIDGE_DISTANCE of each other, as in
    a paddlewheel, Zn4O or Zr6. A carboxylate bridging metals farther
    apart - both ends of an oxalate - stays a linker (round 55; before,
    CALF-20's oxalate was swallowed into a Zn2(C2O4) "cluster" of two
    zinc atoms 5.5 A apart)."""
    from thevoid_core.services.mof_analysis import (
        bridges_one_cluster,
        carboxylate_groups,
    )

    metals = {i for i, s in enumerate(symbols) if is_metal(s)}
    node = set(metals)

    for i, neighbours in enumerate(adjacency):
        if i in metals:
            continue
        metal_neighbours = {(w, image) for w, image in neighbours if w in metals}
        if len(metal_neighbours) >= 2:
            node.add(i)

    for carbon, oxygens, ends in carboxylate_groups(symbols, adjacency, is_metal, position):
        if position is None or bridges_one_cluster(ends, position):
            node.add(carbon)
            node |= set(oxygens)

    return node


def is_zeolite_like(symbols, adjacency) -> bool:
    elements = set(symbols)
    if not elements & T_ATOMS or not elements <= (T_ATOMS | ZEOLITE_OTHER):
        return False
    for i, s in enumerate(symbols):
        if s in T_ATOMS:
            oxygens = [w for w, _ in adjacency[i] if symbols[w] == "O"]
            if len(oxygens) != 4:
                return False
    return True


#: Representations deconstruct() can build.
SINGLE_NODE = "single-node"
ALL_NODE = "all-node"
SHAPES = {1: "rod (1-periodic)", 2: "layer (2-periodic)", 3: "3-periodic inorganic framework"}


def carboxylate_groups_of(symbols, adjacency, atoms) -> list[set]:
    """Carboxylate groups (C with exactly two O and one other heavy
    neighbour) lying wholly within atoms."""
    groups = []
    for c in atoms:
        if symbols[c] != "C":
            continue
        oxygens = [w for w, _ in adjacency[c] if symbols[w] == "O"]
        heavy = [w for w, _ in adjacency[c] if symbols[w] != "H"]
        if len(oxygens) == 2 and len(heavy) == 3 and all(o in atoms for o in oxygens):
            groups.append({c, *oxygens})
    return groups


def _split_rods(rods, symbols, adjacency, is_metal, node_atoms):
    """Split infinite SBUs at their metals (module docstring). Returns
    (node atoms that stay node side, one Component per rod metal, the
    rod's metal-metal joins as (metal, metal, translation), rod info)."""
    rod_atoms = set().union(*(set(r.atoms) for r in rods))
    metals = {a for a in rod_atoms if is_metal(symbols[a])}
    rest = rod_atoms - metals

    # Inorganic bridges: pieces of the rod's non-metal atoms bonded only
    # to metals and hydrogen. They are used up by the joins below; the
    # rest (carboxylates and other organic parts) goes to the linkers.
    pieces, _ = _components(sorted(rest), adjacency, rest, "piece", symbols)
    inorganic = set()
    for piece in pieces:
        atoms = set(piece.atoms)
        if not any(symbols[a] == "C" for a in atoms) and not any(
            w not in atoms and symbols[w] != "H" and not is_metal(symbols[w])
            for a in atoms for w, _ in adjacency[a]
        ):
            inorganic |= atoms
            inorganic |= {w for a in atoms for w, _ in adjacency[a] if symbols[w] == "H"}

    # Two metals bonded to one atom are joined. Each metal is its own
    # component with its stored position as reference, so the join's
    # translation is the difference of the two bonds' images.
    joins = set()
    for atom in rest:
        bound = sorted({(w, image) for w, image in adjacency[atom] if w in metals})
        for i in range(len(bound)):
            for j in range(i + 1, len(bound)):
                (u, tu), (v, tv) = bound[i], bound[j]
                if u == v and tu == tv:
                    continue
                joins.add((u, v, _sub(tv, tu)))

    counts = defaultdict(int)
    for rod in rods:
        counts[(rod.formula, rod.dimensionality,
                sum(1 for a in rod.atoms if is_metal(symbols[a])))] += 1
    info = [{"formula": f, "dimensionality": d, "metals": m, "count": c}
            for (f, d, m), c in sorted(counts.items())]
    singles = [Component("sbu", [m], {m: ZERO}, symbols[m], 0) for m in sorted(metals)]
    return (node_atoms - rod_atoms) | inorganic, singles, joins, info, metals


def deconstruct(structure, bonds, is_metal, representation=SINGLE_NODE) -> Deconstruction:
    """The underlying net of a framework (see the module docstring).
    structure should already have free molecules and minor disorder
    removed (raspa_export.prepare_framework does both).
    representation: SINGLE_NODE, or ALL_NODE, which also makes every
    carboxylate group of a linker its own vertex (used for rod
    frameworks, where it is CoRE's topology_all_node)."""
    n = len(structure.sites)
    symbols = [_symbol(site) for site in structure.sites]
    adjacency = _adjacency(n, bonds)

    # Zeolites first: Al and Ga are metals to pymatgen, but in an
    # aluminosilicate or AlPO they are T atoms like Si and P.
    if is_zeolite_like(symbols, adjacency):
        return _atom_net(n, symbols, adjacency)

    if not any(is_metal(s) for s in symbols):
        t_atoms = [i for i, s in enumerate(symbols) if s in T_ATOMS]
        if t_atoms and set(symbols) <= (T_ATOMS | ZEOLITE_OTHER):
            found = sorted({
                sum(symbols[w] == "O" for w, _ in adjacency[i]) for i in t_atoms
            })
            return Deconstruction(
                status="no_metal",
                message="A silicate/zeotype, but its T atoms are not all "
                "bonded to four oxygens (found "
                + ", ".join(f"{k}-coordinated" for k in found)
                + "); the T-atom net needs TO4 tetrahedra. Check the file "
                "for overlapping or missing atoms.",
            )
        return Deconstruction(
            status="no_metal",
            message="No metal atoms, and not a zeolite-like framework "
            "(T atoms joined through O). Metal-free frameworks such as "
            "COFs are not deconstructed in this version.",
        )

    from thevoid_core.services.mof_analysis import structure_position

    node_atoms = node_side_atoms(
        symbols, adjacency, is_metal, structure_position(structure)
    )
    linker_atoms = set(range(n)) - node_atoms

    sbus, sbu_owner = _components(sorted(node_atoms), adjacency, node_atoms, "sbu", symbols)
    rods = [sbu for sbu in sbus if sbu.dimensionality > 0]
    joins, rod_info, rod_metals = set(), [], set()

    if rods:
        node_atoms, singles, joins, rod_info, rod_metals = _split_rods(
            rods, symbols, adjacency, is_metal, node_atoms)
        others = node_atoms - rod_metals
        rest, _ = _components(sorted(others), adjacency, others, "sbu", symbols)
        node_atoms = node_atoms | rod_metals
        sbus = singles + rest
        sbu_owner = {a: k for k, c in enumerate(sbus) for a in c.atoms}
        for sbu in rest:
            if sbu.dimensionality > 0:  # an inorganic bridge that is itself periodic
                shape = SHAPES[sbu.dimensionality]
                return Deconstruction(
                    status="rod_sbu",
                    rod={"formula": sbu.formula, "dimensionality": sbu.dimensionality},
                    rods=rod_info,
                    message=f"After splitting the infinite SBU at its metals, a "
                    f"metal-free inorganic {shape} is left ({sbu.formula} per "
                    "repeat), so the net is not identified.",
                )

    linker_atoms = set(range(n)) - node_atoms
    if representation == ALL_NODE:
        groups = carboxylate_groups_of(symbols, adjacency, linker_atoms)
        grouped = set().union(*groups) if groups else set()
        remainder = linker_atoms - grouped
        first, _ = _components(sorted(grouped), adjacency, grouped, "linker", symbols)
        second, _ = _components(sorted(remainder), adjacency, remainder, "linker", symbols)
        linkers = first + second
        linker_owner = {a: k for k, c in enumerate(linkers) for a in c.atoms}
    else:
        linkers, linker_owner = _components(sorted(linker_atoms), adjacency, linker_atoms,
                                            "linker", symbols)

    for linker in linkers:
        if linker.dimensionality > 0:
            return Deconstruction(
                status="periodic_linker",
                message=f"An organic part ({linker.formula} per repeat) is "
                "itself periodic, so it cannot be a single node or edge; "
                "the net is not identified.",
            )

    components = sbus + linkers
    offset_sbu = 0
    offset_linker = len(sbus)

    def component_of(atom):
        if atom in sbu_owner:
            return offset_sbu + sbu_owner[atom]
        return offset_linker + linker_owner[atom]

    def offset_of(atom):
        return components[component_of(atom)].offsets[atom]

    edges = set()

    for a in range(n):
        for b, image in adjacency[a]:
            if (b, _neg(image)) < (a, image):
                continue  # each bond once
            x, y = component_of(a), component_of(b)
            if x == y or (a in node_atoms and b in node_atoms):
                # Within one piece; or between node-side pieces, which
                # only a split rod has, and whose links are its joins.
                continue
            translation = _sub(_add(offset_of(a), image), offset_of(b))
            edges.add((x, y, translation))

    for u, v, translation in joins:
        edges.add((component_of(u), component_of(v), translation))

    labels = [(c.kind, c.formula) for c in components]
    result = _simplify(len(components), edges, labels, components)
    result.rods = rod_info
    if representation == ALL_NODE:
        result.representation = ALL_NODE
    elif rod_info:
        result.representation = "single-node, infinite SBUs split at their metals"
    return result


def _atom_net(n, symbols, adjacency) -> Deconstruction:
    edges = set()
    for a in range(n):
        for b, image in adjacency[a]:
            if a < b or (a == b and image > _neg(image)):
                edges.add((a, b, image))
    labels = [("T atom" if s in T_ATOMS else "atom", s) for s in symbols]
    result = _simplify(n, edges, labels, None)
    result.representation = "T-atom net"
    return result


def _simplify(n_vertices, edges, labels, components) -> Deconstruction:
    """Remove <2-connected vertices, turn 2-connected ones into edges,
    merge parallel edges; repeat until nothing changes."""
    # incident[v] = list of (edge id); edges stored as dict id -> (u, w, t)
    store = {}
    incident = defaultdict(set)
    next_id = 0

    def add_edge(u, w, t):
        nonlocal next_id
        store[next_id] = (u, w, tuple(t))
        incident[u].add(next_id)
        incident[w].add(next_id)
        next_id += 1

    def remove_edge(e):
        u, w, _ = store.pop(e)
        incident[u].discard(e)
        incident[w].discard(e)

    def degree(v):
        # a loop counts twice
        return sum(2 if store[e][0] == store[e][1] else 1 for e in incident[v])

    for u, w, t in edges:
        add_edge(u, w, t)

    alive = set(range(n_vertices))
    edge_formulas: dict[str, int] = defaultdict(int)
    dropped: dict[str, int] = defaultdict(int)
    merged = 0
    changed = True

    while changed:
        changed = False

        for v in sorted(alive):
            d = degree(v)
            if d <= 1:
                for e in list(incident[v]):
                    remove_edge(e)
                alive.discard(v)
                dropped[labels[v][1]] += 1
                changed = True
            elif d == 2:
                ids = list(incident[v])
                if len(ids) == 1:  # only a loop: a chain of itself
                    continue
                first, second = (store[e] for e in ids)
                ends = []
                for u, w, t in (first, second):
                    # re-express each edge as v -> other with shift
                    ends.append((w, t) if u == v else (u, _neg(t)))
                (a, ta), (b, tb) = ends
                for e in ids:
                    remove_edge(e)
                alive.discard(v)
                edge_formulas[labels[v][1]] += 1
                shift = _sub(tb, ta)
                if not (a == b and shift == ZERO):
                    add_edge(a, b, shift)
                changed = True

        # merge parallel edges
        seen = {}
        for e in list(store):
            u, w, t = store[e]
            key = (u, w, t) if (u, w, t) <= (w, u, _neg(t)) else (w, u, _neg(t))
            if key in seen:
                remove_edge(e)
                merged += 1
                changed = True
            else:
                seen[key] = e

    vertices = sorted(alive)

    if not vertices:
        return Deconstruction(
            status="no_net",
            edge_formulas=dict(edge_formulas),
            dropped_formulas=dict(dropped),
            message="Nothing is left after removing terminal groups and "
            "turning two-connected pieces into edges: there is no "
            "three-dimensional or layered net (a chain or a molecule).",
        )

    index = {v: k for k, v in enumerate(vertices)}
    net_edges = [(index[u], index[w], t) for u, w, t in store.values()]
    net = PeriodicNet.build(len(vertices), net_edges, 3)

    return Deconstruction(
        status="ok",
        net=net,
        vertex_labels=[labels[v] for v in vertices],
        edge_formulas=dict(edge_formulas),
        dropped_formulas=dict(dropped),
        merged_edges=merged,
    )
