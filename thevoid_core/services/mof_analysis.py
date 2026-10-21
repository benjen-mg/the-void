from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Linker:
    """One organic linker fragment."""

    linker_id: int
    atom_indices: tuple[int, ...]
    formula: str
    #: Round 55: 0 for a finite linker, 1-3 if it is itself periodic.
    dimensionality: int = 0

    @property
    def atom_count(self) -> int:
        return len(self.atom_indices)


@dataclass(frozen=True, slots=True)
class _GraphEdge:
    """A bond in a PeriodicGraph - shaped to satisfy partition_framework's
    existing bonds contract (objects with .atom_a/.atom_b) without
    depending on RenderBond or any other module's bond class.
    """

    atom_a: int
    atom_b: int


@dataclass(frozen=True, slots=True)
class PeriodicGraph:
    """A bond graph built specifically for MOF partitioning - see
    build_periodic_graph_for_mof_analysis for why this needs its own
    completion strategy rather than reusing the application's display-
    oriented RenderModelBuilder.

    symbols and site_images are index-aligned: node i is
    site_images[i] == (site_index, image), an instance of that base-
    structure site at a particular periodic translation, and has
    element symbol symbols[i].
    """

    symbols: list[str]
    bonds: list[_GraphEdge]
    site_images: list[tuple[int, tuple[int, int, int]]]

    #: True if max_atoms was reached before every reachable atom could
    #: be included - should not normally happen (see the builder's
    #: docstring for why this approach terminates for a real MOF where
    #: the naive "complete every molecule" walk would not), but is
    #: reported rather than silently producing a partial graph with no
    #: indication anything was cut short.
    truncated: bool

    #: Round 55: Cartesian position of each node (site at its image),
    #: for partition_framework's carboxylate-cluster rule.
    positions: list | None = None


def build_periodic_graph_for_mof_analysis(
    structure,
    bonds,
    is_metal,
    max_atoms: int = 4000,
) -> PeriodicGraph:
    """Build the bond graph MOF partitioning actually needs, handling
    periodicity itself rather than reusing RenderModelBuilder.

    The problem this solves: RenderModelBuilder's "complete every
    molecule" walk is designed for ordinary molecular and coordination
    crystals, where a molecule that crosses a cell boundary has a real
    edge to complete to. A genuinely 3-periodic MOF has no such edge -
    every linker connects to a metal, which connects to another linker,
    which connects to another metal, forever, in every direction. Fed
    a real MOF, that walk cannot terminate on its own and falls back to
    a bounded coordination-shell instead (confirmed directly: loading a
    real Zn-triazolate-oxalate MOF produced RenderModel.fell_back=True,
    with linkers truncated wherever the shell boundary happened to cut
    through a ring - which is what was reported as "even N shows up as
    a linker": an isolated nitrogen was a triazolate ring sheared in
    half by that arbitrary boundary, not a real chemical fragment).

    The fix is a different, bounded walk: seed every site in the
    structure's reference cell, and expand outward through bonds as
    usual, *except* a metal atom is never used as a launching point to
    discover a periodic image of another atom that hasn't already been
    reached some other way. A metal atom's own bonds are still fully
    recorded (so its ligands and any direct metal-metal bonds are all
    present in the graph) - it just never pulls in *new* territory the
    way an organic atom does. Real organic linkers are finite between
    their metal-coordination points even in an infinite framework, so
    letting every non-metal atom expand freely still completes every
    ring and chain fully; stopping metal atoms from doing the same is
    exactly what makes the walk terminate instead of chasing the
    infinite lattice outward forever.

    max_atoms is a safety cap, not the mechanism the fix relies on - it
    should not normally be reached for a real MOF's asymmetric unit;
    see PeriodicGraph.truncated.
    """
    adjacency: dict[int, list[tuple[int, tuple[int, int, int]]]] = (
        defaultdict(list)
    )

    for bond in bonds:
        adjacency[bond.atom1].append((bond.atom2, bond.image))
        adjacency[bond.atom2].append(
            (bond.atom1, tuple(-v for v in bond.image))
        )

    site_symbols = [_site_symbol(site) for site in structure.sites]

    node_index: dict[tuple[int, tuple[int, int, int]], int] = {}
    site_images: list[tuple[int, tuple[int, int, int]]] = []
    symbols: list[str] = []

    def get_or_create(site_index: int, image: tuple[int, int, int]) -> int:
        key = (site_index, image)

        if key not in node_index:
            node_index[key] = len(site_images)
            site_images.append(key)
            symbols.append(site_symbols[site_index])

        return node_index[key]

    queue: list[tuple[int, tuple[int, int, int]]] = []

    for site_index in range(len(structure.sites)):
        get_or_create(site_index, (0, 0, 0))
        queue.append((site_index, (0, 0, 0)))

    edges_seen: set[tuple[int, int]] = set()
    edges: list[_GraphEdge] = []
    truncated = False
    head = 0

    while head < len(queue):
        if len(site_images) > max_atoms:
            truncated = True
            break

        site_index, image = queue[head]
        head += 1

        current_index = node_index[(site_index, image)]
        current_is_metal = is_metal(site_symbols[site_index])

        for neighbor_site, delta in adjacency[site_index]:
            neighbor_image = tuple(a + b for a, b in zip(image, delta))
            neighbor_key = (neighbor_site, neighbor_image)
            neighbor_is_metal = is_metal(site_symbols[neighbor_site])

            already_present = neighbor_key in node_index

            if not already_present:
                if current_is_metal and not neighbor_is_metal:
                    # The crux of the fix: a metal atom does not pull
                    # in a periodic image of a non-metal atom that
                    # hasn't already been reached through that atom's
                    # own proper expansion. Without this, the walk
                    # would cascade metal -> ligand -> its other metal
                    # -> the next ligand -> ... across the entire
                    # periodic framework, never terminating.
                    continue

                neighbor_index = get_or_create(neighbor_site, neighbor_image)

                if not neighbor_is_metal:
                    # Keep expanding through non-metal atoms so rings
                    # and chains complete fully. A metal neighbour is
                    # recorded (the edge below still gets added) but
                    # never itself enqueued for further expansion,
                    # which is what stops the walk at each metal
                    # centre rather than passing through it.
                    queue.append(neighbor_key)
            else:
                neighbor_index = node_index[neighbor_key]

            if current_index == neighbor_index:
                continue

            edge_key = (
                min(current_index, neighbor_index),
                max(current_index, neighbor_index),
            )

            if edge_key not in edges_seen:
                edges_seen.add(edge_key)
                edges.append(_GraphEdge(atom_a=edge_key[0], atom_b=edge_key[1]))

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    positions = [
        (np.asarray(structure.sites[site].frac_coords, dtype=float)
         + np.asarray(image, dtype=float)) @ lattice
        for site, image in site_images
    ]

    return PeriodicGraph(
        symbols=symbols,
        bonds=edges,
        site_images=site_images,
        truncated=truncated,
        positions=positions,
    )


def _site_symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "?"


@dataclass(frozen=True, slots=True)
class MetalNode:
    """One metal-containing cluster: a single isolated metal centre, or
    a multi-metal secondary building unit held together by a bridging
    atom or a direct metal-metal bond.
    """

    node_id: int
    atom_indices: tuple[int, ...]
    metal_symbols: tuple[str, ...]
    formula: str
    #: Round 55: 0 for a finite cluster, 1 for an infinite rod (MOF-74,
    #: MIL-53), 2 for a layer, 3 for a 3-periodic inorganic framework.
    dimensionality: int = 0

    @property
    def atom_count(self) -> int:
        return len(self.atom_indices)

    @property
    def coordination_number(self) -> int:
        """Number of metal atoms in this node - not the number of
        ligand atoms coordinating it, which is a different, per-metal-
        atom quantity that coordination environment analysis reports
        instead. For a single-metal node this is 1; for a multi-metal
        SBU (a paddlewheel, a Zr6 cluster) it is the metal count.
        """
        return len(self.metal_symbols)


@dataclass(frozen=True, slots=True)
class FrameworkPartition:
    linkers: list[Linker]
    metal_nodes: list[MetalNode]
    #: Round 55: sites left out of the partition as minor disorder
    #: components (partition_structure only; 0 otherwise).
    excluded_disorder_sites: int = 0

    @property
    def linker_count(self) -> int:
        return len(self.linkers)

    @property
    def metal_node_count(self) -> int:
        return len(self.metal_nodes)


def hill_formula(symbol_counts: dict[str, int]) -> str:
    """Hill notation: C first, H second, everything else alphabetical -
    matching the convention already used elsewhere for fragment
    formulas in this application.
    """
    if "C" in symbol_counts:
        ordered = ["C"]

        if "H" in symbol_counts:
            ordered.append("H")

        ordered += sorted(
            symbol for symbol in symbol_counts if symbol not in ("C", "H")
        )
    else:
        ordered = sorted(symbol_counts)

    return "".join(
        f"{symbol}{symbol_counts[symbol] if symbol_counts[symbol] > 1 else ''}"
        for symbol in ordered
    )


def connected_components(node_ids, edges) -> list[list[int]]:
    """Plain connected-components over an explicit node/edge set.

    Deliberately independent of any particular atom/bond class - this
    is reused for both the linker side and the metal-node side of a
    framework partition, which after cutting coordination bonds are
    two disjoint graphs sharing no edges with each other.
    """
    adjacency: dict[int, set[int]] = defaultdict(set)

    for a, b in edges:
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen: set[int] = set()
    components: list[list[int]] = []

    for start in node_ids:
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        component = []

        while stack:
            current = stack.pop()
            component.append(current)

            for neighbour in adjacency.get(current, ()):
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)

        components.append(component)

    return components


def identify_bridging_atoms(symbols, bonds, is_metal) -> set[int]:
    """Non-metal atoms bonded to two or more *different* metal atoms -
    a mu-oxo or mu-hydroxo bridge in a secondary building unit being
    the common case. Treated as part of the metal node rather than the
    organic linker, since chemically that is what they are: the shared
    atom holding a multi-metal cluster together, not a linker atom that
    happens to touch a metal.

    symbols is an indexable sequence of element symbols; bonds is an
    iterable of objects with .atom_a and .atom_b integer index
    attributes (matching RenderBond, but not requiring it - this
    accepts any object shaped that way, including a lightweight test
    double).
    """
    metal_neighbours: dict[int, set[int]] = defaultdict(set)

    for bond in bonds:
        a, b = bond.atom_a, bond.atom_b
        a_is_metal = is_metal(symbols[a])
        b_is_metal = is_metal(symbols[b])

        if a_is_metal and not b_is_metal:
            metal_neighbours[b].add(a)
        elif b_is_metal and not a_is_metal:
            metal_neighbours[a].add(b)

    return {
        atom_index
        for atom_index, metals in metal_neighbours.items()
        if len(metals) >= 2
    }


def partition_framework(symbols, bonds, is_metal, positions=None) -> FrameworkPartition:
    """Split a bonded structure into organic linkers and metal nodes.

    The framework's bond graph is partitioned into two atom sets - the
    "node side" (every metal atom, plus any bridging atom identified by
    identify_bridging_atoms) and everything else (the "linker side") -
    and every bond crossing between the two sets is cut. Connected
    components are then computed separately within each side, since a
    single periodic MOF framework is very often one giant connected
    component if metal-ligand coordination bonds are left in place;
    cutting them at the coordination boundary is what actually
    separates "this is a linker" from "this is a node" at all.

    This is a documented heuristic, not a full topological
    secondary-building-unit perception algorithm. It correctly groups:
      - a single isolated metal ion into its own one-metal node
      - metal centres bridged by a shared atom (mu-oxo, mu-hydroxo,
        and similar) into one multi-metal node
      - metal centres with a direct metal-metal bond into one node

      - metal centres bridged by a carboxylate and within
        CLUSTER_BRIDGE_DISTANCE of each other into one node - a
        Cu2(O2C)4 paddlewheel, whose two Cu share no atom and often no
        bond (round 55; needs `positions`, see below)

    Carboxylate atoms stay with their linker, so a linker's formula is
    the whole molecule (BDC is C8H4O4) - only the grouping of metals
    into nodes changes. A carboxylate bridging metals farther apart than
    CLUSTER_BRIDGE_DISTANCE (an oxalate between two single metal
    centres) does not join them. Topology identification uses the same
    rule but also moves the cluster's carboxylate atoms into the node
    (services/topology/deconstruct.py), as the single-node convention
    requires.

    symbols is an indexable sequence of element symbols, index-aligned
    with atom identity. bonds is an iterable of objects with .atom_a
    and .atom_b integer indices into symbols. positions, when given, is
    the Cartesian position of each atom (PeriodicGraph.positions); it is
    what the carboxylate rule measures metal-metal distances with.
    Without it the rule is skipped and a paddlewheel stays two nodes,
    as before round 55.
    """
    atom_count = len(symbols)

    bridging = identify_bridging_atoms(symbols, bonds, is_metal)
    node_side = {
        index for index in range(atom_count) if is_metal(symbols[index])
    }
    node_side |= bridging

    node_edges = []
    linker_edges = []

    for bond in bonds:
        a, b = bond.atom_a, bond.atom_b

        if a in node_side and b in node_side:
            node_edges.append((a, b))
        elif a not in node_side and b not in node_side:
            linker_edges.append((a, b))
        # else: a coordination bond crossing the node/linker boundary -
        # cut, appearing in neither edge list.

    linker_atom_ids = [
        index for index in range(atom_count) if index not in node_side
    ]
    node_atom_ids = sorted(node_side)

    linkers = []

    for linker_id, component in enumerate(
        connected_components(linker_atom_ids, linker_edges), start=1
    ):
        counts: dict[str, int] = defaultdict(int)

        for index in component:
            counts[symbols[index]] += 1

        linkers.append(
            Linker(
                linker_id=linker_id,
                atom_indices=tuple(sorted(component)),
                formula=hill_formula(counts),
            )
        )

    if positions is not None:
        adjacency = [[] for _ in range(atom_count)]
        zero = (0, 0, 0)
        for bond in bonds:
            adjacency[bond.atom_a].append((bond.atom_b, zero))
            adjacency[bond.atom_b].append((bond.atom_a, zero))

        def position(index, _cell):
            return np.asarray(positions[index], dtype=float)

        for _carbon, _oxygens, ends in carboxylate_groups(symbols, adjacency, is_metal, position):
            metals = sorted({metal for metal, _ in ends})
            if len(metals) >= 2 and bridges_one_cluster(ends, position):
                # A grouping edge only: the carboxylate's own atoms stay
                # with the linker.
                node_edges += [(metals[0], other) for other in metals[1:]]

    metal_nodes = []

    for node_id, component in enumerate(
        connected_components(node_atom_ids, node_edges), start=1
    ):
        counts = defaultdict(int)
        metals = []

        for index in component:
            symbol = symbols[index]
            counts[symbol] += 1

            if is_metal(symbol):
                metals.append(symbol)

        metal_nodes.append(
            MetalNode(
                node_id=node_id,
                atom_indices=tuple(sorted(component)),
                metal_symbols=tuple(sorted(metals)),
                formula=hill_formula(counts),
            )
        )

    return FrameworkPartition(linkers=linkers, metal_nodes=metal_nodes)


# ----------------------------------------------------------------------
# Periodic dimensionality of bonded components (round 52)
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BondedComponent:
    """One connected piece of a periodic bond graph, and how many
    independent lattice directions it extends along.

    dimensionality 0 is a finite molecule or ion (lattice solvent, a
    counter-ion, a discrete complex); 1 a chain, 2 a layer, 3 a
    framework. Interpenetrated frameworks come out as several
    3-periodic components.
    """

    site_indices: tuple[int, ...]
    formula: str
    dimensionality: int
    contains_metal: bool
    #: Sum of atomic numbers. Odd means the fragment, as written, is
    #: either charged or missing hydrogens (a closed-shell neutral
    #: molecule has an even electron count) - used only to flag
    #: probable counter-ions, never to decide anything.
    electron_count: int

    @property
    def atom_count(self) -> int:
        return len(self.site_indices)

    @property
    def is_periodic(self) -> bool:
        return self.dimensionality > 0

    @property
    def possibly_ionic(self) -> bool:
        """A heuristic, not a charge assignment: an isolated metal or
        single atom, or an odd electron count as written."""
        return (
            self.contains_metal
            or self.atom_count == 1
            or self.electron_count % 2 == 1
        )


def bonded_components(structure, bonds, is_metal) -> list[BondedComponent]:
    """Split a structure's sites into bonded components and measure the
    periodic dimensionality of each.

    bonds are the detector's Bond objects (atom1, atom2, image). Every
    bond is fed to the same offset-tracking union-find that pore
    analysis uses for percolation (pore_analysis._OffsetUnionFind): a
    bond that closes a cycle with a nonzero net lattice translation
    means its component repeats across the cell boundary, and the rank
    of all such translations is the component's dimensionality. A
    component with no such cycle is a finite molecule, however many
    cell boundaries its bonds happen to cross.

    Unlike build_periodic_graph_for_mof_analysis this never expands
    periodic images, so it is linear in the bond count and cannot be
    truncated.
    """
    from thevoid_core.services.pore_analysis import (
        _OffsetUnionFind,
        dimensionality_from_vectors,
    )

    n_sites = len(structure.sites)
    union_find = _OffsetUnionFind.for_n_nodes(n_sites)

    for bond in bonds:
        union_find.union(bond.atom1, bond.atom2, tuple(bond.image))

    members: dict[int, list[int]] = defaultdict(list)

    for index in range(n_sites):
        root, _ = union_find.find(index)
        members[root].append(index)

    symbols = [_site_symbol(site) for site in structure.sites]
    components = []

    for root, indices in members.items():
        counts: dict[str, int] = defaultdict(int)
        electrons = 0

        for index in indices:
            counts[symbols[index]] += 1
            electrons += _atomic_number(symbols[index])

        components.append(
            BondedComponent(
                site_indices=tuple(sorted(indices)),
                formula=hill_formula(counts),
                dimensionality=dimensionality_from_vectors(
                    union_find.percolation_vectors_for_root(root)
                ),
                contains_metal=any(is_metal(symbols[i]) for i in indices),
                electron_count=electrons,
            )
        )

    components.sort(key=lambda c: (-c.dimensionality, -c.atom_count, c.site_indices))

    return components


def _atomic_number(symbol: str) -> int:
    try:
        from pymatgen.core import Element

        return int(Element(symbol).Z)
    except Exception:
        return 0


# ----------------------------------------------------------------------
# Carboxylate-bridged metal clusters (round 55)
# ----------------------------------------------------------------------

#: Two metal atoms bridged by one carboxylate are one cluster when they
#: are at most this far apart (angstrom). Metal-metal distances inside
#: carboxylate-bridged clusters are 2.5-3.6 A (Cu2 and Zn2 paddlewheels,
#: Zn4O, Fe3O and Cr3O trimers, Zr6), and syn-syn bridges reach about
#: 4.5 A; a carboxylate joining metals farther apart - the two ends of an
#: oxalate, an anti-anti bridge - is a linker between two clusters, not
#: part of one. CALF-20's oxalate-bridged Zn pairs are 5.5 A apart.
CLUSTER_BRIDGE_DISTANCE = 4.5


def carboxylate_groups(symbols, adjacency, is_metal, position):
    """Every metal-bound carboxylate: (carbon, oxygens, metal ends).

    A carboxylate here is a C with two or more O neighbours that are
    each bonded to a metal. adjacency[i] lists (j, image) pairs (image
    the lattice translation of j relative to i, (0, 0, 0) for a graph of
    explicit site instances). The metal ends are (metal index, cell)
    pairs with cells relative to the carbon's own, and position(index,
    cell) gives a Cartesian position, so distances between the metals a
    carboxylate binds are exact across cell boundaries.
    """
    zero = (0, 0, 0)

    for carbon, neighbours in enumerate(adjacency):
        if symbols[carbon] != "C":
            continue
        oxygens = []
        ends = []
        for oxygen, o_image in neighbours:
            if symbols[oxygen] != "O":
                continue
            metals = [
                (metal, tuple(a + b for a, b in zip(o_image, m_image)))
                for metal, m_image in adjacency[oxygen]
                if is_metal(symbols[metal])
            ]
            if metals:
                oxygens.append(oxygen)
                ends.extend(metals)
        if len(set(oxygens)) >= 2:
            yield carbon, tuple(sorted(set(oxygens))), sorted(set(ends)) or [zero]


def bridges_one_cluster(ends, position, max_distance=CLUSTER_BRIDGE_DISTANCE) -> bool:
    """Whether a carboxylate's metal ends belong to one cluster: a single
    metal (chelating), or metals all within max_distance of each other."""
    points = [position(metal, cell) for metal, cell in ends]
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if float(np.linalg.norm(points[i] - points[j])) > max_distance:
                return False
    return True


def structure_position(structure):
    """position(index, cell) for a periodic structure."""
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    frac = np.array([site.frac_coords for site in structure.sites], dtype=float)

    def position(index, cell):
        return (frac[index] + np.asarray(cell, dtype=float)) @ lattice

    return position


# ----------------------------------------------------------------------
# Per-unit-cell partition (round 55)
# ----------------------------------------------------------------------


def partition_structure(structure, bonds, is_metal, min_occupancy=0.5):
    """Linkers and metal nodes of one unit cell, each exactly once.

    partition_framework works on build_periodic_graph_for_mof_analysis's
    expanded graph, which records the metal neighbours of every linker
    it completes but never expands a metal. Clusters at the edge of that
    patch come out partial: IRMOF-1's 8 Zn4O nodes were listed alongside
    48 lone "Zn" nodes (24 "Zn2" once paddlewheel grouping was added),
    all of them pieces of Zn4O clusters in neighbouring cells. Here the
    split is made on the periodic bond graph itself (bonds with their
    image translations), so every SBU and linker of the cell appears
    once, whole, and an infinite rod SBU is recognised as one (its
    dimensionality is 1) rather than as a string of fragments.

    Grouping is partition_framework's: metals, mu-bridging atoms, direct
    metal-metal bonds, and carboxylate bridges between metals within
    CLUSTER_BRIDGE_DISTANCE; carboxylate atoms stay with their linker.

    A ligand bonded to one node instance only (coordinated water, a
    hydroxide's H, a terminal halide) bridges nothing, so it is counted
    as part of that node - MIL-88B's node is Cr3O4 (Cr3O plus three
    terminal O), not Cr3O with three one-atom "O linkers".

    Sites with occupancy below min_occupancy are left out, as every
    other analysis does (pore_analysis.major_occupancy_structure; the
    same 0.5 the 3D view's disorder toggle uses): taking all of a
    disordered site's alternatives at once joined CALF-20's two
    oxalate orientations into "C4O2" fragments. Site indices are kept,
    so highlighting still maps onto the full structure; the count left
    out is in partition.excluded_disorder_sites. Pass None to keep
    every site.

    Returns (partition, site_images): atom_indices index into
    site_images, a list of (site index, lattice image) pairs placing each
    component near the reference cell, the same form
    build_periodic_graph_for_mof_analysis gives, so the 3D view's
    highlighting works unchanged.
    """
    from thevoid_core.services.pore_analysis import dimensionality_from_vectors

    n = len(structure.sites)
    symbols = [_site_symbol(site) for site in structure.sites]
    zero = (0, 0, 0)
    adjacency = [[] for _ in range(n)]
    excluded = set()
    if min_occupancy is not None:
        excluded = {
            i for i, site in enumerate(structure.sites)
            if site.species.num_atoms < min_occupancy
        }
    kept = set(range(n)) - excluded

    for bond in bonds:
        if bond.atom1 in excluded or bond.atom2 in excluded:
            continue
        image = tuple(int(v) for v in bond.image)
        adjacency[bond.atom1].append((bond.atom2, image))
        adjacency[bond.atom2].append((bond.atom1, tuple(-v for v in image)))

    metals = {i for i in kept if is_metal(symbols[i])}
    node_side = set(metals)

    for i, neighbours in enumerate(adjacency):
        if i in kept and i not in metals and len({(w, img) for w, img in neighbours if w in metals}) >= 2:
            node_side.add(i)

    # Node-side connections: bonds within the side, plus grouping links
    # between the metals a cluster carboxylate bridges.
    node_links = [[] for _ in range(n)]
    for i in node_side:
        for w, image in adjacency[i]:
            if w in node_side:
                node_links[i].append((w, image))

    position = structure_position(structure)
    for _carbon, _oxygens, ends in carboxylate_groups(symbols, adjacency, is_metal, position):
        if len({metal for metal, _ in ends}) >= 2 and bridges_one_cluster(ends, position):
            (first, first_cell), rest = ends[0], ends[1:]
            for metal, cell in rest:
                shift = tuple(b - a for a, b in zip(first_cell, cell))
                node_links[first].append((metal, shift))
                node_links[metal].append((first, tuple(-v for v in shift)))

    linker_links = [
        [(w, image) for w, image in adjacency[i] if w not in node_side]
        if i not in node_side else []
        for i in range(n)
    ]

    frac = np.array([site.frac_coords for site in structure.sites], dtype=float)

    def components(members, links):
        """[(offsets {site: cell}, dimensionality)] by offset BFS."""
        seen = set()
        found = []
        for start in sorted(members):
            if start in seen:
                continue
            offsets = {start: zero}
            seen.add(start)
            stack, cycles = [start], []
            while stack:
                u = stack.pop()
                for w, image in links[u]:
                    expected = tuple(a + b for a, b in zip(offsets[u], image))
                    if w not in offsets:
                        offsets[w] = expected
                        seen.add(w)
                        stack.append(w)
                    elif expected != offsets[w]:
                        cycles.append(tuple(a - b for a, b in zip(expected, offsets[w])))
            found.append([offsets, dimensionality_from_vectors(cycles)])
        return found

    nodes = components(node_side, node_links)
    ligands = components(kept - node_side, linker_links)
    node_of = {site: k for k, (offsets, _) in enumerate(nodes) for site in offsets}

    # A ligand bonded to exactly one node instance - coordinated water,
    # a hydroxide's H, a terminal F, a capping solvent - bridges nothing:
    # it is part of that SBU (Cr3O(H2O)2F), not a linker, and listing it
    # as one ("6 x O" in MIL-88B) was noise.
    linkers_kept = []
    for offsets, dims in ligands:
        ends = set()
        for b, o_b in offsets.items():
            for a, image in adjacency[b]:
                if a in node_side:
                    k = node_of[a]
                    # b's cell in node k's frame: o_a - image ... from a to b is -image'
                    o_a = nodes[k][0][a]
                    shift = tuple(x - y - z for x, y, z in zip(o_a, image, o_b))
                    ends.add((k, shift))
        if dims == 0 and len(ends) == 1:
            (k, shift), = ends
            # shift maps this ligand's frame into node k's frame.
            for b, o_b in offsets.items():
                nodes[k][0][b] = tuple(x + y for x, y in zip(o_b, shift))
        else:
            linkers_kept.append((offsets, dims))

    site_images: list = []

    def materialize(offsets):
        # Place the component around the reference cell.
        centre = np.mean([frac[i] + np.array(o) for i, o in offsets.items()], axis=0)
        shift = np.floor(centre).astype(int)
        indices = []
        for i in sorted(offsets):
            site_images.append((i, tuple(int(v) for v in np.array(offsets[i]) - shift)))
            indices.append(len(site_images) - 1)
        return indices, sorted(offsets)

    linkers = []
    for linker_id, (offsets, dims) in enumerate(linkers_kept, start=1):
        indices, sites = materialize(offsets)
        counts: dict[str, int] = defaultdict(int)
        for i in sites:
            counts[symbols[i]] += 1
        linkers.append(Linker(linker_id, tuple(indices), hill_formula(counts), dims))

    metal_nodes = []
    for node_id, (offsets, dims) in enumerate(nodes, start=1):
        indices, sites = materialize(offsets)
        counts = defaultdict(int)
        metal_symbols = []
        for i in sites:
            counts[symbols[i]] += 1
            if i in metals:
                metal_symbols.append(symbols[i])
        metal_nodes.append(MetalNode(
            node_id, tuple(indices), tuple(sorted(metal_symbols)), hill_formula(counts), dims
        ))

    partition = FrameworkPartition(
        linkers=linkers, metal_nodes=metal_nodes,
        excluded_disorder_sites=len(excluded),
    )
    return partition, site_images
