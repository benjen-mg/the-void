"""Linker functionalisation (round 59, roadmap 3b): hypothetical
variants of a MOF with ring hydrogens of its linkers replaced by
functional groups - IRMOF-1 to IRMOF-3 (-NH2) or IRMOF-2 (-Br).

What is substituted
-------------------
An aromatic C-H of a linker: a hydrogen bonded to a carbon that has
exactly three neighbours and sits in a ring of at most 8 atoms, in a
component that bonds to metal nodes (a linker, not lattice solvent).
Methyl, methylene and N-H hydrogens are left alone.

How a group is placed
---------------------
Rigidly, from standard geometry; the framework is not moved:

* The group's first atom goes along the exterior bisector of the ring
  angle at the carbon (where an ideal C-H points, and where riding-model
  H positions in X-ray structures are put), at a typical aromatic C-X
  bond length (GROUPS; Allen et al., J. Chem. Soc., Perkin Trans. 2
  (1987) S1, the standard tables for X-ray and neutron structures). Its
  other atoms follow from standard lengths and angles in the frame of
  the ring (in-plane and normal directions).
* Conjugated groups (-OH, -OCH3, -NH2, -NO2) are kept in the ring
  plane, and the C-X bond on the bisector, if that gives every contact
  ROOM_RATIO of the van der Waals sum. If not, the C-X bond is first
  bent in the ring plane, by the smallest angle up to MAX_BEND that
  gives the room: real ortho-crowded aromatics open the exocyclic angle
  on the crowded side like this. Only then is the group turned about
  its C-X bond. -CH3 and -CF3 are always turned (15-degree steps,
  refined to 5). Every group is optimised again, locally, once all are
  placed, until none moves, so neighbours see each other.

How a variant is judged
-----------------------
By its closest contact: between a new atom and any atom not within
three bonds of it (1-2, 1-3 and 1-4 pairs are fixed by the geometry),
as a fraction of the two atoms' van der Waals radii (the Void's
Bondi-based radii):

* below CLASH_RATIO (0.75) the variant is rejected;
* below TIGHT_RATIO (0.80) it is written, flagged "tight";
* a hydrogen bond - an N-H or O-H hydrogen, or its N/O, facing an O, N
  or F - is not a clash unless shorter than HBOND_MIN (H...A) or
  HBOND_HEAVY_MIN (D...A). 2-Aminoterephthalate's N-H...O=C is about
  2.0 A, 74% of the vdW sum.

Finally the bond detector is run on the variant: every new atom must
bond to exactly the atoms it was placed to bond to.

Ring turns (round 71)
---------------------
Real MOFs make room for a crowded group by twisting the ring against
its carboxylates (IRMOF-2's -Br rings 25 degrees, MIL-47(V)-Br's 19 and
26). If the flat build is tight or clashes, every substituted ring is
turned rigidly about its linker axis, the line through its two
outward-bonded atoms, so the rest of the linker and the framework stay
put. The angle is the smallest, in 5-degree steps up to 90, that leaves
the variant not tight (or else no longer a clash). Every ring of a
pattern turns by that angle, carried by symmetry (the other way under an
improper operation). The ring atoms that move are checked like the new
ones. A ring with more than two outward bonds (the core of a
tricarboxylate), or with a metal in it, is not turned.

Checked against real structures
-------------------------------
160 functionalised MOFs from CoRE MOF 2024 were checked (-CH3, -OH,
-NH2, -F, -Cl, -NO2, -Br, -I). Each real group was replaced by H and
rebuilt here:

* the rebuilt first atom lies a median 0.08 A from the real one (90%
  within 0.18 A);
* 158 of the 160 pass the check;
* the two that fail are phenolate O atoms 2.3 A from Co, a
  coordination bond rather than a free substituent.

Patterns and symmetry
---------------------
Linkers related by the parent's space group form an orbit. Patterns
are sets of substituted positions on one linker of an orbit, counted
once up to the operations that map that linker onto itself (for
terephthalate: one pattern for one group, ortho / meta / para for
two). A pattern is carried to every linker of the orbit by one
symmetry operation each, so every linker is substituted the same way
("uniform"). This gives one ordered model per pattern. Real
functionalised MOFs are often disordered over the equivalent
positions (IRMOF-3's -NH2 is), and other orderings of the same pattern
are not enumerated.
"""

from __future__ import annotations

import datetime
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from thevoid_core.version import __version__

#: Contacts below this fraction of the vdW radii sum reject a variant.
#: Calibrated by rebuilding the substituents of 160 real functionalised
#: MOFs (CoRE MOF 2024, round 59): their own closest contacts are below
#: 0.75 in 5% of cases, and rebuilt, 2 of the 160 fall below it - both
#: phenolate O at 2.3 A from Co, a coordination bond rather than a
#: substituent the rigid build can place.
CLASH_RATIO = 0.75

#: Contacts below this fraction are written but flagged "tight": closer
#: than in about 90% of real structures, so worth relaxing before the
#: numbers are trusted.
TIGHT_RATIO = 0.80

#: What placement aims for: the smallest bend or twist that gives every
#: contact this much room (a real substituent's median is 0.89).
ROOM_RATIO = 0.85

#: An N-H/O-H ... O/N/F hydrogen bond is only a clash below these (A):
#: H...acceptor, and donor...acceptor. Strong intramolecular hydrogen
#: bonds (a salicylate's O-H...O) reach about 1.6 and 2.4 A.
HBOND_MIN = 1.5
HBOND_HEAVY_MIN = 2.4

#: Rotor search step (degrees).
ROTOR_STEP = 5.0

#: Largest in-plane bend of the C-X bond off the exterior bisector
#: (degrees). On 160 real functionalised MOFs (CoRE MOF 2024) the real
#: substituent sits a median 2.8 degrees off the bisector, 90% within
#: 6.9.
MAX_BEND = 6.0

#: Rotor passes over all groups after the first, at most.
PASSES = 4

#: Round 71: ring turns, tried only when the flat build is tight or
#: clashes - steps (degrees) and the largest turn. Past 90 degrees a
#: ring comes back towards its own flat position, mirrored.
RING_TURN_STEP = 5.0
MAX_RING_TURN = 90.0

#: Groups whose ring carbons are further apart than this (A) cannot
#: touch (two groups reach at most ~3.5 A each, plus the contact limit).
NEIGHBOUR_CUTOFF = 11.0

#: Largest ring that counts as aromatic for "ring C-H".
MAX_RING = 8

#: Distance tolerance (A) for mapping sites under symmetry operations.
SYMMETRY_TOLERANCE = 0.4

#: symprec for the parent's space group.
SYMPREC = 0.1

#: Patterns per orbit and count are capped here, so a large linker with
#: many hydrogens cannot produce thousands of files by accident.
MAX_PATTERNS = 50

TETRAHEDRAL = 180.0 - 109.4712  # angle between a bond and the axis it is staggered about


# ----------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------


def _direction(angle_from_axis_deg, phi_deg):
    """(u, v, w) components of a unit vector at `angle` from u, turned
    by phi about u (phi = 0 is in the ring plane, v)."""
    a, p = math.radians(angle_from_axis_deg), math.radians(phi_deg)
    return np.array([math.cos(a), math.sin(a) * math.cos(p), math.sin(a) * math.sin(p)])


@dataclass(frozen=True)
class Group:
    """A functional group, built in the frame (u, v, w) of its ring
    carbon: u along the new C-X bond, v in the ring plane, w normal."""

    name: str
    description: str
    #: Bond lengths and angles used, for the record.
    geometry: dict
    #: Degrees; 0 for none. The group looks the same after this turn.
    rotor_period: float = 0.0
    #: Try the ring plane first (conjugated groups).
    prefers_planar: bool = False

    def build(self, phi: float):
        """[(element, depth, local position, bonded_to)] - depth is the
        number of bonds from the ring carbon, bonded_to the index of the
        atom (in this list) it hangs from, -1 for the ring carbon."""
        return _BUILDERS[self.name](self.geometry, phi)


def _halogen(g, phi):
    return [(g["element"], 1, np.array([g["C-X"], 0.0, 0.0]), -1)]


def _methyl_like(g, phi):
    c = np.array([g["C-C"], 0.0, 0.0])
    atoms = [("C", 1, c, -1)]
    for k in range(3):
        atoms.append((g["element"], 2, c + g["C-X"] * _direction(TETRAHEDRAL, phi + 120 * k), 0))
    return atoms


def _hydroxy(g, phi):
    o = np.array([g["C-O"], 0.0, 0.0])
    return [("O", 1, o, -1), ("H", 2, o + g["O-H"] * _direction(180 - g["C-O-H"], phi), 0)]


def _methoxy(g, phi):
    o = np.array([g["C-O"], 0.0, 0.0])
    axis = _direction(180 - g["C-O-C"], phi)
    c = o + g["O-C"] * axis
    atoms = [("O", 1, o, -1), ("C", 2, c, 0)]
    # Methyl hydrogens staggered about O-C, one anti to the ring carbon.
    p = -np.array([1.0, 0.0, 0.0]) + np.dot(np.array([1.0, 0.0, 0.0]), axis) * axis
    p = -p / np.linalg.norm(p)
    q = np.cross(axis, p)
    a = math.radians(TETRAHEDRAL)
    for k in range(3):
        t = math.radians(120 * k)
        d = math.cos(a) * axis + math.sin(a) * (math.cos(t) * p + math.sin(t) * q)
        atoms.append(("H", 3, c + g["C-H"] * d, 1))
    return atoms


def _amino(g, phi):
    n = np.array([g["C-N"], 0.0, 0.0])
    angle = 180 - g["C-N-H"]
    return [("N", 1, n, -1),
            ("H", 2, n + g["N-H"] * _direction(angle, phi), 0),
            ("H", 2, n + g["N-H"] * _direction(angle, phi + 180), 0)]


def _nitro(g, phi):
    n = np.array([g["C-N"], 0.0, 0.0])
    angle = 180 - g["C-N-O"]
    return [("N", 1, n, -1),
            ("O", 2, n + g["N-O"] * _direction(angle, phi), 0),
            ("O", 2, n + g["N-O"] * _direction(angle, phi + 180), 0)]


def _cyano(g, phi):
    c = np.array([g["C-C"], 0.0, 0.0])
    return [("C", 1, c, -1), ("N", 2, c + np.array([g["C-N"], 0.0, 0.0]), 0)]


_BUILDERS = {
    "F": _halogen, "Cl": _halogen, "Br": _halogen, "I": _halogen,
    "CH3": _methyl_like, "CF3": _methyl_like,
    "OH": _hydroxy, "OCH3": _methoxy, "NH2": _amino, "NO2": _nitro, "CN": _cyano,
}

GROUPS = {
    "F": Group("F", "fluoro", {"element": "F", "C-X": 1.363}),
    "Cl": Group("Cl", "chloro", {"element": "Cl", "C-X": 1.739}),
    "Br": Group("Br", "bromo", {"element": "Br", "C-X": 1.899}),
    "I": Group("I", "iodo", {"element": "I", "C-X": 2.095}),
    "CH3": Group("CH3", "methyl", {"element": "H", "C-C": 1.506, "C-X": 1.09,
                                   "angles": "tetrahedral"}, rotor_period=120),
    "CF3": Group("CF3", "trifluoromethyl", {"element": "F", "C-C": 1.50, "C-X": 1.336,
                                            "angles": "tetrahedral"}, rotor_period=120),
    "OH": Group("OH", "hydroxy", {"C-O": 1.362, "O-H": 0.967, "C-O-H": 109.0},
                rotor_period=360, prefers_planar=True),
    "OCH3": Group("OCH3", "methoxy", {"C-O": 1.370, "O-C": 1.424, "C-O-C": 117.5,
                                      "C-H": 1.09}, rotor_period=360, prefers_planar=True),
    "NH2": Group("NH2", "amino", {"C-N": 1.380, "N-H": 1.01, "C-N-H": 120.0,
                                  "note": "planar N, as in 2-aminoterephthalate"},
                 rotor_period=180, prefers_planar=True),
    "NO2": Group("NO2", "nitro", {"C-N": 1.468, "N-O": 1.217, "C-N-O": 118.0},
                 rotor_period=180, prefers_planar=True),
    "CN": Group("CN", "cyano", {"C-C": 1.443, "C-N": 1.136}),
}


def group(name: str) -> Group:
    for key, value in GROUPS.items():
        if key.lower() == name.strip().lower() or value.description == name.strip().lower():
            return value
    raise ValueError(f"Unknown group {name!r}; choose from {', '.join(GROUPS)}")


# ----------------------------------------------------------------------
# The parent: substitutable sites, linker orbits, patterns
# ----------------------------------------------------------------------


@dataclass
class Site:
    """A substitutable ring C-H."""

    hydrogen: int
    carbon: int
    #: The carbon's two other (ring) neighbours, (site, image) relative to it.
    ring_neighbours: tuple
    linker: int
    label: str
    ring_size: int = 6


@dataclass
class Orbit:
    """Linkers related by the space group, with the patterns of one."""

    number: int
    linkers: list            # linker ids, representative first
    formula: str
    sites: list              # hydrogens on the representative linker
    patterns: dict = field(default_factory=dict)   # m -> [(pattern tuple, descriptor)]
    #: linker id -> index into Parent.operations taking the rep there.
    transfer: dict = field(default_factory=dict)


@dataclass
class Parent:
    structure: object
    bonds: list
    source: str | None
    space_group: str
    operations: list          # permutation arrays (site -> site)
    sites: dict               # hydrogen index -> Site
    linker_sites: dict        # linker id -> set of site indices
    orbits: list
    notes: list = field(default_factory=list)
    #: +1 / -1 per operation: proper or improper (round 71).
    handedness: list = field(default_factory=list)


def _adjacency(n, bonds):
    adjacency = [[] for _ in range(n)]
    for bond in bonds:
        image = tuple(int(v) for v in bond.image)
        adjacency[bond.atom1].append((bond.atom2, image))
        adjacency[bond.atom2].append((bond.atom1, tuple(-v for v in image)))
    return adjacency


def _smallest_ring(start, adjacency, symbols):
    """Size of the smallest ring of heavy atoms through `start`, up to
    MAX_RING, or None. BFS over (site, image) nodes, one branch per
    first step; a ring closes where two branches meet, and the smallest
    closure of a BFS level is the smallest ring."""
    zero = (0, 0, 0)
    root = (start, zero)
    parent, depth, branch = {root: None}, {root: 0}, {root: None}
    frontier = [root]
    while frontier:
        best, nxt = None, []
        for node in frontier:
            site, cell = node
            for w, image in adjacency[site]:
                if symbols[w] in ("H", "D", "T"):
                    continue
                other = (w, tuple(a + b for a, b in zip(cell, image)))
                if other == parent[node] or other == root:
                    continue
                b = other if node == root else branch[node]
                if other in depth:
                    if branch[other] != b:
                        size = depth[other] + depth[node] + 1
                        best = size if best is None else min(best, size)
                    continue
                parent[other], depth[other], branch[other] = node, depth[node] + 1, b
                nxt.append(other)
        if best is not None:
            return best if best <= MAX_RING else None
        if frontier and depth[frontier[0]] * 2 >= MAX_RING:
            return None
        frontier = nxt
    return None


def substitutable_sites(structure, bonds, is_metal=None):
    """{hydrogen: Site} for every ring C-H on a linker, with the
    partition's linker id; and {linker id: site set}."""
    from thevoid_core.services.mof_analysis import bonded_components, partition_structure

    if is_metal is None:
        from thevoid_core.data.metal_classification import is_metal
    symbols = [site.specie.symbol for site in structure.sites]
    adjacency = _adjacency(len(symbols), bonds)
    partition, site_images = partition_structure(structure, bonds, is_metal, min_occupancy=None)
    free = set()
    for component in bonded_components(structure, bonds, is_metal):
        if not component.is_periodic:
            free.update(component.site_indices)
    linker_sites = {}
    for linker in partition.linkers:
        members = {site_images[k][0] for k in linker.atom_indices}
        if members & free:
            continue  # lattice solvent or a free ion, not a linker
        linker_sites[linker.linker_id] = members
    linker_of = {s: lid for lid, members in linker_sites.items() for s in members}

    labels = [getattr(site, "label", None) or f"{symbols[i]}{i + 1}"
              for i, site in enumerate(structure.sites)]
    sites = {}
    for h, element in enumerate(symbols):
        if element not in ("H", "D", "T") or h not in linker_of or len(adjacency[h]) != 1:
            continue
        c, _image = adjacency[h][0]
        if symbols[c] != "C" or len(adjacency[c]) != 3:
            continue
        others = [(w, img) for w, img in adjacency[c] if w != h]
        if len(others) != 2 or any(symbols[w] in ("H", "D", "T") for w, _ in others):
            continue
        ring = _smallest_ring(c, adjacency, symbols)
        if ring is None:
            continue
        sites[h] = Site(h, c, tuple(others), linker_of[h], labels[h], ring)
    return sites, linker_sites


def symmetry_operations(structure, symprec=SYMPREC, tolerance=SYMMETRY_TOLERANCE):
    """(space group symbol, [permutation arrays]): each of the
    structure's space-group operations as the site permutation it
    induces. An operation that does not map every site onto a site of
    the same species (within `tolerance` A) is dropped."""
    symbol, perms, _ = symmetry_operations_with_handedness(structure, symprec, tolerance)
    return symbol, perms


def symmetry_operations_with_handedness(structure, symprec=SYMPREC,
                                        tolerance=SYMMETRY_TOLERANCE):
    """symmetry_operations, and for each operation +1 if it is proper (a
    rotation) or -1 if improper (a mirror, inversion or rotoinversion):
    an improper operation turns a ring the other way (round 70's ring
    turns are carried to symmetry-equivalent linkers with it)."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    from scipy.spatial import cKDTree

    try:
        analyzer = SpacegroupAnalyzer(structure, symprec=symprec)
        symbol = analyzer.get_space_group_symbol()
        operations = analyzer.get_symmetry_operations(cartesian=False)
    except Exception:
        return "P1", [np.arange(len(structure))], [1]
    frac = np.mod(structure.frac_coords, 1.0)
    species = [site.specie.symbol for site in structure.sites]
    tree = cKDTree(frac, boxsize=1.0 + 1e-12)
    lattice = structure.lattice.matrix
    perms, handedness = [], []
    for op in operations:
        mapped = np.mod(op.operate_multi(frac), 1.0) % 1.0
        _, index = tree.query(mapped)
        delta = mapped - frac[index]
        delta -= np.rint(delta)
        distance = np.linalg.norm(delta @ lattice, axis=1)
        if (distance > tolerance).any() or len(set(index.tolist())) != len(frac):
            continue
        if any(species[i] != species[j] for i, j in enumerate(index)):
            continue
        perms.append(index)
        handedness.append(1 if np.linalg.det(op.rotation_matrix) > 0 else -1)
    if not perms:
        perms, handedness = [np.arange(len(structure))], [1]
    return symbol, perms, handedness


def _ring_distance(c1, c2, adjacency, limit=4):
    """Bonds between two carbons (site indices), ignoring images, up to
    `limit`; None if further."""
    if c1 == c2:
        return 0
    seen, frontier = {c1}, [c1]
    for d in range(1, limit + 1):
        nxt = []
        for u in frontier:
            for w, _ in adjacency[u]:
                if w == c2:
                    return d
                if w not in seen:
                    seen.add(w)
                    nxt.append(w)
        frontier = nxt
    return None


def _descriptor(pattern, sites, adjacency):
    if len(pattern) == 1:
        return "mono"
    if len(pattern) == 2:
        a, b = sites[pattern[0]], sites[pattern[1]]
        d = _ring_distance(a.carbon, b.carbon, adjacency)
        if a.ring_size == b.ring_size == 6 and d in (1, 2, 3):
            return {1: "ortho", 2: "meta", 3: "para"}[d]
        return f"d{d}" if d else ""
    return {3: "tri", 4: "tetra"}.get(len(pattern), f"{len(pattern)}x")


def analyze_parent(structure, bonds=None, *, source=None, max_per_linker=4,
                   bond_detector=None, is_metal=None) -> Parent:
    """Everything needed to generate variants of `structure`: ordered
    and bonded as every analysis sees it, its substitutable sites,
    linker orbits, and the distinct patterns of 1..max_per_linker
    groups per linker."""
    from thevoid_core.services.raspa_export import prepare_framework

    prepared = prepare_framework(structure, bonds, remove_free_molecules=False,
                                 bond_detector=bond_detector)
    structure, bonds = prepared.structure, prepared.bonds
    frac = structure.frac_coords
    wrapped = frac - np.floor(frac)
    wrapped[wrapped >= 1.0] = 0.0  # -1e-17 - floor(...) rounds to exactly 1.0
    if not np.array_equal(wrapped, frac):
        # Every site in [0, 1), so a bond's image, a site's periodic
        # copies and a placed group all use the same cell. (A site at
        # -1e-17 is stored as 1.0 by np.mod but copied to 0.0 by
        # periodic_images: its own carbon then looked like a neighbour
        # 1.5 A away - found by the strip-and-rebuild check.)
        from pymatgen.core import Structure

        from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

        structure = Structure(structure.lattice, [site.species for site in structure.sites],
                              wrapped, labels=[site.label for site in structure.sites])
        bonds = (bond_detector or CovalentBondDetector()).detect(structure)
    sites, linker_sites = substitutable_sites(structure, bonds, is_metal)
    symbol, perms, handedness = symmetry_operations_with_handedness(structure)
    adjacency = _adjacency(len(structure), bonds)

    orbits = []
    assigned = set()
    key_of = {lid: frozenset(members) for lid, members in linker_sites.items()}
    by_key = {v: k for k, v in key_of.items()}
    for lid in sorted(linker_sites):
        if lid in assigned:
            continue
        rep = key_of[lid]
        transfer, stabiliser = {}, []
        for index, perm in enumerate(perms):
            image = frozenset(int(perm[s]) for s in rep)
            target = by_key.get(image)
            if target is None:
                continue
            transfer.setdefault(target, index)
            if target == lid:
                stabiliser.append(perm)
        members = sorted(transfer, key=lambda t: (t != lid, t))
        assigned.update(members)
        rep_sites = sorted(h for h in sites if sites[h].linker == lid)
        counts = {}
        for s in rep:
            element = structure.sites[s].specie.symbol
            counts[element] = counts.get(element, 0) + 1
        from thevoid_core.services.mof_analysis import hill_formula

        orbit = Orbit(len(orbits) + 1, members, hill_formula(counts), rep_sites,
                      transfer=transfer)
        for m in range(1, min(max_per_linker, len(rep_sites)) + 1):
            seen, found = set(), []
            for combo in itertools.combinations(rep_sites, m):
                canonical = min(tuple(sorted(int(p[h]) for h in combo)) for p in stabiliser)
                if canonical in seen:
                    continue
                seen.add(canonical)
                found.append((combo, _descriptor(combo, sites, adjacency)))
                if len(found) >= MAX_PATTERNS:
                    break
            orbit.patterns[m] = found
        orbits.append(orbit)  # also with no sites: the listing says why nothing is made

    notes = list(prepared.notes)
    if not sites:
        notes.append("No ring C-H on a linker: nothing to functionalise.")
    return Parent(structure, bonds, source, symbol, perms, sites, linker_sites, orbits, notes,
                  handedness=handedness)


def site_name(structure, index) -> str:
    """A site's CIF label with its (1-based) index: symmetry-generated
    copies share a label."""
    label = getattr(structure.sites[index], "label", None) or structure.sites[index].specie.symbol
    return f"{label}#{index + 1}"


def pattern_sites(parent: Parent, orbit: Orbit, pattern) -> list[int]:
    """Every hydrogen replaced when `pattern` (on the representative)
    is carried to every linker of the orbit."""
    hydrogens = []
    for lid in orbit.linkers:
        perm = parent.operations[orbit.transfer[lid]]
        hydrogens.extend(int(perm[h]) for h in pattern)
    return sorted(set(hydrogens))


# ----------------------------------------------------------------------
# Placement
# ----------------------------------------------------------------------


@dataclass
class Placed:
    """One group on one ring carbon."""

    hydrogen: int
    carbon: int
    origin: np.ndarray        # the carbon (Cartesian, in the cell)
    frame: np.ndarray         # rows u, v, w
    near: dict                # (site, image) -> bonds from the carbon, up to 3
    phi: float = 0.0
    bend: float = 0.0         # in-plane turn of the C-X bond off the bisector (deg)
    base: np.ndarray | None = None  # the unbent frame
    #: Groups close enough to matter (set once all are known).
    neighbours: list | None = None
    atoms: list = field(default_factory=list)   # [(element, depth, cart, bonded_to)]


def _frame(structure, site: Site):
    lattice = structure.lattice.matrix
    c = structure.sites[site.carbon].coords
    neighbours = [structure.sites[w].frac_coords @ lattice + np.array(img) @ lattice
                  for w, img in site.ring_neighbours]
    a, b = (neighbours[0] - c), (neighbours[1] - c)
    u = -(a / np.linalg.norm(a) + b / np.linalg.norm(b))
    if np.linalg.norm(u) < 1e-6:  # a linear carbon: use the old C-H
        u = structure.sites[site.hydrogen].coords - c
    u /= np.linalg.norm(u)
    w = np.cross(a, b)
    w /= np.linalg.norm(w)
    v = np.cross(w, u)
    return c, np.vstack([u, v, w])


def _near(carbon, adjacency, depth=3):
    zero = (0, 0, 0)
    found = {(carbon, zero): 0}
    frontier = [(carbon, zero)]
    for d in range(1, depth + 1):
        nxt = []
        for site, cell in frontier:
            for w, image in adjacency[site]:
                node = (w, tuple(a + b for a, b in zip(cell, image)))
                if node not in found:
                    found[node] = d
                    nxt.append(node)
        frontier = nxt
    return found


class _Environment:
    """Periodic images of the parent (less the replaced hydrogens) and
    of the placed groups, for contact checks."""

    def __init__(self, structure, removed, radius_provider, reach=6.0):
        from thevoid_core.services.pore_geometry import periodic_images

        self.lattice = structure.lattice.matrix
        self.symbols = [s.specie.symbol for s in structure.sites]
        keep = np.array([i for i in range(len(structure)) if i not in removed])
        self.keep = keep
        frac = np.mod(structure.frac_coords[keep], 1.0)
        positions, _, base, offsets = periodic_images(self.lattice, frac, np.zeros(len(keep)), reach)
        from scipy.spatial import cKDTree

        self.positions = positions
        self.site = keep[base]
        self.offset = offsets
        self.tree = cKDTree(positions)
        self.offset_tuple = [tuple(int(v) for v in row) for row in offsets]
        self.inverse = np.linalg.inv(self.lattice)
        self.radii = radius_provider
        self.hosts = {}     # hydrogen site -> element it is bonded to
        self.donors = set()  # N/O sites carrying a hydrogen
        self._vdw = {}

    def vdw(self, element):
        if element not in self._vdw:
            self._vdw[element] = self.radii.get_vdw_radius(element)
        return self._vdw[element]


def _contacts(placed: Placed, env: _Environment, others, structure, limit=3.2):
    """[(ratio, distance, description, moves)] for the new atoms of `placed`
    against the framework and the other groups, every pair more than
    three bonds apart and within `limit` + the new atom's radius."""
    results = []
    lattice, inverse = env.lattice, env.inverse
    atoms = placed.atoms
    elements = [a[0] for a in atoms]
    donors = {a[3] for a in atoms if a[0] == "H" and a[3] >= 0}
    near = placed.near
    pool = [o for o in (placed.neighbours if placed.neighbours is not None else others)
            if o is not placed and o.atoms]
    if pool:
        other_cart = np.array([a[2] for o in pool for a in o.atoms])
        other_meta = [(o, m) for o in pool for m in range(len(o.atoms))]
    for k, (element, depth, cart, host) in enumerate(atoms):
        host_element = "C" if host < 0 else elements[host]
        new_donor = element in ("N", "O") and k in donors
        r_new = env.vdw(element)
        # Framework
        hits = env.tree.query_ball_point(cart, limit + r_new)
        if hits:
            hits = np.asarray(hits)
            distances = np.linalg.norm(env.positions[hits] - cart, axis=1)
            for j, distance in zip(hits.tolist(), distances.tolist()):
                site = int(env.site[j])
                bonds_away = near.get((site, env.offset_tuple[j]))
                if bonds_away is not None and bonds_away + depth <= 3:
                    continue
                other = env.symbols[site]
                results.append(_judge(
                    distance, element, host_element, new_donor, other,
                    env.hosts.get(site), site in env.donors, env,
                    f"{element} (new) - {structure.sites[site].label or other + str(site + 1)}")
                    + (depth >= 2,))
        # Other groups, through the nearest periodic image.
        if pool:
            delta = other_cart - cart
            shift = np.rint(delta @ inverse)
            delta = delta - shift @ lattice
            distances = np.linalg.norm(delta, axis=1)
            for index in np.nonzero(distances <= limit + r_new)[0].tolist():
                other, m = other_meta[index]
                e2, d2, _cart2, host2 = other.atoms[m]
                image = tuple(int(v) for v in -shift[index])
                between = near.get((other.carbon, image))
                if between is not None and depth + d2 + between <= 3:
                    continue
                h2 = "C" if host2 < 0 else other.atoms[host2][0]
                other_donor = e2 in ("N", "O") and any(
                    a[0] == "H" and a[3] == m for a in other.atoms)
                results.append(_judge(
                    float(distances[index]), element, host_element, new_donor, e2, h2,
                    other_donor, env, f"{element} (new) - {e2} (new, on C{other.carbon + 1})")
                    + (depth >= 2 or d2 >= 2,))
    return results


def _judge(distance, e1, host1, donor1, e2, host2, donor2, env, text):
    """(ratio, distance, text). A hydrogen bond - an N-H or O-H hydrogen,
    or its N/O, facing an O, N or F - is not a clash unless shorter than
    HBOND_MIN (H...A) or HBOND_HEAVY_MIN (D...A); below those it counts
    in proportion, so it can reject the variant."""
    acceptors = ("N", "O", "F")
    h_pair = (e1 == "H" and host1 in ("N", "O") and e2 in acceptors) or \
        (e2 == "H" and host2 in ("N", "O") and e1 in acceptors)
    heavy_pair = (e1 in acceptors and e2 in acceptors) and (donor1 or donor2)
    if h_pair or heavy_pair:
        minimum = HBOND_MIN if h_pair else HBOND_HEAVY_MIN
        ratio = 1.0 if distance >= minimum else CLASH_RATIO * distance / minimum
        return (ratio, distance, text + " (H-bond)")
    return (distance / (env.vdw(e1) + env.vdw(e2)), distance, text)


def _set_bend(placed: Placed, beta: float):
    """Turn the C-X bond by beta degrees in the ring plane, off the
    exterior bisector (the ring stays as it is)."""
    if placed.base is None:
        placed.base = placed.frame.copy()
    u, v, w = placed.base
    b = math.radians(beta)
    u2 = math.cos(b) * u + math.sin(b) * v
    placed.frame = np.vstack([u2, np.cross(w, u2), w])
    placed.bend = beta


def _place(placed: Placed, grp: Group, phi: float):
    placed.phi = phi
    placed.atoms = [(e, d, placed.origin + local @ placed.frame, host)
                    for e, d, local, host in grp.build(phi)]


def _rotor_score(contacts):
    """The three closest contacts that the rotor changes (the first atom
    sits on the rotor axis, so its own contacts are the same at every
    angle and must not decide it), as ratios, closest first."""
    ratios = sorted(round(c[0], 6) for c in contacts if c[3])[:3]
    return tuple(ratios + [math.inf] * (3 - len(ratios)))


def _score(contacts):
    """The three closest contacts as ratios, closest first - compared
    as a tuple, so of two placements with the same closest contact the
    one with more room at the next is preferred."""
    ratios = sorted(round(c[0], 6) for c in contacts)[:3]
    return tuple(ratios + [math.inf] * (3 - len(ratios)))


def _bends():
    """0, then +-1, +-2, ... up to MAX_BEND degrees."""
    return [0.0] + [float(sign * k) for k in range(1, int(MAX_BEND) + 1) for sign in (1, -1)]


def _optimise(placed, grp, env, others, structure, local=False):
    """Bend and rotor angle of one group, the others held fixed.

    The C-X bond stays on the exterior bisector, and a conjugated group
    in the ring plane, unless that leaves a contact below ROOM_RATIO:

    1. a conjugated group is kept planar with the smallest in-plane bend
       (up to MAX_BEND) that clears it - real ortho-crowded aromatics
       open the exocyclic angle on the crowded side by a few degrees;
    2. otherwise the smallest bend at which some rotor angle clears it
       (15-degree steps, refined in ROTOR_STEP);
    3. if nothing clears it, the placement with the most room is kept
       and the variant is judged on it.

    local=True searches only near the current placement (later passes).
    """
    def evaluate(beta, phi):
        _set_bend(placed, beta)
        _place(placed, grp, phi)
        return _score(_contacts(placed, env, others, structure))

    period = grp.rotor_period
    if local:
        b0, p0 = placed.bend, placed.phi
        bends = sorted({max(-MAX_BEND, min(MAX_BEND, b0 + d)) for d in (0, 1, -1, 2, -2)},
                       key=lambda b: (abs(b - b0), abs(b)))
    else:
        bends = _bends()

    planar = ([0.0, 180.0] if period > 180 else [0.0]) if grp.prefers_planar else []
    if planar:
        for beta in bends:
            score, phi = max((evaluate(beta, phi), phi) for phi in planar)
            if score[0] >= ROOM_RATIO:
                _set_bend(placed, beta)
                _place(placed, grp, phi)
                return

    def rotor_angles():
        if not period:
            return [0.0]
        if local:
            return [(placed_phi + d) % period for d in np.arange(-15.0, 15.0 + 1e-9, ROTOR_STEP)]
        return [k * 15.0 for k in range(int(round(period / 15.0)))]

    best = None
    for beta in bends:
        placed_phi = p0 if local else 0.0
        score, phi = max((evaluate(beta, float(phi)), float(phi)) for phi in rotor_angles())
        if period and not local:
            fine = [(evaluate(beta, float((phi + d) % period)), float((phi + d) % period))
                    for d in np.arange(-15.0 + ROTOR_STEP, 15.0, ROTOR_STEP) if d]
            score, phi = max(fine + [(score, phi)])
        candidate = (score, -abs(beta), beta, phi)
        if best is None or candidate > best:
            best = candidate
        if score[0] >= ROOM_RATIO:
            break
    _set_bend(placed, best[2])
    _place(placed, grp, best[3])


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    group: str
    orbit: int
    per_linker: int
    pattern: tuple            # representative hydrogens
    descriptor: str
    replaced: list            # hydrogen site indices
    structure: object = None
    status: str = ""          # "ok" | "tight" | "clash"
    closest: list = field(default_factory=list)   # [(ratio, distance, text)]
    rotors: list = field(default_factory=list)
    bends: list = field(default_factory=list)
    space_group: str = ""
    reason: str = ""
    #: Round 71: the ring turn used (degrees, on the representative
    #: linker; 0 if none), each substituted ring's own, and why.
    ring_turn: float = 0.0
    ring_turns: list = field(default_factory=list)
    ring_note: str = ""

    @property
    def formula(self):
        return self.structure.composition.formula.replace(" ", "") if self.structure is not None else ""


def build_variant(parent: Parent, grp: Group, orbit: Orbit | None, pattern, descriptor="",
                  *, name="", radius_provider=None, hydrogens=None,
                  ring_turn: bool = True) -> Variant:
    """One uniform variant: `grp` on `pattern` of every linker in
    `orbit` - or, with `hydrogens`, on exactly those ring C-H sites.

    ring_turn: if the flat build is tight or clashes, turn the
    substituted rings about their linker axis by the smallest angle that
    clears it (round 71; see turn_rings)."""
    variant = _build_flat(parent, grp, orbit, pattern, descriptor, name=name,
                          radius_provider=radius_provider, hydrogens=hydrogens)
    if ring_turn and variant.status in ("tight", "clash"):
        variant = turn_rings(parent, grp, orbit, pattern, descriptor, variant, name=name,
                             radius_provider=radius_provider, hydrogens=hydrogens)
    return variant


def _build_flat(parent: Parent, grp: Group, orbit: Orbit | None, pattern, descriptor="",
                *, name="", radius_provider=None, hydrogens=None) -> Variant:
    """build_variant without ring turns: the framework exactly as given."""
    from pymatgen.core import Structure

    from thevoid_core.data.element_radii import ElementRadiusProvider

    radius_provider = radius_provider or ElementRadiusProvider()
    structure = parent.structure
    if hydrogens is not None:
        unknown = [h for h in hydrogens if h not in parent.sites]
        if unknown:
            raise ValueError(f"site(s) {unknown} are not substitutable ring C-H")
        replaced = sorted(set(int(h) for h in hydrogens))
    else:
        replaced = pattern_sites(parent, orbit, pattern)
    variant = Variant(name, grp.name, orbit.number if orbit else 0, len(pattern),
                      tuple(pattern), descriptor, replaced)
    adjacency = _adjacency(len(structure), parent.bonds)
    env = _Environment(structure, set(replaced), radius_provider)
    for h in range(len(structure)):
        if structure.sites[h].specie.symbol in ("H", "D", "T") and adjacency[h]:
            host = adjacency[h][0][0]
            env.hosts[h] = structure.sites[host].specie.symbol
            if env.hosts[h] in ("N", "O"):
                env.donors.add(host)

    groups = []
    for h in replaced:
        site = parent.sites[h]
        origin, frame = _frame(structure, site)
        groups.append(Placed(h, site.carbon, origin, frame, _near(site.carbon, adjacency)))
    lattice, inverse = structure.lattice.matrix, np.linalg.inv(structure.lattice.matrix)
    origins = np.array([g.origin for g in groups]) if groups else np.zeros((0, 3))
    for placed in groups:
        delta = origins - placed.origin
        delta -= np.rint(delta @ inverse) @ lattice
        close = np.linalg.norm(delta, axis=1) <= NEIGHBOUR_CUTOFF
        placed.neighbours = [g for g, c in zip(groups, close) if c and g is not placed]
    for placed in groups:  # first pass: each sees those already placed
        _optimise(placed, grp, env, None, structure)
    for _ in range(PASSES):  # then each sees all the others, until nothing moves
        moved = False
        for placed in groups:
            before = (placed.phi, placed.bend)
            _optimise(placed, grp, env, None, structure, local=True)
            moved |= (placed.phi, placed.bend) != before
        if not moved:
            break

    contacts = []
    for placed in groups:
        contacts.extend(_contacts(placed, env, groups, structure))
    contacts.sort()
    variant.closest = [(round(r, 4), round(d, 4), text) for r, d, text, _moves in contacts[:5]]
    variant.rotors = [round(float(p.phi), 1) for p in groups]
    variant.bends = [round(float(p.bend), 1) for p in groups]
    worst = contacts[0][0] if contacts else math.inf
    variant.status = "clash" if worst < CLASH_RATIO else "tight" if worst < TIGHT_RATIO else "ok"
    if variant.status == "clash":
        variant.reason = (f"closest contact {contacts[0][1]:.2f} A ({contacts[0][2]}), "
                          f"{contacts[0][0]:.0%} of the vdW sum")

    keep = [i for i in range(len(structure)) if i not in set(replaced)]
    species = [structure.sites[i].specie.symbol for i in keep]
    coords = [structure.sites[i].coords for i in keep]
    labels = [structure.sites[i].label for i in keep]
    counter = 0
    new_atoms = []
    for placed in groups:
        counter += 1
        for k, (element, _depth, cart, _host) in enumerate(placed.atoms):
            new_atoms.append((len(species), placed, k))
            species.append(element)
            coords.append(cart)
            labels.append(f"{element}f{counter}{chr(97 + k)}")
    variant.structure = Structure(structure.lattice, species, coords, coords_are_cartesian=True,
                                  labels=labels)

    if variant.status != "clash":
        problem = _bond_check(variant.structure, new_atoms, keep, groups)
        if problem:
            variant.status, variant.reason = "clash", problem
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

        variant.space_group = SpacegroupAnalyzer(variant.structure, symprec=SYMPREC) \
            .get_space_group_symbol()
    except Exception:
        variant.space_group = "?"
    return variant


def _bond_check(structure, new_atoms, keep, groups):
    """None if every new atom bonds to exactly what it was placed to
    bond to; otherwise what else it bonds to."""
    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

    new_index = {(id(p), k): i for i, p, k in new_atoms}
    carbon_index = {p.carbon: keep.index(p.carbon) for p in groups}
    expected = {}
    for i, placed, k in new_atoms:
        host = placed.atoms[k][3]
        j = carbon_index[placed.carbon] if host < 0 else new_index[(id(placed), host)]
        expected.setdefault(i, set()).add(j)
        expected.setdefault(j, set()).add(i)
    found = {}
    for bond in CovalentBondDetector().detect(structure):
        for a, b in ((bond.atom1, bond.atom2), (bond.atom2, bond.atom1)):
            found.setdefault(a, set()).add(b)
    for i, _placed, _k in new_atoms:
        extra = found.get(i, set()) - expected.get(i, set())
        missing = expected.get(i, set()) - found.get(i, set())
        if extra:
            j = min(extra)
            return (f"new {structure.sites[i].specie.symbol} would bond to "
                    f"{structure.sites[j].specie.symbol} ({structure.sites[j].label})")
        if missing:
            return f"new {structure.sites[i].specie.symbol} is not bonded where it was placed"
    return None


# ----------------------------------------------------------------------
# Ring turns (round 71)
# ----------------------------------------------------------------------


@dataclass
class RingBody:
    """A substituted ring that can turn about its linker axis: the two
    ring atoms bonded outwards (para to each other), and what moves with
    the ring - its other atoms and their hydrogens. Images are relative
    to the defining carbon's cell. `first` is the axis atom nearer the
    defining carbon, so the axis direction (first -> second) is the same
    on symmetry-equivalent rings."""

    carbon: int
    first: tuple
    second: tuple
    moved: list

    @property
    def key(self):
        return frozenset(site for site, _ in self.moved)


def _ring_path(carbon, adjacency, symbols):
    """[(site, image)] around the smallest heavy-atom ring through
    carbon, starting there (images relative to it); None if none of at
    most MAX_RING atoms."""
    from collections import deque

    zero = (0, 0, 0)
    best = None
    for first, image in adjacency[carbon]:
        if symbols[first] in ("H", "D", "T"):
            continue
        queue = deque([[(carbon, zero), (first, tuple(image))]])
        seen = {(first, tuple(image))}
        while queue:
            path = queue.popleft()
            if len(path) > MAX_RING or (best is not None and len(path) >= len(best)):
                break
            site, cell = path[-1]
            for w, step in adjacency[site]:
                if symbols[w] in ("H", "D", "T"):
                    continue
                node = (w, tuple(a + b for a, b in zip(cell, step)))
                if node == (carbon, zero) and len(path) >= 3:
                    if best is None or len(path) < len(best):
                        best = path
                    continue
                if node in seen or node == path[-2]:
                    continue
                seen.add(node)
                queue.append(path + [node])
    return best


def ring_body(parent: Parent, hydrogen: int, is_metal=None):
    """(RingBody, "") for the ring carrying `hydrogen`'s carbon, or
    (None, why it cannot turn): it must be one ring of even size whose
    only outward bonds are at two opposite (para) atoms, with nothing
    but hydrogens (and the groups being placed) on the others, and no
    metal in it."""
    if is_metal is None:
        from thevoid_core.data.metal_classification import is_metal
    structure = parent.structure
    symbols = [site.specie.symbol for site in structure.sites]
    adjacency = _adjacency(len(symbols), parent.bonds)
    carbon = parent.sites[hydrogen].carbon
    ring = _ring_path(carbon, adjacency, symbols)
    if ring is None:
        return None, "not in a ring"
    if any(is_metal(symbols[site]) for site, _ in ring):
        return None, "a metal is part of the ring"
    members = set(ring)
    outward = {}
    for node in ring:
        site, cell = node
        outward[node] = [(w, tuple(a + b for a, b in zip(cell, step)))
                         for w, step in adjacency[site]
                         if (w, tuple(a + b for a, b in zip(cell, step))) not in members]
    anchors = [node for node in ring
               if any(symbols[w] not in ("H", "D", "T") for w, _ in outward[node])]
    if len(anchors) != 2:
        return None, f"{len(anchors)} ring atoms bond outwards (a ring turns only between two)"
    i, j = ring.index(anchors[0]), ring.index(anchors[1])
    if len(ring) % 2 or abs(i - j) != len(ring) // 2:
        return None, "the ring's two outward bonds are not opposite each other"
    moved = []
    for node in ring:
        if node in anchors:
            continue
        moved.append(node)
        moved.extend(outward[node])  # only hydrogens (checked above)
    # anchors[0] is the one nearer the carbon along the ring (ring[0]).
    first, second = sorted(anchors, key=lambda n: min(ring.index(n), len(ring) - ring.index(n)))
    return RingBody(carbon, first, second, moved), ""


def _rotation(axis, degrees):
    axis = axis / np.linalg.norm(axis)
    t = math.radians(degrees)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(t) * k + (1.0 - math.cos(t)) * (k @ k)


def turned_parent(parent: Parent, turns) -> Parent:
    """A copy of parent with each (RingBody, degrees) ring turned about
    its axis (first -> second, right-handed). Moved sites are wrapped
    back into the cell and their bonds' images adjusted, so bonds, sites
    and periodic images stay consistent."""
    from pymatgen.core import Structure

    from thevoid_core.services.bonds.bond_model import Bond

    structure = parent.structure
    lattice = structure.lattice.matrix
    inverse = np.linalg.inv(lattice)
    frac = structure.frac_coords.copy()

    def cart(site, image):
        return (structure.frac_coords[site] + np.array(image)) @ lattice

    for body, degrees in turns:
        a, b = cart(*body.first), cart(*body.second)
        rotation = _rotation(b - a, degrees)
        for site, image in body.moved:
            new = rotation @ (cart(site, image) - a) + a
            frac[site] = new @ inverse - np.array(image)
    # Wrapped into [0, 1) as analyze_parent wraps: a value that rounds
    # to 1.0 becomes 0.0, one cell on (round 59's cell-edge bug).
    shift = np.floor(frac)
    frac = frac - shift
    edge = frac >= 1.0
    frac[edge] = 0.0
    shift[edge] += 1.0
    shift = shift.astype(int)
    bonds = []
    for bond in parent.bonds:
        image = tuple(int(v) for v in np.array(bond.image) - shift[bond.atom1] + shift[bond.atom2])
        bonds.append(Bond(bond.atom1, bond.atom2, image, bond.length))
    turned = Structure(structure.lattice, [site.species for site in structure.sites], frac,
                       labels=[site.label for site in structure.sites])
    sites = {}
    for h, site in parent.sites.items():
        ring_neighbours = tuple((w, tuple(int(v) for v in np.array(img) - shift[site.carbon]
                                          + shift[w])) for w, img in site.ring_neighbours)
        sites[h] = Site(site.hydrogen, site.carbon, ring_neighbours, site.linker, site.label,
                        site.ring_size)
    import dataclasses

    return dataclasses.replace(parent, structure=turned, bonds=bonds, sites=sites)


def _turned_atom_contacts(parent: Parent, bodies, replaced, radius_provider, limit=3.2):
    """[(ratio, distance, text)] for the atoms a ring turn moved, against
    every atom more than three bonds away that did not move with them."""
    structure = parent.structure
    adjacency = _adjacency(len(structure), parent.bonds)
    env = _Environment(structure, set(replaced), radius_provider)
    for h in range(len(structure)):
        if structure.sites[h].specie.symbol in ("H", "D", "T") and adjacency[h]:
            host = adjacency[h][0][0]
            env.hosts[h] = structure.sites[host].specie.symbol
            if env.hosts[h] in ("N", "O"):
                env.donors.add(host)
    results = []
    for body in bodies:
        rigid = body.key | {body.first[0], body.second[0]}
        for site in sorted(body.key - set(replaced)):
            element = env.symbols[site]
            near = _near(site, adjacency)
            position = structure.sites[site].coords
            for j in env.tree.query_ball_point(position, limit + env.vdw(element)):
                other = int(env.site[j])
                if other in rigid:
                    continue
                away = near.get((other, env.offset_tuple[j]))
                if away is not None and away <= 3:
                    continue
                distance = float(np.linalg.norm(env.positions[j] - position))
                results.append(_judge(
                    distance, element, env.hosts.get(site), site in env.donors,
                    env.symbols[other], env.hosts.get(other), other in env.donors, env,
                    f"{element} (turned ring) - "
                    f"{structure.sites[other].label or env.symbols[other] + str(other + 1)}"))
    return results


def _ring_turn_plan(parent: Parent, orbit: Orbit | None, pattern, replaced):
    """([(RingBody, sign)], "") for every substituted ring, each with the
    sign that carries the representative's turn to it by symmetry, or
    (None, why not)."""
    plan_ = {}
    if orbit is not None and pattern:
        rep_rings = {}
        for h in sorted(pattern):
            body, why = ring_body(parent, h)
            if body is None:
                return None, why
            rep_rings.setdefault(body.key, h)
        for lid in orbit.linkers:
            index = orbit.transfer[lid]
            perm = parent.operations[index]
            sign = parent.handedness[index] if parent.handedness else 1
            for h in rep_rings.values():
                body, why = ring_body(parent, int(perm[h]))
                if body is None:
                    return None, why
                plan_.setdefault(body.key, (body, sign))
    else:
        for h in replaced:
            body, why = ring_body(parent, h)
            if body is None:
                return None, why
            plan_.setdefault(body.key, (body, 1))
    return list(plan_.values()), ""


def turn_rings(parent: Parent, grp: Group, orbit: Orbit | None, pattern, descriptor,
               flat: Variant, *, name="", radius_provider=None, hydrogens=None) -> Variant:
    """The flat variant was tight or clashed: turn every substituted ring
    about its linker axis - the line through its two outward-bonded
    atoms, so the rest of the linker and the framework stay put - by the
    smallest angle, in RING_TURN_STEP steps up to MAX_RING_TURN, at which
    the variant is no longer tight (or, failing that, no longer a
    clash). A pattern's rings all turn by the same angle, carried by
    symmetry (the other way under an improper operation); with explicit
    `hydrogens` each ring turns the same way about its own axis. The
    atoms the turn moves are checked like the new ones. If no angle
    helps, the flat variant is kept, with the reason."""
    from thevoid_core.data.element_radii import ElementRadiusProvider

    radius_provider = radius_provider or ElementRadiusProvider()
    rings, why = _ring_turn_plan(parent, orbit, pattern, flat.replaced)
    if rings is None:
        flat.ring_note = f"ring not turned: {why}"
        return flat
    rank = {"clash": 0, "tight": 1, "ok": 2}
    best = None
    steps = int(round(MAX_RING_TURN / RING_TURN_STEP))
    for k in range(1, steps + 1):
        for direction in (1, -1):
            degrees = direction * k * RING_TURN_STEP
            turned = turned_parent(parent, [(body, sign * degrees) for body, sign in rings])
            variant = _build_flat(turned, grp, orbit, pattern, descriptor, name=name,
                                  radius_provider=radius_provider, hydrogens=hydrogens)
            moved = sorted(_turned_atom_contacts(turned, [b for b, _ in rings],
                                                 variant.replaced, radius_provider))
            if moved and (not variant.closest or moved[0][0] < variant.closest[0][0]):
                ratio, distance, text = moved[0]
                variant.closest = sorted(variant.closest + [(round(ratio, 4), round(distance, 4),
                                                             text)])[:5]
                if ratio < CLASH_RATIO:
                    variant.status = "clash"
                    variant.reason = (f"closest contact {distance:.2f} A ({text}), "
                                      f"{ratio:.0%} of the vdW sum")
                elif ratio < TIGHT_RATIO and variant.status == "ok":
                    variant.status = "tight"
            variant.ring_turn = float(degrees)
            variant.ring_turns = [float(sign * degrees) for _, sign in rings]
            if rank[variant.status] > rank[flat.status] and (
                    best is None or rank[variant.status] > rank[best.status]):
                best = variant
            if best is not None and best.status == "ok":
                break
        if best is not None and best.status == "ok":
            break
    if best is None:
        flat.ring_note = (f"ring not turned: no turn up to {MAX_RING_TURN:.0f} degrees "
                          "made it less crowded")
        return flat
    best.ring_note = (f"{len(rings)} ring(s) turned {abs(best.ring_turn):.0f} degrees about "
                      f"the linker axis, to relieve a {flat.status} flat build "
                      f"(flat: {flat.closest[0][0]:.0%} of the vdW sum)")
    return best


def plan(parent: Parent, groups, *, per_linker=(1,), orbits=None):
    """[(group, orbit, pattern, descriptor, name)] for every combination
    asked for, named <stem>_<group>_L<orbit>_m<count>[_<descriptor>][_p<n>]."""
    stem = Path(parent.source).stem if parent.source else "framework"
    jobs = []
    for grp in groups:
        for orbit in parent.orbits:
            if orbits and orbit.number not in orbits:
                continue
            for m in per_linker:
                patterns = orbit.patterns.get(m, [])
                for index, (pattern, descriptor) in enumerate(patterns, start=1):
                    parts = [stem, grp.name, f"L{orbit.number}", f"m{m}"]
                    same = [d for _, d in patterns if d == descriptor]
                    if descriptor and descriptor != "mono":
                        parts.append(descriptor)
                    if len(same) > 1:
                        parts.append(f"p{index}")
                    jobs.append((grp, orbit, pattern, descriptor, "_".join(parts)))
    return jobs


def build_all(parent: Parent, groups, *, per_linker=(1,), orbits=None, progress=None,
              ring_turn=True) -> list:
    """Build and check every variant plan() lists, in memory.
    progress(name, done, total) returning False stops early. ring_turn:
    see build_variant."""
    jobs = plan(parent, groups, per_linker=per_linker, orbits=orbits)
    variants = []
    for number, (grp, orbit, pattern, descriptor, name) in enumerate(jobs, start=1):
        if progress is not None and progress(name, number - 1, len(jobs)) is False:
            break
        variants.append(build_variant(parent, grp, orbit, pattern, descriptor, name=name,
                                      ring_turn=ring_turn))
    return variants


def write_variants(parent: Parent, variants, groups, out_dir, *, per_linker=(1,),
                   keep_clashes=False, ring_turn=True) -> dict:
    """Write built variants as CIFs (clashes only with keep_clashes),
    the parent, functionalization.json and variants.csv; returns the
    provenance record."""
    from pymatgen.io.cif import CifWriter

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        if variant.status != "clash" or keep_clashes:
            path = out / f"{variant.name}.cif"
            CifWriter(variant.structure).write_file(str(path))
            variant.file = path.name
        else:
            variant.file = ""
    if parent.source and Path(parent.source).is_file():
        import shutil

        shutil.copy(parent.source, out / Path(parent.source).name)
    record = provenance(parent, variants, groups, per_linker, ring_turn=ring_turn)
    (out / "functionalization.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    _write_csv(out / "variants.csv", parent, variants)
    return record


def generate(parent: Parent, groups, out_dir, *, per_linker=(1,), orbits=None,
             keep_clashes=False, progress=None, ring_turn=True) -> dict:
    """Build, check and write every variant; returns the provenance
    record (also written as functionalization.json, with variants.csv),
    with the Variant objects under "_variants"."""
    variants = build_all(parent, groups, per_linker=per_linker, orbits=orbits,
                         progress=progress, ring_turn=ring_turn)
    record = write_variants(parent, variants, groups, out_dir, per_linker=per_linker,
                            keep_clashes=keep_clashes, ring_turn=ring_turn)
    record["_variants"] = variants
    return record


def provenance(parent, variants, groups, per_linker, ring_turn=True):
    from thevoid_core.services.report import file_sha256

    structure = parent.structure
    return {
        "software": {"the_void": __version__},
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent": {
            "file": str(parent.source) if parent.source else None,
            "sha256": file_sha256(parent.source) if parent.source else None,
            "formula": structure.composition.formula.replace(" ", ""),
            "sites": len(structure),
            "space_group": parent.space_group,
            "symmetry_operations_used": len(parent.operations),
            "notes": parent.notes,
        },
        "method": {
            "placement": "rigid; first atom on the exterior bisector of the ring angle; "
                         "framework not moved, except substituted rings turned about "
                         "their linker axis when the flat build is crowded (ring_turn); "
                         "rotors turned for the most room (conjugated groups in the "
                         "ring plane unless tight)",
            "groups": {g.name: {"description": g.description, **g.geometry,
                                "rotor_period_deg": g.rotor_period,
                                "prefers_planar": g.prefers_planar} for g in groups},
            "bond_length_source": "Allen et al., J. Chem. Soc., Perkin Trans. 2 (1987) S1",
            "clash_ratio": CLASH_RATIO, "tight_ratio": TIGHT_RATIO, "room_ratio": ROOM_RATIO,
            "hbond_min_A": HBOND_MIN, "hbond_heavy_min_A": HBOND_HEAVY_MIN,
            "max_bend_deg": MAX_BEND, "rotor_step_deg": ROTOR_STEP,
            "contacts": "new atoms vs every atom more than three bonds away, "
                        "as a fraction of the vdW radii sum (the Void's radii)",
            "per_linker": list(per_linker),
            "patterns": "uniform: one pattern per linker orbit, carried to every "
                        "linker of the orbit by one symmetry operation each",
            "ring_turn": ({
                "when": "only if the flat build is tight or clashes",
                "how": "every substituted ring turned rigidly about the line through its "
                       "two outward-bonded (para) atoms, by the smallest angle that leaves "
                       "it not tight (or else not a clash); the same angle on every ring, "
                       "carried by symmetry (reversed under an improper operation); the "
                       "moved ring atoms checked like the new ones",
                "step_deg": RING_TURN_STEP, "max_deg": MAX_RING_TURN,
            } if ring_turn else "off"),
        },
        "orbits": [{"orbit": o.number, "linkers": len(o.linkers), "formula": o.formula,
                    "ring_CH_per_linker": len(o.sites),
                    "patterns": {str(m): [{"hydrogens": [site_name(structure, h) for h in p],
                                           "descriptor": d} for p, d in ps]
                                 for m, ps in o.patterns.items()}}
                   for o in parent.orbits],
        "variants": [{
            "file": getattr(v, "file", ""), "name": v.name, "group": v.group,
            "orbit": v.orbit, "per_linker": v.per_linker, "descriptor": v.descriptor,
            "replaced_hydrogens": [site_name(structure, h) for h in v.replaced],
            "substituents": len(v.replaced), "status": v.status, "reason": v.reason,
            "closest_contacts": [{"ratio": r, "distance_A": d, "pair": t} for r, d, t in v.closest],
            "rotor_angles_deg": v.rotors, "bends_deg": v.bends, "space_group": v.space_group, "formula": v.formula,
            "ring_turn_deg": v.ring_turn, "ring_turns_deg": v.ring_turns, "ring_note": v.ring_note,
        } for v in variants],
    }


def _write_csv(path, parent, variants):
    import csv

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "fn_parent", "fn_group", "fn_orbit", "fn_per_linker",
                         "fn_pattern", "fn_substituents", "fn_status", "fn_closest_ratio",
                         "fn_closest_A", "fn_space_group", "fn_ring_turn_deg"])
        stem = Path(parent.source).name if parent.source else ""
        if stem:
            # The parent itself, copied beside its variants, so a batch
            # run and its figures can set every variant against it.
            writer.writerow([stem, stem, "(parent)", "", 0, "", 0, "parent", "", "",
                             parent.space_group, 0])
        for v in variants:
            if not getattr(v, "file", ""):
                continue
            ratio, distance = (v.closest[0][0], v.closest[0][1]) if v.closest else ("", "")
            writer.writerow([v.file, stem, v.group, v.orbit, v.per_linker, v.descriptor,
                             len(v.replaced), v.status, ratio, distance, v.space_group,
                             v.ring_turn])
