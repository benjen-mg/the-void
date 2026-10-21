"""Structure identity for the shared index (round 64; design section 3).

Four levels, each answering a question the others cannot:

file_sha256 (services/report.file_sha256)
    The exact input bytes. Changes when a file is rewritten, whatever
    the atoms.
framework_key (framework_key)
    A hash of the structure's chemistry and local bonding, independent
    of the cell chosen, the origin, the orientation, the atom order and
    the file's text. It groups candidates: structures with the same
    composition, local bonding and component dimensionalities share a
    key (polymorphs; dia and qtz nets with the same nodes; one framework
    at two temperatures), so it is an address, not an identity.
structure_id (assign_ids, loose match)
    Within one key group, the same framework in any state:
    StructureMatcher at ltol 0.2, stol 0.3, 5 degrees.
determination_id (assign_ids, tight match)
    Within one structure, the same coordinates to within rounding:
    ltol 0.02, stol 0.05, 1 degree. Only entries of one determination
    are replicates whose numbers must agree.

Measured on all 2,664 CoRE MOF 2024 SI structures (the design's
Appendix A; tools/index_identity_survey.py repeats it):

- The key was unchanged by a pymatgen rewrite, a supercell, rotation
  with an origin shift and shuffled atoms, and reduction to the
  primitive cell, in every case tested.
- The structures form 1,604 keys. The loose match splits 52 of the 661
  shared keys, which gives 1,669 structures.

The recipe below is key version 1. It depends on CovalentBondDetector's
bonds (tolerance 0.4 A, floor 0.7 of the radii sum), so a change to the
detector is a change of key version. tests/test_index_identity.py pins
the key of a fixture, so that no such change can pass unnoticed.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass, field
from functools import reduce

KEY_VERSION = 1

#: Weisfeiler-Lehman refinement rounds (key v1).
WL_ROUNDS = 4

#: StructureMatcher tolerances: the same framework in any state, and
#: the same determination (see the module docstring).
LOOSE = dict(ltol=0.2, stol=0.3, angle_tol=5.0)
TIGHT = dict(ltol=0.02, stol=0.05, angle_tol=1.0)


@dataclass(frozen=True)
class FrameworkKey:
    """framework_key and what went into it."""

    key: str
    formula: str
    #: Atoms in the ordered structure the key was computed from.
    n_atoms: int
    #: Periodic dimensionalities of the bonded components, sorted.
    dimensionalities: tuple
    version: int = KEY_VERSION

    def __str__(self):
        return self.key


def _ordered(structure):
    from thevoid_core.services.pore_analysis import major_occupancy_structure

    return major_occupancy_structure(structure)


def framework_key(structure, bonds=None) -> FrameworkKey:
    """The key of a structure (key v1).

    The inputs are the structure with sites under 0.5 occupancy removed,
    as every analysis does, and its covalent bonds. The key is the
    SHA-256 (first 24 hex digits) of three things:

    - the reduced formula;
    - the colour histogram of 4 rounds of Weisfeiler-Lehman refinement
      on the quotient bond graph, starting from the element, with
      counts divided by their greatest common divisor;
    - the set of the bonded components' periodic dimensionalities.

    A supercell's quotient graph covers the cell's, and WL colours do
    not change under a covering, so after dividing by the gcd the
    histogram is the same for any cell. A bond from an atom to its own
    periodic image counts twice in its neighbours, as it joins two
    distinct atoms in any supercell. The dimensionalities are needed
    because a 2D layer and a 3D net can have the same local bonding.

    bonds: CovalentBondDetector's bonds of the ordered structure, if
    already detected.
    """
    from thevoid_core.data.metal_classification import is_metal
    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector
    from thevoid_core.services.mof_analysis import bonded_components

    ordered = _ordered(structure)
    if bonds is None or ordered is not structure:
        bonds = CovalentBondDetector().detect(ordered)
    neighbours = [[] for _ in ordered.sites]
    for bond in bonds:
        if bond.atom1 == bond.atom2:
            neighbours[bond.atom1] += [bond.atom1, bond.atom1]
        else:
            neighbours[bond.atom1].append(bond.atom2)
            neighbours[bond.atom2].append(bond.atom1)
    colours = [site.specie.symbol for site in ordered.sites]
    for _ in range(WL_ROUNDS):
        colours = [
            hashlib.sha1((colours[i] + "|" + ",".join(sorted(colours[j] for j in neighbours[i])))
                         .encode()).hexdigest()[:16]
            for i in range(len(colours))
        ]
    histogram = Counter(colours)
    divisor = reduce(math.gcd, histogram.values()) if histogram else 1
    normalised = sorted((colour, count // divisor) for colour, count in histogram.items())
    dimensionalities = tuple(sorted({c.dimensionality
                                     for c in bonded_components(ordered, bonds, is_metal)}))
    formula = ordered.composition.reduced_formula
    text = (f"v{KEY_VERSION}|{formula}|" + ";".join(f"{c}:{n}" for c, n in normalised)
            + "|dims=" + ",".join(map(str, dimensionalities)))
    return FrameworkKey(hashlib.sha256(text.encode()).hexdigest()[:24], formula, len(ordered),
                        dimensionalities)


# ----------------------------------------------------------------------
# Matching within a key group
# ----------------------------------------------------------------------


def _matcher(tolerances):
    from pymatgen.analysis.structure_matcher import StructureMatcher

    return StructureMatcher(primitive_cell=True, scale=False, attempt_supercell=True,
                            **tolerances)


def same_structure(a, b) -> bool:
    """The same framework, in any state (the loose match)."""
    return bool(_matcher(LOOSE).fit(_ordered(a), _ordered(b)))


def same_determination(a, b) -> bool:
    """The same coordinates to within rounding (the tight match)."""
    return bool(_matcher(TIGHT).fit(_ordered(a), _ordered(b)))


@dataclass
class Member:
    """One item to place: an id that fixes its order (an entry_id), the
    file's SHA-256, and a way to get the structure."""

    id: str
    file_sha256: str
    key: str
    #: The structure, or a callable returning it (loaded only if a match
    #: is needed). None, or a callable returning None: not available
    #: here (a CSD-derived file another member added).
    structure: object = field(repr=False, default=None)

    def load(self):
        if callable(self.structure):
            self.structure = self.structure()
        return self.structure


@dataclass
class Placement:
    id: str
    key: str
    structure_id: str
    determination_id: str


class MatchCache:
    """Decisions already made, by the pair of file SHA-256s and the
    level, so a rebuild only fits new pairs. Keys are order-free."""

    def __init__(self, decisions: dict | None = None):
        self.decisions = dict(decisions or {})
        self.fits = 0
        #: Pairs that needed a fit but lacked a structure: treated as
        #: different (never merged on a guess), and reported.
        self.undecided: set = set()
        #: Decisions made by this cache, for saving.
        self.new: dict = {}

    @staticmethod
    def _key(level, a, b):
        return f"{level}:{min(a, b)}:{max(a, b)}"

    def decide(self, level, a: Member, b: Member) -> bool:
        if a.file_sha256 == b.file_sha256:
            return True
        key = self._key(level, a.file_sha256, b.file_sha256)
        if key not in self.decisions:
            first, second = a.load(), b.load()
            if first is None or second is None:
                self.undecided.add(key)
                return False
            self.fits += 1
            tolerances = LOOSE if level == "structure" else TIGHT
            self.decisions[key] = self.new[key] = bool(
                _matcher(tolerances).fit(_ordered(first), _ordered(second)))
        return self.decisions[key]


def assign_ids(members, cache: MatchCache | None = None) -> list[Placement]:
    """structure_id and determination_id for every member.

    Members are placed in id order, whatever order they arrive in, so
    everyone building the same entries gets the same ids. Within a key,
    a member joins the first structure whose representative (its first
    member) it matches loosely, or starts the next one. Within that
    structure, it joins the first determination it matches tightly.

    The ids are `<key>.<n>` and `<key>.<n>.<m>`, with n and m counted
    from 1. Identical files never need a fit.
    """
    cache = cache or MatchCache()
    groups: dict[str, list] = {}
    out = []
    for member in sorted(members, key=lambda m: m.id):
        structures = groups.setdefault(member.key, [])
        for n, (representative, determinations) in enumerate(structures, start=1):
            if cache.decide("structure", representative, member):
                break
        else:
            structures.append((member, []))
            n, determinations = len(structures), structures[-1][1]
        for m, first in enumerate(determinations, start=1):
            if cache.decide("determination", first, member):
                break
        else:
            determinations.append(member)
            m = len(determinations)
        out.append(Placement(member.id, member.key, f"{member.key}.{n}",
                             f"{member.key}.{n}.{m}"))
    order = {m.id: i for i, m in enumerate(members)}
    return sorted(out, key=lambda p: order[p.id])
