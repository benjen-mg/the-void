"""Round 71: ring turns in functionalisation.

A rigid build puts a new group on a ring that stays where it was. Real
MOFs make room for a crowded group by twisting the ring against its
carboxylates: RASPA2's IRMOF-2 model has its -Br rings at 25 degrees,
and the real MIL-47(V)-Br (CoRE SI) at 19 and 26. So when the flat build
is tight or clashes, each substituted ring is now turned about its
linker axis (the line through its two outward-bonded atoms) by the
smallest angle, in 5-degree steps, that clears it. The same turn goes to
every linker of the pattern, carried by symmetry: an improper operation
turns the ring the other way. The ring atoms that move are checked like
the new ones.
"""

import math
import os
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services import functionalize as fz  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"         # MIL-53(Cr)
MIL47_BR = FIXTURES / "coremof" / "2019[V][bpq]3[ASR]1.cif"        # real MIL-47(V)-Br
NI_NH2 = FIXTURES / "coremof" / "2023[Ni][dia]3[ASR]1.cif"         # a ring already -NH2
ZN_P21C = FIXTURES / "coremof" / "2017[ZnSi][nan]3[ASR]1.cif"      # P2_1/c
RASPA_STRUCTURES = os.environ.get("RASPA_STRUCTURES")


def _load(path):
    from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PymatgenCIFLoader().load(str(path)).structure


@pytest.fixture(scope="module")
def cr_parent():
    return fz.analyze_parent(_load(CR_BDC), source=str(CR_BDC), max_per_linker=2)


def _plane_normal(points):
    points = np.asarray(points)
    return np.linalg.svd(points - points.mean(axis=0))[2][2]


def _twists(structure, bonds):
    """Angle between each 6-ring's plane and each carboxylate (O-C-O)
    plane bonded to it, degrees."""
    adjacency = fz._adjacency(len(structure), bonds)
    symbols = [s.specie.symbol for s in structure.sites]
    lattice = structure.lattice.matrix

    def at(site, image):
        return (structure.frac_coords[site] + np.array(image)) @ lattice

    out = []
    for c, element in enumerate(symbols):
        oxygens = [(w, im) for w, im in adjacency[c] if symbols[w] == "O"]
        carbons = [(w, im) for w, im in adjacency[c] if symbols[w] == "C"]
        if element != "C" or len(oxygens) != 2 or len(carbons) != 1:
            continue
        ring_c, cell = carbons[0]
        ring = fz._ring_path(ring_c, adjacency, symbols)
        if not ring or len(ring) != 6:
            continue
        n1 = _plane_normal([at(w, tuple(a + b for a, b in zip(cell, im))) for w, im in ring])
        n2 = _plane_normal([at(c, (0, 0, 0))] + [at(w, im) for w, im in oxygens])
        out.append(math.degrees(math.acos(min(1.0, abs(float(n1 @ n2))))))
    return out


# ----------------------------------------------------------------------
# Which rings turn, and about what
# ----------------------------------------------------------------------


def test_a_terephthalate_ring_turns_about_its_para_axis(cr_parent):
    h = cr_parent.orbits[0].patterns[1][0][0][0]
    body, why = fz.ring_body(cr_parent, h)
    assert why == "" and body is not None
    symbols = [s.specie.symbol for s in cr_parent.structure.sites]
    # The four other ring carbons and their four hydrogens move; the two
    # carbons bonded to the carboxylates are the axis.
    assert sorted(symbols[site] for site, _ in body.moved) == ["C"] * 4 + ["H"] * 4
    assert symbols[body.first[0]] == symbols[body.second[0]] == "C"
    assert body.carbon in body.key


def test_a_ring_with_more_than_two_outward_bonds_does_not_turn():
    """This Ni MOF's rings already carry -NH2: three ring atoms bond
    outwards, so there is no single axis to turn about."""
    parent = fz.analyze_parent(_load(NI_NH2), max_per_linker=1)
    body, why = fz.ring_body(parent, next(iter(parent.sites)))
    assert body is None and "3 ring atoms bond outwards" in why


def test_a_turned_parent_is_the_same_framework_with_its_rings_turned(cr_parent):
    """Bond lengths kept, the axis atoms and everything off the ring
    unmoved, the twist against the carboxylates grown by the turn, and
    the stored bonds the ones a bond detector finds in the result."""
    from thevoid_core.services.bonds.bond_model import canonical_key
    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

    h = cr_parent.orbits[0].patterns[1][0][0][0]
    body, _ = fz.ring_body(cr_parent, h)
    before = _twists(cr_parent.structure, cr_parent.bonds)
    turned = fz.turned_parent(cr_parent, [(body, 30.0)])
    moved = body.key
    for i, (a, b) in enumerate(zip(cr_parent.structure.sites, turned.structure.sites)):
        if i not in moved:
            assert np.allclose(a.coords, b.coords, atol=1e-9) or \
                np.allclose(np.mod(a.frac_coords, 1), np.mod(b.frac_coords, 1), atol=1e-9)
    for old, new in zip(cr_parent.bonds, turned.bonds):
        assert (old.atom1, old.atom2) == (new.atom1, new.atom2)
        v_old = (cr_parent.structure.frac_coords[old.atom2] + old.image
                 - cr_parent.structure.frac_coords[old.atom1]) @ cr_parent.structure.lattice.matrix
        v_new = (turned.structure.frac_coords[new.atom2] + new.image
                 - turned.structure.frac_coords[new.atom1]) @ turned.structure.lattice.matrix
        assert np.linalg.norm(v_new) == pytest.approx(np.linalg.norm(v_old), abs=1e-6)
    detected = {canonical_key(b.atom1, b.atom2, b.image)
                for b in CovalentBondDetector().detect(turned.structure)}
    assert detected == {canonical_key(b.atom1, b.atom2, b.image) for b in turned.bonds}
    after = _twists(turned.structure, turned.bonds)
    # Two of the four carboxylates sit on the turned ring: their twist
    # goes from 4.6 to 30.3 degrees (the ring was not quite coplanar
    # with them to start with); the other two do not move.
    moved_twists = [a for a, b in zip(after, before) if abs(a - b) > 1e-6]
    assert len(moved_twists) == 2 and moved_twists == pytest.approx([30.3, 30.3], abs=0.1)
    assert sorted(before) == pytest.approx([4.56] * 4, abs=0.01)


# ----------------------------------------------------------------------
# Only to relieve crowding, and by the smallest angle
# ----------------------------------------------------------------------


def test_a_variant_with_room_is_not_turned(cr_parent):
    orbit = cr_parent.orbits[0]
    variant = fz.build_variant(cr_parent, fz.GROUPS["NH2"], orbit, orbit.patterns[1][0][0])
    assert variant.status == "ok" and variant.ring_turn == 0.0 and variant.ring_note == ""


def test_a_clash_is_relieved_by_the_smallest_turn(cr_parent):
    """-Cl in Cr-BDC's narrow channel clashed flat (a contact at 73% of
    the vdW sum, round 59's example). Turned 15 degrees it clears, and
    10 degrees does not."""
    orbit = cr_parent.orbits[0]
    pattern = orbit.patterns[1][0][0]
    flat = fz.build_variant(cr_parent, fz.GROUPS["Cl"], orbit, pattern, ring_turn=False)
    assert flat.status == "clash"
    variant = fz.build_variant(cr_parent, fz.GROUPS["Cl"], orbit, pattern)
    assert variant.status == "ok" and abs(variant.ring_turn) == 15.0
    assert "turned 15 degrees" in variant.ring_note and "clash" in variant.ring_note
    rings, _ = fz._ring_turn_plan(cr_parent, orbit, pattern, variant.replaced)
    for degrees in (5.0, 10.0):
        smaller = fz._build_flat(fz.turned_parent(cr_parent, [(b, s * degrees) for b, s in rings]),
                                 fz.GROUPS["Cl"], orbit, pattern)
        assert smaller.status != "ok"


def test_the_turned_variant_is_checked_independently(cr_parent):
    """Two -CN facing across the channel (meta) overlap flat; turned
    25 degrees they clear. A plain all-pairs check of the written
    structure - every pair more than three bonds apart, by the bond
    detector - finds the same closest contact the build reports, so the
    moved ring atoms were checked too."""
    from collections import deque

    from thevoid_core.data.element_radii import ElementRadiusProvider
    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

    orbit = cr_parent.orbits[0]
    meta = [p for p, d in orbit.patterns[2] if d == "meta"][0]
    variant = fz.build_variant(cr_parent, fz.GROUPS["CN"], orbit, meta)
    assert variant.status == "ok" and abs(variant.ring_turn) == 25.0
    s = variant.structure
    adjacency = fz._adjacency(len(s), CovalentBondDetector().detect(s))
    radii = ElementRadiusProvider()
    worst = math.inf
    for i in range(len(s)):
        near = {(i, (0, 0, 0)): 0}
        queue = deque(near)
        while queue:
            u, cell = queue.popleft()
            if near[(u, cell)] < 3:
                for w, image in adjacency[u]:
                    node = (w, tuple(a + b for a, b in zip(cell, image)))
                    if node not in near:
                        near[node] = near[(u, cell)] + 1
                        queue.append(node)
        for neighbour in s.get_neighbors(s[i], 3.6):
            key = (neighbour.index, tuple(int(round(v)) for v in neighbour.image))
            if key in near:
                continue
            worst = min(worst, neighbour.nn_distance / (
                radii.get_vdw_radius(s[i].specie.symbol)
                + radii.get_vdw_radius(s[neighbour.index].specie.symbol)))
    assert worst == pytest.approx(variant.closest[0][0], abs=2e-3)


# ----------------------------------------------------------------------
# The same turn on every linker, by symmetry
# ----------------------------------------------------------------------


def test_the_turn_is_carried_by_symmetry():
    """A Zn framework in P2_1/c (CoRE SI): its four rings of one orbit
    are related by proper and improper operations. Turned by symmetry
    (the other way under the improper ones), the framework keeps
    P2_1/c; turned all the same way, it drops to P2_1."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    parent = fz.analyze_parent(_load(ZN_P21C), max_per_linker=1)
    orbit = parent.orbits[0]
    pattern = orbit.patterns[1][0][0]
    rings, why = fz._ring_turn_plan(parent, orbit, pattern, fz.pattern_sites(parent, orbit, pattern))
    assert why == "" and sorted(sign for _, sign in rings) == [-1, -1, 1, 1]

    def space_group(turns):
        return SpacegroupAnalyzer(fz.turned_parent(parent, turns).structure,
                                  symprec=fz.SYMPREC).get_space_group_symbol()

    assert parent.space_group == "P2_1/c"
    assert space_group([(body, sign * 20.0) for body, sign in rings]) == "P2_1/c"
    assert space_group([(body, 20.0) for body, _ in rings]) == "P2_1"


def test_when_no_turn_helps_the_flat_variant_stands():
    """-Br on that framework clashes flat and at every turn up to 90
    degrees: the flat variant is kept, rejected, with the reason."""
    parent = fz.analyze_parent(_load(ZN_P21C), max_per_linker=1)
    orbit = parent.orbits[0]
    variant = fz.build_variant(parent, fz.GROUPS["Br"], orbit, orbit.patterns[1][0][0])
    assert variant.status == "clash" and variant.ring_turn == 0.0
    assert "no turn up to 90 degrees" in variant.ring_note


def test_improper_operations_are_recognised():
    from pymatgen.core import Lattice, Structure

    cubic = Structure(Lattice.cubic(4.0), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    _, perms, handedness = fz.symmetry_operations_with_handedness(cubic)
    assert len(perms) == 48 and handedness.count(-1) == 24


# ----------------------------------------------------------------------
# Against real twisted rings
# ----------------------------------------------------------------------


def test_real_bromo_frameworks_twist_their_rings():
    """The real MIL-47(V)-Br twists its rings 19 and 26 degrees against
    the carboxylates. The unsubstituted Cr-BDC fixture's are within 5
    degrees of flat."""
    parent = fz.analyze_parent(_load(MIL47_BR), max_per_linker=1)
    twists = _twists(parent.structure, parent.bonds)
    assert sorted({round(t) for t in twists}) == [19, 26]
    cr = fz.analyze_parent(_load(CR_BDC), max_per_linker=1)
    assert max(_twists(cr.structure, cr.bonds)) < 5.0


@pytest.mark.skipif(not RASPA_STRUCTURES, reason="needs RASPA2's structures (RASPA_STRUCTURES)")
def test_mil47_and_irmof1_with_bromine_turn_less_than_the_real_ones():
    """MIL-47 + -Br clashes flat and clears at 15 degrees; IRMOF-1 + -Br
    is tight flat and clears at 15. Real ones twist further (MIL-47-Br
    19-26, RASPA2's IRMOF-2 25): the build takes the smallest turn that
    clears, not the relaxed one."""
    root = Path(RASPA_STRUCTURES) / "mofs" / "cif"
    for name, flat_status in (("MIL-47", "clash"), ("IRMOF-1", "tight")):
        parent = fz.analyze_parent(_load(root / f"{name}.cif"), max_per_linker=1)
        orbit = parent.orbits[0]
        pattern = orbit.patterns[1][0][0]
        assert fz.build_variant(parent, fz.GROUPS["Br"], orbit, pattern,
                                ring_turn=False).status == flat_status
        variant = fz.build_variant(parent, fz.GROUPS["Br"], orbit, pattern)
        assert variant.status == "ok" and abs(variant.ring_turn) == 15.0
    model = fz.analyze_parent(_load(root / "IRMOF-2.cif"), max_per_linker=1)
    assert {round(t) for t in _twists(model.structure, model.bonds)} == {25}
