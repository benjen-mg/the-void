"""Round 64: structure identity for the shared index
(services/index/identity.py; docs/shared-index-design.md section 3).

The key has to be stable under everything that does not change the
crystal: a rewritten file, another cell, another origin or orientation,
another atom order. The matches have to split what the key merges: two
nets with the same local bonding, and two determinations of one
framework. The pairs used here are the cases the round-62 survey of
all 2,664 CoRE MOF 2024 SI structures turned up
(tests/fixtures/identity/README.txt).
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402

from thevoid_core.services.index.identity import (  # noqa: E402
    KEY_VERSION,
    MatchCache,
    Member,
    assign_ids,
    framework_key,
    same_determination,
    same_structure,
)

FIXTURES = Path(__file__).parent / "fixtures"
IDENTITY = FIXTURES / "identity"
COREMOF = sorted((FIXTURES / "coremof").glob("*.cif"))


from tests.sample_structures import ZN_DIA  # noqa: E402

def _load(name):
    return Structure.from_file(IDENTITY / f"{name}.cif")


def _moved(structure, seed=0):
    """Rotated in space, origin shifted, atoms shuffled."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    order = rng.permutation(len(structure))
    shift = rng.uniform(0, 1, 3)
    return Structure(Lattice(structure.lattice.matrix @ q.T),
                     [structure[i].species for i in order],
                     [(structure[i].frac_coords + shift) % 1 for i in order])


# ----------------------------------------------------------------------
# The key
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", COREMOF + [ZN_DIA], ids=lambda p: p.stem)
def test_the_key_does_not_depend_on_the_file(path):
    structure = Structure.from_file(path)
    key = framework_key(structure).key
    assert framework_key(Structure.from_str(str(CifWriter(structure)), fmt="cif")).key == key
    assert framework_key(_moved(structure)).key == key
    if len(structure) <= 200:
        assert framework_key(structure * (1, 2, 1)).key == key


def test_the_key_does_not_depend_on_the_cell():
    """A conventional cell and its primitive cell share a key."""
    structure = _load("2020[Zn][qtz]3[ASR]1")
    primitive = structure.get_primitive_structure(tolerance=0.01)
    doubled = structure * (2, 1, 1)
    assert framework_key(doubled).key == framework_key(structure).key
    assert framework_key(primitive).key == framework_key(structure).key


def test_the_key_is_pinned():
    """The recipe is key version 1. If this fails, the key changed (a
    change to the WL rounds, the hashing, or the bond detector): that is
    a new key version, and every stored key goes stale."""
    key = framework_key(Structure.from_file(ZN_DIA))
    assert KEY_VERSION == 1 and key.version == 1
    assert key.key == "a1fd89898fd9e97b9783369b"
    assert (key.formula, key.n_atoms, key.dimensionalities) == ("ZnH10C12(NO)4", 124, (3,))


def test_what_goes_into_the_key():
    key = framework_key(Structure.from_file(COREMOF[0]))
    assert len(key.key) == 24 and str(key) == key.key
    assert key.dimensionalities == (3,)
    assert key.n_atoms > 0


def test_the_key_tells_a_layer_from_a_net():
    """Round 62: without the dimensionalities, a 2D and a 3D Co
    structure from one paper, with the same formula and local bonding,
    shared a key."""
    layer = framework_key(_load("0000[Co][nan]2[ASR]1"))
    net = framework_key(_load("0000[Co][nan]3[ASR]10"))
    assert layer.formula == net.formula
    assert 2 in layer.dimensionalities and 3 in net.dimensionalities
    assert layer.key != net.key


def test_a_change_of_chemistry_changes_the_key():
    structure = Structure.from_file(COREMOF[0])
    other = structure.copy()
    other.replace_species({"Cr": "V"})
    assert framework_key(other).key != framework_key(structure).key


# ----------------------------------------------------------------------
# The matches
# ----------------------------------------------------------------------


def test_the_key_groups_and_the_match_splits():
    """dia and qtz Zn nets from one paper share a key (same formula,
    same local bonding, both 3D) but are not the same structure."""
    dia, qtz = _load("2020[Zn][dia]3[ASR]1"), _load("2020[Zn][qtz]3[ASR]1")
    assert framework_key(dia).key == framework_key(qtz).key
    assert not same_structure(dia, qtz)


def test_one_framework_two_determinations():
    """Two steps of a Cu sql series from one paper: the same structure,
    not the same determination (density differs by 0.14%)."""
    a, b = _load("2022[Cu][sql]2[ASR]2"), _load("2022[Cu][sql]2[ASR]4")
    assert framework_key(a).key == framework_key(b).key
    assert same_structure(a, b)
    assert not same_determination(a, b)


def test_copies_are_one_determination():
    """CoRE's ASR and FSR copies of a structure with nothing to remove
    differ in bytes, not atoms; a rewrite and a supercell are also the
    same determination."""
    asr, fsr = _load("2020[Co][nia]3[ASR]1"), _load("2020[Co][nia]3[FSR]1")
    assert same_determination(asr, fsr)
    rewritten = Structure.from_str(str(CifWriter(asr)), fmt="cif")
    assert same_determination(asr, rewritten)
    assert same_determination(asr, asr * (1, 1, 2))


# ----------------------------------------------------------------------
# Assigning ids
# ----------------------------------------------------------------------


def _members():
    from thevoid_core.services.report import file_sha256

    out = []
    for name in ("2020[Zn][dia]3[ASR]1", "2020[Zn][qtz]3[ASR]1", "2022[Cu][sql]2[ASR]2",
                 "2022[Cu][sql]2[ASR]4", "2020[Co][nia]3[ASR]1", "2020[Co][nia]3[FSR]1"):
        path = IDENTITY / f"{name}.cif"
        structure = Structure.from_file(path)
        out.append(Member(name, file_sha256(path), framework_key(structure).key, structure))
    return out


@pytest.fixture(scope="module")
def members():
    return _members()


def test_ids_for_a_small_index(members):
    placed = {p.id: p for p in assign_ids(members)}
    dia, qtz = placed["2020[Zn][dia]3[ASR]1"], placed["2020[Zn][qtz]3[ASR]1"]
    assert dia.key == qtz.key and dia.structure_id != qtz.structure_id
    cu2, cu4 = placed["2022[Cu][sql]2[ASR]2"], placed["2022[Cu][sql]2[ASR]4"]
    assert cu2.structure_id == cu4.structure_id and cu2.determination_id != cu4.determination_id
    asr, fsr = placed["2020[Co][nia]3[ASR]1"], placed["2020[Co][nia]3[FSR]1"]
    assert asr.determination_id == fsr.determination_id
    # Numbered from 1 in id order: dia (the smaller id) is .1, qtz .2.
    assert dia.structure_id.endswith(".1") and qtz.structure_id.endswith(".2")
    assert cu2.determination_id.endswith(".1.1") and cu4.determination_id.endswith(".1.2")


def test_ids_do_not_depend_on_arrival_order(members):
    forward = {p.id: (p.structure_id, p.determination_id) for p in assign_ids(members)}
    backward = assign_ids(list(reversed(members)))
    assert [p.id for p in backward] == [m.id for m in reversed(members)]  # input order kept
    assert {p.id: (p.structure_id, p.determination_id) for p in backward} == forward


def test_the_cache_saves_every_fit_the_second_time(members):
    cache = MatchCache()
    first = assign_ids(members, cache)
    fits = cache.fits
    assert fits > 0
    again = MatchCache(cache.decisions)
    assert [(p.structure_id, p.determination_id) for p in assign_ids(members, again)] == \
        [(p.structure_id, p.determination_id) for p in first]
    assert again.fits == 0


def test_identical_files_need_no_fit():
    structure = _load("2020[Co][nia]3[ASR]1")
    key = framework_key(structure).key

    def never():
        raise AssertionError("loaded")

    same = [Member("a", "sha-x", key, never), Member("b", "sha-x", key, never)]
    placed = assign_ids(same)
    assert placed[0].determination_id == placed[1].determination_id


def test_structures_are_loaded_only_when_a_match_is_needed():
    loaded = []

    def loader(name):
        def load():
            loaded.append(name)
            return _load(name)
        return load

    names = ("2020[Zn][dia]3[ASR]1", "2022[Cu][sql]2[ASR]2")
    keys = {n: framework_key(_load(n)).key for n in names}
    assign_ids([Member(n, n, keys[n], loader(n)) for n in names])
    assert loaded == []  # different keys: nothing to compare
