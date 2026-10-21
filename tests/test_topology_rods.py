"""Round 70: nets of rod-SBU frameworks.

Before this round a framework whose inorganic SBU is infinite - the
rods of MIL-53, MIL-47 and MOF-74 - was reported "not identified:
rod/layer SBU". CoRE MOF 2024 names these nets with CrystalNets, which
splits a rod at its metals. The Void now does the same. Each metal is a
vertex. Two metals are joined when they share an atom (a mu-O or mu-OH,
or one carboxylate oxygen bridging both). A carboxylate spanning two
metals syn-syn does not join them. MIL-53 is then bpq, CoRE's
`topology`; with every carboxylate its own vertex it is rna, CoRE's
`topology_all_node`.

Every expected net here is CoRE's own label for that file.
"""

from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"   # MIL-53(Cr)
V_BDC = FIXTURES / "coremof" / "2019[V][bpq]3[ASR]1.cif"     # MIL-47(V)-Br
MN_AZOLATE = FIXTURES / "coremof" / "2019[Mn][bpq]3[ASR]1.cif"
ZN_PTS = FIXTURES / "rods" / "2019[Zn][pts]3[ASR]3.cif"
ZN_FGL = FIXTURES / "rods" / "2020[Zn][fgl]3[ASR]1.cif"
CU_TFZ = FIXTURES / "rods" / "2024[Cu][tfz]3[ASR]1.cif"


def _topology(path):
    from pymatgen.core import Structure

    from thevoid_core.services.topology import analyze_topology

    return analyze_topology(Structure.from_file(path))


@pytest.mark.parametrize("path", [CR_BDC, V_BDC], ids=["MIL-53(Cr)", "MIL-47(V)"])
def test_mil53_is_bpq_and_its_all_node_net_rna(path):
    result = _topology(path)
    assert result.status == "identified" and result.summary == "bpq (4,6)-c"
    kinds = {k["kind"]: (k["formula"], k["connectivity"]) for k in result.vertex_kinds}
    # The whole linker (BDC, carboxylates included) is one 4-c vertex;
    # each metal of the rod a 6-c one.
    assert kinds["linker"][0].startswith("C8") and kinds["linker"][1] == 4
    assert kinds["sbu"][1] == 6 and kinds["sbu"][0] in ("Cr", "V")
    rod = result.rod_sbus[0]
    assert rod["dimensionality"] == 1 and rod["metals"] == 2
    assert result.representation == "single-node, infinite SBUs split at their metals"
    assert any("split at its metals" in note for note in result.notes)
    assert result.all_node["rcsr"] == "rna" and result.all_node["summary"] == "rna (3,6)-c"


def test_a_carboxylate_oxygen_shared_by_two_metals_joins_them():
    """Zn rods held together by carboxylates, one O of which bridges
    two Zn: the Zn are joined along the rod, and the net is CoRE's
    pts-x (6-c metals); without the joins it would be pts (4-c)."""
    result = _topology(ZN_PTS)
    assert result.rcsr == "pts-x"
    assert {k["connectivity"] for k in result.vertex_kinds if k["kind"] == "sbu"} == {6}


def test_a_syn_syn_carboxylate_alone_does_not_join_metals():
    """Here the rod's Zn are spanned only by O-C-O bridges, no shared
    atom: they are not joined, and the net is CoRE's fgl (ggl all-node)."""
    result = _topology(ZN_FGL)
    assert result.summary == "fgl 4-c"
    assert result.all_node["rcsr"] == "ggl"


def test_an_inorganic_rod_is_joined_through_its_oxygens():
    """Cu2O2 rods of mu-O: the O are bonded only to Cu, so they are used
    up by the Cu-Cu joins. CoRE: tfz, both ways."""
    result = _topology(CU_TFZ)
    assert result.rcsr == "tfz" and result.all_node["rcsr"] == "tfz"
    assert [(r["formula"], r["count"]) for r in result.rod_sbus] == [("Cu2O2", 3)]
    assert all(k["kind"] == "linker" or k["formula"] == "Cu" for k in result.vertex_kinds)


def test_the_all_node_net_splits_only_carboxylates():
    """A known limit, recorded: the all-node net makes carboxylates
    vertices, not azolates. This Mn azolate framework is bpq as CoRE
    says, but CoRE's all-node rna is not reproduced (bpq again)."""
    result = _topology(MN_AZOLATE)
    assert result.rcsr == "bpq" and result.all_node["rcsr"] == "bpq"


def test_the_report_records_rods_and_the_all_node_net():
    from thevoid_core.services.topology import topology_to_dict

    d = topology_to_dict(_topology(CR_BDC))
    assert d["rcsr"] == "bpq" and d["all_node"]["rcsr"] == "rna"
    assert d["rod_sbus"] == [{"formula": "C4Cr2O10", "dimensionality": 1, "metals": 2,
                              "count": 1}]
    assert "split at their metals" in d["representation"]


def test_a_finite_sbu_framework_is_unchanged():
    """No rod, no split: no rod record, no all-node net, and the
    single-node representation as before."""
    from pymatgen.core import Structure

    from thevoid_core.services.topology import analyze_topology

    from tests.sample_structures import ZN_DIA

    result = analyze_topology(Structure.from_file(ZN_DIA))
    assert result.rcsr == "dia" and result.rod_sbus == [] and result.all_node is None
    assert result.representation == "single-node"


def test_the_coremof_comparison_now_agrees_on_mil53():
    """Before, the Compare window set "not identified: rod/layer SBU"
    against CoRE's bpq."""
    from pymatgen.core import Structure

    from thevoid_core.services.reference import coremof
    from thevoid_core.services.reference.compare import compare_with_coremof

    structure = Structure.from_file(CR_BDC)
    match = coremof.find_matches(structure, path=CR_BDC)[0]
    comparison = compare_with_coremof(structure, match, samples_per_atom=100, oms=False)
    row = comparison.row("topology")
    assert row.agrees is True and row.void.startswith("bpq") and row.reference == "bpq"


def test_a_net_too_large_for_the_exact_solve_is_reported_not_hung(monkeypatch):
    """Splitting a three-periodic inorganic framework at its metals can
    give a net of hundreds of vertices whose placement has large
    denominators; the exact solve then takes hours (CoRE SI
    2019[YK][nan]3[ION]1, 384 vertices). Above EXACT_SOLVE_LIMIT the
    net is reported as too complex instead."""
    from thevoid_core.services.topology import net as N

    cube = N.PeriodicNet.build(1, [(0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1))], 3)
    monkeypatch.setattr(N, "_solve_fast_then_verify", lambda *a: None)
    monkeypatch.setattr(N, "EXACT_SOLVE_LIMIT", 0)
    supercell = N.PeriodicNet.build(2, [(0, 1, (0, 0, 0)), (1, 0, (1, 0, 0)),
                                        (0, 0, (0, 1, 0)), (1, 1, (0, 1, 0)),
                                        (0, 0, (0, 0, 1)), (1, 1, (0, 0, 1))], 3)
    with pytest.raises(ValueError, match="too slow"):
        N.barycentric_placement(supercell)
    assert N.barycentric_placement(cube)  # one vertex: nothing to solve


def test_old_entries_are_stale_only_where_the_net_could_change():
    """In the shared index, an entry from before 0.9.10 is stale in
    topology only if its net was a rod "not identified" or "not in the
    reference library" (nine nets joined the library); a named net
    stands."""
    from thevoid_core.services.index import staleness

    changes = [c for c in staleness.load_changes() if c["version"] == "0.9.10"]
    assert len(changes) == 1 and changes[0]["columns"] == ["topology"]

    def stale(topology):
        entry = {"entry_id": "e", "structure": {"file_name": "a.cif", "file_sha256": "s"},
                 "analysis": {"software": {"the_void": "0.9.9"}, "config": {},
                              "summary": {"topology": topology}}}
        return staleness.staleness_of(entry, changes, "0.9.10").stale

    assert stale("not identified: rod/layer SBU")
    assert stale("(4,6)-c net, not in the reference library")
    assert not stale("pcu 6-c") and not stale("not identified: unstable net")


def test_an_unnamed_split_rod_net_counts_as_no_net_for_the_models():
    """CoRE names a rod framework's net only when RCSR has it: of the
    281 CoRE SI rod frameworks whose split net the Void cannot name,
    272 have no net in CoRE. For the property models' descriptors such
    a net is therefore "no net" (every net feature 0), as it was before
    rods were split, while a named one (bpq) is that net."""
    from thevoid_core.services.ml.descriptors import net_code, net_features

    text = "(6,8)-c net of split rod SBUs, not in the reference library"
    assert net_code(text) == "" and not any(net_features(text).values())
    assert net_code("(3,4)-c net, not in the reference library") == "other"
    assert net_features("bpq (4,6)-c")["net_bpq"] == 1.0
