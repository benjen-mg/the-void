"""Round 56: the bundled CoRE MOF 2024 index, finding a structure in it,
and comparing the Void's numbers with it (services/reference/coremof.py,
compare.py).

The comparison is only a test of the Void if both sides use the same
convention, so it recomputes with Zeo++'s default radii (CCDC) and
CoRE's 1.655 A probe. Those two facts were established in round 56 by
reproducing the database with Zeo++ 0.3 itself (ASA within sampling
noise at 1.655 A and not at 1.86 A; LCD and PLD to the fifth decimal
with -ha and the default radii). What is checked here:

* the index is the whole computation-ready set, with its attribution;
* the radius table is exactly Zeo++'s (tests/fixtures/zeopp_default_radii.json,
  read back from Zeo++ for every element H-Cm);
* matching finds the open Zn-dia sample without a name, and in a
  supercell, and refuses a wrong structure that borrows a name;
* the comparison on the sample and a CoRE SI structure agrees where the
  methods are the same.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from tests.sample_structures import ZN_DIA  # noqa: E402
from thevoid_core.data.element_radii import CCDCRadiusProvider  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.reference import coremof  # noqa: E402
from thevoid_core.services.reference.compare import (  # noqa: E402
    compare_with_coremof,
    comparison_to_dict,
)

FIXTURES = Path(__file__).parent / "fixtures"
CR_BPQ = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"
TASYAR_ID = "2021[Zn][dmc]3[ASR]1"
SAMPLE_ID = "2020[Zn][dia]3[ASR]1"


def _unnamed_sample(tmp_path):
    """The sample under a name that is in no index: file and data block."""
    path = tmp_path / "sample.cif"
    path.write_text(ZN_DIA.read_text().replace(f"data_{SAMPLE_ID}", "data_sample"))
    return path


def _load(path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PymatgenCIFLoader().load(str(path)).structure


# ----------------------------------------------------------------------
# The index
# ----------------------------------------------------------------------


def test_the_index_is_the_whole_computation_ready_set_with_attribution():
    index = coremof.load_index()
    assert len(index) == 17202
    subsets = {s: sum(1 for e in index.entries if e.subset == s) for s in ("ASR", "FSR", "Ion")}
    assert subsets == {"ASR": 8857, "FSR": 7635, "Ion": 710}
    assert sum(e.downloadable for e in index.entries) == 2664

    source = coremof.index_info()
    assert source["licence"] == "CC-BY-4.0"
    assert source["doi"] == "10.1016/j.matt.2025.102140"
    assert len(source["sha256"]) == 64 and len(source["si_zip_sha256"]) == 64

    entry = index.by_id[TASYAR_ID]
    assert entry.csd_refcode == "TASYAR" and entry.topology == "dmc"
    assert entry.pld_A == pytest.approx(2.84946) and entry.n_atoms == 44


def test_the_radius_table_is_zeopps_own():
    recorded = json.loads((FIXTURES / "zeopp_default_radii.json").read_text())
    provider = CCDCRadiusProvider()
    assert len(recorded) == 96
    assert {e: provider.get_vdw_radius(e) for e in recorded} == recorded


def test_search_filters_combine():
    hits = coremof.search(pld_min=6, topology="PCU")
    assert hits and all(e.pld_A >= 6 and e.topology == "pcu" for e in hits)
    open_cu = coremof.search(metals=["Cu"], has_oms=True, downloadable_only=True, limit=5)
    assert len(open_cu) == 5
    assert all("Cu" in e.metal_set and e.has_oms and e.source == "SI" for e in open_cu)


# ----------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------


def test_names_come_from_the_file_and_the_cif(tmp_path):
    path = tmp_path / "RUBTAK01_clean.cif"
    path.write_text("data_UiO-66\n_database_code_CSD 'RUBTAK03'\n")
    names = coremof.candidate_names(path)
    assert names[:2] == ["RUBTAK01_clean", "RUBTAK01"]
    assert "UiO-66" in names and "RUBTAK03" in names


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def test_the_sample_is_found_without_a_name(tmp_path):
    """Metals, cell contents, density and mean atomic mass single out
    the sample's own entry and its FSR twin, and nothing else."""
    path = _unnamed_sample(tmp_path)
    matches = coremof.find_matches(_load(path), path=path)
    assert [(m.entry.id, m.confidence) for m in matches] == [
        (SAMPLE_ID, "possible"), ("2020[Zn][dia]3[FSR]1", "possible")]
    assert all(ok for _, _, ok in matches[0].checks.values())


def test_a_supercell_still_matches():
    """CoRE MOF often stores the primitive cell; the Void may have the
    conventional one, or a supercell. Density decides, not atom count."""
    from thevoid_core.services.pore_analysis import major_occupancy_structure

    doubled = major_occupancy_structure(_load(ZN_DIA)) * (2, 1, 1)
    best = coremof.find_matches(doubled)[0]
    assert best.entry.id == SAMPLE_ID
    assert best.checks["n_atoms"] == (248, 124, True)


def test_a_borrowed_name_is_not_an_identity(tmp_path):
    """A different structure saved as TASYAR.cif: found by name, but
    reported as 'name only', and no comparison is made from it."""
    path = tmp_path / "TASYAR.cif"
    text = CR_BPQ.read_text().replace("data_2010[Cr][bpq]3[ASR]1", "data_TASYAR")
    path.write_text(text)
    matches = coremof.find_matches(_load(path), path=path, search_by_composition=False)
    assert {(m.entry.id, m.confidence) for m in matches} == {
        (TASYAR_ID, "name only"), ("2021[Zn][dmc]3[FSR]1", "name only")}

    from thevoid_core.services.report import reference_to_dict

    record = reference_to_dict(_load(path), str(path))
    # Found as itself by composition (CoRE has it as ASR and FSR)...
    assert record["match"] == "possible"
    assert record["entry"].startswith("2010[Cr][bpq]3[")
    # ...and the borrowed name is listed, not used.
    assert {"entry": TASYAR_ID, "match": "name only", "matched_by": "CSD refcode TASYAR"} \
        in record["candidates"]


def test_a_core_id_file_name_is_identified():
    """CoRE's own files carry the id as file name and data block."""
    best = coremof.find_matches(_load(CR_BPQ), path=CR_BPQ)[0]
    assert (best.entry.id, best.confidence) == ("2010[Cr][bpq]3[ASR]1", "identified")


def test_silicon_counts_as_a_metal_the_way_core_counts_it():
    """Round 58: CoRE's metal column includes metalloids (Si, and in
    other entries As, Sb, Ge, Te): 2017[ZnSi][nan]3 is a Zn framework
    with a tetraphenylsilane linker. The Void's periodic table calls Si a
    non-metal, so before this its metal set {Zn} disagreed with the
    entry's {Zn, Si} and CoRE's own file came out "name only" - found
    live, in a 36-structure fetch, and true of 178 entries."""
    path = FIXTURES / "coremof" / "2017[ZnSi][nan]3[ASR]1.cif"
    structure = _load(path)
    assert coremof.StructureFacts.of(structure).metals == frozenset({"Zn", "Si"})
    best = coremof.find_matches(structure, path=path)[0]
    assert (best.entry.id, best.confidence) == ("2017[ZnSi][nan]3[ASR]1", "identified")


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_comparison(tmp_path_factory):
    path = _unnamed_sample(tmp_path_factory.mktemp("sample"))
    structure = _load(path)
    match = coremof.find_matches(structure, path=path)[0]
    return compare_with_coremof(structure, match, samples_per_atom=500)


def test_the_sample_agrees_with_coremof_where_the_methods_are_the_same(sample_comparison):
    """The structure file is CoRE's own (found here without its name), so
    every difference is method: LCD 5.110 against 5.158 and PLD 4.116
    against 4.149 (Zeo++'s high-accuracy mode sits high),
    ASA 1690 against 1705 m2/g, the same net and no open metal site."""
    c = sample_comparison
    assert c.errors == {}
    lcd, pld = c.row("lcd"), c.row("pld")
    assert lcd.reference == pytest.approx(5.15766) and pld.reference == pytest.approx(4.14913)
    assert -0.15 < lcd.difference < 0 and lcd.agrees
    assert -0.15 < pld.difference < 0 and pld.agrees
    assert c.row("asa").void == pytest.approx(1705.23, rel=0.03)
    assert c.row("topology").void == "dia" and c.row("topology").agrees
    assert c.row("oms").agrees and c.row("density").agrees
    assert c.flagged == []
    assert c.parameters["radii"] == "CCDC (Zeo++ defaults)"
    assert c.parameters["probe_radius_A"] == 1.655
    assert any("possible" in note for note in c.notes)


def test_the_comparison_is_recorded_in_reports(sample_comparison):
    record = comparison_to_dict(sample_comparison)
    json.dumps(record, allow_nan=False)
    assert record["entry"] == SAMPLE_ID and record["licence"] == "CC-BY-4.0"
    assert record["reference"]["doi"] == "10.1039/C9CC06849J"
    assert {r["property"] for r in record["rows"]} == {
        "n_atoms", "density", "lcd", "pld", "asa", "nasa", "topology", "oms"}


def test_a_core_si_structure_matches_its_own_entry():
    """2010[Cr][bpq]3[ASR]1 (a Cr-BDC, CoRE ASA 1627 m2/g): the
    structure file is CoRE's own, so every difference is method."""
    structure = _load(CR_BPQ)
    match = coremof.find_matches(structure, path=CR_BPQ)[0]
    c = compare_with_coremof(structure, match, samples_per_atom=500, topology=False, oms=False)
    assert c.confidence == "identified"
    assert c.row("lcd").agrees and c.row("pld").agrees
    assert c.row("asa").void == pytest.approx(1626.75, rel=0.03)
    assert not any("possible" in note for note in c.notes)
