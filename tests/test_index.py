"""Round 65: the shared index on disk - entries, licences, adding,
building, finding, verifying (services/index/entries.py, licence.py,
store.py; the-void-index; docs/shared-index-design.md sections 4-8).

What would go wrong unnoticed, and so is tested here:

- a local path leaking into a shared entry;
- a CSD-derived structure being copied into the index;
- the same analysis stored twice, or two members' copies conflicting;
- a build that depends on who runs it: on the order of entries, or on
  which structure files a member happens to have;
- a hand-edited or tampered entry or structure passing unnoticed;
- replicates that disagree passing unnoticed.

Real analyses of real fixtures are used throughout: a CoRE MOF folder
written by the-void-fetch (from a stand-in archive, as in
test_reference_fetch.py), a structure carrying the CSD's download
notice, and a pair of determinations of one Cu framework with a
reordered copy of one of them.

The CSD file is a stand-in: an open CoRE structure with the notice
written on at test time (_csd_download), because a real CSD file cannot
be published with the tests. What the index looks at is the notice.
"""

import csv
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Structure  # noqa: E402
from pymatgen.io.cif import CifWriter  # noqa: E402

from thevoid_core.services.index import licence  # noqa: E402
from thevoid_core.services.index.entries import (  # noqa: E402
    analysis_id,
    compute_entry_id,
    make_entry,
    method_key,
)
from thevoid_core.services.index.store import Index, IndexProblem  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CORE_IDS = ["2010[Cr][bpq]3[ASR]1", "2019[Mn][bpq]3[ASR]1"]
CU2, CU4 = "2022[Cu][sql]2[ASR]2", "2022[Cu][sql]2[ASR]4"
CSD = "csd_download.cif"

#: The first lines of the notice the CCDC writes on every CIF it serves.
CSD_NOTICE = """\
#######################################################################
#
# This file contains crystal structure data downloaded from the
# Cambridge Structural Database (CSD) hosted by the Cambridge
# Crystallographic Data Centre (CCDC).
#
#######################################################################

"""


def _csd_download(target: Path) -> Path:
    """An open structure (CoRE MOF 2024 SI, CC-BY-4.0) with the CSD's
    download notice on top: to the index, a file from the CSD."""
    source = FIXTURES / "identity" / "2020[Zn][qtz]3[ASR]1.cif"
    target.write_text(CSD_NOTICE + source.read_text())
    return target


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for entry_id in CORE_IDS:
            archive.writestr(f"CR/{entry_id}.cif",
                             (FIXTURES / "coremof" / f"{entry_id}.cif").read_text())
    return buffer.getvalue()


def _shuffled_copy(source: Path, target: Path):
    """The same atoms in another order, written by pymatgen: another
    file (another SHA-256), the same determination."""
    structure = Structure.from_file(source)
    order = np.random.default_rng(3).permutation(len(structure))
    moved = Structure(structure.lattice, [structure[i].species for i in order],
                      [structure[i].frac_coords for i in order])
    CifWriter(moved).write_file(str(target))


def _analyse(files, out):
    from thevoid_core import cli

    cli.main([str(f) for f in files] + ["--out", str(out), "-q", "--samples", "100",
                                        "--no-topology"])


@pytest.fixture(scope="module")
def world(tmp_path_factory, monkeypatch_module):
    """Analysed folders: a fetched CoRE folder, and a lab folder with
    a CSD file and the Cu determinations."""
    from thevoid_core.services.reference import fetch
    from thevoid_core.services.reference.net import CACHE_ENVIRONMENT_VARIABLE

    root = tmp_path_factory.mktemp("world")
    monkeypatch_module.setenv(CACHE_ENVIRONMENT_VARIABLE, str(root / "cache"))
    core = root / "coremof"
    fetch.fetch_coremof(core, {"topology": "bpq"},
                        opener=lambda url: _archive())
    lab = root / "lab"
    lab.mkdir()
    _csd_download(lab / CSD)
    for name in (CU2, CU4):
        shutil.copy(FIXTURES / "identity" / f"{name}.cif", lab / f"{name}.cif")
    _shuffled_copy(lab / f"{CU2}.cif", lab / "Cu2_reordered.cif")
    _analyse(sorted(core.glob("*.cif")), root / "core_results")
    _analyse([lab / CSD, lab / f"{CU2}.cif", lab / f"{CU4}.cif"],
             root / "lab_results")
    _analyse([lab / "Cu2_reordered.cif"], root / "reordered_results")
    return root


@pytest.fixture(scope="module")
def monkeypatch_module():
    patch = pytest.MonkeyPatch()
    yield patch
    patch.undo()


@pytest.fixture
def index(tmp_path, world):
    idx = Index.init(tmp_path / "idx", name="Test Lab")
    idx.add_results(world / "core_results", contributor="A. B.")
    idx.add_results(world / "lab_results", contributor="A. B.")
    return idx


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _report(world, results, stem):
    return json.loads((world / results / f"{stem}.json").read_text())


# ----------------------------------------------------------------------
# Licences
# ----------------------------------------------------------------------


def test_the_fetch_folder_records_each_files_licence(world):
    rows = _read_csv(world / "coremof" / "reference.csv")
    assert {r["ref_licence"] for r in rows} == {"CC-BY-4.0"}
    assert all("supporting-information archive" in r["ref_licence_basis"] for r in rows)
    verdict = licence.detect(world / "coremof" / f"{CORE_IDS[0]}.cif")
    assert verdict["spdx"] == "CC-BY-4.0" and verdict["redistributable"]


def test_a_csd_file_is_never_redistributable_whatever_is_declared(world):
    path = world / "lab" / CSD
    declared = {"spdx": "CC0-1.0", "basis": "I say so"}
    verdict = licence.detect(path, declared)
    assert verdict == {"spdx": "CSD", "redistributable": False, "basis": "CSD header in file"}


def test_unknown_files_are_withheld_unless_declared(world):
    path = world / "lab" / f"{CU2}.cif"  # a CoRE SI file, but copied without its folder
    assert licence.detect(path) == {"spdx": None, "redistributable": False, "basis": "unknown"}
    declared = licence.detect(path, {"spdx": "CC-BY-4.0", "basis": "CoRE MOF SI"})
    assert declared["redistributable"] and "declared by contributor" in declared["basis"]
    restricted = licence.detect(path, {"spdx": "LicenseRef-lab-only", "basis": "ours"})
    assert not restricted["redistributable"]


def test_folders_fetched_before_the_licence_columns_are_still_recognised(tmp_path, world):
    old = tmp_path / "old"
    shutil.copytree(world / "coremof", old)
    rows = _read_csv(old / "reference.csv")
    columns = [c for c in rows[0] if c not in ("ref_licence", "ref_licence_basis")]
    with open(old / "reference.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    verdict = licence.detect(old / f"{CORE_IDS[0]}.cif")
    assert verdict["spdx"] == "CC-BY-4.0" and "README" in verdict["basis"]


def test_a_mofxdb_file_is_not_stored(tmp_path):
    folder = tmp_path / "hmof"
    folder.mkdir()
    shutil.copy(FIXTURES / "coremof" / f"{CORE_IDS[0]}.cif", folder / "hMOF-1.cif")
    (folder / "reference.csv").write_text(
        "file,ref_licence,ref_licence_basis\nhMOF-1.cif,,MOFX-DB: no licence stated\n")
    verdict = licence.detect(folder / "hMOF-1.cif")
    assert not verdict["redistributable"] and verdict["basis"] == "MOFX-DB: no licence stated"


def test_a_variant_inherits_its_parents_licence(tmp_path):
    folder = tmp_path / "variants"
    folder.mkdir()
    _csd_download(folder / CSD)
    # A variant written by pymatgen has no CSD header of its own.
    text = (folder / CSD).read_text().replace("Cambridge", "C.")
    (folder / "csd_download_NH2_1.cif").write_text(text)
    (folder / "functionalization.json").write_text(json.dumps({
        "parent": {"file": f"/somewhere/{CSD}", "sha256": "abc"},
        "variants": [{"file": "csd_download_NH2_1.cif"}]}))
    verdict = licence.detect(folder / "csd_download_NH2_1.cif")
    assert not verdict["redistributable"]
    assert verdict["basis"] == "functionalised variant; parent: CSD header in file"


# ----------------------------------------------------------------------
# Entries
# ----------------------------------------------------------------------


def _strings(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_an_entry_carries_no_local_path(world):
    report = _report(world, "core_results", CORE_IDS[0])
    assert str(world) in report["input"]["file"]  # the report does
    entry = make_entry(report, world / "coremof" / f"{CORE_IDS[0]}.cif", contributor="A. B.")
    assert not [s for s in _strings(entry) if str(world) in s or s.startswith("/")]
    assert entry["structure"]["file_name"] == f"{CORE_IDS[0]}.cif"


def test_an_entry_is_named_by_its_content(world):
    report = _report(world, "core_results", CORE_IDS[0])
    path = world / "coremof" / f"{CORE_IDS[0]}.cif"
    a = make_entry(report, path, contributor="A. B.", added="2026-09-30")
    b = make_entry(report, path, contributor="A. B.", added="2026-09-30")
    assert a == b and a["entry_id"] == compute_entry_id(a)
    c = make_entry(report, path, contributor="C. D.", added="2026-10-01")
    assert c["entry_id"] != a["entry_id"] and analysis_id(c) == analysis_id(a)


def test_what_an_entry_records(world):
    report = _report(world, "core_results", CORE_IDS[0])
    entry = make_entry(report, world / "coremof" / f"{CORE_IDS[0]}.cif", contributor="A. B.")
    s, a = entry["structure"], entry["analysis"]
    assert s["licence"]["spdx"] == "CC-BY-4.0" and s["origin"] == "unspecified"
    assert {"kind": "coremof", "value": CORE_IDS[0]} in s["aliases"]
    assert s["metals"] == ["Cr"] and s["key_version"] == 1
    assert a["method_key"] == method_key(report["provenance"]["software"]["the_void"],
                                         report["config"])
    assert a["results"]["pore"]["pld_A"] == report["results"]["pore"]["pld_A"]
    assert a["summary"]["pld_A"] == report["results"]["pore"]["pld_A"]


def test_a_changed_file_is_refused(tmp_path, world):
    report = _report(world, "core_results", CORE_IDS[0])
    changed = tmp_path / f"{CORE_IDS[0]}.cif"
    changed.write_text((world / "coremof" / f"{CORE_IDS[0]}.cif").read_text() + "\n# edited\n")
    with pytest.raises(ValueError, match="has changed since it was analysed"):
        make_entry(report, changed, contributor="A. B.")


# ----------------------------------------------------------------------
# Adding and building
# ----------------------------------------------------------------------


def test_adding_stores_only_what_may_be_redistributed(tmp_path, world):
    idx = Index.init(tmp_path / "idx", name="Test Lab")
    core = idx.add_results(world / "core_results", contributor="A. B.")
    assert len(core.added) == 2 and sorted(core.stored) == [f"{i}.cif" for i in CORE_IDS]
    lab = idx.add_results(world / "lab_results", contributor="A. B.")
    assert len(lab.added) == 3 and lab.stored == []
    assert dict(lab.withheld)[CSD] == "CSD header in file"
    assert dict(lab.withheld)[f"{CU2}.cif"] == "unknown"
    assert len(list((tmp_path / "idx" / "structures").glob("*/*.cif"))) == 2
    assert (tmp_path / "idx" / ".gitignore").read_text().startswith("build/")


def test_the_same_analysis_is_added_once(index, world):
    again = index.add_results(world / "core_results", contributor="Someone Else",
                              note="added again")
    assert again.added == [] and len(again.present) == 2


def test_a_moved_structure_file_is_skipped_with_a_reason(tmp_path, world):
    results = tmp_path / "results"
    shutil.copytree(world / "reordered_results", results)
    # The report records where the file was; a copy of the results
    # elsewhere, without the file, cannot be added.
    for report in results.glob("*.json"):
        data = json.loads(report.read_text())
        data["input"]["file"] = str(tmp_path / "gone" / "Cu2_reordered.cif")
        report.write_text(json.dumps(data))
    idx = Index.init(tmp_path / "idx", name="Test Lab")
    result = idx.add_results(results, contributor="A. B.")
    assert result.added == [] and result.skipped == [("Cu2_reordered.cif",
                                                      "structure file not found")]
    found = idx.add_results(results, contributor="A. B.", search_dirs=[world / "lab"])
    assert len(found.added) == 1


def test_build_ids_and_the_batch_results_window(index, world):
    from thevoid_core.services.batch_results import load_batch_results, locate_structure

    index.add_results(world / "reordered_results", contributor="C. D.")
    result = index.build()
    # Cr-BDC, Mn-BDC, the CSD file and the Cu framework: 4 structures; the Cu
    # one has two determinations, one of them in two files.
    assert (result.rows, result.structures, result.determinations) == (6, 4, 5)
    assert result.problems == [] and result.undecided == []
    rows = {r["name"]: r for r in _read_csv(index.path / "build" / "summary.csv")}
    assert rows["Cu2_reordered.cif"]["determination_id"] == rows[f"{CU2}.cif"]["determination_id"]
    assert rows[f"{CU4}.cif"]["structure_id"] == rows[f"{CU2}.cif"]["structure_id"]
    assert rows[f"{CU4}.cif"]["determination_id"] != rows[f"{CU2}.cif"]["determination_id"]
    assert rows[f"{CORE_IDS[0]}.cif"]["structure_stored"] == "yes"
    assert rows[CSD]["licence"] == "CSD"
    batch = load_batch_results(index.path / "build")
    assert len(batch.rows) == 6 and "structure_id" in batch.columns
    located = {r.get("name"): locate_structure(r, batch).status for r in batch.rows}
    assert located[f"{CORE_IDS[0]}.cif"] == "verified"   # the stored copy
    assert located[CSD] == "missing"            # never stored
    info = json.loads((index.path / "build" / "BUILD.json").read_text())
    assert info["entries"] == 6


def test_the_build_does_not_depend_on_who_builds_it(index, tmp_path):
    first = index.build()
    text = (index.path / "build" / "summary.csv").read_text()
    copy = tmp_path / "copy"
    shutil.copytree(index.path, copy, ignore=shutil.ignore_patterns("build"))
    again = Index.open(copy).build()
    assert (copy / "build" / "summary.csv").read_text() == text
    assert (again.structures, again.determinations) == (first.structures, first.determinations)


def test_two_members_copies_merge_without_conflict(tmp_path, world):
    """Each member adds to a copy; the union of the two folders (what a
    Git merge produces when every write is a new file) builds the same
    table as one index with both adds."""
    base = Index.init(tmp_path / "base", name="Test Lab")
    a_path, b_path = tmp_path / "a", tmp_path / "b"
    shutil.copytree(base.path, a_path)
    shutil.copytree(base.path, b_path)
    Index.open(a_path).add_results(world / "core_results", contributor="A. B.",
                                   )
    Index.open(b_path).add_results(world / "lab_results", contributor="C. D.")
    Index.open(b_path).add_results(world / "core_results", contributor="C. D.")  # overlap
    for file in b_path.rglob("*"):
        if file.is_file():
            target = a_path / file.relative_to(b_path)
            if target.exists():
                assert target.read_bytes() == file.read_bytes(), f"conflict: {target}"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(file, target)
    merged = Index.open(a_path).build()
    rows = _read_csv(a_path / "build" / "summary.csv")
    assert merged.rows == 5  # one row per analysis, whoever added it
    also = [r["also_added_by"] for r in rows if r["name"] == f"{CORE_IDS[0]}.cif"]
    assert also and also[0].startswith(("A. B.", "C. D."))


def test_a_member_without_the_files_builds_the_same_ids(tmp_path, world):
    """Decisions made where both files were at hand are recorded, so a
    build without them (the CSD case) needs no match."""
    idx = Index.init(tmp_path / "idx", name="Test Lab")
    idx.add_results(world / "lab_results", contributor="A. B.")
    idx.add_results(world / "reordered_results", contributor="A. B.")
    assert list((idx.path / "matches").glob("*/*.json"))  # nothing stored, all decided
    result = idx.build()
    assert (result.fits, result.undecided) == (0, [])
    rows = {r["name"]: r for r in _read_csv(idx.path / "build" / "summary.csv")}
    assert rows["Cu2_reordered.cif"]["determination_id"] == rows[f"{CU2}.cif"]["determination_id"]
    # Without the decisions, and without the files, nothing is merged on a guess.
    shutil.rmtree(idx.path / "matches")
    (idx.path / "matches").mkdir()
    bare = idx.build()
    assert bare.undecided and bare.structures == 4
    # Given the files, the build decides again, and records it.
    decided = idx.build(search_dirs=[world / "lab"])
    assert decided.undecided == [] and decided.fits > 0 and decided.structures == 2
    assert list((idx.path / "matches").glob("*/*.json"))


# ----------------------------------------------------------------------
# Finding, withdrawing, verifying
# ----------------------------------------------------------------------


def test_find_says_how_a_file_relates_to_the_index(index, world):
    found = index.find(world / "lab" / "Cu2_reordered.cif", search_dirs=[world / "lab"])
    relations = {e["structure"]["file_name"]: r for r, e in found}
    assert relations == {f"{CU2}.cif": "same determination", f"{CU4}.cif": "same structure"}
    same = index.find(world / "coremof" / f"{CORE_IDS[1]}.cif")
    assert [r for r, _ in same] == ["same file"]
    assert index.find(FIXTURES / "coremof" / "2023[Ni][dia]3[ASR]1.cif") == []
    # The dia polymorph of the CSD file's qtz net shares its key. The CSD
    # file is not stored, so the two are told apart only when a copy of
    # it is at hand.
    dia = FIXTURES / "identity" / "2020[Zn][dia]3[ASR]1.cif"
    assert [r for r, _ in index.find(dia)] == ["same key, not compared"]
    assert [r for r, _ in index.find(dia, search_dirs=[world / "lab"])] == [
        "same key, different structure"]


def test_withdrawn_entries_leave_the_build(index):
    entry_id = next(iter(index.load_entries()))
    index.withdraw(entry_id[:12], reason="wrong CIF", by="A. B.")
    assert index.build().rows == 4
    with pytest.raises(IndexProblem, match="matches 0 entries"):
        index.withdraw("zzzz", reason="x", by="y")


def test_verify_finds_nothing_wrong_with_a_clean_index(index):
    assert index.verify() == []


def test_verify_finds_a_hand_edited_entry_and_a_tampered_structure(index):
    entry_file = index.entry_files()[0]
    data = json.loads(entry_file.read_text())
    data["analysis"]["summary"]["pld_A"] = 99.0
    entry_file.write_text(json.dumps(data))
    stored = next((index.path / "structures").glob("*/*.cif"))
    stored.write_text(stored.read_text() + "\n")
    (index.path / "withdrawn" / ("0" * 64 + ".json")).write_text("{}")
    problems = index.verify()
    assert any("does not match its id" in p for p in problems)
    assert any("SHA-256 is not its name" in p for p in problems)
    assert any("no such entry" in p for p in problems)


def test_verify_finds_replicates_that_disagree(index, world):
    """An analysis of the same file with the same method, but another
    PLD, must be flagged; a reordered copy (another file, the same
    determination) that agrees within tolerance must not."""
    index.add_results(world / "reordered_results", contributor="C. D.")
    assert index.verify() == []
    original = next(e for e in index.load_entries().values()
                    if e["structure"]["file_name"] == f"{CU2}.cif")
    forged = json.loads(json.dumps(original))
    forged["analysis"]["summary"]["pld_A"] = original["analysis"]["summary"]["pld_A"] + 0.3
    forged["analysis"]["platform"] = "another machine"
    forged["entry_id"] = compute_entry_id({k: v for k, v in forged.items() if k != "entry_id"})
    index.add_entry(forged)
    # Every disagreeing pair, whatever order the ids fall in: the forged
    # entry against its own file (to rounding) and against the reordered
    # copy (beyond the 0.15 A tolerance); the two honest ones agree.
    problems = index.verify()
    assert len(problems) == 2 and all(forged["entry_id"][:10] in p for p in problems)
    assert {"same file" in p for p in problems} == {True, False}
    assert all("pld_A" in p for p in problems)


def test_an_index_from_a_newer_version_is_refused(tmp_path):
    idx = Index.init(tmp_path / "idx", name="x")
    info = json.loads((idx.path / "INDEX.json").read_text())
    info["entry_format_version"] = 99
    (idx.path / "INDEX.json").write_text(json.dumps(info))
    with pytest.raises(IndexProblem, match="entry format 99"):
        Index.open(idx.path)
    with pytest.raises(IndexProblem, match="not an index"):
        Index.open(tmp_path)
    with pytest.raises(IndexProblem, match="already an index"):
        Index.init(idx.path, name="again")


# ----------------------------------------------------------------------
# The command line
# ----------------------------------------------------------------------


def test_the_command_line(tmp_path, world, capsys):
    from thevoid_core import index_cli

    path = str(tmp_path / "idx")
    assert index_cli.main(["init", path, "--name", "CLI Lab"]) == 0
    assert index_cli.main(["add", path, str(world / "lab_results"), "--contributor", "A. B."]) == 0
    out = capsys.readouterr().out
    assert "3 added" in out and f"not stored {CSD}: CSD header in file" in out
    assert index_cli.main(["build", path]) == 0
    assert "3 entries, 2 structures, 3 determinations" in capsys.readouterr().out
    assert index_cli.main(["find", path, str(world / "lab" / f"{CU4}.cif")]) == 0
    assert capsys.readouterr().out.startswith("same file")
    assert index_cli.main(["verify", path]) == 0
    assert index_cli.main(["verify", str(tmp_path)]) == 2
    assert "not an index" in capsys.readouterr().err
