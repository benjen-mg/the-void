"""Round 67: stale entries, refreshing them, and training from an index
(services/index/staleness.py, Index.stale/refresh, the-void-index stale
and refresh, train_from_batch on an index; data/method_changes.json).

An entry keeps the numbers of the version that computed them. What must
not happen unnoticed:

- an old number ranked beside new ones as if nothing had changed;
- a refresh overwriting, or losing what the old entry knew (its
  licence, aliases, joined columns), or rerunning with other options;
- an Analysis-menu entry "refreshed" with options it never had;
- a model trained on a mixture of methods, or on superseded entries, or
  with one structure on both sides of its test split.

The stale entries here are real analyses whose reports are relabelled
as version 0.9.2, which the changelog says had other surface areas. The
same cycle was run end to end on real 0.9.2 output (round 67).
"""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services.index import staleness  # noqa: E402
from thevoid_core.services.index.store import Index  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"
CU2 = FIXTURES / "identity" / "2022[Cu][sql]2[ASR]2.cif"
FAST = dict(topology=False, oms=False, surface_samples_per_atom=100)


def _report(path, version="0.9.2", **options):
    from pymatgen.core import Structure

    from thevoid_core.services.report import AnalysisConfig, run_analysis

    report = run_analysis(Structure.from_file(path), AnalysisConfig(**{**FAST, **options}),
                          source=str(path))
    report["provenance"]["software"]["the_void"] = version
    return report


@pytest.fixture(scope="module")
def old_reports():
    return {path.name: _report(path) for path in (CR_BDC, CU2)}


@pytest.fixture
def index(tmp_path, old_reports):
    idx = Index.init(tmp_path / "idx", name="Refresh Lab")
    for path in (CR_BDC, CU2):
        idx.add_report(old_reports[path.name], path, contributor="A. B.",
                       declared_licence={"spdx": "CC-BY-4.0", "basis": "CoRE MOF SI"})
    return idx


# ----------------------------------------------------------------------
# The changelog
# ----------------------------------------------------------------------


def test_the_changelog_names_real_versions_and_columns():
    from thevoid_core.services.report import SUMMARY_COLUMNS
    from thevoid_core.version import __version__

    changes = staleness.load_changes()
    assert changes and all(c["what"] and c["round"] for c in changes)
    for change in changes:
        assert (0, 8, 1) < staleness.version_tuple(change["version"]) <= \
            staleness.version_tuple(__version__)
        assert change.get("condition") in (None, "not_cif", "has_metal", "value_matches")
        assert ("metals" in change) == (change.get("condition") == "has_metal")
        assert ("values" in change) == (change.get("condition") == "value_matches")
        assert set(change.get("values", {})) <= set(change["columns"])
        for column in change["columns"]:
            assert column in SUMMARY_COLUMNS or column == "guest_*", column
    versions = [staleness.version_tuple(c["version"]) for c in changes]
    assert versions == sorted(versions)
    # Round 63's surface-area change is recorded (the one these tests use).
    assert any(c["version"] == "0.9.3" and "channel_asa_m2_per_g" in c["columns"]
               for c in changes)


def test_versions_compare_as_numbers():
    vt = staleness.version_tuple
    assert vt("0.9.10") > vt("0.9.9") > vt("0.9") and vt("") == (0,) and vt("x") == (0,)


# ----------------------------------------------------------------------
# What is stale
# ----------------------------------------------------------------------


def _entry(version, summary, file_name="a.cif", config=None, sha="s", entry_id="e"):
    return {"entry_id": entry_id,
            "structure": {"file_name": file_name, "file_sha256": sha},
            "analysis": {"software": {"the_void": version}, "summary": summary,
                         "config": config or {"pore": True}}}


def test_stale_columns_are_changed_columns_that_have_a_value():
    changes = staleness.load_changes()
    summary = {"pld_A": 5.0, "channel_asa_m2_per_g": 100.0, "topology": None}
    old = staleness.staleness_of(_entry("0.8.4", summary), changes, "0.9.7")
    assert set(old.columns) == {"pld_A", "channel_asa_m2_per_g"}  # topology has no value
    assert any(r.startswith("0.8.6: ") for r in old.reasons())
    assert set(staleness.staleness_of(_entry("0.9.2", summary), changes, "0.9.7").columns) == \
        {"channel_asa_m2_per_g"}
    assert not staleness.staleness_of(_entry("0.9.3", summary), changes, "0.9.7").stale
    # A change in a version newer than the running one is not applied.
    assert not staleness.staleness_of(_entry("0.9.2", summary), changes, "0.9.2").stale


def test_the_orientation_change_only_applies_to_non_cif_inputs():
    changes = [c for c in staleness.load_changes() if c["version"] == "0.9.0"]
    summary = {"channel_asa_m2_per_g": 100.0}
    assert not staleness.staleness_of(_entry("0.8.9", summary, "x.cif"), changes, "0.9.7").stale
    assert staleness.staleness_of(_entry("0.8.9", summary, "POSCAR"), changes, "0.9.7").stale


def test_a_newer_analysis_of_the_same_file_and_options_supersedes():
    summary = {"channel_asa_m2_per_g": 100.0}
    entries = {
        "old": _entry("0.9.2", summary, entry_id="old"),
        "new": _entry("0.9.6", summary, entry_id="new"),
        "other": _entry("0.9.2", summary, config={"pore": False}, entry_id="other"),
    }
    verdicts = staleness.assess(entries, current_version="0.9.7")
    assert verdicts["old"].superseded_by == "new" and verdicts["new"].superseded_by is None
    assert verdicts["other"].superseded_by is None and verdicts["other"].stale


# ----------------------------------------------------------------------
# Refreshing
# ----------------------------------------------------------------------


def test_stale_entries_are_listed_with_why(index):
    stale = index.stale()
    assert sorted(e["structure"]["file_name"] for e, _ in stale) == sorted([CR_BDC.name,
                                                                           CU2.name])
    for _entry_, verdict in stale:
        assert "channel_asa_m2_per_g" in verdict.columns and "pld_A" not in verdict.columns
        assert any("canonical atom order" in r for r in verdict.reasons())


def test_refresh_adds_new_entries_and_keeps_the_old(index):
    from thevoid_core.version import __version__

    old = {e["structure"]["file_name"]: e for e in index.load_entries().values()}
    dry = index.refresh(contributor="C. D.", dry_run=True)
    assert len(dry.would_refresh) == 2 and dry.refreshed == [] and len(index.load_entries()) == 2
    result = index.refresh(contributor="C. D.")
    assert len(result.refreshed) == 2 and result.skipped == []
    entries = index.load_entries()
    assert len(entries) == 4 and set(old[CR_BDC.name]["entry_id"] for _ in [0]) <= set(entries)
    new = {e["structure"]["file_name"]: e for k, e in entries.items()
           if e["analysis"]["software"]["the_void"] == __version__}
    fresh, before = new[CR_BDC.name], old[CR_BDC.name]
    assert fresh["analysis"]["config"] == before["analysis"]["config"]   # same options
    assert fresh["structure"]["file_name"] == CR_BDC.name                 # not the stored name
    assert fresh["structure"]["licence"] == before["structure"]["licence"]
    assert {"kind": "coremof", "value": CR_BDC.stem} in fresh["structure"]["aliases"]
    assert fresh["contributor"]["note"].startswith(f"refresh of {before['entry_id'][:12]}")
    assert fresh["analysis"]["summary"]["pld_A"] == pytest.approx(
        before["analysis"]["summary"]["pld_A"], abs=1e-9)                # geometry unchanged
    assert index.stale() == []
    index.build()
    rows = {(r["name"], r["the_void_version"]): r
            for r in csv.DictReader(open(index.path / "build" / "summary.csv"))}
    assert rows[(CR_BDC.name, "0.9.2")]["superseded_by"] == fresh["entry_id"]
    assert rows[(CR_BDC.name, "0.9.2")]["stale_columns"].startswith("channel_asa_m2_per_g")
    assert rows[(CR_BDC.name, __version__)]["superseded_by"] == ""
    assert index.verify() == []


def test_refresh_needs_the_file_and_real_options(tmp_path, old_reports):
    from thevoid_core.services import report as reporting

    idx = Index.init(tmp_path / "idx", name="Refresh Lab")
    idx.add_report(old_reports[CU2.name], CU2, contributor="A. B.")  # licence unknown: not stored
    gui = reporting.new_report(__import__("pymatgen.core").core.Structure.from_file(CR_BDC),
                               source=str(CR_BDC))
    gui["results"] = {"pore": old_reports[CR_BDC.name]["results"]["pore"]}
    gui["provenance"]["software"]["the_void"] = "0.8.4"
    idx.add_report(gui, CR_BDC, contributor="A. B.",
                   declared_licence={"spdx": "CC-BY-4.0", "basis": "CoRE MOF SI"})
    result = idx.refresh(contributor="C. D.")
    reasons = {name: reason for _id, name, reason in result.skipped}
    assert "--structures" in reasons[CU2.name]
    assert "Analysis-menu entry" in reasons[CR_BDC.name]
    assert result.refreshed == []
    found = idx.refresh(contributor="C. D.", search_dirs=[CU2.parent])
    assert len(found.refreshed) == 1


def test_the_command_line(index, capsys):
    from thevoid_core import index_cli

    assert index_cli.main(["stale", str(index.path)]) == 0
    out = capsys.readouterr().out
    assert "2 stale entries" in out and "0.9.3: Surface samples" in out
    assert index_cli.main(["refresh", str(index.path), "--contributor", "C. D.",
                           "--dry-run"]) == 0
    assert "would refresh 2 of 2" in capsys.readouterr().out
    assert index_cli.main(["refresh", str(index.path), "--contributor", "C. D.",
                           "--limit", "1"]) == 0
    assert "refreshed 1 of 2" in capsys.readouterr().out
    assert index_cli.main(["stale", str(index.path)]) == 0
    assert "1 stale entry" in capsys.readouterr().out


# ----------------------------------------------------------------------
# Training from an index
# ----------------------------------------------------------------------


def _labelled_index(tmp_path, n=40):
    """n real files (one structure, its atoms jittered by 1e-4 of a cell
    so each is its own file), each with a real report whose PLD and
    channel ASA are set to known values, and a label that depends on
    them. Half the files are also 'analysed' with other options."""
    from pymatgen.core import Structure

    from thevoid_core.services.report import file_sha256

    base = Structure.from_file(CR_BDC)
    template = _report(CR_BDC, version="0.9.6")
    folder = tmp_path / "files"
    folder.mkdir()
    idx = Index.init(tmp_path / "idx", name="Train Lab")
    rng = np.random.default_rng(0)
    for i in range(n):
        s = base.copy()
        s.perturb(1e-3, min_distance=1e-4)
        path = folder / f"m{i:02d}.cif"
        s.to(filename=str(path))
        pld, asa = rng.uniform(3, 12), rng.uniform(200, 3500)
        for samples in ((100, 300) if i % 2 == 0 else (100,)):
            report = json.loads(json.dumps(template))
            report["input"] = {"file": str(path), "sha256": file_sha256(path)}
            report["config"]["surface_samples_per_atom"] = samples
            report["results"]["pore"]["pld_A"] = pld
            report["results"]["pore"]["lcd_A"] = pld + 2
            report["results"]["surface_area"]["channel_asa_m2_per_g"] = asa
            entry_joined = {"ref_uptake": 0.5 + 0.0008 * asa + 0.15 * pld}
            from thevoid_core.services.index.entries import make_entry

            entry = make_entry(report, path, contributor="A. B.", joined=entry_joined,
                               structure=s)
            idx.add_entry(entry, path)
    return idx


def test_training_from_an_index_uses_one_method_and_structure_groups(tmp_path):
    from thevoid_core.services.ml.batch_training import train_from_batch

    idx = _labelled_index(tmp_path)
    model = train_from_batch(idx.path, "ref_uptake")
    data = model.card["training_data"]
    assert data["index"] == "Train Lab" and data["entries"] == 40   # the 100-sample method
    rows = list(csv.DictReader(open(idx.path / "build" / "summary.csv")))
    keys = {r["method_key"] for r in rows}
    assert len(keys) == 2 and data["method_key"] in keys
    # Split by structure_id. The jittered copies' structures were never
    # at hand together (not stored: no licence), so each is its own.
    assert model.card["rows"]["groups"] == len({r["structure_id"] for r in rows
                                                if r["method_key"] == data["method_key"]})
    other = next(k for k in keys if k != data["method_key"])
    with pytest.raises(ValueError, match="too few"):
        train_from_batch(idx.path, "ref_uptake", method_key=other)       # 20 rows only


def test_thevoid_train_accepts_an_index(tmp_path):
    import thevoid

    idx = _labelled_index(tmp_path)
    model = thevoid.train(thevoid.Index(idx.path), "ref_uptake")
    assert model.card["training_data"]["entries"] == 40


def test_replicates_of_one_structure_stay_on_one_side_of_the_split(tmp_path):
    """Rows that share a structure_id are one group: 40 rows that are
    20 structures, twice each, make 20 groups, not 40."""
    from thevoid_core.services.ml.batch_training import train_from_batch

    rng = np.random.default_rng(1)
    folder = tmp_path / "build"
    folder.mkdir()
    columns = ["file", "formula", "lcd_A", "pld_A", "void_fraction_percent",
               "channel_asa_m2_per_g", "density_g_cm3", "accessible_dimensionality",
               "open_metal_sites", "topology", "errors", "structure_id", "ref_uptake"]
    with open(folder / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(40):
            pld, asa = rng.uniform(3, 12), rng.uniform(200, 3500)
            writer.writerow([f"m{i}.cif", "Zn4 C24 H12 O13", pld + 2, pld, 50, asa, 1.0, 3, 0,
                             "pcu 6-c", "", f"key.{i // 2 + 1}", 0.001 * asa + 0.1 * pld])
    model = train_from_batch(folder, "ref_uptake")
    assert model.card["rows"]["groups"] == 20


# ----------------------------------------------------------------------
# Refreshing several at once (round 68)
# ----------------------------------------------------------------------


def test_parallel_refresh_writes_exactly_what_a_serial_one_does(index, tmp_path):
    """Analyses run in worker processes, entries are added in the stale
    list's order: two jobs must leave byte-identical entry files."""
    copy = tmp_path / "copy"
    shutil.copytree(index.path, copy)
    serial = index.refresh(contributor="C. D.", jobs=1)
    parallel = Index.open(copy).refresh(contributor="C. D.", jobs=2)
    assert serial.refreshed == parallel.refreshed and len(serial.refreshed) == 2
    files = sorted(p.relative_to(index.path) for p in (index.path / "entries").rglob("*.json"))
    assert files == sorted(p.relative_to(copy) for p in (copy / "entries").rglob("*.json"))
    for name in files:
        assert (index.path / name).read_bytes() == (copy / name).read_bytes()


def test_a_changed_or_unreadable_file_is_skipped_not_fatal(index, tmp_path):
    """A stored copy that is no longer the indexed file is not analysed;
    a file that cannot be analysed at all is reported, and the others
    are still refreshed."""
    from thevoid_core.services.index.store import _refresh_one

    stored = next((index.path / "structures").glob("*/*.cif"))
    stored.write_text("not a CIF\n")
    result = index.refresh(contributor="C. D.", jobs=2)
    assert len(result.refreshed) == 1
    assert len(result.skipped) == 1 and "has changed" in result.skipped[0][2]
    bad = tmp_path / "bad.cif"
    bad.write_text("not a CIF\n")
    entry = next(iter(index.load_entries().values()))
    message = _refresh_one(str(bad), "bad.cif", {}, "C. D.", "", entry)
    assert isinstance(message, str) and message.startswith("could not be analysed")
    with pytest.raises(ValueError, match="at least 1"):
        index.refresh(contributor="C. D.", jobs=0)


def test_refresh_from_python(index):
    import thevoid

    view = thevoid.Index(index.path)
    stale = view.stale()
    assert len(stale) == 2 and all("channel_asa_m2_per_g" in s["columns"] for s in stale)
    done = view.refresh(contributor="C. D.", jobs=2)
    assert len(done["refreshed"]) == 2 and done["skipped"] == [] and view.stale() == []
