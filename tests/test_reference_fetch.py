"""Round 56: downloading reference structures into a folder ready for
batch analysis (services/reference/fetch.py, the-void-fetch), and
joining their reference values into a batch (the-void-analyze
--reference-csv / --compare-coremof).

No test here touches the network: every download goes through the
`opener` hook. CoRE MOF's archive is stood in for by a zip of two of its
own structures (tests/fixtures/coremof, CC-BY-4.0); MOFX-DB by two
recorded-format pages (tests/fixtures/mofxdb_page*.json - their CIF is a
stand-in and their isotherm points placeholders, as the files say:
MOFX-DB could not be reached where round 56 was written). One live
test, off unless THE_VOID_LIVE_NETWORK=1, checks the real services.
"""

import csv
import io
import json
import os
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services.reference import coremof, fetch  # noqa: E402
from thevoid_core.services.reference.net import CACHE_ENVIRONMENT_VARIABLE, Cancelled  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CORE = FIXTURES / "coremof"
IDS = ["2010[Cr][bpq]3[ASR]1", "2019[Mn][bpq]3[ASR]1"]


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv(CACHE_ENVIRONMENT_VARIABLE, str(tmp_path / "cache"))


def _archive(tamper=False) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("CR/", "")
        for entry_id in IDS:
            text = (CORE / f"{entry_id}.cif").read_text()
            if tamper and entry_id == IDS[1]:
                text = text.replace("_cell_length_a", "_cell_length_a 1.1\n#", 1)
            archive.writestr(f"CR/{entry_id}.cif", text)
    return buffer.getvalue()


def _read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


# ----------------------------------------------------------------------
# CoRE MOF 2024
# ----------------------------------------------------------------------


def test_the_plan_is_made_offline_and_counts_what_cannot_be_downloaded():
    downloadable, others = fetch.plan_coremof({"pld_min": 6, "topology": "pcu"})
    assert downloadable and others
    assert all(e.source == "SI" for e in downloadable)
    assert all(e.source == "CSD" for e in others)
    capped, _ = fetch.plan_coremof({"pld_min": 6, "topology": "pcu"}, limit=3)
    assert capped == downloadable[:3]


def test_fetching_writes_the_structures_their_values_and_the_attribution(tmp_path):
    calls = []

    def opener(url):
        calls.append(url)
        return _archive()

    out = tmp_path / "out"
    result = fetch.fetch_coremof(out, {"topology": "bpq", "metals": ["Cr"]},
                                 opener=opener)
    assert calls == [coremof.index_info()["si_zip_url"]]

    written = sorted(p.name for p in result.written)
    assert "2010[Cr][bpq]3[ASR]1.cif" in written
    rows = {r["file"]: r for r in _read_csv(out / "reference.csv")}
    row = rows["2010[Cr][bpq]3[ASR]1.cif"]
    assert float(row["ref_pld_A"]) == pytest.approx(7.09493)
    assert row["ref_doi"] == "10.1021/jp102463p"
    assert row["matches_index"] == "yes"

    # Entries of the query that are not in this (stand-in) archive are
    # reported, not silently dropped; CSD ones are listed with refcodes.
    assert all(reason == "not in the downloaded archive" for _, reason in result.skipped)
    if result.not_downloadable:
        listed = _read_csv(out / "not_downloadable.csv")
        assert all(r["csd_refcode"] for r in listed)

    readme = (out / "README.txt").read_text()
    assert "CC-BY-4.0" in readme and "10.1016/j.matt.2025.102140" in readme
    assert "1.655 A probe" in readme
    # The stand-in archive is not the one the index was built from.
    assert any("has changed" in w for w in result.warnings)


def test_the_archive_is_downloaded_once(tmp_path):
    """Cached after the first fetch - even when, as with this stand-in,
    its checksum is not the one the index expects: a cached copy is only
    ever stored whole, so a different checksum means a different
    upstream version, which each fetch reports. refresh=True downloads
    it again."""
    count = []

    def opener(url):
        count.append(url)
        return _archive()

    query = {"topology": "bpq", "metals": ["Mn"]}
    first = fetch.fetch_coremof(tmp_path / "a", query, opener=opener)
    second = fetch.fetch_coremof(tmp_path / "b", query, opener=opener)
    assert len(count) == 1
    assert first.warnings and second.warnings
    fetch.fetch_coremof(tmp_path / "c", query, opener=opener, refresh=True)
    assert len(count) == 2


def test_a_file_that_is_not_its_entry_is_flagged(tmp_path):
    result = fetch.fetch_coremof(tmp_path / "out", {"topology": "bpq", "metals": ["Mn"]},
                                 opener=lambda url: _archive(tamper=True))
    rows = {r["file"]: r for r in _read_csv(result.reference_csv)}
    assert rows["2019[Mn][bpq]3[ASR]1.cif"]["matches_index"].startswith("no: ")


def test_a_fetch_can_be_cancelled(tmp_path):
    with pytest.raises(Cancelled):
        fetch.fetch_coremof(tmp_path / "out", {"topology": "bpq"}, opener=lambda url: _archive(),
                            progress=lambda message, done, total: not message.startswith("Writing"))


# ----------------------------------------------------------------------
# MOFX-DB
# ----------------------------------------------------------------------


def _mofxdb_opener(seen):
    def opener(url):
        seen.append(url)
        page = parse_qs(urlparse(url).query)["page"][0]
        return (FIXTURES / f"mofxdb_page{page}.json").read_bytes()
    return opener


def test_the_query_follows_the_documented_api():
    url = fetch.mofxdb_query_url("hMOF", page=2, pld_min=4, elements=["Zn", "O"])
    parsed = urlparse(url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == "https://mof.tech.northwestern.edu/mofs.json"
    assert parse_qs(parsed.query) == {"database": ["hMOF"], "page": ["2"], "pld_min": ["4"],
                                      "elements[]": ["Zn", "O"]}
    with pytest.raises(ValueError, match="Unknown MOFX-DB database"):
        fetch.fetch_mofxdb("unused", database="hmof")


def test_mofxdb_pages_are_read_until_the_limit(tmp_path):
    seen = []
    result = fetch.fetch_mofxdb(tmp_path, database="hMOF", limit=3, pld_min=3,
                                opener=_mofxdb_opener(seen))
    assert len(seen) == 2
    # hMOF-2 has no CIF: skipped with a reason; hMOF/4's slash is made safe.
    assert [p.name for p in result.written] == ["hMOF-1.cif", "hMOF-3.cif", "hMOF_4.cif"]
    assert result.skipped == [("hMOF-2", "no CIF in the response")]

    rows = _read_csv(tmp_path / "reference.csv")
    assert rows[0]["ref_name"] == "hMOF-1" and float(rows[0]["ref_pld_A"]) == 3.25
    assert rows[0]["ref_elements"] == "Zn,O,C,H"
    assert rows[0]["ref_adsorbates"] == "CarbonDioxide,Methane"
    isotherms = json.loads((tmp_path / "hMOF-1.isotherms.json").read_text())
    assert isotherms[0]["temperature_K"] == 298 and isotherms[0]["points"][0] == [0.1, 1.0]
    assert not (tmp_path / "hMOF-3.isotherms.json").exists()

    readme = (tmp_path / "README.txt").read_text()
    assert "10.1021/acs.jced.2c00583" in readme and "Nat. Chem. 4, 83" in readme


def test_a_failed_page_keeps_what_was_written(tmp_path):
    from thevoid_core.services.reference.net import DownloadError

    def opener(url):
        if "page=2" in url:
            raise DownloadError("offline")
        return (FIXTURES / "mofxdb_page1.json").read_bytes()

    with pytest.raises(DownloadError):
        fetch.fetch_mofxdb(tmp_path, limit=10, opener=opener)
    assert (tmp_path / "hMOF-1.cif").exists()
    assert [r["file"] for r in _read_csv(tmp_path / "reference.csv")] == ["hMOF-1.cif"]


# ----------------------------------------------------------------------
# Command lines
# ----------------------------------------------------------------------


def test_fetch_dry_run(capsys):
    from thevoid_core import fetch_cli

    assert fetch_cli.main(["coremof", "--pld-min", "6", "--topology", "pcu", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "downloadable" in out and "CSD-only" in out


def test_progress_in_a_log_is_a_few_lines_not_one_per_update():
    """Round 58: `the-void-fetch ... 2> fetch.log` wrote a carriage-return
    counter per update - 150 overwritten lines for 36 structures. Off a
    terminal it now writes one line per step and per tenth of it, and
    every "Writing <entry>" message counts as the same step."""
    from thevoid_core.fetch_cli import ProgressPrinter

    log = io.StringIO()
    progress = ProgressPrinter(log, interactive=False)
    for done in range(0, 1001, 7):
        progress("Downloading the CoRE MOF archive", done, 1000)
    for number in range(1, 37):
        progress(f"Writing 20{number:02d}[Zn][pcu]3[ASR]1", number, 36)
    lines = log.getvalue().splitlines()
    assert "\r" not in log.getvalue()
    assert len([x for x in lines if x.startswith("Downloading")]) == 10  # tenths 0-9 (1000 not hit)
    writing = [x for x in lines if x.startswith("Writing")]
    # 1/36 is the tenth 0, 36/36 the tenth 10: eleven lines.
    assert len(writing) == 11 and writing[-1] == "Writing structures: 36/36"

    terminal = io.StringIO()
    ProgressPrinter(terminal, interactive=True)("Downloading", 5, 10)
    assert terminal.getvalue().startswith("\rDownloading: 5/10")
    quiet = io.StringIO()
    ProgressPrinter(quiet, interactive=False, quiet=True)("Downloading", 5, 10)
    assert quiet.getvalue() == ""


def test_the_batch_joins_reference_values_and_compares(tmp_path, monkeypatch):
    """A fetched CoRE folder, analysed with both options: the summary
    carries CoRE's own values (ref_) and the like-for-like comparison
    (ccdc_ against coremof_)."""
    from thevoid_core import cli

    folder = tmp_path / "core"
    fetch.fetch_coremof(folder, {"topology": "bpq", "metals": ["Cr"]},
                        opener=lambda url: _archive())
    target = folder / "2010[Cr][bpq]3[ASR]1.cif"
    out = tmp_path / "results"
    status = cli.main([str(target), "--out", str(out), "-q", "--no-surface", "--no-oms",
                       "--no-topology", "--no-pore", "--samples", "200",
                       "--reference-csv", str(folder / "reference.csv"), "--compare-coremof"])
    assert status == 0
    row = _read_csv(out / "summary.csv")[0]
    assert float(row["ref_pld_A"]) == pytest.approx(7.09493)
    assert row["coremof_id"] == "2010[Cr][bpq]3[ASR]1" and row["coremof_match"] == "identified"
    assert float(row["ccdc_pld_A"]) == pytest.approx(7.09493, abs=0.15)
    assert "## Reference: CoRE MOF 2024" in (out / "2010[Cr][bpq]3[ASR]1.md").read_text()


def test_a_reference_csv_without_a_file_column_is_refused(tmp_path):
    from thevoid_core import cli

    bad = tmp_path / "bad.csv"
    bad.write_text("x,y\n1,2\n")
    with pytest.raises(SystemExit) as exit_info:
        cli.main([str(FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"), "--out", str(tmp_path / "r"),
                  "--reference-csv", str(bad)])
    assert exit_info.value.code == 2


# ----------------------------------------------------------------------
# The real services (off by default)
# ----------------------------------------------------------------------


live = pytest.mark.skipif(os.environ.get("THE_VOID_LIVE_NETWORK") != "1",
                          reason="set THE_VOID_LIVE_NETWORK=1 to test against the live services")


@live
def test_live_coremof_archive_matches_the_bundled_index(tmp_path):
    result = fetch.fetch_coremof(tmp_path, {"topology": "bpq", "metals": ["Cr"]}, limit=2)
    assert result.written and not result.warnings
    assert all(r["matches_index"] == "yes" for r in _read_csv(result.reference_csv))


@live
def test_live_mofxdb_returns_hmof_structures(tmp_path):
    result = fetch.fetch_mofxdb(tmp_path, database="hMOF", name="hMOF-1", limit=2)
    assert result.written
    rows = _read_csv(result.reference_csv)
    assert rows[0]["ref_database"] == "hMOF" and rows[0]["ref_pld_A"]
