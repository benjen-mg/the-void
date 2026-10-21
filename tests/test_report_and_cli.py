"""Round 51: analysis reports and the `the-void-analyze` batch tool."""

import csv
import io
import json
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from thevoid_core import cli  # noqa: E402
from thevoid_core.services import report as reporting  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

FIXTURE = ZN_DIA  # an open Zn MOF

#: Fast settings: the full surface calculation is exercised elsewhere.
FAST = ["--samples", "200"]


def _structure():
    from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader

    return PymatgenCIFLoader().load(str(FIXTURE)).structure


@pytest.fixture(scope="module")
def full_report():
    config = reporting.AnalysisConfig(
        psd=True, xrd=True, guests=["CO2", "tiny=1.0"],
        surface_samples_per_atom=200,
    )
    return reporting.run_analysis(_structure(), config, source=str(FIXTURE))


def test_density_matches_pymatgen():
    structure = _structure()
    assert reporting.crystal_density(structure) == pytest.approx(
        float(structure.density), rel=1e-6
    )


def test_report_records_provenance(full_report):
    p = full_report["provenance"]
    assert p["software"]["the_void"]
    assert p["software"]["pymatgen"] != "unknown"
    assert full_report["input"]["sha256"] == reporting.file_sha256(FIXTURE)
    assert len(full_report["input"]["sha256"]) == 64
    assert set(reporting.METHOD_NOTES) <= set(full_report["methods"])
    assert full_report["config"]["surface_samples_per_atom"] == 200


def test_report_contains_every_requested_result(full_report):
    results = full_report["results"]
    assert full_report["errors"] == {}
    # Round 53 added topology, on by default.
    assert set(results) == {"pore", "surface_area", "oms", "psd", "xrd", "guests", "topology"}
    assert results["pore"]["pld_A"] == pytest.approx(4.078, abs=0.01)
    assert results["pore"]["lcd_pld_method"] == "voronoi"
    assert results["guests"]["tiny"]["status"] == "accessible"
    assert results["guests"]["CO2"]["guest_diameter_A"] == pytest.approx(3.05)
    assert results["oms"]["metal_centres_analyzed"] == 4
    assert results["xrd"]["strongest_peaks"]
    assert results["psd"]["bins"]


def test_json_round_trips_and_is_strict(full_report):
    text = reporting.to_json(full_report)
    # Strict JSON: no NaN / Infinity tokens.
    data = json.loads(text, parse_constant=lambda name: pytest.fail(name))
    assert data["structure"]["formula"] == full_report["structure"]["formula"]


def test_markdown_has_every_section(full_report):
    text = reporting.to_markdown(full_report)
    for heading in ("## Structure", "## Pores", "## Surface area",
                    "## Guest accessibility", "## Open metal sites",
                    "## Pore size distribution", "## Powder XRD", "## Methods"):
        assert heading in text
    assert full_report["input"]["sha256"] in text


def test_a_failing_analysis_is_recorded_and_the_rest_still_run(monkeypatch):
    import thevoid_core.services.oms_detection as oms

    def boom(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(oms, "detect_open_metal_sites", boom)

    report = reporting.run_analysis(
        _structure(),
        reporting.AnalysisConfig(surface_area=False),
    )

    assert "oms" not in report["results"]
    assert "synthetic failure" in report["errors"]["oms"]
    assert "pore" in report["results"]
    assert "## Analyses that failed" in reporting.to_markdown(report)


def test_guest_parsing():
    assert reporting._guest_diameter("N2") == ("N2", pytest.approx(3.64))
    assert reporting._guest_diameter("benzene=5.85") == ("benzene", 5.85)
    with pytest.raises(ValueError):
        reporting._guest_diameter("unobtainium")


def test_summary_csv_has_a_column_per_guest(full_report):
    text = reporting.summary_csv([reporting.summary_row(full_report)])
    rows = list(csv.DictReader(io.StringIO(text)))
    assert rows[0]["file"] == FIXTURE.name
    assert rows[0]["guest_tiny"] == "accessible"
    assert float(rows[0]["pld_A"]) == pytest.approx(4.078, abs=0.01)


# --- command line ---------------------------------------------------------


def test_cli_writes_reports_and_summary(tmp_path, capsys):
    out = tmp_path / "out"
    code = cli.main([str(FIXTURE), "--out", str(out), "--psd", *FAST])

    assert code == 0
    assert (out / f"{FIXTURE.stem}.json").is_file()
    assert (out / f"{FIXTURE.stem}.md").is_file()
    data = json.loads((out / f"{FIXTURE.stem}.json").read_text())
    assert data["structure"]["space_group"] == "P2_12_12"
    assert "psd" in data["results"]
    rows = list(csv.DictReader(io.StringIO((out / "summary.csv").read_text())))
    assert len(rows) == 1 and rows[0]["errors"] == ""
    assert "done" in capsys.readouterr().out


def test_cli_batch_in_parallel_keeps_input_order(tmp_path):
    inputs = []
    for name in ("b_second", "a_first"):
        target = tmp_path / f"{name}.cif"
        shutil.copy(FIXTURE, target)
        inputs.append(str(target))

    out = tmp_path / "out"
    code = cli.main([*inputs, "--out", str(out), "-j", "2", "-q",
                     "--no-surface", "--format", "json"])

    assert code == 0
    rows = list(csv.DictReader(io.StringIO((out / "summary.csv").read_text())))
    assert [r["file"] for r in rows] == ["b_second.cif", "a_first.cif"]
    assert not list(out.glob("*.md"))


def test_cli_reports_missing_and_unreadable_files(tmp_path):
    broken = tmp_path / "broken.cif"
    broken.write_text("data_nothing\n")
    out = tmp_path / "out"

    code = cli.main([str(tmp_path / "missing.cif"), str(broken), "--out", str(out), "-q"])

    assert code == 1
    rows = {r["file"]: r for r in csv.DictReader(io.StringIO((out / "summary.csv").read_text()))}
    assert rows["missing.cif"]["errors"] == "file not found"
    assert "could not read structure" in rows["broken.cif"]["errors"]


def test_cli_rejects_bad_arguments(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(FIXTURE), "--guests", "unobtainium", "--out", str(tmp_path)])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(FIXTURE), "-j", "0", "--out", str(tmp_path)])
    assert excinfo.value.code == 2


def test_cli_refuses_inputs_that_would_overwrite_each_other(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    a = tmp_path / "x" / "same.cif"
    b = tmp_path / "y" / "same.cif"
    shutil.copy(FIXTURE, a)
    shutil.copy(FIXTURE, b)

    with pytest.raises(SystemExit) as excinfo:
        cli.main([str(a), str(b), "--out", str(tmp_path / "out")])
    assert excinfo.value.code == 2


def test_cli_does_not_import_qt():
    import subprocess

    code = (
        "import sys, thevoid_core.cli, thevoid_core.services.report;"
        "import thevoid_core.services.pore_size_distribution;"
        "sys.exit(any(m.startswith('PySide6') for m in sys.modules))"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent.parent)
    assert result.returncode == 0


def test_cli_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    from thevoid_core.version import __version__
    assert __version__ in capsys.readouterr().out
