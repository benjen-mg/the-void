"""Round 53: topology where a person sees it - the Structure panel,
the analysis report, and the command line's per-structure reports and
summary CSV.
"""

import csv
import io
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from thevoid_core import cli  # noqa: E402
from thevoid_core.services import report as reporting  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.topology import TopologyResult  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

FIXTURE = ZN_DIA  # an open Zn MOF
FAST = ["--samples", "200", "--no-surface", "--no-oms", "--no-pore"]


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qt_app, monkeypatch):
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from thevoid_core.ui.windows.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    win = MainWindow()
    started = []
    # Record, rather than run, the background job; tests drive it.
    monkeypatch.setattr(win._pool, "start", lambda worker: started.append(worker))
    win._started = started
    win._on_structure_ready(win._load_and_analyse(str(FIXTURE)))
    yield win
    win.close()


def test_loading_starts_topology_in_the_background(window):
    assert len(window._started) == 1
    assert window.info_panel.topology_label.text() == "computing…"
    assert not window._busy  # other actions stay available


def test_the_panel_shows_the_net_and_explains_it(window):
    token = window._topology_token
    window._on_topology_ready(window._compute_topology(
        window.current_structure, window.current_bonds, token))

    label = window.info_panel.topology_label
    assert label.text() == "dia 4-c"
    tip = label.toolTip()
    assert "C6H5N2O2" in tip and "single-node" in tip and "bundled curated" in tip


def test_a_result_for_an_older_structure_is_ignored(window):
    stale = window._compute_topology(
        window.current_structure, window.current_bonds, window._topology_token)
    window._on_structure_ready(window._load_and_analyse(str(FIXTURE)))
    window._on_topology_ready(stale)
    assert window.current_topology_result is None
    assert window.info_panel.topology_label.text() == "computing…"


def test_the_report_carries_topology_like_space_group(window, tmp_path):
    window._on_topology_ready(window._compute_topology(
        window.current_structure, window.current_bonds, window._topology_token))
    report = window.build_analysis_report()
    assert report["structure"]["topology"] == "dia 4-c"
    assert report["results"]["topology"]["rcsr"] == "dia"

    md = tmp_path / "r.md"
    assert window.export_analysis_report(str(md))
    text = md.read_text(encoding="utf-8")
    assert "| Topology (RCSR) | dia 4-c |" in text
    assert "## Topology" in text and "| Zn | sbu | 4-c |" in text


def test_panel_states_why_a_net_was_not_named(qt_app):
    from thevoid_core.ui.widgets.structure_info_panel import StructureInfoPanel

    panel = StructureInfoPanel()
    panel.set_topology(TopologyResult(status="rod_sbu", message="The inorganic SBU is an infinite rod."))
    assert panel.topology_label.text() == "not identified: rod/layer SBU"
    assert "infinite rod" in panel.topology_label.toolTip()
    panel.set_topology(None)
    assert panel.topology_label.text() == "-"


def test_cli_reports_topology_in_every_output(tmp_path):
    out = tmp_path / "out"
    assert cli.main([str(FIXTURE), "--out", str(out), "-q", *FAST]) == 0
    report = json.loads((out / f"{FIXTURE.stem}.json").read_text())
    assert report["structure"]["topology"] == "dia 4-c"
    assert "topology" in report["methods"]
    rows = list(csv.DictReader(io.StringIO((out / "summary.csv").read_text())))
    assert rows[0]["topology"] == "dia 4-c"
    assert "| Topology (RCSR) | dia 4-c |" in (out / f"{FIXTURE.stem}.md").read_text()


def test_cli_can_skip_topology_or_use_a_full_archive(tmp_path):
    out = tmp_path / "skip"
    cli.main([str(FIXTURE), "--out", str(out), "-q", *FAST, "--no-topology"])
    assert "topology" not in json.loads((out / f"{FIXTURE.stem}.json").read_text())["results"]

    archive = tmp_path / "only_pcu.arc"
    archive.write_text("key      3 1 1 -1 0 0 1 1 0 -1 0 1 1 0 0 -1\nid       pcu\nend\n")
    out = tmp_path / "arc"
    cli.main([str(FIXTURE), "--out", str(out), "-q", *FAST, "--rcsr-archive", str(archive)])
    topology = json.loads((out / f"{FIXTURE.stem}.json").read_text())["results"]["topology"]
    assert topology["status"] == "not_in_library"
    assert "only_pcu.arc" in topology["library"]

    with pytest.raises(SystemExit) as info:
        cli.main([str(FIXTURE), "--out", str(out), "--rcsr-archive", str(tmp_path / "none.arc")])
    assert info.value.code == 2


def test_a_topology_failure_is_recorded_and_the_rest_runs(monkeypatch):
    import thevoid_core.services.topology as topology

    def broken(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(topology, "analyze_topology", broken)
    structure = PymatgenCIFLoader().load(str(FIXTURE)).structure
    report = reporting.run_analysis(
        structure, reporting.AnalysisConfig(surface_area=False, oms=False)
    )
    assert report["errors"]["topology"] == "RuntimeError: boom"
    assert "pore" in report["results"]


def test_the_reference_nets_ship_in_the_package():
    """The wheel only contains non-Python files that pyproject lists. It
    listed none, so an installed copy had no icons - and would have had
    no reference nets."""
    import tomllib

    config = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["thevoid_core"]
    assert "data/*.json" in patterns and "assets/*.png" in patterns
