"""Round 51 GUI features: pore size distribution, analysis report
export, structure density, Help menu, file logging, exception hook."""

import json
import logging
import os
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

FIXTURE = ZN_DIA  # an open Zn MOF


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def window(qt_app, monkeypatch):
    win = MainWindow()
    # Round 67: no background work (the topology, the shared-index
    # lookup). A topology worker that finished before the report was
    # built put "topology" in it, so the first assertion below failed
    # when the machine was busy; nothing here is about background work.
    monkeypatch.setattr(win._pool, "start", lambda worker: None)
    shown = []
    monkeypatch.setattr(win, "_show_psd_dialog", shown.append)
    win._shown_psd = shown
    payload = win._load_and_analyse(str(FIXTURE))
    win._on_structure_ready(payload)
    yield win
    win.close()


def _menu_titles(window):
    return [a.text() for a in window.menuBar().actions()]


def test_help_is_the_last_menu(window):
    titles = _menu_titles(window)
    assert titles[-1] == "Help"
    # Round 52 added Simulate (RASPA2 export) between Analysis and Measure;
    # round 59 Build (functionalised variants) before it.
    assert titles[:-1] == ["File", "View", "Analysis", "Build", "Simulate", "Measure"]


def test_new_actions_exist(window):
    assert window.psd_action.text() == "Pore Size Distribution..."
    assert window.export_report_action.shortcut().toString() == "Ctrl+E"
    for action in (window.psd_action, window.export_report_action,
                   window.about_action, window.open_log_action,
                   window.copy_diagnostics_action):
        assert action.toolTip()


def test_structure_panel_shows_volume_and_density(window):
    structure = window.current_structure
    assert window.info_panel.volume_label.text() == f"{structure.lattice.volume:.2f}"
    assert float(window.info_panel.density_label.text()) == pytest.approx(
        float(structure.density), rel=1e-4
    )


def test_psd_runs_and_is_cached(window):
    result = window._run_psd(window.current_structure, window.viewer.radius_provider)
    window._on_psd_ready(result)

    assert window.current_psd_result is result
    assert window._shown_psd == [result]
    assert "Pore size distribution" in window.statusBar().currentMessage()


def test_report_gathers_only_what_was_computed(window):
    report = window.build_analysis_report()
    assert report["results"] == {}
    assert report["structure"]["space_group"] == window.current_info.space_group

    oms = window._run_oms_detection(window.current_structure, window.current_site_labels)
    window._on_oms_detection_ready(oms)
    result = window._run_psd(window.current_structure, window.viewer.radius_provider)
    window._on_psd_ready(result)

    report = window.build_analysis_report()
    assert set(report["results"]) == {"oms", "psd"}


def test_report_export_md_and_json(window, tmp_path):
    oms = window._run_oms_detection(window.current_structure, window.current_site_labels)
    window._on_oms_detection_ready(oms)

    md = tmp_path / "r.md"
    js = tmp_path / "r.json"
    assert window.export_analysis_report(str(md))
    assert window.export_analysis_report(str(js))

    assert "## Open metal sites" in md.read_text(encoding="utf-8")
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["results"]["oms"]["metal_centres_analyzed"] == 4
    assert data["input"]["file"].endswith(FIXTURE.name)


def test_loading_a_new_structure_forgets_old_results(window):
    result = window._run_psd(window.current_structure, window.viewer.radius_provider)
    window._on_psd_ready(result)
    window.current_xrd_result = object()

    payload = window._load_and_analyse(str(FIXTURE))
    window._on_structure_ready(payload)

    assert window.current_psd_result is None
    assert window.current_xrd_result is None


def test_report_without_structure_is_refused(qt_app):
    win = MainWindow()
    try:
        assert win.build_analysis_report() is None
        assert win.export_analysis_report("/nonexistent/x.md") is False
    finally:
        win.close()


def test_psd_dialog_plots_and_exports(qt_app, tmp_path):
    from thevoid_core.data.element_radii import ElementRadiusProvider
    from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader
    from thevoid_core.services.pore_size_distribution import compute_pore_size_distribution
    from thevoid_core.ui.dialogs.psd_dialog import PoreSizeDistributionDialog

    structure = PymatgenCIFLoader().load(str(FIXTURE)).structure
    result = compute_pore_size_distribution(structure, radius_provider=ElementRadiusProvider())
    dialog = PoreSizeDistributionDialog(result, title="Zn-dia")

    assert "Zn-dia" in dialog.windowTitle()
    assert len(dialog.axes.patches) == len(result.bin_centers)
    assert "dominant pore diameter" in dialog.summary_label.text()

    target = tmp_path / "psd"
    assert dialog.export_csv(str(target))
    lines = (tmp_path / "psd.csv").read_text().splitlines()
    assert lines[1].startswith("diameter_A")
    assert len(lines) == 2 + len(result.bin_centers)


def test_about_dialog_lists_versions_and_references(qt_app):
    from thevoid_core.ui.dialogs.about_dialog import REFERENCES, AboutDialog
    from thevoid_core.version import __version__

    dialog = AboutDialog(log_path="/tmp/x.log")
    assert __version__ in dialog.versions_label.text()
    html = dialog.methods.toPlainText()
    assert "Willems" in html and "Gelb" in html
    assert len(REFERENCES) >= 6


def test_diagnostics_include_version_file_and_log(window, tmp_path, monkeypatch):
    import thevoid_core.infrastructure.logging as log_module
    from thevoid_core.version import __version__

    log = tmp_path / "the_void.log"
    log.write_text("line one\nline two\n")
    monkeypatch.setattr(log_module, "log_file_path", lambda: log)

    text = window.diagnostics_text()
    assert f"the Void {__version__}" in text
    assert FIXTURE.name in text
    assert "line two" in text

    window.copy_diagnostics()
    assert QApplication.clipboard().text() == window.diagnostics_text()


def test_file_logging_goes_to_the_data_directory(tmp_path, monkeypatch):
    import thevoid_core.infrastructure.logging as log_module
    from thevoid_core.infrastructure.settings import Settings

    monkeypatch.setattr(Settings, "data_directory", staticmethod(lambda: tmp_path))
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        path = log_module.configure_logging()
        assert path == tmp_path / "logs" / "the_void.log"
        logging.getLogger("thevoid_core.test").warning("hello log")
        for handler in root.handlers:
            handler.flush()
        assert "hello log" in path.read_text()
        # Idempotent: calling again adds no second file handler.
        log_module.configure_logging()
        assert sum(
            isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path
            for h in root.handlers
        ) == 1
    finally:
        for handler in list(root.handlers):
            if handler not in before:
                root.removeHandler(handler)
                handler.close()


def test_exception_hook_logs_and_reports(qt_app, monkeypatch, caplog):
    import sys

    from thevoid_core.app import application

    shown = []
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(a)))
    original = sys.excepthook
    try:
        application.install_exception_hook()
        try:
            raise ValueError("boom in a slot")
        except ValueError:
            with caplog.at_level(logging.CRITICAL):
                sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = original

    assert "boom in a slot" in caplog.text
    assert shown and "Copy Diagnostics" in shown[0][2]
