"""Round 56: the reference-database hooks where a person uses them -
Analysis > Compare with CoRE MOF 2024... and File > Fetch Reference
Structures... No network: the fetch dialog is driven with an opener
that serves a stand-in archive (as in test_reference_fetch.py).
"""

import io
import os
import zipfile
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BPQ = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"


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
    monkeypatch.setattr(win._pool, "start", lambda worker: None)  # topology: not needed
    win._on_structure_ready(win._load_and_analyse(str(CR_BPQ)))
    yield win
    win.close()


def _compare(win, entry_id=None):
    """Run the comparison the way the menu does, synchronously."""
    from thevoid_core.services.reference import compare as compare_module

    original = compare_module.compare_with_coremof

    def quick(structure, match, **kwargs):
        kwargs.setdefault("samples_per_atom", 300)
        return original(structure, match, topology=False, **kwargs)

    compare_module.compare_with_coremof = quick
    try:
        payload = win._run_reference_comparison(
            win.current_structure, win.current_info.file_path, entry_id)
    finally:
        compare_module.compare_with_coremof = original
    win._on_reference_comparison_ready(payload)
    return win._reference_dialog


def test_the_menu_actions_exist(window):
    assert window.compare_coremof_action.text() == "Compare with CoRE MOF 2024..."
    assert "1.655" in window.compare_coremof_action.toolTip()
    assert window.fetch_reference_action.text() == "Fetch Reference Structures..."


def test_the_comparison_window_shows_the_match_then_the_numbers(window):
    dialog = _compare(window)
    assert dialog.isVisible()
    text = dialog.match_label.text()
    assert "2010[Cr][bpq]3[ASR]1" in text and "identified" in text
    labels = [dialog.table.item(r, 0).text() for r in range(dialog.table.rowCount())]
    assert labels[:4] == ["Atoms per cell", "Density", "LCD", "PLD"]
    pld = labels.index("PLD")
    assert dialog.table.item(pld, 2).text() == "7.095 Å"
    assert dialog.table.item(pld, 4).text() == "agrees"
    assert "CCDC" in dialog.notes_label.text() and "CC-BY-4.0" in dialog.notes_label.text()
    assert dialog.selection_text().splitlines()[0].startswith("Property\tThe Void")
    assert "identified" in window.statusBar().currentMessage()


def test_the_comparison_goes_into_the_report(window):
    _compare(window)
    report = window.build_analysis_report()
    assert report["results"]["reference"]["entry"] == "2010[Cr][bpq]3[ASR]1"
    from thevoid_core.services.report import to_markdown

    assert "## Reference: CoRE MOF 2024" in to_markdown(report)


def test_another_candidate_can_be_chosen(window):
    dialog = _compare(window)
    ids = [dialog.candidate_box.itemData(i) for i in range(dialog.candidate_box.count())]
    assert "2010[Cr][bpq]3[FSR]1" in ids  # the same structure's FSR entry
    dialog = _compare(window, "2010[Cr][bpq]3[FSR]1")
    assert window.current_reference_comparison.entry_id == "2010[Cr][bpq]3[FSR]1"
    assert dialog.candidate_box.currentData() == "2010[Cr][bpq]3[FSR]1"


def test_loading_another_structure_clears_the_comparison(window):
    _compare(window)
    from tests.sample_structures import ZN_DIA

    window._on_structure_ready(window._load_and_analyse(str(ZN_DIA)))
    assert window.current_reference_comparison is None
    assert not window._reference_dialog.isVisible()


# ----------------------------------------------------------------------
# Fetch dialog
# ----------------------------------------------------------------------


def _archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for path in sorted((FIXTURES / "coremof").glob("*.cif")):
            archive.writestr(f"CR/{path.name}", path.read_text())
    return buffer.getvalue()


def test_the_fetch_dialog_plans_offline_and_fetches(qt_app, tmp_path, monkeypatch):
    from thevoid_core.services.reference import fetch, net
    from thevoid_core.ui.dialogs.fetch_reference_dialog import FetchReferenceDialog

    monkeypatch.setenv(net.CACHE_ENVIRONMENT_VARIABLE, str(tmp_path / "cache"))
    original = fetch.fetch_coremof
    monkeypatch.setattr(fetch, "fetch_coremof",
                        lambda **kw: original(opener=lambda url: _archive(), **kw))

    dialog = FetchReferenceDialog(default_folder=str(tmp_path / "out"))
    dialog.topology_edit.setText("bpq")
    dialog.metals_edit.setText("Cr")
    assert "to download" in dialog.plan_label.text()
    assert "not_downloadable.csv" in dialog.plan_label.text()

    dialog.pld_min.setText("wide")
    assert "not a number" in dialog.plan_label.text()
    assert not dialog.fetch_button.isEnabled()
    dialog.pld_min.setText("")

    assert dialog.start()
    QThreadPool.globalInstance().waitForDone(60000)
    qt_app.processEvents()
    assert dialog.result is not None
    assert (tmp_path / "out" / "2010[Cr][bpq]3[ASR]1.cif").exists()
    assert "the-void-analyze" in dialog.status_label.text()
    assert "--compare-coremof" in dialog.status_label.text()
    assert dialog.open_folder_button.isEnabled()
    dialog.close()


def test_the_main_window_opens_the_fetch_dialog(window):
    dialog = window.fetch_reference_structures()
    assert dialog.isVisible()
    assert window.fetch_reference_structures() is dialog
    dialog.source_box.setCurrentIndex(1)
    assert "online" in dialog.plan_label.text()
    dialog.close()
