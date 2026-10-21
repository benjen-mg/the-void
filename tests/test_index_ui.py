"""Round 66: the shared index where a person uses it - File > Shared
Index (Add Current Structure..., Open Index..., Set Index Folder...),
Batch Results > Add to Shared Index..., and the Structure panel's
Previously Analysed line.

Background work is run synchronously here, the way the menu starts it
(test_reference_ui.py does the same), and the settings go to a private
file, so a test never touches a real user's index folder.
"""

import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"
CU2 = FIXTURES / "identity" / "2022[Cu][sql]2[ASR]2.cif"
CU4 = FIXTURES / "identity" / "2022[Cu][sql]2[ASR]4.cif"


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    """A the-void-analyze folder of the two Cu determinations."""
    from thevoid_core import cli

    root = tmp_path_factory.mktemp("ui")
    for path in (CU2, CU4):
        shutil.copy(path, root / path.name)
    cli.main([str(root / CU2.name), str(root / CU4.name), "--out", str(root / "results"), "-q",
              "--samples", "100", "--no-topology", "--no-oms"])
    return root


@pytest.fixture
def window(qt_app, monkeypatch, tmp_path):
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from thevoid_core.ui.windows import main_window as module

    settings_file = str(tmp_path / "settings.ini")
    monkeypatch.setattr(module.MainWindow, "_settings",
                        staticmethod(lambda: QSettings(settings_file, QSettings.Format.IniFormat)))
    messages = []
    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, _n=name, **k: messages.append((_n, a))))
    win = module.MainWindow()
    win.messages = messages
    started = []
    monkeypatch.setattr(win._pool, "start", lambda worker: started.append(worker))
    win.started = started
    yield win
    win.close()


def _index(tmp_path, name="UI Lab"):
    from thevoid_core.services.index.store import Index

    return Index.init(tmp_path / "idx", name=name)


def _load(window, path):
    window._on_structure_ready(window._load_and_analyse(str(path)))


def _run_lookup(window):
    """Run the Previously Analysed lookup the load started."""
    worker = next(w for w in reversed(window.started) if w.fn == window._find_in_index)
    window._on_index_lookup_ready(worker.fn(*worker.args, **worker.kwargs))


def _values(index, **extra):
    return {"index": str(index.path), "contributor": "A. B.", "note": "", **extra,
            "declared_licence": extra.get("declared_licence")}


# ----------------------------------------------------------------------


def test_the_menu_and_the_panel_row_exist(window):
    assert [a.text() for a in window.shared_index_menu.actions() if a.text()] == [
        "Add Current Structure...", "Open Index...", "Set Index Folder..."]
    assert "never a CSD-derived file" in window.index_add_action.toolTip()
    _load(window, CR_BDC)
    assert window.info_panel.index_label.text() == "no shared index set"


def test_setting_an_index_looks_the_structure_up(window, tmp_path, results):
    index = _index(tmp_path)
    index.add_results(results / "results", contributor="C. D.")
    window.choose_shared_index(str(index.path))
    assert window.shared_index_path == str(index.path)
    _load(window, results / CU4.name)
    assert window.info_panel.index_label.text() == "looking in the shared index…"
    _run_lookup(window)
    label = window.info_panel.index_label
    assert label.text() == "yes in UI Lab: 1 same file, 1 same structure"
    assert "C. D." in label.toolTip() and CU2.name in label.toolTip()
    _load(window, CR_BDC)
    _run_lookup(window)
    assert window.info_panel.index_label.text() == "not in UI Lab"


def test_a_stale_lookup_is_ignored(window, tmp_path, results):
    index = _index(tmp_path)
    index.add_results(results / "results", contributor="C. D.")
    window.choose_shared_index(str(index.path))
    _load(window, results / CU4.name)
    old = next(w for w in reversed(window.started) if w.fn == window._find_in_index)
    _load(window, CR_BDC)  # another structure before the first lookup came back
    window._on_index_lookup_ready(old.fn(*old.args, **old.kwargs))
    assert window.info_panel.index_label.text() == "looking in the shared index…"


def test_adding_the_current_structure(window, tmp_path):
    index = _index(tmp_path)
    window.choose_shared_index(str(index.path))
    _load(window, CR_BDC)
    window.add_current_to_index(_values(index))
    assert window.messages[-1][0] == "information"
    assert "No analysis has been run" in window.messages[-1][1][2]
    from thevoid_core.services.pore_analysis import analyze_pores
    from thevoid_core.data.element_radii import ElementRadiusProvider

    window.current_pore_result = analyze_pores(window.current_structure, probe_radius=1.2,
                                               radius_provider=ElementRadiusProvider())
    report = window.build_analysis_report()
    result = window._run_index_add(_values(index), "report", report,
                                   window.current_info.file_path)
    window._on_index_add_ready(result)
    assert len(result.added) == 1 and result.withheld  # no licence known for this copy
    assert "1 added" in window.messages[-1][1][2]
    entry = next(iter(index.load_entries().values()))
    assert entry["analysis"]["config"]["inferred_from_results"] is True
    assert entry["analysis"]["summary"]["pld_A"] == pytest.approx(
        window.current_pore_result.pld)


def test_adding_rows_from_batch_results(window, tmp_path, results):
    index = _index(tmp_path)
    batch = window.open_batch_results(str(results / "results"))
    assert batch.index_button.isEnabled()
    batch.table.selectRow(0)
    first = batch.rows_to_add()
    assert len(first) == 1
    called = {}
    window.add_results_to_index = lambda folder, only=None, values=None: called.update(
        folder=folder, only=only)
    batch.add_to_index(values=_values(index))
    assert called == {"folder": str(results / "results"), "only": first}
    # What the window then runs: only those rows go in.
    result = type(window)._run_index_add(window, _values(index), "results",
                                         called["folder"], called["only"])
    assert len(result.added) == 1


def test_opening_the_index_builds_it_and_shows_it(window, tmp_path, results):
    index = _index(tmp_path)
    index.add_results(results / "results", contributor="C. D.")
    window.choose_shared_index(str(index.path))
    built = window._run_index_build(str(index.path))
    window._on_index_build_ready(built)
    batch = window.batch_window
    assert batch.isVisible() and batch.is_index_build()
    assert not batch.index_button.isEnabled()
    assert batch.windowTitle().startswith("Shared Index: idx")
    assert len(batch.results.rows) == 2 and "structure_id" in batch.results.columns
    assert "2 entries, 1 structures, 2 determinations" in window.statusBar().currentMessage()


def test_the_add_dialog_needs_an_index_a_contributor_and_a_reason(qt_app, tmp_path):
    from PySide6.QtWidgets import QDialogButtonBox

    from thevoid_core.ui.dialogs.shared_index_dialog import SharedIndexAddDialog, describe_add

    index = _index(tmp_path)
    dialog = SharedIndexAddDialog(what="2 rows", index_path=str(tmp_path), contributor="")
    ok = dialog.buttons.button(QDialogButtonBox.StandardButton.Ok)
    assert not ok.isEnabled()                     # not an index, no contributor
    dialog.path_edit.setText(str(index.path))
    dialog.contributor_edit.setText("A. B.")
    assert ok.isEnabled()
    dialog.licence_box.setCurrentIndex(1)         # CC-BY-4.0: needs a reason
    assert not ok.isEnabled() and dialog.basis_edit.isEnabled()
    dialog.basis_edit.setText("our own structures")
    assert ok.isEnabled()
    assert dialog.values()["declared_licence"] == {"spdx": "CC-BY-4.0",
                                                   "basis": "our own structures"}
    from thevoid_core.services.index.store import AddResult

    text = describe_add(AddResult(added=["x"], withheld=[("csd_download.cif", "CSD header in file")]))
    assert "1 added" in text and "csd_download.cif: CSD header in file" in text


def test_the_tooltip_says_when_an_entry_is_stale(qt_app):
    """Round 67: an entry a later version computes differently says so."""
    from thevoid_core.ui.widgets.structure_info_panel import StructureInfoPanel

    entry = {"entry_id": "e", "structure": {"file_name": "x.cif", "file_sha256": "s"},
             "contributor": {"name": "A. B.", "added": "2026-09-01"},
             "analysis": {"software": {"the_void": "0.9.2"},
                          "summary": {"pld_A": 5.0, "channel_asa_m2_per_g": 1000.0}}}
    panel = StructureInfoPanel()
    panel.set_index_matches([("same file", entry)], "Lab")
    tip = panel.index_label.toolTip()
    assert "the Void 0.9.2" in tip and "(stale: channel_asa_m2_per_g)" in tip
