"""Round 54: the Batch Results window - what a person clicks.

The ranking and filtering rules are tested in test_batch_results.py;
these check that the window applies them the way its controls say it
does (a header click ranks, a filter row filters, a bad value is shown
as ignored rather than silently dropping every row), and that opening a
row reaches the main window only with a verified file, or with the
person's explicit say-so.
"""

import csv
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from thevoid_core.ui.windows.batch_results_window import BatchResultsWindow  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

FIXTURE = ZN_DIA  # an open Zn MOF


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class FakeMain:
    """Just what the window uses of the main window."""

    def __init__(self):
        self.opened = []
        self._busy = False
        self.messages = []

    def _open_file(self, path):
        self.opened.append(path)

    def statusBar(self):  # noqa: N802
        main = self

        class Bar:
            def showMessage(self, text, timeout=0):  # noqa: N802
                main.messages.append(text)

        return Bar()

    def raise_(self):
        pass


@pytest.fixture(scope="module")
def batch_template(tmp_path_factory):
    """A results folder with one real, checksummed report (an open Zn MOF, pore
    analysis only) and three summary-only rows. Built once; each test
    gets its own copy, since some move or edit files."""
    pytest.importorskip("pymatgen")
    from thevoid_core import cli

    project = tmp_path_factory.mktemp("template") / "project"
    (project / "cifs").mkdir(parents=True)
    shutil.copy(FIXTURE, project / "cifs" / "sample.cif")
    old = os.getcwd()
    os.chdir(project)
    try:
        cli.main(["cifs/sample.cif", "--out", "results", "-q", "--no-surface",
                  "--no-oms", "--no-topology"])
    finally:
        os.chdir(old)

    summary = project / "results" / "summary.csv"
    rows = list(csv.DictReader(open(summary)))
    columns = list(rows[0])
    extra = [
        {**{c: "" for c in columns}, "file": "big.cif", "pld_A": "6.5", "lcd_A": "13.2"},
        {**{c: "" for c in columns}, "file": "small.cif", "pld_A": "1.1", "lcd_A": "3.0"},
        {**{c: "" for c in columns}, "file": "failed.cif", "errors": "pore: Boom"},
    ]
    with open(summary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows + extra)
    return project


@pytest.fixture
def batch(batch_template, tmp_path):
    project = tmp_path / "project"
    shutil.copytree(batch_template, project)
    return project


@pytest.fixture
def window(qt_app, batch):
    main = FakeMain()
    win = BatchResultsWindow(main_window=main)
    assert win.load(batch / "results")
    yield win
    win.close()


def _files(win):
    return [row.get("file") for row in win.visible_rows]


def test_opening_a_folder_lists_every_structure(window):
    assert _files(window) == ["sample.cif", "big.cif", "small.cif", "failed.cif"]
    assert window.count_label.text() == "4 of 4 shown"
    assert window.model.headerData(0, Qt.Orientation.Horizontal) == "#"


def test_a_header_click_ranks_and_empty_cells_stay_last(window):
    section = window.results.columns.index("pld_A") + 1
    window.model.sort(section, Qt.SortOrder.DescendingOrder)
    assert _files(window) == ["big.cif", "sample.cif", "small.cif", "failed.cif"]
    assert "ranked by PLD" in window.count_label.text()
    assert window.model.data(window.model.index(0, 0)) == "1"
    window.model.sort(0, Qt.SortOrder.AscendingOrder)  # "#" = file order
    assert _files(window) == ["sample.cif", "big.cif", "small.cif", "failed.cif"]


def test_filter_rows_filter_and_a_bad_value_is_shown_not_swallowed(window):
    row = window.add_filter("pld_A", ">=", "2")
    assert _files(window) == ["sample.cif", "big.cif"]

    row.value_edit.setText("wide")
    assert window.filter_error.isVisibleTo(window)
    assert "needs a number" in window.filter_error.text()
    assert len(window.visible_rows) == 4  # ignored, not "no rows match"

    window._remove_filter(row)
    assert not window.filter_error.isVisibleTo(window)


def test_top_n_and_search(window):
    window.sort_by("lcd_A", descending=True)
    window.top_box.setValue(1)
    assert _files(window) == ["big.cif"]
    window.top_box.setValue(0)
    window.search_edit.setText("boom")
    assert _files(window) == ["failed.cif"]


def test_selecting_a_row_shows_its_report(window):
    window.table.selectRow(0)
    assert window.report_view.toPlainText().startswith("Analysis report")
    assert window.open_3d_button.isEnabled()
    window.table.selectRow(3)
    assert "pore: Boom" in window.report_view.toPlainText()


def test_opening_a_row_loads_the_verified_file_in_the_main_window(window, batch):
    path = window.open_row(0)
    assert path == (batch / "cifs" / "sample.cif").resolve()
    assert window.main_window.opened == [str(path)]
    assert "SHA-256 matches" in window.main_window.messages[-1]


def test_a_missing_file_is_located_once_and_remembered(window, batch, tmp_path, monkeypatch):
    shutil.move(str(batch / "cifs"), str(tmp_path / "archive"))
    picked = str(tmp_path / "archive" / "sample.cif")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (picked, "")))
    assert window.open_row(0) == Path(picked)
    assert tmp_path / "archive" in window.extra_dirs

    # Next time no dialog is needed.
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: pytest.fail("asked again")))
    assert window.open_row(0) == (tmp_path / "archive" / "sample.cif").resolve()


def test_a_different_file_is_only_loaded_if_the_person_says_so(window, batch, monkeypatch):
    target = batch / "cifs" / "sample.cif"
    target.write_text(target.read_text() + "\n# edited\n")
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    assert window.open_row(0) is None
    assert window.main_window.opened == []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    assert window.open_row(0) is not None


def test_a_busy_main_window_is_not_given_a_file(window, monkeypatch):
    window.main_window._busy = True
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    assert window.open_row(0) is None
    assert window.main_window.opened == []


def test_the_view_exports_and_copies_as_shown(window, tmp_path):
    window.sort_by("pld_A", descending=True)
    path = tmp_path / "ranked.csv"
    assert window.export_view(str(path))
    assert [r["file"] for r in csv.DictReader(open(path))][:2] == ["big.cif", "sample.cif"]

    window.table.selectRow(0)
    text = window.selection_text()
    assert text.splitlines()[0].startswith("rank\tfile")
    assert text.splitlines()[1].startswith("1\tbig.cif")


def test_a_folder_that_is_not_a_batch_is_refused_politely(qt_app, tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(a[2])))
    win = BatchResultsWindow()
    assert not win.load(tmp_path)
    assert "No summary.csv" in shown[0]
    win.close()


def test_the_main_window_opens_it(qt_app, batch, monkeypatch):
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from thevoid_core.ui.windows.main_window import MainWindow

    main = MainWindow()
    try:
        assert main.open_batch_action.text() == "Open Batch Results..."
        assert main.open_batch_action.toolTip()
        win = main.open_batch_results(str(batch / "results"))
        assert len(win.visible_rows) == 4
        assert main.open_batch_results(str(batch / "results")) is win  # reused
    finally:
        main.close()
