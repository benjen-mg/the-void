"""Round 49: copy/export for results tables (table_tools)."""

import os

import pytest

PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QItemSelectionModel  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QTableWidget,
    QTableWidgetItem,
)

from thevoid_core.ui.widgets.table_tools import (  # noqa: E402
    copy_table,
    install_table_tools,
    rows_to_csv,
    rows_to_tsv,
    table_rows,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _table():
    table = QTableWidget(3, 3)
    table.setHorizontalHeaderLabels(["Label", "Element", "Distance (Å)"])
    data = [["O1", "O", "2.001"], ["O2", "O", "2.010"], ["N1", "N", "2.150"]]
    for r, row in enumerate(data):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(value))
    return table


def test_all_rows_include_the_header(qt_app):
    rows = table_rows(_table(), selected_only=False)

    assert rows[0] == ["Label", "Element", "Distance (Å)"]
    assert rows[1:] == [
        ["O1", "O", "2.001"], ["O2", "O", "2.010"], ["N1", "N", "2.150"]
    ]


def test_selection_gives_a_rectangular_block(qt_app):
    table = _table()
    table.selectRow(2)

    rows = table_rows(table, selected_only=True)

    assert rows == [
        ["Label", "Element", "Distance (Å)"], ["N1", "N", "2.150"]
    ]


def test_scattered_cells_still_align_by_column(qt_app):
    table = _table()
    model = table.selectionModel()
    flag = QItemSelectionModel.SelectionFlag.Select
    model.select(table.model().index(0, 0), flag)
    model.select(table.model().index(2, 2), flag)

    rows = table_rows(table, selected_only=True)

    assert rows == [
        ["Label", "Distance (Å)"], ["O1", "2.001"], ["N1", "2.150"]
    ]


def test_nothing_selected_means_nothing_copied(qt_app):
    table = _table()
    table.clearSelection()

    assert copy_table(table, selected_only=True) is False


def test_copy_all_puts_tsv_on_the_clipboard(qt_app):
    table = _table()

    assert copy_table(table, selected_only=False) is True

    text = QGuiApplication.clipboard().text()
    assert text.splitlines()[0] == "Label\tElement\tDistance (Å)"
    assert text.splitlines()[3] == "N1\tN\t2.150"


def test_tsv_flattens_tabs_and_newlines_inside_cells():
    assert rows_to_tsv([["a\tb", "c\nd"]]) == "a b\tc d"


def test_csv_quotes_commas():
    assert rows_to_csv([["Zn, Cu", "1"]]) == '"Zn, Cu",1\n'


def test_install_adds_a_copy_shortcut_and_context_menu(qt_app):
    from PySide6.QtCore import Qt

    table = _table()
    install_table_tools(table)

    assert any(a.text() == "Copy Selected" for a in table.actions())
    assert (
        table.contextMenuPolicy() == Qt.ContextMenuPolicy.CustomContextMenu
    )
    assert "copy" in table.toolTip().lower()


def test_every_results_table_has_the_tools(qt_app):
    from PySide6.QtCore import Qt

    from thevoid_core.ui.widgets.coordination_panel import (
        CoordinationEnvironmentPanel,
    )
    from thevoid_core.ui.widgets.measurements_panel import MeasurementsPanel
    from thevoid_core.ui.widgets.mof_analysis_panel import MOFAnalysisPanel
    from thevoid_core.ui.widgets.oms_panel import OMSPanel

    # Panels kept referenced so their child tables stay alive.
    panels = [
        OMSPanel(),
        CoordinationEnvironmentPanel(),
        MeasurementsPanel(),
        MOFAnalysisPanel(),
    ]
    tables = [
        table for panel in panels for table in panel.findChildren(QTableWidget)
    ]

    assert len(tables) >= 6
    for table in tables:
        assert (
            table.contextMenuPolicy()
            == Qt.ContextMenuPolicy.CustomContextMenu
        )
