"""Copy and export for every results table in the application.

Before round 49 no table anywhere - OMS sites, coordination neighbours,
MOF linkers/nodes, measurements, XRD peaks - could be copied: getting a
number out into a spreadsheet or a lab notebook meant retyping it. This
module adds the same three things to each of them:

- Ctrl+C (the platform Copy shortcut) copies the selected cells,
- a right-click menu with Copy Selected, Copy All, and Export CSV...,
- both copies as tab-separated text with a header row, which pastes
  straight into Excel, LibreOffice, Google Sheets, or Origin as
  separate columns.
"""

from __future__ import annotations

import csv
import io
import logging

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMenu, QMessageBox, QTableWidget

logger = logging.getLogger(__name__)


def table_rows(table: QTableWidget, *, selected_only: bool) -> list[list[str]]:
    """The table's contents as a header row followed by data rows.

    selected_only restricts the result to rows and columns that contain
    at least one selected cell - a rectangular block, so a selection of
    scattered cells still produces aligned columns rather than ragged
    rows. With nothing selected it returns just the header, so a
    caller can tell "nothing to copy" apart from "empty table".
    """
    row_count = table.rowCount()
    column_count = table.columnCount()

    header_view = table.horizontalHeader()

    if selected_only:
        indexes = table.selectedIndexes()
        rows = sorted({index.row() for index in indexes})
        columns = {index.column() for index in indexes}
    else:
        rows = list(range(row_count))
        columns = set(range(column_count))

    # Columns in the order they appear on screen (a header section may
    # have been moved), skipping hidden ones - a copy should look like
    # what the person is looking at.
    columns = sorted(
        (c for c in columns if not header_view.isSectionHidden(c)),
        key=header_view.visualIndex,
    )

    header = []

    for column in columns:
        item = table.horizontalHeaderItem(column)
        header.append(item.text() if item is not None else f"Column {column + 1}")

    result = [header]

    for row in rows:
        values = []

        for column in columns:
            item = table.item(row, column)
            values.append(item.text() if item is not None else "")

        result.append(values)

    return result


def rows_to_tsv(rows: list[list[str]]) -> str:
    """Tab-separated text, one line per row - the format spreadsheet
    applications split into columns on paste. Tabs or newlines inside a
    cell are flattened to spaces so they cannot shift the columns.
    """
    return "\n".join(
        "\t".join(
            cell.replace("\t", " ").replace("\n", " ") for cell in row
        )
        for row in rows
    )


def rows_to_csv(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def copy_table(table: QTableWidget, *, selected_only: bool) -> bool:
    """Put the table (or its selection) on the clipboard. Returns False
    when there was nothing to copy.
    """
    rows = table_rows(table, selected_only=selected_only)

    if len(rows) <= 1:
        return False

    QGuiApplication.clipboard().setText(rows_to_tsv(rows))
    return True


def export_table_csv(table: QTableWidget, parent=None, default_name="table.csv") -> None:
    rows = table_rows(table, selected_only=False)

    if len(rows) <= 1:
        QMessageBox.information(
            parent or table, "Nothing to Export", "This table is empty."
        )
        return

    filename, _ = QFileDialog.getSaveFileName(
        parent or table, "Export Table as CSV", default_name, "CSV Files (*.csv)"
    )

    if not filename:
        return

    if not filename.lower().endswith(".csv"):
        filename += ".csv"

    try:
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            handle.write(rows_to_csv(rows))
    except OSError as exc:
        QMessageBox.warning(
            parent or table, "Export Failed", f"Could not write {filename}:\n{exc}"
        )


def install_table_tools(table: QTableWidget, *, export_name: str = "table.csv") -> None:
    """Give a results table Ctrl+C, a right-click copy/export menu, and
    a tooltip saying so. Safe to call once per table at construction.
    """
    copy_action = QAction("Copy Selected", table)
    copy_action.setShortcut(QKeySequence(QKeySequence.StandardKey.Copy))
    copy_action.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
    copy_action.triggered.connect(
        lambda: copy_table(table, selected_only=True)
    )
    # Added to the widget so the shortcut is live even without the menu
    # ever being opened.
    table.addAction(copy_action)

    def show_menu(position) -> None:
        menu = QMenu(table)

        has_rows = table.rowCount() > 0
        has_selection = bool(table.selectedIndexes())

        selected = menu.addAction("Copy Selected")
        selected.setEnabled(has_selection)
        selected.triggered.connect(
            lambda: copy_table(table, selected_only=True)
        )

        everything = menu.addAction("Copy All")
        everything.setEnabled(has_rows)
        everything.triggered.connect(
            lambda: copy_table(table, selected_only=False)
        )

        menu.addSeparator()

        export = menu.addAction("Export CSV...")
        export.setEnabled(has_rows)
        export.triggered.connect(
            lambda: export_table_csv(table, default_name=export_name)
        )

        menu.exec(table.viewport().mapToGlobal(position))

    table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
    table.customContextMenuRequested.connect(show_menu)

    if not table.toolTip():
        table.setToolTip(
            "Right-click to copy or export this table; Ctrl+C copies "
            "the selected rows (pastes into a spreadsheet as columns)."
        )
