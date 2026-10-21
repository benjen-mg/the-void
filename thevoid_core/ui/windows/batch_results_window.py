"""File > Open Batch Results...: rank and filter a `the-void-analyze`
results folder, and open any row's structure in the main window
(round 54).

A separate window, so the table and the 3D view can sit side by side.
The table is a model/view pair rather than a QTableWidget: a CoRE-sized
batch has tens of thousands of rows, and rebuilding that many widget
items on every keystroke in the search box would be sluggish. All the
logic - typing, filters, ranking, finding the structure file again - is
in services/batch_results.py; this module only presents it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.services.batch_results import (
    PRESENCE_OPERATORS,
    Filter,
    check_file,
    column_label,
    load_batch_results,
    locate_structure,
    report_markdown,
    view,
    write_view_csv,
)

logger = logging.getLogger(__name__)

MISSING_TEXT = "–"
MISSING_COLOR = QColor("#8a8a8a")
ERROR_COLOR = QColor("#b3541e")


def format_value(value) -> str:
    if value is None:
        return MISSING_TEXT
    if isinstance(value, float):
        return f"{value:.3f}" if abs(value) < 1e5 else f"{value:.4g}"
    return str(value)


class BatchTableModel(QAbstractTableModel):
    """Rows as the service's view() returns them, with a rank column
    in front. Sorting requests are handed back to the window, which
    re-runs view() so filters, ranking and top-N stay one computation."""

    sort_requested = Signal(int, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.results = None
        self.rows: list = []

    def set_rows(self, results, rows) -> None:
        self.beginResetModel()
        self.results = results
        self.rows = list(rows)
        self.endResetModel()

    def column_name(self, section: int) -> str | None:
        if self.results is None or section == 0:
            return None
        return self.results.columns[section - 1]

    # Qt interface ------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()) -> int:  # noqa: N802
        if parent.isValid() or self.results is None:
            return 0
        return len(self.results.columns) + 1

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):  # noqa: N802
        if orientation != Qt.Orientation.Horizontal or self.results is None:
            return None
        column = self.column_name(section)
        if role == Qt.ItemDataRole.DisplayRole:
            return "#" if column is None else column_label(column)
        if role == Qt.ItemDataRole.ToolTipRole:
            return "Rank in the current view (click to return to file order)" \
                if column is None else column
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        column = self.column_name(index.column())
        value = None if column is None else row.get(column)

        if role == Qt.ItemDataRole.DisplayRole:
            return str(index.row() + 1) if column is None else format_value(value)
        if role == Qt.ItemDataRole.TextAlignmentRole:
            if column is None or self.results.is_numeric(column):
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None
        if role == Qt.ItemDataRole.ForegroundRole:
            if column == "errors" and value:
                return ERROR_COLOR
            if value is None and column is not None:
                return MISSING_COLOR
        if role == Qt.ItemDataRole.ToolTipRole and value is not None and column is not None:
            return str(value)
        return None

    def sort(self, column, order=Qt.SortOrder.AscendingOrder) -> None:
        self.sort_requested.emit(column, order)


class FilterRow(QWidget):
    """One condition: column, operator, value, and a remove button."""

    changed = Signal()
    remove_requested = Signal(object)

    def __init__(self, results, parent=None) -> None:
        super().__init__(parent)
        self.results = results

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.column_box = QComboBox()
        for column in results.columns:
            self.column_box.addItem(column_label(column), column)
        self.operator_box = QComboBox()
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("value")
        self.remove_button = QToolButton()
        self.remove_button.setText("✕")
        self.remove_button.setToolTip("Remove this filter")

        layout.addWidget(self.column_box, 2)
        layout.addWidget(self.operator_box, 1)
        layout.addWidget(self.value_edit, 2)
        layout.addWidget(self.remove_button)

        self.column_box.currentIndexChanged.connect(self._on_column_changed)
        self.operator_box.currentIndexChanged.connect(self._on_operator_changed)
        self.value_edit.textChanged.connect(lambda _: self.changed.emit())
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))

        self._on_column_changed()

    def _on_column_changed(self, *_):
        column = self.column_box.currentData()
        self.operator_box.blockSignals(True)
        self.operator_box.clear()
        for operator in self.results.operators_for(column):
            self.operator_box.addItem(operator.replace(">=", "≥").replace("<=", "≤"), operator)
        self.operator_box.blockSignals(False)
        self._on_operator_changed()

    def _on_operator_changed(self, *_):
        self.value_edit.setEnabled(self.operator_box.currentData() not in PRESENCE_OPERATORS)
        self.changed.emit()

    def set_condition(self, column, operator, value="") -> None:
        self.column_box.setCurrentIndex(self.column_box.findData(column))
        self.operator_box.setCurrentIndex(self.operator_box.findData(operator))
        self.value_edit.setText(value)

    def condition(self) -> Filter:
        return Filter(
            self.column_box.currentData(),
            self.operator_box.currentData(),
            self.value_edit.text().strip(),
        )

    def mark_invalid(self, invalid: bool) -> None:
        self.value_edit.setStyleSheet("border: 1px solid #c0392b;" if invalid else "")


class BatchResultsWindow(QMainWindow):
    def __init__(self, main_window=None, parent=None) -> None:
        super().__init__(parent)
        self.main_window = main_window
        self.results = None
        self.filter_rows: list[FilterRow] = []
        self.sort_column: str | None = None
        self.descending = False
        self.visible_rows: list = []
        self._size_columns = True
        #: Folders the person pointed at to find structure files; searched
        #: for every later row too.
        self.extra_dirs: list[Path] = []

        self.setWindowTitle("Batch Results — the Void")
        self.resize(1200, 800)

        central = QWidget()
        layout = QVBoxLayout(central)

        top = QHBoxLayout()
        self.folder_label = QLabel("No results folder open.")
        self.open_button = QPushButton("Open Results Folder...")
        self.open_button.setToolTip(
            "A folder the-void-analyze wrote (its --out): summary.csv plus "
            "a JSON and Markdown report per structure."
        )
        self.open_button.clicked.connect(self.choose_folder)
        top.addWidget(self.folder_label, 1)
        top.addWidget(self.open_button)
        layout.addLayout(top)

        controls = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search every column…")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.refresh)
        self.add_filter_button = QPushButton("Add Filter")
        self.add_filter_button.setToolTip(
            "Keep only rows that satisfy a condition, e.g. PLD (Å) ≥ 3.3 or "
            "Topology = pcu. Several filters must all hold."
        )
        self.add_filter_button.clicked.connect(lambda: self.add_filter())
        self.top_box = QSpinBox()
        self.top_box.setRange(0, 1_000_000)
        self.top_box.setSpecialValueText("Show all rows")
        self.top_box.setPrefix("Show top ")
        self.top_box.setToolTip("Only the first N rows of the current ranking (0 shows all).")
        self.top_box.valueChanged.connect(self.refresh)
        self.export_button = QPushButton("Export View CSV...")
        self.export_button.setToolTip("Save the rows as shown, in this order, with their rank.")
        self.export_button.clicked.connect(lambda: self.export_view())
        # Round 58: publication figures from the rows as shown.
        self.figure_button = QPushButton("Figure...")
        self.figure_button.setToolTip(
            "A publication figure (scatter, histogram, or parity against CoRE MOF) "
            "from the rows as shown: SVG, PDF and 600-dpi PNG at journal column "
            "width, with a JSON of the exact points.")
        self.figure_button.clicked.connect(self.open_figure)
        # Round 66: the group's shared index.
        self.index_button = QPushButton("Add to Shared Index...")
        self.index_button.setToolTip(
            "Add the selected rows (or, with none selected, every row shown) to the "
            "shared index, with who ran them. Structure files are copied in only when "
            "their licence allows (never a CSD-derived file).")
        self.index_button.clicked.connect(lambda: self.add_to_index())
        self.index_button.setEnabled(False)
        self.count_label = QLabel("")
        controls.addWidget(self.search_edit, 2)
        controls.addWidget(self.add_filter_button)
        controls.addWidget(self.top_box)
        controls.addWidget(self.export_button)
        controls.addWidget(self.figure_button)
        controls.addWidget(self.index_button)
        controls.addWidget(self.count_label, 1)
        layout.addLayout(controls)

        self.filters_layout = QVBoxLayout()
        layout.addLayout(self.filters_layout)
        self.filter_error = QLabel("")
        self.filter_error.setStyleSheet("color: #c0392b;")
        self.filter_error.setVisible(False)
        layout.addWidget(self.filter_error)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.model = BatchTableModel(self)
        self.model.sort_requested.connect(self._on_sort_requested)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(lambda index: self.open_row(index.row()))
        splitter.addWidget(self.table)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        row_bar = QHBoxLayout()
        self.selected_label = QLabel("Select a row to see its report.")
        self.open_3d_button = QPushButton("Open in 3D View")
        self.open_3d_button.setToolTip(
            "Load this row's structure in the main window (double-click a "
            "row does the same). The file is checked against the report's "
            "SHA-256 first."
        )
        self.open_3d_button.setEnabled(False)
        self.open_3d_button.clicked.connect(lambda: self.open_row(self.current_row_index()))
        row_bar.addWidget(self.selected_label, 1)
        row_bar.addWidget(self.open_3d_button)
        bottom_layout.addLayout(row_bar)
        self.report_view = QTextBrowser()
        self.report_view.setOpenExternalLinks(False)
        bottom_layout.addWidget(self.report_view)
        splitter.addWidget(bottom)
        splitter.setSizes([480, 320])
        layout.addWidget(splitter, 1)

        self.setCentralWidget(central)
        self.table.selectionModel().currentRowChanged.connect(
            lambda current, _: self.show_row(current.row())
        )
        QShortcut(QKeySequence.StandardKey.Copy, self.table, activated=self.copy_selection)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def choose_folder(self) -> None:
        start = str(self.results.folder) if self.results else ""
        folder = QFileDialog.getExistingDirectory(self, "Open Batch Results Folder", start)
        if folder:
            self.load(folder)

    def load(self, path) -> bool:
        try:
            results = load_batch_results(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Cannot Open Batch Results", str(exc))
            return False

        self.results = results
        self.extra_dirs = []
        self._size_columns = True
        for row in list(self.filter_rows):
            self._remove_filter(row)
        self.sort_column, self.descending = None, False
        self.table.horizontalHeader().setSortIndicator(-1, Qt.SortOrder.AscendingOrder)
        self.folder_label.setText(
            f"{results.summary_path}  —  {len(results.rows)} structure(s)"
        )
        index_build = self.is_index_build()
        self.index_button.setEnabled(not index_build)
        self.index_button.setToolTip(
            "These rows are the shared index itself." if index_build else
            "Add the selected rows (or, with none selected, every row shown) to the "
            "shared index, with who ran them. Structure files are copied in only when "
            "their licence allows (never a CSD-derived file).")
        if index_build:
            self.setWindowTitle(f"Shared Index: {Path(results.folder).parent.name} — the Void")
        else:
            self.setWindowTitle("Batch Results — the Void")
        self.refresh()
        return True

    # ------------------------------------------------------------------
    # Filters, ranking and the table
    # ------------------------------------------------------------------

    def add_filter(self, column=None, operator=None, value="") -> FilterRow | None:
        if self.results is None:
            return None
        row = FilterRow(self.results)
        if column is not None:
            row.set_condition(column, operator, value)
        row.changed.connect(self.refresh)
        row.remove_requested.connect(self._remove_filter)
        self.filter_rows.append(row)
        self.filters_layout.addWidget(row)
        self.refresh()
        return row

    def _remove_filter(self, row) -> None:
        if row in self.filter_rows:
            self.filter_rows.remove(row)
        row.setParent(None)
        row.deleteLater()
        self.refresh()

    def _on_sort_requested(self, section, order) -> None:
        column = self.model.column_name(section)
        self.sort_column = column
        self.descending = order == Qt.SortOrder.DescendingOrder
        self.refresh()

    def sort_by(self, column: str | None, descending: bool = False) -> None:
        """Programmatic ranking (also what a header click does)."""
        self.sort_column, self.descending = column, descending
        section = 0 if column is None else self.results.columns.index(column) + 1
        order = Qt.SortOrder.DescendingOrder if descending else Qt.SortOrder.AscendingOrder
        header = self.table.horizontalHeader()
        header.blockSignals(True)
        header.setSortIndicator(section, order)
        header.blockSignals(False)
        self.refresh()
        # A ranking whose column is scrolled out of sight reads as an
        # arbitrary order: bring it into view.
        if self.visible_rows and section:
            self.table.scrollTo(self.model.index(0, section))

    def refresh(self, *_) -> None:
        if self.results is None:
            return

        valid, errors = [], []
        for row in self.filter_rows:
            condition = row.condition()
            try:
                condition.validate(self.results)
            except ValueError as exc:
                row.mark_invalid(True)
                errors.append(str(exc))
                continue
            row.mark_invalid(False)
            valid.append(condition)

        self.active_filters = valid
        self.filter_error.setText("Ignored: " + " ".join(errors) if errors else "")
        self.filter_error.setVisible(bool(errors))

        top = self.top_box.value() or None
        self.visible_rows = view(
            self.results,
            filters=valid,
            search=self.search_edit.text(),
            sort_column=self.sort_column,
            descending=self.descending,
            top_n=top,
        )
        self.model.set_rows(self.results, self.visible_rows)
        if self._size_columns:
            # Once per folder: re-measuring every column on each
            # keystroke cost ~0.3 s at 20,000 rows, and columns that
            # change width while typing are distracting anyway.
            self.table.resizeColumnsToContents()
            self._size_columns = False
        ranking = (
            f", ranked by {column_label(self.sort_column)} "
            f"({'largest' if self.descending else 'smallest'} first; empty cells last)"
            if self.sort_column else ""
        )
        self.count_label.setText(
            f"{len(self.visible_rows)} of {len(self.results.rows)} shown{ranking}"
        )
        self.show_row(-1)

    # ------------------------------------------------------------------
    # One row: report and structure
    # ------------------------------------------------------------------

    def current_row_index(self) -> int:
        index = self.table.currentIndex()
        return index.row() if index.isValid() else -1

    def show_row(self, row_index: int) -> None:
        if not (0 <= row_index < len(self.visible_rows)):
            self.selected_label.setText("Select a row to see its report.")
            self.report_view.clear()
            self.open_3d_button.setEnabled(False)
            return
        row = self.visible_rows[row_index]
        self.selected_label.setText(
            f"#{row_index + 1}  {row.get('file')}  ({row.get('formula') or '?'})"
        )
        self.report_view.setMarkdown(report_markdown(row))
        self.open_3d_button.setEnabled(self.main_window is not None)

    def open_row(self, row_index: int):
        """Find the row's structure file, check it is the one the report
        was made from, and load it in the main window. Returns the path
        loaded, or None."""
        if self.main_window is None or not (0 <= row_index < len(self.visible_rows)):
            return None
        row = self.visible_rows[row_index]
        located = locate_structure(row, self.results, self.extra_dirs)

        if located.status == "missing":
            picked, _ = QFileDialog.getOpenFileName(
                self,
                f"Locate {Path(located.recorded_path or row.get('file') or '').name}",
                str(self.results.folder.parent),
                "Structure Files (*.cif *.CIF *.json *.vasp POSCAR CONTCAR);;All Files (*)",
            )
            if not picked:
                return None
            located = check_file(picked, row)
            if Path(picked).parent not in self.extra_dirs:
                self.extra_dirs.append(Path(picked).parent)

        if located.status == "mismatch":
            answer = QMessageBox.question(
                self,
                "Different File",
                f"{located.message}\n\nThe numbers in this row were not "
                "computed from it. Load it anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return None

        if getattr(self.main_window, "_busy", False):
            QMessageBox.information(
                self, "Busy",
                "The main window is still working on something; try again "
                "when it has finished.",
            )
            return None

        self.main_window._open_file(str(located.path))
        self.main_window.statusBar().showMessage(
            f"Batch row #{row_index + 1}: {located.message}", 10000
        )
        self.main_window.raise_()
        return located.path

    # ------------------------------------------------------------------
    # Copy and export
    # ------------------------------------------------------------------

    def selection_text(self) -> str:
        rows = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        if not rows or self.results is None:
            return ""
        lines = ["\t".join(["rank"] + self.results.columns)]
        for r in rows:
            row = self.visible_rows[r]
            lines.append("\t".join(
                [str(r + 1)] + ["" if row.get(c) is None else str(row.get(c))
                                for c in self.results.columns]
            ))
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # The shared index (round 66)
    # ------------------------------------------------------------------

    def is_index_build(self) -> bool:
        """Is the open folder an index's build/ (whose rows are already
        in the index)?"""
        return (self.results is not None
                and (Path(self.results.folder).parent / "INDEX.json").is_file()
                and "entry_id" in self.results.columns)

    def rows_to_add(self) -> list:
        """File names of the selected rows, or of every row shown."""
        selected = sorted({i.row() for i in self.table.selectionModel().selectedRows()})
        rows = [self.visible_rows[r] for r in selected] if selected else self.visible_rows
        return [r.get("file") for r in rows if r.get("file")]

    def add_to_index(self, values=None) -> None:
        if self.results is None or self.is_index_build():
            return
        files = self.rows_to_add()
        if not files:
            return
        if self.main_window is not None:
            self.main_window.add_results_to_index(str(self.results.folder), only=files,
                                                  values=values)

    def copy_selection(self) -> None:
        text = self.selection_text()
        if text:
            QGuiApplication.clipboard().setText(text)

    def open_figure(self):
        """The figure dialog for the rows as shown (filters, search and
        top-N applied)."""
        from thevoid_core.ui.dialogs.figure_dialog import FigureDialog

        if self.results is None:
            return None
        dialog = FigureDialog(self.results, self.visible_rows,
                              filters=getattr(self, "active_filters", []),
                              search=self.search_edit.text(), parent=self)
        dialog.show()
        self._figure_dialog = dialog
        return dialog

    def export_view(self, filename: str | None = None) -> bool:
        if self.results is None:
            return False
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export View as CSV",
                str(self.results.folder / "ranked_view.csv"), "CSV Files (*.csv)",
            )
            if not filename:
                return False
        try:
            write_view_csv(self.results, self.visible_rows, filename)
        except OSError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
            return False
        self.statusBar().showMessage(f"Saved {len(self.visible_rows)} rows to {filename}", 8000)
        return True
