"""Analysis > Compare with CoRE MOF 2024... (round 56): the result of
services/reference/compare.py, where a person reads it.

The dialog shows what the match rests on before the numbers - which
entry, found how, and how sure (identified / possible) - because a
difference against the wrong entry is a difference of structure, not of
method. When there are several candidates, another can be chosen and
compared. Every row says whether the two sides agree within the
flagged tolerance, and the notes say under which convention the Void's
values were computed (CoRE's radii and probe, not the Void's defaults).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

COLUMNS = ["Property", "The Void", "CoRE MOF 2024", "Difference", ""]


def _format(value, unit=""):
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        digits = 1 if unit.startswith("m²") else (4 if unit == "g/cm³" else 3)
        return f"{value:.{digits}f}"
    return str(value)


class ReferenceComparisonDialog(QDialog):
    #: Emitted with a candidate's entry id when "Compare with this
    #: entry" is pressed; MainWindow runs the comparison and calls
    #: set_comparison again.
    compare_requested = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare with CoRE MOF 2024")
        self.setModal(False)
        self.resize(760, 520)
        self.comparison = None

        layout = QVBoxLayout(self)
        self.match_label = QLabel()
        self.match_label.setWordWrap(True)
        self.match_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.match_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Candidates:"))
        self.candidate_box = QComboBox()
        self.candidate_box.setToolTip(
            "Every CoRE MOF 2024 entry that is, or may be, this structure. "
            "'identified': found by name and consistent with the structure; "
            "'possible': same metals, cell contents, density and composition; "
            "'name only': shares the name but not the structure."
        )
        row.addWidget(self.candidate_box, 1)
        self.compare_button = QPushButton("Compare with this entry")
        self.compare_button.clicked.connect(self._on_compare_clicked)
        row.addWidget(self.compare_button)
        layout.addLayout(row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)

        self.notes_label = QLabel()
        self.notes_label.setWordWrap(True)
        self.notes_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.notes_label)

        buttons = QHBoxLayout()
        self.copy_button = QPushButton("Copy Table")
        self.copy_button.clicked.connect(self.copy_table)
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------

    def set_candidates(self, matches, current_id=None) -> None:
        self.candidate_box.clear()
        for match in matches:
            self.candidate_box.addItem(
                f"{match.entry.id}  ({match.confidence}; {match.how})", match.entry.id)
        if current_id is not None:
            index = self.candidate_box.findData(current_id)
            if index >= 0:
                self.candidate_box.setCurrentIndex(index)
        self.compare_button.setEnabled(self.candidate_box.count() > 0)

    def set_busy(self, busy: bool) -> None:
        self.compare_button.setEnabled(not busy and self.candidate_box.count() > 0)
        if busy:
            self.match_label.setText("Computing the Void's values under CoRE MOF's convention…")

    def set_no_match(self, message: str) -> None:
        self.comparison = None
        self.match_label.setText(message)
        self.table.setRowCount(0)
        self.notes_label.setText("")

    def set_comparison(self, comparison) -> None:
        self.comparison = comparison
        entry = comparison.reference_values
        where = (f"CSD refcode {entry['csd_refcode']}" if entry.get("csd_refcode")
                 else "structure from the paper's SI")
        colour = {"identified": "#2e7d32", "possible": "#b26a00"}.get(comparison.confidence, "#c62828")
        self.match_label.setText(
            f"<b>{comparison.entry_id}</b> — {where}, doi:{entry.get('doi') or '–'}<br>"
            f"Match: <b style='color:{colour}'>{comparison.confidence}</b> ({comparison.how})"
        )
        self.table.setRowCount(len(comparison.rows))
        for r, row in enumerate(comparison.rows):
            unit = f" {row.unit}" if row.unit else ""
            difference = "" if row.difference is None else (
                ("+" if row.difference >= 0 else "") + _format(float(row.difference), row.unit) + unit)
            status = {True: "agrees", False: "differs", None: ""}[row.agrees]
            values = [row.label, _format(row.void, row.unit) + (unit if row.difference is not None else ""),
                      _format(row.reference, row.unit) + (unit if row.difference is not None else ""),
                      difference, status]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c == 0 and row.note:
                    item.setToolTip(row.note)
                if c == 4 and row.agrees is False:
                    item.setForeground(QColor("#c62828"))
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        from thevoid_core.services.reference.coremof import index_info

        source = index_info()
        notes = list(comparison.notes)
        for name, error in comparison.errors.items():
            notes.append(f"{name} failed: {error}")
        notes.append(f"Data: {source['citation']}, doi:{source['doi']} ({source['licence']}).")
        self.notes_label.setText("\n".join(f"• {n}" for n in notes))
        self.compare_button.setEnabled(self.candidate_box.count() > 0)

    def selection_text(self) -> str:
        lines = ["\t".join(COLUMNS[:4] + ["status"])]
        for r in range(self.table.rowCount()):
            lines.append("\t".join(
                (self.table.item(r, c).text() if self.table.item(r, c) else "")
                for c in range(len(COLUMNS))))
        return "\n".join(lines)

    def copy_table(self) -> None:
        QApplication.clipboard().setText(self.selection_text())

    def _on_compare_clicked(self) -> None:
        entry_id = self.candidate_box.currentData()
        if entry_id:
            self.compare_requested.emit(entry_id)
