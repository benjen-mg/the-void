"""Analysis > Predict Properties... (round 61): what the property
models (services/ml) say about the loaded structure, where a person
reads it.

Every row has its 90% interval beside the value and says whether the
structure is inside the model's applicability domain, because a number
without either reads as more certain than it is. What each model was
trained on (its caveat: CoRE MOF 2024's stabilities are themselves ML
predictions) is on the row's tooltip and in the notes, and the title
says these are models, not measurements. "Add Model Folder..." adds a
model the-void-train wrote.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

COLUMNS = ["Property", "Predicted", "90% interval", "Unit", "Domain"]


def format_value(value, unit="") -> str:
    """As the reports write it (services/report.format_prediction)."""
    from thevoid_core.services.report import format_prediction

    return format_prediction(value, unit)


class PredictionDialog(QDialog):
    #: Emitted with the list of extra model folders when one is added;
    #: MainWindow re-runs the predictions and calls set_result again.
    models_requested = Signal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Predicted Properties")
        self.setModal(False)
        self.resize(760, 420)
        self.result = None
        self.extra_models: list[str] = []

        layout = QVBoxLayout(self)
        self.header_label = QLabel(
            "<b>Model estimates, not measurements.</b> Each value comes with the interval "
            "that held 90% of held-out structures, and a flag for structures unlike the "
            "training set."
        )
        self.header_label.setWordWrap(True)
        self.header_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(self.header_label)

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
        self.add_button = QPushButton("Add Model Folder...")
        self.add_button.setToolTip("A folder the-void-train wrote (card.json + model.joblib). "
                                   "Only add models you trust: they are loaded with joblib.")
        self.add_button.clicked.connect(self._on_add_clicked)
        buttons.addWidget(self.add_button)
        self.copy_button = QPushButton("Copy Table")
        self.copy_button.clicked.connect(self.copy_table)
        buttons.addWidget(self.copy_button)
        buttons.addStretch(1)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        self.add_button.setEnabled(not busy)
        if busy:
            self.notes_label.setText("Computing descriptors and predicting…")

    def set_result(self, result: dict) -> None:
        """result: predict_structure's {"predictions", "descriptors", "notes"}."""
        self.result = result
        predictions = result["predictions"]
        self.table.setRowCount(len(predictions))
        for r, p in enumerate(predictions):
            values = [p.label, format_value(p.value, p.unit),
                      f"{format_value(p.low, p.unit)} – {format_value(p.high, p.unit)}",
                      p.unit or "–", "inside" if p.in_domain else "outside"]
            for c, text in enumerate(values):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setToolTip(f"{p.model}: {p.note}" if p.note else p.model)
                if c == 4:
                    item.setToolTip(f"mean distance to the 5 nearest training structures: "
                                    f"{p.domain_distance:.2f} (standardised descriptors)")
                    if not p.in_domain:
                        item.setForeground(QColor("#c62828"))
                self.table.setItem(r, c, item)
        self.table.resizeColumnsToContents()
        notes = []
        outside = [p.label for p in predictions if not p.in_domain]
        if outside:
            notes.append("Outside the applicability domain (unlike the training structures, "
                         "so treat with extra care): " + ", ".join(outside) + ".")
        for p in predictions:
            if p.note:
                notes.append(f"{p.label}: {p.note}")
        for key, message in result.get("notes", {}).items():
            notes.append(f"{key}: {message} (entered the model as missing)")
        self.notes_label.setText("\n".join(f"• {n}" for n in notes))
        self.add_button.setEnabled(True)

    def selection_text(self) -> str:
        lines = ["\t".join(COLUMNS)]
        for r in range(self.table.rowCount()):
            lines.append("\t".join(
                (self.table.item(r, c).text() if self.table.item(r, c) else "")
                for c in range(len(COLUMNS))))
        return "\n".join(lines)

    def copy_table(self) -> None:
        QApplication.clipboard().setText(self.selection_text())

    def _on_add_clicked(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Model folder (from the-void-train)")
        if folder:
            self.add_model_folder(folder)

    def add_model_folder(self, folder: str) -> None:
        if folder not in self.extra_models:
            self.extra_models.append(folder)
        self.models_requested.emit(list(self.extra_models))
