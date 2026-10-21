"""Batch Results > Figure... (round 58): a publication figure from the
rows the window is showing (services/figures.py).

The preview is the saved figure itself: it is rendered at its print
size (85 or 175 mm wide, 7 pt text) and the picture is scaled to fit,
keeping its proportions - not a chart stretched to the window, which
would show a layout the file does not have. Save writes SVG, PDF and 600-dpi PNG plus a
JSON of the exact points, the filters and search in force, and the
style. Below the preview the dialog says how many rows are plotted and
how many were left out, and why.
"""

from __future__ import annotations

from pathlib import Path

import io

from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from thevoid_core.services import figures
from thevoid_core.services.batch_results import column_label

KINDS = [("Scatter", "scatter"), ("Histogram", "histogram"),
         ("Parity (the Void vs reference)", "parity")]


class FigureDialog(QDialog):
    def __init__(self, results, rows, *, filters=(), search="", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Figure")
        self.results, self.rows = results, list(rows)
        self.filters, self.search = list(filters), search
        self.data = None
        self.last_saved: list = []

        numeric = [c for c in results.columns if results.is_numeric(c)]
        text = [c for c in results.columns
                if not results.is_numeric(c) and c not in ("file", "errors")]

        layout = QHBoxLayout(self)
        controls = QFormLayout()
        self.kind_box = QComboBox()
        for label, key in KINDS:
            self.kind_box.addItem(label, key)
        controls.addRow("Kind:", self.kind_box)
        self.x_box, self.y_box = QComboBox(), QComboBox()
        for box in (self.x_box, self.y_box):
            for column in numeric:
                box.addItem(column_label(column), column)
        controls.addRow("X:", self.x_box)
        controls.addRow("Y:", self.y_box)
        self.colour_box = QComboBox()
        self.colour_box.addItem("none", None)
        for column in text:
            self.colour_box.addItem(column_label(column), column)
        self.colour_box.setToolTip(
            "At most three categories get their own colour and marker; the rest are "
            "shown as Other. Topology is grouped by net (sql 4-c and its "
            "interpenetrated forms are all sql); unidentified nets never take a colour.")
        controls.addRow("Colour by:", self.colour_box)
        self.log_x, self.log_y = QCheckBox("log x"), QCheckBox("log y")
        logs = QHBoxLayout()
        logs.addWidget(self.log_x)
        logs.addWidget(self.log_y)
        controls.addRow("Axes:", logs)
        self.bins_box = QSpinBox()
        self.bins_box.setRange(0, 200)
        self.bins_box.setSpecialValueText("auto")  # 0: numpy's auto rule
        self.bins_box.setValue(0)
        controls.addRow("Bins:", self.bins_box)
        self.width_box = QComboBox()
        self.width_box.addItem("Single column (85 mm)", "single")
        self.width_box.addItem("Double column (175 mm)", "double")
        controls.addRow("Width:", self.width_box)

        self.save_button = QPushButton("Save...")
        self.save_button.setToolTip("SVG, PDF and 600-dpi PNG, with a JSON of the points, "
                                    "filters and style.")
        self.save_button.clicked.connect(lambda: self.save())
        controls.addRow(self.save_button)
        layout.addLayout(controls)

        right = QVBoxLayout()
        self.figure = Figure()
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 256)
        self.preview.setStyleSheet("background: white;")
        self._pixmap = QPixmap()
        right.addWidget(self.preview, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        right.addWidget(self.status_label)
        layout.addLayout(right, 1)

        self._set_defaults(numeric, text)
        for widget in (self.kind_box, self.x_box, self.y_box, self.colour_box, self.width_box):
            widget.currentIndexChanged.connect(self.redraw)
        for widget in (self.log_x, self.log_y):
            widget.toggled.connect(self.redraw)
        self.bins_box.valueChanged.connect(self.redraw)
        self.resize(980, 560)
        self.redraw()

    def _set_defaults(self, numeric, text):
        def choose(box, column):
            index = box.findData(column)
            if index >= 0:
                box.setCurrentIndex(index)

        choose(self.x_box, "pld_A")
        choose(self.y_box, "channel_asa_m2_per_g")
        if "topology" in text:
            choose(self.colour_box, "topology")

    def spec(self) -> figures.FigureSpec:
        kind = self.kind_box.currentData()
        return figures.FigureSpec(
            kind, self.x_box.currentData(),
            None if kind == "histogram" else self.y_box.currentData(),
            colour_by=None if kind == "histogram" else self.colour_box.currentData(),
            log_x=self.log_x.isChecked(), log_y=self.log_y.isChecked(),
            bins=self.bins_box.value() or None,
        )

    def style(self) -> figures.FigureStyle:
        return figures.FigureStyle(width=self.width_box.currentData())

    def redraw(self, *_):
        kind = self.kind_box.currentData()
        self.y_box.setEnabled(kind != "histogram")
        self.colour_box.setEnabled(kind != "histogram")
        self.bins_box.setEnabled(kind == "histogram")
        if kind == "parity" and self.sender() is self.kind_box:
            # Offer the comparison's own pair, if the batch has it.
            for quantity in ("pld_A", "lcd_A", "asa_m2_per_g"):
                x, y = self.x_box.findData(f"coremof_{quantity}"), self.y_box.findData(f"ccdc_{quantity}")
                if x >= 0 and y >= 0:
                    self.x_box.setCurrentIndex(x)
                    self.y_box.setCurrentIndex(y)
                    break
        try:
            _, self.data = figures.draw(self.results, self.rows, self.spec(), self.style(),
                                        figure=self.figure)
        except ValueError as exc:
            self.data = None
            self.figure.clear()
            self.status_label.setText(str(exc))
            self.save_button.setEnabled(False)
            self._pixmap = QPixmap()
            self.preview.clear()
            return
        self._render_preview()
        text = f"{len(self.data.points)} of {len(self.rows)} rows plotted"
        if self.data.excluded:
            reasons = {}
            for _, reason in self.data.excluded:
                reasons[reason] = reasons.get(reason, 0) + 1
            text += "; left out: " + ", ".join(f"{n} ({r})" for r, n in reasons.items())
        if self.data.folded:
            text += f"; {len(self.data.folded)} categories shown as Other"
        if self.filters or self.search:
            text += ". Filters: " + "; ".join(
                [f.describe() for f in self.filters] + ([f'search "{self.search}"'] if self.search else []))
        self.status_label.setText(text)
        self.save_button.setEnabled(bool(self.data.points))

    #: Preview resolution: enough to read 7 pt text on screen.
    PREVIEW_DPI = 200

    def _render_preview(self):
        import matplotlib

        buffer = io.BytesIO()
        with matplotlib.rc_context(getattr(self.figure, "_void_rc", {})):
            self.figure.savefig(buffer, format="png", dpi=self.PREVIEW_DPI)
        self._pixmap = QPixmap()
        self._pixmap.loadFromData(buffer.getvalue(), "PNG")
        self._fit_preview()

    def _fit_preview(self):
        if not self._pixmap.isNull():
            self.preview.setPixmap(self._pixmap.scaled(
                self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit_preview()

    def save(self, base_path: str | None = None) -> list:
        if self.data is None or not self.data.points:
            return []
        if base_path is None:
            spec = self.spec()
            suggestion = str(self.results.folder / "figures" /
                             (f"{spec.kind}_{spec.x}" + (f"_{spec.y}" if spec.y else "")))
            base_path, _ = QFileDialog.getSaveFileName(
                self, "Save Figure", suggestion, "Figure (*.svg *.pdf *.png)")
            if not base_path:
                return []
        self.last_saved = figures.save(self.figure, self.data, self.spec(), self.style(),
                                       Path(base_path), results=self.results,
                                       filters=self.filters, search=self.search)
        self.status_label.setText(
            self.status_label.text().split(" Saved")[0] + " Saved: "
            + ", ".join(p.name for p in self.last_saved))
        return self.last_saved
