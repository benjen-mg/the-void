"""Pore size distribution plot, with CSV export."""

from __future__ import annotations

import csv

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

BAR_COLOUR = "#1f4e79"
LINE_COLOUR = "#d9822b"


def write_psd_csv(result, filename: str) -> None:
    with open(filename, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [f"# geometric pore size distribution, bin width {result.bin_width} A, "
             f"pore volume {result.pore_volume:.3f} A^3 per cell"]
        )
        writer.writerow(
            ["diameter_A", "distribution_per_A", "cumulative_volume_A3_at_lower_edge"]
        )
        for centre, value, volume in result.rows():
            writer.writerow([f"{centre:.4f}", f"{value:.6f}", f"{volume:.4f}"])


class PoreSizeDistributionDialog(QDialog):
    def __init__(self, result, parent=None, *, title: str = "") -> None:
        super().__init__(parent)

        self.result = result
        self.setWindowTitle("Pore Size Distribution" + (f" - {title}" if title else ""))
        self.resize(820, 560)

        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self._plot()

        dominant = result.dominant_diameter
        self.summary_label = QLabel(
            f"Pore volume {result.pore_volume:.1f} Å³ per cell "
            f"({100 * result.void_fraction:.1f} % of the cell); "
            + (
                f"dominant pore diameter {dominant:.2f} Å."
                if dominant is not None
                else "no void space."
            )
            + f"  Bins {result.bin_width:g} Å; grid "
            f"{result.grid_shape[0]}×{result.grid_shape[1]}×"
            f"{result.grid_shape[2]}."
        )
        self.summary_label.setWordWrap(True)
        self.summary_label.setToolTip(
            "Geometric PSD (Gelb & Gubbins): bars show how the pore "
            "volume divides by the largest sphere that can cover each "
            "point; the line is the volume coverable by spheres at "
            "least that large. Resolution is limited by the grid "
            "(0.3 Å), so the last bar can sit up to one bin below "
            "the exact largest cavity diameter."
        )

        self.export_button = QPushButton("Export CSV")
        self.export_button.setToolTip("Save the distribution table as CSV.")
        self.export_button.clicked.connect(lambda: self.export_csv())

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)

        buttons = QHBoxLayout()
        buttons.addWidget(self.export_button)
        buttons.addStretch()
        buttons.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas, stretch=1)
        layout.addWidget(self.summary_label)
        layout.addLayout(buttons)

    def _plot(self) -> None:
        result = self.result
        axes = self.figure.add_subplot(111)
        axes.bar(
            result.bin_centers, result.distribution, width=result.bin_width * 0.9,
            color=BAR_COLOUR, label="distribution",
        )
        axes.set_xlabel("Pore diameter (Å)")
        axes.set_ylabel("Fraction of pore volume per Å", color=BAR_COLOUR)

        twin = axes.twinx()
        twin.plot(
            result.diameters, result.cumulative_volume, color=LINE_COLOUR,
            linewidth=1.8, label="cumulative",
        )
        twin.set_ylabel("Volume coverable by larger spheres (Å³)", color=LINE_COLOUR)
        twin.set_ylim(bottom=0)
        axes.set_xlim(left=0)
        self.axes = axes
        self.twin_axes = twin

    def export_csv(self, filename: str | None = None) -> bool:
        if not filename:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Pore Size Distribution", "psd.csv", "CSV Files (*.csv)"
            )
        if not filename:
            return False
        if not filename.lower().endswith(".csv"):
            filename += ".csv"
        try:
            write_psd_csv(self.result, filename)
        except OSError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
            return False
        return True
