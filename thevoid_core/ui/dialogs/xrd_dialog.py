from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from thevoid_core.ui.widgets.table_tools import install_table_tools
from thevoid_core.exports.xrd_exporter import XRDExporter
from thevoid_core.services.xrd.peak_profile import (
    DEFAULT_FWHM,
    convolve_pattern,
    kalpha_components,
    split_kalpha_doublet,
)
from thevoid_core.services.xrd.xrd_result import (
    DEFAULT_MERGE_TOLERANCE,
    DEFAULT_MIN_RELATIVE_INTENSITY,
)
from thevoid_core.ui.widgets.xrd_plot_widget import XRDPlotWidget

#: Label, threshold as a percentage of the strongest peak.
THRESHOLD_PRESETS = [
    ("Dominant only (\u2265 10%)", 10.0),
    ("Major peaks (\u2265 5%)", 5.0),
    ("Full list (\u2265 1%)", 1.0),
    ("Everything (\u2265 0.1%)", 0.1),
    ("No filtering", 0.0),
]

STICKS = "sticks"
PROFILE = "profile"


class XRDDialog(QDialog):
    """Display a powder XRD pattern and its peak data.

    A full calculation on a low-symmetry cell produces many hundreds of
    reflections, most of them both negligible and unresolvable. The
    dialog shows a reduced list by default and lets the cutoff be moved
    without recalculating, since filtering is pure post-processing.

    Two ways to look at the same pattern are offered: discrete sticks
    (Show / Merge control which are drawn and how nearby ones combine)
    or a continuous trace built by convolving every calculated
    reflection with a pseudo-Voigt peak of a chosen width - closer to
    what an instrument would actually record, including overlap between
    nearby reflections that a stick plot can't show.
    """

    def __init__(self, result: Any, parent=None) -> None:
        super().__init__(parent)

        self.full_result = result
        self.result = result

        # Populated by _refresh() whenever the profile style is active,
        # so export_csv can write the continuous trace rather than the
        # discrete peak list without recomputing it.
        self._profile_two_theta: list[float] | None = None
        self._profile_intensities: list[float] | None = None
        self._profile_fwhm: float | None = None

        self.setWindowTitle("Powder XRD")
        self.resize(1100, 860)

        layout = QVBoxLayout(self)

        layout.addLayout(self._build_style_controls())
        layout.addLayout(self._build_stick_controls())

        self.plot = XRDPlotWidget()
        layout.addWidget(self.plot, stretch=3)

        self.table = QTableWidget()
        install_table_tools(self.table, export_name="xrd_peaks.csv")
        layout.addWidget(self.table, stretch=2)

        layout.addLayout(self._build_buttons())

        self._on_style_changed()

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_style_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        row.addWidget(QLabel("Style:"))

        self.style_box = QComboBox()
        self.style_box.addItem("Sticks", STICKS)
        self.style_box.addItem("Profile (pseudo-Voigt)", PROFILE)
        self.style_box.setToolTip(
            "Sticks: one line per reflection, sized by intensity. "
            "Profile: peaks broadened into a continuous curve, closer "
            "to what a real diffractometer trace looks like."
        )
        self.style_box.currentIndexChanged.connect(self._on_style_changed)

        row.addWidget(self.style_box)

        row.addSpacing(16)

        self.fwhm_label = QLabel("FWHM (\u00b0 2\u03b8):")
        row.addWidget(self.fwhm_label)

        self.fwhm_box = QDoubleSpinBox()
        self.fwhm_box.setDecimals(3)
        self.fwhm_box.setRange(0.01, 2.0)
        self.fwhm_box.setSingleStep(0.01)
        self.fwhm_box.setValue(DEFAULT_FWHM)
        self.fwhm_box.setToolTip(
            "Width of each simulated peak. Narrower looks like a sharp "
            "research instrument; wider looks like a poorly aligned or "
            "low-resolution one."
        )
        self.fwhm_box.valueChanged.connect(self._refresh)

        row.addWidget(self.fwhm_box)

        row.addSpacing(16)

        self.doublet_checkbox = QCheckBox("K\u03b1\u2081/K\u03b1\u2082 doublet")
        self.doublet_checkbox.setToolTip(
            "Draw each reflection as the K\u03b1\u2081 + K\u03b1\u2082 pair "
            "(2:1) a real tube produces, instead of one peak at the "
            "averaged wavelength. The split grows with angle - about "
            "0.3\u00b0 at 2\u03b8 = 90\u00b0 for Cu. Only available for "
            "the averaged K\u03b1 sources."
        )
        has_doublet = (
            kalpha_components(float(self.full_result.wavelength)) is not None
        )
        self.doublet_checkbox.setChecked(has_doublet)
        self.doublet_checkbox.setEnabled(has_doublet)
        self.doublet_checkbox.toggled.connect(self._refresh)
        row.addWidget(self.doublet_checkbox)

        row.addStretch()

        return row

    def _build_stick_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.threshold_label = QLabel("Show:")
        row.addWidget(self.threshold_label)

        self.threshold_box = QComboBox()
        self.threshold_box.setToolTip(
            "Hide reflections weaker than this fraction of the "
            "strongest peak, to declutter the pattern."
        )

        for label, value in THRESHOLD_PRESETS:
            self.threshold_box.addItem(label, value)

        default_index = next(
            (
                i
                for i, (_, value) in enumerate(THRESHOLD_PRESETS)
                if value == DEFAULT_MIN_RELATIVE_INTENSITY
            ),
            0,
        )
        self.threshold_box.setCurrentIndex(default_index)
        self.threshold_box.currentIndexChanged.connect(self._refresh)

        row.addWidget(self.threshold_box)

        row.addSpacing(16)

        self.merge_label = QLabel("Merge within (\u00b0 2\u03b8):")
        row.addWidget(self.merge_label)

        self.merge_box = QDoubleSpinBox()
        self.merge_box.setDecimals(3)
        self.merge_box.setRange(0.0, 1.0)
        self.merge_box.setSingleStep(0.01)
        self.merge_box.setValue(DEFAULT_MERGE_TOLERANCE)
        self.merge_box.setToolTip(
            "Reflections closer together than this are summed into one "
            "peak, because no diffractometer resolves them apart. "
            "Applies to the peak table in both styles."
        )
        self.merge_box.valueChanged.connect(self._refresh)

        row.addWidget(self.merge_box)

        row.addSpacing(16)

        self.summary_label = QLabel()
        row.addWidget(self.summary_label)

        row.addStretch()

        return row

    def _build_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()

        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.setToolTip(
            "Save the peak table (2θ, d-spacing, intensity, hkl) as "
            "a CSV file for use in other software."
        )
        export_png_button = QPushButton("Export PNG")
        export_png_button.setToolTip(
            "Save the pattern currently shown above as a PNG image."
        )

        self.export_csv_button.clicked.connect(self.export_csv)
        export_png_button.clicked.connect(self.export_png)

        row.addWidget(self.export_csv_button)
        row.addWidget(export_png_button)
        row.addStretch()

        return row

    @Slot()
    def _on_style_changed(self) -> None:
        is_profile = self.style_box.currentData() == PROFILE

        self.fwhm_label.setVisible(is_profile)
        self.fwhm_box.setVisible(is_profile)
        self.doublet_checkbox.setVisible(is_profile)

        self._refresh()

    @Slot()
    def _refresh(self) -> None:
        """Redraw and repopulate. No recalculation of the pattern itself:
        both the stick filtering and the profile convolution are pure
        post-processing on the already-simulated reflection list.
        """
        self.result = self.full_result.reduced(
            min_relative_intensity=self.threshold_box.currentData(),
            merge_tolerance=self.merge_box.value(),
        )

        if self.style_box.currentData() == PROFILE:
            fwhm = self.fwhm_box.value()

            peak_positions = self.full_result.two_theta
            peak_heights = self.full_result.intensities

            if self.doublet_checkbox.isChecked():
                peak_positions, peak_heights = split_kalpha_doublet(
                    peak_positions,
                    peak_heights,
                    float(self.full_result.wavelength),
                )

            two_theta, intensities = convolve_pattern(
                peak_positions,
                peak_heights,
                two_theta_range=self.full_result.effective_two_theta_range,
                fwhm=fwhm,
            )

            self._profile_two_theta = two_theta
            self._profile_intensities = intensities
            self._profile_fwhm = fwhm

            self.plot.plot_profile(
                two_theta,
                intensities,
                wavelength=self.full_result.wavelength,
                fwhm=fwhm,
            )

            self.summary_label.setText(
                f"Trace from {self.full_result.peak_count} reflections, "
                f"FWHM {fwhm:g}\u00b0  |  table: {self.result.peak_count} "
                f"peaks \u2265 {self.threshold_box.currentData():g}%"
            )

            self.export_csv_button.setText(
                f"Export CSV ({len(two_theta)} profile points)"
            )
        else:
            self._profile_two_theta = None
            self._profile_intensities = None
            self._profile_fwhm = None

            self.plot.plot_pattern(self.result, full_result=self.full_result)

            self.summary_label.setText(
                f"{self.result.peak_count} of "
                f"{self.full_result.peak_count} reflections"
            )

            self.export_csv_button.setText(
                f"Export CSV ({self.result.peak_count} peaks)"
            )

        self._populate_peak_table()

    # ------------------------------------------------------------------
    # Table
    # ------------------------------------------------------------------

    def _populate_peak_table(self) -> None:
        two_theta = self.result.two_theta
        intensities = self.result.intensities
        d_spacings = self.result.d_spacings
        hkls = getattr(self.result, "hkls", None) or []

        row_count = min(
            len(two_theta),
            len(intensities),
            len(d_spacings),
        )

        self.table.setSortingEnabled(False)
        self.table.clear()
        self.table.setRowCount(row_count)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            [
                "2\u03b8 (\u00b0)",
                "I/I\u2080 (%)",
                "d (\u00c5)",
                "hkl",
            ]
        )

        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setAlternatingRowColors(True)

        for i in range(row_count):
            self.table.setItem(
                i, 0, self._numeric_item(float(two_theta[i]), decimals=3)
            )
            self.table.setItem(
                i, 1, self._numeric_item(float(intensities[i]), decimals=2)
            )
            self.table.setItem(
                i, 2, self._numeric_item(float(d_spacings[i]), decimals=4)
            )

            hkl_text = (
                XRDExporter.format_hkl(hkls[i]) if i < len(hkls) else ""
            )
            self.table.setItem(i, 3, QTableWidgetItem(hkl_text))

        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        self.table.sortItems(1, Qt.SortOrder.DescendingOrder)

    @staticmethod
    def _numeric_item(value: float, *, decimals: int) -> QTableWidgetItem:
        """A table item that displays and sorts as a number."""
        item = QTableWidgetItem()

        item.setData(Qt.ItemDataRole.DisplayRole, round(value, decimals))
        item.setTextAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )

        return item

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_csv(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "",
            "CSV Files (*.csv)",
        )

        if not filename:
            return

        filename = self._ensure_suffix(filename, ".csv")

        try:
            if self._profile_two_theta is not None:
                # Every sampled point, not just discrete peaks - the
                # shape needed to plot a line chart in Excel and get a
                # continuous trace rather than a jagged mess with no
                # points between peaks.
                XRDExporter.export_profile_csv(
                    filename,
                    self._profile_two_theta,
                    self._profile_intensities,
                    wavelength=self.full_result.wavelength,
                    fwhm=self._profile_fwhm,
                )
            else:
                XRDExporter.export_csv(filename, self.result)
        except (OSError, ValueError, TypeError, AttributeError) as error:
            self._show_export_error(
                export_type="CSV",
                filename=filename,
                error=error,
            )

    def export_png(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export PNG",
            "",
            "PNG Files (*.png)",
        )

        if not filename:
            return

        filename = self._ensure_suffix(filename, ".png")

        try:
            self.plot.export_png(filename)
        except (OSError, ValueError, RuntimeError, AttributeError) as error:
            self._show_export_error(
                export_type="PNG",
                filename=filename,
                error=error,
            )

    @staticmethod
    def _ensure_suffix(filename: str, suffix: str) -> str:
        """Append the suffix when the user omitted it.

        Appends rather than using with_suffix, which would rewrite a
        legitimate dot in the stem: 'sample.v2' must not become
        'sample.csv'.
        """
        path = Path(filename)

        if path.suffix.lower() == suffix.lower():
            return str(path)

        return str(path.with_name(path.name + suffix))

    def _show_export_error(
        self,
        *,
        export_type: str,
        filename: str,
        error: Exception,
    ) -> None:
        QMessageBox.critical(
            self,
            f"{export_type} export failed",
            (
                f"The {export_type} file could not be written.\n\n"
                f"File: {filename}\n"
                f"Reason: {error}"
            ),
        )
