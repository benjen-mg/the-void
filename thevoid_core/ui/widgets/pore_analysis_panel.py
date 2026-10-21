from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PLACEHOLDER = "\u2013"

#: A generic small-molecule probe (roughly half the kinetic diameter of
#: N2, the most common probe in the porous-materials literature) - a
#: reasonable default to land on, not a claim that it is the right
#: choice for every material or every question being asked.
DEFAULT_PROBE_RADIUS = 1.2

DIMENSIONALITY_LABELS = {
    0: "None (isolated pockets only)",
    1: "1D (channel)",
    2: "2D (layered)",
    3: "3D (interconnected)",
}


class PoreAnalysisPanel(QWidget):
    """Controls for running pore analysis, and the resulting LCD, PLD,
    void fraction, accessible/non-accessible volume, and accessible
    network dimensionality.
    """

    #: Emitted with the chosen probe radius when "Run Pore Analysis" is
    #: clicked - MainWindow owns actually running the (potentially slow)
    #: analysis on a background thread and calling set_result() back.
    run_requested = Signal(float)

    #: Emitted whenever the visualization checkbox changes state.
    visualization_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.probe_radius_box = QDoubleSpinBox()
        self.probe_radius_box.setRange(0.1, 10.0)
        self.probe_radius_box.setSingleStep(0.1)
        self.probe_radius_box.setDecimals(2)
        self.probe_radius_box.setValue(DEFAULT_PROBE_RADIUS)
        self.probe_radius_box.setSuffix(" \u00c5")
        self.probe_radius_box.setToolTip(
            "The spherical probe's radius, in \u00e5ngstr\u00f6ms - a "
            "rough guide: N\u2082 \u2248 1.82, CO\u2082 \u2248 1.53, "
            "H\u2082 \u2248 1.48, H\u2082O \u2248 1.58. Smaller finds "
            "more pore space; larger is stricter about what counts as "
            "accessible."
        )

        self.run_button = QPushButton("Run Pore Analysis")
        self.run_button.clicked.connect(
            lambda: self.run_requested.emit(self.probe_radius_box.value())
        )

        controls_form = QFormLayout()
        controls_form.addRow("Probe Radius:", self.probe_radius_box)

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(controls_form)
        controls_layout.addWidget(self.run_button)

        self.show_visualization_checkbox = QCheckBox(
            "Show pore space in 3D view"
        )
        self.show_visualization_checkbox.setChecked(True)
        self.show_visualization_checkbox.setEnabled(False)
        self.show_visualization_checkbox.toggled.connect(
            self.visualization_toggled.emit
        )
        controls_layout.addWidget(self.show_visualization_checkbox)

        controls_group = QGroupBox("Settings")
        controls_group.setLayout(controls_layout)

        self.lcd_value = QLabel(PLACEHOLDER)
        self.pld_value = QLabel(PLACEHOLDER)
        self.lcd_value.setToolTip(
            "Largest cavity diameter: the biggest sphere that fits "
            "anywhere in the framework, found by maximising the exact "
            "distance to the atom surfaces (not limited to grid points)."
        )
        self.pld_value.setToolTip(
            "Pore limiting diameter: the biggest sphere that can travel "
            "all the way through the crystal, from the bottleneck of the "
            "widest path along the atoms' Voronoi network (the Zeo++ "
            "method), refined for unequal atom radii."
        )
        self.void_fraction_value = QLabel(PLACEHOLDER)
        self.accessible_volume_value = QLabel(PLACEHOLDER)
        self.non_accessible_volume_value = QLabel(PLACEHOLDER)
        self.dimensionality_value = QLabel(PLACEHOLDER)

        results_form = QFormLayout()
        results_form.addRow("Largest Cavity Diameter (LCD):", self.lcd_value)
        results_form.addRow("Pore Limiting Diameter (PLD):", self.pld_value)
        results_form.addRow("Void Fraction:", self.void_fraction_value)
        results_form.addRow(
            "Accessible Volume:", self.accessible_volume_value
        )
        results_form.addRow(
            "Non-accessible Volume:", self.non_accessible_volume_value
        )
        results_form.addRow(
            "Accessible Network Dimensionality:", self.dimensionality_value
        )

        results_group = QGroupBox("Results")
        results_group.setLayout(results_form)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        layout.addWidget(self.status_label)
        layout.addStretch()

    def set_result(self, result) -> None:
        """Populate from a pore_analysis.PoreAnalysisResult."""
        self.lcd_value.setText(
            f"{result.lcd:.3f} \u00c5"
            if result.lcd not in (float("inf"),)
            else "unbounded (empty structure)"
        )
        self.pld_value.setText(
            f"{result.pld:.3f} \u00c5"
            if result.pld is not None
            else "none (no percolating pathway found)"
        )
        self.void_fraction_value.setText(
            f"{result.void_fraction_percent:.2f} %"
        )
        self.accessible_volume_value.setText(
            f"{result.accessible_volume:.2f} \u00c5\u00b3"
        )
        self.non_accessible_volume_value.setText(
            f"{result.non_accessible_volume:.2f} \u00c5\u00b3"
        )
        self.dimensionality_value.setText(
            DIMENSIONALITY_LABELS.get(
                result.accessible_dimensionality,
                str(result.accessible_dimensionality),
            )
        )

        self.show_visualization_checkbox.setEnabled(True)
        self.status_label.setText(
            f"Probe radius {result.probe_radius:.2f} \u00c5, grid "
            f"{result.grid_shape[0]}\u00d7{result.grid_shape[1]}\u00d7"
            f"{result.grid_shape[2]}. "
            + (
                "LCD/PLD from the atoms' Voronoi network (grid-independent)."
                if getattr(result, "geometry_method", "grid") == "voronoi"
                else "LCD/PLD read off the grid."
            )
        )

    def clear(self) -> None:
        for label in (
            self.lcd_value,
            self.pld_value,
            self.void_fraction_value,
            self.accessible_volume_value,
            self.non_accessible_volume_value,
            self.dimensionality_value,
        ):
            label.setText(PLACEHOLDER)

        self.status_label.setText("")
        self.show_visualization_checkbox.setEnabled(False)
        self.show_visualization_checkbox.setChecked(True)

    def probe_radius(self) -> float:
        return self.probe_radius_box.value()
