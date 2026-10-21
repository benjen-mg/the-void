from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.services.surface_area import PROBE_RADII

PLACEHOLDER = "\u2013"

#: A generic default custom-probe starting value - not itself one of
#: the named probes, just a reasonable number to have in the spin box
#: before someone types their own in.
DEFAULT_CUSTOM_PROBE_RADIUS = 1.5

PROBE_NAMES = list(PROBE_RADII.keys()) + ["Custom"]


class SurfaceAreaPanel(QWidget):
    """Controls for running accessible surface area analysis, and the
    resulting ASA, geometric surface area, per-mass/per-volume figures,
    and total/internal accessible area.
    """

    #: Emitted with (probe_radius, probe_name) when "Run" is clicked.
    run_requested = Signal(float, str)

    #: Emitted whenever either visualization checkbox changes state.
    points_visualization_toggled = Signal(bool)
    mesh_visualization_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.probe_combo = QComboBox()
        self.probe_combo.addItems(PROBE_NAMES)
        self.probe_combo.setToolTip(
            "Which probe molecule to roll over the surface - its "
            "literature radius is used automatically, or pick Custom "
            "to enter your own."
        )
        self.probe_combo.currentTextChanged.connect(
            self._on_probe_selection_changed
        )

        self.custom_probe_radius_box = QDoubleSpinBox()
        self.custom_probe_radius_box.setRange(0.1, 10.0)
        self.custom_probe_radius_box.setSingleStep(0.1)
        self.custom_probe_radius_box.setDecimals(2)
        self.custom_probe_radius_box.setValue(DEFAULT_CUSTOM_PROBE_RADIUS)
        self.custom_probe_radius_box.setSuffix(" \u00c5")
        self.custom_probe_radius_box.setEnabled(False)
        self.custom_probe_radius_box.setToolTip(
            "The probe's radius, in \u00e5ngstr\u00f6ms - only editable "
            "when Custom is selected above."
        )
        # No spin arrows while showing a fixed, non-editable named-
        # probe radius - they would otherwise sit there greyed out,
        # looking like a broken control rather than a value that simply
        # isn't editable in this mode. Restored in
        # _on_probe_selection_changed the moment Custom is chosen.
        self.custom_probe_radius_box.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )

        self.run_button = QPushButton("Run Surface Area Analysis")
        self.run_button.clicked.connect(
            lambda: self.run_requested.emit(
                self.probe_radius(), self.probe_combo.currentText()
            )
        )

        controls_form = QFormLayout()
        controls_form.addRow("Probe Molecule:", self.probe_combo)
        controls_form.addRow(
            "Custom Probe Radius:", self.custom_probe_radius_box
        )

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(controls_form)
        controls_layout.addWidget(self.run_button)

        self.show_points_checkbox = QCheckBox(
            "Show sampled surface points"
        )
        self.show_points_checkbox.setChecked(True)
        self.show_points_checkbox.setEnabled(False)
        self.show_points_checkbox.toggled.connect(
            self.points_visualization_toggled.emit
        )
        controls_layout.addWidget(self.show_points_checkbox)

        self.show_mesh_checkbox = QCheckBox("Show surface mesh overlay")
        self.show_mesh_checkbox.setChecked(False)
        self.show_mesh_checkbox.setEnabled(False)
        self.show_mesh_checkbox.toggled.connect(
            self.mesh_visualization_toggled.emit
        )
        controls_layout.addWidget(self.show_mesh_checkbox)

        controls_group = QGroupBox("Settings")
        controls_group.setLayout(controls_layout)

        self.asa_value = QLabel(PLACEHOLDER)
        self.geometric_value = QLabel(PLACEHOLDER)
        self.asa_per_mass_value = QLabel(PLACEHOLDER)
        self.asa_per_volume_value = QLabel(PLACEHOLDER)
        self.total_accessible_value = QLabel(PLACEHOLDER)
        self.internal_accessible_value = QLabel(PLACEHOLDER)
        self.pocket_value = QLabel(PLACEHOLDER)
        self.channel_asa_per_mass_value = QLabel(PLACEHOLDER)

        results_form = QFormLayout()
        results_form.addRow(
            "Accessible Surface Area (ASA):", self.asa_value
        )
        results_form.addRow(
            "Geometric Surface Area:", self.geometric_value
        )
        results_form.addRow("ASA per mass:", self.asa_per_mass_value)
        results_form.addRow("ASA per volume:", self.asa_per_volume_value)
        results_form.addRow(
            "Total Accessible Area:", self.total_accessible_value
        )
        results_form.addRow(
            "Channel Accessible Area:", self.internal_accessible_value
        )
        results_form.addRow("Pocket Area:", self.pocket_value)
        results_form.addRow(
            "Channel ASA per mass:", self.channel_asa_per_mass_value
        )

        self.internal_accessible_value.setToolTip(
            "Surface bordering the connected channel network - the "
            "part a probe can actually reach from outside the crystal "
            "(Zeo++'s ASA)."
        )
        self.pocket_value.setToolTip(
            "Surface a probe could touch only from inside a sealed "
            "pocket it cannot get into (Zeo++'s NASA)."
        )
        self.channel_asa_per_mass_value.setToolTip(
            "Channel-accessible area per gram - comparable to the "
            "ASA m\u00b2/g figure Zeo++ and most screening studies "
            "report. \"ASA per mass\" above includes pockets."
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

    def _on_probe_selection_changed(self, name: str) -> None:
        is_custom = name == "Custom"
        self.custom_probe_radius_box.setEnabled(is_custom)
        self.custom_probe_radius_box.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.UpDownArrows
            if is_custom
            else QAbstractSpinBox.ButtonSymbols.NoButtons
        )

    def probe_radius(self) -> float:
        name = self.probe_combo.currentText()

        if name == "Custom":
            return self.custom_probe_radius_box.value()

        return PROBE_RADII[name]

    def set_result(self, result) -> None:
        """Populate from a surface_area.SurfaceAreaResult."""
        self.asa_value.setText(
            f"{result.accessible_surface_area:.2f} \u00c5\u00b2"
        )
        self.geometric_value.setText(
            f"{result.geometric_surface_area:.2f} \u00c5\u00b2"
        )
        self.asa_per_mass_value.setText(f"{result.asa_per_mass:.2f} m\u00b2/g")
        self.asa_per_volume_value.setText(
            f"{result.asa_per_volume:.2f} m\u00b2/cm\u00b3"
        )
        self.total_accessible_value.setText(
            f"{result.total_accessible_area:.2f} \u00c5\u00b2"
        )
        self.internal_accessible_value.setText(
            f"{result.internal_accessible_area:.2f} \u00c5\u00b2"
        )
        self.pocket_value.setText(
            f"{getattr(result, 'pocket_accessible_area', 0.0):.2f} "
            "\u00c5\u00b2"
        )
        self.channel_asa_per_mass_value.setText(
            f"{getattr(result, 'channel_asa_per_mass', 0.0):.2f} m\u00b2/g"
        )

        self.show_points_checkbox.setEnabled(True)
        self.show_mesh_checkbox.setEnabled(True)

        self.status_label.setText(
            f"Probe: {result.probe_name} ({result.probe_radius:.2f} "
            f"\u00c5), {result.n_samples_per_atom} samples/atom."
        )

    def clear(self) -> None:
        for label in (
            self.asa_value,
            self.geometric_value,
            self.asa_per_mass_value,
            self.asa_per_volume_value,
            self.total_accessible_value,
            self.internal_accessible_value,
            self.pocket_value,
            self.channel_asa_per_mass_value,
        ):
            label.setText(PLACEHOLDER)

        self.status_label.setText("")
        self.show_points_checkbox.setEnabled(False)
        self.show_points_checkbox.setChecked(True)
        self.show_mesh_checkbox.setEnabled(False)
        self.show_mesh_checkbox.setChecked(False)
