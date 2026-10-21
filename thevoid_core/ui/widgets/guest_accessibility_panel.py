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

from thevoid_core.data.probe_radii import PROBE_RADII
from thevoid_core.services.guest_accessibility import AccessibilityStatus

PLACEHOLDER = "\u2013"

#: A reasonable starting value for the custom-diameter spin box -
#: not itself a named guest, just something to have there before
#: someone types their own value in.
DEFAULT_CUSTOM_GUEST_DIAMETER = 3.0

GUEST_NAMES = ["CO2", "CH4", "N2", "H2", "H2O", "Custom"]

STATUS_LABELS = {
    AccessibilityStatus.ACCESSIBLE: "Accessible",
    AccessibilityStatus.PARTIALLY_ACCESSIBLE: "Partially Accessible",
    AccessibilityStatus.INACCESSIBLE: "Inaccessible",
}

STATUS_COLORS = {
    AccessibilityStatus.ACCESSIBLE: "#2eb872",
    AccessibilityStatus.PARTIALLY_ACCESSIBLE: "#f4a300",
    AccessibilityStatus.INACCESSIBLE: "#d1495b",
}


class GuestAccessibilityPanel(QWidget):
    """Controls for running guest accessibility analysis, and the
    resulting accessibility status, limiting pore size, largest
    accessible cavity, and suggested probe radius.
    """

    #: Emitted with (guest_diameter, guest_name) when "Run" is clicked.
    run_requested = Signal(float, str)

    accessible_visualization_toggled = Signal(bool)
    blocked_visualization_toggled = Signal(bool)
    bottleneck_visualization_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.guest_combo = QComboBox()
        self.guest_combo.addItems(GUEST_NAMES)
        self.guest_combo.setToolTip(
            "Which guest molecule to test - its literature kinetic "
            "diameter is used automatically, or pick Custom to enter "
            "your own."
        )
        self.guest_combo.currentTextChanged.connect(
            self._on_guest_selection_changed
        )

        self.custom_diameter_box = QDoubleSpinBox()
        self.custom_diameter_box.setRange(0.2, 20.0)
        self.custom_diameter_box.setSingleStep(0.1)
        self.custom_diameter_box.setDecimals(2)
        self.custom_diameter_box.setValue(DEFAULT_CUSTOM_GUEST_DIAMETER)
        self.custom_diameter_box.setSuffix(" \u00c5")
        self.custom_diameter_box.setEnabled(False)
        self.custom_diameter_box.setToolTip(
            "The guest's kinetic diameter, in \u00e5ngstr\u00f6ms - only "
            "editable when Custom is selected above."
        )
        # See SurfaceAreaPanel's identical custom_probe_radius_box for
        # why: no spin arrows while showing a fixed, non-editable named-
        # guest diameter. Restored in _on_guest_selection_changed the
        # moment Custom is chosen.
        self.custom_diameter_box.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.NoButtons
        )

        self.run_button = QPushButton("Run Guest Accessibility Analysis")
        self.run_button.clicked.connect(
            lambda: self.run_requested.emit(
                self.guest_diameter(), self.guest_combo.currentText()
            )
        )

        controls_form = QFormLayout()
        controls_form.addRow("Guest Molecule:", self.guest_combo)
        controls_form.addRow(
            "Custom Probe Diameter:", self.custom_diameter_box
        )

        controls_layout = QVBoxLayout()
        controls_layout.addLayout(controls_form)
        controls_layout.addWidget(self.run_button)

        self.show_accessible_checkbox = QCheckBox(
            "Highlight accessible channels"
        )
        self.show_accessible_checkbox.setChecked(True)
        self.show_accessible_checkbox.setEnabled(False)
        self.show_accessible_checkbox.toggled.connect(
            self.accessible_visualization_toggled.emit
        )
        controls_layout.addWidget(self.show_accessible_checkbox)

        self.show_blocked_checkbox = QCheckBox(
            "Highlight inaccessible regions"
        )
        self.show_blocked_checkbox.setChecked(False)
        self.show_blocked_checkbox.setEnabled(False)
        self.show_blocked_checkbox.toggled.connect(
            self.blocked_visualization_toggled.emit
        )
        controls_layout.addWidget(self.show_blocked_checkbox)

        self.show_bottlenecks_checkbox = QCheckBox("Show pore bottlenecks")
        self.show_bottlenecks_checkbox.setChecked(False)
        self.show_bottlenecks_checkbox.setEnabled(False)
        self.show_bottlenecks_checkbox.toggled.connect(
            self.bottleneck_visualization_toggled.emit
        )
        controls_layout.addWidget(self.show_bottlenecks_checkbox)

        controls_group = QGroupBox("Settings")
        controls_group.setLayout(controls_layout)

        self.status_value = QLabel(PLACEHOLDER)
        self.limiting_pore_size_value = QLabel(PLACEHOLDER)
        self.largest_accessible_cavity_value = QLabel(PLACEHOLDER)
        self.suggested_probe_radius_value = QLabel(PLACEHOLDER)

        results_form = QFormLayout()
        results_form.addRow("Accessibility Status:", self.status_value)
        results_form.addRow(
            "Limiting Pore Size:", self.limiting_pore_size_value
        )
        results_form.addRow(
            "Largest Accessible Cavity:",
            self.largest_accessible_cavity_value,
        )
        results_form.addRow(
            "Suggested Probe Radius:", self.suggested_probe_radius_value
        )

        results_group = QGroupBox("Results")
        results_group.setLayout(results_form)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        layout.addWidget(self.detail_label)
        layout.addStretch()

    def _on_guest_selection_changed(self, name: str) -> None:
        is_custom = name == "Custom"
        self.custom_diameter_box.setEnabled(is_custom)
        self.custom_diameter_box.setButtonSymbols(
            QAbstractSpinBox.ButtonSymbols.UpDownArrows
            if is_custom
            else QAbstractSpinBox.ButtonSymbols.NoButtons
        )

    def guest_diameter(self) -> float:
        name = self.guest_combo.currentText()

        if name == "Custom":
            return self.custom_diameter_box.value()

        return 2.0 * PROBE_RADII[name]

    def set_result(self, result) -> None:
        """Populate from a guest_accessibility.GuestAccessibilityResult."""
        status_text = STATUS_LABELS[result.status]
        color = STATUS_COLORS[result.status]
        self.status_value.setText(status_text)
        self.status_value.setStyleSheet(f"color: {color}; font-weight: 600;")

        if result.minimum_window_diameter is not None:
            self.limiting_pore_size_value.setText(
                f"{result.minimum_window_diameter:.3f} \u00c5"
            )
        else:
            self.limiting_pore_size_value.setText(
                "none (no percolating pathway)"
            )

        self.largest_accessible_cavity_value.setText(
            f"{result.largest_accessible_cavity:.3f} \u00c5"
        )

        if result.suggested_probe_radius is not None:
            self.suggested_probe_radius_value.setText(
                f"{result.suggested_probe_radius:.3f} \u00c5"
            )
        else:
            self.suggested_probe_radius_value.setText(PLACEHOLDER)

        for checkbox in (
            self.show_accessible_checkbox,
            self.show_blocked_checkbox,
            self.show_bottlenecks_checkbox,
        ):
            checkbox.setEnabled(True)

        self.detail_label.setText(
            f"Guest: {result.guest_name} ({result.guest_diameter:.2f} "
            f"\u00c5 diameter). Largest Cavity Diameter (any cavity, "
            f"reachable or not): {result.largest_cavity_diameter:.3f} "
            f"\u00c5. {len(result.bottlenecks)} bottleneck(s) found."
        )

    def clear(self) -> None:
        self.status_value.setText(PLACEHOLDER)
        self.status_value.setStyleSheet("")

        for label in (
            self.limiting_pore_size_value,
            self.largest_accessible_cavity_value,
            self.suggested_probe_radius_value,
        ):
            label.setText(PLACEHOLDER)

        self.detail_label.setText("")

        for checkbox, default_checked in (
            (self.show_accessible_checkbox, True),
            (self.show_blocked_checkbox, False),
            (self.show_bottlenecks_checkbox, False),
        ):
            checkbox.setEnabled(False)
            checkbox.setChecked(default_checked)
