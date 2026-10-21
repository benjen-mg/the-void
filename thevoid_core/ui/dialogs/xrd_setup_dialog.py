from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from thevoid_core.services.xrd.powder_xrd_simulator import (
    DEFAULT_TWO_THETA_RANGE,
    DEFAULT_WAVELENGTH,
    RADIATION_SOURCES,
)


class XRDSetupDialog(QDialog):
    """Ask for the 2-theta range and radiation source before simulating.

    Both change the physics of the calculated pattern (which reflections
    fall in range, and where they land), unlike the filtering controls
    in the results dialog, which only affect what is displayed - so
    these are gathered up front rather than folded into that dialog.
    """

    def __init__(
        self,
        parent=None,
        *,
        two_theta_range: tuple[float, float] = DEFAULT_TWO_THETA_RANGE,
        wavelength_key: str = DEFAULT_WAVELENGTH,
    ) -> None:
        super().__init__(parent)

        self.setWindowTitle("Powder XRD Setup")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.min_box = QDoubleSpinBox()
        self.min_box.setRange(0.0, 179.0)
        self.min_box.setDecimals(1)
        self.min_box.setSuffix("\u00b0")
        self.min_box.setValue(two_theta_range[0])
        self.min_box.setToolTip(
            "Low end of the 2\u03b8 range to calculate over."
        )

        self.max_box = QDoubleSpinBox()
        self.max_box.setRange(1.0, 180.0)
        self.max_box.setDecimals(1)
        self.max_box.setSuffix("\u00b0")
        self.max_box.setValue(two_theta_range[1])
        self.max_box.setToolTip(
            "High end of the 2\u03b8 range to calculate over. Must be "
            "greater than 2\u03b8 min."
        )

        form.addRow("2\u03b8 min:", self.min_box)
        form.addRow("2\u03b8 max:", self.max_box)

        self.source_box = QComboBox()
        self.source_box.setToolTip(
            "The lab radiation source's wavelength, which sets where "
            "each reflection falls in 2\u03b8."
        )

        selected_index = 0

        for index, (label, key) in enumerate(RADIATION_SOURCES.items()):
            self.source_box.addItem(label, key)

            if key == wavelength_key:
                selected_index = index

        self.source_box.setCurrentIndex(selected_index)

        form.addRow("Radiation source:", self.source_box)

        layout.addLayout(form)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.button_box = buttons

    def _on_accept(self) -> None:
        if self.min_box.value() >= self.max_box.value():
            self.min_box.setStyleSheet("border: 1px solid #c0392b;")
            self.max_box.setStyleSheet("border: 1px solid #c0392b;")
            self.error_label.setText(
                "2\u03b8 min must be less than 2\u03b8 max."
            )
            self.error_label.setVisible(True)
            return

        self.min_box.setStyleSheet("")
        self.max_box.setStyleSheet("")
        self.error_label.setVisible(False)
        self.accept()

    def two_theta_range(self) -> tuple[float, float]:
        return (self.min_box.value(), self.max_box.value())

    def wavelength_key(self) -> str:
        return self.source_box.currentData()
