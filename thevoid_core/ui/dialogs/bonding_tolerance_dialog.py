from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
)

from thevoid_core.services.bonds.covalent_bond_detector import (
    CovalentBondDetector,
)


class BondingToleranceDialog(QDialog):
    """Adjust the slack added to summed covalent radii when deciding
    whether two atoms are bonded.

    The default (0.4 A) matches the convention most crystal-structure
    databases use and is right for the great majority of files. The
    lever exists for the genuine edge cases: an extended inorganic
    framework with longer, more ionic-character metal-ligand contacts
    can be under-bonded at the default value, and raising the tolerance
    is the direct way to address that without a second bonding
    algorithm to choose between.
    """

    MIN_TOLERANCE = 0.0
    MAX_TOLERANCE = 1.0

    def __init__(self, current: float, parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Bonding Tolerance")

        layout = QVBoxLayout(self)

        note = QLabel(
            "Two atoms are bonded when their separation is no greater "
            "than the sum of their covalent radii, plus this tolerance. "
            "Raise it if a structure with long or ionic-character bonds "
            "(an extended framework, for instance) looks under-bonded; "
            "lower it if unrelated atoms are being drawn as bonded."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()

        self.tolerance_box = QDoubleSpinBox()
        self.tolerance_box.setDecimals(2)
        self.tolerance_box.setRange(self.MIN_TOLERANCE, self.MAX_TOLERANCE)
        self.tolerance_box.setSingleStep(0.05)
        self.tolerance_box.setSuffix(" \u00c5")
        self.tolerance_box.setValue(current)
        self.tolerance_box.setToolTip(
            "Extra slack, in \u00e5ngstr\u00f6ms, added on top of two "
            "atoms' summed covalent radii before a contact stops "
            "counting as a bond."
        )

        form.addRow("Tolerance:", self.tolerance_box)
        layout.addLayout(form)

        default_label = QLabel(
            f"Default: {CovalentBondDetector.TOLERANCE:g} \u00c5"
        )
        default_label.setStyleSheet("color: gray;")
        layout.addWidget(default_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.RestoreDefaults
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(
            QDialogButtonBox.StandardButton.RestoreDefaults
        ).clicked.connect(
            lambda: self.tolerance_box.setValue(
                CovalentBondDetector.TOLERANCE
            )
        )
        layout.addWidget(buttons)

    def tolerance(self) -> float:
        return self.tolerance_box.value()
