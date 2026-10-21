from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class SupercellDialog(QDialog):
    """Ask for arbitrary a x b x c cell repeats.

    The preset menu only offers 1x1x1 / 2x2x2 / 3x3x3; this covers
    everything else, including asymmetric tilings like 4x1x2.
    """

    MIN_DIM = 1
    MAX_DIM = 10

    def __init__(self, current: tuple[int, int, int], parent=None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Custom Supercell")

        layout = QVBoxLayout(self)

        note = QLabel(
            "Number of unit cells to repeat along each lattice vector, "
            "for display only - every analysis still runs on the "
            "single crystallographic cell. Large values (several "
            "hundred cells or more) can be slow to render."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()

        self.a_box = self._make_spinbox(current[0])
        self.b_box = self._make_spinbox(current[1])
        self.c_box = self._make_spinbox(current[2])

        form.addRow("a:", self.a_box)
        form.addRow("b:", self.b_box)
        form.addRow("c:", self.c_box)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _make_spinbox(self, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(self.MIN_DIM, self.MAX_DIM)
        box.setValue(value)
        box.setToolTip(
            f"Repeats along this axis ({self.MIN_DIM}-{self.MAX_DIM})."
        )
        return box

    def dimensions(self) -> tuple[int, int, int]:
        return (
            self.a_box.value(),
            self.b_box.value(),
            self.c_box.value(),
        )
