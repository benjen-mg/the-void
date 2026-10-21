from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLabel,
    QVBoxLayout,
    QWidget,
)

#: Shown when nothing is selected, and as each field's value until the
#: first atom is picked.
PLACEHOLDER = "\u2013"

#: Shown in the frac/cartesian coordinate rows when an atom was
#: selected from the structure tree but isn't part of the current 3D
#: view (hidden by a toggle, filtered by occupancy, or just not the
#: image the render happened to include).
NOT_VISIBLE = "not shown in current view"


class SelectedAtomPanel(QWidget):
    """Details for the one currently selected atom.

    Populated from either selection direction - a 3D pick or a
    structure-tree click - so it always reflects whichever the person
    did most recently, matching the single shared selection state the
    rest of the app keeps.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.label_value = QLabel(PLACEHOLDER)
        self.element_value = QLabel(PLACEHOLDER)
        self.occupancy_value = QLabel(PLACEHOLDER)
        self.fractional_value = QLabel(PLACEHOLDER)
        self.cartesian_value = QLabel(PLACEHOLDER)

        for label in (
            self.label_value,
            self.element_value,
            self.occupancy_value,
            self.fractional_value,
            self.cartesian_value,
        ):
            label.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )

        form = QFormLayout()
        form.addRow("Label:", self.label_value)
        form.addRow("Element:", self.element_value)
        form.addRow("Occupancy:", self.occupancy_value)
        form.addRow("Fractional Coordinates:", self.fractional_value)
        form.addRow("Cartesian Coordinates (\u00c5):", self.cartesian_value)

        group = QGroupBox("Selected Atom")
        group.setLayout(form)

        layout = QVBoxLayout(self)
        layout.addWidget(group)
        layout.addStretch()

        self.clear()

    def clear(self) -> None:
        """Nothing selected - the initial state, and the state after a
        new structure loads (any previous selection no longer applies).
        """
        self.label_value.setText(PLACEHOLDER)
        self.element_value.setText(PLACEHOLDER)
        self.occupancy_value.setText(PLACEHOLDER)
        self.fractional_value.setText(PLACEHOLDER)
        self.cartesian_value.setText(PLACEHOLDER)

    def set_atom(
        self,
        *,
        label: str,
        element: str,
        occupancy: float,
        fractional_coordinates,
        cartesian_coordinates,
    ) -> None:
        """Show a selected atom's details.

        cartesian_coordinates may be None when the selection came from
        the structure tree for a site that isn't part of the current 3D
        view (hidden by a toggle, or simply not drawn) - there is no
        "the" cartesian position to show in that case without implying
        a 3D pick that didn't happen, so a clear note is shown instead
        of a number that would look like real, clicked-on geometry.
        """
        self.label_value.setText(label)
        self.element_value.setText(element)
        self.occupancy_value.setText(f"{occupancy:.3f}")

        x, y, z = fractional_coordinates
        self.fractional_value.setText(f"({x:.5f}, {y:.5f}, {z:.5f})")

        if cartesian_coordinates is None:
            self.cartesian_value.setText(NOT_VISIBLE)
        else:
            cx, cy, cz = cartesian_coordinates
            self.cartesian_value.setText(f"({cx:.4f}, {cy:.4f}, {cz:.4f})")
