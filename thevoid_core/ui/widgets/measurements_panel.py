from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.ui.widgets.table_tools import install_table_tools

COLUMNS = [
    "Type",
    "Atom A",
    "Element A",
    "Atom B",
    "Element B",
    "Atom C",
    "Element C",
    "Value",
    "Atom D",
    "Element D",
]

#: Logical column of the Value cell. Atom D / Element D (torsion, round
#: 49) were appended after it so every existing column keeps its
#: logical index; the header then moves Value visually to the end, so
#: on screen (and in Copy/Export, which follow the visual order) the
#: columns still read A, B, C, D, Value.
VALUE_COLUMN = COLUMNS.index("Value")

#: Text shown in the Atom C / Element C columns for a distance row,
#: which only ever involves two atoms. A visible placeholder rather
#: than an empty cell, so it reads as "not applicable to this row"
#: instead of looking like missing data.
NOT_APPLICABLE = "\u2013"


class MeasurementsPanel(QWidget):
    """An accumulating record of every distance and angle measurement
    taken so far - the "Allow multiple measurements" requirement made
    visible, rather than each new measurement silently overwriting the
    last one's on-screen readout.

    One row per completed measurement, with a dedicated Atom C /
    Element C column pair for an angle's third atom (the one that
    isn't the vertex) so all three atoms in an angle are genuinely
    visible in their own columns rather than two of them appearing as
    normal columns and the third only readable inside the Value
    column's text. Left blank (Atom C / Element C show "-") for a
    distance row, which only ever involves two atoms.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.table = QTableWidget()
        self.table.setColumnCount(len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setAlternatingRowColors(True)
        header = self.table.horizontalHeader()
        header.moveSection(
            header.visualIndex(VALUE_COLUMN), len(COLUMNS) - 1
        )
        install_table_tools(self.table, export_name="measurements.csv")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)

    def add_distance(
        self,
        *,
        label_a: str,
        element_a: str,
        label_b: str,
        element_b: str,
        distance_angstrom: float,
    ) -> None:
        self._add_row(
            kind="Distance",
            label_a=label_a,
            element_a=element_a,
            label_b=label_b,
            element_b=element_b,
            label_c=NOT_APPLICABLE,
            element_c=NOT_APPLICABLE,
            value_text=f"{distance_angstrom:.3f} \u00c5",
        )

    def add_angle(
        self,
        *,
        label_a: str,
        element_a: str,
        label_vertex: str,
        element_vertex: str,
        label_c: str,
        element_c: str,
        angle_degrees: float,
    ) -> None:
        self._add_row(
            kind="Angle",
            label_a=label_a,
            element_a=element_a,
            label_b=label_vertex,
            element_b=element_vertex,
            label_c=label_c,
            element_c=element_c,
            value_text=f"{angle_degrees:.1f}\u00b0",
        )

    def add_torsion(
        self,
        *,
        label_a: str,
        element_a: str,
        label_b: str,
        element_b: str,
        label_c: str,
        element_c: str,
        label_d: str,
        element_d: str,
        torsion_degrees: float,
    ) -> None:
        """A signed A-B-C-D dihedral about the B-C bond."""
        self._add_row(
            kind="Torsion",
            label_a=label_a,
            element_a=element_a,
            label_b=label_b,
            element_b=element_b,
            label_c=label_c,
            element_c=element_c,
            label_d=label_d,
            element_d=element_d,
            value_text=f"{torsion_degrees:.1f}\u00b0",
        )

    def clear(self) -> None:
        self.table.setRowCount(0)

    def _add_row(
        self,
        *,
        kind: str,
        label_a: str,
        element_a: str,
        label_b: str,
        element_b: str,
        label_c: str,
        element_c: str,
        value_text: str,
        label_d: str = NOT_APPLICABLE,
        element_d: str = NOT_APPLICABLE,
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        values = (
            kind,
            label_a,
            element_a,
            label_b,
            element_b,
            label_c,
            element_c,
            value_text,
            label_d,
            element_d,
        )

        for column, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
                if column > 0
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            self.table.setItem(row, column, item)

        self.table.resizeColumnsToContents()
        self.table.scrollToBottom()
