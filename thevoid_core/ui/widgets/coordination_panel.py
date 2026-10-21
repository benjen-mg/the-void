from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.ui.widgets.table_tools import install_table_tools

PLACEHOLDER = "\u2013"

NEIGHBOR_COLUMNS = ["Label", "Element", "Distance (\u00c5)"]


class CoordinationEnvironmentPanel(QWidget):
    """Coordination number, neighbour list, and bond length statistics
    for whichever atom's coordination environment was most recently
    analyzed.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.center_value = QLabel(PLACEHOLDER)
        self.coordination_number_value = QLabel(PLACEHOLDER)
        self.average_value = QLabel(PLACEHOLDER)
        self.min_value = QLabel(PLACEHOLDER)
        self.max_value = QLabel(PLACEHOLDER)

        summary_form = QFormLayout()
        summary_form.addRow("Centre Atom:", self.center_value)
        summary_form.addRow(
            "Coordination Number:", self.coordination_number_value
        )
        summary_form.addRow("Average Bond Length:", self.average_value)
        summary_form.addRow("Minimum Bond Length:", self.min_value)
        summary_form.addRow("Maximum Bond Length:", self.max_value)

        summary_group = QGroupBox("Coordination Environment")
        summary_group.setLayout(summary_form)

        self.neighbor_table = QTableWidget()
        self.neighbor_table.setColumnCount(len(NEIGHBOR_COLUMNS))
        self.neighbor_table.setHorizontalHeaderLabels(NEIGHBOR_COLUMNS)
        self.neighbor_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.neighbor_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.neighbor_table.setAlternatingRowColors(True)
        install_table_tools(
            self.neighbor_table, export_name="coordination.csv"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(summary_group)
        layout.addWidget(self.neighbor_table)

    def set_environment(self, environment) -> None:
        """Populate from a coordination.CoordinationEnvironment."""
        self.center_value.setText(
            f"{environment.center_label} ({environment.center_symbol})"
        )
        self.coordination_number_value.setText(
            str(environment.coordination_number)
        )

        if environment.coordination_number == 0:
            self.average_value.setText(PLACEHOLDER)
            self.min_value.setText(PLACEHOLDER)
            self.max_value.setText(PLACEHOLDER)
        else:
            self.average_value.setText(
                f"{environment.average_bond_length:.3f} \u00c5"
            )
            self.min_value.setText(f"{environment.min_bond_length:.3f} \u00c5")
            self.max_value.setText(f"{environment.max_bond_length:.3f} \u00c5")

        self.neighbor_table.setRowCount(len(environment.neighbors))

        for row, neighbor in enumerate(environment.neighbors):
            for column, text in enumerate(
                (
                    neighbor.label,
                    neighbor.symbol,
                    f"{neighbor.distance:.3f}",
                )
            ):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                    if column > 0
                    else Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                )
                self.neighbor_table.setItem(row, column, item)

        self.neighbor_table.resizeColumnsToContents()

    def clear(self) -> None:
        self.center_value.setText(PLACEHOLDER)
        self.coordination_number_value.setText(PLACEHOLDER)
        self.average_value.setText(PLACEHOLDER)
        self.min_value.setText(PLACEHOLDER)
        self.max_value.setText(PLACEHOLDER)
        self.neighbor_table.setRowCount(0)
