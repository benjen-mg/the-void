from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.ui.widgets.table_tools import install_table_tools

#: Column: header label, data key on the underlying result object.
LINKER_COLUMNS = ["ID", "Formula", "Atom Count"]
#: "Metal Atoms" was headed "Coordination Number" before round 55, but
#: the value has always been the node's metal count (see
#: MetalNode.coordination_number), not a coordination number. "Extent"
#: says whether the node is a finite cluster or an infinite rod/layer,
#: in which case its formula is the part within one unit cell.
NODE_COLUMNS = ["ID", "Metal Type(s)", "Composition", "Metal Atoms", "Extent"]

EXTENT_LABELS = {0: "cluster", 1: "rod (1D)", 2: "layer (2D)", 3: "3D inorganic"}

#: Qt.ItemDataRole.UserRole slot each row's atom_indices are stored
#: under, for the click-to-highlight handler to retrieve.
ATOM_INDICES_ROLE = Qt.ItemDataRole.UserRole


def summary_text(partition) -> str:
    """One line for the panel: counts per unit cell, plus what a reader
    needs to read them right - that a rod node's formula is one cell's
    share of an infinite chain, and how many disorder sites were left
    out."""
    text = (
        f"{partition.linker_count} linker(s), "
        f"{partition.metal_node_count} metal node(s) per unit cell"
    )
    periodic = [n for n in partition.metal_nodes if getattr(n, "dimensionality", 0)]
    if periodic:
        kinds = sorted({EXTENT_LABELS.get(n.dimensionality, "?") for n in periodic})
        text += (
            f"; {len(periodic)} node(s) are infinite ({', '.join(kinds)}) - "
            "composition is the part inside one cell"
        )
    excluded = getattr(partition, "excluded_disorder_sites", 0)
    if excluded:
        text += f"; {excluded} minor-disorder site(s) left out"
    return text


class MOFAnalysisPanel(QWidget):
    """Linkers and metal nodes from the most recent MOF partition,
    each in their own table. Selecting a row highlights that linker or
    node in the 3D view - the two tables use different highlight
    colours so switching between them stays unambiguous.
    """

    #: Emitted with (atom_indices, colour) when a row is selected -
    #: MainWindow forwards this straight to
    #: CrystalViewer.highlight_group() without needing to know
    #: anything about which table the click came from.
    highlight_requested = Signal(object, str)

    LINKER_COLOR = "#ff8800"
    NODE_COLOR = "#3399ff"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.summary_label = QLabel("No MOF analysis run yet.")

        self.linker_table = self._make_table(LINKER_COLUMNS)
        self.linker_table.itemSelectionChanged.connect(
            lambda: self._on_selection_changed(
                self.linker_table, self.LINKER_COLOR
            )
        )

        self.node_table = self._make_table(NODE_COLUMNS)
        self.node_table.itemSelectionChanged.connect(
            lambda: self._on_selection_changed(
                self.node_table, self.NODE_COLOR
            )
        )

        # Round 73: linkers and nodes side by side, each under its own
        # heading with its count, instead of on two tabs of a dock that
        # left room for one row or none. The splitter between them can
        # be dragged.
        self.linker_heading = QLabel("Linkers")
        self.node_heading = QLabel("Metal Nodes")
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        for heading, table in ((self.linker_heading, self.linker_table),
                               (self.node_heading, self.node_table)):
            heading.setStyleSheet("font-weight: bold;")
            column = QWidget()
            column_layout = QVBoxLayout(column)
            column_layout.setContentsMargins(0, 0, 0, 0)
            column_layout.addWidget(heading)
            column_layout.addWidget(table, 1)
            self.splitter.addWidget(column)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)

        self.summary_label.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.splitter, 1)

    @staticmethod
    def _make_table(columns: list[str]) -> QTableWidget:
        table = QTableWidget()
        table.setColumnCount(len(columns))
        table.setHorizontalHeaderLabels(columns)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        table.setAlternatingRowColors(True)
        install_table_tools(table, export_name="mof_analysis.csv")
        return table

    def set_partition(self, partition) -> None:
        """Populate both tables from a mof_analysis.FrameworkPartition."""
        self.summary_label.setText(summary_text(partition))
        self.summary_label.setWordWrap(True)

        self.linker_table.setRowCount(len(partition.linkers))

        for row, linker in enumerate(partition.linkers):
            self._set_row(
                self.linker_table,
                row,
                [
                    str(linker.linker_id),
                    linker.formula,
                    str(linker.atom_count),
                ],
                linker.atom_indices,
            )

        self.node_table.setRowCount(len(partition.metal_nodes))

        for row, node in enumerate(partition.metal_nodes):
            metal_types = ", ".join(sorted(set(node.metal_symbols)))

            self._set_row(
                self.node_table,
                row,
                [
                    str(node.node_id),
                    metal_types,
                    node.formula,
                    str(node.coordination_number),
                    EXTENT_LABELS.get(getattr(node, "dimensionality", 0), "?"),
                ],
                node.atom_indices,
            )

        self.linker_heading.setText(f"Linkers ({len(partition.linkers)})")
        self.node_heading.setText(f"Metal Nodes ({len(partition.metal_nodes)})")
        self.linker_table.resizeColumnsToContents()
        self.node_table.resizeColumnsToContents()

    def clear(self) -> None:
        self.summary_label.setText("No MOF analysis run yet.")
        self.linker_heading.setText("Linkers")
        self.node_heading.setText("Metal Nodes")
        self.linker_table.setRowCount(0)
        self.node_table.setRowCount(0)

    @staticmethod
    def _set_row(table, row, values, atom_indices) -> None:
        for column, text in enumerate(values):
            item = QTableWidgetItem(text)
            item.setTextAlignment(
                Qt.AlignmentFlag.AlignCenter
                if column > 0
                else Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )

            if column == 0:
                item.setData(ATOM_INDICES_ROLE, list(atom_indices))

            table.setItem(row, column, item)

    def _on_selection_changed(self, table, color: str) -> None:
        selected = table.selectedItems()

        if not selected:
            return

        # The atom indices are stored on column 0's item regardless of
        # which cell in the row was actually clicked.
        row = selected[0].row()
        id_item = table.item(row, 0)

        atom_indices = id_item.data(ATOM_INDICES_ROLE) if id_item else None

        if atom_indices is not None:
            self.highlight_requested.emit(atom_indices, color)
