from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.ui.widgets.table_tools import install_table_tools
from thevoid_core.data.metal_classification import OMS_SUPPORTED_METALS
from thevoid_core.services.oms_detection import OMSClassification

PLACEHOLDER = "\u2013"

SITE_COLUMNS = ["Label", "Metal", "CN", "Status", "Asymmetry"]
NEIGHBOR_COLUMNS = ["Label", "Element", "Distance (\u00c5)"]

CLASSIFICATION_LABELS = {
    OMSClassification.OPEN: "Open",
    OMSClassification.SATURATED: "Saturated",
}
CLASSIFICATION_COLORS = {
    OMSClassification.OPEN: "#ff5ea8",
    OMSClassification.SATURATED: "#6b7280",
}

#: Index (into a row's underlying OMSSite) stored on the row's first
#: cell, so a click can look the site back up without re-running
#: detection or re-deriving it from displayed text.
SITE_INDEX_ROLE = Qt.ItemDataRole.UserRole


class OMSPanel(QWidget):
    """Controls for running Open Metal Site detection, a table of
    every analyzed metal centre, and inspection detail (coordination
    environment, location, missing positions) for whichever row is
    selected.
    """

    run_requested = Signal()

    #: Emitted with the OMSSite object for the selected row - "OMS
    #: inspection mode" - or None when the selection is cleared.
    site_selected = Signal(object)

    highlight_all_toggled = Signal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.run_button = QPushButton("Run OMS Detection")
        self.run_button.setToolTip(
            "Find undercoordinated metal centres by the geometric gap "
            "in their coordination sphere. Only checks these metals: "
            + ", ".join(sorted(OMS_SUPPORTED_METALS))
            + ". Any other metal in the structure is skipped and named "
            "in the result summary as not checked."
        )
        self.run_button.clicked.connect(self.run_requested.emit)

        self.highlight_all_checkbox = QCheckBox(
            "Highlight all open metal sites"
        )
        self.highlight_all_checkbox.setChecked(True)
        self.highlight_all_checkbox.setEnabled(False)
        self.highlight_all_checkbox.toggled.connect(
            self.highlight_all_toggled.emit
        )

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.run_button)
        controls_layout.addWidget(self.highlight_all_checkbox)

        controls_group = QGroupBox("Settings")
        controls_group.setLayout(controls_layout)

        self.summary_label = QLabel("No OMS detection run yet.")
        self.summary_label.setWordWrap(True)

        self.site_table = QTableWidget()
        self.site_table.setColumnCount(len(SITE_COLUMNS))
        self.site_table.setHorizontalHeaderLabels(SITE_COLUMNS)
        self.site_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.site_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.site_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.site_table.setAlternatingRowColors(True)
        install_table_tools(self.site_table, export_name="oms_sites.csv")
        self.site_table.itemSelectionChanged.connect(
            self._on_selection_changed
        )

        # --- Inspection detail (OMS inspection mode) ---

        self.detail_metal_value = QLabel(PLACEHOLDER)
        self.detail_cn_value = QLabel(PLACEHOLDER)
        self.detail_status_value = QLabel(PLACEHOLDER)
        self.detail_location_value = QLabel(PLACEHOLDER)
        self.detail_missing_value = QLabel(PLACEHOLDER)
        self.detail_gap_value = QLabel(PLACEHOLDER)
        self.detail_gap_value.setToolTip(
            "The widest empty cone around the metal: how far (in "
            "degrees) the emptiest direction is from its nearest "
            "ligand. Saturated polyhedra stay below about 71 "
            "(tetrahedron); a vacant site gives about 90. Above 80 "
            "counts as open even when the vacancies are symmetric "
            "(e.g. square-planar with both axial sites empty)."
        )

        detail_form = QFormLayout()
        detail_form.addRow("Metal Type:", self.detail_metal_value)
        detail_form.addRow("Coordination Number:", self.detail_cn_value)
        detail_form.addRow("OMS Classification:", self.detail_status_value)
        detail_form.addRow("OMS Location:", self.detail_location_value)
        detail_form.addRow(
            "Missing Position(s):", self.detail_missing_value
        )
        detail_form.addRow("Widest Empty Cone:", self.detail_gap_value)

        detail_group = QGroupBox("Selected Site - Local Environment")
        detail_layout = QVBoxLayout()
        detail_layout.addLayout(detail_form)

        self.neighbor_table = QTableWidget()
        self.neighbor_table.setColumnCount(len(NEIGHBOR_COLUMNS))
        self.neighbor_table.setHorizontalHeaderLabels(NEIGHBOR_COLUMNS)
        self.neighbor_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.neighbor_table.setAlternatingRowColors(True)
        install_table_tools(
            self.neighbor_table, export_name="oms_site_neighbours.csv"
        )
        detail_layout.addWidget(self.neighbor_table)
        detail_group.setLayout(detail_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(controls_group)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.site_table)
        layout.addWidget(detail_group)

    def set_result(self, result) -> None:
        """Populate from an oms_detection.OMSDetectionResult."""
        summary = (
            f"{result.oms_count} open metal site(s) found "
            f"({len(result.sites)} metal centre(s) analyzed)."
        )

        unsupported = getattr(result, "unsupported_metals", ())

        if unsupported:
            summary += (
                " Not checked (outside OMS detection's supported set): "
                + ", ".join(unsupported)
                + "."
            )

        self.summary_label.setText(summary)

        self.site_table.setRowCount(len(result.sites))

        for row, site in enumerate(result.sites):
            values = (
                site.label,
                site.metal_symbol,
                str(site.coordination_number),
                CLASSIFICATION_LABELS[site.classification],
                f"{site.asymmetry_score:.3f}",
            )

            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                    if column > 0
                    else Qt.AlignmentFlag.AlignLeft
                    | Qt.AlignmentFlag.AlignVCenter
                )

                if column == 3:
                    item.setForeground(
                        _qcolor(CLASSIFICATION_COLORS[site.classification])
                    )

                if column == 0:
                    item.setData(SITE_INDEX_ROLE, row)

                self.site_table.setItem(row, column, item)

        self.site_table.resizeColumnsToContents()
        self.highlight_all_checkbox.setEnabled(result.oms_count > 0)

    def clear(self) -> None:
        self.summary_label.setText("No OMS detection run yet.")
        self.site_table.setRowCount(0)
        self.highlight_all_checkbox.setEnabled(False)
        self.highlight_all_checkbox.setChecked(True)
        self._clear_detail()

    def _clear_detail(self) -> None:
        for label in (
            self.detail_metal_value,
            self.detail_cn_value,
            self.detail_status_value,
            self.detail_location_value,
            self.detail_missing_value,
            self.detail_gap_value,
        ):
            label.setText(PLACEHOLDER)

        self.neighbor_table.setRowCount(0)

    def _on_selection_changed(self) -> None:
        selected = self.site_table.selectedItems()

        if not selected:
            self._clear_detail()
            self.site_selected.emit(None)
            return

        row = selected[0].row()
        id_item = self.site_table.item(row, 0)
        site_index = id_item.data(SITE_INDEX_ROLE) if id_item else None

        self.site_selected.emit(site_index)

    def show_site_detail(self, site) -> None:
        """Populate the inspection detail for one OMSSite - called back
        by MainWindow after it resolves site_selected's row index to
        the actual OMSSite object, since the panel itself does not
        keep the last analysis result around.
        """
        if site is None:
            self._clear_detail()
            return

        self.detail_metal_value.setText(site.metal_symbol)
        self.detail_cn_value.setText(str(site.coordination_number))
        self.detail_status_value.setText(
            CLASSIFICATION_LABELS[site.classification]
        )
        self.detail_status_value.setStyleSheet(
            f"color: {CLASSIFICATION_COLORS[site.classification]}; "
            "font-weight: 600;"
        )
        self.detail_location_value.setText(
            "({:.3f}, {:.3f}, {:.3f}) \u00c5".format(*site.position)
        )

        if site.missing_positions:
            directions = ", ".join(
                "({:.2f}, {:.2f}, {:.2f})".format(*mp.direction)
                for mp in site.missing_positions
            )
            self.detail_missing_value.setText(directions)
        else:
            self.detail_missing_value.setText(PLACEHOLDER)

        gap = getattr(site, "angular_gap", None)
        self.detail_gap_value.setText(
            f"{gap:.1f}\u00b0" if gap else PLACEHOLDER
        )

        neighbors = site.coordination_environment.neighbors
        self.neighbor_table.setRowCount(len(neighbors))

        for row, neighbor in enumerate(neighbors):
            for column, text in enumerate(
                (neighbor.label, neighbor.symbol, f"{neighbor.distance:.3f}")
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


def _qcolor(hex_color: str):
    from PySide6.QtGui import QColor

    return QColor(hex_color)
