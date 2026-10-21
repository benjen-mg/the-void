from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.services.bonds.bond_model import site_occupancy
from thevoid_core.services.site_labels import compute_site_labels

PARTIAL_OCCUPANCY_BRUSH = QBrush(QColor("#b3541e"))

#: Column holding the site index, stored via Qt.ItemDataRole.UserRole -
#: not a visible column, just where each row's underlying site index
#: lives so a row can be mapped back to a site and vice versa.
SITE_INDEX_COLUMN = 0


class StructureTree(QWidget):
    """Flat list of the sites in the loaded structure.

    Selecting a row here selects the same atom in the 3D view, and
    selecting an atom in the 3D view selects the corresponding row
    here - see select_site_index() for the tree-following-3D direction
    and the atom_selected signal for the 3D-following-tree direction.
    """

    #: Emitted with a site index when the user selects a row by
    #: clicking it - not when a row is selected programmatically via
    #: select_site_index(), which would otherwise loop straight back
    #: to whatever just selected this row in the first place.
    atom_selected = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(
            [
                "Label",
                "Element",
                "Occ.",
                "Fractional Coordinates",
            ]
        )

        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionBehavior(
            QTreeWidget.SelectionBehavior.SelectRows
        )
        self.tree.setSelectionMode(
            QTreeWidget.SelectionMode.SingleSelection
        )

        self.tree.itemSelectionChanged.connect(self._on_selection_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree)

        self._items_by_site_index: dict[int, QTreeWidgetItem] = {}

        # Set while a selection change is being applied programmatically
        # (from select_site_index()) so _on_selection_changed knows not
        # to treat it as a user click and re-emit atom_selected.
        self._suppress_selection_signal = False

    def clear(self) -> None:
        self.tree.clear()
        self._items_by_site_index = {}

    def populate(self, structure) -> None:
        self.tree.clear()
        self._items_by_site_index = {}

        labels = compute_site_labels(structure)

        items = []

        for site_index, (site, label) in enumerate(
            zip(structure.sites, labels)
        ):
            species = list(site.species.keys())
            element = species[0].symbol if species else "?"

            occupancy = site_occupancy(site)

            x, y, z = site.frac_coords

            item = QTreeWidgetItem(
                [
                    label,
                    element,
                    f"{occupancy:.3f}",
                    f"({x:.5f}, {y:.5f}, {z:.5f})",
                ]
            )
            item.setData(
                SITE_INDEX_COLUMN, Qt.ItemDataRole.UserRole, site_index
            )

            # Partially occupied sites are where spurious short contacts
            # come from, so make them findable at a glance.
            if occupancy < 0.99:
                for column in range(self.tree.columnCount()):
                    item.setForeground(column, PARTIAL_OCCUPANCY_BRUSH)

                item.setToolTip(
                    2,
                    "Partially occupied. Usually one of several "
                    "alternative positions for the same atom.",
                )

            items.append(item)
            self._items_by_site_index[site_index] = item

        self.tree.addTopLevelItems(items)

        for column in range(self.tree.columnCount()):
            self.tree.resizeColumnToContents(column)

    def select_site_index(self, site_index: int) -> bool:
        """Select and scroll to the row for a site, without emitting
        atom_selected - this is the 3D-view-drove-the-change direction,
        so telling the 3D view about its own selection again would be
        pointless at best and a feedback loop at worst.

        Returns False (and does nothing) if no row exists for that
        site index - the structure tree is empty, or hasn't been
        populated for the structure this index belongs to.
        """
        item = self._items_by_site_index.get(site_index)

        if item is None:
            return False

        self._suppress_selection_signal = True

        try:
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)
        finally:
            self._suppress_selection_signal = False

        return True

    def clear_selection(self) -> None:
        self._suppress_selection_signal = True

        try:
            self.tree.clearSelection()
            self.tree.setCurrentItem(None)
        finally:
            self._suppress_selection_signal = False

    def _on_selection_changed(self) -> None:
        if self._suppress_selection_signal:
            return

        selected = self.tree.selectedItems()

        if not selected:
            return

        site_index = selected[0].data(
            SITE_INDEX_COLUMN, Qt.ItemDataRole.UserRole
        )

        if site_index is not None:
            self.atom_selected.emit(site_index)
