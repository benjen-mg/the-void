from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class ColorCircle(QWidget):
    """A small filled circle showing an element colour."""

    DIAMETER = 16

    def __init__(self, color, parent=None) -> None:
        super().__init__(parent)

        self.color = QColor(
            int(color[0] * 255),
            int(color[1] * 255),
            int(color[2] * 255),
        )

        self.setFixedSize(self.DIAMETER, self.DIAMETER)

    def paintEvent(self, event):  # noqa: N802 - Qt naming
        painter = QPainter(self)

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(70, 70, 70), 1))
        painter.setBrush(self.color)

        painter.drawEllipse(1, 1, self.DIAMETER - 2, self.DIAMETER - 2)


class ElementLegend(QWidget):
    """Legend mapping element symbols to their render colours.

    Round 73: columns > 1 lays the entries out in a grid, for the
    compact key at the top of the Inspector's Structure tab, and
    title=False leaves out the "Elements" heading (the group box there
    has its own)."""

    def __init__(self, color_provider, parent=None, *, columns: int = 1,
                 title: bool = True) -> None:
        super().__init__(parent)

        self.color_provider = color_provider
        self.columns = max(1, int(columns))

        # Named _layout, not layout: assigning to self.layout shadows the
        # inherited QWidget.layout() method.
        self._layout = QVBoxLayout(self)

        if title:
            heading = QLabel("Elements")
            heading.setStyleSheet("font-weight: bold;")
            self._layout.addWidget(heading)
        else:
            self._layout.setContentsMargins(0, 0, 0, 0)

        self._grid = QGridLayout()
        self._grid.setHorizontalSpacing(14)
        self._grid.setVerticalSpacing(4)
        self._layout.addLayout(self._grid)
        self._layout.addStretch()

        #: Shown when there is nothing to list.
        self.empty_label = QLabel("No structure loaded.")
        self.empty_label.setStyleSheet("color: gray;")
        self._grid.addWidget(self.empty_label, 0, 0)

        self._rows: list[QWidget] = []

    def update_elements(self, elements) -> None:
        """elements: symbols, or {symbol: sites in the cell}, whose
        counts are then shown beside each symbol."""
        for row in self._rows:
            self._grid.removeWidget(row)
            row.setParent(None)
            row.deleteLater()

        self._rows.clear()
        counts = elements if isinstance(elements, dict) else None

        for index, symbol in enumerate(sorted(elements)):
            row = QWidget()

            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(5)

            row_layout.addWidget(
                ColorCircle(self.color_provider.get_color(symbol))
            )
            row_layout.addWidget(QLabel(symbol))
            if counts is not None:
                count = QLabel(str(counts[symbol]))
                count.setStyleSheet("color: gray;")
                row_layout.addWidget(count)
                row.setToolTip(f"{symbol}: {counts[symbol]} site(s) in the cell")
            row_layout.addStretch()

            self._grid.addWidget(row, index // self.columns, index % self.columns)

            self._rows.append(row)

        self.empty_label.setVisible(not self._rows)

        # This widget is a manually-positioned floating child of the
        # viewer (not managed by a parent layout that would resize it
        # automatically), so its actual on-screen size stays frozen at
        # whatever it was when last explicitly resized - typically its
        # tiny, empty-state size from construction - unless something
        # here tells it to catch up with its own new content. Confirmed
        # directly to be a real bug, not a hypothetical one: without
        # this, sizeHint() correctly grows to fit every row added, but
        # the widget's actual size() does not follow it, so everything
        # past whatever first sliver of content already fit renders
        # completely clipped and invisible.
        #
        # A plain adjustSize() call here is not enough on its own: Qt
        # defers a layout's own recalculation to the next event loop
        # iteration rather than updating it synchronously the moment a
        # widget is added or removed, so sizeHint() still reports the
        # *previous* content's size immediately afterward - confirmed
        # directly (querying it right after insertWidget() gives the
        # stale value; only after the pending layout request has
        # actually been processed does it reflect the new rows).
        # self._layout.activate() and updateGeometry() were both tried
        # first and neither forced it either - processEvents() is what
        # actually flushes the pending QEvent.LayoutRequest, confirmed
        # directly by checking sizeHint() before and after.
        app = QApplication.instance()
        if app is not None:
            app.processEvents()
        self.adjustSize()

    def clear(self) -> None:
        self.update_elements([])


def element_counts(structure) -> dict:
    """{symbol: number of sites} of a structure, by each site's first
    species - what the 3D view colours it as."""
    counts: dict = {}
    for site in structure.sites:
        symbol = next(iter(site.species)).symbol
        counts[symbol] = counts.get(symbol, 0) + 1
    return counts
