from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from thevoid_core.ui.widgets.element_legend import ColorCircle


class ColorCustomizationDialog(QDialog):
    """Lets the person change the render colour of any element
    currently present in the loaded structure, or reset them back to
    the default Jmol colours.

    Applies changes to color_provider immediately as each one is made
    (not held back for an OK/Cancel choice) and emits colors_changed
    so the caller can redraw the 3D view right away - the same
    "change it and see it happen" feel as every other display toggle
    in this application (display style, opacity, background), rather
    than a settings dialog you have to close before anything happens.
    """

    colors_changed = Signal()

    def __init__(self, color_provider, elements, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Customize Atom Colors")
        self.resize(320, 420)

        self.color_provider = color_provider
        self._rows: dict[str, QWidget] = {}
        self._swatches: dict[str, ColorCircle] = {}

        layout = QVBoxLayout(self)

        rows_container = QWidget()
        self._rows_layout = QVBoxLayout(rows_container)

        for symbol in sorted(elements):
            self._add_row(symbol)

        self._rows_layout.addStretch()

        scroll_area = QScrollArea()
        scroll_area.setWidget(rows_container)
        scroll_area.setWidgetResizable(True)
        layout.addWidget(scroll_area)

        self.reset_all_button = QPushButton("Reset All to Defaults")
        self.reset_all_button.setToolTip(
            "Restore every element below to its default Jmol colour."
        )
        self.reset_all_button.clicked.connect(self._on_reset_all_clicked)
        layout.addWidget(self.reset_all_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)

    def _add_row(self, symbol: str) -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        swatch = ColorCircle(self.color_provider.get_color(symbol))
        self._swatches[symbol] = swatch
        row_layout.addWidget(swatch)

        row_layout.addWidget(QLabel(symbol))
        row_layout.addStretch()

        change_button = QPushButton("Change...")
        change_button.setToolTip(f"Pick a new render colour for {symbol}.")
        change_button.clicked.connect(
            lambda: self._on_change_clicked(symbol)
        )
        row_layout.addWidget(change_button)

        reset_button = QPushButton("Reset")
        reset_button.setToolTip(f"Restore {symbol}'s default Jmol colour.")
        reset_button.clicked.connect(lambda: self._on_reset_clicked(symbol))
        row_layout.addWidget(reset_button)

        self._rows_layout.addWidget(row)
        self._rows[symbol] = row

    def _on_change_clicked(self, symbol: str) -> None:
        current = self.color_provider.get_color(symbol)
        initial = QColor(
            int(current[0] * 255),
            int(current[1] * 255),
            int(current[2] * 255),
        )

        chosen = QColorDialog.getColor(
            initial, self, f"Choose a color for {symbol}"
        )

        if not chosen.isValid():
            return

        rgb = (chosen.redF(), chosen.greenF(), chosen.blueF())
        self.color_provider.set_color(symbol, rgb)
        self._refresh_swatch(symbol)
        self.colors_changed.emit()

    def _on_reset_clicked(self, symbol: str) -> None:
        self.color_provider.reset_color(symbol)
        self._refresh_swatch(symbol)
        self.colors_changed.emit()

    def _on_reset_all_clicked(self) -> None:
        self.color_provider.reset_all_colors()

        for symbol in self._swatches:
            self._refresh_swatch(symbol)

        self.colors_changed.emit()

    def _refresh_swatch(self, symbol: str) -> None:
        old_swatch = self._swatches[symbol]
        row = self._rows[symbol]
        row_layout = row.layout()

        new_swatch = ColorCircle(self.color_provider.get_color(symbol))
        row_layout.replaceWidget(old_swatch, new_swatch)
        old_swatch.deleteLater()
        self._swatches[symbol] = new_swatch
