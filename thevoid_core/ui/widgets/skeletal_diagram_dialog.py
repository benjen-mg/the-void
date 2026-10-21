import io

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

#: Export formats PIL's Image.save handles directly by file extension.
#: SVG is not offered - the underlying diagram is RDKit's own
#: rasterised grid rendering (a PIL Image), not a vector drawing, so
#: there is no vector format to honestly offer here.
EXPORT_FILTER = "PNG Files (*.png);;PDF Files (*.pdf)"


class SkeletalDiagramDialog(QDialog):
    """Shows a rendered 2D skeletal diagram (a PIL Image, RDKit's own
    composed grid of fragment diagrams) with a Save As... button to
    export it as PNG or PDF.
    """

    def __init__(self, image, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("2D Skeletal Diagram")
        self.resize(700, 600)

        self.image = image

        self.image_label = QLabel()
        self.image_label.setPixmap(self._render_pixmap(image))
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        scroll_area = QScrollArea()
        scroll_area.setWidget(self.image_label)
        scroll_area.setWidgetResizable(False)

        self.save_button = QPushButton("Save As...")
        self.save_button.clicked.connect(self._on_save_clicked)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)

        button_row = QHBoxLayout()
        button_row.addStretch()
        button_row.addWidget(self.save_button)
        button_row.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll_area)
        layout.addLayout(button_row)

    @staticmethod
    def _render_pixmap(image) -> QPixmap:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        pixmap = QPixmap()
        pixmap.loadFromData(buffer.read())

        return pixmap

    def _on_save_clicked(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save 2D Skeletal Diagram",
            "skeletal_diagram.png",
            EXPORT_FILTER,
        )

        if not filename:
            return

        try:
            image = self.image
            if filename.lower().endswith(".pdf") and image.mode != "RGB":
                # PIL's PDF writer requires an RGB (not RGBA) image.
                image = image.convert("RGB")
            image.save(filename)
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(
                self,
                "Export failed",
                f"The diagram could not be saved.\n\nFile: {filename}\n"
                f"Reason: {error}",
            )
