"""SkeletalDiagramDialog: displays a rendered PIL Image and offers a
save action, through the real Qt widget."""

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("PIL")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from thevoid_core.ui.widgets.skeletal_diagram_dialog import (  # noqa: E402
    SkeletalDiagramDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _image():
    return Image.new("RGB", (300, 200), color="white")


def test_dialog_shows_a_non_empty_pixmap(qt_app):
    dialog = SkeletalDiagramDialog(_image())

    pixmap = dialog.image_label.pixmap()

    assert pixmap is not None
    assert not pixmap.isNull()
    assert pixmap.width() > 0
    assert pixmap.height() > 0


def test_save_dialog_writes_the_image(qt_app, tmp_path, monkeypatch):
    dialog = SkeletalDiagramDialog(_image())
    out_path = tmp_path / "diagram.png"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )

    dialog._on_save_clicked()

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_cancelling_save_dialog_writes_nothing(qt_app, tmp_path, monkeypatch):
    dialog = SkeletalDiagramDialog(_image())

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )

    before = list(tmp_path.iterdir())
    dialog._on_save_clicked()
    after = list(tmp_path.iterdir())

    assert before == after


def test_save_supports_pdf_export(qt_app, tmp_path, monkeypatch):
    dialog = SkeletalDiagramDialog(_image())
    out_path = tmp_path / "diagram.pdf"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )

    dialog._on_save_clicked()

    assert out_path.exists()
    assert out_path.read_bytes().startswith(b"%PDF")


def test_save_supports_rgba_image_exported_as_pdf(qt_app, tmp_path, monkeypatch):
    """RDKit's own rendered images can come back as RGBA; PIL's PDF
    writer requires RGB, so this must be converted rather than fail -
    confirmed directly with an RGBA source image, not just RGB.
    """
    rgba_image = Image.new("RGBA", (300, 200), color=(255, 255, 255, 255))
    dialog = SkeletalDiagramDialog(rgba_image)
    out_path = tmp_path / "diagram.pdf"

    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(out_path), "")),
    )

    dialog._on_save_clicked()

    assert out_path.exists()


def test_close_button_accepts_the_dialog(qt_app):
    dialog = SkeletalDiagramDialog(_image())

    received = []
    dialog.accepted.connect(lambda: received.append(True))

    dialog.close_button.click()

    assert received == [True]
