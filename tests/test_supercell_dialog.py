"""SupercellDialog: the custom a x b x c tiling entry, exercised
through the real Qt widget.

The preset menu covers 1x1x1 / 2x2x2 / 3x3x3; this dialog is the only
route to anything else, including the asymmetric tilings (4x1x2) that
are the reason it exists.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialogButtonBox,
)

from thevoid_core.ui.dialogs.supercell_dialog import (  # noqa: E402
    SupercellDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_dialog_starts_at_the_current_dimensions(qt_app):
    dialog = SupercellDialog((2, 3, 4))

    assert dialog.dimensions() == (2, 3, 4)


def test_asymmetric_dimensions_round_trip(qt_app):
    """The whole point of the dialog: the presets are all cubic, so an
    asymmetric tiling must survive unchanged.
    """
    dialog = SupercellDialog((1, 1, 1))

    dialog.a_box.setValue(4)
    dialog.b_box.setValue(1)
    dialog.c_box.setValue(2)

    assert dialog.dimensions() == (4, 1, 2)


def test_values_are_clamped_to_the_allowed_range(qt_app):
    dialog = SupercellDialog((1, 1, 1))

    dialog.a_box.setValue(SupercellDialog.MAX_DIM + 5)
    dialog.b_box.setValue(SupercellDialog.MIN_DIM - 5)

    assert dialog.dimensions()[0] == SupercellDialog.MAX_DIM
    assert dialog.dimensions()[1] == SupercellDialog.MIN_DIM


def test_zero_repeats_are_impossible(qt_app):
    """A 0 along any axis would mean an empty cell, so the spinbox
    floor has to be 1, not 0.
    """
    assert SupercellDialog.MIN_DIM == 1

    dialog = SupercellDialog((1, 1, 1))
    dialog.c_box.setValue(0)

    assert dialog.dimensions()[2] == 1


def test_ok_accepts_the_dialog(qt_app):
    dialog = SupercellDialog((2, 2, 2))

    button_box = dialog.findChild(QDialogButtonBox)
    button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    assert dialog.result() == SupercellDialog.DialogCode.Accepted


def test_cancel_rejects_the_dialog(qt_app):
    dialog = SupercellDialog((2, 2, 2))

    button_box = dialog.findChild(QDialogButtonBox)
    button_box.button(QDialogButtonBox.StandardButton.Cancel).click()

    assert dialog.result() == SupercellDialog.DialogCode.Rejected


def test_the_upper_bound_keeps_the_atom_count_survivable(qt_app):
    """10 x 10 x 10 is already a thousand copies of the cell. The cap
    exists so a typo cannot ask the renderer for a million atoms.
    """
    assert SupercellDialog.MAX_DIM <= 10
