"""BondingToleranceDialog: value round-trip and the Restore Defaults
button, exercised through the real Qt widget."""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.bonds.covalent_bond_detector import (  # noqa: E402
    CovalentBondDetector,
)
from thevoid_core.ui.dialogs.bonding_tolerance_dialog import (  # noqa: E402
    BondingToleranceDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_dialog_starts_at_the_given_value(qt_app):
    dialog = BondingToleranceDialog(0.55)

    assert dialog.tolerance() == pytest.approx(0.55)


def test_dialog_reflects_spinbox_edits(qt_app):
    dialog = BondingToleranceDialog(0.4)

    dialog.tolerance_box.setValue(0.75)

    assert dialog.tolerance() == pytest.approx(0.75)


def test_restore_defaults_resets_to_the_detector_default(qt_app):
    dialog = BondingToleranceDialog(0.9)

    from PySide6.QtWidgets import QDialogButtonBox

    button_box = dialog.findChild(QDialogButtonBox)
    restore_button = button_box.button(
        QDialogButtonBox.StandardButton.RestoreDefaults
    )
    restore_button.click()

    assert dialog.tolerance() == pytest.approx(CovalentBondDetector.TOLERANCE)


def test_value_is_clamped_to_the_allowed_range(qt_app):
    dialog = BondingToleranceDialog(0.4)

    dialog.tolerance_box.setValue(5.0)  # above MAX_TOLERANCE

    assert dialog.tolerance() <= BondingToleranceDialog.MAX_TOLERANCE
