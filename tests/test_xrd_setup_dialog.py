"""XRDSetupDialog: the 2-theta range and radiation source asked for
before the pattern is simulated.

Both inputs change the physics of the calculated pattern rather than
what is displayed, so getting them wrong produces a wrong pattern
rather than an ugly one - which is why the invalid-range guard matters
enough to test.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QDialogButtonBox,
)

from thevoid_core.services.xrd.powder_xrd_simulator import (  # noqa: E402
    DEFAULT_TWO_THETA_RANGE,
    DEFAULT_WAVELENGTH,
    RADIATION_SOURCES,
)
from thevoid_core.ui.dialogs.xrd_setup_dialog import (  # noqa: E402
    XRDSetupDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_dialog_starts_at_the_defaults(qt_app):
    dialog = XRDSetupDialog()

    assert dialog.two_theta_range() == DEFAULT_TWO_THETA_RANGE
    assert dialog.wavelength_key() == DEFAULT_WAVELENGTH


def test_dialog_starts_at_the_values_it_is_given(qt_app):
    dialog = XRDSetupDialog(
        two_theta_range=(10.0, 60.0), wavelength_key="MoKa"
    )

    assert dialog.two_theta_range() == (10.0, 60.0)
    assert dialog.wavelength_key() == "MoKa"


def test_every_radiation_source_is_offered(qt_app):
    dialog = XRDSetupDialog()

    offered = {
        dialog.source_box.itemData(i)
        for i in range(dialog.source_box.count())
    }

    assert offered == set(RADIATION_SOURCES.values())


def test_an_unknown_wavelength_key_falls_back_to_the_first_source(qt_app):
    """A stale key from a previous session must not leave the combo box
    on a negative index with no data at all.
    """
    dialog = XRDSetupDialog(wavelength_key="NotARealSource")

    assert dialog.wavelength_key() in RADIATION_SOURCES.values()


def test_edited_range_round_trips(qt_app):
    dialog = XRDSetupDialog()

    dialog.min_box.setValue(7.5)
    dialog.max_box.setValue(45.0)

    assert dialog.two_theta_range() == (7.5, 45.0)


def test_ok_is_refused_when_min_is_not_below_max(qt_app):
    """An inverted or empty range would hand the simulator a window
    containing no reflections at all. The dialog must not close.
    """
    dialog = XRDSetupDialog()

    dialog.min_box.setValue(60.0)
    dialog.max_box.setValue(30.0)

    dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    assert dialog.result() != XRDSetupDialog.DialogCode.Accepted


def test_an_equal_min_and_max_is_also_refused(qt_app):
    dialog = XRDSetupDialog()

    dialog.min_box.setValue(30.0)
    dialog.max_box.setValue(30.0)

    dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    assert dialog.result() != XRDSetupDialog.DialogCode.Accepted


def test_the_invalid_range_is_marked_on_both_boxes(qt_app):
    dialog = XRDSetupDialog()

    dialog.min_box.setValue(60.0)
    dialog.max_box.setValue(30.0)
    dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    assert dialog.min_box.styleSheet()
    assert dialog.max_box.styleSheet()


def test_correcting_the_range_clears_the_marking_and_accepts(qt_app):
    dialog = XRDSetupDialog()

    dialog.min_box.setValue(60.0)
    dialog.max_box.setValue(30.0)
    dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    dialog.max_box.setValue(90.0)
    dialog.button_box.button(QDialogButtonBox.StandardButton.Ok).click()

    assert dialog.result() == XRDSetupDialog.DialogCode.Accepted
    assert dialog.min_box.styleSheet() == ""
    assert dialog.max_box.styleSheet() == ""


def test_cancel_rejects_the_dialog(qt_app):
    dialog = XRDSetupDialog()

    dialog.button_box.button(
        QDialogButtonBox.StandardButton.Cancel
    ).click()

    assert dialog.result() == XRDSetupDialog.DialogCode.Rejected
