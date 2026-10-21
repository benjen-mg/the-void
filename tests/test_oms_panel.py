"""OMSPanel: results table, inspection detail, and controls, through
the real Qt widget."""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.coordination import (  # noqa: E402
    CoordinationEnvironment,
    Neighbor,
)
from thevoid_core.services.oms_detection import (  # noqa: E402
    MissingPosition,
    OMSClassification,
    OMSDetectionResult,
    OMSSite,
)
from thevoid_core.ui.widgets.oms_panel import OMSPanel, PLACEHOLDER  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _env():
    return CoordinationEnvironment(
        center_site_index=0, center_label="Cu1", center_symbol="Cu",
        neighbors=(
            Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),
            Neighbor(2, "O", "O2", 2.0, (0, 0, 0)),
        ),
    )


def _open_site(index=0):
    return OMSSite(
        site_index=index, label="Cu1", metal_symbol="Cu",
        position=(1.0, 2.0, 3.0), coordination_environment=_env(),
        classification=OMSClassification.OPEN, asymmetry_score=0.2,
        missing_positions=(
            MissingPosition(direction=(0.0, 0.0, -1.0), confidence=0.2),
        ),
    )


def _saturated_site(index=1):
    return OMSSite(
        site_index=index, label="Zn1", metal_symbol="Zn",
        position=(4.0, 5.0, 6.0), coordination_environment=_env(),
        classification=OMSClassification.SATURATED, asymmetry_score=0.0,
        missing_positions=(),
    )


def test_starts_empty(qt_app):
    panel = OMSPanel()

    assert panel.site_table.rowCount() == 0
    assert "No OMS detection" in panel.summary_label.text()
    assert not panel.highlight_all_checkbox.isEnabled()


def test_set_result_populates_the_summary(qt_app):
    panel = OMSPanel()
    result = OMSDetectionResult(sites=(_open_site(), _saturated_site()))

    panel.set_result(result)

    assert "1 open metal site" in panel.summary_label.text()
    assert "2 metal centre" in panel.summary_label.text()


def test_set_result_populates_the_table(qt_app):
    panel = OMSPanel()
    result = OMSDetectionResult(sites=(_open_site(), _saturated_site()))

    panel.set_result(result)

    assert panel.site_table.rowCount() == 2
    assert panel.site_table.item(0, 0).text() == "Cu1"
    assert panel.site_table.item(0, 1).text() == "Cu"
    assert panel.site_table.item(0, 2).text() == "2"
    assert panel.site_table.item(0, 3).text() == "Open"
    assert panel.site_table.item(1, 3).text() == "Saturated"


def test_highlight_toggle_enabled_only_when_oms_found(qt_app):
    panel = OMSPanel()

    panel.set_result(OMSDetectionResult(sites=(_saturated_site(),)))
    assert not panel.highlight_all_checkbox.isEnabled()

    panel.set_result(OMSDetectionResult(sites=(_open_site(),)))
    assert panel.highlight_all_checkbox.isEnabled()


def test_selecting_a_row_emits_its_index(qt_app):
    panel = OMSPanel()
    result = OMSDetectionResult(sites=(_open_site(), _saturated_site()))
    panel.set_result(result)

    received = []
    panel.site_selected.connect(received.append)

    panel.site_table.selectRow(1)

    assert received == [1]


def test_show_site_detail_populates_every_field(qt_app):
    panel = OMSPanel()

    panel.show_site_detail(_open_site())

    assert panel.detail_metal_value.text() == "Cu"
    assert panel.detail_cn_value.text() == "2"
    assert panel.detail_status_value.text() == "Open"
    assert "1.000" in panel.detail_location_value.text()
    assert "0.00" in panel.detail_missing_value.text()
    assert panel.neighbor_table.rowCount() == 2


def test_show_site_detail_for_saturated_site_has_no_missing_positions(qt_app):
    panel = OMSPanel()

    panel.show_site_detail(_saturated_site())

    assert panel.detail_missing_value.text() == PLACEHOLDER


def test_show_site_detail_with_none_clears_it(qt_app):
    panel = OMSPanel()
    panel.show_site_detail(_open_site())

    panel.show_site_detail(None)

    assert panel.detail_metal_value.text() == PLACEHOLDER
    assert panel.neighbor_table.rowCount() == 0


def test_run_button_emits_run_requested(qt_app):
    panel = OMSPanel()

    received = []
    panel.run_requested.connect(lambda: received.append(True))

    panel.run_button.click()

    assert received == [True]


def test_highlight_toggle_emits_its_state(qt_app):
    panel = OMSPanel()
    panel.set_result(OMSDetectionResult(sites=(_open_site(),)))

    received = []
    panel.highlight_all_toggled.connect(received.append)

    panel.highlight_all_checkbox.setChecked(False)

    assert received == [False]


def test_clear_resets_everything(qt_app):
    panel = OMSPanel()
    panel.set_result(OMSDetectionResult(sites=(_open_site(),)))
    panel.show_site_detail(_open_site())

    panel.clear()

    assert panel.site_table.rowCount() == 0
    assert "No OMS detection" in panel.summary_label.text()
    assert panel.detail_metal_value.text() == PLACEHOLDER
    assert not panel.highlight_all_checkbox.isEnabled()
    assert panel.highlight_all_checkbox.isChecked()


def test_summary_names_metals_that_were_not_checked(qt_app):
    """A result that skipped unsupported metals must say so, rather
    than reading like a structure with no metals at all.
    """
    panel = OMSPanel()
    panel.set_result(
        OMSDetectionResult(sites=(), unsupported_metals=("Na", "Sr"))
    )

    text = panel.summary_label.text()
    assert "0 metal centre" in text
    assert "Not checked" in text
    assert "Na, Sr" in text


def test_summary_has_no_not_checked_note_when_everything_was_covered(qt_app):
    panel = OMSPanel()
    panel.set_result(OMSDetectionResult(sites=(_open_site(),)))

    assert "Not checked" not in panel.summary_label.text()
