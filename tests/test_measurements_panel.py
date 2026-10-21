"""MeasurementsPanel: accumulating distance/angle records, exercised
through the real Qt widget.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.ui.widgets.measurements_panel import (  # noqa: E402
    NOT_APPLICABLE,
    MeasurementsPanel,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_starts_empty(qt_app):
    panel = MeasurementsPanel()

    assert panel.table.rowCount() == 0


def test_add_distance_appends_a_row(qt_app):
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="C1", element_a="C", label_b="O1", element_b="O",
        distance_angstrom=1.234,
    )

    assert panel.table.rowCount() == 1
    assert panel.table.item(0, 0).text() == "Distance"
    assert panel.table.item(0, 1).text() == "C1"
    assert panel.table.item(0, 2).text() == "C"
    assert panel.table.item(0, 3).text() == "O1"
    assert panel.table.item(0, 4).text() == "O"
    assert panel.table.item(0, 7).text() == "1.234 \u00c5"


def test_distance_row_shows_a_placeholder_for_atom_c(qt_app):
    """A distance only ever involves two atoms - the Atom C / Element C
    columns must not be left blank (which could look like missing
    data) or contain something misleading.
    """
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="C1", element_a="C", label_b="O1", element_b="O",
        distance_angstrom=1.234,
    )

    assert panel.table.item(0, 5).text() == NOT_APPLICABLE
    assert panel.table.item(0, 6).text() == NOT_APPLICABLE


def test_multiple_measurements_accumulate_as_separate_rows(qt_app):
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="C1", element_a="C", label_b="O1", element_b="O",
        distance_angstrom=1.2,
    )
    panel.add_distance(
        label_a="O1", element_a="O", label_b="O2", element_b="O",
        distance_angstrom=2.5,
    )
    panel.add_distance(
        label_a="Na1", element_a="Na", label_b="Cl1", element_b="Cl",
        distance_angstrom=2.8,
    )

    assert panel.table.rowCount() == 3
    assert panel.table.item(2, 1).text() == "Na1"


def test_add_angle_shows_all_three_atoms_in_their_own_columns(qt_app):
    """The point of this table: every atom in an angle measurement -
    not just the two that happen to fit the A/B shape a distance
    measurement uses - must be genuinely visible in its own column,
    not folded into the Value column's text.
    """
    panel = MeasurementsPanel()

    panel.add_angle(
        label_a="H1", element_a="H",
        label_vertex="O1", element_vertex="O",
        label_c="H2", element_c="H",
        angle_degrees=104.5,
    )

    assert panel.table.item(0, 0).text() == "Angle"
    assert panel.table.item(0, 1).text() == "H1"  # atom A
    assert panel.table.item(0, 2).text() == "H"  # element A
    assert panel.table.item(0, 3).text() == "O1"  # the vertex, as atom B
    assert panel.table.item(0, 4).text() == "O"  # element B
    assert panel.table.item(0, 5).text() == "H2"  # atom C
    assert panel.table.item(0, 6).text() == "H"  # element C
    assert panel.table.item(0, 7).text() == "104.5\u00b0"  # value alone


def test_clear_removes_every_row(qt_app):
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="C1", element_a="C", label_b="O1", element_b="O",
        distance_angstrom=1.2,
    )
    panel.add_distance(
        label_a="O1", element_a="O", label_b="O2", element_b="O",
        distance_angstrom=2.5,
    )

    panel.clear()

    assert panel.table.rowCount() == 0


def test_clear_removes_angle_rows_too(qt_app):
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="C1", element_a="C", label_b="O1", element_b="O",
        distance_angstrom=1.2,
    )
    panel.add_angle(
        label_a="H1", element_a="H", label_vertex="O1", element_vertex="O",
        label_c="H2", element_c="H", angle_degrees=104.5,
    )

    panel.clear()

    assert panel.table.rowCount() == 0


def test_distance_value_is_formatted_to_three_decimals(qt_app):
    panel = MeasurementsPanel()

    panel.add_distance(
        label_a="A", element_a="X", label_b="B", element_b="Y",
        distance_angstrom=1.23456789,
    )

    assert panel.table.item(0, 7).text() == "1.235 \u00c5"


def test_angle_value_is_formatted_to_one_decimal(qt_app):
    panel = MeasurementsPanel()

    panel.add_angle(
        label_a="A", element_a="X", label_vertex="B", element_vertex="Y",
        label_c="C", element_c="Z", angle_degrees=104.4999,
    )

    assert panel.table.item(0, 7).text() == "104.5\u00b0"
