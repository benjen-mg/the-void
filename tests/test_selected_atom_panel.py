"""SelectedAtomPanel: display formatting logic, exercised through the
real Qt widget."""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.ui.widgets.selected_atom_panel import (  # noqa: E402
    NOT_VISIBLE,
    PLACEHOLDER,
    SelectedAtomPanel,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_starts_with_placeholders(qt_app):
    panel = SelectedAtomPanel()

    assert panel.label_value.text() == PLACEHOLDER
    assert panel.element_value.text() == PLACEHOLDER
    assert panel.occupancy_value.text() == PLACEHOLDER
    assert panel.fractional_value.text() == PLACEHOLDER
    assert panel.cartesian_value.text() == PLACEHOLDER


def test_set_atom_populates_every_field(qt_app):
    panel = SelectedAtomPanel()

    panel.set_atom(
        label="Si1",
        element="Si",
        occupancy=1.0,
        fractional_coordinates=(0.1, 0.2, 0.3),
        cartesian_coordinates=(1.5, 2.5, 3.5),
    )

    assert panel.label_value.text() == "Si1"
    assert panel.element_value.text() == "Si"
    assert panel.occupancy_value.text() == "1.000"
    assert panel.fractional_value.text() == "(0.10000, 0.20000, 0.30000)"
    assert panel.cartesian_value.text() == "(1.5000, 2.5000, 3.5000)"


def test_partial_occupancy_is_formatted(qt_app):
    panel = SelectedAtomPanel()

    panel.set_atom(
        label="O2",
        element="O",
        occupancy=0.5,
        fractional_coordinates=(0.0, 0.0, 0.0),
        cartesian_coordinates=(0.0, 0.0, 0.0),
    )

    assert panel.occupancy_value.text() == "0.500"


def test_missing_cartesian_coordinates_shows_a_clear_note(qt_app):
    """None (not a number, not a placeholder dash) means the tree
    selected a site that isn't part of the current 3D view - must not
    be confused with an actual (0, 0, 0) position.
    """
    panel = SelectedAtomPanel()

    panel.set_atom(
        label="H3",
        element="H",
        occupancy=1.0,
        fractional_coordinates=(0.2, 0.2, 0.2),
        cartesian_coordinates=None,
    )

    assert panel.cartesian_value.text() == NOT_VISIBLE


def test_clear_resets_every_field(qt_app):
    panel = SelectedAtomPanel()

    panel.set_atom(
        label="C1",
        element="C",
        occupancy=1.0,
        fractional_coordinates=(0.5, 0.5, 0.5),
        cartesian_coordinates=(2.0, 2.0, 2.0),
    )

    panel.clear()

    assert panel.label_value.text() == PLACEHOLDER
    assert panel.cartesian_value.text() == PLACEHOLDER


def test_selecting_a_second_atom_replaces_the_first(qt_app):
    panel = SelectedAtomPanel()

    panel.set_atom(
        label="Na1",
        element="Na",
        occupancy=1.0,
        fractional_coordinates=(0.0, 0.0, 0.0),
        cartesian_coordinates=(0.0, 0.0, 0.0),
    )
    panel.set_atom(
        label="Cl1",
        element="Cl",
        occupancy=1.0,
        fractional_coordinates=(0.5, 0.5, 0.5),
        cartesian_coordinates=(3.0, 3.0, 3.0),
    )

    assert panel.label_value.text() == "Cl1"
    assert panel.element_value.text() == "Cl"
