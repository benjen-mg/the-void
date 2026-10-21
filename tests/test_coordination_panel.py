"""CoordinationEnvironmentPanel: display formatting, through the real
Qt widget.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.coordination import (  # noqa: E402
    CoordinationEnvironment,
    Neighbor,
)
from thevoid_core.ui.widgets.coordination_panel import (  # noqa: E402
    PLACEHOLDER,
    CoordinationEnvironmentPanel,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def test_starts_with_placeholders(qt_app):
    panel = CoordinationEnvironmentPanel()

    assert panel.center_value.text() == PLACEHOLDER
    assert panel.coordination_number_value.text() == PLACEHOLDER
    assert panel.neighbor_table.rowCount() == 0


def test_set_environment_populates_the_summary(qt_app):
    panel = CoordinationEnvironmentPanel()

    env = CoordinationEnvironment(
        center_site_index=0, center_label="Zn1", center_symbol="Zn",
        neighbors=(
            Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),
            Neighbor(2, "O", "O2", 2.2, (0, 0, 0)),
        ),
    )
    panel.set_environment(env)

    assert panel.center_value.text() == "Zn1 (Zn)"
    assert panel.coordination_number_value.text() == "2"
    assert panel.average_value.text() == "2.100 \u00c5"
    assert panel.min_value.text() == "2.000 \u00c5"
    assert panel.max_value.text() == "2.200 \u00c5"


def test_set_environment_populates_the_neighbor_table(qt_app):
    panel = CoordinationEnvironmentPanel()

    env = CoordinationEnvironment(
        center_site_index=0, center_label="Zn1", center_symbol="Zn",
        neighbors=(
            Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),
            Neighbor(2, "N", "N1", 2.2, (0, 0, 0)),
        ),
    )
    panel.set_environment(env)

    assert panel.neighbor_table.rowCount() == 2
    assert panel.neighbor_table.item(0, 0).text() == "O1"
    assert panel.neighbor_table.item(0, 1).text() == "O"
    assert panel.neighbor_table.item(0, 2).text() == "2.000"
    assert panel.neighbor_table.item(1, 0).text() == "N1"


def test_zero_coordination_shows_placeholders_for_bond_stats(qt_app):
    panel = CoordinationEnvironmentPanel()

    env = CoordinationEnvironment(
        center_site_index=0, center_label="X1", center_symbol="X",
        neighbors=(),
    )
    panel.set_environment(env)

    assert panel.coordination_number_value.text() == "0"
    assert panel.average_value.text() == PLACEHOLDER
    assert panel.min_value.text() == PLACEHOLDER
    assert panel.max_value.text() == PLACEHOLDER
    assert panel.neighbor_table.rowCount() == 0


def test_clear_resets_everything(qt_app):
    panel = CoordinationEnvironmentPanel()

    env = CoordinationEnvironment(
        center_site_index=0, center_label="Zn1", center_symbol="Zn",
        neighbors=(Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),),
    )
    panel.set_environment(env)

    panel.clear()

    assert panel.center_value.text() == PLACEHOLDER
    assert panel.coordination_number_value.text() == PLACEHOLDER
    assert panel.neighbor_table.rowCount() == 0


def test_selecting_a_new_environment_replaces_the_old_one(qt_app):
    panel = CoordinationEnvironmentPanel()

    panel.set_environment(
        CoordinationEnvironment(
            center_site_index=0, center_label="Zn1", center_symbol="Zn",
            neighbors=(Neighbor(1, "O", "O1", 2.0, (0, 0, 0)),),
        )
    )
    panel.set_environment(
        CoordinationEnvironment(
            center_site_index=5, center_label="Cu1", center_symbol="Cu",
            neighbors=(
                Neighbor(6, "N", "N1", 1.9, (0, 0, 0)),
                Neighbor(7, "N", "N2", 1.95, (0, 0, 0)),
            ),
        )
    )

    assert panel.center_value.text() == "Cu1 (Cu)"
    assert panel.neighbor_table.rowCount() == 2
