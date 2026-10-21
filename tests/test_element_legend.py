"""ElementLegend and ColorCircle: the element-to-colour key.

The rebuild path is the part that matters. update_elements is called
again on every structure load and every colour change, so a row that is
removed from the layout but not actually destroyed shows up as a
duplicate entry or a stale colour - the sort of bug that only appears
on the second file somebody opens.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from thevoid_core.data.element_colors import (  # noqa: E402
    ElementColorProvider,
)
from thevoid_core.ui.widgets.element_legend import (  # noqa: E402
    ColorCircle,
    ElementLegend,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def legend(qt_app):
    widget = ElementLegend(ElementColorProvider())
    yield widget
    widget.deleteLater()


def _symbols_shown(legend):
    return [
        label.text()
        for row in legend._rows
        for label in row.findChildren(QLabel)
    ]


def test_a_new_legend_lists_nothing(legend):
    assert legend._rows == []


def test_every_element_gets_a_row(legend):
    legend.update_elements(["Zn", "O", "C"])

    assert len(legend._rows) == 3


def test_elements_are_listed_alphabetically(legend):
    legend.update_elements(["Zn", "C", "O", "H"])

    assert _symbols_shown(legend) == ["C", "H", "O", "Zn"]


def test_each_row_carries_the_element_colour(legend):
    provider = ElementColorProvider()
    legend.update_elements(["O"])

    circle = legend._rows[0].findChild(ColorCircle)
    expected = provider.get_color("O")

    assert circle.color.red() == int(expected[0] * 255)
    assert circle.color.green() == int(expected[1] * 255)
    assert circle.color.blue() == int(expected[2] * 255)


def test_rebuilding_replaces_rather_than_appends(legend):
    """The duplicate-row bug: opening a second structure must not leave
    the first structure's elements in the legend.
    """
    legend.update_elements(["Zn", "O", "C", "H"])
    legend.update_elements(["Fe", "S"])

    assert _symbols_shown(legend) == ["Fe", "S"]
    assert len(legend._rows) == 2


def test_clear_empties_the_legend(legend):
    legend.update_elements(["Zn", "O"])
    legend.clear()

    assert legend._rows == []
    assert _symbols_shown(legend) == []


def test_the_title_survives_every_rebuild(legend):
    """The title label and the trailing stretch are not rows; a rebuild
    that removed them would leave the legend unlabelled.
    """
    legend.update_elements(["Zn"])
    legend.clear()
    legend.update_elements(["O"])

    titles = [
        label.text()
        for label in legend.findChildren(QLabel)
        if label.text() == "Elements"
    ]

    assert titles == ["Elements"]


def test_rows_are_inserted_before_the_trailing_stretch(legend):
    """Appended after it, every row would be pushed to the bottom of
    the panel instead of stacking under the title.
    """
    legend.update_elements(["Zn", "O"])

    last_item = legend._layout.itemAt(legend._layout.count() - 1)

    assert last_item.widget() is None


def test_an_unknown_symbol_still_renders(legend):
    """The colour provider falls back rather than raising, so a legend
    for an exotic or mislabelled element must not take the window down.
    """
    legend.update_elements(["Uue"])

    assert len(legend._rows) == 1


def test_the_colour_circle_has_a_fixed_size(qt_app):
    circle = ColorCircle((1.0, 0.0, 0.0))

    assert circle.width() == ColorCircle.DIAMETER
    assert circle.height() == ColorCircle.DIAMETER
