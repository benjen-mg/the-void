"""StructureTree selection sync: clicking a row must notify listeners
(atom_selected), but a programmatic select_site_index() call - the
3D-view-drove-this-change direction - must not, or the two widgets
would ping-pong forever.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.ui.widgets.structure_tree import StructureTree  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


class FakeSpecies:
    def __init__(self, symbol):
        self.symbol = symbol


class FakeSite:
    def __init__(self, symbol, frac_coords, occupancy=1.0):
        self.species = {FakeSpecies(symbol): occupancy}
        self.frac_coords = frac_coords


class FakeStructure:
    def __init__(self, sites):
        self.sites = sites


@pytest.fixture
def populated_tree(qt_app):
    tree = StructureTree()
    structure = FakeStructure(
        [
            FakeSite("Si", [0.0, 0.0, 0.0]),
            FakeSite("O", [0.25, 0.25, 0.25]),
            FakeSite("O", [0.75, 0.75, 0.75]),
        ]
    )
    tree.populate(structure)

    return tree


def test_populate_labels_rows_per_element(populated_tree):
    labels = [
        populated_tree.tree.topLevelItem(i).text(0) for i in range(3)
    ]

    assert labels == ["Si1", "O1", "O2"]


def test_clicking_a_row_emits_atom_selected(populated_tree):
    received = []
    populated_tree.atom_selected.connect(received.append)

    item = populated_tree.tree.topLevelItem(1)
    populated_tree.tree.setCurrentItem(item)

    assert received == [1]


def test_select_site_index_selects_the_right_row(populated_tree):
    assert populated_tree.select_site_index(2) is True

    assert populated_tree.tree.currentItem() is (
        populated_tree.tree.topLevelItem(2)
    )


def test_select_site_index_does_not_emit_atom_selected(populated_tree):
    """The direction this method exists for: the 3D view already knows
    what was selected, so telling it again would be a no-op at best.
    """
    received = []
    populated_tree.atom_selected.connect(received.append)

    populated_tree.select_site_index(1)

    assert received == []


def test_select_site_index_returns_false_for_an_unknown_index(
    populated_tree,
):
    assert populated_tree.select_site_index(999) is False


def test_clear_selection_leaves_nothing_current(populated_tree):
    populated_tree.select_site_index(0)

    populated_tree.clear_selection()

    assert populated_tree.tree.currentItem() is None


def test_clear_selection_does_not_emit_atom_selected(populated_tree):
    populated_tree.select_site_index(0)

    received = []
    populated_tree.atom_selected.connect(received.append)

    populated_tree.clear_selection()

    assert received == []


def test_populate_clears_any_previous_rows(populated_tree):
    smaller_structure = FakeStructure([FakeSite("C", [0.5, 0.5, 0.5])])

    populated_tree.populate(smaller_structure)

    assert populated_tree.tree.topLevelItemCount() == 1
    assert populated_tree.select_site_index(2) is False  # gone now
