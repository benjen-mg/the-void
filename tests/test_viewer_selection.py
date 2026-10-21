"""CrystalViewer atom selection, exercised end to end except for the
literal mouse click - _on_selection_pick is called directly with a
picked point, the same argument pyvista's picking machinery would hand
it (see test_viewer_measurement.py for the same approach and why it's
the right substitute for a live click in a sandbox with no GPU).
"""

import os

import pytest

pv = pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.ui.widgets.viewer_widget import (  # noqa: E402
    CrystalViewer,
    DisplayStyle,
)


class FakeAtom:
    def __init__(self, symbol, position, site_index, image=(0, 0, 0)):
        self.symbol = symbol
        self.position = tuple(position)
        self.site_index = site_index
        self.image = image


class FakeModel:
    def __init__(self, atoms):
        self.atoms = atoms


@pytest.fixture
def viewer_with_atoms():
    pv.OFF_SCREEN = True

    viewer = CrystalViewer.__new__(CrystalViewer)
    viewer.picking_enabled = True
    viewer.plotter = pv.Plotter(off_screen=True, window_size=(200, 200))
    viewer.display_style = DisplayStyle.BALL_AND_STICK
    viewer.radius_provider = ElementRadiusProvider()
    viewer._selected_atom_index = None
    viewer._selection_actor = None
    viewer.on_atom_selected = None

    viewer._current_model = FakeModel(
        [
            FakeAtom("C", (0.0, 0.0, 0.0), site_index=0),
            FakeAtom("O", (1.5, 0.0, 0.0), site_index=1),
            # Two images of the same site (a supercell tiling it) -
            # index 2 is the reference cell, index 3 an image.
            FakeAtom("Na", (5.0, 5.0, 5.0), site_index=2, image=(0, 0, 0)),
            FakeAtom("Na", (15.0, 5.0, 5.0), site_index=2, image=(1, 0, 0)),
        ]
    )

    yield viewer

    viewer.plotter.close()


def test_clicking_an_atom_selects_it(viewer_with_atoms):
    viewer = viewer_with_atoms

    viewer._on_selection_pick((0.05, 0.0, 0.0))  # near atom 0 (C)

    assert viewer._selected_atom_index == 0


def test_selecting_invokes_on_atom_selected(viewer_with_atoms):
    viewer = viewer_with_atoms
    received = []
    viewer.on_atom_selected = received.append

    viewer._on_selection_pick((1.4, 0.05, 0.0))  # near atom 1 (O)

    assert len(received) == 1
    assert received[0].symbol == "O"
    assert received[0].site_index == 1


def test_only_one_atom_selected_at_a_time(viewer_with_atoms):
    viewer = viewer_with_atoms

    viewer._on_selection_pick((0.05, 0.0, 0.0))  # atom 0
    first_actor_count = len(viewer.plotter.renderer.actors)

    viewer._on_selection_pick((1.4, 0.0, 0.0))  # atom 1
    second_actor_count = len(viewer.plotter.renderer.actors)

    # Selecting a second atom must not add a second highlight actor -
    # add_mesh(..., name=...) replaces the one with that name.
    assert viewer._selected_atom_index == 1
    assert second_actor_count == first_actor_count


def test_a_click_far_from_any_atom_does_not_change_the_selection(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms

    viewer._on_selection_pick((0.05, 0.0, 0.0))  # select atom 0
    viewer._on_selection_pick((50.0, 50.0, 50.0))  # miss

    assert viewer._selected_atom_index == 0


def test_a_click_far_from_any_atom_does_not_fire_the_callback(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms
    received = []
    viewer.on_atom_selected = received.append

    viewer._on_selection_pick((50.0, 50.0, 50.0))

    assert received == []


def test_clear_selection_removes_the_highlight(viewer_with_atoms):
    viewer = viewer_with_atoms

    viewer._on_selection_pick((0.05, 0.0, 0.0))
    before = len(viewer.plotter.renderer.actors)

    viewer.clear_selection()

    assert viewer._selected_atom_index is None
    assert len(viewer.plotter.renderer.actors) == before - 1


def test_select_atom_by_site_index_finds_the_matching_atom(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms

    atom = viewer.select_atom_by_site_index(1)

    assert atom is not None
    assert atom.symbol == "O"
    assert viewer._selected_atom_index == 1


def test_select_atom_by_site_index_prefers_the_reference_cell_image(
    viewer_with_atoms,
):
    """Site 2 (Na) has two drawn copies: index 2 at image (0,0,0) and
    index 3 at image (1,0,0). The reference-cell copy must win, for a
    predictable, stable choice.
    """
    viewer = viewer_with_atoms

    atom = viewer.select_atom_by_site_index(2)

    assert atom.image == (0, 0, 0)
    assert viewer._selected_atom_index == 2


def test_select_atom_by_site_index_does_not_invoke_the_callback(
    viewer_with_atoms,
):
    """The tree-click direction: the caller already knows what it
    selected, so echoing back would be redundant at best.
    """
    viewer = viewer_with_atoms
    received = []
    viewer.on_atom_selected = received.append

    viewer.select_atom_by_site_index(0)

    assert received == []


def test_select_atom_by_site_index_returns_none_when_not_in_view(
    viewer_with_atoms,
):
    viewer = viewer_with_atoms

    result = viewer.select_atom_by_site_index(999)

    assert result is None
    assert viewer._selected_atom_index is None


def test_select_atom_by_site_index_clears_a_stale_selection_when_missing(
    viewer_with_atoms,
):
    """If something was already selected and the tree asks for a site
    that isn't drawn, the stale highlight must not linger.
    """
    viewer = viewer_with_atoms

    viewer.select_atom_by_site_index(0)
    assert viewer._selected_atom_index == 0

    viewer.select_atom_by_site_index(999)

    assert viewer._selected_atom_index is None


def test_selection_survives_switching_atoms_of_different_elements(
    viewer_with_atoms,
):
    """The highlight radius is derived from the atom's own element and
    the current display style - must not crash or misbehave when
    switching between very differently sized atoms.
    """
    viewer = viewer_with_atoms

    viewer.select_atom_by_site_index(0)  # C
    viewer.select_atom_by_site_index(2)  # Na, different radius

    assert viewer._selected_atom_index == 2


def test_picking_with_no_model_loaded_does_nothing(viewer_with_atoms):
    viewer = viewer_with_atoms
    viewer._current_model = None

    viewer._on_selection_pick((0.0, 0.0, 0.0))

    assert viewer._selected_atom_index is None
