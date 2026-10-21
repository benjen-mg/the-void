"""Regression tests for a real reported bug: after running MOF
analysis, nothing visible happened anywhere, and the result only
became reachable after minimizing and maximizing the window. Root
cause: the MOF Structure panel was tabified behind Measurements via
tabifyDockWidget(), which does not raise the newly grouped dock, and
nothing in the result handler ever raised it either - the content
updated in a tab the person wasn't looking at.

Fixed by consolidating from eight separate QDockWidgets down to three
(Structure Tree, Inspector, Results), each holding related panels as
tabs inside one dock, and having every handler that updates a panel's
content explicitly switch to and raise that panel's tab.
"""

import os

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

CIF = """\
data_test
_cell_length_a 10.0
_cell_length_b 10.0
_cell_length_c 10.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Zn1 Zn 0.25 0.50 0.50 1.0
Ob  O  0.35 0.50 0.50 1.0
Zn2 Zn 0.45 0.50 0.50 1.0
O1a O  0.16 0.50 0.50 1.0
C1  C  0.10 0.50 0.50 1.0
O1b O  0.04 0.50 0.50 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def loaded_window(qt_app, tmp_path):
    cif_path = tmp_path / "test.cif"
    cif_path.write_text(CIF)

    window = MainWindow()
    # isVisible() on a child widget is only ever true if every ancestor
    # up to the top-level window is shown too - without this, the
    # "does the dock actually become visible" tests below would fail
    # regardless of what _show_results_tab does, since nothing in the
    # window hierarchy would be visible at all.
    window.show()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window

    window.close()


def test_only_three_top_level_docks_exist(loaded_window):
    """Down from eight - Structure Tree, Inspector, Results."""
    from PySide6.QtWidgets import QDockWidget

    docks = loaded_window.findChildren(QDockWidget)

    assert len(docks) == 3


def test_inspector_holds_seven_panels_as_tabs(loaded_window):
    """Round 73: the element legend moved from its own tab to the top of
    the Structure tab (test_layout_round73.py)."""
    window = loaded_window

    assert window.inspector_tabs.count() == 7
    titles = [
        window.inspector_tabs.tabText(i)
        for i in range(window.inspector_tabs.count())
    ]
    assert titles == [
        "Structure", "Selected Atom", "Coordination", "Pores",
        "Surface Area", "Guest Accessibility", "Open Metal Sites",
    ]


def test_results_holds_two_panels_as_tabs(loaded_window):
    window = loaded_window

    assert window.results_tabs.count() == 2
    titles = [
        window.results_tabs.tabText(i)
        for i in range(window.results_tabs.count())
    ]
    assert titles == ["Measurements", "MOF Structure"]


def test_loading_a_structure_lands_on_the_structure_tab(loaded_window):
    window = loaded_window

    assert window.inspector_tabs.currentWidget() is window.info_panel


def test_mof_analysis_result_is_immediately_the_visible_tab(loaded_window):
    """The exact reported bug: running MOF analysis must make its
    result the active, visible tab - not something left behind another
    tab that requires a manual click (or worse) to discover.
    """
    window = loaded_window

    # Start on a different tab, as a person who had just been looking
    # at something else would be.
    window.results_tabs.setCurrentWidget(window.measurements_panel)

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )
    window._on_mof_partition_ready((partition, site_images))

    assert window.results_tabs.currentWidget() is window.mof_analysis_panel


def test_coordination_analysis_result_is_immediately_the_visible_tab(
    loaded_window,
):
    window = loaded_window

    zn_index = next(
        i
        for i, site in enumerate(window.current_structure.sites)
        if list(site.species.keys())[0].symbol == "Zn"
    )
    window._on_atom_selected_in_tree(zn_index)

    # The atom selection itself switches to "Selected Atom" - move
    # away from it before running coordination analysis, to confirm
    # that action independently raises its own tab too.
    window.inspector_tabs.setCurrentWidget(window.info_panel)

    env = window._run_coordination_analysis(
        window.current_structure,
        window.current_selected_site_index,
        window.current_site_labels,
    )
    window._on_coordination_environment_ready(env)

    assert window.inspector_tabs.currentWidget() is window.coordination_panel


def test_selecting_an_atom_shows_the_selected_atom_tab(loaded_window):
    window = loaded_window

    window.inspector_tabs.setCurrentWidget(window.coordination_panel)

    zn_index = next(
        i
        for i, site in enumerate(window.current_structure.sites)
        if list(site.species.keys())[0].symbol == "Zn"
    )
    window._on_atom_selected_in_tree(zn_index)

    assert window.inspector_tabs.currentWidget() is window.selected_atom_panel


def test_measuring_shows_the_measurements_tab(loaded_window):
    window = loaded_window

    window.results_tabs.setCurrentWidget(window.mof_analysis_panel)

    window.viewer._measurement_mode = "distance"
    atoms = window.viewer._current_model.atoms
    zn_atoms = [a for a in atoms if a.symbol == "Zn"]

    results = []
    window.viewer.on_measurement = results.append
    window.viewer._measurement_picks = []
    window.viewer._on_measurement_pick(zn_atoms[0].position)
    window.viewer._on_measurement_pick(zn_atoms[1].position)
    window._on_measurement_result(results[0])

    assert window.results_tabs.currentWidget() is window.measurements_panel


def test_inspector_and_results_docks_are_visible_after_showing_a_tab(
    loaded_window,
):
    """The reported symptom included the panel needing a window
    minimize/maximize to even appear - _show_results_tab must ensure
    the dock itself is shown, not just that the correct tab is active
    inside an already-hidden dock.
    """
    window = loaded_window

    window._results_dock.hide()
    assert not window._results_dock.isVisible()

    partition, site_images = window._run_mof_partition(
        window.current_structure, window.current_bonds
    )
    window._on_mof_partition_ready((partition, site_images))

    assert window._results_dock.isVisible()


def test_window_title_reflects_the_current_product_name(loaded_window):
    assert loaded_window.windowTitle() == "the Void"


def test_window_icon_is_set_and_not_null(loaded_window):
    assert not loaded_window.windowIcon().isNull()
