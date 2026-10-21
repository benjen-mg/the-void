"""Round 73: panel sizes and the element key, from the owner's report.

- The Results dock opened about 100 px tall, so after Analysis > MOF
  Structure the linker and node lists showed one row or none, each
  behind its own tab.
- The element colour key was on a Legend tab of its own, the last of
  eight, scrolled off the Inspector's tab bar: it had to be looked for.

Now the Results dock starts at 30% of the window's height, is grown back
to at least 260 px whenever a result is shown in it, and linkers and
nodes sit side by side. The element key, with each element's colour
and its sites per cell, is the first thing on the Structure tab. The
window and dock sizes are kept from one session to the next.
"""

import os

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from pathlib import Path  # noqa: E402

from PySide6.QtCore import QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

SAMPLE = ZN_DIA  # the structure loaded


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def settings_file(tmp_path):
    return str(tmp_path / "settings.ini")


@pytest.fixture
def make_window(qt_app, monkeypatch, settings_file):
    from thevoid_core.ui.windows import main_window as module

    monkeypatch.setattr(module.MainWindow, "_settings",
                        staticmethod(lambda: QSettings(settings_file, QSettings.Format.IniFormat)))
    for name in ("information", "warning"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(lambda *a, **k: None))
    made = []

    def make():
        window = module.MainWindow()
        window.show()
        qt_app.processEvents()
        made.append(window)
        return window

    yield make
    for window in made:
        if window.isVisible():
            window.close()


def _load(window, path=SAMPLE):
    window._on_structure_ready(window._load_and_analyse(str(path)))


def test_the_results_dock_starts_with_room(make_window):
    window = make_window()
    assert window._results_dock.height() >= window.RESULTS_MIN_HEIGHT
    assert window._results_dock.height() >= 0.25 * window.height()


def test_showing_mof_results_gives_the_tables_room(make_window, qt_app):
    window = make_window()
    _load(window)
    window.resizeDocks([window._results_dock], [90], Qt.Orientation.Vertical)
    qt_app.processEvents()
    assert window._results_dock.height() < window.RESULTS_MIN_HEIGHT
    window._on_mof_partition_ready(window._run_mof_partition(window.current_structure,
                                                             window.current_bonds))
    qt_app.processEvents()
    assert window._results_dock.height() >= window.RESULTS_MIN_HEIGHT
    panel = window.mof_analysis_panel
    # Side by side, both visible at once, with their counts.
    assert panel.linker_table.isVisible() and panel.node_table.isVisible()
    assert panel.linker_heading.text() == f"Linkers ({panel.linker_table.rowCount()})"
    assert panel.node_heading.text() == f"Metal Nodes ({panel.node_table.rowCount()})"
    assert panel.linker_table.rowCount() > 0 and panel.node_table.rowCount() > 0
    # At least three rows' height for each list.
    row = panel.linker_table.verticalHeader().defaultSectionSize()
    assert panel.linker_table.viewport().height() >= 3 * row
    # A taller dock is left alone.
    window.resizeDocks([window._results_dock], [420], Qt.Orientation.Vertical)
    qt_app.processEvents()
    height = window._results_dock.height()
    window._show_results_tab(window.mof_analysis_panel)
    qt_app.processEvents()
    assert window._results_dock.height() == height


def test_the_element_key_is_first_on_the_structure_tab(make_window, qt_app):
    from thevoid_core.ui.widgets.element_legend import ColorCircle

    window = make_window()
    _load(window)
    qt_app.processEvents()
    assert window.inspector_tabs.currentWidget() is window.info_panel
    assert window.element_legend.isVisible()
    shown = {}
    for row in window.element_legend._rows:
        labels = [label.text() for label in row.findChildren(QLabel)]
        circle = row.findChild(ColorCircle)
        shown[labels[0]] = (int(labels[1]), circle.color.name())
    structure = window.current_structure
    for symbol, (count, color) in shown.items():
        assert count == sum(1 for s in structure.sites
                            if next(iter(s.species)).symbol == symbol)
        r, g, b = window.viewer.color_provider.get_color(symbol)
        assert color == "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    assert set(shown) == {"C", "H", "N", "O", "Zn"}
    # Above everything else on the tab.
    group = window.info_panel.elements_group
    others = [g for g in window.info_panel.findChildren(type(group)) if g is not group]
    assert all(group.mapTo(window.info_panel, group.rect().topLeft()).y()
               < g.mapTo(window.info_panel, g.rect().topLeft()).y() for g in others)
    # No Legend tab left to look for.
    tabs = [window.inspector_tabs.tabText(i) for i in range(window.inspector_tabs.count())]
    assert "Legend" not in tabs


def test_the_element_key_follows_colour_changes(make_window, qt_app):
    from thevoid_core.ui.widgets.element_legend import ColorCircle

    window = make_window()
    _load(window)
    window.viewer.color_provider.set_color("Zn", (0.0, 1.0, 0.0))
    window._on_atom_colors_changed()
    zinc = next(r for r in window.element_legend._rows
                if r.findChildren(QLabel)[0].text() == "Zn")
    assert zinc.findChild(ColorCircle).color.name() == "#00ff00"


def test_sizes_are_kept_between_sessions(make_window, qt_app):
    """(The window is first made to fit the offscreen test screen,
    800 x 800, which a restored geometry is clamped to.)"""
    first = make_window()
    first.resize(780, 720)
    qt_app.processEvents()
    first.resizeDocks([first._results_dock], [330], Qt.Orientation.Vertical)
    first.resizeDocks([first._inspector_dock], [300], Qt.Orientation.Horizontal)
    qt_app.processEvents()
    kept = (first._results_dock.height(), first._inspector_dock.width())
    first.close()
    second = make_window()
    assert abs(second._results_dock.height() - kept[0]) <= 2
    assert abs(second._inspector_dock.width() - kept[1]) <= 2


def test_a_closed_dock_comes_back_next_session(make_window):
    first = make_window()
    first._results_dock.close()
    first.close()
    second = make_window()
    assert second._results_dock.isVisible()


def test_an_old_or_foreign_layout_is_ignored(make_window, settings_file):
    settings = QSettings(settings_file, QSettings.Format.IniFormat)
    settings.setValue("layout/version", 0)
    settings.setValue("layout/state", b"not a layout")
    settings.sync()
    window = make_window()
    assert window._results_dock.height() >= window.RESULTS_MIN_HEIGHT
