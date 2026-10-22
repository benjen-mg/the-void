"""Select/View cursor mode at the MainWindow level: a single toolbar
location (not also duplicated as a View menu entry), and the action
group correctly reflecting which mode is active - the checked state is
what theme.py's QToolButton:checked styling depends on to actually be
visible, so this is also an indirect regression test for "there's no
indication one has been selected".
"""

import os

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QMenu,
    QMessageBox,
    QToolBar,
)

from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def window(qt_app):
    win = MainWindow()
    yield win
    win.close()


def _view_menu(window):
    for action in window.menuBar().actions():
        if action.text() == "View":
            return action.menu()
    raise AssertionError("View menu not found")


def test_cursor_mode_is_not_a_view_menu_entry(window):
    """The duplicate being removed: Select/View used to appear both as
    a View > Cursor Mode submenu and as a toolbar - only the toolbar
    should remain.
    """
    view_menu = _view_menu(window)
    submenu_titles = [
        action.text() for action in view_menu.actions() if action.menu()
    ]

    assert "Cursor Mode" not in submenu_titles


def test_select_and_view_actions_are_not_in_any_menu(window):
    """Neither action is associated with any QMenu widget - only the
    toolbar - confirming the menu route was actually removed, not just
    renamed or hidden.
    """
    for action in (window.select_mode_action, window.view_mode_action):
        assert not any(
            isinstance(widget, QMenu)
            for widget in action.associatedObjects()
        )


def test_exactly_one_toolbar_holds_the_cursor_mode_actions(window):
    toolbars = window.findChildren(QToolBar)
    holding = [
        tb
        for tb in toolbars
        if window.select_mode_action in tb.actions()
        and window.view_mode_action in tb.actions()
    ]

    assert len(holding) == 1


def test_select_is_checked_by_default(window):
    assert window.select_mode_action.isChecked()
    assert not window.view_mode_action.isChecked()


def test_choosing_view_mode_checks_it_and_unchecks_select(window):
    window.view_mode_action.trigger()

    assert window.view_mode_action.isChecked()
    assert not window.select_mode_action.isChecked()


def test_choosing_select_mode_again_checks_it_back(window):
    """Where VTK can pick atoms, Select is checked again. Where it cannot
    (no working VTK interactor, as on GitHub's test machines), the
    window falls back to View rather than showing a Select mode that
    does nothing."""
    window.view_mode_action.trigger()
    window.select_mode_action.trigger()

    if window.viewer.picking_enabled:
        assert window.select_mode_action.isChecked()
        assert not window.view_mode_action.isChecked()
    else:
        assert window.view_mode_action.isChecked()
        assert not window.select_mode_action.isChecked()


def test_the_two_actions_are_mutually_exclusive(window):
    """Exactly one of the two is ever checked - confirmed through the
    action group itself, which is what the visible highlight in the
    toolbar reflects.
    """
    assert window.cursor_mode_group.isExclusive()
    checked = [
        a
        for a in (window.select_mode_action, window.view_mode_action)
        if a.isChecked()
    ]
    assert len(checked) == 1
