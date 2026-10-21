"""End-to-end 2D Skeletal Diagram generation, through the real
MainWindow's actual menu-triggered path, on real structures rather than
a small synthetic case: an openly licensed Zn MOF.
"""

import os

import pytest

pytest.importorskip("rdkit")
pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtCore import QThreadPool  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from thevoid_core.services.skeletal_diagram_render import (  # noqa: E402
    render_skeletal_diagram,
)
from thevoid_core.ui.widgets.skeletal_diagram_dialog import (  # noqa: E402
    SkeletalDiagramDialog,
)
from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

SAMPLE_CIF = str(ZN_DIA)


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


def _window(path):
    window = MainWindow()
    window._on_structure_ready(window._load_and_analyse(str(path)))
    return window


@pytest.fixture
def sample_window(qt_app):
    window = _window(SAMPLE_CIF)
    yield window
    window.close()


def test_the_sample_deduplicates_its_fragments(sample_window):
    """The open Zn MOF's 8 identical linkers collapse to one picture."""
    result = sample_window._generate_skeletal_diagram(
        sample_window.current_structure, sample_window.current_bonds)
    assert result.skipped_count == 0
    total = sum(f.occurrence_count for f in result.fragments)
    assert 0 < len(result.fragments) < total


def test_connected_node_and_linker_clusters_appear_for_the_real_structure(
    sample_window,
):
    """The feature this round added: a metal centre shown connected to
    its full surrounding linkers, not just isolated fragments -
    confirmed against the real structure by label, and confirmed the
    resulting mol is genuinely bigger than the node's own bare
    coordination sphere (it pulled in whole linkers, not just their
    first connecting atom).
    """
    window = sample_window

    result = window._generate_skeletal_diagram(
        window.current_structure, window.current_bonds
    )

    cluster_fragments = [
        f for f in result.fragments if f.label.endswith("+ linkers")
    ]
    simple_node_fragments = [
        f
        for f in result.fragments
        if f.label.endswith("node") and not f.label.endswith("+ linkers")
    ]

    assert len(cluster_fragments) > 0
    assert len(simple_node_fragments) > 0

    largest_cluster = max(f.mol.GetNumAtoms() for f in cluster_fragments)
    largest_simple_node = max(
        f.mol.GetNumAtoms() for f in simple_node_fragments
    )
    assert largest_cluster > largest_simple_node


def test_at_least_one_fragment_has_occurrence_count_above_one(
    sample_window,
):
    window = sample_window

    result = window._generate_skeletal_diagram(
        window.current_structure, window.current_bonds
    )

    assert any(f.occurrence_count > 1 for f in result.fragments)


def test_every_fragment_mol_has_a_2d_conformer(sample_window):
    window = sample_window

    result = window._generate_skeletal_diagram(
        window.current_structure, window.current_bonds
    )

    for fragment in result.fragments:
        assert fragment.mol.GetNumConformers() > 0


def test_rendered_image_is_produced_without_error(sample_window):
    window = sample_window

    result = window._generate_skeletal_diagram(
        window.current_structure, window.current_bonds
    )
    image = render_skeletal_diagram(result)

    assert image.size[0] > 0
    assert image.size[1] > 0


def test_menu_action_end_to_end_shows_a_dialog(sample_window, monkeypatch):
    """The actual menu-triggered path: _on_skeletal_diagram_menu_action
    dispatches to a real background QThreadPool worker, so this test
    waits for that worker to actually finish and for Qt's queued
    cross-thread signal delivery to run, rather than checking
    immediately after the call.
    """
    window = sample_window

    shown_dialogs = []

    def fake_exec(self):
        shown_dialogs.append(self)
        return 0

    monkeypatch.setattr(SkeletalDiagramDialog, "exec", fake_exec)

    window._on_skeletal_diagram_menu_action()

    QThreadPool.globalInstance().waitForDone(10_000)
    for _ in range(20):
        QApplication.processEvents()
        if shown_dialogs:
            break

    assert len(shown_dialogs) == 1
    assert shown_dialogs[0].windowTitle() == "2D Skeletal Diagram"
    assert not shown_dialogs[0].image_label.pixmap().isNull()


def test_no_structure_loaded_shows_a_message(qt_app, monkeypatch):
    window = MainWindow()

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: shown.append(a)),
    )

    window._on_skeletal_diagram_menu_action()

    assert len(shown) == 1
    window.close()


def test_missing_rdkit_shows_a_clear_actionable_message_not_a_raw_error(
    sample_window, monkeypatch
):
    """A real reported bug: without rdkit installed, the menu action
    used to let a bare ModuleNotFoundError('No module named rdkit')
    propagate all the way to the generic worker-failure dialog - a
    correct but genuinely unhelpful message for someone who doesn't
    know what rdkit is or how to install it. Simulated here via
    importlib.util.find_spec, since rdkit is actually installed in
    this environment for every other test in this file.
    """
    import importlib.util

    window = sample_window

    real_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *a, **k):
        if name == "rdkit":
            return None
        return real_find_spec(name, *a, **k)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: shown.append(a)),
    )

    window._on_skeletal_diagram_menu_action()

    assert len(shown) == 1
    message = shown[0][2]
    assert "rdkit" in message.lower()
    assert "pip install" in message
    assert not window._busy  # never dispatched to the worker at all
