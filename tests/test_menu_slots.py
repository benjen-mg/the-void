"""Round 72: menu entries and buttons must not pass Qt's `checked`
argument into slots that take an optional argument.

QAction.triggered and QAbstractButton.clicked send `checked` (False for
an ordinary action). Connected straight to a method like
export_raspa_input_set(directory=None), that False arrived as the
directory: Simulate > Export RASPA2 GCMC Input Set... skipped its
dialogs and failed with "argument should be a str or an os.PathLike
object ..., not 'bool'". Open Batch Results, Compare with CoRE MOF,
Predict Properties and Batch Results > Add to Shared Index had the same
wiring. The tests called the methods directly, so none saw it.
"""

import ast
import os
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

UI = Path(__file__).resolve().parents[1] / "thevoid_core" / "ui"


def test_no_click_passes_its_checked_state_as_an_argument():
    """Every `.triggered/.clicked.connect(self.method)` in the UI whose
    method takes a parameter: only QActionGroup.triggered, which sends
    the QAction the method expects, may do that."""
    offenders = []
    for path in UI.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        methods = {f.name: [a.arg for a in f.args.args[1:]]
                   for c in ast.walk(tree) if isinstance(c, ast.ClassDef)
                   for f in c.body if isinstance(f, ast.FunctionDef)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "connect" and node.args):
                continue
            signal, slot = node.func.value, node.args[0]
            if not (isinstance(signal, ast.Attribute) and signal.attr in ("triggered", "clicked")):
                continue
            if not (isinstance(slot, ast.Attribute) and isinstance(slot.value, ast.Name)
                    and slot.value.id == "self" and methods.get(slot.attr)):
                continue
            emitter = signal.value
            if isinstance(emitter, ast.Attribute) and emitter.attr.endswith("_group") \
                    and methods[slot.attr] == ["action"]:
                continue  # QActionGroup.triggered(QAction)
            offenders.append(f"{path.name}:{node.lineno} {slot.attr}({', '.join(methods[slot.attr])})")
    assert offenders == []


@pytest.fixture
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_menu_entries_call_their_slots_with_nothing(qt_app, monkeypatch):
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from thevoid_core.ui.windows.main_window import MainWindow

    received = {}
    slots = {"simulate_raspa_action": "export_raspa_input_set",
             "export_raspa_action": "export_raspa_input_set",
             "open_batch_action": "open_batch_results",
             "compare_coremof_action": "compare_with_coremof",
             "predict_action": "predict_properties",
             "export_report_action": "export_analysis_report"}
    # The recorders take one optional argument, as the real slots do:
    # PySide6 passes `checked` to a slot that can take it.
    for method in set(slots.values()):
        monkeypatch.setattr(MainWindow, method,
                            lambda self, argument=None, _m=method: received.setdefault(_m, argument))
    window = MainWindow()
    try:
        for action, method in slots.items():
            received.pop(method, None)
            getattr(window, action).setEnabled(True)
            getattr(window, action).trigger()
            assert received[method] is None, f"{action} passed {received[method]!r}"
    finally:
        window.close()


def test_buttons_call_their_slots_with_nothing(qt_app, monkeypatch):
    from thevoid_core.ui.windows import batch_results_window

    received = []
    monkeypatch.setattr(batch_results_window.BatchResultsWindow, "export_view",
                        lambda self, filename=None: received.append(("export_view", filename)))
    monkeypatch.setattr(batch_results_window.BatchResultsWindow, "add_to_index",
                        lambda self, values=None: received.append(("add_to_index", values)))
    batch = batch_results_window.BatchResultsWindow()
    for button in (batch.export_button, batch.index_button):
        button.setEnabled(True)
        button.click()
    assert received == [("export_view", None), ("add_to_index", None)]
    batch.close()
