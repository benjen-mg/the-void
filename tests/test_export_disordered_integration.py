"""The disordered-structure export path, through the real MainWindow.

Covers what the service-level tests cannot: that the person is asked
before their minor disorder components are thrown away, that answering
No abandons the export without writing anything, and that an
unexpected writer failure reaches a dialog instead of propagating out
of a Qt slot.
"""

import os

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QMessageBox,
)

from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

#: Na at full occupancy plus four quarter-occupancy Cl alternatives:
#: one disordered site written the way a deposited CIF writes one.
DISORDERED_CIF = """\
data_disordered
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
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
Na1 Na 0.0 0.0 0.0 1.0
Cl1 Cl 0.50 0.50 0.50 0.25
Cl2 Cl 0.56 0.50 0.50 0.25
Cl3 Cl 0.50 0.56 0.50 0.25
Cl4 Cl 0.50 0.50 0.56 0.25
"""

ORDERED_CIF = """\
data_ordered
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
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
Na1 Na 0.0 0.0 0.0 1.0
Cl1 Cl 0.5 0.5 0.5 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


def _window_for(qt_app, tmp_path, cif_text, name):
    cif_path = tmp_path / name
    cif_path.write_text(cif_text)

    window = MainWindow()
    window._on_structure_ready(window._load_and_analyse(str(cif_path)))

    return window


@pytest.fixture
def disordered_window(qt_app, tmp_path):
    window = _window_for(qt_app, tmp_path, DISORDERED_CIF, "disordered.cif")
    yield window, tmp_path
    window.close()


@pytest.fixture
def ordered_window(qt_app, tmp_path):
    window = _window_for(qt_app, tmp_path, ORDERED_CIF, "ordered.cif")
    yield window, tmp_path
    window.close()


def _stub_save_dialog(monkeypatch, path):
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(path), "")),
    )


def _stub_question(monkeypatch, answer):
    asked = []

    def fake_question(*args, **kwargs):
        asked.append(args)
        return answer

    monkeypatch.setattr(QMessageBox, "question", staticmethod(fake_question))

    return asked


def _stub_critical(monkeypatch):
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: shown.append(a)),
    )
    return shown


@pytest.mark.parametrize("format_key,name", [
    ("poscar", "POSCAR"),
    ("quantum_espresso", "out.in"),
])
def test_disordered_export_writes_after_confirmation(
    disordered_window, monkeypatch, format_key, name
):
    """The bug in full: before the fix, POSCAR raised ValueError here
    and Quantum ESPRESSO raised an AttributeError that escaped the
    handler. Both must now produce a real file.
    """
    window, tmp_path = disordered_window
    out_path = tmp_path / name

    _stub_save_dialog(monkeypatch, out_path)
    asked = _stub_question(monkeypatch, QMessageBox.StandardButton.Yes)
    shown = _stub_critical(monkeypatch)

    window.export_structure_as(format_key)

    assert asked, "the person must be asked before atoms are discarded"
    assert not shown, "a confirmed export must not report a failure"
    assert out_path.exists()
    assert out_path.read_text().strip()


def test_the_written_file_holds_only_the_major_components(
    disordered_window, monkeypatch
):
    window, tmp_path = disordered_window
    out_path = tmp_path / "POSCAR"

    _stub_save_dialog(monkeypatch, out_path)
    _stub_question(monkeypatch, QMessageBox.StandardButton.Yes)

    window.export_structure_as("poscar")

    from pymatgen.io.vasp import Poscar

    reparsed = Poscar.from_str(out_path.read_text()).structure

    assert len(reparsed) == 1
    assert reparsed[0].specie.symbol == "Na"


def test_declining_writes_nothing(disordered_window, monkeypatch):
    window, tmp_path = disordered_window
    out_path = tmp_path / "POSCAR"

    _stub_save_dialog(monkeypatch, out_path)
    _stub_question(monkeypatch, QMessageBox.StandardButton.No)

    window.export_structure_as("poscar")

    assert not out_path.exists()


def test_the_loaded_structure_is_not_modified_by_exporting(
    disordered_window, monkeypatch
):
    window, tmp_path = disordered_window

    _stub_save_dialog(monkeypatch, tmp_path / "POSCAR")
    _stub_question(monkeypatch, QMessageBox.StandardButton.Yes)

    window.export_structure_as("poscar")

    assert len(window.current_structure) == 5
    assert not window.current_structure.is_ordered


@pytest.mark.parametrize(
    "format_key,name",
    [
        ("cif", "out.cif"),
        ("xyz", "out.xyz"),
        ("gaussian", "out.gjf"),
        ("cp2k", "out.inp"),
    ],
)
def test_the_other_formats_export_disorder_without_asking(
    disordered_window, monkeypatch, format_key, name
):
    window, tmp_path = disordered_window
    out_path = tmp_path / name

    _stub_save_dialog(monkeypatch, out_path)
    asked = _stub_question(monkeypatch, QMessageBox.StandardButton.Yes)

    window.export_structure_as(format_key)

    assert not asked, f"{format_key} can hold this structure as it is"
    assert out_path.exists()


@pytest.mark.parametrize("format_key,name", [
    ("poscar", "POSCAR"),
    ("quantum_espresso", "out.in"),
])
def test_an_ordered_structure_is_never_queried(
    ordered_window, monkeypatch, format_key, name
):
    window, tmp_path = ordered_window
    out_path = tmp_path / name

    _stub_save_dialog(monkeypatch, out_path)
    asked = _stub_question(monkeypatch, QMessageBox.StandardButton.Yes)

    window.export_structure_as(format_key)

    assert not asked
    assert out_path.exists()


def test_an_unexpected_writer_failure_becomes_a_dialog(
    ordered_window, monkeypatch
):
    """The reason the handler is deliberately broad. A writer raising
    something outside (OSError, ValueError, RuntimeError) must still be
    reported rather than propagate out of the slot - AttributeError
    being the exact type that used to get through.
    """
    window, tmp_path = ordered_window
    out_path = tmp_path / "out.cif"

    _stub_save_dialog(monkeypatch, out_path)
    shown = _stub_critical(monkeypatch)

    def exploding_writer(_structure):
        raise AttributeError("attr='specie' not found on PeriodicSite")

    monkeypatch.setitem(
        MainWindow.EXPORT_FORMATS,
        "cif",
        (exploding_writer, "Export CIF", "structure.cif", "CIF Files (*.cif)"),
    )

    window.export_structure_as("cif")

    assert shown, "the failure must reach an Export failed dialog"
    assert not out_path.exists()
