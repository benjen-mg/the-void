"""End-to-end quantum chemistry export, through the real MainWindow's
File > Export Structure menu action path rather than the service
functions in isolation.
"""

import os

import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("pyvista")
pytest.importorskip("pyvistaqt")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QMessageBox,
)

from thevoid_core.ui.windows.main_window import MainWindow  # noqa: E402

TRICLINIC_CIF = """\
data_test
_cell_length_a 5.64
_cell_length_b 6.10
_cell_length_c 7.30
_cell_angle_alpha 80
_cell_angle_beta 95
_cell_angle_gamma 105
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Zn1 Zn 0.1 0.2 0.3 1.0
O1 O 0.5 0.5 0.5 1.0
C1 C 0.9 0.8 0.7 1.0
"""


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    QMessageBox.warning = staticmethod(lambda *a, **k: None)
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    return app


@pytest.fixture
def export_window(qt_app, tmp_path):
    cif_path = tmp_path / "test.cif"
    cif_path.write_text(TRICLINIC_CIF)

    window = MainWindow()

    payload = window._load_and_analyse(str(cif_path))
    window._on_structure_ready(payload)

    yield window, tmp_path

    window.close()


def _stub_save_dialog(monkeypatch, path):
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (path, "")),
    )


@pytest.mark.parametrize(
    "format_key,filename",
    [
        ("poscar", "POSCAR"),
        ("cif", "out.cif"),
        ("xyz", "out.xyz"),
        ("gaussian", "out.gjf"),
        ("cp2k", "out.inp"),
        ("quantum_espresso", "out.in"),
    ],
)
def test_each_format_writes_a_real_file(
    export_window, monkeypatch, format_key, filename
):
    window, tmp_path = export_window
    out_path = tmp_path / filename
    _stub_save_dialog(monkeypatch, str(out_path))

    window.export_structure_as(format_key)

    assert out_path.exists()
    assert out_path.read_text().strip() != ""


def test_poscar_export_round_trips_the_loaded_structure(
    export_window, monkeypatch
):
    from pymatgen.io.vasp import Poscar

    window, tmp_path = export_window
    out_path = tmp_path / "POSCAR"
    _stub_save_dialog(monkeypatch, str(out_path))

    window.export_structure_as("poscar")

    reparsed = Poscar.from_str(out_path.read_text()).structure
    assert np.allclose(
        reparsed.lattice.matrix,
        window.current_structure.lattice.matrix,
        atol=1e-4,
    )


def test_cif_export_round_trips_the_loaded_structure(
    export_window, monkeypatch
):
    from pymatgen.core import Structure

    window, tmp_path = export_window
    out_path = tmp_path / "out.cif"
    _stub_save_dialog(monkeypatch, str(out_path))

    window.export_structure_as("cif")

    reparsed = Structure.from_str(out_path.read_text(), fmt="cif")
    assert np.allclose(
        reparsed.lattice.matrix,
        window.current_structure.lattice.matrix,
        atol=1e-4,
    )


def test_quantum_espresso_export_round_trips_the_loaded_structure(
    export_window, monkeypatch
):
    from pymatgen.io.pwscf import PWInput

    window, tmp_path = export_window
    out_path = tmp_path / "out.in"
    _stub_save_dialog(monkeypatch, str(out_path))

    window.export_structure_as("quantum_espresso")

    reparsed = PWInput.from_str(out_path.read_text())
    assert np.allclose(
        reparsed.structure.lattice.matrix,
        window.current_structure.lattice.matrix,
        atol=1e-4,
    )


def test_gaussian_export_preserves_lattice_via_tv_vectors(
    export_window, monkeypatch
):
    window, tmp_path = export_window
    out_path = tmp_path / "out.gjf"
    _stub_save_dialog(monkeypatch, str(out_path))

    window.export_structure_as("gaussian")

    text = out_path.read_text()
    tv_lines = [
        line for line in text.splitlines() if line.strip().startswith("TV")
    ]
    tv_vectors = np.array(
        [[float(v) for v in line.split()[1:4]] for line in tv_lines]
    )

    assert np.allclose(
        tv_vectors, window.current_structure.lattice.matrix
    )


def test_cp2k_export_preserves_exact_cell_vectors(
    export_window, monkeypatch
):
    window, tmp_path = export_window
    out_path = tmp_path / "out.inp"
    _stub_save_dialog(monkeypatch, str(out_path))

    window.export_structure_as("cp2k")

    text = out_path.read_text()
    cell_block = text.split("&CELL")[1].split("&END CELL")[0]
    cell_lines = {
        line.strip().split()[0]: line
        for line in cell_block.splitlines()
        if line.strip() and line.strip().split()[0] in ("A", "B", "C")
    }
    parsed_matrix = np.array(
        [
            [float(v) for v in cell_lines[label].split()[1:4]]
            for label in "ABC"
        ]
    )

    assert np.allclose(
        parsed_matrix, window.current_structure.lattice.matrix
    )


def test_export_with_no_structure_loaded_shows_a_message(
    qt_app, monkeypatch
):
    window = MainWindow()

    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        staticmethod(lambda *a, **k: shown.append(a)),
    )

    window.export_structure_as("cif")

    assert len(shown) == 1
    window.close()


def test_cancelling_the_save_dialog_does_not_write_anything(
    export_window, monkeypatch, tmp_path
):
    window, _ = export_window
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),  # user cancelled
    )

    before = list(tmp_path.iterdir())
    window.export_structure_as("cif")
    after = list(tmp_path.iterdir())

    assert before == after
