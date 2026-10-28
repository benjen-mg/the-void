"""Version 1.0.1: every CIF the Void writes has unique atom labels.

The CIF standard requires each _atom_site_label to be unique. pymatgen
keeps a site's label from the file it was read from, so a structure read
from a CIF with symmetry (one Zn1, expanded to its copies) was written
back with every copy called Zn1, and every species of a disordered site
got the same label. Olex2, SHELX and other CIF readers reject or rename
such files. What is checked here:

- a symmetry-expanded structure, and a disordered site, are written with
  unique labels;
- the labels are the structure tree's own (Zn1 Zn2 ... C1 C2 ...) and
  fit SHELX's four characters;
- nothing else changes: cell, positions and occupancies read back the
  same;
- every writer goes through the one function that does this.
"""

import warnings
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402
from pymatgen.io.cif import CifParser  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.export import export_cif, unique_cif_labels  # noqa: E402
from thevoid_core.services.site_labels import compute_site_labels  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

#: One Zn, one O and one C listed once each, in P-1: the file has three
#: labels, the structure six sites.
SYMMETRIC_CIF = """\
data_symmetric
_cell_length_a 7.0
_cell_length_b 8.0
_cell_length_c 9.0
_cell_angle_alpha 90
_cell_angle_beta 100
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P -1'
loop_
_symmetry_equiv_pos_as_xyz
'x, y, z'
'-x, -y, -z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
Zn1 Zn 0.10 0.20 0.30 1.0
O1 O 0.30 0.20 0.25 1.0
C1 C 0.40 0.35 0.10 1.0
"""


def _labels(cif_text: str) -> list[str]:
    """The _atom_site_label column of a CIF written by pymatgen."""
    lines = cif_text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "_atom_site_label")
    while lines[start - 1].strip().startswith("_atom_site"):
        start -= 1
    header = []
    i = start
    while lines[i].strip().startswith("_atom_site"):
        header.append(lines[i].strip())
        i += 1
    column = header.index("_atom_site_label")
    rows = []
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(("loop_", "_")):
        rows.append(lines[i].split()[column])
        i += 1
    return rows


def _read_back(cif_text: str, tmp_path) -> Structure:
    path = tmp_path / "written.cif"
    path.write_text(cif_text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CifParser(str(path)).parse_structures(primitive=False)[0]


def test_symmetry_copies_get_their_own_labels(tmp_path):
    source = tmp_path / "symmetric.cif"
    source.write_text(SYMMETRIC_CIF)
    structure = PymatgenCIFLoader().load(str(source)).structure
    # The bug's precondition: the copies carry the file's label.
    assert sorted(site.label for site in structure) == ["C1", "C1", "O1", "O1", "Zn1", "Zn1"]

    text = export_cif(structure)
    labels = _labels(text)
    assert len(labels) == len(set(labels)) == 6
    assert labels == compute_site_labels(structure)

    again = _read_back(text, tmp_path)
    assert again.lattice.abc == pytest.approx(structure.lattice.abc)
    assert again.matches(structure, scale=False)


def test_a_disordered_site_has_one_label_per_species(tmp_path):
    structure = Structure(
        Lattice.cubic(5.0),
        [{"Na": 0.5, "K": 0.5}, "Cl", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]],
        labels=["M1", "Cl1", "Cl1"],
    )
    text = export_cif(structure)
    labels = _labels(text)
    assert sorted(labels) == ["Cl1", "Cl2", "K1", "Na1"]

    again = _read_back(text, tmp_path)
    assert again.composition == structure.composition  # occupancies kept
    assert again[0].species == structure[0].species


def test_labels_fit_shelx_and_match_the_structure_tree():
    structure = PymatgenCIFLoader().load(str(ZN_DIA)).structure
    labels = _labels(export_cif(structure))
    assert labels == compute_site_labels(structure)
    assert max(len(label) for label in labels) <= 4
    assert unique_cif_labels(["Zn2+", "O2-", "Zn", "C"]) == ["Zn1", "O1", "Zn2", "C1"]


def test_framework_and_batch_write_unique_labels(tmp_path):
    thevoid = pytest.importorskip("thevoid")
    source = tmp_path / "symmetric.cif"
    source.write_text(SYMMETRIC_CIF)

    written = thevoid.load(source).write_cif(tmp_path / "framework.cif")
    labels = _labels(written.read_text())
    assert len(labels) == len(set(labels)) == 6

    structure = PymatgenCIFLoader().load(str(source)).structure
    thevoid.analyze_batch([structure], out=tmp_path / "batch", surface_samples_per_atom=50,
                          topology=False)
    (batch_cif,) = (tmp_path / "batch").glob("*.cif")
    labels = _labels(batch_cif.read_text())
    assert len(labels) == len(set(labels)) == 6


def test_every_cif_writer_goes_through_cif_writer():
    """No code but services/export.py builds a CifWriter itself."""
    direct = []
    for path in list((ROOT / "thevoid_core").rglob("*.py")) + list((ROOT / "thevoid").rglob("*.py")):
        if path.name == "export.py" and path.parent.name == "services":
            continue
        if "CifWriter(" in path.read_text(encoding="utf-8"):
            direct.append(str(path.relative_to(ROOT)))
    assert direct == []
