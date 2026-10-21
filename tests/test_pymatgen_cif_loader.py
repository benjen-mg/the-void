"""PymatgenCIFLoader tests.

The disorder-group reading in particular needs a real, small CIF on
disk: it deliberately reads the raw file rather than the parsed
Structure (see the loader's docstring for why), so a fake structure
object can't exercise it.
"""

import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services.cif.pymatgen_cif_loader import (  # noqa: E402
    PymatgenCIFLoader,
)

DISORDERED_CIF = """\
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
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
_atom_site_disorder_group
O1 O 0.10 0.10 0.10 0.5 1
O2 O 0.15 0.10 0.10 0.5 2
C1 C 0.50 0.50 0.50 1.0 .
"""

ORDERED_CIF = """\
data_test
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
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
C1 C 0.50 0.50 0.50 1.0
"""


@pytest.fixture
def disordered_cif(tmp_path):
    path = tmp_path / "disordered.cif"
    path.write_text(DISORDERED_CIF)
    return str(path)


@pytest.fixture
def ordered_cif(tmp_path):
    path = tmp_path / "ordered.cif"
    path.write_text(ORDERED_CIF)
    return str(path)


def test_disorder_groups_are_read_from_the_raw_cif(disordered_cif):
    info = PymatgenCIFLoader().load(disordered_cif)

    assert info.disorder_groups == {"O1": "1", "O2": "2"}
    assert info.has_explicit_disorder_groups


def test_unset_group_marker_is_excluded(disordered_cif):
    """C1's group is '.' (not applicable) - it must not appear."""
    info = PymatgenCIFLoader().load(disordered_cif)

    assert "C1" not in info.disorder_groups


def test_no_disorder_column_gives_an_empty_dict(ordered_cif):
    info = PymatgenCIFLoader().load(ordered_cif)

    assert info.disorder_groups == {}
    assert not info.has_explicit_disorder_groups


def test_partial_occupancy_is_still_counted_independently(disordered_cif):
    """Occupancy detection (from the parsed Structure) and disorder
    group reading (from the raw file) are two separate code paths;
    both should agree something is disordered here.
    """
    info = PymatgenCIFLoader().load(disordered_cif)

    assert info.partial_occupancy_sites == 2
    assert not info.is_ordered


def test_basic_fields_are_populated(ordered_cif):
    info = PymatgenCIFLoader().load(ordered_cif)

    assert info.site_count == 1
    assert info.a == pytest.approx(5.0)
    assert info.b == pytest.approx(5.0)
    assert info.c == pytest.approx(5.0)
    assert info.structure is not None
