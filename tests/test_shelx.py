"""Version 1.1.0: reading SHELX .res / .ins files (services/shelx).

Olex2 and SHELXL keep a structure being refined as a .res file. Reading
it wrongly would be silent: a plausible structure with the wrong
occupancies, a missing symmetry copy, or hydrogens of the wrong part.
So the reader is checked three ways:

- against SHELXL itself: I-43d.res and the CIF SHELXL wrote from the
  same refinement must give the same structure (sites, positions,
  occupancies, disorder groups, space group);
- against the file's own UNIT instruction: the cell contents read must
  be the contents the crystallographer declared;
- on small files written here, where each rule of the format (10m + p
  coding, free variables, special positions, PART sof, riding
  hydrogens, centring, comments and continuation lines) has a known
  answer.
"""

import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.shelx import ShelxError, parse_shelx, read_shelx  # noqa: E402
from thevoid_core.services.shelx.reader import (  # noqa: E402
    decode,
    equivalent_u,
    operator_xyz,
    parse_operator,
    space_group_operators,
)

FIXTURES = Path(__file__).parent / "fixtures" / "shelx"
I43D_RES, I43D_CIF, P21C_RES = (FIXTURES / n for n in ("I-43d.res", "I-43d.cif", "p21c.res"))


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PymatgenCIFLoader().load(str(path))


# ----------------------------------------------------------------------
# Against SHELXL's own CIF, and the file's own UNIT
# ----------------------------------------------------------------------


def test_a_res_file_gives_the_structure_shelxl_wrote_as_a_cif():
    res, cif = _load(I43D_RES), _load(I43D_CIF)
    assert res.space_group == cif.space_group == "I-43d"
    assert res.site_count == cif.site_count == 2860

    a, b = cif.structure, res.structure
    distances = a.lattice.get_all_distances(a.frac_coords, b.frac_coords)
    # The CIF rounds coordinates to their standard uncertainty.
    assert distances.min(axis=1).max() < 0.02 and distances.min(axis=0).max() < 0.02
    for i, j in enumerate(distances.argmin(axis=1)):
        assert a[i].species.elements == b[j].species.elements
        assert a[i].species.num_atoms == pytest.approx(b[j].species.num_atoms, abs=2e-4)

    assert res.disorder_groups == cif.disorder_groups
    assert set(res.disorder_groups.values()) == {"-1", "-2", "-3"}


def test_the_cell_holds_what_unit_declares():
    """UNIT is the crystallographer's own count of each element in the
    cell. I-43d: C644 H876 N60 P48 Cl28 Ni16."""
    model = read_shelx(I43D_RES)
    composition = _load(I43D_RES).structure.composition
    for element, count in zip(model.sfac, model.unit):
        assert composition[element.capitalize()] == pytest.approx(count, abs=0.01)


def test_special_positions_and_part_occupancies_in_the_real_file():
    atoms = {a.name: a for a in read_shelx(I43D_RES).atoms}
    ni, cl2 = atoms["NI1"], atoms["CL2"]
    # Ni on the threefold axis (x, x, x): sof 1/3, a full site.
    assert ni.sof == pytest.approx(1 / 3, abs=1e-4) and ni.site_symmetry_order == 3
    assert ni.occupancy == 1.0
    # Cl2 at (1/2, 1/4, 5/8), site symmetry -4: sof 1/4, a full site.
    assert cl2.sof == pytest.approx(0.25) and cl2.site_symmetry_order == 4
    assert cl2.occupancy == 1.0
    # "PART -1 10.25": a quarter-occupied molecule, off the special position.
    part = [a for a in atoms.values() if a.part == -1]
    assert part and all(a.occupancy == pytest.approx(0.25) for a in part)
    assert all(a.site_symmetry_order == 1 for a in part)


def test_free_variable_parts_and_residues_in_the_real_file():
    """p21c.res: "PART 2 -31" is 1 - fv(3), "PART 1 31" is fv(3); the
    two parts of each disordered group sum to one, so the cell's
    contents are whole numbers. Atom names repeat between residues and
    are told apart as SHELXL's CIF does (O1_4)."""
    model = read_shelx(P21C_RES)
    assert model.fvar == pytest.approx([0.08684, 0.48096, 0.55764])
    first = model.atoms[0]
    assert (first.label, first.part, first.residue) == ("O1_4", 2, 4)
    assert first.occupancy == pytest.approx(1 - 0.55764)
    assert any(a.part == 1 and a.occupancy == pytest.approx(0.55764) for a in model.atoms)

    info = _load(P21C_RES)
    assert info.space_group == "P2_1/c"
    amounts = info.structure.composition.get_el_amt_dict()
    assert amounts == pytest.approx({"Al": 4, "Ga": 4, "H": 96, "C": 136, "O": 16, "F": 144},
                                    abs=1e-3)
    labels = [a.label for a in model.atoms]
    assert len(labels) == len(set(labels))


# ----------------------------------------------------------------------
# The format's rules, one at a time
# ----------------------------------------------------------------------


def test_numbers_coded_as_10m_plus_p():
    fvar = [0.5, 0.7, 0.2]
    assert decode(11.0, fvar) == pytest.approx(1.0)        # fixed at 1
    assert decode(10.5, fvar) == pytest.approx(0.5)        # fixed at 0.5
    assert decode(0.25, fvar) == pytest.approx(0.25)       # refined
    assert decode(-0.3, fvar) == pytest.approx(-0.3)
    assert decode(21.0, fvar) == pytest.approx(0.7)        # fv(2)
    assert decode(-21.0, fvar) == pytest.approx(0.3)       # 1 - fv(2)
    assert decode(30.5, fvar) == pytest.approx(0.1)        # 0.5 x fv(3)
    assert decode(-30.5, fvar) == pytest.approx(0.4)       # 0.5 x (1 - fv(3))
    with pytest.raises(ShelxError, match="free variable 4"):
        decode(41.0, fvar)


def test_symmetry_operators_are_read_and_written():
    rotation, translation = parse_operator("-X+Y, 1/2-X, 0.25+Z")
    assert rotation.tolist() == [[-1, 1, 0], [-1, 0, 0], [0, 0, 1]]
    assert translation == (0, Fraction(1, 2), Fraction(1, 4))
    assert operator_xyz((rotation, translation)) == "-x+y,-x+1/2,z+1/4"
    assert operator_xyz(parse_operator("0.5-X, -Y, 0.5+Z")) == "-x+1/2,-y,z+1/2"
    with pytest.raises(ShelxError):
        parse_operator("X, Y")


def test_latt_gives_the_centring_and_the_inversion():
    twofold = [parse_operator("-X, 0.5+Y, 0.5-Z")]
    assert len(space_group_operators(1, twofold)) == 4       # P2(1)/c
    assert len(space_group_operators(-1, twofold)) == 2      # no inversion
    assert len(space_group_operators(2, twofold)) == 8       # I-centred
    assert len(space_group_operators(-4, [])) == 4           # F, one operator
    assert len(read_shelx(I43D_RES).operators) == 48         # I-43d: 24 x 2
    with pytest.raises(ShelxError, match="LATT"):
        space_group_operators(9, [])


#: A file written here: an I-centred orthorhombic cell with an atom on
#: a mirror-related general position, one on the origin (site symmetry
#: mmm, order 8 of 16), a two-part disorder tied to free variable 2, a
#: riding hydrogen, a comment line, a continuation and a Q peak.
SMALL = """\
TITL small test in Immm
CELL 0.71073 6.0 7.0 8.0 90 90 90
ZERR 2 0.001 0.001 0.001 0 0 0
LATT 2
SYMM -X, -Y, Z
SYMM -X, Y, -Z
SYMM X, -Y, -Z
SFAC C H ZN
UNIT 16 8 2
  this line starts with a space, so it is a comment
TEMP -100
FVAR 1.0 0.6
ZN1  3  0.0 0.0 0.0  10.125  0.02 0.03 =
     0.04 0 0 0
C1   1  0.2 0.3 0.1  11.0  0.05   ! a comment
H1   2  0.25 0.35 0.15  11.0  -1.2
PART 1
C2   1  0.1 0.4 0.3  21.0  0.05
PART 2
C3   1  0.15 0.45 0.35  -21.0  0.05
PART 0
HKLF 4
REM R1 =  0.0250 for    900 Fo > 4sig(Fo)  and  0.0300 for all   1000 data
REM     50 parameters refined using      2 restraints
END
REM Highest difference peak  0.300,  deepest hole -0.200,  1-sigma level  0.050
Q1   1  0.5 0.5 0.5  11.0  0.05  0.30
"""


def test_a_small_file_built_so_every_answer_is_known(tmp_path):
    model = parse_shelx(SMALL)
    assert len(model.operators) == 16                       # mmm (8) x I (2)
    atoms = {a.name: a for a in model.atoms}
    assert set(atoms) == {"ZN1", "C1", "H1", "C2", "C3"}    # the Q peak is not an atom

    zn = atoms["ZN1"]
    assert zn.site_symmetry_order == 8 and zn.occupancy == 1.0
    assert zn.u_aniso == (0.02, 0.03, 0.04, 0, 0, 0)
    assert zn.u_iso == pytest.approx(0.03)                  # orthogonal cell: the mean
    assert atoms["H1"].u_iso == pytest.approx(1.2 * 0.05)   # rides on C1
    assert atoms["C2"].occupancy == pytest.approx(0.6)
    assert atoms["C3"].occupancy == pytest.approx(0.4)
    assert (atoms["C2"].part, atoms["C3"].part) == (1, 2)
    assert model.temperature_C == -100 and model.difference_peaks == 1
    assert model.refinement == {"r1": 0.025, "reflections_observed": 900, "r1_all": 0.03,
                                "reflections_all": 1000, "parameters": 50, "restraints": 2,
                                "peak": 0.3, "hole": -0.2}

    path = tmp_path / "small.res"
    path.write_text(SMALL)
    info = _load(path)
    amounts = info.structure.composition.get_el_amt_dict()
    # Zn: 2 per cell (the origin and the body centre). C1 and H1 on
    # general positions: 16 each. C2 + C3: 16 x (0.6 + 0.4).
    assert amounts == pytest.approx({"Zn": 2, "C": 32, "H": 16})
    assert info.space_group == "Immm"
    assert info.disorder_groups == {"C2": "1", "C3": "2"}
    assert info.partial_occupancy_sites == 32


def test_equivalent_u_in_a_cell_that_is_not_orthogonal():
    """An isotropic tensor written in a monoclinic cell: U11 = U22 =
    U33 = U, U13 = U cos(beta*), the others zero. Its Ueq is U."""
    beta = 110.0
    beta_star = 180.0 - beta
    u = 0.04
    tensor = (u, u, u, 0.0, u * np.cos(np.radians(beta_star)), 0.0)
    assert equivalent_u(tensor, (7, 8, 9, 90, beta, 90)) == pytest.approx(u)


def test_files_that_are_not_shelx_say_why(tmp_path):
    with pytest.raises(ShelxError, match="no CELL"):
        parse_shelx("TITL nothing here\nEND\n")
    with pytest.raises(ShelxError, match="no SFAC"):
        parse_shelx("CELL 0.71 5 5 5 90 90 90\nC1 1 0 0 0\nEND\n")
    with pytest.raises(ShelxError, match="no atoms"):
        parse_shelx("CELL 0.71 5 5 5 90 90 90\nSFAC C\nHKLF 4\nEND\n")
    with pytest.raises(ShelxError, match="cannot read this line"):
        parse_shelx("CELL 0.71 5 5 5 90 90 90\nSFAC C\nC1 7 0 0 0\nEND\n")


def test_an_ins_file_and_the_other_ways_in(tmp_path):
    """A .ins (not yet refined) loads the same way; the command line and
    the Python API read SHELX files through the same loader."""
    thevoid = pytest.importorskip("thevoid")
    ins = tmp_path / "small.ins"
    ins.write_text(SMALL.split("HKLF 4")[0] + "HKLF 4\nEND\n")
    assert _load(ins).structure.composition.get_el_amt_dict() == pytest.approx(
        {"Zn": 2, "C": 32, "H": 16})

    fw = thevoid.load(I43D_RES)
    assert (fw.n_sites, fw.space_group) == (2860, "I-43d")

    from thevoid_core.ui.windows.main_window import MainWindow

    assert {".res", ".ins"} <= MainWindow._DROPPABLE_SUFFIXES
