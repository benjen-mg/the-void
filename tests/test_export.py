"""Export writers: POSCAR, CIF, and Quantum ESPRESSO verified by
round-trip (write, then re-parse with the corresponding reader, and
confirm the structure comes back out correctly - the strongest
available check). XYZ, Gaussian, and CP2K have no corresponding reader
available here, so they are checked by direct, exact content
inspection instead - every cell vector and every atomic position
value confirmed present in the output, not merely "the file looks
plausible".
"""

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.export import (  # noqa: E402
    CP2K_DEFAULT_BASIS,
    CP2K_DEFAULT_POTENTIAL,
    export_cif,
    export_cp2k,
    export_gaussian,
    export_poscar,
    export_quantum_espresso,
    export_xyz,
)


@pytest.fixture
def triclinic_structure():
    """A deliberately low-symmetry (triclinic, three distinct
    elements, non-trivial fractional coordinates) structure - any
    format that only happens to work for a high-symmetry cubic test
    case would not be a meaningful check.
    """
    lattice = Lattice.from_parameters(5.64, 6.10, 7.30, 80, 95, 105)
    return Structure(
        lattice,
        ["Zn", "O", "C"],
        [[0.1, 0.2, 0.3], [0.5, 0.5, 0.5], [0.9, 0.8, 0.7]],
    )


def _as_unordered_set(structure, decimals=6):
    """(symbol, rounded fractional coordinates) pairs, order-
    independent - CIF round-trips are known to reorder sites
    alphabetically by element rather than preserving file order
    (confirmed directly against this exact structure before writing
    this helper), so an order-sensitive comparison would fail for
    reasons that have nothing to do with whether the data survived.
    """
    return {
        (site.specie.symbol, tuple(np.round(site.frac_coords, decimals)))
        for site in structure.sites
    }


# --- POSCAR ---


def test_poscar_round_trips_lattice_and_positions_exactly(
    triclinic_structure,
):
    from pymatgen.io.vasp import Poscar

    text = export_poscar(triclinic_structure)
    reparsed = Poscar.from_str(text).structure

    assert np.allclose(
        reparsed.lattice.matrix, triclinic_structure.lattice.matrix
    )
    assert np.allclose(
        reparsed.frac_coords, triclinic_structure.frac_coords, atol=1e-6
    )


# --- CIF ---


def test_cif_round_trips_lattice_and_positions(triclinic_structure):
    text = export_cif(triclinic_structure)
    reparsed = Structure.from_str(text, fmt="cif")

    assert np.allclose(
        reparsed.lattice.matrix,
        triclinic_structure.lattice.matrix,
        atol=1e-4,
    )
    assert _as_unordered_set(reparsed) == _as_unordered_set(
        triclinic_structure
    )


# --- XYZ (extended) ---


def test_xyz_atom_count_and_header(triclinic_structure):
    text = export_xyz(triclinic_structure)
    lines = text.splitlines()

    assert lines[0] == "3"
    assert "Lattice=" in lines[1]
    assert 'pbc="T T T"' in lines[1]


def test_xyz_lattice_values_are_exact(triclinic_structure):
    text = export_xyz(triclinic_structure)
    header = text.splitlines()[1]

    lattice_str = header.split('Lattice="')[1].split('"')[0]
    values = [float(v) for v in lattice_str.split()]
    lattice_from_file = np.array(values).reshape(3, 3)

    assert np.allclose(lattice_from_file, triclinic_structure.lattice.matrix)


def test_xyz_atom_lines_have_correct_cartesian_coordinates(
    triclinic_structure,
):
    text = export_xyz(triclinic_structure)
    atom_lines = text.splitlines()[2:]

    assert len(atom_lines) == 3

    for line, site in zip(atom_lines, triclinic_structure.sites):
        parts = line.split()
        assert parts[0] == site.specie.symbol
        coords = np.array([float(v) for v in parts[1:4]])
        assert np.allclose(coords, site.coords)


# --- Gaussian ---


def test_gaussian_route_line_reflects_requested_level_of_theory(
    triclinic_structure,
):
    text = export_gaussian(
        triclinic_structure, functional="B3LYP", basis_set="6-311G(d,p)"
    )

    assert "B3LYP/6-311G(d,p)" in text.splitlines()[0]


def test_gaussian_route_line_declares_3d_periodic_boundary_conditions(
    triclinic_structure,
):
    text = export_gaussian(triclinic_structure)

    assert "PBC=3D" in text.splitlines()[0]


def test_gaussian_charge_and_multiplicity_line(triclinic_structure):
    text = export_gaussian(triclinic_structure, charge=-1, multiplicity=2)

    assert "-1 2" in text.splitlines()


def test_gaussian_atom_block_has_correct_cartesian_positions(
    triclinic_structure,
):
    text = export_gaussian(triclinic_structure)
    lines = text.splitlines()

    charge_line_index = lines.index("0 1")
    atom_lines = lines[charge_line_index + 1: charge_line_index + 4]

    for line, site in zip(atom_lines, triclinic_structure.sites):
        parts = line.split()
        assert parts[0] == site.specie.symbol
        coords = np.array([float(v) for v in parts[1:4]])
        assert np.allclose(coords, site.coords)


def test_gaussian_has_exactly_three_tv_lines_matching_the_lattice(
    triclinic_structure,
):
    """Three TV pseudo-atoms, one per lattice vector - Gaussian's own
    real mechanism for expressing periodicity, each TV line's
    "coordinates" being the exact Cartesian components of that lattice
    vector.
    """
    text = export_gaussian(triclinic_structure)

    tv_lines = [
        line for line in text.splitlines() if line.strip().startswith("TV")
    ]

    assert len(tv_lines) == 3

    tv_vectors = np.array(
        [[float(v) for v in line.split()[1:4]] for line in tv_lines]
    )
    assert np.allclose(tv_vectors, triclinic_structure.lattice.matrix)


# --- CP2K ---


def test_cp2k_cell_vectors_are_exact(triclinic_structure):
    text = export_cp2k(triclinic_structure)

    cell_block = text.split("&CELL")[1].split("&END CELL")[0]
    cell_lines = {
        line.strip().split()[0]: line
        for line in cell_block.splitlines()
        if line.strip() and line.strip().split()[0] in ("A", "B", "C")
    }

    assert set(cell_lines.keys()) == {"A", "B", "C"}

    parsed_matrix = np.array(
        [
            [float(v) for v in cell_lines[label].split()[1:4]]
            for label in "ABC"
        ]
    )
    assert np.allclose(parsed_matrix, triclinic_structure.lattice.matrix)


def test_cp2k_coord_block_uses_scaled_fractional_coordinates(
    triclinic_structure,
):
    text = export_cp2k(triclinic_structure)

    assert "SCALED_COORDS TRUE" in text

    coord_block = text.split("&COORD")[1].split("&END COORD")[0]
    lines = [line for line in coord_block.splitlines() if line.strip()]
    # First line is SCALED_COORDS TRUE, the rest are atoms.
    atom_lines = lines[1:]

    assert len(atom_lines) == 3

    for line, site in zip(atom_lines, triclinic_structure.sites):
        parts = line.split()
        assert parts[0] == site.specie.symbol
        coords = np.array([float(v) for v in parts[1:4]])
        assert np.allclose(coords, site.frac_coords)


def test_cp2k_has_one_kind_block_per_unique_element(triclinic_structure):
    text = export_cp2k(triclinic_structure)

    for symbol in ("Zn", "O", "C"):
        assert f"&KIND {symbol}" in text

    assert text.count("&KIND") == 3
    assert CP2K_DEFAULT_BASIS in text
    assert CP2K_DEFAULT_POTENTIAL in text


def test_cp2k_brackets_are_balanced(triclinic_structure):
    """A basic structural-validity check: every &SECTION has a
    matching &END SECTION.
    """
    text = export_cp2k(triclinic_structure)

    opens = text.count("&") - text.count("&END")
    ends = text.count("&END")

    assert opens == ends


# --- Quantum ESPRESSO ---


def test_qe_round_trips_lattice_and_positions(triclinic_structure):
    from pymatgen.io.pwscf import PWInput

    text = export_quantum_espresso(triclinic_structure)
    reparsed = PWInput.from_str(text)

    assert np.allclose(
        reparsed.structure.lattice.matrix,
        triclinic_structure.lattice.matrix,
        atol=1e-5,
    )
    assert np.allclose(
        reparsed.structure.frac_coords,
        triclinic_structure.frac_coords,
        atol=1e-5,
    )


def test_qe_placeholder_pseudopotentials_are_labelled(triclinic_structure):
    text = export_quantum_espresso(triclinic_structure)

    assert "placeholder" in text.lower()


def test_qe_custom_pseudopotentials_suppress_the_placeholder_notice(
    triclinic_structure,
):
    text = export_quantum_espresso(
        triclinic_structure,
        pseudopotentials={
            "Zn": "Zn.real.UPF", "O": "O.real.UPF", "C": "C.real.UPF",
        },
    )

    assert "placeholder" not in text.lower()
    assert "Zn.real.UPF" in text


def test_qe_ecutwfc_and_kpoints_are_reflected(triclinic_structure):
    text = export_quantum_espresso(
        triclinic_structure, ecutwfc=60.0, kpoints_grid=(6, 6, 6)
    )

    assert "ecutwfc = 60" in text
    assert "6 6 6" in text
