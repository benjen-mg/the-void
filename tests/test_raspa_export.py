"""Round 52: the RASPA2 export (services/raspa_export.py).

The numbers here decide what a GCMC run simulates, so each is checked
against something independent rather than against itself: the UFF
table against RDKit's own UFF implementation and against the values
RASPA ships; unit-cell counts against perpendicular widths worked out
by hand; the framework/guest split against structures built so the
right answer is known; pocket blocking against a cell whose single
sealed cage is known exactly. The last test hands an export to a real
RASPA2 binary, when one is installed.

The structure is the open Zn-dia sample without its charges
(tests/fixtures/open/), so the export is the uncharged case.
"""

import json
import math
import os
import shutil
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.metal_classification import is_metal  # noqa: E402
from thevoid_core.services.bonds.covalent_bond_detector import (  # noqa: E402
    CovalentBondDetector,
)
from thevoid_core.services.cif.pymatgen_cif_loader import (  # noqa: E402
    PymatgenCIFLoader,
)
from thevoid_core.services.mof_analysis import bonded_components  # noqa: E402
from tests.sample_structures import (  # noqa: E402
    ZN_DIA_UNCHARGED,
)
from thevoid_core.services.raspa_export import (  # noqa: E402
    ADSORBATES,
    KCAL_PER_MOL_IN_K,
    MANIFEST_NAME,
    UFF_VDW,
    RaspaExportSettings,
    build_raspa_export,
    compute_blocking,
    existing_export_files,
    prepare_framework,
    read_cif_charges,
    summary_for_report,
    uff_lennard_jones,
    unit_cells_for_cutoff,
    write_raspa_export,
)

FIXTURE = ZN_DIA_UNCHARGED
NAME = FIXTURE.stem

warnings.filterwarnings("ignore", message="No Pauling electronegativity")


def _sample():
    return PymatgenCIFLoader().load(str(FIXTURE)).structure


def _wire_framework_with_guests(coordinated_water=False):
    """A 3-periodic carbon 'pcu' wire framework (C every 1.5 A along the
    edges of a 6 A cube, so each wire bonds to its own periodic image)
    with a free water molecule in the middle of the cage and an isolated
    Na in a face - two things that are not framework, of two kinds."""
    lattice = Lattice.cubic(6.0)
    species, coords = [], []

    for t in (0.0, 0.25, 0.5, 0.75):
        for point in ((t, 0, 0), (0, t, 0), (0, 0, t)):
            if point not in coords:
                species.append("C")
                coords.append(point)

    # free water at the cage centre
    species += ["O", "H", "H"]
    coords += [(0.5, 0.5, 0.5), (0.5 + 0.7572 / 6, 0.5 + 0.586 / 6, 0.5),
               (0.5 - 0.7572 / 6, 0.5 + 0.586 / 6, 0.5)]
    # isolated sodium in a face centre
    species.append("Na")
    coords.append((0.5, 0.5, 0.0))

    if coordinated_water:
        # an O 1.43 A from the C at (0, 0, 0.25): bonded, so framework
        # (and 3.7 A from the Na, so it does not capture the ion)
        species.append("O")
        coords.append((0.0, 1.43 / 6, 0.25))

    return Structure(lattice, species, coords)


def _cage_and_slit():
    """Ar atoms on a square lattice (5 A) in two layers 5 A apart,
    stacked with a 9 A gap: the two layers enclose one cubic cage per
    cell, and the gap is an open 2D slit. For a 1.90 A probe (CH4, the
    application's own value) with Bondi's 1.88 A for Ar, the cage centre
    has clearance 5*sqrt(3)/2 - 1.88 = 2.450 A and every window
    5/sqrt(2) - 1.88 = 1.656 A: the cage is a sealed pocket and the
    slit percolates."""
    return Structure(
        Lattice.tetragonal(5.0, 14.0), ["Ar", "Ar"], [[0, 0, 0], [0, 0, 5 / 14]]
    )


# ----------------------------------------------------------------------
# Force field
# ----------------------------------------------------------------------


def test_uff_table_matches_rdkit_for_every_element_rdkit_can_type():
    """The table was copied from RDKit's compiled UFF parameters;
    this re-derives every value RDKit will type from a real molecule, so
    a transcription slip in any row fails here."""
    Chem = pytest.importorskip("rdkit.Chem")
    from rdkit import RDLogger
    from rdkit.Chem import rdForceFieldHelpers

    RDLogger.DisableLog("rdApp.*")
    checked = 0

    for symbol, (x_i, d_i) in UFF_VDW.items():
        for smiles in (f"[{symbol}]", f"[{symbol}+2]", f"[{symbol}H2]",
                       f"[{symbol}H3]", f"[{symbol}H4]",
                       f"[{symbol}](F)(F)(F)(F)(F)F"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            try:
                params = rdForceFieldHelpers.GetUFFVdWParams(
                    Chem.AddHs(mol), 0, 0
                )
            except Exception:
                params = None
            if params:
                assert params == pytest.approx((x_i, d_i)), symbol
                checked += 1
                break

    # RDKit can type most of the table from simple molecules; the rest
    # (Zr, Ni, Hf, noble gases...) are pinned by the next test.
    assert checked >= 70


def test_uff_conversion_reproduces_raspas_own_uff_values():
    """RASPA2's bundled GenericMOFs force field uses UFF for metals;
    its Zn, Zr and Ni rows are eps/k 62.3992 K / sigma 2.46155 A,
    34.7221 / 2.78317 and 7.54829 / 2.52481. Same conversion (D_i in
    kcal/mol over R; x_i / 2^(1/6)), same numbers."""
    for symbol, eps, sigma in (("Zn", 62.3992, 2.46155),
                               ("Zr", 34.7221, 2.78317),
                               ("Ni", 7.54829, 2.52481)):
        got_eps, got_sigma = uff_lennard_jones(symbol)
        assert got_eps == pytest.approx(eps, abs=2e-3)
        assert got_sigma == pytest.approx(sigma, abs=1e-5)

    assert KCAL_PER_MOL_IN_K == pytest.approx(503.2195, abs=1e-3)


def test_every_element_has_parameters_and_unknowns_are_refused():
    from pymatgen.core import Element

    for z in range(1, 104):
        assert Element.from_Z(z).symbol in UFF_VDW
    assert uff_lennard_jones("D") == uff_lennard_jones("H")
    with pytest.raises(ValueError):
        uff_lennard_jones("X")


def test_adsorbate_models_are_trappe_and_centred():
    """TraPPE values (Potoff & Siepmann 2001; Martin & Siepmann 1998),
    and bead 0 at the molecular centre - RASPA tests blocking on bead 0,
    so a CO2 whose oxygen came first would be blocked 1.16 A off the
    point the pocket classification is about."""
    co2 = {a.name: a for a in ADSORBATES["CO2"].pseudo_atoms}
    assert (co2["C_co2"].epsilon_K, co2["C_co2"].sigma_A, co2["C_co2"].charge) == (27.0, 2.80, 0.70)
    assert (co2["O_co2"].epsilon_K, co2["O_co2"].sigma_A, co2["O_co2"].charge) == (79.0, 3.05, -0.35)
    n2 = {a.name: a for a in ADSORBATES["N2"].pseudo_atoms}
    assert (n2["N_n2"].epsilon_K, n2["N_n2"].sigma_A, n2["N_n2"].charge) == (36.0, 3.31, -0.482)
    assert n2["N_com"].epsilon_K is None and n2["N_com"].charge == 0.964
    ch4 = ADSORBATES["CH4"].pseudo_atoms[0]
    assert (ch4.epsilon_K, ch4.sigma_A, ch4.charge) == (148.0, 3.73, 0.0)

    for model in ADSORBATES.values():
        charges = {a.name: a.charge for a in model.pseudo_atoms}
        assert sum(charges[name] for name, *_ in model.atoms) == pytest.approx(0.0)
        assert model.atoms[0][1:] == (0.0, 0.0, 0.0)
        centre = np.mean([xyz for _, *xyz in model.atoms], axis=0)
        assert np.allclose(centre, 0.0)


# ----------------------------------------------------------------------
# Unit cells
# ----------------------------------------------------------------------


def test_unit_cells_for_a_monoclinic_cell_match_hand_computed_widths():
    """A monoclinic cell (a 8.914, b 9.694, c 9.484 A, beta 115.9):
    widths 8.019, 9.694 and 8.531 A.
    2 x 12.8 / 8.531 = 3.0009, so three cells along c would be 0.008 A
    too thin and RASPA would refuse; four are needed."""
    lattice = Lattice.from_parameters(8.9138, 9.6935, 9.4836, 90, 115.895, 90)
    beta = math.radians(lattice.beta)
    widths = (lattice.a * math.sin(beta), lattice.b, lattice.c * math.sin(beta))
    assert widths == pytest.approx((8.019, 9.6935, 8.531), abs=1e-3)
    assert unit_cells_for_cutoff(lattice.matrix, 12.8) == (4, 3, 4)


def test_unit_cells_for_the_sample():
    """Orthorhombic, so the widths are the edges: 9.881, 14.739 and
    14.786 A. 25.6 / 9.881 = 2.59 and 25.6 / 14.739 = 1.74: 3 x 2 x 2."""
    lattice = _sample().lattice
    assert lattice.angles == pytest.approx((90, 90, 90))
    assert lattice.abc == pytest.approx((9.8812, 14.7391, 14.7855), abs=1e-4)
    assert unit_cells_for_cutoff(lattice.matrix, 12.8) == (3, 2, 2)


def test_unit_cells_are_strictly_wider_than_twice_the_cutoff():
    """Exact equality is not enough for the minimum-image convention."""
    assert unit_cells_for_cutoff(Lattice.cubic(12.8).matrix, 12.8) == (3, 3, 3)
    assert unit_cells_for_cutoff(Lattice.cubic(25.61).matrix, 12.8) == (1, 1, 1)
    # a strongly sheared cell needs more copies than its edge suggests
    sheared = Lattice.from_parameters(20, 20, 20, 90, 90, 30).matrix
    assert unit_cells_for_cutoff(sheared, 12.8) == (3, 3, 2)


# ----------------------------------------------------------------------
# Which atoms are framework
# ----------------------------------------------------------------------


def test_bonded_components_find_the_framework_and_the_guests():
    structure = _wire_framework_with_guests()
    bonds = CovalentBondDetector().detect(structure)
    components = bonded_components(structure, bonds, is_metal)
    summary = sorted((c.formula, c.dimensionality) for c in components)

    assert summary == [("C10", 3), ("H2O", 0), ("Na", 0)]
    by_formula = {c.formula: c for c in components}
    assert by_formula["Na"].possibly_ionic
    assert not by_formula["H2O"].possibly_ionic  # 10 electrons, closed shell


def test_the_sample_is_one_three_periodic_framework():
    prepared = prepare_framework(_sample())
    assert prepared.framework_dimensionality == 3
    assert prepared.removed == []
    assert prepared.disorder_sites_dropped == 0
    assert len(prepared.structure) == 124


def test_free_molecules_are_removed_and_listed_but_coordinated_ones_kept():
    """Solvent bonded to the framework is framework; solvent that is not
    is removed by default, grouped by formula, with the ion flagged."""
    structure = _wire_framework_with_guests(coordinated_water=True)
    prepared = prepare_framework(structure)

    removed = {m.formula: m for m in prepared.removed}
    assert set(removed) == {"H2O", "Na"}
    assert removed["Na"].possibly_ionic and not removed["H2O"].possibly_ionic
    assert prepared.structure.composition.formula.replace(" ", "") == "C10O1"


def test_keeping_free_molecules_is_honoured():
    structure = _wire_framework_with_guests()
    prepared = prepare_framework(structure, remove_free_molecules=False)
    assert prepared.removed == []
    assert len(prepared.structure) == len(structure)
    assert any("kept at your request" in n for n in prepared.notes)


def test_nothing_is_removed_when_nothing_is_periodic():
    """A molecular crystal has no framework; removing every 'free'
    molecule would export an empty box."""
    prepared = prepare_framework(_cage_and_slit())
    assert prepared.removed == []
    assert len(prepared.structure) == 2
    assert prepared.framework_dimensionality == 0


def test_an_even_split_disorder_cannot_be_exported():
    structure = Structure(
        Lattice.cubic(5.0), [{"Na": 0.5, "K": 0.5}], [[0, 0, 0]]
    )
    with pytest.raises(ValueError, match="RASPA cannot represent"):
        prepare_framework(structure)


# ----------------------------------------------------------------------
# Charges
# ----------------------------------------------------------------------


def _charged_cif(tmp_path, charges_text=None):
    structure = prepare_framework(_sample()).structure
    q = {"Zn": 0.9, "H": 0.1, "O": -0.5, "N": -0.2, "C": 2.0 / 12}
    lat = structure.lattice
    lines = [
        "data_charged",
        f"_cell_length_a {lat.a}", f"_cell_length_b {lat.b}",
        f"_cell_length_c {lat.c}", f"_cell_angle_alpha {lat.alpha}",
        f"_cell_angle_beta {lat.beta}", f"_cell_angle_gamma {lat.gamma}",
        "_symmetry_space_group_name_H-M 'P 1'",
        "loop_", "_symmetry_equiv_pos_as_xyz", "'x, y, z'",
        "loop_", "_atom_site_label", "_atom_site_type_symbol",
        "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
        "_atom_site_charge",
    ]
    for i, site in enumerate(structure.sites):
        el = site.specie.symbol
        charge = charges_text(i, el) if charges_text else f"{q[el]:.6f}"
        x, y, z = (v % 1 for v in site.frac_coords)
        lines.append(f"{el}{i + 1} {el} {x:.6f} {y:.6f} {z:.6f} {charge}")
    path = tmp_path / "charged.cif"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_cif_charges_are_read_and_carried_into_the_export(tmp_path):
    path = _charged_cif(tmp_path)
    charges = read_cif_charges(path)
    assert len(charges) == 124
    assert charges["Zn1"] == pytest.approx(0.9)

    structure = PymatgenCIFLoader().load(str(path)).structure
    export = build_raspa_export(structure, settings=_fast(), source=path)

    assert export.summary["charges"]["source"] == "cif"
    assert "UseChargesFromCIFFile         yes" in export.files["simulation.input"]
    cif = export.files["charged.cif"]
    assert "_atom_site_charge" in cif
    written = [float(line.split()[-1]) for line in cif.splitlines()
               if line.startswith("Zn ")]
    assert written == pytest.approx([0.9] * 4)
    assert not any("UNCHARGED" in w for w in export.warnings)


def test_incomplete_charges_are_not_used(tmp_path):
    """Half a set of charges is worse than none."""
    path = _charged_cif(tmp_path, lambda i, el: "?" if i == 3 else "0.0")
    assert read_cif_charges(path) is None
    assert read_cif_charges(FIXTURE) is None
    assert read_cif_charges(None) is None


def test_an_uncharged_framework_is_flagged_only_for_charged_adsorbates():
    structure = _sample()
    co2 = build_raspa_export(structure, settings=_fast(), source=FIXTURE)
    ch4 = build_raspa_export(structure, settings=_fast(adsorbate="CH4"), source=FIXTURE)

    assert any("UNCHARGED FRAMEWORK" in w for w in co2.warnings)
    assert "UNCHARGED FRAMEWORK" in co2.files["simulation.input"]
    assert "UNCHARGED FRAMEWORK" in co2.files["README.txt"]
    assert not any("UNCHARGED" in w for w in ch4.warnings)
    assert "ChargeMethod                  None" in ch4.files["simulation.input"]


def test_charge_equilibration_is_opt_in_and_runs_as_its_own_step():
    """RASPA's in-GCMC charge equilibration spans the whole simulation
    box and had not finished for a 4 x 3 x 4 box of a small Zn MOF after 25
    minutes; on one unit cell it takes seconds and gives the same
    charges. So it is a separate first step, and must not average the
    charges per element (every atom of an element shares a pseudo-atom
    type in this export)."""
    export = build_raspa_export(
        _sample(), settings=_fast(charge_equilibration=True), source=FIXTURE
    )
    stage1 = export.files["charges/simulation.input"]
    assert "ChargeFromChargeEquilibration yes" in stage1
    assert "SymmetrizeFrameworkCharges    no" in stage1
    assert "UnitCells                     1 1 1" in stage1
    assert "NumberOfCycles                0" in stage1
    assert "Component" not in stage1

    main = export.files["simulation.input"]
    assert "ChargeEquilibration" not in main
    assert f"FrameworkName                 {NAME}_qeq" in main
    assert "UseChargesFromCIFFile         yes" in main

    script = export.files["run_raspa.sh"]
    assert f"Framework_0_final_1_1_1_P1.cif {NAME}_qeq.cif" in script
    assert export.files[f"charges/{NAME}.cif"] == export.files[f"{NAME}.cif"]
    assert export.summary["charges"]["source"] == "charge_equilibration"
    assert "run_raspa.sh" in export.files["README.txt"]

    plain = build_raspa_export(_sample(), settings=_fast(), source=FIXTURE)
    assert not any(name.startswith("charges/") for name in plain.files)
    assert "run_raspa.sh" not in plain.files


def test_charge_equilibration_is_skipped_for_an_uncharged_adsorbate():
    export = build_raspa_export(
        _sample(), settings=_fast(adsorbate="CH4", charge_equilibration=True),
        source=FIXTURE,
    )
    assert export.summary["charges"]["source"] == "none"
    assert "run_raspa.sh" not in export.files
    assert any("skipped" in note for note in export.summary["notes"])


def test_the_run_script_is_executable(tmp_path):
    export = build_raspa_export(
        _sample(), settings=_fast(charge_equilibration=True), source=FIXTURE
    )
    write_raspa_export(export, tmp_path)
    assert os.access(tmp_path / "run_raspa.sh", os.X_OK)
    assert (tmp_path / "charges" / "simulation.input").is_file()


# ----------------------------------------------------------------------
# Pocket blocking
# ----------------------------------------------------------------------


def test_the_sealed_cage_gets_one_sphere_at_its_centre():
    blocking = compute_blocking(_cage_and_slit(), 1.90)

    assert blocking.status == "blocked"
    assert blocking.sphere_count == 1
    assert blocking.accessible_dimensionality == 2
    x, y, z, radius = blocking.spheres[0]
    centre = np.array([0.5, 0.5, 2.5 / 14])
    offset = (np.array([x, y, z]) - centre) @ _cage_and_slit().lattice.matrix
    # within a grid spacing of the true centre; radius = clearance there
    assert np.linalg.norm(offset) < 0.3
    assert radius == pytest.approx(5 * math.sqrt(3) / 2 - 1.88, abs=0.05)
    # and it stops short of the window plane into the slit (z = 5 A)
    assert z * 14 + radius < 5.0


def test_blocking_covers_every_pocket_node_and_no_accessible_node():
    """The two properties that make blocking safe: nothing sealed is
    left insertable, and nothing reachable is forbidden."""
    from thevoid_core.data.element_radii import ElementRadiusProvider
    from thevoid_core.services.pore_analysis import (
        classify_void_grid,
        grid_shape_for,
    )

    structure = _cage_and_slit()
    blocking = compute_blocking(structure, 1.90)
    radii = np.array([ElementRadiusProvider().get_vdw_radius("Ar")] * 2)
    grid = classify_void_grid(
        structure, radii, probe_radius=1.90, grid_shape=grid_shape_for(structure)
    )
    lattice = structure.lattice.matrix

    def inside_any(frac_points):
        hit = np.zeros(len(frac_points), dtype=bool)
        for *centre, radius in blocking.spheres:
            delta = frac_points - np.array(centre)
            delta -= np.round(delta)
            hit |= np.linalg.norm(delta @ lattice, axis=1) < radius
        return hit

    assert inside_any(grid.grid_frac[grid.pocket_mask]).all()
    assert not inside_any(grid.grid_frac[grid.sweep.accessible]).any()


def _assert_blocking_matches_pore_analysis(structure, probes):
    from thevoid_core.data.element_radii import ElementRadiusProvider
    from thevoid_core.services.pore_analysis import analyze_pores

    for probe in probes:
        blocking = compute_blocking(structure, probe)
        pores = analyze_pores(structure, probe_radius=probe,
                              radius_provider=ElementRadiusProvider())
        assert blocking.accessible_volume_A3 == pytest.approx(pores.accessible_volume)
        assert blocking.pocket_volume_A3 == pytest.approx(pores.non_accessible_volume)
        assert blocking.accessible_dimensionality == pores.accessible_dimensionality


def test_blocking_classifies_void_exactly_as_pore_analysis_does():
    """Same grid code, same answer. On the sample a 1.3 A probe finds no
    sealed void; 1.6 A leaves a sliver sealed (0.22 A^3) and still
    percolates in 3D; at 1.9 A the channels are 1D. Pore Analysis
    reports the same for each probe."""
    structure = _sample()
    _assert_blocking_matches_pore_analysis(structure, (1.3, 1.6, 1.9))
    assert compute_blocking(structure, 1.3).status == "no pockets"
    assert compute_blocking(structure, 1.6).status == "blocked"
    assert compute_blocking(structure, 1.9).accessible_dimensionality == 1


def test_a_fully_open_structure_has_no_pockets():
    open_cell = Structure(Lattice.cubic(12.0), ["Ar"], [[0, 0, 0]])
    assert compute_blocking(open_cell, 1.9).status == "no pockets"


def test_block_file_and_input_agree(tmp_path):
    export = build_raspa_export(
        _cage_and_slit(), settings=_fast(adsorbate="CH4", framework_name="cage")
    )
    block = export.files["cage.block"].splitlines()
    assert block[0] == "1"
    assert len(block[1].split()) == 4
    text = export.files["simulation.input"]
    assert "BlockPockets             yes" in text
    assert "BlockPocketsFileName     cage" in text


# ----------------------------------------------------------------------
# Files and records
# ----------------------------------------------------------------------


def _fast(**overrides):
    defaults = dict(cycles=10, initialization_cycles=10, print_every=5)
    defaults.update(overrides)
    return RaspaExportSettings(**defaults)


def test_input_set_is_complete_and_self_consistent(tmp_path):
    export = build_raspa_export(_sample(), settings=_fast(pressures_Pa=(1e4, 1e5)), source=FIXTURE)
    names = set(export.files)
    assert names == {f"{NAME}.cif", "pseudo_atoms.def", "force_field_mixing_rules.def",
                     "force_field.def", "CO2.def", "simulation.input",
                     "README.txt", MANIFEST_NAME}

    text = export.files["simulation.input"]
    assert "UnitCells                     3 2 2" in text
    assert "ExternalPressure              10000 100000" in text
    assert f"FrameworkName                 {NAME}" in text
    assert "Forcefield                    Local" in text
    assert "MoleculeDefinition       Local" in text

    # every pseudo-atom the CIF and the molecule use is defined, with an
    # exact name - a trailing '_' would be a RASPA prefix wildcard
    pseudo = export.files["pseudo_atoms.def"].splitlines()
    defined = {line.split()[0] for line in pseudo[3:]}
    mixing = export.files["force_field_mixing_rules.def"].splitlines()
    rules = {line.split()[0] for line in mixing[7:-2]}
    assert defined == rules == {"Zn", "H", "C", "N", "O", "C_co2", "O_co2"}
    assert not any(name.endswith("_") for name in rules)
    assert int(pseudo[1]) == len(defined) and int(mixing[5]) == len(rules)

    labels = {line.split()[0] for line in export.files[f"{NAME}.cif"].splitlines()
              if line[:2].strip() in {"Zn", "H", "C", "N", "O"} and len(line.split()) >= 5}
    assert labels <= defined

    manifest = json.loads(export.files[MANIFEST_NAME])
    assert manifest["input"]["sha256"] and manifest["unit_cells"] == [3, 2, 2]
    assert manifest["force_field"]["framework_parameters"]["Zn"]["epsilon_K"] == pytest.approx(62.3992, abs=2e-3)

    written = write_raspa_export(export, tmp_path / "set")
    assert len(written) == len(export.files)
    assert existing_export_files(export, tmp_path / "set") == sorted(export.files)
    assert existing_export_files(export, tmp_path / "empty") == []


def test_ch4_has_no_rotation_move_and_linear_molecules_do():
    ch4 = build_raspa_export(_sample(), settings=_fast(adsorbate="CH4"), source=FIXTURE)
    n2 = build_raspa_export(_sample(), settings=_fast(adsorbate="N2"), source=FIXTURE)
    assert "RotationProbability" not in ch4.files["simulation.input"]
    assert "RotationProbability" in n2.files["simulation.input"]
    assert "0 N_com" in n2.files["N2.def"]


def test_readme_states_the_assumptions():
    readme = build_raspa_export(_sample(), settings=_fast(), source=FIXTURE).files["README.txt"]
    for phrase in ("STARTING POINT", "UFF", "TraPPE", "Lorentz-Berthelot",
                   "3 x 2 x 2", "124 atoms", "SHA-256",
                   "Helium void fraction"):
        assert phrase in readme, phrase
    assert "minor-disorder" not in readme


def test_bad_settings_are_refused():
    for bad in (dict(adsorbate="Kr"), dict(pressures_Pa=()), dict(pressures_Pa=(-1.0,)),
                dict(temperature_K=0), dict(cycles=0), dict(blocking_probe_radius_A=0)):
        with pytest.raises(ValueError):
            build_raspa_export(_sample(), settings=_fast(**bad))


def test_report_summary_is_json_safe():
    export = build_raspa_export(_sample(), settings=_fast(), source=FIXTURE)
    summary = summary_for_report(export, f"/somewhere/{NAME}_RASPA_CO2")
    json.dumps(summary, allow_nan=False)
    assert summary["unit_cells"] == [3, 2, 2]
    assert summary["charges"] == "none"


# ----------------------------------------------------------------------
# The real thing
# ----------------------------------------------------------------------


def _raspa_binary():
    candidate = os.environ.get("RASPA_SIMULATE") or shutil.which("simulate")
    return candidate if candidate and Path(candidate).exists() else None


@pytest.mark.skipif(_raspa_binary() is None,
                    reason="RASPA2 not installed (set RASPA_SIMULATE)")
def test_raspa2_reads_and_runs_the_export(tmp_path):
    """Everything above checks the files against what they should say;
    this checks RASPA agrees. It reads the local force field (not its
    share directory), builds the 3 x 2 x 2 box, mixes Zn-O_co2 by
    Lorentz-Berthelot and finishes without warnings."""
    export = build_raspa_export(
        _sample(), settings=_fast(cycles=5, initialization_cycles=0), source=FIXTURE
    )
    folder = tmp_path / "set"
    write_raspa_export(export, folder)
    env = dict(os.environ, RASPA_DIR=str(tmp_path / "no_share"))
    subprocess.run([_raspa_binary(), "simulation.input"], cwd=folder, env=env,
                   check=True, capture_output=True, timeout=600)

    output = next((folder / "Output" / "System_0").glob("*.data")).read_text()
    assert "Number of unitcells [a]: 3" in output
    assert "Number of unitcells [b]: 2" in output
    assert "Number of unitcells [c]: 2" in output
    assert "Number of framework atoms in the unit cell: 124" in output
    assert "Zn -   O_co2 [LENNARD_JONES] p_0/k_B:  70.21066 [K], p_1: 2.75577 [A]" in output
    assert "Simulation finished,  0 warnings" in output


def _run_charge_equilibration(tmp_path, structure, source):
    """Run the two-step export; return the per-element sets of charges
    step 1 wrote and step 2's output."""
    export = build_raspa_export(
        structure,
        settings=_fast(cycles=5, initialization_cycles=0, charge_equilibration=True),
        source=source,
    )
    folder = tmp_path / "set"
    write_raspa_export(export, folder)
    env = dict(os.environ, RASPA_DIR=str(tmp_path / "no_share"),
               SIMULATE=_raspa_binary())
    subprocess.run(["./run_raspa.sh"], cwd=folder, env=env, check=True,
                   capture_output=True, timeout=900)

    charges = {}
    in_loop = False
    for line in (folder / f"{Path(source).stem}_qeq.cif").read_text().splitlines():
        if "_atom_site_charge" in line:
            in_loop = True
            continue
        parts = line.split()
        if in_loop and len(parts) >= 6:
            charges.setdefault(parts[0], set()).add(round(float(parts[5]), 4))
    output = next((folder / "Output" / "System_0").glob("*.data")).read_text()
    return charges, output


@pytest.mark.skipif(_raspa_binary() is None,
                    reason="RASPA2 not installed (set RASPA_SIMULATE)")
def test_raspa2_runs_the_charge_equilibration_route(tmp_path):
    """Step 1 writes per-atom charges, step 2 reads them. The sample has
    four symmetry-distinct oxygens (the four oxygen charges of its CoRE
    file); they get four different charges, which the per-element
    averaging RASPA does by default would have erased."""
    charges, output = _run_charge_equilibration(tmp_path, _sample(), FIXTURE)
    assert len(charges["O"]) == 4
    assert sum(len(v) for v in charges.values()) > 5
    assert "Simulation finished,  0 warnings" in output


@pytest.mark.skipif(_raspa_binary() is None,
                    reason="RASPA2 not installed (set RASPA_SIMULATE)")
def test_raspa2_finds_and_applies_the_block_file(tmp_path):
    """RASPA silently ignores a block file it cannot find ("not found
    and therefore not used"), so check it reports the sphere, in the
    right place, replicated over the box. (Longer runs made while
    writing this showed the sphere removing the cage population and
    nothing in the slit: 6000 cycles of CH4 at 298 K, 10 bar gave
    0.0752 molecules/cell unblocked, 0.0590 blocked, and 0.0151 with
    the block inverted to count only what sits inside the sphere.)"""
    export = build_raspa_export(
        _cage_and_slit(),
        settings=_fast(adsorbate="CH4", cycles=2, initialization_cycles=0,
                       framework_name="cage"),
    )
    folder = tmp_path / "set"
    write_raspa_export(export, folder)
    env = dict(os.environ, RASPA_DIR=str(tmp_path / "no_share"))
    subprocess.run([_raspa_binary(), "simulation.input"], cwd=folder, env=env,
                   check=True, capture_output=True, timeout=600)

    output = next((folder / "Output" / "System_0").glob("*.data")).read_text()
    cells = export.summary["unit_cells"]
    assert "Pockets are blocked for this component" in output
    assert f"Number of pockets blocked in a unitcell: {cells[0] * cells[1] * cells[2]}" in output
    x, y, z, radius = _first_sphere(export)
    assert f"Block center: {x * 5:.6f} {y * 5:.6f} {z * 14:.6f}  distance: {radius:.6f}" in output


def _first_sphere(export):
    line = export.files["cage.block"].splitlines()[1]
    return tuple(float(v) for v in line.split())
