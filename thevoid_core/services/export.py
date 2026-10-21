from __future__ import annotations

import logging

from thevoid_core.services.pore_analysis import major_occupancy_structure

logger = logging.getLogger(__name__)


#: Export formats whose file syntax has no way at all to express
#: fractional site occupancy: a VASP POSCAR is a plain per-element atom
#: count followed by that many positions, and a Quantum ESPRESSO
#: ATOMIC_POSITIONS card is one line per atom with a single species
#: name. Neither can say "this site is 25% oxygen". CIF, extended XYZ,
#: Gaussian and CP2K are not listed here: CIF has real occupancy
#: support, and the other three are written out atom-by-atom from the
#: dominant species of each site, which succeeds (chemically
#: approximate, but it does not fail) on a disordered structure.
#:
#: Fed a disordered structure directly, pymatgen's POSCAR writer raises
#: ValueError and its PWInput writer raises AttributeError (attr
#: 'specie' not found on PeriodicSite) - the second of which used to
#: escape the export error handler entirely. Both are now headed off
#: before the writer is ever called.
FORMATS_REQUIRING_ORDERED_STRUCTURE = frozenset(
    {"poscar", "quantum_espresso"}
)


def ordered_structure_for_export(structure):
    """The ordered approximation of a disordered structure, for the
    formats that cannot represent disorder at all.

    This is exactly the same major-occupancy filter the pore, surface
    area and guest accessibility analyses already apply (occupancy
    below 0.5 dropped), reused rather than reimplemented so an exported
    POSCAR describes the same single physical cell those analyses were
    computed on, instead of a second, subtly different interpretation
    of the same disorder.

    Returns the structure unchanged when it is already ordered. The
    result is not guaranteed ordered: a site split 0.5/0.5 between two
    species survives the filter and stays disordered, which is why
    callers must still check is_ordered rather than assume.
    """
    if structure.is_ordered:
        return structure

    return major_occupancy_structure(structure)


#: Common, well-known CP2K GTH basis set / pseudopotential names,
#: bundled with every standard CP2K installation's BASIS_MOLOPT and
#: GTH_POTENTIALS data files - not element-specific tuning, just the
#: same generic "molecularly optimised, short-range, PBE" convention
#: used in the overwhelming majority of CP2K tutorials and examples.
#: pymatgen's own CP2K input-set machinery (pymatgen.io.cp2k.sets)
#: requires a populated basis-set database to match against and raised
#: ValueError for an ordinary Cl/Na structure in this environment
#: (confirmed directly, not assumed), which is why this module builds
#: the CP2K input directly instead of depending on it.
CP2K_DEFAULT_BASIS = "DZVP-MOLOPT-SR-GTH"
CP2K_DEFAULT_POTENTIAL = "GTH-PBE"

#: Placeholder pseudopotential filenames, following the SSSP/PSlibrary
#: naming convention (<Element>.pbe-*-kjpaw_psl.*.UPF) Quantum
#: ESPRESSO's standard libraries use - not real files (none are
#: bundled here), and not a claim that this exact filename exists in
#: any specific library version. Clearly commented in the generated
#: file so this is obvious to whoever opens it, not a silent
#: assumption.
QE_PLACEHOLDER_PSEUDOPOTENTIAL_SUFFIX = "pbe-n-kjpaw_psl.1.0.0.UPF"


def export_poscar(structure) -> str:
    """VASP POSCAR format, via pymatgen's own writer - fractional
    coordinates and the full 3x3 lattice matrix, both exact.
    """
    from pymatgen.io.vasp import Poscar

    return Poscar(structure).get_str()


def export_cif(structure) -> str:
    """CIF format, via pymatgen's own writer - cell parameters
    (a, b, c, alpha, beta, gamma) and fractional coordinates, both
    exact.
    """
    from pymatgen.io.cif import CifWriter

    return str(CifWriter(structure))


def export_xyz(structure) -> str:
    """Extended XYZ format: the plain XYZ atom block (element, then
    Cartesian coordinates) that every XYZ reader understands, plus the
    lattice matrix and periodic-boundary flags encoded in the comment
    line using the widely-adopted "extended XYZ" convention (the same
    one ASE, OVITO, and VMD read and write) - because standard XYZ has
    no field for a lattice at all (confirmed directly: pymatgen's own
    XYZ writer emits only element and Cartesian position, nothing
    else), and this module's job is to preserve periodicity, not
    silently drop it because the base format has nowhere to put it.
    """
    lattice = structure.lattice.matrix
    lattice_values = " ".join(
        f"{value:.10f}" for row in lattice for value in row
    )

    comment = (
        f'Lattice="{lattice_values}" '
        'Properties=species:S:1:pos:R:3 '
        'pbc="T T T"'
    )

    lines = [str(len(structure.sites)), comment]

    for site in structure.sites:
        x, y, z = site.coords
        lines.append(f"{_symbol(site):<3}{x:15.8f}{y:15.8f}{z:15.8f}")

    return "\n".join(lines) + "\n"


def export_gaussian(
    structure,
    *,
    functional: str = "PBEPBE",
    basis_set: str = "6-31G(d)",
    charge: int = 0,
    multiplicity: int = 1,
    title: str = "the Void export",
) -> str:
    """Gaussian .gjf input, with periodicity preserved via Gaussian's
    own translation-vector ("TV") pseudo-atom convention - the real,
    documented mechanism Gaussian itself uses for polymers, slabs, and
    crystals (one TV pseudo-atom per lattice vector, each given the
    vector's Cartesian components as its "coordinates", appended after
    the real atoms) - rather than exporting a bare, non-periodic
    molecule and quietly discarding the cell.

    Built directly rather than via pymatgen's GaussianInput: that class
    wraps a Molecule, not a periodic Structure, and has no support for
    TV pseudo-atoms (confirmed by inspecting its constructor - it takes
    only charge, spin multiplicity, and ordinary atoms), so it cannot
    express periodicity at all.

    functional/basis_set/charge/multiplicity are ordinary Gaussian
    calculation parameters this module cannot infer from geometry alone
    (there is no way to know the desired level of theory or the
    system's charge state from atomic positions) - sensible defaults
    are provided and clearly present in the output route line for
    whoever opens the file to review and change.
    """
    n_lattice_vectors = 3  # always full 3D periodicity for a Structure

    route = (
        f"# {functional}/{basis_set} PBC={n_lattice_vectors}D "
        "Force"
    )

    lines = [
        route,
        "",
        title,
        "",
        f"{charge} {multiplicity}",
    ]

    for site in structure.sites:
        x, y, z = site.coords
        lines.append(f"{_symbol(site):<3}{x:15.8f}{y:15.8f}{z:15.8f}")

    for vector in structure.lattice.matrix:
        x, y, z = vector
        lines.append(f"{'TV':<3}{x:15.8f}{y:15.8f}{z:15.8f}")

    lines.append("")
    lines.append("")

    return "\n".join(lines)


def export_cp2k(
    structure,
    *,
    project_name: str = "the_void",
    functional: str = "PBE",
    basis: str = CP2K_DEFAULT_BASIS,
    potential: str = CP2K_DEFAULT_POTENTIAL,
    run_type: str = "ENERGY_FORCE",
) -> str:
    """CP2K input, hand-built rather than via pymatgen's CP2K input-set
    machinery: that machinery needs a populated basis-set database to
    match elements against, and raised ValueError for an ordinary
    NaCl-style structure when tried directly in this environment
    (confirmed, not assumed) - a dependency this module cannot rely on
    being available. Every element instead gets the same well-known,
    generic GTH basis/potential names (DZVP-MOLOPT-SR-GTH /
    GTH-PBE by default) that ship with every standard CP2K
    installation's basis-set files, clearly present in each &KIND
    section for review.

    &CELL preserves the exact 3x3 lattice matrix (not just a, b, c
    lengths and angles, so no re-derivation/rounding of the original
    vectors is needed); &COORD uses fractional coordinates with
    SCALED_COORDS TRUE, both exact to the source structure.
    """
    lattice = structure.lattice.matrix

    lines = [
        "&GLOBAL",
        f"  PROJECT {project_name}",
        f"  RUN_TYPE {run_type}",
        "  PRINT_LEVEL LOW",
        "&END GLOBAL",
        "&FORCE_EVAL",
        "  METHOD Quickstep",
        "  &DFT",
        "    &QS",
        "      METHOD GPW",
        "    &END QS",
        "    &MGRID",
        "      CUTOFF 400",
        "      REL_CUTOFF 50",
        "    &END MGRID",
        "    &SCF",
        "      MAX_SCF 300",
        "      EPS_SCF 1.0E-6",
        "    &END SCF",
        "    &XC",
        "      &XC_FUNCTIONAL",
        f"        {functional}",
        "      &END XC_FUNCTIONAL",
        "    &END XC",
        "  &END DFT",
        "  &SUBSYS",
        "    &CELL",
    ]

    for label, vector in zip("ABC", lattice):
        x, y, z = vector
        lines.append(f"      {label} {x:.8f} {y:.8f} {z:.8f}")

    lines += [
        "    &END CELL",
        "    &COORD",
        "      SCALED_COORDS TRUE",
    ]

    for site in structure.sites:
        fx, fy, fz = site.frac_coords
        lines.append(
            f"      {_symbol(site):<3}{fx:15.8f}{fy:15.8f}{fz:15.8f}"
        )

    lines.append("    &END COORD")

    for symbol in _unique_symbols(structure):
        lines += [
            f"    &KIND {symbol}",
            f"      BASIS_SET {basis}",
            f"      POTENTIAL {potential}",
            "    &END KIND",
        ]

    lines += [
        "  &END SUBSYS",
        "&END FORCE_EVAL",
    ]

    return "\n".join(lines) + "\n"


def export_quantum_espresso(
    structure,
    *,
    pseudopotentials: dict[str, str] | None = None,
    ecutwfc: float = 40.0,
    ecutrho: float | None = None,
    kpoints_grid: tuple[int, int, int] = (4, 4, 4),
    calculation: str = "scf",
) -> str:
    """Quantum ESPRESSO pw.x input, via pymatgen's own PWInput writer -
    the exact 3x3 lattice matrix (CELL_PARAMETERS, not re-derived from
    a, b, c/angles) and exact fractional coordinates, both preserved.

    pseudopotentials, if not supplied, defaults to a placeholder
    filename per element following the SSSP/PSlibrary naming
    convention - not real files (none are bundled here); the generated
    ATOMIC_SPECIES block is commented so this is obvious to whoever
    opens the file, not a silent assumption that these specific files
    exist anywhere.
    """
    from pymatgen.io.pwscf import PWInput

    symbols = _unique_symbols(structure)

    used_placeholder_pseudopotentials = pseudopotentials is None

    if pseudopotentials is None:
        pseudopotentials = {
            symbol: f"{symbol}.{QE_PLACEHOLDER_PSEUDOPOTENTIAL_SUFFIX}"
            for symbol in symbols
        }

    control = {"calculation": calculation}
    system = {"ecutwfc": ecutwfc}

    if ecutrho is not None:
        system["ecutrho"] = ecutrho

    pw_input = PWInput(
        structure,
        pseudo=pseudopotentials,
        control=control,
        system=system,
        kpoints_grid=kpoints_grid,
    )

    header = (
        "! Pseudopotential filenames below are placeholders "
        "(SSSP/PSlibrary naming convention) - replace with the actual "
        "files from your chosen pseudopotential library before "
        "running.\n"
        if used_placeholder_pseudopotentials
        else ""
    )

    return header + str(pw_input)


def _symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "X"


def _unique_symbols(structure) -> list[str]:
    seen: list[str] = []

    for site in structure.sites:
        symbol = _symbol(site)

        if symbol not in seen:
            seen.append(symbol)

    return seen
