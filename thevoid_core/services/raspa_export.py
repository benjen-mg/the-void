"""RASPA2 export: a characterized framework as a ready-to-run GCMC input
set (round 52).

The Void does geometry, not simulation. This module is the bridge: it
takes a structure the application has already loaded and bonded, and
writes everything RASPA2 needs to run a grand-canonical Monte Carlo
adsorption calculation on it - the framework, a force field, the
adsorbate model, the pockets to block and simulation.input - into one
folder. Run `simulate` in that folder and nothing else is needed.

What it produces is a documented starting point, not a finished
simulation setup. Every assumption (force field, charges, cycle counts,
cutoff, which molecules were removed) is written into README.txt and
the JSON manifest beside the input files, and is shown to the person
before they export. The defaults were chosen because they are the most
common published choices for rigid-framework screening, not because
they are right for every material.

Force field
    Framework atoms: UFF Lennard-Jones (Rappe et al., J. Am. Chem. Soc.
    1992, 114, 10024), eps = D_i, sigma = x_i / 2^(1/6), one type per
    element (UFF's van der Waals terms do not depend on hybridisation;
    every UFF type of an element carries the same x_i and D_i). The
    table below was copied from the parameter set compiled into RDKit's
    UFF implementation and is checked against RDKit at test time.
    Adsorbates: TraPPE (Potoff & Siepmann, AIChE J. 2001, 47, 1676 for
    CO2 and N2; Martin & Siepmann, J. Phys. Chem. B 1998, 102, 2569 for
    united-atom CH4). Cross terms: Lorentz-Berthelot. LJ shifted at the
    cutoff, no tail corrections, the RASPA "GenericMOFs" convention.

Why everything is written locally instead of using RASPA's bundled
force fields: RASPA reads ./pseudo_atoms.def, ./force_field_mixing_
rules.def, ./force_field.def and ./<molecule>.def from the working
directory before its share directory (confirmed in RASPA2 2.0.4's
molecule.c and potentials.c). The bundled "GenericMOFs" set is
DREIDING for main-group atoms and UFF for metals, and its entries are
prefix wildcards: "C_" matches every pseudo-atom whose name starts with
C, so a framework labelled Cu, Co, Cr, Cd or Cl silently picks up
carbon's parameters unless a more specific rule happens to follow it.
Exact, local names avoid all of that and make the export independent of
which RASPA version or share directory is installed.

Nothing here imports Qt.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from thevoid_core.version import __version__

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

#: kcal/mol -> K (divide by the gas constant): 4184 / 8.314462618.
KCAL_PER_MOL_IN_K = 4184.0 / 8.314462618

#: UFF van der Waals parameters per element: (x_i in angstrom - the
#: LJ minimum distance - and D_i, the well depth, in kcal/mol), from
#: Rappe et al. 1992 Table 1 as compiled into RDKit's UFF parameter set
#: (UFF's "Lw" is lawrencium). Every UFF atom type of one element has
#: the same x_i and D_i, which was checked across all 127 types.
UFF_VDW: dict[str, tuple[float, float]] = {
    "H": (2.886, 0.044), "He": (2.362, 0.056), "Li": (2.451, 0.025),
    "Be": (2.745, 0.085), "B": (4.083, 0.18), "C": (3.851, 0.105),
    "N": (3.66, 0.069), "O": (3.5, 0.06), "F": (3.364, 0.05),
    "Ne": (3.243, 0.042), "Na": (2.983, 0.03), "Mg": (3.021, 0.111),
    "Al": (4.499, 0.505), "Si": (4.295, 0.402), "P": (4.147, 0.305),
    "S": (4.035, 0.274), "Cl": (3.947, 0.227), "Ar": (3.868, 0.185),
    "K": (3.812, 0.035), "Ca": (3.399, 0.238), "Sc": (3.295, 0.019),
    "Ti": (3.175, 0.017), "V": (3.144, 0.016), "Cr": (3.023, 0.015),
    "Mn": (2.961, 0.013), "Fe": (2.912, 0.013), "Co": (2.872, 0.014),
    "Ni": (2.834, 0.015), "Cu": (3.495, 0.005), "Zn": (2.763, 0.124),
    "Ga": (4.383, 0.415), "Ge": (4.28, 0.379), "As": (4.23, 0.309),
    "Se": (4.205, 0.291), "Br": (4.189, 0.251), "Kr": (4.141, 0.22),
    "Rb": (4.114, 0.04), "Sr": (3.641, 0.235), "Y": (3.345, 0.072),
    "Zr": (3.124, 0.069), "Nb": (3.165, 0.059), "Mo": (3.052, 0.056),
    "Tc": (2.998, 0.048), "Ru": (2.963, 0.056), "Rh": (2.929, 0.053),
    "Pd": (2.899, 0.048), "Ag": (3.148, 0.036), "Cd": (2.848, 0.228),
    "In": (4.463, 0.599), "Sn": (4.392, 0.567), "Sb": (4.42, 0.449),
    "Te": (4.47, 0.398), "I": (4.5, 0.339), "Xe": (4.404, 0.332),
    "Cs": (4.517, 0.045), "Ba": (3.703, 0.364), "La": (3.522, 0.017),
    "Ce": (3.556, 0.013), "Pr": (3.606, 0.01), "Nd": (3.575, 0.01),
    "Pm": (3.547, 0.009), "Sm": (3.52, 0.008), "Eu": (3.493, 0.008),
    "Gd": (3.368, 0.009), "Tb": (3.451, 0.007), "Dy": (3.428, 0.007),
    "Ho": (3.409, 0.007), "Er": (3.391, 0.007), "Tm": (3.374, 0.006),
    "Yb": (3.355, 0.228), "Lu": (3.64, 0.041), "Hf": (3.141, 0.072),
    "Ta": (3.17, 0.081), "W": (3.069, 0.067), "Re": (2.954, 0.066),
    "Os": (3.12, 0.037), "Ir": (2.84, 0.073), "Pt": (2.754, 0.08),
    "Au": (3.293, 0.039), "Hg": (2.705, 0.385), "Tl": (4.347, 0.68),
    "Pb": (4.297, 0.663), "Bi": (4.37, 0.518), "Po": (4.709, 0.325),
    "At": (4.75, 0.284), "Rn": (4.765, 0.248), "Fr": (4.9, 0.05),
    "Ra": (3.677, 0.404), "Ac": (3.478, 0.033), "Th": (3.396, 0.026),
    "Pa": (3.424, 0.022), "U": (3.395, 0.022), "Np": (3.424, 0.019),
    "Pu": (3.424, 0.016), "Am": (3.381, 0.014), "Cm": (3.326, 0.013),
    "Bk": (3.339, 0.013), "Cf": (3.313, 0.013), "Es": (3.299, 0.012),
    "Fm": (3.286, 0.012), "Md": (3.274, 0.011), "No": (3.248, 0.011),
    "Lr": (3.236, 0.011),
}

#: Deuterium and tritium get hydrogen's parameters and name.
_ELEMENT_ALIASES = {"D": "H", "T": "H"}


def uff_lennard_jones(symbol: str) -> tuple[float, float]:
    """(epsilon / k_B in K, sigma in angstrom) for one element."""
    symbol = _ELEMENT_ALIASES.get(symbol, symbol)

    if symbol not in UFF_VDW:
        raise ValueError(
            f"No UFF parameters for {symbol!r}; RASPA export needs a "
            "real element on every site."
        )

    x_i, d_i = UFF_VDW[symbol]

    return d_i * KCAL_PER_MOL_IN_K, x_i / 2.0 ** (1.0 / 6.0)


@dataclass(frozen=True, slots=True)
class PseudoAtom:
    """One RASPA pseudo-atom: a row of pseudo_atoms.def plus its row of
    force_field_mixing_rules.def."""

    name: str
    element: str
    mass: float
    charge: float
    epsilon_K: float | None  # None: no LJ site ("none" interaction)
    sigma_A: float | None
    printed: bool = True


@dataclass(frozen=True, slots=True)
class AdsorbateModel:
    """A rigid adsorbate as RASPA needs it. The first atom is the
    molecule's centre: RASPA tests pocket blocking on the starting
    bead (bead 0, see BlockedPocket calls in RASPA2's cbmc.c), and the
    blocking spheres are built from where a probe *centre* fits, so
    bead 0 has to be the centre for the two to mean the same thing.
    """

    name: str
    description: str
    reference: str
    critical_temperature_K: float
    critical_pressure_Pa: float
    acentric_factor: float
    #: (pseudo-atom name, x, y, z in angstrom), centre first.
    atoms: tuple[tuple[str, float, float, float], ...]
    pseudo_atoms: tuple[PseudoAtom, ...]
    #: Linear molecules get rotation moves; a single bead does not.
    rotates: bool
    #: Probe radius the rest of the application uses for this gas
    #: (data/probe_radii.py) - the default for pocket blocking.
    probe_radius_key: str

    @property
    def is_charged(self) -> bool:
        return any(abs(atom.charge) > 0 for atom in self.pseudo_atoms)


#: Critical constants are the NIST values RASPA2's own TraPPE molecule
#: files carry (they feed the Peng-Robinson fugacity).
ADSORBATES: dict[str, AdsorbateModel] = {
    "CO2": AdsorbateModel(
        name="CO2",
        description="carbon dioxide, rigid 3-site TraPPE",
        reference="Potoff & Siepmann, AIChE J. 2001, 47, 1676",
        critical_temperature_K=304.1282,
        critical_pressure_Pa=7377300.0,
        acentric_factor=0.22394,
        atoms=(
            ("C_co2", 0.0, 0.0, 0.0),
            ("O_co2", 0.0, 0.0, 1.16),
            ("O_co2", 0.0, 0.0, -1.16),
        ),
        pseudo_atoms=(
            PseudoAtom("C_co2", "C", 12.0107, 0.70, 27.0, 2.80),
            PseudoAtom("O_co2", "O", 15.9994, -0.35, 79.0, 3.05),
        ),
        rotates=True,
        probe_radius_key="CO2",
    ),
    "N2": AdsorbateModel(
        name="N2",
        description="nitrogen, rigid TraPPE with a massless centre charge",
        reference="Potoff & Siepmann, AIChE J. 2001, 47, 1676",
        critical_temperature_K=126.192,
        critical_pressure_Pa=3395800.0,
        acentric_factor=0.0372,
        atoms=(
            ("N_com", 0.0, 0.0, 0.0),
            ("N_n2", 0.0, 0.0, 0.55),
            ("N_n2", 0.0, 0.0, -0.55),
        ),
        pseudo_atoms=(
            PseudoAtom("N_n2", "N", 14.0067, -0.482, 36.0, 3.31),
            PseudoAtom("N_com", "N", 0.0, 0.964, None, None, printed=False),
        ),
        rotates=True,
        probe_radius_key="N2",
    ),
    "CH4": AdsorbateModel(
        name="CH4",
        description="methane, united-atom TraPPE (one uncharged site)",
        reference="Martin & Siepmann, J. Phys. Chem. B 1998, 102, 2569",
        critical_temperature_K=190.564,
        critical_pressure_Pa=4599200.0,
        acentric_factor=0.01142,
        atoms=(("CH4_sp3", 0.0, 0.0, 0.0),),
        pseudo_atoms=(
            PseudoAtom("CH4_sp3", "C", 16.04246, 0.0, 148.0, 3.73),
        ),
        rotates=False,
        probe_radius_key="CH4",
    ),
}


@dataclass
class RaspaExportSettings:
    """Everything a person can choose about the exported simulation.
    The defaults are a common screening set-up, recorded in every
    export as assumptions rather than presented as recommendations."""

    adsorbate: str = "CO2"
    temperature_K: float = 298.0
    #: RASPA runs one isotherm point per pressure, in order.
    pressures_Pa: tuple[float, ...] = (1.0e5,)
    initialization_cycles: int = 5000
    cycles: int = 10000
    print_every: int = 1000
    #: Applied to both LJ and the Ewald real-space sum.
    cutoff_A: float = 12.8
    #: Opt-in: RASPA's built-in charge equilibration, computed at run
    #: time. Ignored when the input CIF already carries charges.
    charge_equilibration: bool = False
    #: Drop 0-periodic molecules (lattice solvent, free counter-ions).
    remove_free_molecules: bool = True
    block_pockets: bool = True
    #: None: the probe radius the rest of the application uses for the
    #: adsorbate (data/probe_radii.py).
    blocking_probe_radius_A: float | None = None
    grid_spacing_A: float = 0.3
    #: Base name for the CIF and block file; defaults to the input
    #: file's stem.
    framework_name: str | None = None

    def validate(self) -> None:
        if self.adsorbate not in ADSORBATES:
            raise ValueError(
                f"Unknown adsorbate {self.adsorbate!r}; choose one of "
                f"{', '.join(ADSORBATES)}."
            )
        if not self.pressures_Pa or any(p <= 0 for p in self.pressures_Pa):
            raise ValueError("Pressures must be positive.")
        if self.temperature_K <= 0:
            raise ValueError("Temperature must be positive.")
        if self.cutoff_A <= 0:
            raise ValueError("The cutoff must be positive.")
        if self.cycles < 1 or self.initialization_cycles < 0:
            raise ValueError("Cycle counts must be positive.")
        if self.print_every < 1:
            raise ValueError("PrintEvery must be at least 1.")
        if (
            self.blocking_probe_radius_A is not None
            and self.blocking_probe_radius_A <= 0
        ):
            raise ValueError("The blocking probe radius must be positive.")

    def as_dict(self) -> dict:
        data = asdict(self)
        data["pressures_Pa"] = list(self.pressures_Pa)
        return data


# ----------------------------------------------------------------------
# Framework preparation
# ----------------------------------------------------------------------


@dataclass
class RemovedMolecule:
    formula: str
    count: int
    atoms_each: int
    possibly_ionic: bool


@dataclass
class PreparedFramework:
    """The exact set of atoms that will be written, and why."""

    structure: object
    #: Per-site charges index-aligned with structure, or None.
    charges: list[float] | None
    removed: list[RemovedMolecule]
    framework_dimensionality: int
    disorder_sites_dropped: int
    notes: list[str] = field(default_factory=list)
    #: Bonds of `structure` itself (indices remapped after removals),
    #: so later steps - topology, round 53 - need not re-detect them.
    bonds: list = field(default_factory=list)


def read_cif_charges(path) -> dict[str, float] | None:
    """_atom_site_label -> _atom_site_charge, read from the raw CIF
    (pymatgen's Structure does not keep the column). None when the
    file is not a CIF, has no charge column, or any charge is missing
    or unparsable - half a set of charges is worse than none, so this
    is all or nothing.
    """
    if path is None or Path(path).suffix.lower() != ".cif":
        return None

    try:
        from pymatgen.io.cif import CifParser

        parser = CifParser(str(path))

        for block in parser._cif.data.values():
            labels = block.data.get("_atom_site_label")
            charges = block.data.get("_atom_site_charge")

            if not labels or not charges:
                continue

            result = {}

            for label, charge in zip(labels, charges):
                text = str(charge).split("(")[0]

                if text in (".", "?", ""):
                    return None

                result[label] = float(text)

            return result
    except Exception:
        logger.debug("Could not read charges from %r.", path, exc_info=True)

    return None


def prepare_framework(
    structure,
    bonds=None,
    *,
    remove_free_molecules: bool = True,
    cif_charges: dict[str, float] | None = None,
    bond_detector=None,
) -> PreparedFramework:
    """Decide which atoms are the framework RASPA should see.

    1. Disorder: RASPA has no notion of partial occupancy, so minor
       components are dropped with the same 0.5 filter every void-space
       analysis uses (export.ordered_structure_for_export). A site
       split exactly 50/50 between species survives that filter and
       cannot be written; that raises ValueError.
    2. Bonds: the caller's already-detected bonds are used as-is for an
       ordered structure. If step 1 removed sites, their indices no
       longer apply and bonds are re-detected on the ordered cell with
       the same covalent detector.
    3. Free molecules: bonded components that are not periodic in any
       direction (mof_analysis.bonded_components) are lattice solvent
       or free ions, not framework, and are removed when asked. Nothing
       is removed from a structure with no periodic component at all -
       a molecular crystal has no framework to keep.
    """
    from thevoid_core.data.metal_classification import is_metal
    from thevoid_core.services.export import ordered_structure_for_export
    from thevoid_core.services.mof_analysis import bonded_components

    notes: list[str] = []
    ordered = ordered_structure_for_export(structure)

    if not ordered.is_ordered:
        raise ValueError(
            "This structure has sites shared between two or more species "
            "at 0.5 occupancy or more, which RASPA cannot represent. "
            "Resolve the disorder before exporting."
        )

    disorder_dropped = len(structure) - len(ordered)

    if disorder_dropped:
        notes.append(
            f"{disorder_dropped} minor disorder component site(s) "
            "(occupancy below 0.5) were dropped, as in every void-space "
            "analysis."
        )

    if bonds is None or ordered is not structure:
        if bond_detector is None:
            from thevoid_core.services.bonds.covalent_bond_detector import (
                CovalentBondDetector,
            )

            bond_detector = CovalentBondDetector()

        bonds = bond_detector.detect(ordered)

    charges = None

    if cif_charges is not None:
        labels = [site.label for site in ordered.sites]

        if all(label in cif_charges for label in labels):
            charges = [float(cif_charges[label]) for label in labels]
        else:
            notes.append(
                "The input CIF has an _atom_site_charge column, but not "
                "every site's label could be matched to a charge, so no "
                "charges were used."
            )

    components = bonded_components(ordered, bonds, is_metal)
    dimensionality = max((c.dimensionality for c in components), default=0)

    free = [c for c in components if not c.is_periodic]
    removed: list[RemovedMolecule] = []
    keep = list(range(len(ordered)))

    if dimensionality == 0:
        notes.append(
            "No bonded component extends periodically, so there is no "
            "framework to separate from guests; every atom was kept. "
            "Check the bonding tolerance if this is meant to be a "
            "framework."
        )
    elif free and remove_free_molecules:
        drop = {index for c in free for index in c.site_indices}
        keep = [index for index in keep if index not in drop]

        grouped: dict[tuple[str, bool], RemovedMolecule] = {}

        for component in free:
            key = (component.formula, component.possibly_ionic)

            if key not in grouped:
                grouped[key] = RemovedMolecule(
                    formula=component.formula,
                    count=0,
                    atoms_each=component.atom_count,
                    possibly_ionic=component.possibly_ionic,
                )

            grouped[key].count += 1

        removed = sorted(grouped.values(), key=lambda m: (-m.count, m.formula))
    elif free:
        notes.append(
            f"{len(free)} free (non-periodic) molecule(s) were kept at "
            "your request; RASPA will treat them as fixed framework atoms."
        )

    if len(keep) != len(ordered):
        framework = ordered.copy()
        framework.remove_sites(
            [index for index in range(len(ordered)) if index not in set(keep)]
        )

        if charges is not None:
            charges = [charges[index] for index in keep]

        from thevoid_core.services.bonds.bond_model import Bond

        # remove_sites keeps the survivors in order, so old index ->
        # position in keep.
        new_index = {old: new for new, old in enumerate(keep)}
        bonds = [
            Bond(new_index[b.atom1], new_index[b.atom2], b.image, b.length)
            for b in bonds
            if b.atom1 in new_index and b.atom2 in new_index
        ]
    else:
        framework = ordered

    return PreparedFramework(
        structure=framework,
        charges=charges,
        removed=removed,
        framework_dimensionality=dimensionality,
        disorder_sites_dropped=disorder_dropped,
        notes=notes,
        bonds=list(bonds),
    )


# ----------------------------------------------------------------------
# Unit cells and pockets
# ----------------------------------------------------------------------


def unit_cells_for_cutoff(lattice_matrix, cutoff: float) -> tuple[int, int, int]:
    """How many copies of the cell RASPA needs along each axis.

    The minimum-image convention needs the simulation box's
    perpendicular width to exceed twice the cutoff in every direction;
    RASPA refuses to run otherwise. floor(2 rc / w) + 1 is the
    smallest count that is strictly larger (plain ceil would allow
    exact equality).
    """
    from thevoid_core.services.pore_analysis import perpendicular_widths

    widths = perpendicular_widths(lattice_matrix)

    return tuple(int(math.floor(2.0 * cutoff / w)) + 1 for w in widths)


@dataclass
class BlockingResult:
    """Spheres (fractional centre, radius in angstrom) that forbid
    insertions into pockets the adsorbate cannot reach."""

    probe_radius_A: float
    spheres: list[tuple[float, float, float, float]]
    pocket_volume_A3: float
    accessible_volume_A3: float
    accessible_dimensionality: int
    grid_shape: tuple[int, int, int]
    #: "blocked", "no pockets", or "not percolating" (see compute_blocking).
    status: str
    #: Largest node spacing actually used, angstrom.
    grid_spacing_A: float = 0.0

    @property
    def sphere_count(self) -> int:
        return len(self.spheres)


#: Beyond this many spheres the block file is written anyway but the
#: export warns: RASPA tests every sphere on every insertion.
MAX_REASONABLE_BLOCK_SPHERES = 2000

#: Node cap per axis for the pocket grid. Pore Analysis caps at 60,
#: which for a 3 x 3 x 3 CALF-20 cell (27 A) means 0.45 A spacing - too
#: coarse to resolve its 2.7 A windows, so a 1.2 A probe that
#: percolates in the unit cell came out "not percolating". At 100 the
#: 0.3 A target holds up to 30 A cells (checked: the same accessible
#: volume per cell as the unit cell, 49.4 A^3, in 11 s).
BLOCKING_MAX_GRID_POINTS_PER_AXIS = 100


def compute_blocking(
    structure,
    probe_radius: float,
    *,
    radius_provider=None,
    grid_spacing: float = 0.3,
    max_grid_points_per_axis: int = BLOCKING_MAX_GRID_POINTS_PER_AXIS,
) -> BlockingResult:
    """Cover every sealed pocket with spheres RASPA can block.

    Void space is classified with pore analysis's own grid code
    (pore_analysis.classify_void_grid): a node is a pocket node when a
    probe centre fits there but no probe-sized path joins it to a
    percolating network.

    Covering is greedy: take the uncovered pocket node with the
    largest clearance, put a sphere there, remove every pocket node it
    contains, repeat. A sphere's radius is its centre's clearance - the
    largest empty sphere that fits there, Zeo++'s blocking convention -
    but never more than the distance to the nearest accessible node
    minus a quarter of a grid spacing, so no grid node the probe can
    reach is ever inside a sphere. The clearance exceeds the extent of
    the region a probe centre can occupy by the probe radius itself,
    so the continuous pocket between nodes is covered too. Each sphere
    contains at least its own centre, so the loop always terminates.

    RASPA tests blocking on a molecule's starting bead, which every
    AdsorbateModel puts at the molecular centre - the same point the
    probe-centre classification is about.

    Two cases write no spheres, by design:
      - "no pockets": nothing to block.
      - "not percolating": the probe reaches no channel at all. Blocking
        would then forbid every insertion and the GCMC would report
        exactly zero, which is never a useful input - hard-sphere
        radii are conservative, and real adsorbates often do get
        through windows slightly narrower than the probe (CALF-20 and
        CO2 are the textbook example, and the application's own
        fixture). Nothing is blocked and the export warns instead.
    """
    from thevoid_core.data.element_radii import ElementRadiusProvider
    from thevoid_core.services.pore_analysis import (
        classify_void_grid,
        grid_shape_for,
    )

    radius_provider = radius_provider or ElementRadiusProvider()

    symbols = [_symbol(site) for site in structure.sites]
    atom_radii = np.array([radius_provider.get_vdw_radius(s) for s in symbols])
    grid_shape = grid_shape_for(structure, grid_spacing, max_grid_points_per_axis)
    grid = classify_void_grid(
        structure, atom_radii, probe_radius=probe_radius, grid_shape=grid_shape
    )

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    cell_volume = float(abs(np.linalg.det(lattice)))
    n_nodes = int(np.prod(grid_shape))
    node_volume = cell_volume / n_nodes

    accessible = grid.sweep.accessible
    pocket = grid.pocket_mask
    dimensionality = max(grid.sweep.dimensionality_by_root.values(), default=0)

    actual_spacing = float(
        max(np.linalg.norm(lattice, axis=1) / np.array(grid_shape))
    )

    def result(status, spheres):
        return BlockingResult(
            grid_spacing_A=actual_spacing,
            probe_radius_A=probe_radius,
            spheres=spheres,
            pocket_volume_A3=float(pocket.sum()) * node_volume,
            accessible_volume_A3=float(accessible.sum()) * node_volume,
            accessible_dimensionality=dimensionality,
            grid_shape=tuple(grid_shape),
            status=status,
        )

    if not pocket.any():
        return result("no pockets", [])

    if not accessible.any():
        return result("not percolating", [])

    step = lattice / np.array(grid_shape)[:, None]
    # Stop this far short of the nearest accessible node: a quarter of
    # the smallest node spacing keeps the boundary between two nodes
    # of different classes split the way the grid itself splits it.
    margin = 0.25 * float(np.min(np.linalg.norm(step, axis=1)))

    pocket_frac = grid.grid_frac[pocket]
    clearance = grid.distance_field[pocket]
    accessible_frac = grid.grid_frac[accessible]

    from scipy.spatial import cKDTree

    # Periodic nearest-accessible distances for a triclinic cell: the
    # accessible nodes in all 27 neighbouring cells, in Cartesian space.
    shifts = np.array(
        [(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
        dtype=float,
    )
    images = (accessible_frac[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    tree = cKDTree(images @ lattice)

    uncovered = np.ones(len(pocket_frac), dtype=bool)
    order = np.argsort(-clearance)
    spheres: list[tuple[float, float, float, float]] = []

    for candidate in order:
        if not uncovered[candidate]:
            continue

        centre = pocket_frac[candidate]
        nearest_accessible, _ = tree.query(centre @ lattice)
        radius = min(
            float(clearance[candidate]),
            float(nearest_accessible) - margin,
        )

        delta = pocket_frac[uncovered] - centre
        delta -= np.round(delta)
        inside = np.linalg.norm(delta @ lattice, axis=1) < radius

        still = np.flatnonzero(uncovered)
        uncovered[still[inside]] = False
        uncovered[candidate] = False

        spheres.append((*map(float, centre), radius))

    return result("blocked", spheres)


# ----------------------------------------------------------------------
# File writers
# ----------------------------------------------------------------------


def _symbol(site) -> str:
    species = list(site.species.keys())
    symbol = species[0].symbol if species else "X"

    return _ELEMENT_ALIASES.get(symbol, symbol)


def _element_mass(symbol: str) -> float:
    from pymatgen.core import Element

    return float(Element(symbol).atomic_mass)


def framework_pseudo_atoms(structure, charges=None) -> list[PseudoAtom]:
    """One pseudo-atom per element, named exactly by its symbol (no
    wildcards, see the module docstring). A pseudo-atom's own charge
    is 0: per-atom charges, when there are any, come from the CIF's
    _atom_site_charge column (UseChargesFromCIFFile yes)."""
    seen: list[str] = []

    for site in structure.sites:
        symbol = _symbol(site)

        if symbol not in seen:
            seen.append(symbol)

    atoms = []

    for symbol in seen:
        epsilon, sigma = uff_lennard_jones(symbol)
        atoms.append(
            PseudoAtom(symbol, symbol, _element_mass(symbol), 0.0, epsilon, sigma)
        )

    return atoms


def write_framework_cif(structure, name: str, charges=None) -> str:
    """A P1 CIF in the form RASPA2's reader expects: the full cell
    written out, labels that are exactly the pseudo-atom names (RASPA
    looks pseudo-atoms up by _atom_site_label), and an _atom_site_charge
    column when there are charges."""
    lattice = structure.lattice
    lines = [
        f"data_{name}",
        "",
        f"_audit_creation_method  'the Void {__version__} RASPA export'",
        f"_cell_length_a     {lattice.a:.6f}",
        f"_cell_length_b     {lattice.b:.6f}",
        f"_cell_length_c     {lattice.c:.6f}",
        f"_cell_angle_alpha  {lattice.alpha:.6f}",
        f"_cell_angle_beta   {lattice.beta:.6f}",
        f"_cell_angle_gamma  {lattice.gamma:.6f}",
        f"_cell_volume       {lattice.volume:.6f}",
        "",
        "_symmetry_space_group_name_H-M  'P 1'",
        "_symmetry_Int_Tables_number     1",
        "_symmetry_cell_setting          triclinic",
        "",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "'x, y, z'",
        "",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]

    if charges is not None:
        lines.append("_atom_site_charge")

    for index, site in enumerate(structure.sites):
        symbol = _symbol(site)
        fx, fy, fz = (float(v) % 1.0 for v in site.frac_coords)
        row = f"{symbol:<4} {symbol:<3} {fx:.8f} {fy:.8f} {fz:.8f}"

        if charges is not None:
            row += f" {charges[index]: .6f}"

        lines.append(row)

    return "\n".join(lines) + "\n"


def write_pseudo_atoms_def(pseudo_atoms: list[PseudoAtom]) -> str:
    from thevoid_core.data.element_radii import ElementRadiusProvider

    radii = ElementRadiusProvider()
    lines = [
        "#number of pseudo atoms",
        str(len(pseudo_atoms)),
        "#type      print   as    chem  oxidation   mass        charge   "
        "polarization B-factor radii  connectivity anisotropic "
        "anisotropic-type   tinker-type",
    ]

    for atom in pseudo_atoms:
        radius = radii.get_covalent_radius(atom.element)
        lines.append(
            f"{atom.name:<10} {'yes' if atom.printed else 'no':<7} "
            f"{atom.element:<5} {atom.element:<5} 0           "
            f"{atom.mass:<11.6f} {atom.charge:<8.4f} 0.0          1.0      "
            f"{radius:<6.3f} 0            0           relative           0"
        )

    return "\n".join(lines) + "\n"


def write_mixing_rules_def(pseudo_atoms: list[PseudoAtom]) -> str:
    lines = [
        "# general rule for shifted vs truncated",
        "shifted",
        "# general rule tailcorrections",
        "no",
        "# number of defined interactions",
        str(len(pseudo_atoms)),
        "# type interaction, parameters (epsilon/kB in K, sigma in A). "
        "Exact names only - no '_' wildcards.",
    ]

    for atom in pseudo_atoms:
        if atom.epsilon_K is None:
            lines.append(f"{atom.name:<10} none")
        else:
            lines.append(
                f"{atom.name:<10} lennard-jones {atom.epsilon_K:10.4f} "
                f"{atom.sigma_A:9.5f}"
            )

    lines += [
        "# general mixing rule for Lennard-Jones",
        "Lorentz-Berthelot",
    ]

    return "\n".join(lines) + "\n"


FORCE_FIELD_DEF = (
    "# rules to overwrite\n0\n"
    "# number of defined interactions\n0\n"
    "# mixing rules to overwrite\n0\n"
)


def write_molecule_def(model: AdsorbateModel) -> str:
    n_atoms = len(model.atoms)
    lines = [
        "# critical constants: Temperature [T], Pressure [Pa], and "
        "Acentric factor [-]",
        f"{model.critical_temperature_K}",
        f"{model.critical_pressure_Pa}",
        f"{model.acentric_factor}",
        "# Number Of Atoms",
        str(n_atoms),
        "# Number of groups",
        "1",
        f"# {model.name}-group",
        "rigid",
        "# number of atoms",
        str(n_atoms),
        "# atomic positions",
    ]

    for index, (name, x, y, z) in enumerate(model.atoms):
        lines.append(f"{index} {name:<9} {x:12.6f} {y:12.6f} {z:12.6f}")

    n_bonds = n_atoms - 1
    lines += [
        "# Chiral centers Bond  BondDipoles Bend  UrayBradley InvBend  "
        "Torsion Imp. Torsion Bond/Bond Stretch/Bend Bend/Bend "
        "Stretch/Torsion Bend/Torsion IntraVDW IntraCoulomb",
        f"               0    {n_bonds}            0    0            0"
        "       0        0            0         0            0         0"
        "               0            0        0            0",
    ]

    if n_bonds:
        lines.append("# Bond stretch: atom n1-n2, type, parameters")
        lines += [f"0 {i} RIGID_BOND" for i in range(1, n_atoms)]

    lines += ["# Number of config moves", "0"]

    return "\n".join(lines) + "\n"


def write_block_file(blocking: BlockingResult) -> str:
    """RASPA's .block format: a count, then one line per sphere of
    fractional centre (unit-cell coordinates) and radius in angstrom -
    as read by ReadBlockingPockets in RASPA2's framework.c."""
    lines = [str(blocking.sphere_count)]
    lines += [
        f"{x:.6f} {y:.6f} {z:.6f} {r:.4f}" for x, y, z, r in blocking.spheres
    ]

    return "\n".join(lines) + "\n"


def write_simulation_input(
    settings: RaspaExportSettings,
    *,
    framework_name: str,
    unit_cells: tuple[int, int, int],
    charge_source: str,
    block: bool,
) -> str:
    """The GCMC run. charge_source is "cif" (charges in the CIF this
    folder ships), "charge_equilibration" (charges in <name>_qeq.cif,
    which run_raspa.sh makes first - see write_charge_equilibration_input)
    or "none"."""
    model = ADSORBATES[settings.adsorbate]
    needs_ewald = model.is_charged
    pressures = " ".join(f"{p:.6g}" for p in settings.pressures_Pa)
    use_cif_charges = charge_source in ("cif", "charge_equilibration")
    cif_name = (
        f"{framework_name}{QEQ_SUFFIX}"
        if charge_source == "charge_equilibration"
        else framework_name
    )

    charge_note = {
        "cif": "framework charges from the input CIF's _atom_site_charge column",
        "charge_equilibration": f"framework charges from {cif_name}.cif, made "
        "by RASPA charge equilibration in step 1 of run_raspa.sh",
        "none": "UNCHARGED FRAMEWORK - no framework-adsorbate electrostatics",
    }[charge_source]

    lines = [
        f"# Written by the Void {__version__}. A documented STARTING POINT,",
        "# not a finished simulation setup - read README.txt before "
        "trusting results.",
        f"# Force field: UFF (framework) + TraPPE ({model.name}), "
        "Lorentz-Berthelot, LJ shifted, no tail corrections.",
        f"# Charges: {charge_note}.",
        "",
        "SimulationType                MonteCarlo",
        f"NumberOfCycles                {settings.cycles}",
        f"NumberOfInitializationCycles  {settings.initialization_cycles}",
        f"PrintEvery                    {settings.print_every}",
        "RestartFile                   no",
        "",
        "Forcefield                    Local",
        f"CutOffVDW                     {settings.cutoff_A}",
    ]

    if needs_ewald:
        lines += [
            "ChargeMethod                  Ewald",
            f"CutOffChargeCharge            {settings.cutoff_A}",
            "EwaldPrecision                1e-6",
        ]
    else:
        lines.append("ChargeMethod                  None")

    lines += [
        f"UseChargesFromCIFFile         {'yes' if use_cif_charges else 'no'}",
    ]

    lines += [
        "",
        "Framework 0",
        f"FrameworkName                 {cif_name}",
        "RemoveAtomNumberCodeFromLabel no",
        f"UnitCells                     {unit_cells[0]} {unit_cells[1]} "
        f"{unit_cells[2]}",
        f"ExternalTemperature           {settings.temperature_K}",
        f"ExternalPressure              {pressures}",
        "",
        f"Component 0 MoleculeName             {model.name}",
        "            MoleculeDefinition       Local",
        "            TranslationProbability   0.5",
    ]

    if model.rotates:
        lines.append("            RotationProbability      0.5")

    lines += [
        "            ReinsertionProbability   0.5",
        "            SwapProbability          1.0",
        "            CreateNumberOfMolecules  0",
    ]

    if block:
        lines += [
            "            BlockPockets             yes",
            f"            BlockPocketsFileName     {framework_name}",
        ]

    return "\n".join(lines) + "\n"


#: The charged framework run_raspa.sh copies out of step 1.
QEQ_SUFFIX = "_qeq"
QEQ_FOLDER = "charges"
RUN_SCRIPT = "run_raspa.sh"


def write_charge_equilibration_input(framework_name: str, lattice_matrix) -> str:
    """Step 1 of the charge-equilibration route: a zero-cycle run on ONE
    unit cell whose only job is to make RASPA compute framework charges
    and write them out (Movies/System_0/Framework_0_final_1_1_1_P1.cif).

    Why a separate step instead of ChargeFromChargeEquilibration in the
    GCMC input: RASPA equilibrates over every atom in the simulation box
    with periodic Ewald sums, so the cost grows with the square of the
    number of unit cells. For CALF-20 (44 atoms, a 4 x 3 x 4 box) the
    in-GCMC version had not finished after 25 minutes here; on one unit
    cell it takes 8 seconds. The periodic charges are those of the
    crystal either way - RASPA's Qeq takes its own lattice sums out to
    250 A, independent of the box and of the cutoffs below (charge_
    equilibration.c, getJ).

    SymmetrizeFrameworkCharges no: RASPA's default averages the charges
    of all atoms sharing a pseudo-atom type, and this export gives every
    atom of an element the same type, so the default would give every
    oxygen one charge regardless of environment. Confirmed on CALF-20:
    with the default, its 8 O came out identical.

    The cutoffs only need to satisfy RASPA's box check for a run that
    samples nothing, so they are set just under half the narrowest cell
    width.
    """
    from thevoid_core.services.pore_analysis import perpendicular_widths

    cutoff = round(0.49 * float(min(perpendicular_widths(lattice_matrix))), 3)

    return "\n".join([
        f"# Written by the Void {__version__}. STEP 1 of 2 (run_raspa.sh runs",
        "# both): framework charges by RASPA's charge equilibration (its default",
        "# method), on one unit cell, sampling nothing.",
        "",
        "SimulationType                MonteCarlo",
        "NumberOfCycles                0",
        "NumberOfInitializationCycles  0",
        "PrintEvery                    1",
        "RestartFile                   no",
        "",
        "Forcefield                    Local",
        f"CutOffVDW                     {cutoff}",
        "ChargeMethod                  Ewald",
        f"CutOffChargeCharge            {cutoff}",
        "EwaldPrecision                1e-6",
        "UseChargesFromCIFFile         no",
        "ChargeFromChargeEquilibration yes",
        "SymmetrizeFrameworkCharges    no",
        "",
        "Framework 0",
        f"FrameworkName                 {framework_name}",
        "RemoveAtomNumberCodeFromLabel no",
        "UnitCells                     1 1 1",
    ]) + "\n"


def write_run_script(framework_name: str) -> str:
    charged = f"{framework_name}{QEQ_SUFFIX}.cif"
    return "\n".join([
        "#!/bin/sh",
        f"# Written by the Void {__version__}: charge equilibration, then GCMC.",
        "# Needs RASPA2's 'simulate' on PATH (or set SIMULATE=/path/to/simulate).",
        "set -e",
        'cd "$(dirname "$0")"',
        'SIMULATE="${SIMULATE:-simulate}"',
        "echo 'Step 1: framework charges (one unit cell)...'",
        f'(cd {QEQ_FOLDER} && "$SIMULATE" simulation.input)',
        f"cp {QEQ_FOLDER}/Movies/System_0/Framework_0_final_1_1_1_P1.cif {charged}",
        f"echo 'Step 2: GCMC with the charges in {charged}...'",
        '"$SIMULATE" simulation.input',
    ]) + "\n"


# ----------------------------------------------------------------------
# The whole export
# ----------------------------------------------------------------------


@dataclass
class RaspaExport:
    #: file name -> content, all written into one folder.
    files: dict[str, str]
    #: JSON-safe record of what was assumed and done (also written as
    #: the_void_raspa_export.json, and folded into analysis reports).
    summary: dict
    warnings: list[str]


def _safe_name(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return cleaned.strip("_") or "framework"


def build_raspa_export(
    structure,
    bonds=None,
    settings: RaspaExportSettings | None = None,
    *,
    source=None,
    radius_provider=None,
    bond_detector=None,
    progress=None,
) -> RaspaExport:
    """Build (but do not write) a complete RASPA2 input set.

    Slow enough for the worker thread on a large cell: the pocket
    classification is a full pore-analysis grid. Raises ValueError for
    anything that makes the export impossible (bad settings, an
    unresolvable disorder, an unknown element); everything that is
    merely questionable becomes a warning.
    """
    from thevoid_core.data.probe_radii import PROBE_RADII
    from thevoid_core.services.report import file_sha256, provenance

    settings = settings or RaspaExportSettings()
    settings.validate()
    model = ADSORBATES[settings.adsorbate]

    def step(message):
        if progress is not None:
            progress(message)

    name = _safe_name(
        settings.framework_name
        or (Path(source).stem if source else "framework")
    )

    step("Preparing framework atoms")
    cif_charges = read_cif_charges(source)
    prepared = prepare_framework(
        structure,
        bonds,
        remove_free_molecules=settings.remove_free_molecules,
        cif_charges=cif_charges,
        bond_detector=bond_detector,
    )
    framework = prepared.structure

    if len(framework) == 0:
        raise ValueError("No framework atoms are left to export.")

    warnings: list[str] = []
    use_cif_charges = prepared.charges is not None
    net_charge = sum(prepared.charges) if use_cif_charges else 0.0

    if use_cif_charges and abs(net_charge) > 1e-3:
        warnings.append(
            f"The framework's CIF charges sum to {net_charge:+.4f} e per "
            "cell (after any removals). Ewald sums assume a neutral cell."
        )

    if not use_cif_charges and model.is_charged:
        if settings.charge_equilibration:
            warnings.append(
                "Framework charges come from RASPA's charge equilibration, "
                f"run as a separate first step: use {RUN_SCRIPT} (or run "
                f"{QEQ_FOLDER}/ first and copy its charged CIF, as the script "
                "does). Qeq-type charges are known to be poor for some "
                "metals; check the values in the charged CIF."
            )
        else:
            warnings.append(
                f"UNCHARGED FRAMEWORK. The input has no partial charges, "
                f"so {model.name}'s quadrupole does not interact with the "
                "framework at all. This usually underestimates uptake, "
                "sometimes severely. Supply a CIF with _atom_site_charge "
                "(e.g. DDEC or EQeq charges) or opt into charge "
                "equilibration before relying on the numbers."
            )

    if any(m.possibly_ionic for m in prepared.removed):
        ions = ", ".join(m.formula for m in prepared.removed if m.possibly_ionic)
        warnings.append(
            f"Removed molecules that may be counter-ions: {ions} (isolated "
            "metal or single atom, or an odd electron count as written, "
            "which can also mean missing hydrogens). If they balance a "
            "charged framework, removing them leaves the framework "
            "charged."
        )

    symbols = {_symbol(site) for site in framework.sites}

    if "C" in symbols and "H" not in symbols:
        warnings.append(
            "The framework contains carbon but no hydrogen. Many CIFs omit "
            "H atoms; without them the framework is less dense and more "
            "open than the real material."
        )

    unit_cells = unit_cells_for_cutoff(framework.lattice.matrix, settings.cutoff_A)

    blocking = None
    blocked = False

    if settings.block_pockets:
        probe = (
            settings.blocking_probe_radius_A
            if settings.blocking_probe_radius_A is not None
            else PROBE_RADII[model.probe_radius_key]
        )
        step(f"Finding pockets a {probe:.3f} A probe cannot reach")
        blocking = compute_blocking(
            framework,
            probe,
            radius_provider=radius_provider,
            grid_spacing=settings.grid_spacing_A,
        )
        blocked = blocking.status == "blocked"

        if blocking.status == "not percolating":
            warnings.append(
                f"No channel is open to a {probe:.3f} A probe (the "
                f"hard-sphere size used for {model.name}), so nothing was "
                "blocked: blocking every cavity would force zero uptake. "
                "Real guests often pass windows slightly narrower than "
                "their hard-sphere size, but treat the GCMC result with "
                "care and consider whether diffusion into these pores is "
                "physical."
            )
        if blocking.grid_spacing_A > 1.25 * settings.grid_spacing_A:
            warnings.append(
                f"The cell is large, so pockets were classified on a "
                f"{blocking.grid_spacing_A:.2f} A grid rather than "
                f"{settings.grid_spacing_A} A. Windows close to the probe "
                "size may be misclassified; compare with Pore Analysis on "
                "the unit cell."
            )

        if blocked and blocking.sphere_count > MAX_REASONABLE_BLOCK_SPHERES:
            warnings.append(
                f"{blocking.sphere_count} blocking spheres were needed; "
                "RASPA checks each on every insertion, which will be slow."
            )

    charge_source = (
        "cif" if use_cif_charges
        else "charge_equilibration"
        if settings.charge_equilibration and model.is_charged
        else "none"
    )

    if settings.charge_equilibration and not model.is_charged and not use_cif_charges:
        prepared.notes.append(
            f"Charge equilibration was requested but skipped: {model.name} "
            "is uncharged, so framework charges would have no effect."
        )

    step("Writing input files")
    pseudo_atoms = framework_pseudo_atoms(framework) + list(model.pseudo_atoms)

    files = {
        f"{name}.cif": write_framework_cif(
            framework, name, prepared.charges if use_cif_charges else None
        ),
        "pseudo_atoms.def": write_pseudo_atoms_def(pseudo_atoms),
        "force_field_mixing_rules.def": write_mixing_rules_def(pseudo_atoms),
        "force_field.def": FORCE_FIELD_DEF,
        f"{model.name}.def": write_molecule_def(model),
        "simulation.input": write_simulation_input(
            settings,
            framework_name=name,
            unit_cells=unit_cells,
            charge_source=charge_source,
            block=blocked,
        ),
    }

    if blocked:
        files[f"{name}.block"] = write_block_file(blocking)

    if charge_source == "charge_equilibration":
        for shared in (f"{name}.cif", "pseudo_atoms.def",
                       "force_field_mixing_rules.def", "force_field.def"):
            files[f"{QEQ_FOLDER}/{shared}"] = files[shared]
        files[f"{QEQ_FOLDER}/simulation.input"] = (
            write_charge_equilibration_input(name, framework.lattice.matrix)
        )
        files[RUN_SCRIPT] = write_run_script(name)

    summary = {
        "provenance": provenance(),
        "input": {
            "file": str(source) if source else None,
            "sha256": file_sha256(source) if source else None,
        },
        "framework_name": name,
        "settings": settings.as_dict(),
        "adsorbate": {
            "name": model.name,
            "model": model.description,
            "reference": model.reference,
        },
        "force_field": {
            "framework": "UFF Lennard-Jones (Rappe et al. 1992), one type "
            "per element",
            "mixing": "Lorentz-Berthelot",
            "lj_treatment": "shifted at the cutoff, no tail corrections",
            "cutoff_A": settings.cutoff_A,
            "framework_parameters": {
                atom.name: {"epsilon_K": atom.epsilon_K, "sigma_A": atom.sigma_A}
                for atom in pseudo_atoms
                if atom.name in symbols
            },
        },
        "charges": {
            "source": charge_source,
            "net_framework_charge_e": net_charge if use_cif_charges else None,
        },
        "framework": {
            "atoms": len(framework),
            "formula": framework.composition.reduced_formula,
            "cell_formula": framework.composition.formula,
            "dimensionality": prepared.framework_dimensionality,
            "disorder_sites_dropped": prepared.disorder_sites_dropped,
            "removed_molecules": [asdict(m) for m in prepared.removed],
        },
        "unit_cells": list(unit_cells),
        "blocking": None if blocking is None else {
            "status": blocking.status,
            "probe_radius_A": blocking.probe_radius_A,
            "spheres": blocking.sphere_count,
            "pocket_volume_A3": blocking.pocket_volume_A3,
            "accessible_volume_A3": blocking.accessible_volume_A3,
            "accessible_dimensionality": blocking.accessible_dimensionality,
            "grid_shape": list(blocking.grid_shape),
            "grid_spacing_A": blocking.grid_spacing_A,
        },
        "files": sorted(files) + ["README.txt", MANIFEST_NAME],
        "notes": list(prepared.notes),
        "warnings": list(warnings),
    }

    files["README.txt"] = write_readme(summary)
    files[MANIFEST_NAME] = json.dumps(_json_safe(summary), indent=2) + "\n"

    return RaspaExport(files=files, summary=summary, warnings=warnings)


MANIFEST_NAME = "the_void_raspa_export.json"


def _json_safe(value):
    from thevoid_core.services.report import _clean

    return _clean(value)


def write_raspa_export(export: RaspaExport, directory) -> list[Path]:
    """Write every file into directory (created if needed)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    for name, content in export.files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

        if name == RUN_SCRIPT:
            path.chmod(0o755)

        written.append(path)

    return written


def existing_export_files(export: RaspaExport, directory) -> list[str]:
    """Which of this export's files already exist in directory."""
    directory = Path(directory)
    return sorted(name for name in export.files if (directory / name).exists())


def write_readme(summary: dict) -> str:
    s = summary["settings"]
    fw = summary["framework"]
    ff = summary["force_field"]
    ads = summary["adsorbate"]
    charges = summary["charges"]
    blocking = summary["blocking"]
    name = summary["framework_name"]
    pressures = ", ".join(f"{p / 1e5:g} bar" for p in s["pressures_Pa"])

    lines = [
        f"RASPA2 input set for {name}, written by the Void "
        f"{summary['provenance']['software']['the_void']}",
        "=" * 72,
        "",
        "THIS IS A STARTING POINT, NOT A FINISHED SIMULATION SETUP.",
        "Every choice below was made by default; review each before you",
        "rely on, or publish, anything RASPA computes from these files.",
        "",
    ]

    if summary["charges"]["source"] == "charge_equilibration":
        lines += [
            f"Run:  ./{RUN_SCRIPT}  (two steps: framework charges by charge",
            f"equilibration in {QEQ_FOLDER}/, then the GCMC run here). Everything",
            "RASPA needs is here; nothing is read from its share directory.",
            "",
        ]
    else:
        lines += [
            "Run:  cd into this folder, then  simulate  (RASPA2). Everything",
            "RASPA needs is here; nothing is read from its share directory.",
            "",
        ]

    if summary["warnings"]:
        lines += ["WARNINGS", "--------"]
        for warning in summary["warnings"]:
            lines += _wrap(f"* {warning}")
        lines.append("")

    lines += [
        "What will be simulated",
        "----------------------",
        f"GCMC adsorption of {ads['name']} ({ads['model']}; {ads['reference']})",
        f"at {s['temperature_K']} K and {pressures} (one isotherm point per "
        "pressure).",
        f"{s['initialization_cycles']} initialisation + {s['cycles']} "
        f"production cycles, printed every {s['print_every']}.",
        "Fugacity from the Peng-Robinson equation of state (RASPA default).",
        "Moves: translation, "
        + ("rotation, " if ads["name"] != "CH4" else "")
        + "reinsertion, swap (insertion/deletion).",
        "",
        "Framework",
        "---------",
        f"{name}.cif: {fw['atoms']} atoms, {fw['cell_formula']} per cell, "
        "written in P1.",
        f"Rigid. Periodic in {fw['dimensionality']} dimension(s) by bond "
        "connectivity.",
        f"Unit cells: {' x '.join(str(n) for n in summary['unit_cells'])}, the "
        "fewest that keep every perpendicular width above twice the",
        f"{ff['cutoff_A']} A cutoff (minimum-image convention).",
    ]

    if fw["disorder_sites_dropped"]:
        lines.append(
            f"{fw['disorder_sites_dropped']} minor-disorder site(s) (occupancy "
            "< 0.5) were dropped."
        )

    if fw["removed_molecules"]:
        lines.append("Removed as non-framework (not bonded periodically):")
        for m in fw["removed_molecules"]:
            flag = "  <- may be a counter-ion" if m["possibly_ionic"] else ""
            lines.append(f"  {m['count']} x {m['formula']}{flag}")
    else:
        lines.append("No free molecules were removed.")

    lines += [
        "",
        "Force field",
        "-----------",
        f"Framework: {ff['framework']}, epsilon = D_i, sigma = x_i / 2^(1/6).",
        f"Adsorbate: TraPPE. Cross terms: {ff['mixing']}. LJ {ff['lj_treatment']}.",
        "Parameters (epsilon/kB K, sigma A):",
    ]

    for element, p in ff["framework_parameters"].items():
        lines.append(f"  {element:<3} {p['epsilon_K']:9.4f} {p['sigma_A']:8.5f}")

    lines += ["", "Charges", "-------"]

    if charges["source"] == "cif":
        lines.append(
            "Per-atom framework charges from the input CIF's _atom_site_charge "
            f"column (net {charges['net_framework_charge_e']:+.4f} e per cell)."
        )
    elif charges["source"] == "charge_equilibration":
        lines += _wrap(
            "Framework charges from RASPA's charge equilibration (its default "
            f"method), computed per atom on one unit cell by {QEQ_FOLDER}/"
            f"simulation.input and copied to {name}{QEQ_SUFFIX}.cif by "
            f"{RUN_SCRIPT}; the GCMC reads them with UseChargesFromCIFFile. "
            "Not averaged per element (SymmetrizeFrameworkCharges no)."
        )
    else:
        lines += _wrap(
            "None. The framework is uncharged: only Lennard-Jones "
            "framework-adsorbate interactions are included. Adsorbate-"
            "adsorbate electrostatics are still computed (Ewald) for charged "
            "adsorbates."
        )

    lines += ["", "Pocket blocking", "---------------"]

    if blocking is None:
        lines.append("Not requested.")
    elif blocking["status"] == "blocked":
        lines += _wrap(
            f"{blocking['spheres']} sphere(s) in {name}.block cover "
            f"{blocking['pocket_volume_A3']:.1f} A^3 of void a "
            f"{blocking['probe_radius_A']:.3f} A probe can fit into but not "
            "reach; insertions there are forbidden. Classified on the pore-"
            f"analysis grid ({' x '.join(map(str, blocking['grid_shape']))})."
        )
    elif blocking["status"] == "no pockets":
        lines.append(
            f"No inaccessible pockets for a {blocking['probe_radius_A']:.3f} A "
            "probe; nothing to block."
        )
    else:
        lines += _wrap(
            f"No channel percolates for a {blocking['probe_radius_A']:.3f} A "
            "probe, so nothing was blocked (see WARNINGS)."
        )

    lines += [
        "",
        "Not included, and worth considering",
        "-----------------------------------",
    ]
    lines += _wrap(
        "* Helium void fraction: RASPA's 'excess' loadings need "
        "HeliumVoidFraction for this framework (a Widom run with helium). "
        "Absolute loadings do not."
    )
    lines += _wrap(
        "* Framework flexibility, polarisation, and any adsorbate-specific "
        "metal-site parameters (UFF is generic and known to underbind open "
        "metal sites)."
    )
    lines += _wrap(
        "* Convergence: check that the loading has stopped drifting over "
        "the initialisation cycles, and that block averages agree."
    )

    if summary["notes"]:
        lines += ["", "Notes", "-----"]
        for note in summary["notes"]:
            lines += _wrap(f"* {note}")

    source = summary["input"]
    lines += ["", "Provenance", "----------"]

    if source["file"]:
        lines.append(f"Input: {Path(source['file']).name}")
        lines.append(f"SHA-256: {source['sha256']}")

    lines.append(f"Generated: {summary['provenance']['generated']}")
    lines.append(f"Full record: {MANIFEST_NAME}")

    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 72) -> list[str]:
    import textwrap

    return textwrap.wrap(text, width=width, subsequent_indent="  ") or [""]


def summary_for_report(export: RaspaExport, directory=None) -> dict:
    """The part of an export's record an analysis report carries."""
    s = export.summary
    return _json_safe(
        {
            "directory": str(directory) if directory else None,
            "framework_name": s["framework_name"],
            "adsorbate": s["adsorbate"]["name"],
            "temperature_K": s["settings"]["temperature_K"],
            "pressures_Pa": s["settings"]["pressures_Pa"],
            "cycles": s["settings"]["cycles"],
            "initialization_cycles": s["settings"]["initialization_cycles"],
            "cutoff_A": s["settings"]["cutoff_A"],
            "unit_cells": s["unit_cells"],
            "charges": s["charges"]["source"],
            "framework_atoms": s["framework"]["atoms"],
            "removed_molecules": [
                f"{m['count']} x {m['formula']}"
                for m in s["framework"]["removed_molecules"]
            ],
            "blocking": s["blocking"],
            "warnings": s["warnings"],
        }
    )
