"""Reading SHELX .res and .ins files (version 1.1.0).

Olex2, ShelXle and SHELXL itself keep a structure being refined as a
SHELX instruction file: the cell, the symmetry, the atoms with their
occupancies, displacement parameters and disorder parts, and (in a
.res) what the last refinement gave. Reading it lets the Void analyse a
structure straight from the refinement folder, before any CIF is made.

Only the model is read. Restraints, constraints and the weighting
scheme are instructions to the refinement program and are skipped. The
file is never written or changed.

The format's rules that matter here (SHELXL manual, "the .ins file"):

- An instruction starts in column 1. A line that starts with a space is
  a comment, unless the line before it ended in '=', which continues
  it. '!' starts a comment.
- An atom is: name, scattering-factor number (its place in SFAC), x, y,
  z, then optionally the site occupation factor (sof) and either one
  isotropic U or six anisotropic ones (U11 U22 U33 U23 U13 U12).
- A number may be coded as 10m + p: m = 1 fixes it at p; m > 1 makes it
  p times free variable m; m < -1 makes it p times (free variable -m
  minus 1), written out as |p| x (1 - fv). The free variables are on
  FVAR; the first is the overall scale factor.
- A sof includes the site symmetry: an atom on a twofold axis has sof
  0.5 when the site is full. The chemical occupancy, which is what a
  CIF records and what every analysis needs, is the sof times the
  number of symmetry operations that leave the atom where it is.
- "PART n sof" puts the atoms that follow in disorder part n, and, when
  the sof is given, gives them that sof.
- A negative isotropic U (between -0.5 and -5) is a riding hydrogen's:
  minus that many times the equivalent U of the last atom before it
  that is not a hydrogen.
- LATT gives the centring (1 P, 2 I, 3 R obverse, 4 F, 5 A, 6 B, 7 C),
  negative when the structure is not centrosymmetric. SYMM lists the
  other operators, without the identity, the centring or the inversion.
- The atom list ends at HKLF. After END, a .res holds the difference
  peaks (Q1, Q2 ...), which are not atoms.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

#: Every SHELXL instruction (its first four characters), so that an atom
#: is told from an instruction: an atom's name may not be one of these.
INSTRUCTIONS = frozenset("""
ABIN ACTA AFIX ANIS ANSC ANSR BASF BIND BLOC BOND BUMP CELL CGLS CHIV CONF
CONN DAMP DANG DEFS DELU DFIX DISP EADP END EQIV EXTI EXYZ FEND FLAT FMAP
FRAG FREE FVAR GRID HFIX HKLF HTAB ISOR LATT LAUE LIST L.S. MERG MOLE MORE
MOVE MPLA NCSY NEUT OMIT PART PLAN PRIG REM RESI RIGU RTAB SADI SAME SFAC
SHEL SIMU SIZE SPEC STIR SUMP SWAT SYMM TEMP TIME TITL TREF TWIN TWST UNIT
WGHT WIGL WPDB XNPD ZERR
""".split())

_CENTRING = {
    1: [(0, 0, 0)],
    2: [(0, 0, 0), (Fraction(1, 2),) * 3],
    3: [(0, 0, 0), (Fraction(2, 3), Fraction(1, 3), Fraction(1, 3)),
        (Fraction(1, 3), Fraction(2, 3), Fraction(2, 3))],
    4: [(0, 0, 0), (0, Fraction(1, 2), Fraction(1, 2)),
        (Fraction(1, 2), 0, Fraction(1, 2)), (Fraction(1, 2), Fraction(1, 2), 0)],
    5: [(0, 0, 0), (0, Fraction(1, 2), Fraction(1, 2))],
    6: [(0, 0, 0), (Fraction(1, 2), 0, Fraction(1, 2))],
    7: [(0, 0, 0), (Fraction(1, 2), Fraction(1, 2), 0)],
}


class ShelxError(ValueError):
    """The file is not a SHELX file this reader can use; says why."""


@dataclass
class ShelxAtom:
    name: str
    element: str
    frac: tuple
    #: The site occupation factor as SHELX holds it (site symmetry
    #: included), free variables applied.
    sof: float
    #: The chemical occupancy: sof x site-symmetry order.
    occupancy: float
    part: int = 0
    residue: int = 0
    #: Isotropic U (A^2), or the equivalent one of an anisotropic atom.
    u_iso: float | None = None
    #: (U11, U22, U33, U23, U13, U12), or None for an isotropic atom.
    u_aniso: tuple | None = None
    #: How many symmetry operations leave the atom where it is.
    site_symmetry_order: int = 1

    @property
    def label(self) -> str:
        """The name as SHELXL writes it in a CIF: C1, or C1_3 in residue 3."""
        return f"{self.name}_{self.residue}" if self.residue else self.name


@dataclass
class ShelxModel:
    title: str = ""
    wavelength: float | None = None
    cell: tuple = ()
    cell_esd: tuple = ()
    z: float | None = None
    latt: int = 1
    #: Every operator of the space group, as (3x3 integer matrix,
    #: translation of Fractions), identity first.
    operators: list = field(default_factory=list)
    sfac: list = field(default_factory=list)
    unit: list = field(default_factory=list)
    fvar: list = field(default_factory=list)
    atoms: list = field(default_factory=list)
    temperature_C: float | None = None
    hklf: int | None = None
    twinned: bool = False
    solvent_mask: bool = False
    #: What the last refinement reported (a .res only): r1, r1_all,
    #: wr2, goof, reflections_observed, reflections_all, parameters,
    #: restraints, peak, hole.
    refinement: dict = field(default_factory=dict)
    difference_peaks: int = 0

    @property
    def centrosymmetric(self) -> bool:
        return self.latt > 0


# ----------------------------------------------------------------------
# Numbers and symmetry
# ----------------------------------------------------------------------


def decode(value: float, fvar: list) -> float:
    """A SHELX number coded as 10m + p (module docstring)."""
    m = math.ceil((value - 5.0) / 10.0)
    p = value - 10.0 * m
    if abs(m) <= 1:
        return p
    index = abs(m)
    if index > len(fvar):
        raise ShelxError(f"an atom uses free variable {index}, but FVAR has only {len(fvar)}")
    free = fvar[index - 1]
    return p * free if m > 0 else p * (free - 1.0)


def parse_operator(text: str):
    """'-X, 0.5+Y, 1/2-Z' -> (3x3 integer matrix, translation)."""
    parts = [p.strip().upper().replace(" ", "") for p in text.split(",")]
    if len(parts) != 3:
        raise ShelxError(f"cannot read the symmetry operator {text!r}")
    rotation = np.zeros((3, 3), dtype=int)
    translation = []
    for row, part in enumerate(parts):
        shift = Fraction(0)
        for sign, number, axis in re.findall(r"([+-]?)(\d*\.?\d*(?:/\d+)?)([XYZ]?)", part):
            if not number and not axis:
                continue
            factor = -1 if sign == "-" else 1
            if axis:
                if number not in ("", "1"):
                    raise ShelxError(f"cannot read the symmetry operator {text!r}")
                rotation[row, "XYZ".index(axis)] += factor
            else:
                shift += factor * Fraction(number).limit_denominator(12)
        translation.append(shift % 1)
    if abs(round(float(np.linalg.det(rotation)))) != 1:
        raise ShelxError(f"cannot read the symmetry operator {text!r}")
    return rotation, tuple(translation)


def operator_xyz(operator) -> str:
    """(matrix, translation) -> 'x,y,z' as a CIF writes it."""
    rotation, translation = operator
    rows = []
    for row in range(3):
        text = ""
        for column, axis in enumerate("xyz"):
            value = rotation[row, column]
            if value:
                text += ("-" if value < 0 else "+" if text else "") + axis
        shift = Fraction(translation[row]) % 1
        if shift:
            text += f"+{shift.numerator}/{shift.denominator}"
        rows.append(text or "0")
    return ",".join(rows)


def space_group_operators(latt: int, symm: list) -> list:
    """Every operator: identity and SYMM, with the inversion when LATT
    is positive, each with every centring translation."""
    identity = (np.eye(3, dtype=int), (Fraction(0),) * 3)
    base = [identity] + list(symm)
    if latt > 0:
        base += [(-rotation, tuple((-t) % 1 for t in translation))
                 for rotation, translation in list(base)]
    centring = _CENTRING.get(abs(latt))
    if centring is None:
        raise ShelxError(f"LATT {latt} is not a lattice type (1 to 7)")
    operators, seen = [], set()
    for shift in centring:
        for rotation, translation in base:
            moved = tuple((Fraction(t) + Fraction(s)) % 1 for t, s in zip(translation, shift))
            key = (tuple(rotation.flatten()), moved)
            if key not in seen:
                seen.add(key)
                operators.append((rotation, moved))
    return operators


def lattice_matrix(cell) -> np.ndarray:
    """Rows a, b, c in Cartesian angstroms (a along x, b in the xy plane)."""
    a, b, c, alpha, beta, gamma = cell
    alpha, beta, gamma = (math.radians(v) for v in (alpha, beta, gamma))
    cx = c * math.cos(beta)
    cy = c * (math.cos(alpha) - math.cos(beta) * math.cos(gamma)) / math.sin(gamma)
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, 0.0, 0.0],
                     [b * math.cos(gamma), b * math.sin(gamma), 0.0],
                     [cx, cy, cz]])


def site_symmetry_order(frac, operators, matrix, tolerance: float = 0.01) -> int:
    """How many operators leave the position where it is (to within
    `tolerance` angstroms, lattice translations allowed)."""
    position = np.asarray(frac, dtype=float)
    count = 0
    for rotation, translation in operators:
        image = rotation @ position + np.array([float(t) for t in translation])
        delta = image - position
        delta -= np.round(delta)
        if np.linalg.norm(delta @ matrix) < tolerance:
            count += 1
    return max(count, 1)


def equivalent_u(u_aniso, cell) -> float:
    """Ueq = one third of the trace of the orthogonalised U tensor."""
    matrix = lattice_matrix(cell)
    reciprocal = np.linalg.inv(matrix).T          # rows a*, b*, c*
    lengths = np.linalg.norm(reciprocal, axis=1)
    u11, u22, u33, u23, u13, u12 = u_aniso
    u = np.array([[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]])
    scaled = np.diag(lengths) @ u @ np.diag(lengths)
    cartesian = matrix.T @ scaled @ matrix
    return float(np.trace(cartesian) / 3.0)


# ----------------------------------------------------------------------
# The file
# ----------------------------------------------------------------------


def _logical_lines(text: str):
    """The instructions, continuation lines joined, comments removed.
    Yields (keyword or None, the line)."""
    pending = None
    for raw in text.splitlines():
        line = raw.split("!", 1)[0].rstrip()
        if pending is not None:
            line = pending + " " + line.strip()
            pending = None
        elif not line or line[0] in " \t":
            continue
        if line.endswith("="):
            pending = line[:-1].rstrip()
            continue
        yield line
    if pending:
        yield pending


def _keyword(token: str) -> str:
    return token.split("_", 1)[0].upper()[:4]


def _floats(tokens):
    try:
        return [float(t) for t in tokens]
    except ValueError:
        return None


_REFINEMENT_PATTERNS = (
    (re.compile(r"R1\s*=\s*([\d.]+)\s+for\s+(\d+)\s+Fo\s*>\s*4sig\(Fo\)\s+and\s+([\d.]+)"
                r"\s+for\s+all\s+(\d+)\s+data"),
     ("r1", "reflections_observed", "r1_all", "reflections_all")),
    (re.compile(r"wR2\s*=\s*([\d.]+),\s*GooF\s*=\s*S\s*=\s*([\d.]+)"), ("wr2", "goof")),
    (re.compile(r"(\d+)\s+parameters\s+refined\s+using\s+(\d+)\s+restraints"),
     ("parameters", "restraints")),
    (re.compile(r"Highest difference peak\s+([-\d.]+),\s*deepest hole\s+([-\d.]+)"),
     ("peak", "hole")),
)


def _read_refinement(line: str, into: dict) -> None:
    for pattern, names in _REFINEMENT_PATTERNS:
        match = pattern.search(line)
        if match:
            for name, value in zip(names, match.groups()):
                into[name] = int(value) if value.isdigit() else float(value)


def read_shelx(path) -> ShelxModel:
    """The model in a SHELX .res or .ins file."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_shelx(text)


def parse_shelx(text: str) -> ShelxModel:
    model = ShelxModel()
    symm = []
    part, part_sof, residue = 0, None, 0
    in_fragment = False
    ended = False
    raw_atoms = []

    for line in _logical_lines(text):
        tokens = line.split()
        keyword = _keyword(tokens[0])

        if ended:
            if keyword == "REM":
                _read_refinement(line, model.refinement)
            elif re.fullmatch(r"Q\d+", tokens[0], re.IGNORECASE):
                model.difference_peaks += 1
            continue

        if keyword == "REM":
            _read_refinement(line, model.refinement)
        elif keyword == "TITL":
            model.title = line[4:].strip()
        elif keyword == "CELL":
            numbers = _floats(tokens[1:8])
            if not numbers or len(numbers) != 7:
                raise ShelxError("CELL must give the wavelength and six cell parameters")
            model.wavelength, model.cell = numbers[0], tuple(numbers[1:])
        elif keyword == "ZERR":
            numbers = _floats(tokens[1:8]) or []
            if numbers:
                model.z, model.cell_esd = numbers[0], tuple(numbers[1:])
        elif keyword == "LATT":
            model.latt = int(float(tokens[1])) if len(tokens) > 1 else 1
        elif keyword == "SYMM":
            symm.append(parse_operator(line[4:]))
        elif keyword == "SFAC":
            numbers = _floats(tokens[2:])
            if numbers and len(numbers) >= 9:
                model.sfac.append(tokens[1])       # one element, explicit factors
            else:
                model.sfac.extend(tokens[1:])
        elif keyword == "UNIT":
            model.unit = _floats(tokens[1:]) or []
        elif keyword == "FVAR":
            model.fvar.extend(_floats(tokens[1:]) or [])
        elif keyword == "TEMP":
            numbers = _floats(tokens[1:2])
            model.temperature_C = numbers[0] if numbers else None
        elif keyword == "PART":
            part = int(float(tokens[1])) if len(tokens) > 1 else 0
            part_sof = float(tokens[2]) if len(tokens) > 2 else None
        elif keyword == "RESI":
            numbers = [t for t in tokens[1:] if re.fullmatch(r"-?\d+", t)]
            residue = int(numbers[0]) if numbers else 0
        elif keyword == "FRAG":
            in_fragment = True
        elif keyword == "FEND":
            in_fragment = False
        elif keyword in ("TWIN", "BASF"):
            model.twinned = True
        elif keyword == "ABIN":
            model.solvent_mask = True
        elif keyword == "HKLF":
            numbers = _floats(tokens[1:2])
            model.hklf = int(numbers[0]) if numbers else None
            ended = True
        elif keyword == "END":
            ended = True
        elif keyword in INSTRUCTIONS or tokens[0].startswith("+"):
            continue
        elif not in_fragment:
            numbers = _floats(tokens[1:])
            if (numbers is None or len(numbers) < 4 or numbers[0] != int(numbers[0])
                    or not 1 <= int(numbers[0]) <= max(len(model.sfac), 1)):
                raise ShelxError(f"cannot read this line as an atom or an instruction: "
                                 f"{line[:60]!r}")
            raw_atoms.append((tokens[0], numbers, part, part_sof, residue))

    if len(model.cell) != 6:
        raise ShelxError("no CELL instruction: this does not look like a SHELX file")
    if not model.sfac:
        raise ShelxError("no SFAC instruction, so the atoms' elements are not known")

    model.operators = space_group_operators(model.latt, symm)
    matrix = lattice_matrix(model.cell)
    last_heavy_u = None

    for name, numbers, atom_part, atom_part_sof, atom_residue in raw_atoms:
        element = _element(model.sfac[int(numbers[0]) - 1])
        frac = tuple(decode(v, model.fvar) for v in numbers[1:4])
        raw_sof = atom_part_sof if atom_part_sof is not None else (
            numbers[4] if len(numbers) > 4 else 11.0)
        sof = decode(raw_sof, model.fvar)
        order = site_symmetry_order(frac, model.operators, matrix)
        occupancy = round(sof * order, 5)
        if abs(occupancy - 1.0) < 2e-4:
            occupancy = 1.0

        u_values = numbers[5:]
        u_iso, u_aniso = None, None
        if len(u_values) >= 6:
            u_aniso = tuple(decode(v, model.fvar) for v in u_values[:6])
            u_iso = equivalent_u(u_aniso, model.cell)
        elif u_values:
            value = u_values[0]
            if -5.0 < value <= -0.5:
                u_iso = -value * last_heavy_u if last_heavy_u is not None else None
            else:
                u_iso = decode(value, model.fvar)
        if element not in ("H", "D") and u_iso is not None:
            last_heavy_u = u_iso

        model.atoms.append(ShelxAtom(
            name=name, element=element, frac=frac, sof=sof, occupancy=occupancy,
            part=atom_part, residue=atom_residue, u_iso=u_iso, u_aniso=u_aniso,
            site_symmetry_order=order))

    if not model.atoms:
        raise ShelxError("the file has a cell but no atoms")
    return model


def _element(sfac_label: str) -> str:
    """'CL' -> 'Cl', 'Ni2+' -> 'Ni'."""
    match = re.match(r"[A-Za-z]{1,2}", sfac_label)
    if not match:
        raise ShelxError(f"SFAC entry {sfac_label!r} is not an element")
    return match.group(0).capitalize()


# ----------------------------------------------------------------------
# As a CIF
# ----------------------------------------------------------------------


def shelx_to_cif(model: ShelxModel, name: str = "shelx") -> str:
    """The model as CIF text, so that it is read exactly as a CIF is
    (the same symmetry expansion, occupancies and disorder groups)."""
    a, b, c, alpha, beta, gamma = model.cell
    block = re.sub(r"[^A-Za-z0-9_.-]", "_", name) or "shelx"
    lines = [
        f"data_{block}",
        f"_cell_length_a {a}", f"_cell_length_b {b}", f"_cell_length_c {c}",
        f"_cell_angle_alpha {alpha}", f"_cell_angle_beta {beta}",
        f"_cell_angle_gamma {gamma}",
    ]
    if model.z is not None:
        lines.append(f"_cell_formula_units_Z {model.z:g}")
    lines += ["loop_", "_space_group_symop_operation_xyz"]
    lines += [f"'{operator_xyz(op)}'" for op in model.operators]
    lines += ["loop_", "_atom_site_label", "_atom_site_type_symbol",
              "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z",
              "_atom_site_U_iso_or_equiv", "_atom_site_adp_type",
              "_atom_site_occupancy", "_atom_site_disorder_group"]
    labels = unique_labels(model.atoms)
    for atom, label in zip(model.atoms, labels):
        u = f"{atom.u_iso:.5f}" if atom.u_iso is not None else "?"
        kind = "Uani" if atom.u_aniso else "Uiso"
        x, y, z = atom.frac
        lines.append(f"{label} {atom.element} {x:.6f} {y:.6f} {z:.6f} {u} {kind} "
                     f"{atom.occupancy:.5f} {atom.part if atom.part else '.'}")
    anisotropic = [(label, atom) for atom, label in zip(model.atoms, labels) if atom.u_aniso]
    if anisotropic:
        lines += ["loop_", "_atom_site_aniso_label", "_atom_site_aniso_U_11",
                  "_atom_site_aniso_U_22", "_atom_site_aniso_U_33",
                  "_atom_site_aniso_U_23", "_atom_site_aniso_U_13",
                  "_atom_site_aniso_U_12"]
        for label, atom in anisotropic:
            lines.append(label + " " + " ".join(f"{v:.5f}" for v in atom.u_aniso))
    return "\n".join(lines) + "\n"


def unique_labels(atoms) -> list:
    """Each atom's CIF label; a repeated one gets a letter (C1, C1B ...),
    which only a file with a repeated name needs."""
    seen: dict[str, int] = {}
    labels = []
    for atom in atoms:
        label = atom.label
        count = seen.get(label, 0)
        seen[label] = count + 1
        labels.append(label if count == 0 else f"{label}{chr(ord('A') + count)}")
    return labels
