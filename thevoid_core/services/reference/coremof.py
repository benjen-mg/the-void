"""CoRE MOF 2024: the bundled property index, and finding a structure's
entry in it (round 56).

The index (data/coremof2024.json.gz, built by
tools/build_coremof_index.py) holds the 17,202 computation-ready
entries of CoRE MOF 2024 - Zhao et al., Matter 8, 102140 (2025),
CC-BY-4.0 - with their Zeo++ pore geometry, topology, open metal sites
and stability data. It is a trimmed copy, frozen when the Void was
released; `index_info()` says which version.

Finding an entry
----------------
A structure is looked up by name first, because names are how these
files travel: a CoRE MOF 2024 id ("2010[Zr][fcu]3[ASR]1") or source
refcode as the file name, a CSD refcode as the file name ("RUBTAK01.cif",
CoRE MOF 2019's "RUBTAK01_clean.cif"), the CIF's data block, or its
_database_code_CSD. A name alone does not make a match, though: a
refcode can have ASR and FSR entries (with and without bound solvent),
and a file can be renamed. Every candidate is checked against the
structure itself - atom count, density, metals - and a name whose
checks fail is reported as a name match only.

With no usable name, entries are searched by what the index records
about the structure: same metals, atom counts in a whole-number ratio
(CoRE MOF often stores the primitive cell), density within 0.5% and
mean atomic mass within 0.2% (a composition check the index allows
without storing formulas). The space group is not used: CoRE records
different ones for an ASR entry and its identical FSR twin (74 and 15
for 2010[Cr][bpq]3), so it would reject true matches. A hit is reported as
"possible" - a composition-and-density coincidence is not an identity.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

INDEX_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "coremof2024.json.gz"

#: Relative density difference within which a candidate is the same
#: structure. CoRE's densities are Zeo++'s (cell mass / volume); the
#: Void's are pymatgen's, with the same atomic masses to 4-5 digits.
DENSITY_TOLERANCE = 0.005

#: Elements CoRE MOF lists among an entry's metals that the Void's own
#: classification does not call metals (178 entries: Si 143, As 13,
#: Sb 11, Ge 10, Te 1). Counted as metals for matching, or every one of
#: those entries fails its metals check (round 58: found fetching
#: 2017[ZnSi][nan]3[ASR]1, flagged "no: metals" against its own entry).
COREMOF_METALLOIDS = frozenset({"Si", "As", "Sb", "Ge", "Te"})

#: Relative difference in mean atomic mass (cell mass / atoms) within
#: which two structures have the same composition. CALF-20 (16.13 amu)
#: and VONDEJ (13.72 amu) share metal, density and space group.
MASS_TOLERANCE = 0.002

#: Largest whole-number ratio between a structure's cell and an
#: entry's cell still taken as the same crystal (a conventional F or I
#: cell holds 4 or 2 primitive cells; 8 leaves room for supercells).
MAX_CELL_RATIO = 8

#: A CSD refcode: six letters, optionally two digits.
_REFCODE = re.compile(r"^[A-Z]{6}(\d{2})?$")
#: Suffixes CoRE MOF (2014, 2019, 2024) and common tools add to a
#: refcode in a file name.
_SUFFIXES = re.compile(
    r"(_clean(_h)?|_charged|_ddec|_repeat|_pacman|_asr|_fsr|_ion|_ncr|_p1|_conventional"
    r"|_primitive|_opt)+$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CoreMofEntry:
    """One CoRE MOF 2024 entry; field names follow the index columns."""

    values: dict

    def __getattr__(self, name):
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name) from None

    @property
    def metal_set(self) -> frozenset:
        return frozenset(m for m in (self.values.get("metals") or "").split(",") if m)

    @property
    def downloadable(self) -> bool:
        """Only structures from papers' SI are redistributed by CoRE MOF;
        CSD-derived ones need a CSD licence."""
        return self.values.get("source") == "SI"


@dataclass
class CoreMofIndex:
    source: dict
    columns: list
    entries: list  # [CoreMofEntry]
    by_id: dict = field(default_factory=dict)
    by_refcode: dict = field(default_factory=dict)
    by_csd: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.entries)


@lru_cache(maxsize=2)
def load_index(path: str | None = None) -> CoreMofIndex:
    """The bundled index (or one at `path`), loaded once."""
    with gzip.open(path or INDEX_PATH, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    columns = data["columns"]
    entries = [CoreMofEntry(dict(zip(columns, row))) for row in data["rows"]]
    index = CoreMofIndex(source=data["source"], columns=columns, entries=entries)
    for entry in entries:
        index.by_id[entry.id] = entry
        index.by_refcode.setdefault(entry.refcode.upper(), []).append(entry)
        if entry.csd_refcode:
            index.by_csd.setdefault(entry.csd_refcode.upper(), []).append(entry)
    return index


def index_info() -> dict:
    """Where the bundled values came from, for reports and the UI."""
    return dict(load_index().source)


# ----------------------------------------------------------------------
# Names
# ----------------------------------------------------------------------


def names_from_cif_text(text: str) -> list[str]:
    """Data block names and _database_code_CSD values in a CIF."""
    names = re.findall(r"^\s*data_(\S+)", text, re.MULTILINE)
    names += re.findall(r"^\s*_database_code_CSD\s+'?([A-Za-z0-9]+)'?", text,
                        re.MULTILINE | re.IGNORECASE)
    return names


def candidate_names(path=None, cif_text: str | None = None) -> list[str]:
    """Names to look a structure up by, most specific first."""
    raw = []
    if path is not None:
        raw.append(Path(path).stem)
    if cif_text is None and path is not None and str(path).lower().endswith(".cif"):
        try:
            cif_text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            cif_text = None
    if cif_text:
        raw += names_from_cif_text(cif_text)

    names = []
    for name in raw:
        for candidate in (name, _SUFFIXES.sub("", name), re.split(r"[_\-. ]", name)[0]):
            candidate = candidate.strip()
            if candidate and candidate not in names:
                names.append(candidate)
    return names


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


@dataclass
class StructureFacts:
    """What a match is checked against."""

    n_atoms: int
    density_g_cm3: float
    metals: frozenset
    space_group_number: int | None = None
    #: Cell mass / atom count, amu: independent of the cell chosen, and
    #: different for different compositions even at equal density.
    mean_atomic_mass: float | None = None

    @classmethod
    def of(cls, structure):
        from thevoid_core.data.metal_classification import is_metal
        from thevoid_core.services.pore_analysis import major_occupancy_structure
        from thevoid_core.services.report import crystal_density

        ordered = major_occupancy_structure(structure)
        metals = frozenset(
            site.specie.symbol for site in ordered.sites
            if hasattr(site, "specie")
            and (is_metal(site.specie.symbol) or site.specie.symbol in COREMOF_METALLOIDS)
        )
        number = None  # not used for matching (see the module docstring)
        mean_mass = float(ordered.composition.weight) / len(ordered) if len(ordered) else None
        return cls(len(ordered), crystal_density(ordered), metals, number, mean_mass)


@dataclass
class Match:
    entry: CoreMofEntry
    #: "identified": found by name and consistent with the structure.
    #: "name only": found by name, but the structure differs (atom count
    #: or density) - a related entry, not this structure's values.
    #: "possible": not by name; same metals, cell contents, density and
    #: mean atomic mass.
    confidence: str
    how: str
    checks: dict

    @property
    def is_identity(self) -> bool:
        return self.confidence == "identified"


def cell_ratio(n_structure: int, n_entry) -> int | None:
    """How many of the entry's cells make this structure's cell, when
    that is a whole number up to MAX_CELL_RATIO (either way round): CoRE
    MOF stores many structures as primitive cells - IRMOF-1 has 106
    atoms there, 424 in its conventional cell. None if neither cell is a
    whole multiple of the other."""
    if not n_entry or not n_structure:
        return None
    big, small = max(n_structure, n_entry), min(n_structure, n_entry)
    if big % small:
        return None
    ratio = big // small
    return ratio if ratio <= MAX_CELL_RATIO else None


def check(entry: CoreMofEntry, facts: StructureFacts) -> dict:
    """Per-property agreement between an entry and a structure:
    {property: (structure's value, entry's value, agrees)}. Atom counts
    agree when one cell is a whole multiple of the other (see
    cell_ratio); density, which does not depend on the cell chosen, is
    what then tells two determinations apart."""
    density = entry.density_g_cm3
    ratio = cell_ratio(facts.n_atoms, entry.n_atoms)
    result = {
        "n_atoms": (facts.n_atoms, entry.n_atoms, ratio is not None),
        "density_g_cm3": (
            facts.density_g_cm3, density,
            density is not None and abs(facts.density_g_cm3 - density) <= DENSITY_TOLERANCE * density,
        ),
        "metals": (
            ",".join(sorted(facts.metals)), entry.metals,
            not entry.metal_set or facts.metals == entry.metal_set,
        ),
    }
    mass = entry.mean_atomic_mass
    if facts.mean_atomic_mass is not None and mass:
        result["mean_atomic_mass"] = (
            facts.mean_atomic_mass, mass,
            abs(facts.mean_atomic_mass - mass) <= MASS_TOLERANCE * mass,
        )
    return result


def _consistent(checks: dict) -> bool:
    return all(ok for _, _, ok in checks.values())


def find_matches(structure=None, *, path=None, cif_text=None, facts=None,
                 index: CoreMofIndex | None = None, search_by_composition=True) -> list[Match]:
    """Entries of CoRE MOF 2024 that are, or may be, this structure -
    best first. An empty list means nothing plausible was found."""
    index = index or load_index()
    if facts is None:
        facts = StructureFacts.of(structure)

    seen = set()
    matches = []
    for name in candidate_names(path, cif_text):
        hits = []
        if name in index.by_id:
            hits = [(index.by_id[name], f"CoRE MOF 2024 id {name}")]
        elif name.upper() in index.by_refcode:
            hits = [(e, f"source file name {name}") for e in index.by_refcode[name.upper()]]
        elif _REFCODE.match(name.upper()) and name.upper() in index.by_csd:
            hits = [(e, f"CSD refcode {name.upper()}") for e in index.by_csd[name.upper()]]
        for entry, how in hits:
            if entry.id in seen:
                continue
            seen.add(entry.id)
            checks = check(entry, facts)
            confidence = "identified" if _consistent(checks) else "name only"
            matches.append(Match(entry, confidence, how, checks))

    # Run even after an identification: other entries of the same
    # structure (its FSR twin, another determination) are listed too.
    if search_by_composition:
        for entry in index.entries:
            if entry.id in seen or cell_ratio(facts.n_atoms, entry.n_atoms) is None:
                continue
            if entry.metal_set != facts.metals:
                continue
            checks = check(entry, facts)
            if not _consistent(checks):
                continue
            seen.add(entry.id)
            matches.append(Match(entry, "possible",
                                 "same metals, cell contents, density and composition", checks))

    order = {"identified": 0, "possible": 1, "name only": 2}

    def closeness(match):
        ours, theirs, _ = match.checks["density_g_cm3"]
        return abs(ours - theirs) / theirs if theirs else 1.0

    matches.sort(key=lambda m: (order[m.confidence], closeness(m)))
    return matches


def search(index: CoreMofIndex | None = None, *, subset=None, source=None,
           pld_min=None, pld_max=None, lcd_min=None, lcd_max=None,
           asa_min=None, metals=None, topology=None, has_oms=None,
           downloadable_only=False, limit=None) -> list[CoreMofEntry]:
    """Entries passing every given filter, in index order. `metals`
    matches entries containing all the named metals; `topology` the
    single-node net code, case-insensitively."""
    index = index or load_index()
    wanted_metals = {m.strip() for m in (metals or []) if m.strip()}
    found = []
    for entry in index.entries:
        if subset and entry.subset != subset:
            continue
        if source and entry.source != source:
            continue
        if downloadable_only and not entry.downloadable:
            continue
        if not _within(entry.pld_A, pld_min, pld_max):
            continue
        if not _within(entry.lcd_A, lcd_min, lcd_max):
            continue
        if asa_min is not None and not _within(entry.asa_m2_g, asa_min, None):
            continue
        if wanted_metals and not wanted_metals <= entry.metal_set:
            continue
        if topology and (entry.topology or "").lower() != topology.lower():
            continue
        if has_oms is not None and entry.has_oms is not has_oms:
            continue
        found.append(entry)
        if limit and len(found) >= limit:
            break
    return found


def _within(value, low, high) -> bool:
    if low is None and high is None:
        return True
    if value is None:
        return False
    return (low is None or value >= low) and (high is None or value <= high)
