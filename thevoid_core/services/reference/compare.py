"""The Void's numbers beside a reference database's (round 56).

A reference value is only a check on the Void if both were computed the
same way. CoRE MOF 2024 used Zeo++ with its default (CCDC) van der
Waals radii and a 1.655 A probe; the Void's own analyses use other radii
(data/element_radii.py) and, by default, a 1.82 A N2 probe. Comparing
those directly mixes a difference of convention with any difference of
method or structure. So the comparison recomputes what it compares
under the reference's convention - CCDCRadiusProvider, the reference
probe - and says so on every row. The Void's standard analyses are
untouched.

What is compared, and why only this:
    LCD, PLD                       geometry only, no probe
    channel ASA, pocket (N)ASA     m^2/g, at the reference probe
    topology                       the single-node net code
    open metal sites               present or not
    density, atoms per cell        identity checks, not results
Void fraction and pore volume are left out: CoRE's file does not say
which probe or definition they use, and a comparison that guesses the
definition would test the guess.

A comparison against a "possible" or "name only" match is still made,
but labelled: its differences may be differences of structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

#: CoRE MOF 2024's Zeo++ probe for surface areas and volumes, A; the
#: database's ASA/NASA values are reproduced with it (round 56).
COREMOF_PROBE_RADIUS = 1.655

#: Differences worth a flag. Radii are matched, so what remains is
#: method and sampling: the Void's LCD/PLD agree with Zeo++ to a few
#: hundredths of an angstrom on the same radii (round 50), and surface
#: sampling scatters by 1-1.5% (round 50).
FLAG_DIAMETER_A = 0.15
FLAG_AREA_RELATIVE = 0.05
FLAG_AREA_ABSOLUTE_M2_G = 25.0


@dataclass
class ComparisonRow:
    key: str
    label: str
    void: object
    reference: object
    unit: str = ""
    #: void - reference for numbers; None for categories.
    difference: float | None = None
    #: True when the two agree (numbers: within the flag thresholds).
    agrees: bool | None = None
    note: str = ""


@dataclass
class ReferenceComparison:
    database: str
    entry_id: str
    confidence: str
    how: str
    rows: list[ComparisonRow]
    parameters: dict
    notes: list[str] = field(default_factory=list)
    reference_values: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)
    seconds: float = 0.0

    def row(self, key) -> ComparisonRow | None:
        return next((r for r in self.rows if r.key == key), None)

    @property
    def flagged(self) -> list[ComparisonRow]:
        return [r for r in self.rows if r.agrees is False]


def _numeric_row(key, label, void, reference, unit, *, absolute=None, relative=None,
                 floor=None, note=""):
    if void is None or reference is None:
        return ComparisonRow(key, label, void, reference, unit, None, None, note)
    difference = float(void) - float(reference)
    tolerance = 0.0
    if absolute is not None:
        tolerance = max(tolerance, absolute)
    if relative is not None:
        tolerance = max(tolerance, relative * abs(float(reference)))
    if floor is not None:
        tolerance = max(tolerance, floor)
    return ComparisonRow(key, label, float(void), float(reference), unit, difference,
                         abs(difference) <= tolerance + 1e-12, note)


def compare_with_coremof(structure, match, *, samples_per_atom: int = 1000,
                         grid_spacing: float = 0.3, topology: bool = True,
                         oms: bool = True, progress=None) -> ReferenceComparison:
    """Compute the Void's values under CoRE MOF's convention and set
    them beside `match.entry`'s (a coremof.Match). Each part that fails
    is recorded in .errors and the others still run."""
    from thevoid_core.data.element_radii import CCDCRadiusProvider
    from thevoid_core.services.pore_analysis import major_occupancy_structure
    from thevoid_core.services.reference.coremof import StructureFacts, cell_ratio

    start = time.time()
    entry = match.entry
    radii = CCDCRadiusProvider()
    structure = major_occupancy_structure(structure)
    facts = StructureFacts.of(structure)
    errors = {}
    rows = []

    def step(name):
        if progress is not None:
            progress(name)

    ratio = cell_ratio(facts.n_atoms, entry.n_atoms)
    rows.append(ComparisonRow(
        "n_atoms", "Atoms per cell", facts.n_atoms, entry.n_atoms, "",
        None, ratio is not None,
        note=("" if ratio in (None, 1) else
              f"cells differ by a factor of {ratio} (primitive vs conventional)"),
    ))
    rows.append(_numeric_row("density", "Density", facts.density_g_cm3,
                             entry.density_g_cm3, "g/cm³", relative=0.005,
                             note="identity check"))

    lcd = pld = None
    step("pore geometry")
    try:
        from thevoid_core.services.pore_analysis import analyze_pores

        pores = analyze_pores(structure, probe_radius=COREMOF_PROBE_RADIUS,
                              radius_provider=radii, grid_spacing=grid_spacing)
        lcd, pld = pores.lcd, pores.pld
    except Exception as exc:
        errors["pore"] = f"{type(exc).__name__}: {exc}"
    rows.append(_numeric_row("lcd", "LCD", lcd, entry.lcd_A, "Å", absolute=FLAG_DIAMETER_A))
    rows.append(_numeric_row("pld", "PLD", pld, entry.pld_A, "Å", absolute=FLAG_DIAMETER_A))

    asa = nasa = None
    step("surface area")
    try:
        from thevoid_core.services.surface_area import analyze_surface_area

        area = analyze_surface_area(structure, probe_radius=COREMOF_PROBE_RADIUS,
                                    probe_name="CoRE MOF (1.655 Å)", radius_provider=radii,
                                    n_samples_per_atom=samples_per_atom)
        asa = area.channel_asa_per_mass
        # Pocket area per gram, as channel_asa_per_mass is computed:
        # A^2 per cell -> m^2, over the cell's mass in grams.
        nasa = area.pocket_accessible_area * 1.0e-20 / area.cell_mass_grams
    except Exception as exc:
        errors["surface_area"] = f"{type(exc).__name__}: {exc}"
    rows.append(_numeric_row("asa", "Channel ASA", asa, entry.asa_m2_g, "m²/g",
                             relative=FLAG_AREA_RELATIVE, floor=FLAG_AREA_ABSOLUTE_M2_G))
    rows.append(_numeric_row("nasa", "Pocket ASA (NASA)", nasa, entry.nasa_m2_g, "m²/g",
                             relative=FLAG_AREA_RELATIVE, floor=FLAG_AREA_ABSOLUTE_M2_G))

    if topology:
        step("topology")
        void_net = None
        try:
            from thevoid_core.services.topology import analyze_topology

            result = analyze_topology(structure)
            void_net = result.rcsr or result.summary
        except Exception as exc:
            errors["topology"] = f"{type(exc).__name__}: {exc}"
        reference_net = entry.topology or None
        agrees = None
        if void_net and reference_net:
            agrees = void_net.split()[0].lower() == reference_net.lower()
        rows.append(ComparisonRow(
            "topology", "Topology (single node)", void_net, reference_net or "not assigned",
            "", None, agrees,
            note="CoRE MOF: CrystalNets; the Void: its own single-node deconstruction",
        ))

    if oms:
        step("open metal sites")
        void_oms = None
        try:
            from thevoid_core.services.oms_detection import detect_open_metal_sites
            from thevoid_core.services.site_labels import compute_site_labels

            found = detect_open_metal_sites(structure, compute_site_labels(structure))
            open_metals = sorted({s.metal_symbol for s in found.sites if s.is_open})
            void_oms = ("yes (" + ", ".join(open_metals) + ")") if open_metals else "no"
        except Exception as exc:
            errors["oms"] = f"{type(exc).__name__}: {exc}"
        reference_oms = None
        if entry.has_oms is not None:
            reference_oms = ("yes (" + entry.oms_metals + ")") if entry.has_oms else "no"
        agrees = None
        if void_oms is not None and reference_oms is not None:
            agrees = void_oms.startswith("yes") == reference_oms.startswith("yes")
        rows.append(ComparisonRow("oms", "Open metal sites", void_oms, reference_oms, "",
                                  None, agrees,
                                  note="CoRE MOF: MOFChecker-style rule; the Void: CrystalNN + geometry"))

    notes = [
        "The Void's values here are computed as CoRE MOF 2024 computes its "
        "own: Zeo++'s default (CCDC) van der Waals radii and a "
        f"{COREMOF_PROBE_RADIUS} Å probe. They differ from the Void's standard "
        "analyses, which use its own radii and the N₂ probe.",
    ]
    if match.confidence != "identified":
        notes.append(
            f"The match is '{match.confidence}' ({match.how}): the entry may be "
            "a different determination or a related structure, and differences "
            "can be differences of structure, not of method."
        )

    from thevoid_core.services.reference.coremof import index_info

    source = index_info()
    return ReferenceComparison(
        database="CoRE MOF 2024",
        entry_id=entry.id,
        confidence=match.confidence,
        how=match.how,
        rows=rows,
        parameters={
            "radii": "CCDC (Zeo++ defaults)",
            "probe_radius_A": COREMOF_PROBE_RADIUS,
            "samples_per_atom": samples_per_atom,
            "grid_spacing_A": grid_spacing,
            "index_sha256": source.get("sha256"),
            "index_retrieved": source.get("retrieved"),
        },
        notes=notes,
        reference_values=dict(entry.values),
        errors=errors,
        seconds=time.time() - start,
    )


def comparison_to_dict(comparison: ReferenceComparison) -> dict:
    """JSON-ready form for reports."""
    from thevoid_core.services.reference.coremof import index_info

    source = index_info()
    return {
        "database": comparison.database,
        "entry": comparison.entry_id,
        "match": comparison.confidence,
        "matched_by": comparison.how,
        "parameters": comparison.parameters,
        "rows": [
            {"property": r.key, "label": r.label, "the_void": r.void,
             "reference": r.reference, "unit": r.unit, "difference": r.difference,
             "agrees": r.agrees, "note": r.note}
            for r in comparison.rows
        ],
        "notes": comparison.notes,
        "errors": comparison.errors,
        "reference": {k: comparison.reference_values.get(k) for k in (
            "id", "subset", "refcode", "csd_refcode", "source", "doi", "year")},
        "citation": source.get("citation"),
        "licence": source.get("licence"),
    }
