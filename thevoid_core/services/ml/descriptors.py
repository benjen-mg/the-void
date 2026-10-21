"""Descriptors for the property models (round 61, roadmap 4a).

A model is only as good as the agreement between the numbers it was
trained on and the numbers it is given. So each descriptor set is
defined twice, once from the training data and once from a structure
the Void analyses, and the two are checked to agree
(tests/test_ml.py).

"coremof-v1": the CoRE MOF 2024 models.
    Training: the bundled CoRE MOF 2024 index (Zeo++ geometry, CCDC
    radii, 1.655 Å probe) and the linker chemistry of CoRE's MOFid
    strings (data/coremof2024_linkers.json.gz).
    A structure: the Void's geometry computed the same way (the CoRE
    comparison's convention: radii="ccdc", probe 1.655 Å), which agrees
    with CoRE's to a median 0.03 Å (PLD) and about 1% (ASA) on 76
    structures (rounds 56-58). Its linker chemistry comes from the
    organic fragments of the bond graph (organic_fragments).
"void-v1": models trained on a the-void-analyze batch (the-void-train).
    Both sides are the same summary-CSV columns plus composition
    features parsed from its formula, so training and prediction go
    through the same code.
"""

from __future__ import annotations

import math

#: Heteroatoms whose share of a linker's heavy atoms is a descriptor.
LINKER_ELEMENTS = ("N", "O", "F", "Cl", "Br", "S", "P")

#: The 15 most common single-node nets of CoRE MOF 2024 (counted over the
#: whole index), in order; any other net is "other", and a net that
#: CrystalNets (CoRE) or the Void calls unstable is "unstable".
COREMOF_NETS = ("pcu", "dia", "sql", "pts", "hcb", "rtl", "bpq", "srs", "flu", "kgd",
                "nbo", "ths", "tbo", "sod", "bcu")

COREMOF_V1 = (
    "lcd_A", "pld_A", "asa_m2_g", "nasa_m2_g", "density_g_cm3", "mean_atomic_mass",
    "dimension", "catenation", "has_oms",
    "n_metal_species", "metal_Z_max", "metal_Z_min", "metal_electronegativity_mean",
    "metal_row_max", "metal_group_max", "has_lanthanide", "has_alkali_or_alkaline_earth",
    *(f"linker_{e}_fraction" for e in LINKER_ELEMENTS),
    "linker_types", "linker_heavy_atoms_max",
    *(f"net_{n}" for n in COREMOF_NETS), "net_other", "net_unstable",
)

VOID_V1_BASE = (
    "lcd_A", "pld_A", "void_fraction_percent", "channel_asa_m2_per_g", "density_g_cm3",
    "accessible_dimensionality", "has_oms", "mean_atomic_mass",
    "n_metal_species", "metal_Z_max", "metal_Z_min", "metal_electronegativity_mean",
    "metal_row_max", "metal_group_max", "has_lanthanide", "has_alkali_or_alkaline_earth",
    *(f"{e}_fraction" for e in LINKER_ELEMENTS),
)

NAN = float("nan")


def oms_checkable(metals) -> bool:
    """Whether the Void's open-metal-site detection checks every metal
    here (oms_detection.OMS_SUPPORTED_METALS: 16 metals until round 69,
    which added the lanthanides with Y, Th-Pu, In and Ag, calibrated on
    CoRE's labels). For a structure with any other (alkali metals, Pb,
    Bi, ...) it cannot say "no open metal sites", so has_oms is missing
    on both sides, training and prediction. Round 61 found this:
    reporting 0 there had moved predicted water Henry coefficients of
    Ce, Sm, Eu, Gd and U MOFs by 9-11 decades."""
    from thevoid_core.services.oms_detection import OMS_SUPPORTED_METALS

    return all(m in OMS_SUPPORTED_METALS for m in metals)


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return NAN
    return number if math.isfinite(number) else NAN


def metal_features(metals) -> dict:
    """Features of the set of metals (and CoRE's metalloids)."""
    from pymatgen.core import Element

    elements = []
    for symbol in metals:
        try:
            elements.append(Element(symbol))
        except (ValueError, KeyError):
            continue
    if not elements:
        return {"n_metal_species": 0.0, "metal_Z_max": NAN, "metal_Z_min": NAN,
                "metal_electronegativity_mean": NAN, "metal_row_max": NAN,
                "metal_group_max": NAN, "has_lanthanide": 0.0,
                "has_alkali_or_alkaline_earth": 0.0}
    negativities = [e.X for e in elements if e.X == e.X]
    return {
        "n_metal_species": float(len(elements)),
        "metal_Z_max": float(max(e.Z for e in elements)),
        "metal_Z_min": float(min(e.Z for e in elements)),
        "metal_electronegativity_mean": (sum(negativities) / len(negativities)
                                         if negativities else NAN),
        "metal_row_max": float(max(e.row for e in elements)),
        "metal_group_max": float(max(e.group for e in elements)),
        "has_lanthanide": float(any(e.is_lanthanoid or e.is_actinoid for e in elements)),
        "has_alkali_or_alkaline_earth": float(any(e.is_alkali or e.is_alkaline
                                                  for e in elements)),
    }


def linker_features(fragments) -> dict:
    """fragments: element-count dicts, one per distinct linker. Shares
    of heavy atoms, averaged over the distinct linkers (each linker
    type once, as MOFid lists them)."""
    names = [f"linker_{e}_fraction" for e in LINKER_ELEMENTS]
    rows = []
    for counts in fragments:
        heavy = sum(v for k, v in counts.items() if k not in ("H", "D", "T"))
        if heavy and counts.get("C"):
            rows.append(([counts.get(e, 0) / heavy for e in LINKER_ELEMENTS], heavy))
    if not rows:
        return {**{n: NAN for n in names}, "linker_types": 0.0, "linker_heavy_atoms_max": NAN}
    out = {n: sum(r[0][i] for r in rows) / len(rows) for i, n in enumerate(names)}
    out["linker_types"] = float(len(rows))
    out["linker_heavy_atoms_max"] = float(max(r[1] for r in rows))
    return out


def net_code(text: str | None) -> str:
    """The net of a topology cell: an RCSR code; "other" for a net
    outside the reference library; "unstable"; or "" when no net was
    assigned. Reads CoRE's codes and the Void's summaries alike
    ("pcu 6-c, 2-fold interpenetrated", "(3,4)-c net, not in the
    reference library", "not identified: unstable net")."""
    text = (text or "").strip().lower()
    if not text:
        return ""
    if "unstable" in text:
        return "unstable"
    if "split rod sbus" in text and "not in the reference library" in text:
        # Round 70: CoRE leaves a rod framework's net unnamed unless RCSR
        # has it - 272 of the 281 CoRE SI rod frameworks whose split net
        # the Void cannot name have no net in CoRE - so this is "no net",
        # as it was when the Void did not split rods at all.
        return ""
    if "not in the reference library" in text or text == "unlisted":
        return "other"
    if text.startswith(("not ", "no ")):
        return ""
    return text.split()[0]


def net_features(net: str | None, nets=COREMOF_NETS) -> dict:
    code = net_code(net)
    out = {f"net_{n}": 1.0 if code == n else 0.0 for n in nets}
    out["net_other"] = 1.0 if code and code not in nets and code != "unstable" else 0.0
    out["net_unstable"] = 1.0 if code == "unstable" else 0.0
    return out


# ----------------------------------------------------------------------
# coremof-v1 from the training data
# ----------------------------------------------------------------------


def coremof_row(entry_values: dict, linkers: dict | None) -> dict:
    """coremof-v1 descriptors of one CoRE MOF 2024 index entry.
    linkers: {"fragments": [element counts...]} from the linker table,
    or None when CoRE gives no MOFid for the entry."""
    v = entry_values
    oms = v.get("has_oms")
    row = {
        "lcd_A": _number(v.get("lcd_A")), "pld_A": _number(v.get("pld_A")),
        "asa_m2_g": _number(v.get("asa_m2_g")), "nasa_m2_g": _number(v.get("nasa_m2_g")),
        "density_g_cm3": _number(v.get("density_g_cm3")),
        "mean_atomic_mass": _number(v.get("mean_atomic_mass")),
        "dimension": _number(v.get("dimension")), "catenation": _number(v.get("catenation")),
        "has_oms": NAN if oms is None else float(bool(oms)),
    }
    metals = [m for m in (v.get("metals") or "").split(",") if m]
    if not oms_checkable(metals):
        row["has_oms"] = NAN
    row.update(metal_features(metals))
    row.update(linker_features(linkers["fragments"]) if linkers else
               linker_features([]) | {"linker_types": NAN})
    row.update(net_features(v.get("topology")))
    return row


# ----------------------------------------------------------------------
# coremof-v1 from a structure
# ----------------------------------------------------------------------


def organic_fragments(structure, bonds=None) -> list:
    """Element counts of each distinct linker: a carbon-containing
    component of the bond graph with its metals removed (as MOFid
    splits a MOF), that bonds to at least two metal atoms - a ligand on
    a single metal (a coordinated solvent, a terminal formate) links
    nothing. On 124 CoRE SI structures this gives exactly the linker
    chemistry of CoRE's MOFid strings for 118."""
    from thevoid_core.data.metal_classification import is_metal
    from thevoid_core.services.pore_analysis import major_occupancy_structure

    ordered = major_occupancy_structure(structure)
    if bonds is None or ordered is not structure:
        from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

        bonds = CovalentBondDetector().detect(ordered)
    symbols = [site.specie.symbol for site in ordered.sites]
    metal = [is_metal(s) for s in symbols]
    parent = list(range(len(symbols)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for bond in bonds:
        if not (metal[bond.atom1] or metal[bond.atom2]):
            parent[find(bond.atom1)] = find(bond.atom2)
    counts, contacts = {}, {}
    for i, symbol in enumerate(symbols):
        if not metal[i]:
            counts.setdefault(find(i), {})
            counts[find(i)][symbol] = counts[find(i)].get(symbol, 0) + 1
    for bond in bonds:
        a, b = bond.atom1, bond.atom2
        if metal[a] == metal[b]:
            continue
        organic, other, image = (b, a, tuple(bond.image)) if metal[a] else \
            (a, b, tuple(-v for v in bond.image))
        contacts.setdefault(find(organic), set()).add((organic, other, image))
    distinct = {}
    for root, c in counts.items():
        if c.get("C") and len(contacts.get(root, ())) >= 2:
            distinct[tuple(sorted(c.items()))] = c
    return list(distinct.values())


def _supported():
    from thevoid_core.services.oms_detection import OMS_SUPPORTED_METALS

    return set(OMS_SUPPORTED_METALS)


def coremof_descriptors(structure, *, samples_per_atom: int = 1000, pores=None, surface=None,
                        oms=None, topology=None) -> tuple[dict, dict]:
    """(coremof-v1 descriptors, notes) of a structure, computed under
    CoRE MOF 2024's convention (CCDC radii, 1.655 Å probe). pores,
    surface, oms and topology: results already computed that way
    (internal result objects), so a caller with a cache (thevoid's
    Framework) does not repeat them."""
    from thevoid_core.data.element_radii import CCDCRadiusProvider
    from thevoid_core.data.metal_classification import is_metal
    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector
    from thevoid_core.services.mof_analysis import bonded_components
    from thevoid_core.services.pore_analysis import analyze_pores, major_occupancy_structure
    from thevoid_core.services.reference.coremof import StructureFacts
    from thevoid_core.services.surface_area import analyze_surface_area

    notes = {}
    radii = CCDCRadiusProvider()
    if pores is None:
        pores = analyze_pores(structure, probe_radius=1.655, radius_provider=radii)
    if surface is None:
        surface = analyze_surface_area(structure, probe_radius=1.655, probe_name="CoRE (1.655 Å)",
                                       radius_provider=radii, n_samples_per_atom=samples_per_atom)
    facts = StructureFacts.of(structure)
    mass = surface.cell_mass_grams
    nasa = surface.pocket_accessible_area * 1.0e-20 / mass if mass else NAN
    row = {
        "lcd_A": _number(pores.lcd), "pld_A": _number(pores.pld),
        "asa_m2_g": _number(surface.channel_asa_per_mass), "nasa_m2_g": _number(nasa),
        "density_g_cm3": _number(facts.density_g_cm3),
        "mean_atomic_mass": _number(facts.mean_atomic_mass),
    }
    ordered = major_occupancy_structure(structure)
    bonds = CovalentBondDetector().detect(ordered)
    components = [c for c in bonded_components(ordered, bonds, is_metal) if c.contains_metal]
    top = max((c.dimensionality for c in components), default=0)
    row["dimension"] = float(top)
    row["catenation"] = float(sum(1 for c in components if c.dimensionality == top)) if top else NAN
    if oms is None:
        try:
            from thevoid_core.services.oms_detection import detect_open_metal_sites
            from thevoid_core.services.site_labels import compute_site_labels

            oms = detect_open_metal_sites(structure, compute_site_labels(structure))
        except Exception as exc:
            notes["has_oms"] = f"open metal sites failed: {type(exc).__name__}: {exc}"
    row["has_oms"] = float(oms.oms_count > 0) if oms is not None else NAN
    if not oms_checkable(facts.metals):
        row["has_oms"] = NAN
        notes.setdefault("has_oms", "open metal sites are not checked for "
                         + ", ".join(sorted(set(facts.metals) - _supported())) + ": left missing")
    row.update(metal_features(sorted(facts.metals)))
    row.update(linker_features(organic_fragments(ordered, bonds)))
    if topology is None:
        try:
            from thevoid_core.services.topology import analyze_topology

            topology = analyze_topology(structure)
        except Exception as exc:
            notes["net"] = f"topology failed: {type(exc).__name__}: {exc}"
    net = None
    if topology is not None:
        net = topology.rcsr or {"not_in_library": "unlisted",
                                "unstable": "unstable"}.get(topology.status)
        if net == "unlisted" and getattr(topology, "rod_sbus", None):
            net = None  # see net_code: CoRE leaves these unnamed
        if net is None:
            # No net (a periodic organic part, a metal-free framework,
            # a failed analysis): every net feature is 0, as for CoRE's
            # entries without a net. Rod SBUs are split at their metals
            # since round 70, as CoRE's CrystalNets does (MIL-53: bpq).
            notes["net"] = f"no net assigned ({topology.summary}): every net feature is 0"
    row.update(net_features(net))
    return row, notes


# ----------------------------------------------------------------------
# void-v1: from a the-void-analyze summary row
# ----------------------------------------------------------------------


def composition_features(formula: str) -> dict:
    from pymatgen.core import Composition

    from thevoid_core.data.metal_classification import is_metal

    try:
        composition = Composition(formula)
    except Exception:
        return {**metal_features([]), "mean_atomic_mass": NAN,
                **{f"{e}_fraction": NAN for e in LINKER_ELEMENTS}}
    amounts = composition.get_el_amt_dict()
    atoms = sum(amounts.values())
    heavy = sum(v for k, v in amounts.items() if k not in ("H", "D", "T"))
    out = {"mean_atomic_mass": composition.weight / atoms if atoms else NAN}
    out.update(metal_features(sorted(k for k in amounts if is_metal(k))))
    out.update({f"{e}_fraction": amounts.get(e, 0.0) / heavy if heavy else NAN
                for e in LINKER_ELEMENTS})
    return out


def void_row(summary_row: dict, nets) -> dict:
    """void-v1 descriptors of a the-void-analyze summary row."""
    oms = summary_row.get("open_metal_sites")
    row = {
        "lcd_A": _number(summary_row.get("lcd_A")), "pld_A": _number(summary_row.get("pld_A")),
        "void_fraction_percent": _number(summary_row.get("void_fraction_percent")),
        "channel_asa_m2_per_g": _number(summary_row.get("channel_asa_m2_per_g")),
        "density_g_cm3": _number(summary_row.get("density_g_cm3")),
        "accessible_dimensionality": _number(summary_row.get("accessible_dimensionality")),
        "has_oms": NAN if oms in (None, "") else float(_number(oms) > 0),
    }
    row.update(composition_features(summary_row.get("formula") or ""))
    row.update(net_features(summary_row.get("topology"), nets))
    try:
        from pymatgen.core import Composition

        from thevoid_core.data.metal_classification import is_metal

        metals = [e.symbol for e in Composition(summary_row.get("formula") or "") if is_metal(e.symbol)]
    except Exception:
        metals = []
    if not oms_checkable(metals):
        row["has_oms"] = NAN  # as coremof-v1: 0 would claim a check that was not made
    return row


def void_feature_names(nets) -> tuple:
    return VOID_V1_BASE + tuple(f"net_{n}" for n in nets) + ("net_other", "net_unstable")
