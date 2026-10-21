"""Analysis reports: every result as plain, serialisable data.

One module serves the command-line batch tool (thevoid_core.cli) and
File > Export Analysis Report... in the application, so a number means
the same thing wherever it comes from. Each report records how it was
made: the software and library versions, a SHA-256 of the input file,
and every parameter and method choice. That record is what a paper's
supporting information, or anyone re-running the analysis, needs.

Nothing here imports Qt.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from thevoid_core.version import __version__

REPORT_FORMAT_VERSION = 1

AVOGADRO_NUMBER = 6.02214076e23


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass
class AnalysisConfig:
    """What to run, and with which parameters. The defaults match the
    application's own panel defaults."""

    pore: bool = True
    pore_probe_radius: float = 1.2
    surface_area: bool = True
    surface_probe: str = "N2"
    surface_probe_radius: float | None = None  # overrides surface_probe
    surface_samples_per_atom: int = 1000
    oms: bool = True
    psd: bool = False
    psd_bin_width: float = 0.2
    guests: list[str] = field(default_factory=list)  # names or "name=diameter"
    xrd: bool = False
    xrd_wavelength: str = "CuKa"
    xrd_two_theta_range: tuple[float, float] = (5.0, 50.0)
    xrd_top_peaks: int = 20
    grid_spacing: float = 0.3
    #: Round 53: the underlying net and its RCSR code.
    topology: bool = True
    #: A Systre .arc file to match against instead of the bundled set.
    rcsr_archive: str | None = None
    #: Round 56: find the structure in CoRE MOF 2024 and compare, with
    #: the Void's values recomputed under CoRE's convention.
    compare_coremof: bool = False
    #: Round 61: property models to apply - names of bundled models
    #: ("water_kh", "thermal_stability", "solvent_stability",
    #: "water_stability", or "all") or folders the-void-train wrote.
    predict: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = dict(self.__dict__)
        data["xrd_two_theta_range"] = list(self.xrd_two_theta_range)
        return data


# ----------------------------------------------------------------------
# Structure summary
# ----------------------------------------------------------------------


def summarize_structure(structure, *, space_group: str | None = None) -> dict:
    lattice = structure.lattice
    composition = structure.composition
    volume = float(lattice.volume)
    mass_per_cell = float(composition.weight)  # g/mol per cell contents

    if space_group is None:
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            space_group = SpacegroupAnalyzer(
                structure, symprec=0.01
            ).get_space_group_symbol()
        except Exception:
            space_group = "unknown"

    partial = sum(
        1 for site in structure.sites if site.species.num_atoms < 0.99
    )

    return {
        "formula": composition.reduced_formula,
        "cell_formula": composition.formula,
        "space_group": space_group,
        "sites_in_cell": len(structure),
        "atoms_in_cell": float(composition.num_atoms),
        "partial_occupancy_sites": partial,
        "cell": {
            "a": lattice.a, "b": lattice.b, "c": lattice.c,
            "alpha": lattice.alpha, "beta": lattice.beta,
            "gamma": lattice.gamma,
        },
        "cell_volume_A3": volume,
        "cell_mass_g_per_mol": mass_per_cell,
        "density_g_cm3": crystal_density(structure),
    }


def crystal_density(structure) -> float:
    """Crystallographic density in g/cm^3 (cell mass / cell volume),
    counting partially occupied sites by their occupancy."""
    volume = float(structure.lattice.volume)

    if volume <= 0:
        return 0.0

    grams = float(structure.composition.weight) / AVOGADRO_NUMBER
    return grams / (volume * 1e-24)


# ----------------------------------------------------------------------
# Result -> dict converters (also used by the GUI export)
# ----------------------------------------------------------------------


def _clean(value):
    """JSON-safe: numpy scalars to Python, inf/nan to None."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if hasattr(value, "value") and hasattr(value, "name"):  # Enum
        return value.value
    return value


def pore_to_dict(result) -> dict:
    return _clean(
        {
            "probe_radius_A": result.probe_radius,
            "lcd_A": result.lcd,
            "pld_A": result.pld,
            "void_fraction_percent": result.void_fraction_percent,
            "accessible_volume_A3": result.accessible_volume,
            "non_accessible_volume_A3": result.non_accessible_volume,
            "accessible_dimensionality": result.accessible_dimensionality,
            "grid_shape": list(result.grid_shape),
            "lcd_pld_method": getattr(result, "geometry_method", "grid"),
            "accessibility_method": getattr(result, "accessibility_method", "grid"),
        }
    )


def surface_area_to_dict(result) -> dict:
    return _clean(
        {
            "probe": result.probe_name,
            "probe_radius_A": result.probe_radius,
            "samples_per_atom": result.n_samples_per_atom,
            "accessible_surface_area_A2": result.accessible_surface_area,
            "channel_surface_area_A2": result.internal_accessible_area,
            "pocket_surface_area_A2": getattr(
                result, "pocket_accessible_area", None
            ),
            "geometric_surface_area_A2": result.geometric_surface_area,
            "asa_m2_per_g_including_pockets": result.asa_per_mass,
            "channel_asa_m2_per_g": getattr(
                result, "channel_asa_per_mass", None
            ),
            "asa_m2_per_cm3": result.asa_per_volume,
        }
    )


def guest_to_dict(result) -> dict:
    return _clean(
        {
            "guest": result.guest_name,
            "guest_diameter_A": result.guest_diameter,
            "status": result.status,
            "pld_A": result.minimum_window_diameter,
            "lcd_A": result.largest_cavity_diameter,
            "largest_accessible_cavity_A": result.largest_accessible_cavity,
            "bottleneck_count": len(result.bottlenecks),
        }
    )


def oms_to_dict(result) -> dict:
    from thevoid_core.services.oms_detection import calibration_of

    calibrated = {}
    for site in result.sites:
        found = calibration_of(site.metal_symbol)
        if found is not None:
            calibrated[site.metal_symbol] = found
    return _clean(
        {
            "open_metal_sites": result.oms_count,
            "metal_centres_analyzed": len(result.sites),
            "not_checked_metals": list(
                getattr(result, "unsupported_metals", ())
            ),
            # Round 69: metals here whose saturated coordination number
            # was calibrated on CoRE MOF 2024 (data/oms_calibration.json).
            "calibrated_metals": calibrated,
            "sites": [
                {
                    "label": site.label,
                    "metal": site.metal_symbol,
                    "coordination_number": site.coordination_number,
                    "classification": site.classification,
                    "asymmetry": site.asymmetry_score,
                    "widest_empty_cone_deg": getattr(site, "angular_gap", None),
                    "saturated_coordination": getattr(site, "saturated_at", None),
                    "missing_directions": [
                        list(mp.direction) for mp in site.missing_positions
                    ],
                }
                for site in result.sites
            ],
        }
    )


def psd_to_dict(result) -> dict:
    return _clean(
        {
            "bin_width_A": result.bin_width,
            "pore_volume_A3": result.pore_volume,
            "void_fraction": result.void_fraction,
            "dominant_diameter_A": result.dominant_diameter,
            "grid_shape": list(result.grid_shape),
            "bins": [
                {"diameter_A": c, "distribution_per_A": p, "cumulative_volume_A3": v}
                for c, p, v in result.rows()
            ],
        }
    )


def xrd_to_dict(result, top: int = 20) -> dict:
    order = np.argsort(result.intensities)[::-1][:top]
    peaks = []

    for index in sorted(order, key=lambda i: result.two_theta[i]):
        hkls = result.hkls[index] if index < len(result.hkls) else []
        peaks.append(
            {
                "two_theta_deg": result.two_theta[index],
                "d_A": result.d_spacings[index],
                "relative_intensity": result.intensities[index],
                "hkl": [list(h.get("hkl", ())) for h in hkls],
            }
        )

    return _clean(
        {
            "wavelength_A": result.wavelength,
            "two_theta_range_deg": list(result.two_theta_range or ()),
            "reflections": len(result.two_theta),
            "strongest_peaks": peaks,
        }
    )


# ----------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------


def file_sha256(path) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def provenance() -> dict:
    versions = {"the_void": __version__, "python": platform.python_version()}

    for module in ("numpy", "scipy", "pymatgen"):
        try:
            if module == "pymatgen":
                from importlib.metadata import version

                versions[module] = version("pymatgen")
            else:
                versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "unknown"

    return {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "software": versions,
        "platform": platform.platform(),
        "report_format_version": REPORT_FORMAT_VERSION,
    }


METHOD_NOTES = {
    "radii": "van der Waals: Bondi (1964), Batsanov (2001) where Bondi is "
    "missing; covalent (bonding): Cordero et al. (2008) + 0.4 A",
    "disorder": "sites below 0.5 occupancy removed before every void-space "
    "analysis",
    "lcd_pld": "LCD: exact maximum of the distance to atom surfaces; PLD: "
    "radical Voronoi network bottleneck (Zeo++ method), near-critical "
    "windows and vertices refined along actual paths whose clearance is "
    "computed exactly (round 56); where radii differ, also a split-sphere "
    "network (larger atoms split into spheres of the smallest radius, "
    "edges valued exactly; round 57, always since round 58); never below "
    "what the grid's path proves",
    "volumes": "cell-centred grid, 0.3 A target spacing, 6-connected "
    "periodic percolation, with the Voronoi network joined in so windows "
    "narrower than the grid can resolve still connect (round 55), and the "
    "same split-sphere network as the PLD used when nothing percolates "
    "through the radical one (rounds 57-58)",
    "surface_area": "rotated Fibonacci sampling of probe-inflated spheres; "
    "channel/pocket split as Zeo++ ASA/NASA",
    "psd": "geometric PSD (Gelb & Gubbins 1999) by grid morphological "
    "opening; resolution limited by the grid, so the largest bin can sit "
    "up to one bin below the exact LCD",
    "oms": "CrystalNN coordination; open if below the metal's reference "
    "coordination number and either asymmetric (>0.15) or with an empty "
    "cone wider than 80 degrees; the reference numbers of the lanthanides, "
    "Y, Th-Pu, In and Ag are calibrated on CoRE MOF 2024's labels "
    "(data/oms_calibration.json, round 69)",
    "xrd": "pymatgen XRDCalculator (no Debye-Waller factors)",
    "topology": "single-node deconstruction (SBUs = metals, mu-bridging "
    "atoms and carboxylates that chelate one metal or bridge metals within "
    "4.5 A; >=3-connected linkers are nodes, "
    "2-connected pieces edges; zeolites as T-atom nets), then an exact "
    "Systre-style isomorphism test (barycentric placement, minimal cell) "
    "against RCSR nets (O'Keeffe et al. 2008)",
    "reference": "CoRE MOF 2024 comparison (only when requested): entry "
    "found by name (id, refcode, CSD refcode) and checked by atom count, "
    "density, metals and mean atomic mass, or by those alone; the Void's "
    "values recomputed with Zeo++'s default (CCDC) radii and CoRE's "
    "1.655 A probe; bundled CoRE MOF 2024 index, Zhao et al., Matter 8, "
    "102140 (2025), CC-BY-4.0",
    "raspa_export": "RASPA2 GCMC input set (only when exported): UFF "
    "Lennard-Jones framework (Rappe et al. 1992) + TraPPE adsorbate, "
    "Lorentz-Berthelot, shifted LJ, no tail corrections; framework "
    "charges only from the CIF's _atom_site_charge column (otherwise "
    "uncharged unless RASPA charge equilibration was chosen); "
    "non-periodic bonded molecules removed; pockets unreachable by the "
    "adsorbate's probe blocked with clearance spheres (Zeo++ convention)",
}


def new_report(structure, *, source: str | None = None, space_group=None) -> dict:
    return {
        "provenance": provenance(),
        "input": {
            "file": str(source) if source else None,
            "sha256": file_sha256(source) if source else None,
        },
        "structure": _clean(summarize_structure(structure, space_group=space_group)),
        "methods": dict(METHOD_NOTES),
        "results": {},
        "errors": {},
    }


# ----------------------------------------------------------------------
# Running everything
# ----------------------------------------------------------------------


def _guest_diameter(entry: str):
    from thevoid_core.data.probe_radii import PROBE_RADII

    if "=" in entry:
        name, value = entry.split("=", 1)
        return name.strip(), float(value)

    name = entry.strip()

    if name not in PROBE_RADII:
        raise ValueError(
            f"Unknown guest {name!r}; use one of {sorted(PROBE_RADII)} "
            "or name=diameter"
        )

    return name, 2.0 * PROBE_RADII[name]


def run_analysis(structure, config: AnalysisConfig | None = None, *,
                 source=None, radius_provider=None, progress=None) -> dict:
    """Run every analysis `config` asks for and return the report.

    A failing analysis is recorded under "errors" and the rest still
    run, so one awkward structure doesn't lose a whole batch's worth of
    results.
    """
    from thevoid_core.data.element_radii import ElementRadiusProvider

    config = config or AnalysisConfig()
    radius_provider = radius_provider or ElementRadiusProvider()
    report = new_report(structure, source=source)
    report["config"] = _clean(config.as_dict())

    def step(name, function):
        if progress is not None:
            progress(name)
        try:
            report["results"][name] = function()
        except Exception as exc:  # recorded, not raised
            report["errors"][name] = f"{type(exc).__name__}: {exc}"

    if config.topology:
        from thevoid_core.services.topology import (
            analyze_topology,
            topology_to_dict,
        )
        from thevoid_core.services.topology.library import default_library

        def topology():
            result = analyze_topology(
                structure, library=default_library(config.rcsr_archive)
            )
            report["structure"]["topology"] = result.summary
            return _clean(topology_to_dict(result))

        step("topology", topology)

    if config.pore:
        from thevoid_core.services.pore_analysis import analyze_pores

        step("pore", lambda: pore_to_dict(analyze_pores(
            structure, probe_radius=config.pore_probe_radius,
            radius_provider=radius_provider, grid_spacing=config.grid_spacing,
        )))

    if config.surface_area:
        from thevoid_core.data.probe_radii import PROBE_RADII
        from thevoid_core.services.surface_area import analyze_surface_area

        def surface():
            radius = config.surface_probe_radius
            name = config.surface_probe
            if radius is None:
                radius = PROBE_RADII[name]
            else:
                name = "Custom"
            return surface_area_to_dict(analyze_surface_area(
                structure, probe_radius=radius, probe_name=name,
                radius_provider=radius_provider,
                n_samples_per_atom=config.surface_samples_per_atom,
            ))

        step("surface_area", surface)

    for entry in config.guests:
        from thevoid_core.services.guest_accessibility import (
            analyze_guest_accessibility,
        )

        def guest(entry=entry):
            name, diameter = _guest_diameter(entry)
            return guest_to_dict(analyze_guest_accessibility(
                structure, guest_diameter=diameter, guest_name=name,
                radius_provider=radius_provider,
                grid_spacing=config.grid_spacing,
            ))

        label = entry.split("=")[0].strip()
        if progress is not None:
            progress(f"guest {label}")
        try:
            report["results"].setdefault("guests", {})[label] = guest()
        except Exception as exc:
            report["errors"][f"guest {label}"] = f"{type(exc).__name__}: {exc}"

    if config.oms:
        from thevoid_core.services.oms_detection import detect_open_metal_sites
        from thevoid_core.services.site_labels import compute_site_labels

        step("oms", lambda: oms_to_dict(detect_open_metal_sites(
            structure, compute_site_labels(structure)
        )))

    if config.psd:
        from thevoid_core.services.pore_size_distribution import (
            compute_pore_size_distribution,
        )

        step("psd", lambda: psd_to_dict(compute_pore_size_distribution(
            structure, radius_provider=radius_provider,
            bin_width=config.psd_bin_width, grid_spacing=config.grid_spacing,
        )))

    if config.xrd:
        from thevoid_core.services.xrd.powder_xrd_simulator import (
            PowderXRDSimulator,
        )

        step("xrd", lambda: xrd_to_dict(PowderXRDSimulator().simulate(
            structure, two_theta_range=tuple(config.xrd_two_theta_range),
            wavelength=config.xrd_wavelength,
        ), top=config.xrd_top_peaks))

    if config.compare_coremof:
        step("reference", lambda: reference_to_dict(
            structure, source, samples_per_atom=config.surface_samples_per_atom,
            grid_spacing=config.grid_spacing,
        ))

    if config.predict:
        from thevoid_core.services.ml.predict import predict_structure, predictions_to_dict

        models = None if list(config.predict) == ["all"] else list(config.predict)

        def predictions():
            result = predict_structure(structure, models, summary_row=summary_row(report),
                                       samples_per_atom=config.surface_samples_per_atom)
            return predictions_to_dict(result)

        step("predictions", predictions)

    return report


def reference_to_dict(structure, source=None, **options) -> dict:
    """The CoRE MOF 2024 comparison for a report: the best match's
    comparison, the other candidates, or why there is none."""
    from thevoid_core.services.reference.compare import (
        compare_with_coremof,
        comparison_to_dict,
    )
    from thevoid_core.services.reference.coremof import find_matches

    matches = find_matches(structure, path=source)
    usable = [m for m in matches if m.confidence != "name only"]
    others = [
        {"entry": m.entry.id, "match": m.confidence, "matched_by": m.how}
        for m in matches[1:] if usable
    ] or [
        {"entry": m.entry.id, "match": m.confidence, "matched_by": m.how}
        for m in matches
    ]
    if not usable:
        return _clean({
            "database": "CoRE MOF 2024",
            "match": None,
            "message": (
                "No CoRE MOF 2024 entry matches this structure."
                if not matches else
                "Entries share this structure's name, but none matches its atom "
                "count, density or composition; no comparison made."
            ),
            "candidates": others,
        })
    record = comparison_to_dict(compare_with_coremof(structure, usable[0], **options))
    record["candidates"] = others
    return _clean(record)


# ----------------------------------------------------------------------
# Output formats
# ----------------------------------------------------------------------


def to_json(report: dict) -> str:
    return json.dumps(_clean(report), indent=2, ensure_ascii=False)


def _fmt(value, digits=3):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown(report: dict) -> str:
    """A human-readable version, suitable for a lab notebook or a
    paper's supporting information."""
    s = report["structure"]
    r = report.get("results", {})
    p = report["provenance"]
    lines = [
        f"# Analysis report: {s['formula']}",
        "",
        f"Generated {p['generated']} by the Void {p['software']['the_void']} "
        f"(pymatgen {p['software'].get('pymatgen')}, numpy "
        f"{p['software'].get('numpy')}, scipy {p['software'].get('scipy')}).",
    ]

    if report["input"].get("file"):
        lines += ["", f"Input: `{Path(report['input']['file']).name}`"
                  f" (SHA-256 `{report['input'].get('sha256')}`)"]

    cell = s["cell"]
    lines += [
        "", "## Structure", "",
        "| Property | Value |", "|---|---|",
        f"| Formula | {s['formula']} ({s['cell_formula']} per cell) |",
        f"| Space group | {s['space_group']} |",
    ]
    if s.get("topology"):
        lines.append(f"| Topology (RCSR) | {s['topology']} |")
    lines += [
        f"| Cell (Å, °) | a {cell['a']:.4f}, b {cell['b']:.4f}, c {cell['c']:.4f}; "
        f"α {cell['alpha']:.2f}, β {cell['beta']:.2f}, γ {cell['gamma']:.2f} |",
        f"| Cell volume | {s['cell_volume_A3']:.2f} Å³ |",
        f"| Density | {s['density_g_cm3']:.4f} g/cm³ |",
        f"| Sites (partial occupancy) | {s['sites_in_cell']} ({s['partial_occupancy_sites']}) |",
    ]

    if "topology" in r:
        q = r["topology"]
        lines += ["", "## Topology", "",
                  f"{q['summary']} ({q['representation']} representation; "
                  f"matched against the {q['library']})."]
        if q["message"]:
            lines += ["", q["message"]]
        if q["nodes"]:
            lines += ["", "| Node | Kind | Connectivity | Per cell |", "|---|---|---|---|"]
            for node in q["nodes"]:
                lines.append(f"| {node['formula']} | {node['kind']} | "
                             f"{node['connectivity']}-c | {node['count']} |")
        if q["edges_from"]:
            lines += ["", "Edges of the net come from: "
                      + ", ".join(sorted(q["edges_from"])) + "."]
        for note in q["notes"]:
            lines += ["", f"- {note}"]

    if "pore" in r:
        q = r["pore"]
        lines += [
            "", "## Pores", "",
            "| Property | Value |", "|---|---|",
            f"| Largest cavity diameter | {_fmt(q['lcd_A'])} Å |",
            f"| Pore limiting diameter | {_fmt(q['pld_A'])} Å |",
            f"| Void fraction (probe {q['probe_radius_A']:.2f} Å) | {_fmt(q['void_fraction_percent'], 2)} % |",
            f"| Accessible / non-accessible volume | {_fmt(q['accessible_volume_A3'], 1)} / {_fmt(q['non_accessible_volume_A3'], 1)} Å³ |",
            f"| Accessible network | {q['accessible_dimensionality']}D |",
            f"| LCD/PLD method | {q['lcd_pld_method']} |",
        ]

    if "surface_area" in r:
        q = r["surface_area"]
        lines += [
            "", f"## Surface area ({q['probe']}, probe {q['probe_radius_A']:.3f} Å)", "",
            "| Property | Value |", "|---|---|",
            f"| Channel ASA | {_fmt(q['channel_surface_area_A2'], 2)} Å² ({_fmt(q['channel_asa_m2_per_g'], 1)} m²/g) |",
            f"| Pocket surface | {_fmt(q['pocket_surface_area_A2'], 2)} Å² |",
            f"| Total ASA (incl. pockets) | {_fmt(q['accessible_surface_area_A2'], 2)} Å² ({_fmt(q['asa_m2_per_g_including_pockets'], 1)} m²/g) |",
            f"| Geometric surface | {_fmt(q['geometric_surface_area_A2'], 2)} Å² |",
        ]

    if r.get("guests"):
        lines += ["", "## Guest accessibility", "",
                  "| Guest | Diameter (Å) | Verdict |", "|---|---|---|"]
        for name, q in r["guests"].items():
            lines.append(f"| {name} | {_fmt(q['guest_diameter_A'], 2)} | {q['status']} |")

    if "oms" in r:
        q = r["oms"]
        lines += ["", "## Open metal sites", "",
                  f"{q['open_metal_sites']} open of {q['metal_centres_analyzed']} metal centres analyzed."]
        if q["not_checked_metals"]:
            lines.append(f"Not checked: {', '.join(q['not_checked_metals'])}.")
        for metal, c in (q.get("calibrated_metals") or {}).items():
            lines.append(
                f"{metal} counts as saturated at coordination {c['saturated_coordination']}, "
                f"a number calibrated on CoRE MOF 2024's labels ({c['group']}; agreement "
                f"{c['held_out_agreement']:.0%} on {c['held_out_items']} held-out cases).")
        if q["sites"]:
            lines += ["", "| Site | Metal | CN | Status | Asymmetry | Widest cone (°) |",
                      "|---|---|---|---|---|---|"]
            for site in q["sites"]:
                lines.append(
                    f"| {site['label']} | {site['metal']} | {site['coordination_number']} | "
                    f"{site['classification']} | {_fmt(site['asymmetry'])} | "
                    f"{_fmt(site['widest_empty_cone_deg'], 1)} |"
                )

    if "psd" in r:
        q = r["psd"]
        lines += ["", "## Pore size distribution", "",
                  f"Dominant pore diameter {_fmt(q['dominant_diameter_A'], 2)} Å; "
                  f"pore volume {_fmt(q['pore_volume_A3'], 1)} Å³ per cell "
                  f"(bins of {q['bin_width_A']} Å; full table in the JSON report)."]

    if "xrd" in r:
        q = r["xrd"]
        lines += ["", f"## Powder XRD (λ = {q['wavelength_A']:.5f} Å)", "",
                  "| 2θ (°) | d (Å) | I (rel.) | hkl |", "|---|---|---|---|"]
        for peak in q["strongest_peaks"]:
            hkl = "; ".join("".join(str(i) for i in h) for h in peak["hkl"])
            lines.append(
                f"| {peak['two_theta_deg']:.3f} | {peak['d_A']:.4f} | "
                f"{peak['relative_intensity']:.1f} | {hkl} |"
            )

    if "predictions" in r:
        lines += ["", "## Predicted properties (models, not measurements)", "",
                  "| Model | Prediction | 90% interval | In the training domain | Note |",
                  "|---|---|---|---|---|"]
        for name, q in r["predictions"].get("models", {}).items():
            unit = q.get("unit") or ""
            value, low, high = (format_prediction(q.get(k), unit) for k in ("value", "low", "high"))
            lines.append(
                f"| {q['label']} | {value}{' ' + unit if unit else ''} | {low} to {high} | "
                f"{'yes' if q.get('in_domain') else 'NO: an extrapolation'} | {q.get('note', '')} |")
        for key, message in (r["predictions"].get("notes") or {}).items():
            lines.append(f"\nDescriptor note ({key}): {message}.")

    if "raspa_export" in r:
        q = r["raspa_export"]
        pressures = ", ".join(f"{p / 1e5:g}" for p in q["pressures_Pa"])
        blocking = q.get("blocking") or {}
        block_text = {
            "blocked": f"{blocking.get('spheres')} sphere(s)",
            "no pockets": "no pockets to block",
            "not percolating": "not blocked (no channel open to the probe)",
        }.get(blocking.get("status"), "not requested")
        lines += [
            "", "## RASPA2 export", "",
            "A GCMC input set was written; it is a starting point, not a "
            "simulation result. See its README.txt.", "",
            "| Property | Value |", "|---|---|",
            f"| Folder | `{q['directory']}` |",
            f"| Adsorbate | {q['adsorbate']} (TraPPE) |",
            f"| Conditions | {q['temperature_K']} K; {pressures} bar |",
            f"| Cycles | {q['initialization_cycles']} initialisation + {q['cycles']} production |",
            f"| Force field | UFF framework, Lorentz-Berthelot, cutoff {q['cutoff_A']} Å |",
            f"| Framework charges | {q['charges']} |",
            f"| Unit cells | {' × '.join(str(n) for n in q['unit_cells'])} |",
            f"| Framework atoms | {q['framework_atoms']} |",
            f"| Removed molecules | {', '.join(q['removed_molecules']) or 'none'} |",
            f"| Pocket blocking | {block_text} |",
        ]
        for warning in q.get("warnings", []):
            lines.append(f"\n> **Warning:** {warning}")

    if "reference" in r:
        q = r["reference"]
        lines += ["", f"## Reference: {q['database']}", ""]
        if not q.get("match"):
            lines.append(q["message"])
        else:
            ref = q["reference"]
            where = f"CSD {ref['csd_refcode']}" if ref.get("csd_refcode") else "paper SI"
            lines += [
                f"Entry `{q['entry']}` ({where}; doi:{ref.get('doi')}), match: "
                f"**{q['match']}** ({q['matched_by']}).", "",
                "| Property | The Void | CoRE MOF 2024 | Difference | |",
                "|---|---|---|---|---|",
            ]
            for row in q["rows"]:
                unit = f" {row['unit']}" if row["unit"] else ""
                diff = "" if row["difference"] is None else f"{row['difference']:+.3f}{unit}"
                mark = {True: "agrees", False: "**differs**", None: ""}[row["agrees"]]
                lines.append(f"| {row['label']} | {_fmt(row['the_void'])}{unit if row['difference'] is not None else ''} | "
                             f"{_fmt(row['reference'])}{unit if row['difference'] is not None else ''} | {diff} | {mark} |")
            for note in q["notes"]:
                lines += ["", f"- {note}"]
            lines += ["", f"- Data: {q['citation']}; licence {q['licence']}."]
        if q.get("candidates"):
            lines += ["", "Other candidates: " + ", ".join(
                f"`{c['entry']}` ({c['match']})" for c in q["candidates"]) + "."]

    if report.get("errors"):
        lines += ["", "## Analyses that failed", ""]
        lines += [f"- {k}: {v}" for k, v in report["errors"].items()]

    lines += ["", "## Methods", ""]
    lines += [f"- **{k}**: {v}" for k, v in report["methods"].items()]

    return "\n".join(lines) + "\n"


SUMMARY_COLUMNS = [
    "file", "formula", "space_group", "topology", "density_g_cm3", "cell_volume_A3",
    "lcd_A", "pld_A", "void_fraction_percent", "accessible_dimensionality",
    "channel_asa_m2_per_g", "channel_surface_area_A2", "pocket_surface_area_A2",
    "open_metal_sites", "dominant_pore_diameter_A", "errors",
]


def format_prediction(value, unit: str = "") -> str:
    """A predicted value as reports and the Predict Properties window
    show it: three significant figures, scientific for very small or
    large numbers (Henry coefficients span ten decades), whole degrees."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "–"
    value = float(value)
    if value != 0 and (abs(value) < 1e-2 or abs(value) >= 1e5):
        return f"{value:.2e}"
    if unit == "°C":
        return f"{value:.0f}"
    return f"{value:.3g}"


def summary_row(report: dict) -> dict:
    r = report.get("results", {})
    pore = r.get("pore", {})
    surface = r.get("surface_area", {})
    oms = r.get("oms", {})
    psd = r.get("psd", {})
    s = report["structure"]
    row = {
        "file": Path(report["input"]["file"]).name if report["input"].get("file") else "",
        "formula": s["formula"],
        "space_group": s["space_group"],
        "topology": s.get("topology"),
        "density_g_cm3": s["density_g_cm3"],
        "cell_volume_A3": s["cell_volume_A3"],
        "lcd_A": pore.get("lcd_A"),
        "pld_A": pore.get("pld_A"),
        "void_fraction_percent": pore.get("void_fraction_percent"),
        "accessible_dimensionality": pore.get("accessible_dimensionality"),
        "channel_asa_m2_per_g": surface.get("channel_asa_m2_per_g"),
        "channel_surface_area_A2": surface.get("channel_surface_area_A2"),
        "pocket_surface_area_A2": surface.get("pocket_surface_area_A2"),
        "open_metal_sites": oms.get("open_metal_sites"),
        "dominant_pore_diameter_A": psd.get("dominant_diameter_A"),
        "errors": "; ".join(f"{k}: {v}" for k, v in report.get("errors", {}).items()),
    }
    for name, guest in r.get("guests", {}).items():
        row[f"guest_{name}"] = guest["status"]
    for name, p in (r.get("predictions") or {}).get("models", {}).items():
        row[f"pred_{name}"] = p.get("value")
        row[f"pred_{name}_low"] = p.get("low")
        row[f"pred_{name}_high"] = p.get("high")
        row[f"pred_{name}_in_domain"] = p.get("in_domain")
    if "raspa_export" in r:
        directory = r["raspa_export"].get("directory")
        row["raspa_folder"] = Path(directory).name if directory else ""
    if "reference" in r:
        q = r["reference"]
        row["coremof_id"] = q.get("entry") or ""
        row["coremof_match"] = q.get("match") or "none"
        by_key = {x["property"]: x for x in q.get("rows", [])}
        for key, column in (("lcd", "lcd_A"), ("pld", "pld_A"), ("asa", "asa_m2_per_g")):
            if key in by_key:
                row[f"ccdc_{column}"] = by_key[key]["the_void"]
                row[f"coremof_{column}"] = by_key[key]["reference"]
        if "topology" in by_key:
            row["coremof_topology"] = by_key["topology"]["reference"]
        row["coremof_differs"] = ", ".join(
            x["label"] for x in q.get("rows", []) if x["agrees"] is False)
    return row


def summary_csv(rows: list[dict]) -> str:
    columns = list(SUMMARY_COLUMNS)
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {k: ("" if row.get(k) is None else row.get(k)) for k in columns}
        )
    return buffer.getvalue()
