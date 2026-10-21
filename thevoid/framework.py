"""The Framework object: one structure and everything the Void can
compute about it, with the same code and defaults as the GUI and
the-void-analyze."""

from __future__ import annotations

from pathlib import Path

from thevoid import results as R

#: Which radii an analysis uses: the Void's own (Bondi-based; the GUI's
#: and the-void-analyze's), or Zeo++'s default CCDC radii, as CoRE MOF
#: 2024 computed its values.
RADII = ("void", "ccdc")


def _radius_provider(radii: str):
    from thevoid_core.data.element_radii import CCDCRadiusProvider, ElementRadiusProvider

    if radii == "void":
        return ElementRadiusProvider()
    if radii == "ccdc":
        return CCDCRadiusProvider()
    raise ValueError(f"radii must be one of {RADII}, not {radii!r}")


def probe_radius(probe) -> tuple[str, float]:
    """(name, radius in Å) for a probe given by name ("N2", "CO2", "H2",
    "H2O", "CH4") or as a radius in Å."""
    from thevoid_core.data.probe_radii import PROBE_RADII

    if isinstance(probe, str):
        if probe not in PROBE_RADII:
            raise ValueError(f"Unknown probe {probe!r}; use one of {sorted(PROBE_RADII)} "
                             "or a radius in Å")
        return probe, PROBE_RADII[probe]
    radius = float(probe)
    if radius < 0:
        raise ValueError(f"A probe radius cannot be negative: {radius}")
    return "Custom", radius


class Framework:
    """A crystal structure and the Void's analyses of it.

    Make one with `thevoid.load(path_or_structure_or_atoms)`. Results are
    cached by their arguments, so asking twice is free; `clear_cache()`
    forgets them. The structure itself is never modified: analyses that
    need an ordered structure (disorder below 0.5 occupancy dropped) make
    their own copy, as the GUI does.
    """

    def __init__(self, structure, *, source=None, space_group=None, name=None):
        self._structure = structure
        self.source = str(source) if source else None
        self._space_group = space_group
        self.name = name or (Path(source).stem if source else structure.composition.reduced_formula)
        self._bonds = None
        self._cache = {}

    # -- the structure -------------------------------------------------

    @property
    def structure(self):
        """A copy of the pymatgen Structure."""
        return self._structure.copy()

    @property
    def formula(self) -> str:
        return self._structure.composition.formula.replace(" ", "")

    @property
    def n_sites(self) -> int:
        return len(self._structure)

    @property
    def space_group(self) -> str:
        if self._space_group is None:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

            try:
                self._space_group = SpacegroupAnalyzer(self._structure, symprec=0.01) \
                    .get_space_group_symbol()
            except Exception:
                self._space_group = "unknown"
        return self._space_group

    @property
    def density_g_cm3(self) -> float:
        from thevoid_core.services.report import crystal_density

        return crystal_density(self._structure)

    @property
    def cell_volume_A3(self) -> float:
        return float(self._structure.volume)

    @property
    def bonds(self) -> list:
        """Covalent bonds, detected as everywhere else in the Void."""
        if self._bonds is None:
            from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

            self._bonds = CovalentBondDetector().detect(self._structure)
        return list(self._bonds)

    def summary(self) -> dict:
        """The report's structure summary (formula, cell, density...)."""
        from thevoid_core.services.report import summarize_structure

        return summarize_structure(self._structure, space_group=self.space_group)

    def __repr__(self):
        return f"Framework({self.name!r}, {self.formula}, {self.space_group}, {self.n_sites} sites)"

    def clear_cache(self) -> None:
        """Forget every cached result (they are recomputed on request)."""
        self._cache.clear()

    def _cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    # -- analyses --------------------------------------------------------

    def pores(self, probe=1.2, *, grid_spacing: float = 0.3, radii: str = "void") -> R.PoreResult:
        """LCD, PLD, void fraction and accessible volume for a probe
        (a name such as "N2", or a radius in Å; default 1.2 Å, as
        Analysis > Pore Analysis and the-void-analyze)."""
        from thevoid_core.services.pore_analysis import analyze_pores
        from thevoid_core.services.report import pore_to_dict

        _name, radius = probe_radius(probe)

        def compute():
            raw = analyze_pores(self._structure, probe_radius=radius,
                                radius_provider=_radius_provider(radii), grid_spacing=grid_spacing)
            d = pore_to_dict(raw)
            return R.PoreResult(
                d["probe_radius_A"], d["lcd_A"], d.get("pld_A"), d["void_fraction_percent"],
                d["accessible_volume_A3"], d["non_accessible_volume_A3"],
                d["accessible_dimensionality"], d["lcd_pld_method"], d["accessibility_method"],
                tuple(d["grid_shape"]), d, raw)

        return self._cached(("pores", radius, grid_spacing, radii), compute)

    def surface_area(self, probe="N2", *, samples_per_atom: int = 1000, seed: int = 0,
                     radii: str = "void") -> R.SurfaceAreaResult:
        """Accessible surface area by probe sampling (default N₂,
        1000 points per atom, as the GUI)."""
        from thevoid_core.services.report import surface_area_to_dict
        from thevoid_core.services.surface_area import analyze_surface_area

        name, radius = probe_radius(probe)

        def compute():
            raw = analyze_surface_area(self._structure, probe_radius=radius, probe_name=name,
                                       radius_provider=_radius_provider(radii),
                                       n_samples_per_atom=samples_per_atom, seed=seed)
            d = surface_area_to_dict(raw)
            return R.SurfaceAreaResult(
                d["probe"], d["probe_radius_A"], d["samples_per_atom"],
                d["accessible_surface_area_A2"], d["channel_surface_area_A2"],
                d.get("pocket_surface_area_A2") or 0.0, d["geometric_surface_area_A2"],
                d["asa_m2_per_g_including_pockets"], d.get("channel_asa_m2_per_g") or 0.0,
                d["asa_m2_per_cm3"], d, raw)

        return self._cached(("surface", radius, name, samples_per_atom, seed, radii), compute)

    def guest(self, guest="CO2", *, diameter: float | None = None, grid_spacing: float = 0.3,
              radii: str = "void") -> R.GuestResult:
        """Can a guest get in? `guest` is a name (N2, CO2, H2, H2O, CH4:
        the kinetic diameter is twice its probe radius) or, with
        `diameter` (Å), any label."""
        from thevoid_core.services.guest_accessibility import analyze_guest_accessibility
        from thevoid_core.services.report import _guest_diameter, guest_to_dict

        if diameter is None:
            name, diameter = _guest_diameter(guest)
        else:
            name, diameter = str(guest), float(diameter)

        def compute():
            raw = analyze_guest_accessibility(
                self._structure, guest_diameter=diameter, guest_name=name,
                radius_provider=_radius_provider(radii), grid_spacing=grid_spacing)
            d = guest_to_dict(raw)
            status = getattr(raw.status, "value", raw.status)
            return R.GuestResult(d["guest"], d["guest_diameter_A"], str(status),
                                 d.get("pld_A"), d["lcd_A"], d["largest_accessible_cavity_A"],
                                 d["bottleneck_count"], d, raw)

        return self._cached(("guest", name, diameter, grid_spacing, radii), compute)

    def open_metal_sites(self) -> R.OpenMetalSitesResult:
        """Undercoordinated metal centres (CrystalNN environments)."""
        from thevoid_core.services.oms_detection import detect_open_metal_sites
        from thevoid_core.services.report import oms_to_dict
        from thevoid_core.services.site_labels import compute_site_labels

        def compute():
            raw = detect_open_metal_sites(self._structure, compute_site_labels(self._structure))
            d = oms_to_dict(raw)
            sites = tuple(R.MetalSite(s["label"], s["metal"], s["coordination_number"],
                                      s["classification"], s.get("asymmetry"),
                                      s.get("widest_empty_cone_deg")) for s in d["sites"])
            return R.OpenMetalSitesResult(d["open_metal_sites"], d["metal_centres_analyzed"],
                                          tuple(d["not_checked_metals"]), sites, d, raw)

        return self._cached(("oms",), compute)

    def topology(self, *, rcsr_archive=None) -> R.TopologyResult:
        """The underlying net, named by RCSR code where the library has
        it (95 nets bundled; rcsr_archive: a Systre .arc for all)."""
        from thevoid_core.services.topology import analyze_topology, topology_to_dict
        from thevoid_core.services.topology.library import default_library

        def compute():
            raw = analyze_topology(self._structure, library=default_library(rcsr_archive))
            d = topology_to_dict(raw)
            return R.TopologyResult(raw.summary, raw.status, raw.rcsr, raw.interpenetration,
                                    d, raw)

        return self._cached(("topology", str(rcsr_archive)), compute)

    def partition(self) -> R.PartitionResult:
        """Linkers and metal nodes of one unit cell (MOF Analysis)."""
        from thevoid_core.data.metal_classification import is_metal
        from thevoid_core.services.mof_analysis import partition_structure

        def compute():
            raw, _images = partition_structure(self._structure, self.bonds, is_metal)
            linkers = tuple(R.Component(x.formula, x.atom_count, x.dimensionality)
                            for x in raw.linkers)
            nodes = tuple(R.Component(x.formula, x.atom_count, x.dimensionality,
                                      tuple(x.metal_symbols)) for x in raw.metal_nodes)
            d = {"linkers": [c.__dict__ for c in linkers],
                 "nodes": [dict(c.__dict__, metals=list(c.metals)) for c in nodes],
                 "excluded_disorder_sites": raw.excluded_disorder_sites}
            return R.PartitionResult(linkers, nodes, raw.excluded_disorder_sites, d, raw)

        return self._cached(("partition",), compute)

    def pore_size_distribution(self, *, bin_width: float = 0.2, grid_spacing: float = 0.3,
                               radii: str = "void") -> R.PoreSizeDistributionResult:
        """Geometric PSD (Gelb & Gubbins 1999), as Analysis > Pore Size
        Distribution."""
        from thevoid_core.services.pore_size_distribution import compute_pore_size_distribution
        from thevoid_core.services.report import psd_to_dict

        def compute():
            raw = compute_pore_size_distribution(self._structure,
                                                 radius_provider=_radius_provider(radii),
                                                 bin_width=bin_width, grid_spacing=grid_spacing)
            d = psd_to_dict(raw)
            bins = d["bins"]
            return R.PoreSizeDistributionResult(
                d["bin_width_A"], d["pore_volume_A3"], d["void_fraction"],
                d.get("dominant_diameter_A"),
                tuple(b["diameter_A"] for b in bins), tuple(b["distribution_per_A"] for b in bins),
                tuple(b["cumulative_volume_A3"] for b in bins), d, raw)

        return self._cached(("psd", bin_width, grid_spacing, radii), compute)

    def xrd(self, *, wavelength: str = "CuKa", two_theta_range=(5.0, 50.0),
            min_relative_intensity: float = 5.0, merge_tolerance: float = 0.05) -> R.XRDResult:
        """Simulated powder pattern, with the GUI's peak reduction
        (reflections within merge_tolerance° summed, then those below
        min_relative_intensity % of the strongest dropped)."""
        from thevoid_core.services.report import xrd_to_dict
        from thevoid_core.services.xrd.powder_xrd_simulator import PowderXRDSimulator

        def compute():
            full = PowderXRDSimulator().simulate(self._structure,
                                                 two_theta_range=tuple(two_theta_range),
                                                 wavelength=wavelength)
            raw = full.reduced(min_relative_intensity=min_relative_intensity,
                               merge_tolerance=merge_tolerance)
            d = xrd_to_dict(raw, top=len(raw.two_theta))
            hkl = tuple(tuple(tuple(h.get("hkl", ())) for h in hs) for hs in raw.hkls)
            return R.XRDResult(float(raw.wavelength), tuple(float(v) for v in raw.two_theta),
                               tuple(float(v) for v in raw.intensities),
                               tuple(float(v) for v in raw.d_spacings), hkl, d, raw)

        return self._cached(("xrd", wavelength, tuple(two_theta_range), min_relative_intensity,
                             merge_tolerance), compute)

    # -- CoRE MOF 2024 ----------------------------------------------------

    def coremof_matches(self) -> list:
        """[(entry id, confidence, how it was found)], best first."""
        from thevoid_core.services.reference.coremof import find_matches

        return [(m.entry.id, m.confidence, m.how)
                for m in find_matches(self._structure, path=self.source)]

    def compare_coremof(self, *, entry: str | None = None,
                        samples_per_atom: int = 1000) -> R.ComparisonResult:
        """The Void's LCD, PLD, surface areas, topology and open metal
        sites beside CoRE MOF 2024's, computed CoRE's way. entry: a CoRE
        id to compare against instead of the best match."""
        from thevoid_core.services.reference.compare import (
            compare_with_coremof,
            comparison_to_dict,
        )
        from thevoid_core.services.reference.coremof import find_matches

        def compute():
            matches = find_matches(self._structure, path=self.source)
            if entry is not None:
                chosen = [m for m in matches if m.entry.id == entry]
                if not chosen:
                    raise ValueError(f"{entry} is not among this structure's CoRE matches: "
                                     f"{[m.entry.id for m in matches] or 'none'}")
                match = chosen[0]
            elif matches:
                match = matches[0]
            else:
                raise ValueError("No CoRE MOF 2024 entry matches this structure")
            raw = compare_with_coremof(self._structure, match, samples_per_atom=samples_per_atom)
            d = comparison_to_dict(raw)
            rows = tuple(R.ComparisonRow(r.key, r.label, r.void, r.reference, r.unit,
                                         r.difference, r.agrees, r.note) for r in raw.rows)
            return R.ComparisonResult(raw.entry_id, raw.confidence, raw.how, rows,
                                      tuple(raw.notes), d, raw)

        return self._cached(("coremof", entry, samples_per_atom), compute)

    def predict(self, models=None, *, samples_per_atom: int = 1000):
        """Property predictions (Predictions: index by model name).
        models: names of bundled models (thevoid.models()), folders
        the-void-train wrote, or Models; None: every bundled model.
        Needs scikit-learn (pip install -e ".[ml]").

        Bundled models read CoRE MOF 2024's descriptors, computed as
        CoRE does (CCDC radii, 1.655 Å probe): the pores, surface area,
        open metal sites and topology behind them are cached like any
        other result here, so pores(1.655, radii="ccdc") afterwards is
        free. Models trained on a batch run analyze() with the batch's
        options."""
        from thevoid.models import Model, Predictions

        from thevoid_core.services.ml.predict import predict_structure, resolve

        chosen = resolve([m.raw if isinstance(m, Model) else m for m in (
            [models] if isinstance(models, (str, Path, Model)) else models)]
            if models is not None else None)
        cached = {}
        if any(m.descriptor_set == "coremof-v1" for m in chosen):
            cached["pores"] = self.pores(1.655, radii="ccdc").raw
            cached["surface"] = self.surface_area(1.655, samples_per_atom=samples_per_atom,
                                                  radii="ccdc").raw
            for key, compute in (("oms", self.open_metal_sites), ("topology", self.topology)):
                try:
                    cached[key] = compute().raw
                except Exception:  # coremof_descriptors records the failure itself
                    pass
        summary = None
        void_models = [m for m in chosen if m.descriptor_set == "void-v1"]
        if void_models:
            from thevoid_core.services.report import summary_row

            options = void_models[0].card.get("analysis_options", {})
            summary = summary_row(self.analyze(**options).data)
        raw = predict_structure(self._structure, chosen, summary_row=summary,
                                samples_per_atom=samples_per_atom, cached=cached)
        return Predictions(raw, name=self.name)

    # -- everything at once ------------------------------------------------

    def analyze(self, **options):
        """Every analysis the-void-analyze would run with these options
        (AnalysisConfig's fields: pore, pore_probe_radius, surface_area,
        surface_probe, oms, psd, guests, xrd, topology, compare_coremof,
        ...), as a Report. A failing analysis is recorded in
        report.errors, not raised."""
        from thevoid.report import Report, analysis_config

        from thevoid_core.services.report import run_analysis

        report = run_analysis(self._structure, analysis_config(options), source=self.source)
        report["structure"]["space_group"] = self.space_group
        return Report(report, name=self.name)

    # -- building ----------------------------------------------------------

    def functionalization_sites(self, max_per_linker: int = 4):
        """The ring C-H sites, linker orbits and distinct patterns that
        functionalize() can use (the-void-functionalize --list)."""
        from thevoid.variants import Sites

        return Sites(self._functionalization_parent(max_per_linker))

    def _functionalization_parent(self, max_per_linker):
        from thevoid_core.services import functionalize as fz

        def compute():
            return fz.analyze_parent(self._structure, source=self.source,
                                     max_per_linker=max_per_linker)

        return self._cached(("fz-parent", max_per_linker), compute)

    def functionalize(self, groups=("NH2",), *, per_linker=(1,), orbits=None,
                      ring_turn=True):
        """Variants with ring hydrogens of the linkers replaced by
        `groups` ("F", "Cl", "Br", "I", "CH3", "CF3", "OH", "OCH3",
        "NH2", "NO2", "CN"), `per_linker` groups on each linker, the same
        pattern on every linker of an orbit. Each variant is placed
        rigidly and checked; rejected ones are kept in the list with
        status "clash" and their reason. ring_turn: turn a crowded
        variant's rings about the linker axis by the smallest angle that
        clears it (the angle is each Variant's ring_turn_deg). Returns
        Variants; write them with .write(folder)."""
        from thevoid_core.services import functionalize as fz
        from thevoid.variants import Variants

        if isinstance(groups, str):
            groups = (groups,)
        if isinstance(per_linker, int):
            per_linker = (per_linker,)
        chosen = [fz.group(g) for g in groups]
        per_linker = tuple(sorted(set(int(m) for m in per_linker)))
        parent = self._functionalization_parent(max(max(per_linker), 1))
        built = fz.build_all(parent, chosen, per_linker=per_linker, orbits=orbits,
                             ring_turn=ring_turn)
        return Variants(parent, built, chosen, per_linker, parent_name=self.name,
                        ring_turn=ring_turn)

    # -- writing -----------------------------------------------------------

    def write_cif(self, path) -> Path:
        """The structure as a CIF (as written, disorder included)."""
        from pymatgen.io.cif import CifWriter

        path = Path(path)
        CifWriter(self._structure).write_file(str(path))
        return path

    def raspa_input(self, folder, **settings) -> Path:
        """A RASPA2 GCMC input folder (Simulate > Export RASPA2 GCMC
        Input Set); settings are RaspaExportSettings' fields
        (adsorbate="CO2", temperature_K=298, pressures_Pa=(1e5,), ...)."""
        from dataclasses import fields as dc_fields

        from thevoid_core.services.raspa_export import (
            RaspaExportSettings,
            build_raspa_export,
            write_raspa_export,
        )

        valid = {f.name for f in dc_fields(RaspaExportSettings)}
        unknown = sorted(set(settings) - valid)
        if unknown:
            raise TypeError(f"Unknown RASPA setting(s) {unknown}; valid: {sorted(valid)}")
        if "pressures_Pa" in settings:
            settings["pressures_Pa"] = tuple(settings["pressures_Pa"])
        options = RaspaExportSettings(**settings)
        options.validate()
        export = build_raspa_export(self._structure, None, options, source=self.source)
        folder = Path(folder)
        write_raspa_export(export, folder)
        return folder

    def to_pymatgen(self):
        """A copy of the pymatgen Structure (as .structure)."""
        return self.structure

    def to_ase(self):
        """An ASE Atoms copy (needs ASE; ordered structures only, since
        ASE has no partial occupancy)."""
        from pymatgen.io.ase import AseAtomsAdaptor

        if not self._structure.is_ordered:
            raise ValueError("ASE cannot hold partial occupancy; this structure is disordered")
        return AseAtomsAdaptor.get_atoms(self._structure)
