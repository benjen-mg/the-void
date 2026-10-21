"""Result objects returned by the public API.

Every result is a frozen dataclass whose fields are named exactly as
the same numbers are in the Void's JSON reports and summary CSV
(`lcd_A`, `pld_A`, `channel_asa_m2_per_g`, ...). A value computed in
Python, in a batch or in the GUI therefore goes by one name, with its
unit in the name. `to_dict()` gives the report's own dictionary, and
`raw` is the internal object behind the result. `raw` is not part of
the stable API: it can change between versions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields


def _public_fields(cls):
    return [f.name for f in fields(cls) if f.name not in ("raw", "details")]


class _Result:
    def to_dict(self) -> dict:
        """The report's own dictionary for this result (JSON-ready)."""
        return dict(self.details)

    def __repr__(self):
        shown = ", ".join(f"{name}={getattr(self, name)!r}" for name in _public_fields(type(self))
                          if not isinstance(getattr(self, name), (list, tuple)) or
                          len(getattr(self, name)) <= 4)
        return f"{type(self).__name__}({shown})"


@dataclass(frozen=True, repr=False)
class PoreResult(_Result):
    """Analysis > Pore Analysis: LCD, PLD, void fraction and volumes.

    lcd_A / pld_A: largest cavity and pore limiting diameters (Å);
    pld_A is None when nothing percolates. void_fraction_percent is the
    geometric free volume; accessible_volume_A3 is the part a probe of
    probe_radius_A can reach from a channel, non_accessible_volume_A3
    the pockets. accessible_dimensionality: 0 (none) to 3."""

    probe_radius_A: float
    lcd_A: float
    pld_A: float | None
    void_fraction_percent: float
    accessible_volume_A3: float
    non_accessible_volume_A3: float
    accessible_dimensionality: int
    lcd_pld_method: str
    accessibility_method: str
    grid_shape: tuple
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


@dataclass(frozen=True, repr=False)
class SurfaceAreaResult(_Result):
    """Analysis > Surface Area: accessible surface by probe sampling.

    channel_asa_m2_per_g is what Zeo++ and most screening studies call
    ASA (channel surface only); asa_m2_per_g_including_pockets adds the
    pockets a probe cannot reach from outside."""

    probe: str
    probe_radius_A: float
    samples_per_atom: int
    accessible_surface_area_A2: float
    channel_surface_area_A2: float
    pocket_surface_area_A2: float
    geometric_surface_area_A2: float
    asa_m2_per_g_including_pockets: float
    channel_asa_m2_per_g: float
    asa_m2_per_cm3: float
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


@dataclass(frozen=True, repr=False)
class GuestResult(_Result):
    """Analysis > Guest Accessibility: can a guest of guest_diameter_A
    get through? status is "accessible", "partially_accessible" (fits in
    a cavity but no channel carries it through) or "inaccessible"."""

    guest: str
    guest_diameter_A: float
    status: str
    pld_A: float | None
    lcd_A: float
    largest_accessible_cavity_A: float
    bottleneck_count: int
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


@dataclass(frozen=True)
class MetalSite:
    label: str
    metal: str
    coordination_number: int
    classification: str
    asymmetry: float | None
    widest_empty_cone_deg: float | None


@dataclass(frozen=True, repr=False)
class OpenMetalSitesResult(_Result):
    """Analysis > Open Metal Sites: every supported metal centre and its
    classification; open_metal_sites counts the open ones."""

    open_metal_sites: int
    metal_centres_analyzed: int
    not_checked_metals: tuple
    sites: tuple
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


@dataclass(frozen=True, repr=False)
class TopologyResult(_Result):
    """The underlying net: summary like "pcu 6-c" or "dia 4-c, 2-fold
    interpenetrated"; rcsr the code alone (None if not identified)."""

    summary: str
    status: str
    rcsr: str | None
    interpenetration: int
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


@dataclass(frozen=True, repr=False)
class PoreSizeDistributionResult(_Result):
    """Geometric pore size distribution (Gelb & Gubbins): diameters_A
    are bin centres, distribution_per_A the PSD, cumulative_volume_A3
    the volume in pores at least that wide."""

    bin_width_A: float
    pore_volume_A3: float
    void_fraction: float
    dominant_diameter_A: float | None
    diameters_A: tuple
    distribution_per_A: tuple
    cumulative_volume_A3: tuple
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


@dataclass(frozen=True, repr=False)
class XRDResult(_Result):
    """Simulated powder pattern: every peak after the 5% relative
    intensity cut and 0.05° merge (the GUI's defaults)."""

    wavelength_A: float
    two_theta_deg: tuple
    relative_intensity: tuple
    d_A: tuple
    hkl: tuple
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)

    def strongest(self, n: int = 10):
        """[(two_theta_deg, relative_intensity, d_A, hkl)] of the n
        strongest peaks, in 2-theta order."""
        order = sorted(range(len(self.relative_intensity)),
                       key=lambda i: -self.relative_intensity[i])[:n]
        return [(self.two_theta_deg[i], self.relative_intensity[i], self.d_A[i], self.hkl[i])
                for i in sorted(order, key=lambda i: self.two_theta_deg[i])]


@dataclass(frozen=True)
class ComparisonRow:
    key: str
    label: str
    the_void: object
    reference: object
    unit: str
    difference: float | None
    agrees: bool | None
    note: str


@dataclass(frozen=True, repr=False)
class ComparisonResult(_Result):
    """Analysis > Compare with CoRE MOF 2024. The Void's side is
    recomputed as CoRE computes its own values (Zeo++'s CCDC radii,
    1.655 Å probe). confidence: "identified", "possible" or "name
    only"; only "identified" compares this structure with itself."""

    entry_id: str
    confidence: str
    how: str
    rows: tuple
    notes: tuple
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)

    @property
    def flagged(self):
        """Rows where the two sides differ by more than the tolerance."""
        return tuple(r for r in self.rows if r.agrees is False)

    def row(self, key: str):
        for r in self.rows:
            if r.key == key:
                return r
        raise KeyError(key)


@dataclass(frozen=True)
class Component:
    """A linker or metal node of MOF Analysis."""

    formula: str
    atoms: int
    dimensionality: int
    metals: tuple = ()


@dataclass(frozen=True, repr=False)
class PartitionResult(_Result):
    """Analysis > Analyze MOF Structure: linkers and nodes of one cell,
    each once."""

    linkers: tuple
    nodes: tuple
    excluded_disorder_sites: int
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)

    def linker_formulas(self) -> dict:
        """{formula: count}."""
        counts = {}
        for c in self.linkers:
            counts[c.formula] = counts.get(c.formula, 0) + 1
        return counts

    def node_formulas(self) -> dict:
        counts = {}
        for c in self.nodes:
            counts[c.formula] = counts.get(c.formula, 0) + 1
        return counts
