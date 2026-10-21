from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import numpy as np

from thevoid_core.data.metal_classification import (
    OMS_CALIBRATED_SATURATION,
    OMS_SUPPORTED_METALS,
    is_metal,
)
from thevoid_core.services.coordination import (
    CoordinationEnvironment,
    analyze_coordination_environment,
)

logger = logging.getLogger(__name__)


class OMSClassification(enum.Enum):
    """Whether a metal centre's coordination sphere shows a genuine
    directional gap (OPEN) or is reasonably well covered on every side
    (SATURATED).
    """

    OPEN = "open"
    SATURATED = "saturated"


#: A reference "typical maximum" coordination number per supported
#: metal, used only to gate the asymmetry check - a site already at
#: this coordination number is treated as fully used up regardless of
#: any residual geometric distortion (real coordination spheres are
#: rarely perfectly symmetric even when genuinely saturated), so only
#: sites below their metal's typical maximum are even considered as
#: OMS candidates. These are common, representative values for MOF/
#: inorganic coordination chemistry, not a claim that no valid geometry
#: for that metal ever exceeds this number - Zr in particular commonly
#: reaches 7 or 8 in real MOF nodes (e.g. UiO-66's Zr6 clusters), which
#: is exactly why it is given a higher reference value than the rest
#: (and Hf, which forms the same Hf6 clusters, likewise).
TYPICAL_MAX_COORDINATION: dict[str, int] = {
    "Zn": 6,
    "Cu": 6,
    "Co": 6,
    "Ni": 6,
    "Fe": 6,
    "Cr": 6,
    "Ti": 6,
    "Al": 6,
    "Mg": 6,
    "Mn": 6,
    "Cd": 6,
    "V": 6,
    "Sc": 6,
    "Ga": 6,
    "Zr": 8,
    "Hf": 8,
    # Round 69: the metals calibrated on CoRE MOF 2024's labels
    # (data/oms_calibration.json). These are fitted numbers, not
    # textbook ones: the one that makes this same test agree best with
    # CoRE on half its SI structures.
    **OMS_CALIBRATED_SATURATION,
}

#: The asymmetry-magnitude threshold above which a site is classified
#: OPEN. Chosen by direct numerical calculation against reference
#: coordination geometries (see compute_coordination_asymmetry's
#: docstring for the method), not picked arbitrarily: every genuinely
#: symmetric, saturated geometry checked (tetrahedral, octahedral,
#: square planar) gives exactly 0.0, while every single-vacancy
#: geometry checked (square pyramidal missing its axial position - the
#: single most common real OMS case, as in desolvated Cu paddlewheels
#: and M-MOF-74 nodes - trigonal bipyramidal missing one equatorial
#: position, a three-coordinate T-shape) gives 0.20-0.33. 0.15 sits
#: comfortably below every missing-ligand case checked and comfortably
#: above every saturated one, with real margin on both sides.
DEFAULT_ASYMMETRY_THRESHOLD = 0.15

#: Round 50: the second, complementary OMS test - the widest empty cone
#: in the coordination sphere, i.e. the largest angle any direction
#: from the metal can be from its nearest ligand. The asymmetry vector
#: above is blind to vacancies that come in opposite pairs, because a
#: pair cancels: square-planar Cu/Ni/Pd/Pt with both axial sites open
#: (the metalloporphyrin nodes of PCN-222 and friends), trigonal-planar
#: three-coordinate sites, and linear two-coordinate ones all have
#: asymmetry exactly 0 and used to come out SATURATED.
#:
#: Reference widest gaps (computed, and checked in the tests):
#: octahedron 54.7, trigonal bipyramid 63.4, tetrahedron 70.5 - all
#: saturated - against square pyramid 90, square plane 90, trigonal
#: plane 90, seesaw 90+, linear 90. 80 degrees sits in the clear gap
#: between the two groups, 9.5 degrees above the tetrahedron.
DEFAULT_ANGULAR_GAP_THRESHOLD = 80.0

#: Atoms this close to a metal (angstrom) block a direction even when
#: CrystalNN does not count them as coordinated - chiefly the second
#: metal of a paddlewheel (Cu-Cu about 2.6 A), which sits in one of the
#: two axial positions of each square-planar Cu and must not be
#: reported as a vacancy.
BLOCKING_DISTANCE = 3.0

#: Most missing positions reported for one symmetric-vacancy site.
MAX_GAP_POSITIONS = 2


def compute_coordination_asymmetry(
    direction_vectors,
) -> tuple[np.ndarray, float]:
    """The core geometric measurement behind OMS detection: how lopsided
    a coordination sphere is, and which direction it is lopsided toward.

    Each ligand's position is reduced to a unit vector from the metal
    centre. For a genuinely symmetric, saturated coordination
    environment - ligands evenly distributed all the way around the
    metal, as in a real tetrahedron, octahedron, or square plane - these
    unit vectors cancel out and sum to (very close to) zero, the same
    property confirmed directly against a real rock-salt structure's
    octahedral Na environment via CrystalNN, not just idealised
    geometry (see test_coordination.py). When one ligand is missing
    from an otherwise-complete polyhedron, the vectors no longer
    cancel: the remaining sum points *away* from the gap, so the
    missing direction is the negative of that sum, normalised.

    Returns (missing_direction, magnitude): missing_direction is a unit
    vector (zeros if magnitude is ~0, since the direction of a
    near-zero vector is meaningless noise); magnitude is the vector
    sum's length divided by the number of ligands - 0.0 for perfect
    symmetry, up to 1.0 for a single ligand with nothing opposing it at
    all.
    """
    vectors = np.asarray(direction_vectors, dtype=float)

    if len(vectors) == 0:
        return np.zeros(3), 0.0

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms

    total = np.sum(unit_vectors, axis=0)
    magnitude = float(np.linalg.norm(total)) / len(vectors)

    if np.linalg.norm(total) < 1e-9:
        return np.zeros(3), magnitude

    missing_direction = -total / np.linalg.norm(total)

    return missing_direction, magnitude


def _sphere_directions(n: int = 2000) -> np.ndarray:
    index = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / n
    ring = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = np.pi * (3.0 - np.sqrt(5.0)) * index
    return np.column_stack([ring * np.cos(phi), ring * np.sin(phi), z])


_DIRECTIONS = _sphere_directions()


def _gap_at(direction, unit_vectors) -> float:
    """Angle (degrees) from direction to the nearest of unit_vectors."""
    d = direction / np.linalg.norm(direction)
    cosines = np.clip(unit_vectors @ d, -1.0, 1.0)
    return float(np.degrees(np.arccos(np.max(cosines))))


def angular_gaps(direction_vectors, *, threshold: float, separation: float = 60.0):
    """The widest empty cone in a coordination sphere.

    Returns (widest_gap_degrees, directions): directions are the
    centres of every distinct gap wider than threshold (at least
    `separation` degrees apart), widest first, each refined from a
    2000-direction survey by local maximisation.
    """
    from scipy.optimize import minimize

    vectors = np.asarray(direction_vectors, dtype=float)

    if len(vectors) == 0:
        return 180.0, []

    units = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    cosines = np.clip(_DIRECTIONS @ units.T, -1.0, 1.0)
    survey = np.degrees(np.arccos(np.max(cosines, axis=1)))

    def refine(start):
        def objective(angles):
            theta, phi = angles
            d = np.array(
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
            )
            return -_gap_at(d, units)

        theta0 = float(np.arccos(np.clip(start[2], -1.0, 1.0)))
        phi0 = float(np.arctan2(start[1], start[0]))
        result = minimize(
            objective, [theta0, phi0], method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 400},
        )
        theta, phi = result.x
        d = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        value = -float(result.fun)

        if value < _gap_at(start, units):
            return start, _gap_at(start, units)

        return d, value

    order = np.argsort(survey)[::-1]
    widest_direction, widest = refine(_DIRECTIONS[order[0]])

    directions = []

    for index in order:
        if survey[index] < threshold - 5.0:
            break

        start = _DIRECTIONS[index]

        if any(
            np.degrees(np.arccos(np.clip(start @ d, -1, 1))) < separation
            for d, _ in directions
        ):
            continue

        refined, value = refine(start)

        if value <= threshold:
            continue

        if any(
            np.degrees(np.arccos(np.clip(refined @ d, -1, 1))) < separation
            for d, _ in directions
        ):
            continue

        directions.append((refined, value))

    directions.sort(key=lambda item: -item[1])

    return max(widest, max((v for _, v in directions), default=0.0)), directions


@dataclass(frozen=True, slots=True)
class MissingPosition:
    """One direction, from the metal centre, where the coordination
    sphere shows a gap - a candidate location for a guest molecule or
    solvent ligand to (re)coordinate.
    """

    direction: tuple[float, float, float]
    confidence: float


@dataclass(frozen=True, slots=True)
class OMSSite:
    """One analyzed metal centre - every supported-metal site in the
    structure gets one of these, whether or not it turns out to be
    open, so the panel can show a complete table rather than only the
    positive hits.
    """

    site_index: int
    label: str
    metal_symbol: str
    position: tuple[float, float, float]
    coordination_environment: CoordinationEnvironment
    classification: OMSClassification
    asymmetry_score: float
    missing_positions: tuple[MissingPosition, ...]

    #: Round 50: the widest empty cone in the coordination sphere, in
    #: degrees (see DEFAULT_ANGULAR_GAP_THRESHOLD).
    angular_gap: float = 0.0

    #: Round 69: the coordination number at which this metal counts as
    #: saturated (TYPICAL_MAX_COORDINATION), recorded with the result
    #: because for the calibrated metals it is a fitted number.
    saturated_at: int = 0

    @property
    def coordination_number(self) -> int:
        return self.coordination_environment.coordination_number

    @property
    def is_open(self) -> bool:
        return self.classification is OMSClassification.OPEN


@dataclass(frozen=True, slots=True)
class OMSDetectionResult:
    sites: tuple[OMSSite, ...]

    #: Element symbols of metals present in the structure that OMS
    #: detection does not cover (sorted, each listed once), so a result
    #: with nothing analyzed can say *why* - "this structure's Na is
    #: outside the supported set" - instead of reading exactly like a
    #: structure with no metals in it at all.
    unsupported_metals: tuple[str, ...] = ()

    @property
    def oms_count(self) -> int:
        return sum(1 for site in self.sites if site.is_open)

    @property
    def open_sites(self) -> tuple[OMSSite, ...]:
        return tuple(site for site in self.sites if site.is_open)


def classify_site(
    coordination_number: int,
    asymmetry_magnitude: float,
    typical_max_coordination: int,
    threshold: float = DEFAULT_ASYMMETRY_THRESHOLD,
    angular_gap: float | None = None,
    gap_threshold: float = DEFAULT_ANGULAR_GAP_THRESHOLD,
) -> OMSClassification:
    """OPEN requires both a real directional gap (asymmetry above
    threshold) and headroom below the metal's typical maximum
    coordination - a site already at that maximum is treated as fully
    used up regardless of residual geometric distortion, which matters
    because real coordination spheres are essentially never perfectly
    symmetric even when genuinely saturated; requiring both conditions
    together is what keeps ordinary distortion in a fully-coordinated
    site from being misread as an open site.
    """
    if coordination_number >= typical_max_coordination:
        return OMSClassification.SATURATED

    if asymmetry_magnitude > threshold:
        return OMSClassification.OPEN

    # Round 50: a symmetric pair (or triple) of vacancies cancels in the
    # asymmetry vector but still leaves a wide empty cone.
    if angular_gap is not None and angular_gap > gap_threshold:
        return OMSClassification.OPEN

    return OMSClassification.SATURATED


def detect_open_metal_sites(
    structure,
    labels,
    *,
    threshold: float = DEFAULT_ASYMMETRY_THRESHOLD,
    gap_threshold: float = DEFAULT_ANGULAR_GAP_THRESHOLD,
    supported_metals=OMS_SUPPORTED_METALS,
) -> OMSDetectionResult:
    """Run Open Metal Site detection over every supported-metal site in
    a structure.

    For each site whose element is in supported_metals, this reuses
    analyze_coordination_environment (CrystalNN) - the same, already-
    verified coordination analysis the Coordination Environment feature
    uses - to get its coordination number and neighbour geometry, then
    applies compute_coordination_asymmetry to that geometry's direction
    vectors to decide whether it shows a genuine gap.

    Every supported-metal site is included in the result, open or not,
    so a caller can show a complete inventory rather than only the
    positive hits.
    """
    sites = []
    unsupported = set()

    for site_index, site in enumerate(structure.sites):
        symbol = _symbol(site)

        if symbol not in supported_metals:
            if is_metal(symbol):
                unsupported.add(symbol)
            continue

        environment = analyze_coordination_environment(
            structure, site_index, labels
        )

        direction_vectors = [
            neighbor.direction
            for neighbor in environment.neighbors
            if neighbor.direction is not None
        ]

        missing_direction, asymmetry = compute_coordination_asymmetry(
            direction_vectors
        )

        blocking = list(direction_vectors) + _close_contacts(
            structure, site_index, environment
        )
        widest_gap, gap_directions = angular_gaps(
            blocking, threshold=gap_threshold
        )

        typical_max = TYPICAL_MAX_COORDINATION.get(symbol, 6)

        classification = classify_site(
            environment.coordination_number,
            asymmetry,
            typical_max,
            threshold,
            angular_gap=widest_gap,
            gap_threshold=gap_threshold,
        )

        if classification is not OMSClassification.OPEN:
            missing_positions = ()
        elif asymmetry > threshold:
            # One-sided vacancy: the asymmetry vector points straight
            # at it (exactly, for an ideal polyhedron).
            missing_positions = (
                MissingPosition(
                    direction=tuple(float(v) for v in missing_direction),
                    confidence=asymmetry,
                ),
            )
        else:
            # Symmetric vacancies: one position per distinct wide gap.
            # (At most MAX_GAP_POSITIONS: a linear two-coordinate site's
            # "gap" is a whole ring, and listing arbitrary points of it
            # would say nothing more.)
            missing_positions = tuple(
                MissingPosition(
                    direction=tuple(float(v) for v in direction),
                    confidence=float(value) / 180.0,
                )
                for direction, value in gap_directions[:MAX_GAP_POSITIONS]
            )

        sites.append(
            OMSSite(
                site_index=site_index,
                label=environment.center_label,
                metal_symbol=symbol,
                position=tuple(float(v) for v in site.coords),
                coordination_environment=environment,
                classification=classification,
                asymmetry_score=asymmetry,
                missing_positions=missing_positions,
                angular_gap=widest_gap,
                saturated_at=typical_max,
            )
        )

    return OMSDetectionResult(
        sites=tuple(sites), unsupported_metals=tuple(sorted(unsupported))
    )


def calibration_of(metal: str) -> dict | None:
    """For a metal calibrated in round 69: its group, saturated
    coordination number and held-out agreement with CoRE MOF 2024, from
    data/oms_calibration.json. None for the other metals."""
    if metal not in OMS_CALIBRATED_SATURATION:
        return None
    for name, group in _calibration()["groups"].items():
        if metal in group["metals"]:
            held = group["held_out"]
            return {"group": name, "saturated_coordination": group["saturated_coordination"],
                    "held_out_items": held["items"], "held_out_agreement": held["agreement"]}
    return None


_CALIBRATION = None


def _calibration() -> dict:
    global _CALIBRATION
    if _CALIBRATION is None:
        import json

        from thevoid_core.data.metal_classification import OMS_CALIBRATION_FILE

        with open(OMS_CALIBRATION_FILE, encoding="utf-8") as handle:
            _CALIBRATION = json.load(handle)
    return _CALIBRATION


def _close_contacts(structure, site_index, environment) -> list:
    """Direction vectors to atoms within BLOCKING_DISTANCE of the metal
    that are *not* already among its coordinating neighbours - see
    BLOCKING_DISTANCE. They block a direction for the angular-gap test
    without changing the coordination number.
    """
    try:
        site = structure.sites[site_index]
        found = structure.get_neighbors(site, BLOCKING_DISTANCE)
    except Exception:
        logger.debug("Close-contact search failed.", exc_info=True)
        return []

    known = [
        np.asarray(n.direction, dtype=float)
        for n in environment.neighbors
        if n.direction is not None
    ]
    extra = []

    for neighbor in found:
        vector = np.asarray(neighbor.coords, dtype=float) - np.asarray(
            site.coords, dtype=float
        )

        if np.linalg.norm(vector) < 1e-6:
            continue

        if any(np.linalg.norm(vector - k) < 1e-3 for k in known):
            continue

        extra.append(vector)

    return extra


def _symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "?"
