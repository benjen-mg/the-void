from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

import numpy as np

from thevoid_core.services.pore_analysis import (
    build_grid_edges,
    refined_lcd_pld,
    build_periodic_atoms,
    compute_distance_field,
    major_occupancy_structure,
    network_assisted_sweep,
    sweep_connectivity,
)

logger = logging.getLogger(__name__)


class AccessibilityStatus(enum.Enum):
    """A guest molecule's relationship to a framework's pore network.

    ACCESSIBLE: the guest can diffuse all the way through the material
    via at least one percolating channel (its diameter does not exceed
    the pore-limiting diameter of that channel).

    PARTIALLY_ACCESSIBLE: the guest fits inside at least one cavity
    somewhere in the structure (its diameter does not exceed the
    largest cavity diameter), but every percolating channel is too
    narrow for it to actually travel between cavities - it could be
    loaded into an accessible pocket from outside, or be present as a
    trapped/kinetically-restricted guest, but cannot freely diffuse
    through the bulk material the way an ACCESSIBLE guest can.

    INACCESSIBLE: the guest does not fit anywhere in the structure at
    all - not even the single largest cavity is big enough for it.
    """

    ACCESSIBLE = "accessible"
    PARTIALLY_ACCESSIBLE = "partially_accessible"
    INACCESSIBLE = "inaccessible"


def classify_accessibility(
    lcd: float, pld: float | None, guest_diameter: float
) -> AccessibilityStatus:
    """The three-way classification, from the same two numbers pore
    analysis already reports: Largest Cavity Diameter (the widest point
    anywhere, whether or not it is reachable) and Pore Limiting
    Diameter (the narrowest bottleneck along the best percolating
    path, undefined if nothing percolates at all).

    A guest that fits through the narrowest bottleneck of some
    percolating channel (guest_diameter <= pld) can obviously also fit
    into every cavity along that channel, so ACCESSIBLE implies the
    guest would also satisfy the PARTIALLY_ACCESSIBLE condition - the
    three categories are checked in an order (ACCESSIBLE first) that
    relies on that implication rather than needing to re-derive it.
    """
    if pld is not None and guest_diameter <= pld:
        return AccessibilityStatus.ACCESSIBLE

    if guest_diameter <= lcd:
        return AccessibilityStatus.PARTIALLY_ACCESSIBLE

    return AccessibilityStatus.INACCESSIBLE


def largest_accessible_cavity(
    distance_field: np.ndarray, accessible_mask: np.ndarray
) -> float:
    """Diameter of the largest cavity that is actually part of the
    accessible (percolating) network - not just the largest cavity
    anywhere, which pore_analysis's own LCD already reports and which
    can be a sealed, unreachable pocket. A guest small enough to fit
    into that sealed pocket still could not get there, so it is not a
    meaningful "largest accessible cavity" for guest-diffusion purposes.

    0.0 if nothing in the structure is accessible at all.
    """
    accessible_values = distance_field[accessible_mask]

    if len(accessible_values) == 0:
        return 0.0

    return 2.0 * float(np.max(accessible_values))


@dataclass(frozen=True, slots=True)
class Bottleneck:
    """One local constriction point along the accessible network - a
    grid position where the local pore radius is at least as small as
    every accessible neighbour's, i.e. a point a guest has to squeeze
    through rather than pass freely.
    """

    grid_index: tuple[int, int, int]
    position: np.ndarray
    radius: float


def identify_bottlenecks(
    distance_field: np.ndarray,
    grid_shape: tuple[int, int, int],
    accessible_mask: np.ndarray,
    grid_positions: np.ndarray,
) -> list[Bottleneck]:
    """Every accessible grid point whose local pore radius is a local
    minimum among its accessible neighbours - a strict minimum against
    at least one neighbour, so a flat plateau of equally-wide points
    does not get every single point on it labelled a "bottleneck".

    Plural by design: a framework can have more than one distinct
    narrow point along its channel system (or more than one separate
    channel, each with its own bottleneck), and all of them are
    relevant to where a guest gets held up, not just the single
    narrowest one already reported as the pore-limiting diameter.
    """
    edges = build_grid_edges(grid_shape)

    neighbors: dict[int, list[int]] = {}
    for a, b, _offset in edges:
        neighbors.setdefault(a, []).append(b)
        neighbors.setdefault(b, []).append(a)

    bottlenecks = []

    for node in range(len(distance_field)):
        if not accessible_mask[node]:
            continue

        node_radius = distance_field[node]
        accessible_neighbors = [
            neighbor
            for neighbor in neighbors.get(node, ())
            if accessible_mask[neighbor]
        ]

        if not accessible_neighbors:
            continue

        neighbor_radii = distance_field[accessible_neighbors]

        is_local_minimum = np.all(node_radius <= neighbor_radii)
        is_strict_somewhere = np.any(node_radius < neighbor_radii)

        if is_local_minimum and is_strict_somewhere:
            i = node // (grid_shape[1] * grid_shape[2])
            j = (node // grid_shape[2]) % grid_shape[1]
            k = node % grid_shape[2]

            bottlenecks.append(
                Bottleneck(
                    grid_index=(i, j, k),
                    position=grid_positions[node],
                    radius=float(node_radius),
                )
            )

    return bottlenecks


@dataclass(frozen=True, slots=True)
class GuestAccessibilityResult:
    """Everything the Guest Accessibility panel needs to display, plus
    enough to render channels, blocked regions, and bottlenecks in the
    3D view.
    """

    guest_name: str
    guest_diameter: float

    status: AccessibilityStatus

    #: Pore-limiting diameter - the narrowest bottleneck along the best
    #: percolating path, independent of the guest (a property of the
    #: framework alone), reported here as "minimum window diameter" and
    #: again as "maximum guest diameter" (the same number, viewed from
    #: two directions: how narrow the window is, and the largest guest
    #: that could still get through it) since both are literally the
    #: same underlying pore-limiting-diameter quantity.
    minimum_window_diameter: float | None
    maximum_guest_diameter: float | None

    #: Largest Cavity Diameter - the widest point anywhere, reachable
    #: or not (a property of the framework alone, independent of the
    #: guest), for comparison against largest_accessible_cavity below.
    largest_cavity_diameter: float

    #: Diameter of the largest cavity that is actually reachable via
    #: the accessible (percolating) network - see
    #: largest_accessible_cavity's own docstring for why this can
    #: differ from largest_cavity_diameter.
    largest_accessible_cavity: float

    #: Every local constriction found in the accessible network.
    bottlenecks: list[Bottleneck]

    #: The largest probe radius that would still be fully ACCESSIBLE
    #: (not merely partially accessible) for this structure - literally
    #: minimum_window_diameter / 2, offered as a directly actionable
    #: number for screening ("use a probe up to this size and it will
    #: percolate through this framework"). None if nothing percolates
    #: at all, whatever probe size is used.
    suggested_probe_radius: float | None

    grid_shape: tuple[int, int, int]

    #: Grid positions (Cartesian), for the 3D view: accessible_points
    #: are part of the percolating channel network; blocked_points are
    #: real void space (not inside any atom) that is too narrow for
    #: this specific guest, whether or not that void space belongs to
    #: an otherwise-accessible region elsewhere.
    accessible_points: np.ndarray
    blocked_points: np.ndarray


def analyze_guest_accessibility(
    structure,
    *,
    guest_diameter: float,
    guest_name: str,
    radius_provider,
    grid_spacing: float = 0.3,
    max_grid_points_per_axis: int = 60,
) -> GuestAccessibilityResult:
    """Run the full guest-accessibility pipeline on a structure for one
    guest molecule (or a custom diameter).

    guest_diameter, not guest_radius: guest molecules are conventionally
    discussed and reported by their kinetic *diameter* in this context
    (the same convention data.probe_radii's underlying kinetic-diameter
    values come from before being halved into radii there) - halved to
    a radius only internally, at the point where it is actually used
    as a probe radius against the framework's van der Waals surfaces.
    """
    structure = major_occupancy_structure(structure)

    guest_radius = guest_diameter / 2.0

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)

    grid_shape = tuple(
        int(
            min(
                max_grid_points_per_axis,
                max(4, round(length / grid_spacing)),
            )
        )
        for length in lengths
    )

    symbols = [_site_symbol(site) for site in structure.sites]

    if len(symbols) == 0:
        logger.warning(
            "Guest accessibility analysis requested on a structure "
            "with no atoms; reporting an entirely open, fully "
            "accessible cell."
        )
        return GuestAccessibilityResult(
            guest_name=guest_name,
            guest_diameter=guest_diameter,
            status=AccessibilityStatus.ACCESSIBLE,
            minimum_window_diameter=None,
            maximum_guest_diameter=None,
            largest_cavity_diameter=float("inf"),
            largest_accessible_cavity=float("inf"),
            bottlenecks=[],
            suggested_probe_radius=None,
            grid_shape=grid_shape,
            accessible_points=np.zeros((0, 3)),
            blocked_points=np.zeros((0, 3)),
        )

    vdw_radii = np.array(
        [radius_provider.get_vdw_radius(symbol) for symbol in symbols]
    )
    max_radius = float(np.max(vdw_radii))
    cutoff = max_radius + guest_radius + 2.0

    atom_positions, atom_radii = build_periodic_atoms(
        structure, vdw_radii, cutoff
    )

    nx, ny, nz = grid_shape
    frac_x = (np.arange(nx) + 0.5) / nx
    frac_y = (np.arange(ny) + 0.5) / ny
    frac_z = (np.arange(nz) + 0.5) / nz
    grid_frac = np.stack(
        np.meshgrid(frac_x, frac_y, frac_z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid_positions = grid_frac @ lattice

    distance_field = compute_distance_field(
        grid_positions, atom_positions, atom_radii
    )

    lcd = 2.0 * float(np.max(distance_field))

    sweep_result = sweep_connectivity(
        distance_field, grid_shape, guest_radius
    )
    grid_pld_radius = sweep_result.pld_radius
    # Round 55: accessibility with the Voronoi network's help (see
    # pore_geometry.network_accessibility); the network is reused below
    # for the exact LCD/PLD.
    sweep_result, _method, prepared = network_assisted_sweep(
        np.asarray(structure.lattice.matrix, dtype=float),
        [site.frac_coords for site in structure.sites],
        vdw_radii,
        grid_frac=grid_frac,
        distance_field=distance_field,
        grid_shape=grid_shape,
        probe_radius=guest_radius,
        grid_sweep=sweep_result,
    )

    pld = (
        2.0 * sweep_result.pld_radius
        if sweep_result.pld_radius is not None
        else None
    )
    lcd, pld, _method = refined_lcd_pld(
        structure,
        vdw_radii,
        grid_points=grid_positions,
        distance_field=distance_field,
        grid_lcd=lcd,
        grid_pld=pld,
        prepared=prepared,
        grid_shape=grid_shape,
    )

    status = classify_accessibility(lcd, pld, guest_diameter)

    if (
        status is AccessibilityStatus.ACCESSIBLE
        and not sweep_result.accessible.any()
        and grid_pld_radius is not None
    ):
        # The exact PLD lets this guest through but the coarser grid
        # does not quite resolve the window (its PLD is lower). Show the
        # network the grid does resolve - traced at its own bottleneck -
        # rather than an empty picture that contradicts the verdict.
        sweep_result = sweep_connectivity(
            distance_field, grid_shape, grid_pld_radius
        )

    accessible_cavity = largest_accessible_cavity(
        distance_field, sweep_result.accessible
    )

    bottlenecks = identify_bottlenecks(
        distance_field, grid_shape, sweep_result.accessible, grid_positions
    )

    suggested_radius = pld / 2.0 if pld is not None else None

    # "Blocked" for this guest specifically: real void space (not
    # inside any atom) whose local radius is below the guest's own
    # radius - narrower than the guest needs, not merely "not part of
    # the percolating network" (which includes plenty of perfectly
    # wide-open space that just happens to be in a sealed pocket).
    blocked_mask = (distance_field >= 0) & (distance_field < guest_radius)

    return GuestAccessibilityResult(
        guest_name=guest_name,
        guest_diameter=guest_diameter,
        status=status,
        minimum_window_diameter=pld,
        maximum_guest_diameter=pld,
        largest_cavity_diameter=lcd,
        largest_accessible_cavity=accessible_cavity,
        bottlenecks=bottlenecks,
        suggested_probe_radius=suggested_radius,
        grid_shape=grid_shape,
        accessible_points=grid_positions[sweep_result.accessible],
        blocked_points=grid_positions[blocked_mask],
    )


def _site_symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "?"
