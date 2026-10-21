from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

#: PROBE_RADII now lives in data.probe_radii (shared with
#: guest_accessibility.py, which needs the same table plus CH4) and is
#: re-exported here unchanged, since this module's own callers already
#: import it from this path. Listed in __all__ so it isn't flagged as
#: an unused import - it's unused *within* this module by design, but
#: is this module's own public re-export.
from thevoid_core.data.probe_radii import PROBE_RADII
from thevoid_core.services.pore_analysis import (
    build_periodic_atoms,
    compute_distance_field,
    major_occupancy_structure,
    network_assisted_sweep,
    sweep_connectivity,
)

__all__ = [
    "PROBE_RADII",
    "SurfaceAreaResult",
    "analyze_surface_area",
    "sample_atom_surfaces",
    "canonical_rotations",
    "canonical_order",
    "sample_sphere_points",
    "exposed_mask",
]

logger = logging.getLogger(__name__)


AVOGADRO_NUMBER = 6.02214076e23

#: 1 A^2 = 1e-20 m^2; 1 A^3 = 1e-24 cm^3 - derived once, used in the
#: per-mass and per-volume conversions below.
SQUARE_ANGSTROM_TO_SQUARE_METER = 1.0e-20
CUBIC_ANGSTROM_TO_CUBIC_CM = 1.0e-24


def sample_sphere_points(
    n: int, radius: float, center, rng: np.random.Generator
) -> np.ndarray:
    """n uniformly-distributed random points on a sphere of the given
    radius and centre.

    Standard normalised-Gaussian method: a 3D Gaussian is spherically
    symmetric, so normalising a Gaussian vector to unit length gives a
    uniform distribution over the sphere - unlike sampling spherical
    angles uniformly, which would bunch points near the poles.
    """
    vectors = rng.normal(size=(n, 3))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    return np.asarray(center, dtype=float) + vectors * radius


#: The golden angle, in radians - successive points of a Fibonacci
#: (golden-spiral) sphere lattice are this far apart in longitude.
_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))


def fibonacci_sphere_points(
    n: int, radius: float, center, rotation: np.ndarray | None = None
) -> np.ndarray:
    """n nearly evenly spaced points on a sphere: the Fibonacci
    (golden-spiral) lattice - equal-area latitude bands, successive
    points a golden angle apart in longitude - optionally rotated.

    Used instead of independent random points since round 50: estimating
    the exposed fraction from an evenly spread point set converges much
    faster than from random points (roughly n^-3/4 against n^-1/2), so
    the same 500 samples per atom give a several times smaller error,
    and the result no longer changes from run to run with the seed.
    """
    index = np.arange(n, dtype=float) + 0.5
    z = 1.0 - 2.0 * index / n
    ring = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    phi = _GOLDEN_ANGLE * index

    unit = np.column_stack([ring * np.cos(phi), ring * np.sin(phi), z])

    if rotation is not None:
        unit = unit @ rotation.T

    return np.asarray(center, dtype=float) + unit * radius


def lattice_frame(lattice) -> np.ndarray | None:
    """The rotation from the standard orientation of this cell (the one
    pymatgen builds from a, b, c, alpha, beta, gamma, as every CIF is
    read) to the cell as it lies in space; None when they are the same.

    Round 60: the sampling rotations are drawn in the standard frame and
    turned by this, so a crystal gets the same sample points whatever
    the orientation of its axes. Before, the same Cr-BDC read from its
    CIF (standard orientation) and through ASE (a along x) gave channel
    ASA 1505 and 1450 m2/g with the same seed - the same number within
    sampling scatter, but not the same number. A structure in the
    standard orientation (every CIF) samples exactly as before."""
    from pymatgen.core import Lattice

    matrix = np.asarray(lattice.matrix, dtype=float)
    standard = Lattice.from_parameters(*lattice.abc, *lattice.angles).matrix
    transform = np.linalg.solve(standard, matrix)   # x_actual = x_standard @ transform
    if np.allclose(transform, np.eye(3), atol=1e-9):
        return None
    u, _, vt = np.linalg.svd(transform)  # nearest orthogonal matrix
    return u @ vt


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random 3x3 rotation matrix (from a random unit
    quaternion). Each atom's Fibonacci lattice gets its own, so that
    neighbouring atoms' lattices are never aligned with each other or
    with the crystal axes - which could otherwise bias the exposed
    fraction systematically in a highly symmetric structure.
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q

    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


#: Fractional coordinates are rounded to this before atoms are put in
#: canonical order (canonical_rotations, round 63).
CANONICAL_ROUNDING = 1e-4


def canonical_order(frac_coords, labels) -> np.ndarray:
    """The atoms' indices sorted by element, then by fractional
    coordinates wrapped into [0, 1) and rounded to 1e-4: an order that
    belongs to the crystal as placed in its cell, not to the file."""
    frac = np.mod(np.round(np.asarray(frac_coords, dtype=float) / CANONICAL_ROUNDING)
                  * CANONICAL_ROUNDING, 1.0)
    frac = np.where(np.isclose(frac, 1.0), 0.0, frac)
    keys = [(str(labels[i]), *np.round(frac[i], 6)) for i in range(len(labels))]
    return np.array(sorted(range(len(labels)), key=lambda i: keys[i]), dtype=int)


def canonical_rotations(frac_coords, labels, seed: int = 0) -> np.ndarray:
    """One sample rotation per atom, drawn from the seeded stream in
    canonical order (canonical_order) instead of file order.

    Round 63. Before, the rotations were drawn in file order, so the
    same crystal with its atoms listed in another order gave a
    different channel ASA: 0.08-3.5% on 12 CoRE structures at 500
    samples per atom. A shared index needs two analyses of one
    determination to agree, so the rotations now follow the atoms, not
    the file. Every atom still gets its own independent rotation, as
    before, so sampling scatter is unchanged. (A rotation derived from
    each atom's environment was tried first: it also made supercells
    exact, but gave every translation-equivalent atom the same samples,
    which raised the seed-to-seed scatter of a small centred cell from
    1.6% to 4.3%.)
    """
    n_atoms = len(labels)
    rotations = np.empty((n_atoms, 3, 3))
    rng = np.random.default_rng(seed)
    for i in canonical_order(frac_coords, labels):
        rotations[i] = random_rotation(rng)
    return rotations


def exposed_mask(
    sample_points: np.ndarray,
    tree,
    neighbor_radii: np.ndarray,
    max_neighbor_radius: float,
) -> np.ndarray:
    """For a batch of sampled surface points (from any atom or atoms),
    which of them are exposed - not inside any other atom's own
    (already probe-inflated, if applicable) sphere.

    tree is a cKDTree already built over every relevant atom instance
    (periodic images included); neighbor_radii is index-aligned with
    tree's own points and already has any probe radius added in.

    A small epsilon tolerance (rather than a strict "distance <
    radius" check) absorbs floating-point rounding at exactly the
    sampled atom's own surface, which by construction sits at a
    distance from its own centre exactly equal to its own radius - the
    boundary case a point on an atom's own surface should never be
    misclassified as "inside itself" due to rounding, without needing
    to explicitly track and exclude that one specific atom instance
    from every query.
    """
    epsilon = 1e-6

    n_points = len(sample_points)
    exposed = np.ones(n_points, dtype=bool)

    if n_points == 0 or tree.n == 0:
        return exposed

    # Vectorised (round 50): one batched k-nearest query instead of a
    # Python loop over every sample point. A point is buried if any
    # neighbour within max_neighbor_radius contains it; only a point
    # with k neighbours all inside that range could have a further,
    # unexamined one, and those few are re-checked exhaustively.
    k = int(min(64, tree.n))
    distances, indices = tree.query(
        sample_points, k=k, distance_upper_bound=max_neighbor_radius
    )

    if k == 1:
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    valid = np.isfinite(distances)
    safe_indices = np.where(valid, indices, 0)
    inside = valid & (distances < neighbor_radii[safe_indices] - epsilon)
    exposed = ~np.any(inside, axis=1)

    saturated = np.nonzero(valid[:, -1] & exposed)[0]

    for point_index in saturated:
        candidates = np.asarray(
            tree.query_ball_point(
                sample_points[point_index], r=max_neighbor_radius
            ),
            dtype=int,
        )
        deltas = sample_points[point_index] - tree.data[candidates]
        d = np.linalg.norm(deltas, axis=1)

        if np.any(d < neighbor_radii[candidates] - epsilon):
            exposed[point_index] = False

    return exposed


def sample_atom_surfaces(
    base_cartesian: np.ndarray,
    base_vdw_radii: np.ndarray,
    periodic_positions: np.ndarray,
    periodic_vdw_radii: np.ndarray,
    probe_radius: float,
    n_samples_per_atom: int,
    rng: np.random.Generator | None = None,
    frame: np.ndarray | None = None,
    rotations: np.ndarray | None = None,
):
    """One sampling pass - geometric if probe_radius is 0,
    accessible otherwise - over every atom in the base cell.

    Each atom's sphere is sampled with a randomly rotated Fibonacci
    lattice (fibonacci_sphere_points / random_rotation) rather than
    independent random points - see fibonacci_sphere_points for why.
    rotations (one per atom; analyze_surface_area passes
    canonical_rotations) or, without them, rng chooses the rotations. frame (see lattice_frame) turns
    them into the cell's own orientation, so a crystal gives the same
    points however its axes are laid out in space.

    base_cartesian/base_vdw_radii describe only the base cell's own
    atoms (what gets sampled - each contributes its own share of the
    unit cell's total surface exactly once); periodic_positions/
    periodic_vdw_radii describe every atom instance, base cell and
    images alike (what sampled points get checked against, since a
    point near a cell face can be blocked by a neighbour's periodic
    image). Both radii arrays are plain van der Waals radii - the probe
    is added once, inside this function, to whichever role each array
    plays (effective sampling radius for the atom being sampled;
    effective exclusion radius for every neighbour being checked
    against).

    Returns (total_area, per_atom_areas, sampled_points, exposed).
    """
    from scipy.spatial import cKDTree

    n_atoms = len(base_cartesian)

    effective_radii = base_vdw_radii + probe_radius
    neighbor_radii = periodic_vdw_radii + probe_radius

    max_neighbor_radius = (
        float(np.max(neighbor_radii)) if len(neighbor_radii) else 0.0
    )

    tree = cKDTree(periodic_positions)

    total_area = 0.0
    per_atom_areas = np.zeros(n_atoms)
    all_points = []
    all_exposed = []

    for atom_index in range(n_atoms):
        radius = effective_radii[atom_index]
        center = base_cartesian[atom_index]

        rotation = rotations[atom_index] if rotations is not None else random_rotation(rng)
        if frame is not None:
            rotation = frame.T @ rotation
        points = fibonacci_sphere_points(
            n_samples_per_atom, radius, center, rotation
        )

        exposed = exposed_mask(
            points, tree, neighbor_radii, max_neighbor_radius
        )

        exposed_fraction = float(np.mean(exposed))
        area = exposed_fraction * 4.0 * np.pi * radius * radius

        per_atom_areas[atom_index] = area
        total_area += area

        all_points.append(points)
        all_exposed.append(exposed)

    if all_points:
        sampled_points = np.concatenate(all_points, axis=0)
        exposed_all = np.concatenate(all_exposed, axis=0)
    else:
        sampled_points = np.zeros((0, 3))
        exposed_all = np.zeros(0, dtype=bool)

    return total_area, per_atom_areas, sampled_points, exposed_all


@dataclass(frozen=True, slots=True)
class SurfaceAreaResult:
    """Everything the Accessible Surface Area panel needs to display,
    plus enough to render the sampled surface in the 3D view.

    accessible_surface_area and geometric_surface_area are both totals
    for the unit cell (angstrom^2): accessible_surface_area is the
    solvent-accessible-style area computed with the probe inflating
    every atom's radius (the standard Duren/Zeo++-style definition);
    geometric_surface_area is the same computation with the probe
    radius forced to zero - the true molecular (van der Waals) surface.

    total_accessible_area is simply accessible_surface_area again,
    named to match the requested output directly rather than making a
    caller reach for a differently-named field for the same number.
    """

    probe_name: str
    probe_radius: float
    n_samples_per_atom: int

    accessible_surface_area: float
    geometric_surface_area: float

    asa_per_mass: float
    asa_per_volume: float

    total_accessible_area: float

    #: The portion of total_accessible_area that borders the connected,
    #: percolating pore network - the same percolation classification
    #: pore_analysis.sweep_connectivity computes for pore volume -
    #: rather than a sealed, non-percolating pocket. For a bulk
    #: periodic calculation there is no outer "external" particle
    #: surface to subtract, which is why this is reported as the
    #: percolating-network split rather than an external/internal
    #: split in the more usual finite-particle sense: documented
    #: explicitly here since the term could otherwise be read either
    #: way, and this is a deliberate, disclosed choice rather than an
    #: assumed one.
    internal_accessible_area: float

    cell_volume: float
    cell_mass_grams: float

    #: Every sampled point from the accessible-surface pass (both
    #: accessible and blocked), for rendering. Row-aligned with
    #: accessible_mask. The geometric pass's own points are not kept -
    #: it exists to produce one comparison number, not something to
    #: visualise separately.
    sampled_points: np.ndarray
    accessible_mask: np.ndarray

    #: Round 50. The rest of total_accessible_area: surface a probe
    #: could touch only from inside a sealed pocket (Zeo++'s NASA).
    pocket_accessible_area: float = 0.0

    #: internal_accessible_area per gram - the value Zeo++ reports as
    #: "ASA_m^2/g", which counts channel surface only. asa_per_mass
    #: above counts pockets too.
    channel_asa_per_mass: float = 0.0

    #: Row-aligned with sampled_points: True for exposed points that
    #: border the channel network (a subset of accessible_mask).
    channel_mask: np.ndarray | None = None


#: Grid used only to decide which surface points border the
#: percolating channel network and which border sealed pockets - the
#: same spacing and cap as pore analysis's own default grid.
_PERCOLATION_GRID_SPACING = 0.3
_PERCOLATION_MAX_GRID_POINTS_PER_AXIS = 60


def classify_channel_points(
    structure,
    vdw_radii: np.ndarray,
    probe_radius: float,
    points: np.ndarray,
    normals: np.ndarray,
    *,
    grid_spacing: float = _PERCOLATION_GRID_SPACING,
    max_grid_points_per_axis: int = _PERCOLATION_MAX_GRID_POINTS_PER_AXIS,
) -> np.ndarray:
    """For each exposed surface point, True if it borders the
    percolating (channel) part of the probe-accessible space, False if
    it borders a sealed pocket - Zeo++'s ASA / NASA split.

    Every exposed point is, by construction, a position the probe's
    *centre* can occupy, sitting exactly on the edge of the region
    open to it. It is stepped one grid spacing further out along its
    own atom's surface normal, into that open region, and then takes
    the channel/pocket status of the nearest grid node that is itself
    open to the probe, out of the 27 nodes around it (125 if none of
    those is open).

    Replaces the pre-round-50 lookup, which took whichever node of a
    0.6 A corner-aligned grid sat at the lower corner of the grid cell
    the point fell in - up to a full cell diagonal (1 A) away and, as
    often as not, inside an atom and therefore "not accessible". On
    CALF-20 with a 1 A probe that put 132.6 of 186.6 A^2 in channels and
    the rest in "pockets"; Zeo++ finds one channel, no pockets, and
    190.7 A^2 - all of it channel.
    """
    n_points = len(points)
    result = np.zeros(n_points, dtype=bool)

    if n_points == 0:
        return result

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)

    shape = np.array(
        [
            int(min(max_grid_points_per_axis, max(4, round(length / grid_spacing))))
            for length in lengths
        ]
    )

    periodic_positions, periodic_radii = build_periodic_atoms(
        structure, vdw_radii, float(np.max(vdw_radii)) + probe_radius + 2.0
    )

    nx, ny, nz = shape
    grid_frac = np.stack(
        np.meshgrid(
            (np.arange(nx) + 0.5) / nx,
            (np.arange(ny) + 0.5) / ny,
            (np.arange(nz) + 0.5) / nz,
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 3)

    distance_field = compute_distance_field(
        grid_frac @ lattice, periodic_positions, periodic_radii
    )
    sweep = sweep_connectivity(distance_field, tuple(shape), probe_radius)
    # Round 55: with the Voronoi network's help, so a window slightly
    # wider than the probe - but narrower than the grid can resolve -
    # still counts as a channel (UiO-66 and N2 were all "pocket").
    sweep, _method, _prepared = network_assisted_sweep(
        lattice,
        [site.frac_coords for site in structure.sites],
        vdw_radii,
        grid_frac=grid_frac,
        distance_field=distance_field,
        grid_shape=tuple(shape),
        probe_radius=probe_radius,
        grid_sweep=sweep,
    )

    free = distance_field >= probe_radius
    channel = sweep.accessible

    step = float(np.min(lengths / shape))
    nudged = points + normals * step

    # Continuous grid coordinates (node i sits at i + 0.5).
    g = nudged @ np.linalg.inv(lattice) * shape - 0.5
    nearest = np.rint(g).astype(int)

    def resolve(reach, todo):
        span = np.arange(-reach, reach + 1)
        offsets = np.array(
            [(a, b, c) for a in span for b in span for c in span]
        )

        cand = nearest[todo][:, None, :] + offsets[None, :, :]
        wrapped = np.mod(cand, shape)
        flat = (wrapped[..., 0] * ny + wrapped[..., 1]) * nz + wrapped[..., 2]

        node_cart = ((cand + 0.5) / shape) @ lattice
        dist = np.linalg.norm(node_cart - nudged[todo][:, None, :], axis=2)
        dist = np.where(free[flat], dist, np.inf)

        best = np.argmin(dist, axis=1)
        found = np.isfinite(dist[np.arange(len(todo)), best])
        chosen = flat[np.arange(len(todo)), best]

        return found, channel[chosen]

    todo = np.arange(n_points)
    found, status = resolve(1, todo)
    result[todo[found]] = status[found]

    remaining = todo[~found]

    if len(remaining):
        found, status = resolve(2, remaining)
        result[remaining[found]] = status[found]
        unresolved = int(np.sum(~found))

        if unresolved:
            # Nothing open to the probe within two grid spacings: the
            # point sits in a crevice narrower than the grid resolves.
            # Such a crevice cannot hold a probe that also reaches the
            # channels, so it is counted with the pockets.
            logger.debug(
                "%d surface point(s) had no open grid node nearby; "
                "counted as pocket surface.",
                unresolved,
            )

    return result


def _grid_cartesian_points(grid_shape, lattice) -> np.ndarray:
    nx, ny, nz = grid_shape

    fx, fy, fz = np.meshgrid(
        np.arange(nx) / nx,
        np.arange(ny) / ny,
        np.arange(nz) / nz,
        indexing="ij",
    )
    grid_frac = np.stack([fx.ravel(), fy.ravel(), fz.ravel()], axis=1)

    return grid_frac @ lattice


def analyze_surface_area(
    structure,
    *,
    probe_radius: float,
    probe_name: str,
    radius_provider,
    n_samples_per_atom: int = 1000,
    seed: int = 0,
) -> SurfaceAreaResult:
    """Run the full accessible-surface-area pipeline on a structure:
    one Monte Carlo sampling pass with the probe radius applied (the
    accessible surface), and one with it forced to zero (the geometric,
    true-molecular-surface comparison number).

    n_samples_per_atom is a per-atom Monte Carlo sample count, not a
    global grid resolution the way pore analysis's grid_spacing is -
    every atom gets its own fixed sample budget regardless of its size,
    since the statistical noise in an exposed *fraction* depends on the
    sample count, not on the sphere's radius.

    Round 50 changes, all checked against Zeo++ 0.2 with the same radii:
    evenly spread (Fibonacci) rather than random samples, and 1000 of
    them per atom rather than 500 - now affordable, the exposure test
    being vectorised - so CALF-20's ASA at a 1 A probe varies by under
    1% between seeds rather than 6%, and converges to 190.2 A^2
    (Zeo++: 190.7); and a corrected channel/pocket split
    (classify_channel_points), which previously put 30% of CALF-20's
    surface in pockets Zeo++ says do not exist.
    """
    structure = major_occupancy_structure(structure)


    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    frame = lattice_frame(structure.lattice)
    symbols = [_site_symbol(site) for site in structure.sites]

    vdw_radii = np.array(
        [radius_provider.get_vdw_radius(symbol) for symbol in symbols]
    )

    base_cartesian = np.array(
        [site.coords for site in structure.sites], dtype=float
    )

    cell_volume = float(
        abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2])))
    )
    cell_mass_grams = (
        float(structure.composition.weight) / AVOGADRO_NUMBER
        if len(structure.sites)
        else 0.0
    )

    if len(structure.sites) == 0:
        logger.warning(
            "Surface area analysis requested on a structure with no "
            "atoms; reporting zero area."
        )
        return SurfaceAreaResult(
            probe_name=probe_name,
            probe_radius=probe_radius,
            n_samples_per_atom=n_samples_per_atom,
            accessible_surface_area=0.0,
            geometric_surface_area=0.0,
            asa_per_mass=0.0,
            asa_per_volume=0.0,
            total_accessible_area=0.0,
            internal_accessible_area=0.0,
            cell_volume=cell_volume,
            cell_mass_grams=0.0,
            sampled_points=np.zeros((0, 3)),
            accessible_mask=np.zeros(0, dtype=bool),
        )

    max_vdw = float(np.max(vdw_radii))
    cutoff = max_vdw + probe_radius + 2.0

    periodic_positions, periodic_vdw_radii = build_periodic_atoms(
        structure, vdw_radii, cutoff
    )

    # Round 63: rotations are drawn in the atoms' canonical order, not
    # file order (canonical_rotations). The same rotations serve both
    # passes.
    rotations = canonical_rotations(structure.frac_coords, symbols, seed)

    accessible_area, _, sampled_points, accessible_mask = (
        sample_atom_surfaces(
            base_cartesian,
            vdw_radii,
            periodic_positions,
            periodic_vdw_radii,
            probe_radius,
            n_samples_per_atom,
            frame=frame,
            rotations=rotations,
        )
    )

    geometric_area, _, _, _ = sample_atom_surfaces(
        base_cartesian,
        vdw_radii,
        periodic_positions,
        periodic_vdw_radii,
        0.0,
        n_samples_per_atom,
        frame=frame,
        rotations=rotations,
    )

    # Each sample stands for its own atom's share of that atom's
    # sphere (radii differ, so shares differ) - tracked exactly per
    # point rather than pooled into one average weight.
    atom_of_point = np.repeat(np.arange(len(vdw_radii)), n_samples_per_atom)
    effective = vdw_radii + probe_radius
    point_area = (4.0 * np.pi * effective**2 / n_samples_per_atom)[atom_of_point]

    normals = sampled_points - base_cartesian[atom_of_point]
    normals /= np.maximum(
        np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
    )

    exposed_index = np.nonzero(accessible_mask)[0]
    is_channel = classify_channel_points(
        structure,
        vdw_radii,
        probe_radius,
        sampled_points[exposed_index],
        normals[exposed_index],
    )
    channel_mask = np.zeros(len(sampled_points), dtype=bool)
    channel_mask[exposed_index[is_channel]] = True

    internal_area = float(np.sum(point_area[channel_mask]))
    pocket_area = max(0.0, accessible_area - internal_area)

    asa_square_meters = accessible_area * SQUARE_ANGSTROM_TO_SQUARE_METER

    asa_per_mass = (
        asa_square_meters / cell_mass_grams if cell_mass_grams > 0 else 0.0
    )
    asa_per_volume = (
        asa_square_meters
        / (cell_volume * CUBIC_ANGSTROM_TO_CUBIC_CM)
        if cell_volume > 0
        else 0.0
    )

    channel_square_meters = internal_area * SQUARE_ANGSTROM_TO_SQUARE_METER
    channel_asa_per_mass = (
        channel_square_meters / cell_mass_grams if cell_mass_grams > 0 else 0.0
    )

    return SurfaceAreaResult(
        probe_name=probe_name,
        probe_radius=probe_radius,
        n_samples_per_atom=n_samples_per_atom,
        accessible_surface_area=accessible_area,
        geometric_surface_area=geometric_area,
        asa_per_mass=asa_per_mass,
        asa_per_volume=asa_per_volume,
        total_accessible_area=accessible_area,
        internal_accessible_area=internal_area,
        cell_volume=cell_volume,
        cell_mass_grams=cell_mass_grams,
        sampled_points=sampled_points,
        accessible_mask=accessible_mask,
        pocket_accessible_area=pocket_area,
        channel_asa_per_mass=channel_asa_per_mass,
        channel_mask=channel_mask,
    )


def _site_symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "?"
