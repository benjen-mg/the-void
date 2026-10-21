from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Periodic percolation detection
# ----------------------------------------------------------------------
#
# The scientific crux of this module: a grid point can satisfy "there is
# room here for a probe of this radius" and still not be part of any
# real pore a guest molecule could use, if that room is a sealed pocket
# with no way out. Telling those apart - and telling a channel that
# runs in one direction from a sheet that runs in two, or a fully 3D
# interconnected network - requires knowing not just "are these two
# points connected" but "does tracing a path between them ever require
# crossing a cell boundary, and if you come back to the same point by a
# different route, do you end up having drifted by a whole lattice
# vector". That second question is exactly what a plain
# connected-components algorithm cannot answer, and exactly what a
# union-find that also tracks each node's accumulated lattice offset
# relative to its set's root can.


@dataclass
class _OffsetUnionFind:
    """Union-find where every node also carries an integer lattice
    offset relative to its set's root.

    Standard weighted-union-with-path-compression, extended to
    propagate an (i, j, k) integer offset alongside the usual parent
    pointer: offset[node] is the lattice translation that must be added
    to node's own grid position to land on its root's grid position
    (mod the grid's periodicity - the offsets accumulate in whole
    lattice vectors, they are not wrapped back into a single cell).

    When union() is asked to join two nodes that turn out to already be
    in the same set, the *expected* relative offset from the edge being
    added is compared against the *actual* relative offset already
    implied by the tree. Agreement means the edge is redundant (it
    closes a cycle that returns to the same physical point with zero
    net translation - unremarkable). Disagreement means the cycle
    returns to the same grid point having drifted by a nonzero lattice
    vector: the network this set belongs to repeats itself across a
    cell boundary, i.e., it percolates - and the specific nonzero
    vector found is a genuine translational symmetry of that network,
    which is exactly what the dimensionality calculation needs.
    """

    parent: list[int] = field(default_factory=list)
    rank: list[int] = field(default_factory=list)
    offset: list[tuple[int, int, int]] = field(default_factory=list)

    #: For each set's root: the list of nonzero net-translation vectors
    #: found by cycles within that set, discovered so far. Populated by
    #: union(); read via percolation_vectors_for_root().
    _cycle_vectors: dict[int, list[tuple[int, int, int]]] = field(
        default_factory=dict
    )

    @classmethod
    def for_n_nodes(cls, n: int) -> "_OffsetUnionFind":
        return cls(
            parent=list(range(n)),
            rank=[0] * n,
            offset=[(0, 0, 0)] * n,
        )

    def find(self, node: int) -> tuple[int, tuple[int, int, int]]:
        """Root of node's set, and node's cumulative offset to it."""
        path = []
        current = node
        total = (0, 0, 0)

        while self.parent[current] != current:
            path.append(current)
            total = _add(total, self.offset[current])
            current = self.parent[current]

        root = current

        # Path compression: point every visited node straight at the
        # root, updating each one's stored offset to be its (now
        # correct, already-computed) direct offset to that root.
        cumulative = (0, 0, 0)
        for visited in reversed(path):
            cumulative = _add(cumulative, self.offset[visited])
            self.offset[visited] = cumulative
            self.parent[visited] = root

        return root, total

    def union(self, a: int, b: int, edge_offset: tuple[int, int, int]) -> None:
        """Record that b is reachable from a via edge_offset - i.e., if
        a is at some grid position, b is at that position plus
        edge_offset (before wrapping into a single cell).
        """
        root_a, offset_a = self.find(a)
        root_b, offset_b = self.find(b)

        # The offset from root_a to root_b implied by this specific edge:
        # a is offset_a from root_a; b = a + edge_offset; b is offset_b
        # from root_b, so root_b = b - offset_b = a + edge_offset - offset_b
        # = root_a + offset_a + edge_offset - offset_b.
        implied = _add(_subtract(offset_a, offset_b), edge_offset)

        if root_a == root_b:
            if implied != (0, 0, 0):
                self._cycle_vectors.setdefault(root_a, []).append(implied)
            return

        # Union by rank; the new root inherits offset (0,0,0) by
        # definition, and the OTHER root gets an offset making the
        # bookkeeping above consistent for anything already pointing at
        # it (path compression will fix descendants lazily on next find).
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
            implied = _negate(implied)

        self.parent[root_b] = root_a
        self.offset[root_b] = implied

        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        # Merge any cycle vectors already found under the absorbed root.
        absorbed = self._cycle_vectors.pop(root_b, [])
        if absorbed:
            self._cycle_vectors.setdefault(root_a, []).extend(absorbed)

    def percolation_vectors_for_root(
        self, root: int
    ) -> list[tuple[int, int, int]]:
        return list(self._cycle_vectors.get(root, []))


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _subtract(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _negate(a):
    return (-a[0], -a[1], -a[2])


def dimensionality_from_vectors(vectors: list[tuple[int, int, int]]) -> int:
    """Rank of the integer lattice spanned by a set of translation
    vectors - 0 if empty (no percolation at all: an isolated pocket),
    1 for a single independent direction (a channel), 2 for a sheet, 3
    for a fully interconnected 3D network.

    Computed as the rank of the vectors over the rationals (equivalent
    to the rank over the integers for this purpose - any nonzero
    integer combination spans the same space as long as we only care
    about dimensionality, not the specific sublattice).
    """
    if not vectors:
        return 0

    matrix = np.array(vectors, dtype=float)

    return int(np.linalg.matrix_rank(matrix))


# ----------------------------------------------------------------------
# Grid graph and the combined PLD / accessibility sweep
# ----------------------------------------------------------------------


def build_grid_edges(shape: tuple[int, int, int]):
    """6-connected periodic grid graph over a (nx, ny, nz) grid.

    6-connectivity (face-adjacent neighbours only, not diagonals) is
    the standard, simplest choice for a distance-transform pore grid
    and is what is used here; it slightly underestimates true
    connectivity compared to a diagonal-inclusive scheme, in the same
    direction and for the same reason a discretised grid always slightly
    misrepresents smooth geometry - finer grid spacing is the correction
    for that, not more exotic connectivity.

    Returns a flat list of (node_a, node_b, offset) triples, node
    indices being row-major flattened grid positions and offset the
    integer lattice translation (in whole cells) crossed by that edge -
    nonzero only where the edge wraps around the grid's own periodic
    boundary.
    """
    nx, ny, nz = shape

    def index(i, j, k):
        return (i % nx) * ny * nz + (j % ny) * nz + (k % nz)

    edges = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                node = index(i, j, k)

                for axis, size in enumerate((nx, ny, nz)):
                    delta = [0, 0, 0]
                    delta[axis] = 1

                    ni, nj, nk = i + delta[0], j + delta[1], k + delta[2]
                    neighbor = index(ni, nj, nk)

                    offset = [0, 0, 0]
                    if [i, j, k][axis] + 1 >= size:
                        offset[axis] = 1

                    edges.append((node, neighbor, tuple(offset)))

    return edges


@dataclass(frozen=True, slots=True)
class ConnectivitySweepResult:
    """Everything the combined Kruskal-style sweep over the grid graph
    produces in one pass: the geometric bottleneck (independent of any
    chosen probe), and, for a specific probe radius, which grid nodes
    end up in an accessible (percolating) network versus an isolated
    pocket.
    """

    #: Radius (not diameter) of the widest bottleneck along the best
    #: percolating pathway found anywhere in the grid, at the point
    #: percolation first appears when edges are considered from widest
    #: to narrowest. None if nothing in the grid percolates at all.
    pld_radius: float | None

    #: For each grid node (flat index): True if, restricted to edges
    #: whose two endpoints both have local radius >= probe_radius, that
    #: node belongs to a percolating (accessible) network. False for
    #: nodes below probe_radius entirely, and for nodes above it that
    #: are only part of a sealed, non-percolating pocket.
    accessible: np.ndarray

    #: Root-node dimensionality for each of the accessible sets found,
    #: keyed by that set's representative node index at the probe_radius
    #: cutoff - exposed for callers that want per-network detail; most
    #: callers just want the single largest network's dimensionality,
    #: which analyze_pores reports.
    dimensionality_by_root: dict[int, int]

    #: Round 55: the accessibility sweep's union-find itself, so the
    #: Voronoi-assisted classification can extend it rather than redo
    #: it. Not part of the result's meaning.
    access_union_find: object = None


def sweep_connectivity(
    radius_values: np.ndarray,
    shape: tuple[int, int, int],
    probe_radius: float,
) -> ConnectivitySweepResult:
    """The combined PLD + probe-accessibility computation.

    radius_values is a flat array (row-major, matching build_grid_edges'
    indexing) of the local pore radius at every grid node - the
    distance from that point to the nearest atom's van der Waals
    surface, with no probe inflation applied (see compute_distance_field).

    Two separate sweeps over the same edge list, each simple and
    independently correct, rather than one clever combined pass: an
    earlier version tried to find both answers in a single sweep and
    stopped early once edge capacity dropped below probe_radius, which
    is wrong whenever the material's true geometric bottleneck (PLD) is
    *narrower* than whatever probe radius happens to be requested - PLD
    is a property of the material alone, and stopping the sweep at
    probe_radius before that narrower bottleneck was ever reached would
    silently under-report it (or miss it) instead of computing it from
    the full range of edges.

    Sweep 1 - PLD (independent of probe_radius): process every edge, in
    decreasing order of capacity (the smaller of its two endpoints'
    local radii - the largest probe that could actually cross that
    edge), unioning endpoints in a periodic-offset-tracking union-find.
    The first time any union reveals a nonzero net lattice translation,
    that edge's capacity is, by construction, the largest value at
    which the grid has *any* percolating path - the pore-limiting
    radius.

    Sweep 2 - accessibility at probe_radius: a fresh union-find, only
    processing edges whose capacity is at least probe_radius (an edge
    narrower than that cannot be crossed by a probe of this size at
    all). Whichever nodes with local radius >= probe_radius end up in a
    set that accumulated a nonzero translation are accessible to that
    probe; the rest - even if their own local radius is large enough -
    are sealed pockets a probe of this size could never reach without
    passing through a constriction narrower than itself.
    """
    n_nodes = int(np.prod(shape))
    edges = build_grid_edges(shape)

    edge_capacity = [
        min(radius_values[a], radius_values[b]) for a, b, _ in edges
    ]

    order = sorted(
        range(len(edges)), key=lambda i: edge_capacity[i], reverse=True
    )

    # --- Sweep 1: PLD, independent of probe_radius ---

    pld_uf = _OffsetUnionFind.for_n_nodes(n_nodes)
    pld_radius: float | None = None

    for edge_index in order:
        capacity = edge_capacity[edge_index]

        if capacity <= 0:
            # Zero or negative capacity means the two endpoints are
            # touching or inside an atom's own volume, not real void
            # space - the grid's periodic topology alone would
            # otherwise eventually "percolate" through such edges
            # regardless of the actual structure (every grid, however
            # solid, is periodic), reporting a physically meaningless
            # zero-radius channel through solid material. Sorted
            # descending, every remaining edge is <= 0 too, so there is
            # nothing further to find.
            break

        a, b, offset = edges[edge_index]

        pld_uf.union(a, b, offset)

        if pld_radius is None:
            root_a, _ = pld_uf.find(a)
            if pld_uf.percolation_vectors_for_root(root_a):
                pld_radius = capacity
                break

    # --- Sweep 2: accessibility at the requested probe_radius ---

    access_uf = _OffsetUnionFind.for_n_nodes(n_nodes)

    for edge_index in order:
        capacity = edge_capacity[edge_index]

        if capacity < probe_radius or capacity <= 0:
            # Sorted descending: every remaining edge is narrower still,
            # so nothing past this point can be crossed by this probe.
            break

        a, b, offset = edges[edge_index]
        access_uf.union(a, b, offset)

    accessible = np.zeros(n_nodes, dtype=bool)
    root_of_node = np.full(n_nodes, -1, dtype=np.int64)

    for node in range(n_nodes):
        if radius_values[node] < probe_radius:
            continue

        root, _ = access_uf.find(node)

        if access_uf.percolation_vectors_for_root(root):
            accessible[node] = True
            root_of_node[node] = root

    # Dimensionality involves an SVD-based rank computation - cheap
    # once, but only once per distinct network, not once per node in
    # it. Measured directly: computing it per-node instead of per-root
    # took roughly 11 seconds on a grid with one large accessible
    # network of ~26,000 nodes, doing the same redundant computation
    # that many times over; per-root, the same case is a handful of
    # calls.
    unique_roots = set(root_of_node[accessible].tolist())
    dimensionality_by_root = {
        root: dimensionality_from_vectors(
            access_uf.percolation_vectors_for_root(root)
        )
        for root in unique_roots
    }

    return ConnectivitySweepResult(
        pld_radius=pld_radius,
        accessible=accessible,
        dimensionality_by_root=dimensionality_by_root,
        access_union_find=access_uf,
    )


# ----------------------------------------------------------------------
# Geometry: atoms -> distance field
# ----------------------------------------------------------------------

#: Below this occupancy, a site is treated as a minor disorder
#: component and excluded from void-space analysis - see
#: major_occupancy_structure's docstring for why this matters. The
#: same 0.5 threshold the 3D viewer's own "Show Minor Disorder
#: Components" toggle already uses (MainWindow.min_occupancy), kept
#: consistent rather than inventing a second convention.
DEFAULT_MIN_OCCUPANCY = 0.5


def major_occupancy_structure(structure, min_occupancy: float = DEFAULT_MIN_OCCUPANCY):
    """Drop minor disorder components before any void-space analysis.

    A disordered site split across several partial-occupancy
    alternatives (e.g. four positions at 0.25 occupancy each) is a
    crystallographic ensemble average: in any single, physically real
    unit cell, only *one* of those alternatives is actually present,
    not all of them simultaneously. Distance-field-based analysis
    (pore volume, accessible surface area, guest accessibility) has no
    concept of "ensemble average" - every atom passed in is treated as
    permanently, simultaneously present and blocking space with its
    full van der Waals radius. Left unfiltered, a structure with
    disorder gets every alternative position counted as solid at once,
    which can plausibly close off pores and channels that are actually
    open in the real, physical structure.

    This is a real, confirmed effect, not a theoretical concern: a
    real deposited MOF structure (CALF-20, CCDC 2084733) has 12 sites
    at 0.25 occupancy: including them all as permanently-present atoms
    dropped its computed Largest Cavity Diameter to 2.49 A and Pore
    Limiting Diameter to 0.96 A - smaller than nearly any real guest
    molecule, and CALF-20 is specifically well known for practical CO2
    capture. Excluding the minor components (occupancy below 0.5, the
    same threshold the 3D viewer's own disorder toggle already uses)
    raised those to 4.05 A and 2.38 A - a completely different,
    far more plausible answer for a real, functional gas-sorption
    material. (Those are the grid values of the time; round 50's
    grid-independent LCD/PLD - see pore_geometry.py - puts the filtered
    structure at 4.28 A / 2.71 A, in line with Zeo++; round 56's path-
    checked PLD refinement at 4.28 A / 2.80 A.)

    Returns a new Structure (the original is left untouched) with
    every site below min_occupancy removed, keeping the atoms that
    remain at their original indices' relative order.
    """
    keep_indices = [
        index
        for index, site in enumerate(structure.sites)
        if site.species.num_atoms >= min_occupancy
    ]

    if len(keep_indices) == len(structure.sites):
        return structure

    filtered = structure.copy()
    remove_indices = [
        index for index in range(len(structure.sites))
        if index not in keep_indices
    ]
    filtered.remove_sites(remove_indices)

    return filtered


def perpendicular_widths(lattice_matrix) -> np.ndarray:
    """The cell's thickness along each lattice vector, measured
    perpendicular to the opposite face - volume divided by that face's
    area. Standard crystallographic quantity, used here to work out how
    many periodic image cells are needed in each direction to guarantee
    that no atom within a given cutoff distance of the base cell is
    missed, however skewed the cell's shape.
    """
    a, b, c = np.asarray(lattice_matrix, dtype=float)
    volume = abs(np.dot(a, np.cross(b, c)))

    return np.array(
        [
            volume / np.linalg.norm(np.cross(b, c)),
            volume / np.linalg.norm(np.cross(c, a)),
            volume / np.linalg.norm(np.cross(a, b)),
        ]
    )


def build_periodic_atoms(structure, atom_radii: np.ndarray, cutoff: float):
    """Every atom in the base cell, plus enough periodic images in each
    direction that no atom within cutoff of the base cell's volume is
    missing from the list.

    atom_radii is index-aligned with structure.sites. Returns
    (positions, radii) as flat (N, 3) and (N,) arrays covering the base
    cell and all included images together.
    """
    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    widths = perpendicular_widths(lattice)

    image_range = [
        max(1, int(np.ceil(cutoff / width))) if width > 0 else 1
        for width in widths
    ]

    frac_coords = np.array(
        [site.frac_coords for site in structure.sites], dtype=float
    )

    offsets = [
        (i, j, k)
        for i in range(-image_range[0], image_range[0] + 1)
        for j in range(-image_range[1], image_range[1] + 1)
        for k in range(-image_range[2], image_range[2] + 1)
    ]
    offset_array = np.array(offsets, dtype=float)

    # Every (site, image) combination at once: shape
    # (n_offsets, n_sites, 3) fractional, then flattened and converted.
    shifted_frac = (
        frac_coords[np.newaxis, :, :] + offset_array[:, np.newaxis, :]
    )
    positions = shifted_frac.reshape(-1, 3) @ lattice
    radii = np.tile(atom_radii, len(offsets))

    return positions, radii


def compute_distance_field(
    grid_points: np.ndarray,
    atom_positions: np.ndarray,
    atom_radii: np.ndarray,
) -> np.ndarray:
    """For every grid point, the distance to the nearest atom's van der
    Waals surface - positive outside every atom, negative if the point
    falls inside one.

    The nearest atom *centre* is not necessarily the nearest atom
    *surface* once radii differ, so this considers each point's k
    nearest atom centres (not just the single nearest) and takes the
    minimum surface distance among those - fully vectorised via one
    batched cKDTree query rather than a per-point Python loop, which
    measured at roughly 11 seconds for a modest 27,000-point grid and
    would only get worse for the finer grids and larger cells real use
    needs; this is milliseconds to a couple of seconds for the same
    grid, depending on cKDTree's own query cost.

    k=32 is a generous margin, not a tight one: verified directly
    against a deliberately constructed case where the nearest atom by
    raw centre distance is *not* the one with the nearest surface (a
    small nearby atom versus a much larger, farther one whose surface
    actually reaches closer) with only 2 atoms total, and in ordinary
    use k neighbours out of what is usually hundreds to thousands of
    candidate atom instances (base cell plus periodic images) is very
    unlikely to exclude the true nearest surface for any realistic
    element and geometry - van der Waals radii only range over about
    2 A across the whole periodic table, so an atom would need to be
    both farther than all 32 closer candidates *and* have a radius
    large enough to still win despite that gap, which does not happen
    for physically reasonable packing densities. Documented as a
    deliberate, bounded approximation rather than a hidden one.
    """
    if len(atom_positions) == 0:
        return np.full(len(grid_points), np.inf)

    from scipy.spatial import cKDTree

    tree = cKDTree(atom_positions)

    k = min(32, len(atom_positions))

    distances, indices = tree.query(grid_points, k=k)

    if k == 1:
        # scipy squeezes the last axis away when k=1; put it back so
        # the rest of this function can assume a consistent 2D shape.
        distances = distances[:, np.newaxis]
        indices = indices[:, np.newaxis]

    surface_distances = distances - atom_radii[indices]

    return np.min(surface_distances, axis=1)


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoreAnalysisResult:
    """Everything the Pore Analysis panel needs to display, plus enough
    to draw the accessible pore network in the 3D view.

    LCD and PLD are diameters (2x the corresponding radius), in
    angstrom, matching how these are conventionally reported in the
    literature. LCD is a pure geometric property of the framework
    itself, independent of any probe. PLD is also independent of the
    probe used for the accessible/non-accessible split below - it
    answers "what is the largest sphere that could ever percolate
    through this material at all" - and can come out smaller, equal to,
    or (if nothing percolates) undefined regardless of what probe_radius
    was requested for the accessibility calculation.

    void_fraction_percent is the *total* geometric free-volume fraction
    for the given probe_radius - every grid point with enough local room
    for that probe, whether or not it is part of a connected,
    percolating network. accessible_volume and non_accessible_volume
    (both absolute, in cubic angstrom) split that same total by whether
    each point's region actually percolates: accessible_volume +
    non_accessible_volume == void_fraction_percent / 100 * cell_volume,
    by construction.
    """

    probe_radius: float
    grid_shape: tuple[int, int, int]

    lcd: float
    pld: float | None

    void_fraction_percent: float
    accessible_volume: float
    non_accessible_volume: float
    cell_volume: float

    #: 0 if no accessible network exists at all; otherwise the highest
    #: dimensionality among the accessible networks found (a framework
    #: can have more than one distinct channel system).
    accessible_dimensionality: int

    #: Cartesian positions of every grid point classified as part of an
    #: accessible (percolating) network, for the 3D viewer to render.
    accessible_points: np.ndarray

    #: How LCD and PLD were obtained: "voronoi" (round 50: exact
    #: continuous LCD, Voronoi-network PLD - see pore_geometry.py) or
    #: "grid" (read off grid nodes, as before round 50; used as the
    #: fallback, or when refine_geometry=False).
    geometry_method: str = "grid"

    #: Round 55: how accessible volume, points and dimensionality were
    #: found - "voronoi+grid" (the Voronoi network supplies connections
    #: through windows the grid cannot resolve) or "grid".
    accessibility_method: str = "grid"


@dataclass(frozen=True, slots=True)
class VoidGrid:
    """The grid-level classification analyze_pores is built on, kept
    for callers that need more than the summary numbers - round 52's
    RASPA export uses it to find the sealed pockets it must block.

    Arrays are flat, index-aligned with each other, in the row-major
    (x slowest) order build_grid_edges uses.
    """

    grid_shape: tuple[int, int, int]
    #: Fractional and Cartesian coordinates of every node (cell-centred).
    grid_frac: np.ndarray
    grid_points: np.ndarray
    #: Distance from each node to the nearest van der Waals surface.
    distance_field: np.ndarray
    sweep: "ConnectivitySweepResult"
    probe_radius: float
    #: Round 55: "voronoi+grid" when the Voronoi network supplied the
    #: connections through narrow windows (pore_geometry.
    #: network_accessibility), "grid" when it could not be built.
    accessibility_method: str = "grid"
    #: (network, clearance) for reuse by the LCD/PLD refinement.
    prepared_network: tuple | None = None

    @property
    def free_mask(self) -> np.ndarray:
        """Nodes with room for the probe centre, reachable or not."""
        return self.distance_field >= self.probe_radius

    @property
    def pocket_mask(self) -> np.ndarray:
        """Nodes with room for the probe centre that the probe cannot
        reach from any percolating network: sealed pockets."""
        return self.free_mask & ~self.sweep.accessible


def classify_void_grid(
    structure,
    atom_radii: np.ndarray,
    *,
    probe_radius: float,
    grid_shape: tuple[int, int, int],
    use_network: bool = True,
) -> VoidGrid:
    """Distance field and accessible/pocket split on the cell-centred
    grid - the shared first half of analyze_pores (moved here unchanged
    in round 52 so the RASPA export's pocket blocking classifies void
    space with exactly the same code as the Pore Analysis panel).

    structure must already be filtered for disorder, and atom_radii
    index-aligned with its sites.
    """
    lattice = np.asarray(structure.lattice.matrix, dtype=float)

    max_radius = float(atom_radii.max())
    cutoff = max_radius + probe_radius + 2.0

    atom_positions, expanded_radii = build_periodic_atoms(
        structure, atom_radii, cutoff
    )

    nx, ny, nz = grid_shape
    frac_x = (np.arange(nx) + 0.5) / nx
    frac_y = (np.arange(ny) + 0.5) / ny
    frac_z = (np.arange(nz) + 0.5) / nz

    grid_frac = np.stack(
        np.meshgrid(frac_x, frac_y, frac_z, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    grid_points = grid_frac @ lattice

    distance_field = compute_distance_field(
        grid_points, atom_positions, expanded_radii
    )

    sweep = sweep_connectivity(distance_field, grid_shape, probe_radius)
    method, prepared = "grid", None

    if use_network:
        sweep, method, prepared = network_assisted_sweep(
            lattice,
            [site.frac_coords for site in structure.sites],
            atom_radii,
            grid_frac=grid_frac,
            distance_field=distance_field,
            grid_shape=grid_shape,
            probe_radius=probe_radius,
            grid_sweep=sweep,
        )

    return VoidGrid(
        grid_shape=tuple(grid_shape),
        grid_frac=grid_frac,
        grid_points=grid_points,
        distance_field=distance_field,
        sweep=sweep,
        probe_radius=probe_radius,
        accessibility_method=method,
        prepared_network=prepared,
    )


#: Round 57 tried the split-sphere network for accessibility only when
#: the grid's own best path came within this (radius, A) of the probe.
#: Since round 58 the PLD always uses the split network when radii
#: differ, and accessibility does too whenever nothing percolates - the
#: same (cached) network, so the two cannot disagree. Kept for reference.
SPLIT_ACCESS_WINDOW = 0.35


def network_assisted_sweep(
    lattice, frac_coords, atom_radii, *, grid_frac, distance_field,
    grid_shape, probe_radius, grid_sweep, prepared=None,
):
    """(sweep, method, prepared): grid_sweep with its accessible nodes
    and dimensionalities recomputed with the Voronoi network's help
    (pore_geometry.network_accessibility). The grid PLD is kept as it
    was. Falls back to the plain grid sweep - method "grid" - if the
    network cannot be built or the combined percolation fails."""
    from thevoid_core.services import pore_geometry

    try:
        if prepared is None:
            prepared = pore_geometry.prepare_network(
                lattice, frac_coords, atom_radii, distance_field
            )
        network, clearance = prepared
        if network is None or not len(network.vertex_positions):
            return grid_sweep, "grid", prepared
        result = pore_geometry.network_accessibility(
            lattice,
            grid_frac=grid_frac,
            distance_field=distance_field,
            grid_shape=grid_shape,
            probe_radius=probe_radius,
            network=network,
            clearance=clearance,
            grid_union_find=grid_sweep.access_union_find,
        )
    except Exception:
        logger.warning(
            "Voronoi-assisted accessibility failed; keeping the grid's.",
            exc_info=True,
        )
        return grid_sweep, "grid", prepared

    method = "voronoi+grid"
    grid_radius = grid_sweep.pld_radius
    if (not result.dimensionality_by_root
            and np.ptp(np.asarray(atom_radii, dtype=float)) > 0.02):
        # Round 57/58: nothing percolates through the radical network,
        # which can miss a channel the split-sphere network finds
        # (pore_geometry.split_network). The PLD uses the same network,
        # built once and kept on this one, so the two cannot disagree
        # about whether this probe passes.
        try:
            split = pore_geometry.cached_split_network(
                network, lattice, frac_coords, atom_radii, clearance,
                pore_geometry.split_reach(pore_geometry.network_reach(atom_radii, distance_field)),
                exact_above=min(probe_radius, grid_radius if grid_radius is not None
                                else probe_radius) - pore_geometry.SPLIT_EXACT_MARGIN,
            )
            split_result = pore_geometry.network_accessibility(
                lattice, grid_frac=grid_frac, distance_field=distance_field,
                grid_shape=grid_shape, probe_radius=probe_radius, network=split,
                clearance=clearance, grid_union_find=None,
            )
            if split_result.dimensionality_by_root:
                result, method = split_result, "split+grid"
        except Exception:
            logger.warning("Split-sphere accessibility failed; keeping the "
                           "Voronoi network's.", exc_info=True)

    return (
        ConnectivitySweepResult(
            pld_radius=grid_sweep.pld_radius,
            accessible=result.accessible,
            dimensionality_by_root=result.dimensionality_by_root,
        ),
        method,
        prepared,
    )


def grid_shape_for(
    structure,
    grid_spacing: float = 0.3,
    max_grid_points_per_axis: int = 60,
) -> tuple[int, int, int]:
    """Per-axis node counts for a target spacing, capped - the rule
    analyze_pores has always used."""
    lengths = np.linalg.norm(
        np.asarray(structure.lattice.matrix, dtype=float), axis=1
    )

    return tuple(
        int(min(max_grid_points_per_axis, max(4, round(length / grid_spacing))))
        for length in lengths
    )


def analyze_pores(
    structure,
    *,
    probe_radius: float,
    radius_provider,
    grid_spacing: float = 0.3,
    max_grid_points_per_axis: int = 60,
    refine_geometry: bool = True,
) -> PoreAnalysisResult:
    """Run the full pore analysis pipeline on a structure.

    grid_spacing is the target distance between grid points, in
    angstrom; the actual per-axis point count is rounded from the
    cell's own dimensions and capped at max_grid_points_per_axis to
    keep the O(grid_points) distance-field and O(grid_edges log
    grid_edges) connectivity sweep bounded for very large or very
    finely-specified cells - this analysis is run on demand, from a
    background thread, not every frame, but it still has to finish in a
    reasonable time.

    LCD and PLD are then made grid-independent by
    pore_geometry.compute_pore_geometry (refine_geometry=True, the
    default): read straight off grid nodes they were off by up to
    0.5-0.7 A whenever a cavity or window centre fell between nodes.
    The volumes, accessible points, and dimensionality stay grid-based.
    """
    structure = major_occupancy_structure(structure)

    lattice = np.asarray(structure.lattice.matrix, dtype=float)

    grid_shape = grid_shape_for(
        structure, grid_spacing, max_grid_points_per_axis
    )

    symbols = [_site_symbol(site) for site in structure.sites]
    atom_radii = np.array(
        [radius_provider.get_vdw_radius(symbol) for symbol in symbols]
    )

    cell_volume = float(abs(np.dot(lattice[0], np.cross(lattice[1], lattice[2]))))

    n_total = int(np.prod(grid_shape))

    if len(atom_radii) == 0:
        # An empty structure is entirely void - not a real material to
        # report pore statistics for, but handled explicitly rather
        # than let a downstream division or infinite distance value
        # propagate somewhere confusing.
        logger.warning(
            "Pore analysis requested on a structure with no atoms; "
            "reporting an entirely open cell."
        )
        return PoreAnalysisResult(
            probe_radius=probe_radius,
            grid_shape=grid_shape,
            lcd=float("inf"),
            pld=None,
            void_fraction_percent=100.0,
            accessible_volume=cell_volume,
            non_accessible_volume=0.0,
            cell_volume=cell_volume,
            accessible_dimensionality=0,
            accessible_points=np.zeros((0, 3)),
        )

    grid = classify_void_grid(
        structure,
        atom_radii,
        probe_radius=probe_radius,
        grid_shape=grid_shape,
    )
    grid_points = grid.grid_points
    distance_field = grid.distance_field
    sweep_result = grid.sweep

    lcd = 2.0 * float(np.max(distance_field))

    pld = (
        2.0 * sweep_result.pld_radius
        if sweep_result.pld_radius is not None
        else None
    )

    geometry_method = "grid"

    if refine_geometry:
        lcd, pld, geometry_method = refined_lcd_pld(
            structure,
            atom_radii,
            grid_points=grid_points,
            distance_field=distance_field,
            grid_lcd=lcd,
            grid_pld=pld,
            prepared=grid.prepared_network,
            grid_shape=grid_shape,
        )

    free_mask = distance_field >= probe_radius
    void_fraction_percent = 100.0 * float(free_mask.sum()) / n_total

    accessible_count = int(sweep_result.accessible.sum())
    accessible_fraction = accessible_count / n_total
    non_accessible_fraction = (
        float(free_mask.sum()) - accessible_count
    ) / n_total

    accessible_volume = accessible_fraction * cell_volume
    non_accessible_volume = non_accessible_fraction * cell_volume

    accessible_dimensionality = max(
        sweep_result.dimensionality_by_root.values(), default=0
    )

    accessible_points = grid_points[sweep_result.accessible]

    return PoreAnalysisResult(
        probe_radius=probe_radius,
        grid_shape=grid_shape,
        lcd=lcd,
        pld=pld,
        void_fraction_percent=void_fraction_percent,
        accessible_volume=accessible_volume,
        non_accessible_volume=non_accessible_volume,
        cell_volume=cell_volume,
        accessible_dimensionality=accessible_dimensionality,
        accessible_points=accessible_points,
        geometry_method=geometry_method,
        accessibility_method=grid.accessibility_method,
    )


def refined_lcd_pld(
    structure, atom_radii, *, grid_points, distance_field, grid_lcd, grid_pld,
    prepared=None, grid_shape=None,
):
    """(lcd, pld, method) from pore_geometry, falling back to the grid
    values - unchanged, method "grid" - if anything goes wrong. Shared by
    pore analysis and guest accessibility so the two always agree.

    Round 56: the grid's own percolating path also gives a floor. Its
    nodes are at least grid_pld / 2 from every atom surface, adjacent
    nodes are at most h apart (the largest grid spacing), and the
    clearance changes by at most the distance moved, so the straight
    steps between them clear grid_pld / 2 - h / 2: a probe of diameter
    grid_pld - h certainly passes. Where the Voronoi PLD is below that
    it has provably missed a path - the radical network cannot
    represent every channel when radii differ a lot (layered CoRE MOF
    2024 structures: during round 56, 2023[Co][sql]2 came out at 4.54 A
    where its grid proves 4.73 A) - and the floor is reported instead,
    with method "voronoi, grid floor" and a logged warning.
    """
    from thevoid_core.services.pore_geometry import compute_pore_geometry

    try:
        geometry = compute_pore_geometry(
            structure.lattice.matrix,
            [site.frac_coords for site in structure.sites],
            atom_radii,
            grid_points=grid_points,
            distance_field=distance_field,
            grid_pld_radius=grid_pld / 2.0 if grid_pld is not None else None,
            prepared=prepared,
        )
    except Exception:
        logger.warning(
            "Grid-independent LCD/PLD failed; keeping the grid values.",
            exc_info=True,
        )
        return grid_lcd, grid_pld, "grid"

    lcd, pld, method = geometry.lcd, geometry.pld, geometry.method
    if grid_pld is not None and grid_shape is not None:
        spacing = float(np.max(
            np.linalg.norm(structure.lattice.matrix, axis=1) / np.asarray(grid_shape)))
        floor = grid_pld - spacing
        if pld is None or floor > pld:
            logger.warning(
                "The Voronoi PLD (%s) is below what the grid's own path proves "
                "(%.3f A); reporting the grid floor. The true PLD is at least this.",
                "none" if pld is None else f"{pld:.3f} A", floor)
            pld, method = floor, f"{method}, grid floor"
    return lcd, pld, method


def _site_symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "?"
