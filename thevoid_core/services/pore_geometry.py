"""Grid-independent pore geometry: the largest cavity diameter (LCD) and
pore limiting diameter (PLD), computed from the atoms themselves rather
than read off grid nodes.

Why this module exists (round 50)
---------------------------------
Pore analysis and guest accessibility used to report LCD and PLD as
whatever the largest / bottleneck value happened to be *at a grid
node*. The distance-to-nearest-surface function those values come from
has a kink, not a smooth peak, at every cavity centre and window
centre, so the grid error there is first order in the grid spacing,
not second order. Measured on a simple-cubic lattice of 2 A spheres,
with the default 0.3 A spacing, just by changing the cell edge from
10.00 to 10.20 A (so the cavity and window centres stop landing exactly
on grid nodes):

    LCD  13.147 A  vs  13.667 A exact   (-0.52 A)
    PLD  10.004 A  vs  10.425 A exact   (-0.42 A)

and -0.72 / -0.59 A in a 25 A cell, where the 60-point-per-axis cap
widens the spacing. That is larger than the gap between the kinetic
diameters of N2 (3.64 A) and CO2 (3.3 A) - enough, on its own, to flip
an "accessible" / "not accessible" answer.

The method
----------
PLD follows the approach Zeo++ established for this problem (Willems
et al., Microporous Mesoporous Mater. 149, 134 (2012)): the widest path
a sphere can take through the framework runs along the edges of the
atoms' Voronoi network, so the pore limiting diameter is the
bottleneck of the best percolating path through that network.

1. The radical (power) Voronoi diagram of the atoms and enough periodic
   images, with each atom weighted by its van der Waals radius, is
   built as the lower convex hull of the atoms lifted to 4D
   (x, y, z, |x|^2 - r^2) - scipy's Qhull, no extra dependency. Each
   hull facet is a tetrahedron of four atoms; its power centre is a
   Voronoi vertex.
2. Vertices are identified across periodic images by wrapping them
   back into the cell, so the network is periodic, and adjacent
   tetrahedra give the network's edges. Each edge's capacity is the
   largest sphere that can pass along it: the smallest clearance, along
   the edge segment, to the three atoms of the face the two
   tetrahedra share.
3. Edges are swept from widest to narrowest into the same
   lattice-offset-tracking union-find the grid analysis uses; the
   capacity at which a network first wraps onto itself across a cell
   boundary is the PLD.
4. The radical Voronoi diagram is exact for equal radii but only
   approximates the true "equidistant from atom surfaces" ridge when
   radii differ (the same approximation Zeo++ makes). Near-critical
   edges are therefore refined: the true clearance is maximised over
   the plane perpendicular to the edge through its bottleneck point,
   which is where the real window lies, and the sweep is re-run.
   Since round 56 a refined window only counts once an actual path
   through it between the (refined) vertices has been found, bent
   round atoms where needed, and valued by the exact clearance along
   it (voronoi_pld). Before, the window's value was taken on trust:
   MOF-205 came out 0.115 A too wide, and a vertex whose four-atom
   solution did not exist left a CoRE MOF 2024 structure 0.38 A too
   narrow.

LCD is the largest value of the exact clearance function anywhere,
found by local maximisation (Nelder-Mead) started from the best
Voronoi vertices and the best grid nodes.

The grid is still used for everything that is genuinely a volume -
void fraction, accessible versus pocket volume, the accessible points
drawn in 3D, dimensionality - where its errors are second order and
average out. If the Voronoi construction fails for any reason, the
grid values are kept and the result says so (method "grid").
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from thevoid_core.services.pore_analysis import (
    _OffsetUnionFind,
    perpendicular_widths,
)

logger = logging.getLogger(__name__)

#: How far below the first-pass PLD an edge's capacity may be and still
#: be re-examined by the in-plane refinement in step 4. The radical
#: Voronoi offset is bounded by roughly (r1 - r2)/2 * (1 - (r1+r2)/d)
#: for two atoms of radii r1, r2 a distance d apart - about 0.15 A for
#: a transition metal next to hydrogen - so 0.5 A is a comfortable
#: margin.
REFINEMENT_WINDOW = 0.5

#: How far (angstrom) the in-plane window refinement may move away
#: from the Voronoi bottleneck point - enough to reach the true window
#: centre, not so far that it wanders into a different window.
REFINEMENT_REACH = 1.5


@dataclass(frozen=True, slots=True)
class PoreGeometry:
    """LCD / PLD as diameters in angstrom, plus where they are."""

    lcd: float
    lcd_center: np.ndarray
    pld: float | None
    pld_point: np.ndarray | None
    #: "voronoi" when the Voronoi network produced the PLD, "grid" when
    #: it could not be built and the grid value was kept.
    method: str


class ClearanceFunction:
    """Exact distance from any point to the nearest atom *surface*
    (negative inside an atom), periodic in the cell.

    Points are wrapped into the cell before evaluation, so callers (and
    optimisers) can wander across cell faces freely.
    """

    def __init__(self, lattice, frac_coords, radii, reach: float) -> None:
        from scipy.spatial import cKDTree

        self.lattice = np.asarray(lattice, dtype=float)
        self.inverse = np.linalg.inv(self.lattice)

        positions, image_radii, _base, _offsets = periodic_images(
            self.lattice, frac_coords, radii, reach
        )
        self.positions = positions
        self.radii = image_radii
        self.tree = cKDTree(positions)
        self.k = int(min(32, len(positions)))

    def wrap(self, points) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=float))
        frac = np.mod(points @ self.inverse, 1.0)
        return frac @ self.lattice

    def __call__(self, points) -> np.ndarray:
        wrapped = self.wrap(points)
        distances, indices = self.tree.query(wrapped, k=self.k)

        if self.k == 1:
            distances = distances[:, np.newaxis]
            indices = indices[:, np.newaxis]

        return np.min(distances - self.radii[indices], axis=1)

    def at(self, point) -> float:
        return float(self(point)[0])


def periodic_images(lattice, frac_coords, radii, reach: float):
    """Every atom plus each periodic image lying within `reach`
    angstrom of the cell (measured perpendicular to the cell faces).

    Returns (positions, radii, base_index, offsets).
    """
    lattice = np.asarray(lattice, dtype=float)
    frac_coords = np.mod(np.asarray(frac_coords, dtype=float), 1.0)
    radii = np.asarray(radii, dtype=float)

    widths = perpendicular_widths(lattice)
    margins = reach / widths
    span = np.ceil(margins).astype(int)

    offsets = np.array(
        [
            (i, j, k)
            for i in range(-span[0], span[0] + 1)
            for j in range(-span[1], span[1] + 1)
            for k in range(-span[2], span[2] + 1)
        ],
        dtype=float,
    )

    shifted = frac_coords[np.newaxis, :, :] + offsets[:, np.newaxis, :]
    inside = np.all(
        (shifted >= -margins) & (shifted <= 1.0 + margins), axis=2
    )

    offset_index, base_index = np.nonzero(inside)
    frac = shifted[offset_index, base_index]

    return (
        frac @ lattice,
        radii[base_index],
        base_index,
        offsets[offset_index].astype(int),
    )


# ----------------------------------------------------------------------
# LCD
# ----------------------------------------------------------------------


def _maximise(clearance: ClearanceFunction, start, *, step: float,
              maxiter: int = 2000, xatol: float = 1e-5, fatol: float = 1e-7):
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=float)
    initial = np.vstack([start, start + np.eye(3) * step])

    result = minimize(
        lambda x: -clearance.at(x),
        start,
        method="Nelder-Mead",
        options={
            "initial_simplex": initial,
            "xatol": xatol,
            "fatol": fatol,
            "maxiter": maxiter,
        },
    )

    value = -float(result.fun)
    start_value = clearance.at(start)

    if value < start_value:
        return start, start_value

    return np.asarray(result.x, dtype=float), value


def refine_lcd(clearance: ClearanceFunction, candidates, *, step: float = 0.3):
    """Largest clearance anywhere, by local maximisation from each
    candidate point. Returns (radius, centre).
    """
    best_value = -np.inf
    best_point = None

    for point in candidates:
        refined_point, value = _maximise(clearance, point, step=step)

        if value > best_value:
            best_value = value
            best_point = refined_point

    return best_value, clearance.wrap(best_point)[0]


def top_candidates(points, values, count: int, separation: float):
    """Up to `count` of the highest-valued points, at least
    `separation` apart, so maximisation starts in distinct cavities.
    """
    order = np.argsort(values)[::-1]
    chosen = []

    for index in order:
        point = points[index]

        if all(np.linalg.norm(point - other) >= separation for other in chosen):
            chosen.append(point)

        if len(chosen) >= count:
            break

    return chosen


# ----------------------------------------------------------------------
# Voronoi network and PLD
# ----------------------------------------------------------------------


@dataclass
class VoronoiNetwork:
    vertex_positions: np.ndarray  # (n, 3), wrapped into the cell
    vertex_clearance: np.ndarray  # (n,)
    edges: list  # (a, b, offset, capacity, point, direction, face positions, face radii, face-only capacity)
    #: Per vertex: (positions (4, 3), radii (4,)) of its tetrahedron's
    #: atoms, as placed around the unwrapped power centre, and that
    #: centre - what the exact-vertex refinement needs.
    vertex_atoms: list | None = None
    #: Round 57: the split-sphere network (split_network), built the
    #: first time it is needed and kept here for the other analyses of
    #: the same structure.
    split: object = None
    #: For a split-sphere network: every edge with capacity at or above
    #: this has its capacity computed exactly (segment_minimum); its
    #: edges as first built, with the pieces' capacities, are raw_edges.
    exact_above: float | None = None
    raw_edges: list | None = None


def build_voronoi_network(lattice, frac_coords, radii, clearance, reach):
    """The periodic radical-Voronoi network of the structure.

    Raises on Qhull failure; callers fall back to the grid.
    """
    from scipy.spatial import ConvexHull, cKDTree

    lattice = np.asarray(lattice, dtype=float)
    inverse = np.linalg.inv(lattice)

    positions, image_radii, _base, _offsets = periodic_images(
        lattice, frac_coords, radii, reach
    )

    if len(positions) < 5:
        raise ValueError("Too few atoms for a Voronoi network.")

    # Centre the coordinates: the lifted coordinate is |x|^2, and
    # keeping it small keeps Qhull well conditioned.
    origin = lattice.sum(axis=0) / 2.0
    local = positions - origin
    lifted = np.column_stack(
        [local, np.sum(local**2, axis=1) - image_radii**2]
    )

    try:
        hull = ConvexHull(lifted)
    except Exception:
        hull = ConvexHull(lifted, qhull_options="QJ")

    equations = hull.equations
    lower = equations[:, 3] < -1e-12

    centres = np.full((len(equations), 3), np.nan)
    centres[lower] = (
        -equations[lower, :3] / (2.0 * equations[lower, 3:4]) + origin
    )

    simplices = hull.simplices
    neighbors = hull.neighbors

    frac = centres @ inverse
    cell_offset = np.floor(frac + 1e-9)
    owned = lower & np.all(cell_offset == 0, axis=1)

    owned_indices = np.nonzero(owned)[0]

    if len(owned_indices) == 0:
        raise ValueError("No Voronoi vertices inside the cell.")

    owned_frac = np.mod(frac[owned_indices], 1.0)
    # Periodic lookup of "which owned vertex is this" by fractional
    # position; the tolerance is far below any real vertex spacing but
    # far above floating-point noise between images.
    lookup = cKDTree(np.mod(owned_frac, 1.0), boxsize=1.0 + 1e-12)

    vertex_positions = owned_frac @ lattice
    vertex_clearance = clearance(vertex_positions)
    vertex_atoms = [
        (positions[simplices[f]], image_radii[simplices[f]], centres[f])
        for f in owned_indices
    ]


    # Round 58: every (vertex, neighbour) pair at once, in the order the
    # loop over owned vertices and their four neighbours used to visit
    # them, so the edges come out identical and in the same order.
    facet_node = np.full(len(equations), -1, dtype=int)
    facet_node[owned_indices] = np.arange(len(owned_indices))
    node_a_all = np.repeat(np.arange(len(owned_indices)), neighbors.shape[1])
    facet_all = np.repeat(owned_indices, neighbors.shape[1])
    neighbor_all = neighbors[owned_indices].reshape(-1)
    valid = neighbor_all >= 0
    valid[valid] = lower[neighbor_all[valid]]
    node_a_all, facet_all, neighbor_all = node_a_all[valid], facet_all[valid], neighbor_all[valid]
    node_b_all = facet_node[neighbor_all]
    outside = node_b_all < 0
    if outside.any():
        distance, match = lookup.query(np.mod(frac[neighbor_all[outside]], 1.0))
        # A facet near the edge of the image cloud, not a real vertex, is
        # left out; the same edge is found again from the other end,
        # inside the cell.
        node_b_all[outside] = np.where(distance > 1e-6, -1, match)
    keep = node_b_all >= 0
    node_a_all, node_b_all = node_a_all[keep], node_b_all[keep]
    facet_all, neighbor_all = facet_all[keep], neighbor_all[keep]

    # Round 55: the cell offset is measured between where the two
    # vertices are actually stored, not from floor(neighbor_frac). The two
    # differ by a whole cell when a vertex lies on a cell face to
    # within the 1e-6 matching tolerance - its stored copy wrapped
    # to the opposite face - which in high-symmetry frameworks
    # (ZIF-8 has vertices exactly on its cell faces) produced
    # phantom edges one or more cells long, and with them
    # connections across windows far narrower than the probe.
    # The same holds for vertex a itself: an owned vertex within
    # 1e-9 of a face is stored wrapped to the opposite face, so
    # its own shift has to come off too.
    shift_a = np.rint(frac[facet_all] - owned_frac[node_a_all])
    offsets = (np.rint(frac[neighbor_all] - owned_frac[node_b_all]) - shift_a).astype(int)

    # The three atoms two neighbouring tetrahedra share.
    a_sets, b_sets = simplices[facet_all], simplices[neighbor_all]
    shared = (a_sets[:, :, None] == b_sets[:, None, :]).any(axis=2)
    three = shared.sum(axis=1) == 3

    pending = []
    seen = set()
    for index, (node_a, node_b, offset) in enumerate(
            zip(node_a_all.tolist(), node_b_all.tolist(), map(tuple, offsets.tolist()))):
        key = (min(node_a, node_b), max(node_a, node_b),
               offset if node_a <= node_b else tuple(-v for v in offset))
        if key in seen:
            continue
        seen.add(key)
        if three[index]:
            face = sorted(a_sets[index][shared[index]].tolist())
        else:
            face = sorted(set(a_sets[index].tolist()) & set(b_sets[index].tolist()))
        pending.append((node_a, node_b, offset, face, int(facet_all[index]),
                        int(neighbor_all[index])))

    # Round 58: the bottleneck and clearance of every edge at once. Done
    # edge by edge (74,000 edges, one KD-tree query each) this was half
    # the cost of a split-sphere network.
    edges = []
    if pending:
        starts = centres[[e[4] for e in pending]]
        ends = centres[[e[5] for e in pending]]
        capacities = np.empty(len(pending))
        points = (starts + ends) / 2.0
        by_size = {}
        for index, entry in enumerate(pending):
            by_size.setdefault(len(entry[3]), []).append(index)
        for size, indices in by_size.items():
            indices = np.asarray(indices)
            faces = np.array([pending[i][3] for i in indices], dtype=int).reshape(len(indices), size)
            capacities[indices], points[indices] = _edge_bottlenecks(
                starts[indices], ends[indices], positions[faces], image_radii[faces])
        at_points = clearance(points) if len(points) else np.empty(0)
        for index, (node_a, node_b, offset, face, facet, neighbor) in enumerate(pending):
            # Round 56: what the edge's own three atoms allow - Zeo++'s
            # edge radius, which ignores any other atom reaching in
            # closer - kept for comparison and diagnostics.
            face_capacity = float(capacities[index])
            capacity = min(
                face_capacity,
                float(at_points[index]),
                float(vertex_clearance[node_a]),
                float(vertex_clearance[node_b]),
            )
            direction = ends[index] - starts[index]
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > 1e-12 else None
            edges.append(
                (node_a, node_b, offset, capacity, points[index], direction,
                 positions[face], image_radii[face], face_capacity)
            )

    return VoronoiNetwork(
        vertex_positions, vertex_clearance, edges, vertex_atoms
    )


def _edge_bottlenecks(starts, ends, atom_positions, atom_radii):
    """_edge_bottleneck for many segments at once: starts/ends (n, 3),
    atom_positions (n, k, 3), atom_radii (n, k). The same arithmetic,
    and the same atom wins a tie (the first)."""
    segments = ends - starts
    length_squared = np.einsum("ij,ij->i", segments, segments)
    relative = atom_positions - starts[:, None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.einsum("ikj,ij->ik", relative, segments) / length_squared[:, None]
    t = np.where(length_squared[:, None] < 1e-18, 0.0, np.clip(t, 0.0, 1.0))
    points = starts[:, None, :] + t[:, :, None] * segments[:, None, :]
    values = np.linalg.norm(atom_positions - points, axis=2) - atom_radii
    best = np.argmin(values, axis=1)
    rows = np.arange(len(starts))
    return values[rows, best], points[rows, best]


def _edge_bottleneck(start, end, atom_positions, atom_radii):
    """Smallest clearance to the given atoms along the segment
    start-end, and where it occurs.

    Each atom's clearance along a straight segment is convex, so its
    minimum is at the atom's projection onto the segment (clamped to
    the ends).
    """
    segment = end - start
    length_squared = float(np.dot(segment, segment))

    best = np.inf
    best_point = (start + end) / 2.0

    for position, radius in zip(atom_positions, atom_radii):
        if length_squared < 1e-18:
            t = 0.0
        else:
            t = float(np.clip(np.dot(position - start, segment) / length_squared, 0.0, 1.0))

        point = start + t * segment
        value = float(np.linalg.norm(position - point) - radius)

        if value < best:
            best = value
            best_point = point

    return best, best_point


def _exact_vertex_point(clearance, atom_positions, atom_radii, start):
    """(point, clearance) of the true vertex of four atoms of different
    radii - the point equidistant from all four *surfaces*, solved by
    Newton's method from the power centre - or None if Newton does not
    converge within REFINEMENT_REACH, or some fifth atom is closer. For
    equal radii it is the power centre itself; for unequal radii the
    power centre sits slightly off it, closer to the larger atoms, and
    underestimates the room there (confirmed on CALF-20, round 50)."""
    x = np.asarray(start, dtype=float).copy()
    d0 = np.linalg.norm(atom_positions - x, axis=1) - atom_radii
    t = float(np.mean(d0))
    for _ in range(30):
        delta = x - atom_positions
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances < 1e-9):
            return None
        residual = distances - atom_radii - t
        jacobian = np.column_stack([delta / distances[:, None], -np.ones(4)])
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return None
        x = x + step[:3]
        t = t + step[3]
        if np.max(np.abs(step)) < 1e-10:
            break
    else:
        return None
    if np.linalg.norm(x - start) > REFINEMENT_REACH:
        return None
    value = clearance.at(x)
    if value < t - 1e-6:
        return None
    return x, value


def _refine_window_point(clearance, point, direction):
    """Maximise the true clearance over the plane through `point`
    perpendicular to `direction`, within REFINEMENT_REACH of `point` -
    the window's actual centre, which the radical Voronoi edge only
    approximates when radii differ. Returns (value, point)."""
    from scipy.optimize import minimize

    point = np.asarray(point, dtype=float)
    base_value = clearance.at(point)

    if direction is None:
        return base_value, point

    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, direction)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    v = np.cross(direction, u)

    def objective(st):
        offset = st[0] * u + st[1] * v
        distance = np.linalg.norm(offset)
        penalty = max(0.0, distance - REFINEMENT_REACH) * 10.0
        return -clearance.at(point + offset) + penalty

    result = minimize(
        objective,
        np.zeros(2),
        method="Nelder-Mead",
        options={
            "initial_simplex": np.array([[0, 0], [0.2, 0], [0, 0.2]]),
            # Round 56: only a waypoint and an optimistic value now -
            # every route through it is valued exactly - so it need not
            # be found to 1e-5 A.
            "xatol": 1e-3,
            "fatol": 1e-5,
            "maxiter": 200,
        },
    )

    offset = result.x[0] * u + result.x[1] * v

    if np.linalg.norm(offset) > REFINEMENT_REACH or -float(result.fun) <= base_value:
        return base_value, point

    return -float(result.fun), point + offset


def segment_minimum(clearance: ClearanceFunction, start, end):
    """Exact smallest clearance along the straight segment start-end,
    over every atom that could be nearest anywhere on it, and where it
    occurs: (value, point).

    Along a straight segment each atom's clearance is convex, with its
    minimum at the atom's projection onto the segment, so the minimum
    over atoms is exact - no sampling. An atom can only matter if it is
    within clearance(midpoint) + its radius + half the segment's length
    of the midpoint, which bounds the atoms considered.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    middle = (start + end) / 2.0
    shift = clearance.wrap(middle)[0] - middle
    a, b = start + shift, end + shift
    segment = b - a
    length_squared = float(np.dot(segment, segment))
    bound = (max(clearance.at(middle), 0.0) + float(np.max(clearance.radii))
             + 0.5 * np.sqrt(length_squared) + 1e-9)
    nearby = clearance.tree.query_ball_point((a + b) / 2.0, bound)
    if not nearby:
        return clearance.at(middle), middle
    positions = clearance.positions[nearby]
    radii = clearance.radii[nearby]
    if length_squared < 1e-18:
        t = np.zeros(len(nearby))
    else:
        t = np.clip((positions - a) @ segment / length_squared, 0.0, 1.0)
    points = a + t[:, None] * segment
    values = np.linalg.norm(positions - points, axis=1) - radii
    best = int(np.argmin(values))
    return float(values[best]), points[best] - shift


def _vertex_point(clearance, atom_positions, atom_radii, centre):
    """Where a path through this Voronoi vertex should pass, and the
    clearance there: the true vertex of its four atoms (_exact_vertex_point)
    when it exists nearby and beats the power centre; otherwise the
    local maximum of the clearance within REFINEMENT_REACH of the
    centre, or the power centre itself.

    With very different radii side by side (a carbon among hydrogens:
    1.70 and 1.09 A), the power centre can sit well inside the room
    the true surfaces leave, and the four-atom vertex may not exist
    nearby (another atom is closer): round 56 found a PLD 0.38 A low on
    a CoRE MOF structure this way. The true vertex is preferred over
    the local maximum because it lies on the ridges the network's edges
    follow; a maximum further into a cavity is higher but can be worse
    to route through. Either is a real point, and whether a path can
    reach it is checked separately (segment_minimum), so nothing here
    can overstate the PLD.
    """
    centre = np.asarray(centre, dtype=float)
    base = clearance.at(centre)
    exact = _exact_vertex_point(clearance, atom_positions, atom_radii, centre)
    if exact is not None:
        return (exact[0], exact[1]) if exact[1] >= base else (centre, base)
    # Only a route point: it need not be precise, as every route
    # through it is valued exactly.
    point, value = _maximise(clearance, centre, step=0.2, maxiter=100, xatol=1e-3, fatol=1e-5)
    if np.linalg.norm(point - centre) <= REFINEMENT_REACH and value > base:
        return point, value
    return centre, base


def _percolation_capacity(n_nodes, edges, capacities):
    """Largest capacity at which the network percolates, and the index
    of the edge that makes it percolate (None, None if it never does).
    """
    order = sorted(range(len(edges)), key=lambda i: capacities[i], reverse=True)
    union_find = _OffsetUnionFind.for_n_nodes(n_nodes)

    for index in order:
        capacity = capacities[index]

        if capacity <= 0:
            break

        a, b, offset = edges[index][0], edges[index][1], edges[index][2]
        union_find.union(a, b, offset)

        root, _ = union_find.find(a)

        if union_find.percolation_vectors_for_root(root):
            return capacity, index

    return None, None


def _route_value(clearance, points) -> float:
    """Smallest exact clearance along the polyline through `points`."""
    return min(
        segment_minimum(clearance, points[k], points[k + 1])[0]
        for k in range(len(points) - 1)
    )


#: How many times _best_route may bend a path to go round the atom that
#: limits it.
ROUTE_REPAIRS = 12


def _nearest_atom(clearance, point):
    wrapped = clearance.wrap(point)[0]
    distances, indices = clearance.tree.query(wrapped, k=clearance.k)
    values = distances - clearance.radii[indices]
    best = int(np.argmin(values))
    return clearance.positions[indices[best]] - (wrapped - point), float(values[best])


def _best_route(clearance, start, end, window, bottleneck, target) -> float:
    """The best clearance found along a path from `start` to `end` near
    one Voronoi edge - every candidate an actual polyline, valued
    exactly (segment_minimum), so the result is achievable.

    Tried in order, stopping once `target` is reached: straight; bent
    through the refined window centre; bent through the radical edge's
    bottleneck; then the best of those repaired: the point where the
    path comes closest to an atom is pushed straight away from that
    atom until it clears `target`, splitting the segment there, up to
    ROUTE_REPAIRS times. The repair is for ridges that curve - in
    MOF-205, straight lines from a window centre to the neighbouring
    vertices dip 0.06 A below both.
    """
    candidates = [[start, end], [start, window, end], [start, bottleneck, end]]
    best_value, best_path = -np.inf, None
    for path in candidates:
        value = _route_value(clearance, path)
        if value > best_value:
            best_value, best_path = value, path
        if best_value >= target:
            return best_value

    path = [np.asarray(q, dtype=float) for q in best_path]
    for _ in range(ROUTE_REPAIRS):
        worst, where, at = np.inf, None, None
        for k in range(len(path) - 1):
            value, point = segment_minimum(clearance, path[k], path[k + 1])
            if value < worst:
                worst, where, at = value, k, point
        best_value = max(best_value, worst)
        if worst >= target:
            return worst
        atom, value = _nearest_atom(clearance, at)
        away = at - atom
        norm = np.linalg.norm(away)
        if norm < 1e-9:
            break
        pushed = at + away / norm * (target - value + 1e-4)
        path.insert(where + 1, pushed)
    return max(best_value, _route_value(clearance, path))


class RefinedNetwork:
    """The radical Voronoi network plus, near a level, a second node per
    vertex at a better point and a second route per edge between those
    points - each route used only once an actual path has been found
    for it (round 56). Shared by the PLD (voronoi_pld) and by
    accessibility at a probe radius (network_accessibility), so the two
    can never disagree about whether a window is open.

    Why: the refinement of rounds 50-55 raised an edge's capacity to
    min(window centre, vertex values at its ends), taking for granted
    that a path joined the window to each vertex at that level, and
    kept a vertex at its power centre when the true four-atom vertex
    could not be found. The first overstated MOF-205's PLD by 0.115 A
    (8.433 against Zeo++'s high-accuracy 8.318); the second understated
    a CoRE MOF 2024 structure's by 0.38 A (5.57 where a 0.15 A grid
    shows at least 5.68, and Zeo++ 5.95).

    Nodes 0..n-1 are the network's own vertices (at their power
    centres); n.. are refined vertices, each joined to its original by
    the exact clearance of the segment between them (a link). A refined
    vertex sits at its true four-atom vertex, found up front, or - where
    that does not exist - at the local maximum of the clearance near
    it, looked for only once a route through it is needed (until then
    its value counts as unbounded). A route's value starts optimistic,
    min(refined window, its vertices), or what the original edge and
    the two links certainly give if that is more; `certify(level)`
    replaces the optimistic values of the routes that matter at `level`
    by the best actual path _best_route finds.
    """

    def __init__(self, network: VoronoiNetwork, clearance: ClearanceFunction, threshold: float,
                 ceiling: float = np.inf, eager_windows: bool = False):
        """Edges with capacity in [threshold, ceiling) get a second route,
        vertices with clearance there (and every such edge's ends) a
        second node. Accessibility at one probe radius passes the radius
        as the ceiling - an edge that already admits the probe needs no
        other route - which keeps the refinement to the edges round-55
        accessibility refined. eager_windows refines every candidate
        window up front, which the PLD's search over levels needs (with
        windows unknown, every route looks open and each loop verifies
        far more of them); a single level is cheaper lazily."""
        self.network, self.clearance = network, clearance
        self.n = n = len(network.vertex_positions)
        edges = network.edges
        self.capacities = [edge[3] for edge in edges]
        self.near_edges = [i for i, c in enumerate(self.capacities) if threshold <= c < ceiling]
        near = {i for i, v in enumerate(network.vertex_clearance) if threshold <= v < ceiling}
        for index in self.near_edges:
            near.update((edges[index][0], edges[index][1]))

        self.node, self.point, self.value, self.upgradable = {}, {}, {}, set()
        self.link_edges, self.link_caps, self.link_of = [], [], {}
        for vertex in sorted(near):
            stored = network.vertex_positions[vertex]
            point, value = stored, float(network.vertex_clearance[vertex])
            if network.vertex_atoms is not None:
                atoms, atom_radii, centre = network.vertex_atoms[vertex]
                exact = _exact_vertex_point(clearance, atoms, atom_radii, centre)
                if exact is None:
                    # Only worth a search if the vertex does not already
                    # clear the ceiling where it is.
                    if value < ceiling:
                        self.upgradable.add(vertex)
                elif exact[1] >= value:
                    point, value = exact[0] + (stored - centre), exact[1]
            self.node[vertex] = n + len(self.node)
            self.point[vertex] = point
            self.value[vertex] = value
            self.link_of[vertex] = len(self.link_edges)
            self.link_edges.append((vertex, self.node[vertex], (0, 0, 0)))
            self.link_caps.append(value if point is stored
                                  else segment_minimum(clearance, stored, point)[0])

        self.route_edges, self.routes_of, self.windows = [], {}, []
        for k, index in enumerate(self.near_edges):
            a, b, offset = edges[index][0], edges[index][1], edges[index][2]
            self.route_edges.append((self.node[a], self.node[b], offset))
            self.routes_of.setdefault(a, []).append(k)
            self.routes_of.setdefault(b, []).append(k)
            # Refined when first needed (_window) unless eager: most
            # candidate edges never are, and each refinement is a
            # Nelder-Mead search.
            self.windows.append(None)
            if eager_windows:
                self._window(k)

        self.values, self.verified = [], []
        for k in range(len(self.route_edges)):
            value, certain = self._reset(k)
            self.values.append(value)
            self.verified.append(certain)

    @property
    def n_nodes(self) -> int:
        return self.n + len(self.node)

    def edges_and_capacities(self):
        edges = [tuple(e[:3]) for e in self.network.edges] + self.link_edges + self.route_edges
        return edges, self.capacities + self.link_caps + self.values

    def node_position(self, node):
        """(position, clearance) of any node."""
        if node < self.n:
            return self.network.vertex_positions[node], float(self.network.vertex_clearance[node])
        vertex = self.link_edges[node - self.n][0]
        return self.point[vertex], self.value[vertex]

    # ------------------------------------------------------------------

    def _optimistic(self, vertex):
        return np.inf if vertex in self.upgradable else self.value[vertex]

    def _through_original(self, k):
        """A route that is certainly there: refined point -> original
        vertex -> the original edge -> original vertex -> refined point."""
        edge = self.network.edges[self.near_edges[k]]
        return min(self.capacities[self.near_edges[k]],
                   self.link_caps[self.link_of[edge[0]]], self.link_caps[self.link_of[edge[1]]])

    def _window(self, k):
        """(bottleneck, refined window centre, its clearance) for route k.
        The stored bottleneck point is in the frame of vertex a's facet
        (its power centre); the refinement starts there."""
        if self.windows[k] is None:
            edge = self.network.edges[self.near_edges[k]]
            bottleneck = np.asarray(edge[4], dtype=float)
            value, window = _refine_window_point(self.clearance, bottleneck, edge[5])
            self.windows[k] = (bottleneck, window, value)
        return self.windows[k]

    def _optimistic_route(self, k):
        edge = self.network.edges[self.near_edges[k]]
        window = np.inf if self.windows[k] is None else self.windows[k][2]
        return max(self._through_original(k),
                   min(window, self._optimistic(edge[0]), self._optimistic(edge[1])))

    def _reset(self, k):
        known, hope = self._through_original(k), self._optimistic_route(k)
        return hope, hope <= known

    def _upgrade(self, vertex) -> bool:
        atoms, atom_radii, centre = self.network.vertex_atoms[vertex]
        stored = self.network.vertex_positions[vertex]
        point, value = _vertex_point(self.clearance, atoms, atom_radii, centre)
        self.upgradable.discard(vertex)
        if value <= self.value[vertex]:
            return False
        self.point[vertex] = point + (stored - centre)
        self.value[vertex] = value
        self.link_caps[self.link_of[vertex]] = segment_minimum(
            self.clearance, stored, self.point[vertex])[0]
        # Routes already checked from this vertex's old point start
        # somewhere else now.
        for other in self.routes_of[vertex]:
            self.values[other], self.verified[other] = self._reset(other)
        return True

    def _verify(self, k, level):
        """Check route k, needed at `level`. The window is refined first:
        if it cannot reach the level, the route's value just drops to
        what it can hope for and the costly vertex search is skipped."""
        index = self.near_edges[k]
        a, b, offset = self.network.edges[index][:3]
        self._window(k)
        hope = self._optimistic_route(k)
        if hope < level:
            self.values[k] = hope
            return
        for vertex in (a, b):
            if vertex in self.upgradable:
                self._upgrade(vertex)
        if self.verified[k]:
            return
        lattice = self.clearance.lattice
        target = self._optimistic_route(k)
        if target <= self._through_original(k):
            self.values[k], self.verified[k] = target, True
            return
        pa = self.point[a]
        pb = self.point[b] + np.asarray(offset, dtype=float) @ lattice
        bottleneck, window, _ = self.windows[k]
        shift = np.rint((pa - bottleneck) @ self.clearance.inverse) @ lattice
        found = _best_route(self.clearance, pa, pb, window + shift, bottleneck + shift,
                            target=target)
        self.values[k] = max(self._through_original(k), min(target, found))
        self.verified[k] = True

    def certify(self, level) -> None:
        """Make every route whose value is at least `level` an actual
        path at that value (repeating until no unchecked route claims
        the level - checking one can reopen others)."""
        while True:
            pending = [k for k, value in enumerate(self.values)
                       if value >= level and not self.verified[k]]
            if not pending:
                return
            for k in pending:
                if not self.verified[k] and self.values[k] >= level:
                    self._verify(k, level)

    def pld(self):
        """(radius, index of the critical edge in edges_and_capacities)
        of the largest level at which the refined network percolates
        using only certified routes."""
        while True:
            edges, capacities = self.edges_and_capacities()
            radius, critical = _percolation_capacity(self.n_nodes, edges, capacities)
            if radius is None:
                return None, None
            pending = [k for k, value in enumerate(self.values)
                       if value >= radius and not self.verified[k]]
            if not pending:
                return radius, critical
            for k in pending:
                if not self.verified[k] and self.values[k] >= radius:
                    self._verify(k, radius)

    def location(self, critical):
        edges = self.network.edges
        if critical < len(edges):
            return edges[critical][4]
        critical -= len(edges)
        if critical < len(self.link_edges):
            return self.point[self.link_edges[critical][0]]
        return self._window(critical - len(self.link_edges))[1]


def voronoi_pld(network: VoronoiNetwork, clearance: ClearanceFunction):
    """PLD radius and its location from the Voronoi network, with
    near-critical windows and vertices refined (method step 4; see
    RefinedNetwork for what "refined" means since round 56). Never
    below the radical-Voronoi value - the original network is kept -
    and every refined value it rests on is that of an actual path.
    """
    n_nodes = len(network.vertex_positions)
    edges = network.edges
    capacities = [edge[3] for edge in edges]

    first, first_edge = _percolation_capacity(n_nodes, edges, capacities)

    if first is None:
        return None, None

    refined = RefinedNetwork(network, clearance, first - REFINEMENT_WINDOW, eager_windows=True)
    radius, critical = refined.pld()

    if radius is None or radius <= first:
        return first, clearance.wrap(edges[first_edge][4])[0]
    return radius, clearance.wrap(refined.location(critical))[0]


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def compute_pore_geometry(
    lattice,
    frac_coords,
    radii,
    *,
    grid_points: np.ndarray,
    distance_field: np.ndarray,
    grid_pld_radius: float | None,
    prepared=None,
) -> PoreGeometry:
    """Grid-independent LCD and PLD (as diameters) for a structure.

    grid_points / distance_field / grid_pld_radius are the grid
    analysis's own results, used as optimisation starting points, to
    size the image cloud, and as the fallback.
    """
    lattice = np.asarray(lattice, dtype=float)
    radii = np.asarray(radii, dtype=float)
    max_radius = float(np.max(radii))

    grid_lcd_radius = float(np.max(distance_field))
    # Every empty sphere touching the cell has radius at most the LCD
    # radius plus the largest atom radius; its atoms lie within that of
    # its centre, and neighbouring Voronoi vertices within about twice
    # that - so this reach keeps every tetrahedron the cell needs
    # correct.
    sphere_reach = max(grid_lcd_radius, 0.0) + max_radius + 0.5
    reach = 2.0 * sphere_reach + 1.0

    candidates = top_candidates(grid_points, distance_field, 6, separation=1.0)

    if prepared is not None:
        # Round 55: the (network, clearance) pair classify_void_grid
        # already built from the same inputs, rather than a second
        # construction.
        network, clearance = prepared
    else:
        clearance = ClearanceFunction(lattice, frac_coords, radii, sphere_reach + 1.0)
        try:
            network = build_voronoi_network(
                lattice, frac_coords, radii, clearance, reach
            )
        except Exception:
            logger.warning(
                "Voronoi network could not be built; using grid LCD/PLD.",
                exc_info=True,
            )
            network = None

    if network is not None and len(network.vertex_positions):
        candidates += top_candidates(
            network.vertex_positions, network.vertex_clearance, 6,
            separation=1.0,
        )

    lcd_radius, lcd_center = refine_lcd(clearance, candidates)
    lcd_radius = max(lcd_radius, grid_lcd_radius)

    method = "grid"
    pld_radius = grid_pld_radius
    pld_point = None

    if network is not None:
        try:
            voronoi_radius, voronoi_point = voronoi_pld(network, clearance)
        except Exception:
            logger.warning(
                "Voronoi PLD failed; using grid PLD.", exc_info=True
            )
            voronoi_radius, voronoi_point = None, None

        if voronoi_radius is not None:
            pld_radius = voronoi_radius
            pld_point = voronoi_point
            method = "voronoi"
        elif grid_pld_radius is not None:
            logger.warning(
                "Voronoi network did not percolate but the grid did "
                "(PLD radius %.3f); keeping the grid value.",
                grid_pld_radius,
            )
        else:
            method = "voronoi"

    if (network is not None and pld_radius is not None and method == "voronoi"
            and np.ptp(radii) > 0.02):
        # Round 57 built this only when the grid's PLD came close to the
        # network's; round 58 found three CoRE structures 0.18-0.34 A low
        # that the signal missed, and no signal that separates them (see
        # the history note above split_reach). Whenever the radii differ, the
        # split network is now built.
        try:
            split = cached_split_network(
                network, lattice, frac_coords, radii, clearance, split_reach(reach),
                exact_above=pld_radius - SPLIT_EXACT_MARGIN)
            split_radius, split_edge = _percolation_capacity(
                len(split.vertex_positions), split.edges, [e[3] for e in split.edges])
        except Exception:
            logger.warning("Split-sphere network failed; keeping the Voronoi PLD.",
                           exc_info=True)
            split_radius = None
        if split_radius is not None and split_radius > pld_radius:
            logger.info("Split-sphere network: PLD %.3f A (radical network %.3f A).",
                        2 * split_radius, 2 * pld_radius)
            pld_radius = split_radius
            pld_point = clearance.wrap(split.edges[split_edge][4])[0]
            method = "voronoi+split"

    if pld_radius is not None:
        # A path's bottleneck can never be wider than the widest cavity.
        pld_radius = min(pld_radius, lcd_radius)

    return PoreGeometry(
        lcd=2.0 * lcd_radius,
        lcd_center=lcd_center,
        pld=2.0 * pld_radius if pld_radius is not None else None,
        pld_point=pld_point,
        method=method,
    )


# ----------------------------------------------------------------------
# Accessibility with the Voronoi network (round 55)
# ----------------------------------------------------------------------


#: Round 57. Each atom larger than the smallest is replaced, for the
#: split-sphere network only, by a sphere of the smallest radius at its
#: centre and this many more pushed out to touch its surface (Fibonacci
#: directions). 12 directions gave 2.545 A on 2020[Cd][hcb]2[ASR]1, 20
#: gave 2.601 and 30 2.644 (Zeo++ -ha: 2.651).
SPLIT_DIRECTIONS = 30

#: History (no longer used). Round 57 tried the split-sphere network only
#: when the grid's node-based PLD came within 0.015 A (radius) of the
#: Voronoi PLD: on its 40 CoRE structures that singled out the two whose
#: Voronoi PLD was provably low. On round 58's 36 (Zn, live-fetched) the
#: split network raised 9, three of them by 0.13-0.29 A, with grid gaps
#: of 0.05-0.16 A - inside the 0.05-0.23 A range of the 27 it left
#: unchanged. The network is now always built when radii differ.


def split_reach(reach: float) -> float:
    """The image-cloud reach of the split-sphere network: half the
    radical network's (sphere_reach + 1 rather than 2 sphere_reach + 1).

    Every vertex in the cell has its atoms within sphere_reach, so the
    vertices are right; what a thinner cloud can get wrong is a
    tetrahedron next to one at its edge, and so an edge. That can only
    lose a path, never invent a wide one: every edge that can decide the
    PLD is valued by the exact clearance along it (make_exact), whether
    or not the diagram around it is exact. On round 58's nine structures
    where the split network changed the PLD, half the reach gave the same
    PLD to 1e-4 A in half the time (the cloud is 31 spheres per atom).
    """
    return (reach - 1.0) / 2.0 + 1.0

#: Margin below the level of interest down to which split-network edge
#: capacities are computed exactly.
SPLIT_EXACT_MARGIN = 0.5


def split_network(lattice, frac_coords, radii, clearance, reach, *, exact_above,
                  directions: int = SPLIT_DIRECTIONS) -> VoronoiNetwork:
    """A Voronoi network that follows the true channels when atom radii
    differ a lot (round 57).

    The radical Voronoi diagram - the one Zeo++ and the Void use - is the
    exact "equidistant from the surfaces" skeleton only for equal radii.
    With a carbon (1.70 A) among hydrogens (1.09 A) its edges can run
    through the wrong places altogether, and in layered frameworks the
    channel between the layers is not represented at all: on
    2020[Cd][hcb]2[ASR]1 the radical network's best path is 2.341 A
    where a 0.1 A grid proves at least 2.48 A.

    Here every atom larger than the smallest is split into spheres of
    the smallest radius (Zeo++'s high-accuracy idea), so the radical
    diagram of the pieces is close to the true one. The pieces only
    decide where the network runs: vertex clearances are the true
    clearance at the vertex, and every edge whose capacity could matter
    (at or above `exact_above`) is valued by the exact clearance along
    the straight segment between its vertices (segment_minimum), so the
    network's PLD is a real path's - a lower bound on the true PLD, as
    the radical network's is. Zeo++'s high-accuracy mode, which values
    the network with the pieces instead, can come out above.
    """
    lattice = np.asarray(lattice, dtype=float)
    radii = np.asarray(radii, dtype=float)
    cart = np.mod(np.asarray(frac_coords, dtype=float), 1.0) @ lattice
    r_min = float(radii.min())
    k = np.arange(directions) + 0.5
    polar = np.arccos(1.0 - 2.0 * k / directions)
    azimuth = np.pi * (1.0 + 5.0 ** 0.5) * k
    unit = np.stack([np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar),
                     np.cos(polar)], axis=1)
    points = []
    for centre, radius in zip(cart, radii):
        points.append(centre)
        if radius > r_min + 0.02:
            points.extend(centre + unit * (radius - r_min))
    points = np.array(points)
    pieces = build_voronoi_network(lattice, points @ np.linalg.inv(lattice),
                                   np.full(len(points), r_min), clearance, reach)
    split = VoronoiNetwork(pieces.vertex_positions, pieces.vertex_clearance,
                           list(pieces.edges), vertex_atoms=None, exact_above=np.inf)
    split.raw_edges = pieces.edges
    make_exact(split, clearance, lattice, exact_above)
    return split


def make_exact(split, clearance, lattice, exact_above) -> None:
    """Value, by the exact clearance along it, every edge of a
    split-sphere network whose capacity could be at or above
    `exact_above` and is not exact yet."""
    lattice = np.asarray(lattice, dtype=float)
    for index, edge in enumerate(split.raw_edges):
        optimistic = max(edge[3], edge[8])
        if exact_above <= optimistic < split.exact_above or (
                split.exact_above == np.inf and optimistic >= exact_above):
            start = split.vertex_positions[edge[0]]
            end = split.vertex_positions[edge[1]] + np.asarray(edge[2], dtype=float) @ lattice
            capacity, point = segment_minimum(clearance, start, end)
            split.edges[index] = (edge[0], edge[1], edge[2], capacity, point) + tuple(edge[5:])
    split.exact_above = min(split.exact_above, exact_above)


def cached_split_network(network, lattice, frac_coords, radii, clearance, reach,
                         exact_above):
    """network.split, built once, with its exact capacities extended
    down to `exact_above` if an earlier use needed fewer."""
    split = network.split
    if split is None:
        split = split_network(lattice, frac_coords, radii, clearance, reach,
                              exact_above=exact_above)
        network.split = split
    elif split.exact_above > exact_above:
        make_exact(split, clearance, lattice, exact_above)
    return split


def network_reach(radii, distance_field) -> float:
    """The image-cloud reach both networks are built with (see
    compute_pore_geometry)."""
    sphere_reach = max(float(np.max(distance_field)), 0.0) + float(np.max(radii)) + 0.5
    return 2.0 * sphere_reach + 1.0


def prepare_network(lattice, frac_coords, radii, distance_field):
    """(network or None, clearance): the Voronoi network and exact
    clearance function, built the way compute_pore_geometry builds them,
    so one construction can serve both the PLD and the accessibility
    classification."""
    lattice = np.asarray(lattice, dtype=float)
    radii = np.asarray(radii, dtype=float)
    max_radius = float(np.max(radii))
    grid_lcd_radius = float(np.max(distance_field))
    sphere_reach = max(grid_lcd_radius, 0.0) + max_radius + 0.5
    reach = 2.0 * sphere_reach + 1.0
    clearance = ClearanceFunction(lattice, frac_coords, radii, sphere_reach + 1.0)

    try:
        network = build_voronoi_network(lattice, frac_coords, radii, clearance, reach)
    except Exception:
        logger.warning("Voronoi network could not be built.", exc_info=True)
        network = None

    return network, clearance


#: How far (angstrom) from a reachable Voronoi vertex a stranded grid
#: node is looked at, and how many nodes of one stranded blob are tried
#: before giving up on it.
STRANDED_REACH = 3.0
STRANDED_ATTEMPTS = 8


def _segment_is_clear(clearance, start, end, radius, step: float = 0.05) -> bool:
    """Whether a probe of `radius` can move along the straight segment.
    The clearance is sampled every `step` angstrom; being 1-Lipschitz it
    can dip at most step/2 below the smaller of two neighbouring samples,
    so requiring every sample to clear radius + step/2 makes the answer
    rigorous (it can only err towards "not clear")."""
    length = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(length / step)) + 1)
    t = np.linspace(0.0, 1.0, count)[:, None]
    values = clearance(start[None, :] + t * (end - start)[None, :])
    spacing = length / (count - 1)
    return bool(np.min(values) >= radius + spacing / 2.0)


@dataclass
class NetworkAccessibility:
    accessible: np.ndarray
    dimensionality_by_root: dict
    #: How many (grid node, Voronoi vertex) links were made.
    links: int


def network_accessibility(
    lattice,
    *,
    grid_frac: np.ndarray,
    distance_field: np.ndarray,
    grid_shape,
    probe_radius: float,
    network: VoronoiNetwork,
    clearance: ClearanceFunction,
    grid_union_find=None,
) -> NetworkAccessibility:
    """Which grid nodes a probe of this radius can reach from a
    percolating channel, with the Voronoi network supplying the
    connections the grid cannot resolve.

    Why (round 55). The grid decides that two neighbouring nodes are
    connected when both have room for the probe. Near a window only
    slightly wider than the probe, no pair of nodes may happen to sit
    close enough to the window's centre to see it as open: UiO-66, with
    an exact PLD of 3.793 A, came out with no N2 (3.64 A) channel at all
    on its 0.35 A grid, and its whole surface was counted as pocket.
    Windows are exactly what the Voronoi network describes well - its
    edges run through them and their capacities are the (refined) window
    radii that give the Zeo++-matching PLD.

    So the percolation is done on one graph holding both:
      - grid nodes and grid edges, as before (both ends have room);
      - Voronoi vertices with room for the probe, and the edges between
        them whose capacity - refined, as for the PLD, when it is within
        REFINEMENT_WINDOW of the probe radius - admits it;
      - links from each such vertex to the grid nodes within
        (clearance - probe radius) of it. The clearance function is
        1-Lipschitz, so the straight segment between them never comes
        closer to an atom surface than the probe radius: every link is
        a path the probe can really take. (All nodes in that ball have
        room for the probe and, once it is wider than a grid spacing or
        so, are already joined by grid edges, so one link per vertex
        image does.)
    Everything is joined in the same lattice-offset union-find the grid
    uses, so percolation and dimensionality come out as before; the
    network can only add connections the probe really has, never
    remove any.
    """
    from scipy.spatial import cKDTree

    from thevoid_core.services.pore_analysis import (
        build_grid_edges,
        dimensionality_from_vectors,
    )

    lattice = np.asarray(lattice, dtype=float)
    r = float(probe_radius)
    n_grid = int(np.prod(grid_shape))

    # Round 56: the same refined network the PLD uses - a window counts
    # as open here exactly when the PLD would say the probe passes it.
    # Its routes are only checked where one could change the answer:
    # a route between two points the grid and network already join, the
    # same way round, adds nothing, and checking it can cost a
    # Nelder-Mead search or two.
    refined = RefinedNetwork(network, clearance, r - REFINEMENT_WINDOW, ceiling=r)
    n_nodes = refined.n_nodes

    if grid_union_find is not None:
        # Extend the grid sweep's own union-find (grid edges at this probe
        # radius already joined) instead of re-joining ~650,000 edges.
        union_find = grid_union_find
        union_find.parent.extend(range(n_grid, n_grid + n_nodes))
        union_find.rank.extend([0] * n_nodes)
        union_find.offset.extend([(0, 0, 0)] * n_nodes)
    else:
        union_find = _OffsetUnionFind.for_n_nodes(n_grid + n_nodes)
        for a, b, offset in build_grid_edges(tuple(grid_shape)):
            if distance_field[a] >= r and distance_field[b] >= r:
                union_find.union(a, b, offset)

    edges, capacities = refined.edges_and_capacities()
    n_fixed = len(network.edges) + len(refined.link_edges)
    for index, ((a, b, offset), capacity) in enumerate(zip(edges, capacities)):
        if capacity >= r and (index < n_fixed or refined.verified[index - n_fixed]):
            union_find.union(n_grid + a, n_grid + b, offset)

    free = np.flatnonzero(distance_field >= r)
    links = 0
    tree = cKDTree(grid_frac[free] @ lattice) if len(free) else None
    spacing = float(np.max(np.linalg.norm(lattice, axis=1) / np.asarray(grid_shape)))
    shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
    refined_vertex = {node: vertex for vertex, node in refined.node.items()}

    def link_to_grid(nodes):
        """Join each node's images to the free grid nodes within its
        clearance minus the probe radius (a straight step there is clear)."""
        nonlocal links
        if tree is None:
            return
        for v in nodes:
            position, value = refined.node_position(v)
            reach = value - r
            if reach <= 0:
                continue
            centres = position[None, :] + shifts @ lattice
            distances, nearest = tree.query(centres, k=1, distance_upper_bound=reach)
            for image in np.flatnonzero(distances <= reach):
                if reach < 1.5 * spacing:
                    # A ball this small may hold grid nodes that are not
                    # face-neighbours of each other: link them all.
                    targets = tree.query_ball_point(centres[image], reach)
                else:
                    # Every node inside the ball has room for the probe,
                    # and a discrete ball this large is 6-connected, so
                    # the grid has already joined them all.
                    targets = [int(nearest[image])]
                for hit in targets:
                    # The node image sits `shift` cells from the grid
                    # node's own cell.
                    union_find.union(int(free[hit]), n_grid + v,
                                     tuple(int(x) for x in shifts[image]))
                    links += 1

    # A refined vertex still waiting for its local maximum may yet move:
    # it is linked once it has settled.
    link_to_grid([v for v in range(n_nodes)
                  if refined_vertex.get(v) not in refined.upgradable])

    def proxy(node):
        """The union-find node standing for refined node `node` (its
        index minus n_grid), and the cell it sits in relative to that
        node. A refined vertex not joined to anything yet is represented
        by the free grid node nearest to it, within two grid spacings -
        where the route would lead, if it were checked. Used only to
        decide whether checking is worth it."""
        own = n_grid + node
        root, _ = union_find.find(own)
        if root != own or tree is None:
            return own, (0, 0, 0)
        position = refined.node_position(node)[0]
        distances, nearest = tree.query(position[None, :] + shifts @ lattice, k=1,
                                        distance_upper_bound=2.0 * spacing)
        image = int(np.argmin(distances))
        if not np.isfinite(distances[image]):
            return own, (0, 0, 0)
        # The vertex image in cell shifts[image] lies next to the grid
        # node in cell 0 - the same relation link_to_grid records with
        # union(grid node, vertex, shifts[image]).
        return int(free[int(nearest[image])]), tuple(int(x) for x in shifts[image])

    def adds_something(a, b, offset):
        node_a, cell_a = proxy(a - n_grid)
        node_b, cell_b = proxy(b - n_grid)
        root_a, offset_a = union_find.find(node_a)
        root_b, offset_b = union_find.find(node_b)
        if root_a != root_b:
            return True
        offset_a = tuple(x + y for x, y in zip(offset_a, cell_a))
        offset_b = tuple(x + y for x, y in zip(offset_b, cell_b))
        implied = tuple(x - y + z for x, y, z in zip(offset_a, offset_b, offset))
        if implied == (0, 0, 0):
            return False
        known = union_find.percolation_vectors_for_root(root_a)
        return dimensionality_from_vectors(list(known) + [implied]) > \
            dimensionality_from_vectors(list(known))

    # A route that adds nothing now never will (the union-find only
    # grows), so it is set aside for good.
    redundant = set()
    while True:
        candidates = sorted(
            (k for k, value in enumerate(refined.values)
             if value >= r and not refined.verified[k] and k not in redundant),
            key=lambda i: -refined.values[i])
        if not candidates:
            break
        for k in candidates:
            if refined.verified[k] or refined.values[k] < r:
                continue
            a, b, offset = refined.route_edges[k]
            if not adds_something(n_grid + a, n_grid + b, offset):
                redundant.add(k)
                continue
            before = set(refined.upgradable)
            refined._verify(k, r)
            settled = [refined.node[v] for v in before - refined.upgradable]
            for node in settled:
                link = refined.link_of[refined_vertex[node]]
                if refined.link_caps[link] >= r:
                    la, lb, loffset = refined.link_edges[link]
                    union_find.union(n_grid + la, n_grid + lb, loffset)
            link_to_grid(settled)
            if refined.values[k] >= r:
                if refined.verified[k]:
                    union_find.union(n_grid + a, n_grid + b, offset)
                else:
                    # Nothing better was found: do not try it again.
                    refined.verified[k] = True
                    refined.values[k] = refined._through_original(k)

    link_to_grid([refined.node[v] for v in refined.upgradable])
    node_positions = np.array([refined.node_position(v)[0] for v in range(n_nodes)])
    node_clearance = np.array([refined.node_position(v)[1] for v in range(n_nodes)])

    def classify():
        accessible = np.zeros(n_grid, dtype=bool)
        roots = {}
        for node in free:
            root, _ = union_find.find(int(node))
            if root not in roots:
                roots[root] = bool(union_find.percolation_vectors_for_root(root))
            if roots[root]:
                accessible[node] = True
        return accessible, roots

    accessible, roots = classify()

    # Free nodes still cut off may sit in a small cavity whose own
    # Voronoi vertex is reachable (through windows the network resolves)
    # but whose free blob is too thin for any grid node to lie within
    # (clearance - probe) of that vertex - HKUST-1's small pockets for N2
    # are exactly this. Join such a blob to a reachable vertex when the
    # straight segment between them is clear, checked with the exact
    # clearance function (see _segment_is_clear); one success per blob
    # is enough, since the blob is already joined internally.
    stranded = free[~accessible[free]]
    reachable = [
        v for v in range(n_nodes)
        if node_clearance[v] >= r
        and union_find.percolation_vectors_for_root(union_find.find(n_grid + v)[0])
    ]

    if len(stranded) and reachable:
        reachable = np.array(reachable)
        shifts = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])
        images = (node_positions[reachable][:, None, :] + (shifts @ lattice)[None, :, :]).reshape(-1, 3)
        image_vertex = np.repeat(reachable, len(shifts))
        image_shift = np.tile(np.arange(len(shifts)), len(reachable))
        vertex_tree = cKDTree(images)
        node_points = grid_frac[stranded] @ lattice
        distances, nearest = vertex_tree.query(node_points, k=1, distance_upper_bound=STRANDED_REACH)
        by_blob: dict = {}
        for k in np.argsort(distances):
            if not np.isfinite(distances[k]):
                break
            blob, _ = union_find.find(int(stranded[k]))
            attempts = by_blob.setdefault(blob, 0)
            if attempts < 0 or attempts >= STRANDED_ATTEMPTS:
                continue  # already joined, or tried enough of its nodes
            by_blob[blob] = attempts + 1
            image = int(nearest[k])
            if _segment_is_clear(clearance, node_points[k], images[image], r):
                union_find.union(
                    int(stranded[k]), n_grid + int(image_vertex[image]),
                    tuple(int(x) for x in shifts[image_shift[image]]),
                )
                links += 1
                by_blob[blob] = -1

        accessible, roots = classify()

    dimensionality = {
        root: dimensionality_from_vectors(union_find.percolation_vectors_for_root(root))
        for root, percolates in roots.items() if percolates
    }

    return NetworkAccessibility(accessible, dimensionality, links)
