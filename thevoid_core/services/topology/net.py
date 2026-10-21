"""Periodic nets: quotient graphs, their lattices, and an exact canonical
form (round 53).

A periodic net is stored the way crystallographers and Systre store
one: its *quotient graph* - one vertex per vertex orbit under the
translation lattice, and one edge per edge orbit, each carrying the
integer lattice translation from its first to its second end. Two
quotient graphs can describe the same net in very different ways (a
different cell, a supercell, a relabelling), so identifying a net means
bringing it to a form that does not depend on any of those choices.

The method is Systre's (Delgado-Friedrichs & O'Keeffe, Acta Cryst.
A59, 351 (2003)), implemented here from its description with exact
rational arithmetic throughout:

1. Restrict to one connected component and re-express its translations
   in a basis of the lattice that component actually spans (an
   interpenetrated net's components each span a sublattice; a layer
   spans a rank-2 lattice).
2. Barycentric placement: put every vertex at the average of its
   neighbours. For a connected periodic net this placement is unique up
   to translation, and every automorphism of the net acts on it as an
   affine map - which is what makes geometry usable as a canonical
   reference.
3. Stability: if two vertices land on the same point the placement
   cannot tell them apart and the method does not apply. Such nets are
   reported as unstable rather than keyed.
4. Minimal cell: translations of the placement that map the net onto
   itself but are not lattice vectors mean the input cell was a
   supercell; the net is reduced to its true primitive cell.
5. Canonical key: for every vertex v and every ordered choice of d
   edges at v whose vectors are linearly independent, express the net
   in the coordinate system those edges define, number vertices in
   breadth-first order (neighbours sorted by edge vector), and list the
   edges. The lexicographically smallest such listing is the key. Two
   nets are isomorphic exactly when their keys are equal.

Nothing here knows about atoms; services/topology/deconstruct.py turns
a structure into a net, and services/topology/library.py names it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations

Vector = tuple  # of int or Fraction


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _neg(a):
    return tuple(-x for x in a)


def _is_zero(a):
    return all(x == 0 for x in a)


# ----------------------------------------------------------------------
# The quotient graph
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PeriodicNet:
    """A quotient graph: vertices 0..n-1, and undirected edges
    (i, j, shift) meaning vertex i in cell 0 is joined to vertex j in
    cell `shift`. Each undirected edge is listed once."""

    n_vertices: int
    edges: tuple
    dimension: int

    def adjacency(self) -> list[list[tuple[int, Vector]]]:
        """Directed neighbour lists: both directions of every edge, and
        both directions of a loop (a vertex bonded to its own image)."""
        adjacency: list[list] = [[] for _ in range(self.n_vertices)]

        for i, j, shift in self.edges:
            adjacency[i].append((j, tuple(shift)))
            adjacency[j].append((i, _neg(shift)))

        return adjacency

    def degrees(self) -> list[int]:
        return [len(neighbours) for neighbours in self.adjacency()]

    @staticmethod
    def build(n_vertices: int, edges, dimension: int) -> "PeriodicNet":
        """Normalise and de-duplicate: each undirected edge stored once,
        oriented i < j (or, for a loop, with a positive shift)."""
        seen = set()
        normalised = []

        for i, j, shift in edges:
            shift = tuple(int(s) for s in shift)

            if i > j or (i == j and shift < _neg(shift)):
                i, j, shift = j, i, _neg(shift)

            if i == j and _is_zero(shift):
                continue  # a vertex joined to itself in the same cell

            key = (i, j, shift)

            if key not in seen:
                seen.add(key)
                normalised.append(key)

        normalised.sort()

        return PeriodicNet(n_vertices, tuple(normalised), dimension)


def parse_systre_key(text: str) -> PeriodicNet:
    """A Systre key ("3 1 2 0 0 0 1 3 0 0 0 ...": dimension, then
    1-based `i j t1..td` per edge), as stored in Systre's .arc files."""
    tokens = [int(t) for t in text.split()]
    dimension = tokens[0]
    body = tokens[1:]
    stride = 2 + dimension

    if len(body) % stride:
        raise ValueError("Malformed Systre key.")

    edges = []
    n = 0

    for k in range(0, len(body), stride):
        i, j = body[k] - 1, body[k + 1] - 1
        edges.append((i, j, tuple(body[k + 2:k + stride])))
        n = max(n, i + 1, j + 1)

    return PeriodicNet.build(n, edges, dimension)


# ----------------------------------------------------------------------
# Exact linear algebra
# ----------------------------------------------------------------------


def solve_rational(matrix, rhs):
    """Solve matrix @ X = rhs exactly (Gauss-Jordan over Fractions).
    matrix is n x n, rhs n x m. Raises ValueError if singular."""
    n = len(matrix)
    m = len(rhs[0]) if rhs else 0
    a = [
        [Fraction(v) for v in matrix[r]] + [Fraction(v) for v in rhs[r]]
        for r in range(n)
    ]

    for col in range(n):
        pivot = next((r for r in range(col, n) if a[r][col] != 0), None)

        if pivot is None:
            raise ValueError("Singular system.")

        a[col], a[pivot] = a[pivot], a[col]
        inverse = 1 / a[col][col]
        row = [v * inverse for v in a[col]]
        a[col] = row

        for r in range(n):
            if r != col and a[r][col] != 0:
                factor = a[r][col]
                a[r] = [x - factor * y for x, y in zip(a[r], row)]

    return [a[r][n:n + m] for r in range(n)]


def integer_row_basis(vectors) -> list[list[int]]:
    """A Z-basis of the lattice spanned by integer vectors: row echelon
    form by Euclid's algorithm down each column (a Hermite-style
    reduction; unimodular row operations only, so the span is kept)."""
    rows = [list(int(x) for x in v) for v in vectors if any(v)]

    if not rows:
        return []

    d = len(rows[0])
    top = 0

    for col in range(d):
        while True:
            live = [r for r in range(top, len(rows)) if rows[r][col] != 0]
            if len(live) <= 1:
                break
            pivot = min(live, key=lambda r: abs(rows[r][col]))
            for r in live:
                if r != pivot:
                    q = rows[r][col] // rows[pivot][col]
                    rows[r] = [x - q * y for x, y in zip(rows[r], rows[pivot])]

        live = [r for r in range(top, len(rows)) if rows[r][col] != 0]

        if live:
            rows[top], rows[live[0]] = rows[live[0]], rows[top]
            top += 1

        rows = rows[:top] + [r for r in rows[top:] if any(r)]

    return rows[:top]


def coordinates_in_basis(vector, basis) -> tuple:
    """Coefficients c with c @ basis == vector (basis rows linearly
    independent, possibly fewer than the ambient dimension)."""
    r = len(basis)
    d = len(vector)
    # Normal equations (B B^T) c = B v, exact.
    gram = [[sum(basis[i][k] * basis[j][k] for k in range(d)) for j in range(r)]
            for i in range(r)]
    rhs = [[sum(basis[i][k] * vector[k] for k in range(d))] for i in range(r)]
    solution = solve_rational(gram, rhs)
    coefficients = tuple(solution[i][0] for i in range(r))

    reconstructed = tuple(
        sum(coefficients[i] * basis[i][k] for i in range(r)) for k in range(d)
    )

    if reconstructed != tuple(Fraction(v) for v in vector):
        raise ValueError("Vector is not in the span of the basis.")

    return coefficients


def _inverse(matrix):
    n = len(matrix)
    identity = [[int(i == j) for j in range(n)] for i in range(n)]
    return solve_rational(matrix, identity)


def _row_times(vector, matrix):
    """Row vector times matrix."""
    return tuple(
        sum(vector[k] * matrix[k][j] for k in range(len(vector)))
        for j in range(len(matrix[0]))
    )


def _rank(vectors) -> int:
    rows = [[Fraction(v) for v in vec] for vec in vectors]
    rank = 0
    cols = len(rows[0]) if rows else 0

    for col in range(cols):
        pivot = next((r for r in range(rank, len(rows)) if rows[r][col] != 0), None)

        if pivot is None:
            continue

        rows[rank], rows[pivot] = rows[pivot], rows[rank]

        for r in range(len(rows)):
            if r != rank and rows[r][col] != 0:
                factor = rows[r][col] / rows[rank][col]
                rows[r] = [x - factor * y for x, y in zip(rows[r], rows[rank])]

        rank += 1

    return rank


# ----------------------------------------------------------------------
# Components and their lattices
# ----------------------------------------------------------------------


def connected_components(net: PeriodicNet) -> list[PeriodicNet]:
    """Each connected component as a net of its own, with translations
    re-expressed in a basis of the lattice that component spans. Its
    dimension is that lattice's rank (0 for a finite cluster)."""
    adjacency = net.adjacency()
    seen = [False] * net.n_vertices
    result = []

    for start in range(net.n_vertices):
        if seen[start]:
            continue

        # Tree lifts: each vertex's cell offset along a spanning tree.
        offset = {start: tuple([0] * net.dimension)}
        order = [start]
        seen[start] = True
        queue = deque([start])

        while queue:
            u = queue.popleft()
            for w, shift in adjacency[u]:
                if not seen[w]:
                    seen[w] = True
                    offset[w] = _add(offset[u], shift)
                    order.append(w)
                    queue.append(w)

        members = set(order)
        cycle_vectors = []
        component_edges = []

        for i, j, shift in net.edges:
            if i in members:
                vector = _sub(_add(offset[i], shift), offset[j])
                component_edges.append((i, j, vector))
                if not _is_zero(vector):
                    cycle_vectors.append(vector)

        basis = integer_row_basis(cycle_vectors)
        index = {v: k for k, v in enumerate(order)}
        edges = []

        for i, j, vector in component_edges:
            coordinates = (
                coordinates_in_basis(vector, basis) if basis else ()
            )
            if any(c.denominator != 1 for c in coordinates):
                raise ValueError("Cycle vector outside its own lattice.")
            edges.append((index[i], index[j], tuple(int(c) for c in coordinates)))

        component = PeriodicNet.build(len(order), edges, len(basis))
        result.append((component, order))

    return result


# ----------------------------------------------------------------------
# Invariants
# ----------------------------------------------------------------------


def coordination_sequences(net: PeriodicNet, shells: int = 10) -> list[tuple[int, ...]]:
    """For each vertex: how many vertices lie 1, 2, ..., shells bonds
    away in the infinite net (the standard topological invariant)."""
    adjacency = net.adjacency()
    result = []

    for start in range(net.n_vertices):
        origin = (start, tuple([0] * net.dimension))
        previous = {origin}
        current = {origin}
        sequence = []

        for _ in range(shells):
            following = set()
            for vertex, cell in current:
                for w, shift in adjacency[vertex]:
                    node = (w, _add(cell, shift))
                    if node not in previous and node not in current:
                        following.add(node)
            previous, current = current, following
            sequence.append(len(current))

        result.append(tuple(sequence))

    return result


# ----------------------------------------------------------------------
# Barycentric placement, stability, minimal cell
# ----------------------------------------------------------------------


class UnstableNetError(ValueError):
    """Two vertices collide in the barycentric placement."""

    def __init__(self, message, collisions=()):
        super().__init__(message)
        #: Groups of vertex indices sharing a position.
        self.collisions = [list(group) for group in collisions]


def barycentric_placement(net: PeriodicNet) -> list[tuple]:
    """Exact barycentric positions (fractional, in the net's lattice
    basis), vertex 0 at the origin. Requires a connected net."""
    n, d = net.n_vertices, net.dimension

    if n == 1:
        return [tuple([Fraction(0)] * d)]

    adjacency = net.adjacency()
    # Unknowns: positions of vertices 1..n-1. Equation for vertex i:
    # sum_j (p_j + t - p_i) = 0.
    matrix = [[0] * (n - 1) for _ in range(n - 1)]
    rhs = [[0] * d for _ in range(n - 1)]

    for i in range(1, n):
        row = i - 1
        for j, shift in adjacency[i]:
            matrix[row][row] -= 1
            if j != 0:
                matrix[row][j - 1] += 1
            for k in range(d):
                rhs[row][k] -= shift[k]

    solution = _solve_fast_then_verify(matrix, rhs, adjacency)

    if solution is None:
        if len(matrix) > EXACT_SOLVE_LIMIT:
            raise ValueError(
                f"The net has {len(matrix) + 1} vertices per cell and its barycentric "
                "placement has no small denominators; the exact solve is too slow "
                f"above {EXACT_SOLVE_LIMIT + 1} vertices.")
        solution = solve_rational(matrix, rhs)

    return [tuple([Fraction(0)] * d)] + [tuple(row) for row in solution]


#: Round 70: largest system the exact (Fraction) solve is tried on when
#: the fast one fails. It is O(n^3) in Fractions whose size grows with
#: n: a minute at 192 vertices, and hours for the 384-vertex all-node net
#: of a Y-K-O framework whose three-periodic inorganic SBU is split at
#: its 96 metals (CoRE SI 2019[YK][nan]3[ION]1). Such a net is reported
#: as too complex instead.
EXACT_SOLVE_LIMIT = 200


def _solve_fast_then_verify(matrix, rhs, adjacency):
    """Floating-point solve, rounded to nearby rationals, then checked
    exactly against the barycentric equations. The exact Gauss-Jordan
    solve is O(n^3) in Fractions - a minute for a 192-vertex cell such
    as ZIF-20's - while barycentric coordinates are rationals with small
    denominators for every net met so far. The check makes the shortcut
    safe: a rounded answer that does not satisfy every equation exactly
    is thrown away (None) and the exact solve runs instead."""
    try:
        import numpy as np

        approx = np.linalg.solve(np.array(matrix, dtype=float), np.array(rhs, dtype=float))
    except Exception:
        return None

    n = len(matrix) + 1
    d = len(rhs[0])

    for limit in (1000, 100_000):
        rows = [tuple(Fraction(float(x)).limit_denominator(limit) for x in row) for row in approx]
        positions = [tuple([Fraction(0)] * d)] + rows
        if all(
            all(
                sum((positions[j][k] + shift[k] - positions[i][k]) for j, shift in adjacency[i]) == 0
                for k in range(d)
            )
            for i in range(n)
        ):
            return [list(row) for row in rows]

    return None


def _mod1(vector):
    return tuple(x - (x.numerator // x.denominator) for x in vector)


def check_stable(positions) -> None:
    groups: dict = {}
    for index, position in enumerate(positions):
        groups.setdefault(_mod1(position), []).append(index)
    collisions = [g for g in groups.values() if len(g) > 1]
    if collisions:
        raise UnstableNetError(
            "Two vertices share a position in the barycentric placement "
            "(an unstable net); it cannot be keyed this way.",
            collisions,
        )


def _edge_vectors(net, positions):
    """For each vertex, its outgoing edges as (target, vector) where
    vector = p_target + shift - p_vertex."""
    return [
        [(j, _sub(_add(positions[j], shift), positions[i])) for j, shift in neighbours]
        for i, neighbours in enumerate(net.adjacency())
    ]


def minimal_net(net: PeriodicNet):
    """Reduce a connected, stable net to its primitive cell. Returns
    (minimal net, its barycentric positions, index of the input cell
    over the primitive one)."""
    positions = barycentric_placement(net)
    check_stable(positions)
    d = net.dimension

    if d == 0:
        return net, positions, 1

    vectors = _edge_vectors(net, positions)
    by_position = {_mod1(p): i for i, p in enumerate(positions)}
    signature = [sorted(v for _, v in vs) for vs in vectors]

    translations = []

    for j in range(1, net.n_vertices):
        tau = _sub(positions[j], positions[0])
        ok = True

        for i in range(net.n_vertices):
            image = by_position.get(_mod1(_add(positions[i], tau)))
            if image is None or signature[image] != signature[i]:
                ok = False
                break

        if ok:
            translations.append(_mod1(tau))

    if not translations:
        return net, positions, 1

    # Lattice generated by Z^d and the extra translations.
    denominator = 1
    for tau in translations:
        for x in tau:
            denominator = denominator * x.denominator // _gcd(denominator, x.denominator)

    generators = [[denominator * int(k == c) for c in range(d)] for k in range(d)]
    generators += [[int(x * denominator) for x in tau] for tau in translations]
    basis = [[Fraction(v, denominator) for v in row] for row in integer_row_basis(generators)]
    inverse = _inverse(basis)

    # Orbits of vertices under the translations.
    representative = {}
    reps = []
    for i in range(net.n_vertices):
        if i in representative:
            continue
        representative[i] = len(reps)
        reps.append(i)
        for tau in translations:
            other = by_position[_mod1(_add(positions[i], tau))]
            representative.setdefault(other, representative[i])

    new_positions = [_row_times(positions[r], inverse) for r in reps]
    new_positions = [_mod1(p) for p in new_positions]
    edges = []

    for new_i, old_i in enumerate(reps):
        for target, vector in vectors[old_i]:
            new_j = representative[target]
            end = _add(new_positions[new_i], _row_times(vector, inverse))
            shift = _sub(end, new_positions[new_j])
            if any(s.denominator != 1 for s in shift):
                raise ValueError("Minimal-cell reduction produced a fractional shift.")
            edges.append((new_i, new_j, tuple(int(s) for s in shift)))

    reduced = PeriodicNet.build(len(reps), edges, d)
    index = net.n_vertices // reduced.n_vertices

    return reduced, barycentric_placement(reduced), index


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


# ----------------------------------------------------------------------
# Canonical key
# ----------------------------------------------------------------------


class _Abort(Exception):
    pass


def canonical_key(net: PeriodicNet, *, max_candidates: int = 200_000) -> tuple:
    """The canonical form of a connected, stable, periodic net: equal
    for two nets exactly when they are isomorphic. Reduces to the
    minimal cell first, so a supercell or relabelled copy gives the
    same key."""
    reduced, positions, _ = minimal_net(net)
    d = reduced.dimension

    if d == 0:
        raise ValueError("A finite cluster has no periodic net.")

    vectors = _edge_vectors(reduced, positions)
    sequences = coordination_sequences(reduced, shells=6)
    # Only start from vertices with the smallest invariant: any
    # isomorphism maps these onto each other, so nothing is lost.
    best_invariant = min((len(vectors[v]), sequences[v]) for v in range(reduced.n_vertices))
    starts = [v for v in range(reduced.n_vertices)
              if (len(vectors[v]), sequences[v]) == best_invariant]

    best = None
    tried = 0

    for start in starts:
        candidates = _basis_candidates(reduced, vectors, start, d)
        for basis in permutations(candidates, d):
            if _rank(basis) < d:
                continue
            tried += 1
            if tried > max_candidates:
                raise ValueError("Too many candidate bases to key this net.")
            transform = _inverse([list(b) for b in basis])
            try:
                key = _traverse(reduced, vectors, start, transform, best)
            except _Abort:
                continue
            if best is None or key < best:
                best = key

    if best is None:
        raise ValueError("No basis of the lattice could be chosen at any vertex.")

    return (d, tuple(best))


def _basis_candidates(net, vectors, start, d):
    """Vectors from `start` to every vertex within k bonds, for the
    smallest k at which they span the lattice. The edges at a vertex
    usually span it (k = 1), but not always: a trigonal-planar vertex
    in a 3D net such as srs or tbo has three coplanar edges, so the
    second shell is needed. The set is defined by the net alone, so the
    choice stays canonical."""
    reached = {(start, tuple([Fraction(0)] * d))}
    frontier = list(reached)
    found: list = []

    for _ in range(4):
        following = []
        for vertex, position in frontier:
            for target, vector in vectors[vertex]:
                node = (target, _add(position, vector))
                if node not in reached:
                    reached.add(node)
                    following.append(node)
                    found.append(node[1])
        frontier = following
        unique = sorted(set(found))
        if _rank(unique) == d:
            return unique

    return sorted(set(found))


def _traverse(net, vectors, start, transform, best, *, exact=False):
    """One breadth-first numbering in the coordinates `transform`
    defines. Aborts as soon as the listing is certain to exceed `best`
    or, with exact=True, as soon as it differs from `best` at all."""
    number = {start: 0}
    lift = {start: tuple([Fraction(0)] * net.dimension)}
    order = [start]
    key = []
    position = 0
    head = 0

    while head < len(order):
        u = order[head]
        u_number = head
        head += 1
        outgoing = sorted(
            (_row_times(vector, transform), target) for target, vector in vectors[u]
        )

        for vector, target in outgoing:
            end = _add(lift[u], vector)
            if target not in number:
                number[target] = len(order)
                lift[target] = end
                order.append(target)
            t_number = number[target]
            shift = _sub(end, lift[target])

            if t_number < u_number:
                continue
            if t_number == u_number and not (shift > _neg(shift)):
                continue

            entry = (u_number, t_number, shift)

            if best is not None:
                if position >= len(best):
                    raise _Abort
                if exact:
                    if entry != best[position]:
                        raise _Abort
                elif entry > best[position]:
                    raise _Abort
                elif entry < best[position]:
                    best = None  # strictly better from here on
            key.append(entry)
            position += 1

    return key


def key_string(key: tuple) -> str:
    """A compact, printable form of a canonical key."""
    d, entries = key
    parts = [str(d)]
    for i, j, shift in entries:
        parts.append(f"{i} {j} " + " ".join(str(s) for s in shift))
    return " | ".join(parts)


# ----------------------------------------------------------------------
# Isomorphism
# ----------------------------------------------------------------------


@dataclass
class PreparedNet:
    """A connected, stable net reduced to its primitive cell, with
    everything the isomorphism test reuses."""

    net: PeriodicNet
    vectors: list
    sequences: list
    starts: list
    #: The traversal from the first start and first candidate basis:
    #: this net's fixed reference listing.
    reference: list | None = None

    @property
    def invariant(self) -> tuple:
        """Cheap isomorphism invariants: dimension, vertex count, and
        the multiset of (degree, coordination sequence)."""
        return (
            self.net.dimension,
            self.net.n_vertices,
            tuple(sorted(
                (len(self.vectors[v]), self.sequences[v])
                for v in range(self.net.n_vertices)
            )),
        )


def prepare(net: PeriodicNet, shells: int = 10) -> PreparedNet:
    reduced, positions, _ = minimal_net(net)
    if reduced.dimension == 0:
        raise ValueError("A finite cluster has no periodic net.")
    vectors = _edge_vectors(reduced, positions)
    sequences = coordination_sequences(reduced, shells=shells)
    best = min((len(vectors[v]), sequences[v]) for v in range(reduced.n_vertices))
    starts = [v for v in range(reduced.n_vertices)
              if (len(vectors[v]), sequences[v]) == best]
    return PreparedNet(reduced, vectors, sequences, starts)


def _candidate_transforms(prepared: PreparedNet, start: int):
    d = prepared.net.dimension
    candidates = _basis_candidates(prepared.net, prepared.vectors, start, d)
    for basis in permutations(candidates, d):
        if _rank(basis) == d:
            yield _inverse([list(b) for b in basis])


def reference_listing(prepared: PreparedNet) -> list:
    if prepared.reference is None:
        start = prepared.starts[0]
        transform = next(_candidate_transforms(prepared, start))
        prepared.reference = _traverse(
            prepared.net, prepared.vectors, start, transform, None
        )
    return prepared.reference


def isomorphic(a: PreparedNet, b: PreparedNet, *, max_candidates: int = 500_000) -> bool:
    """Exact isomorphism test for two prepared nets.

    b's listing from one fixed (start vertex, basis) is its reference.
    If the nets are isomorphic, the isomorphism carries that choice to
    one of a's candidate choices - the candidate sets are defined by
    the nets alone - and from there a's traversal reproduces the
    reference exactly. So a matches b iff some candidate of a
    reproduces b's reference. Each wrong candidate is abandoned at its
    first differing entry, which is what keeps this fast.
    """
    if a.invariant != b.invariant:
        return False

    target = reference_listing(b)
    tried = 0

    for start in a.starts:
        for transform in _candidate_transforms(a, start):
            tried += 1
            if tried > max_candidates:
                raise ValueError("Too many candidate bases to compare these nets.")
            try:
                listing = _traverse(a.net, a.vectors, start, transform, target, exact=True)
            except _Abort:
                continue
            if listing == target:
                return True

    return False
