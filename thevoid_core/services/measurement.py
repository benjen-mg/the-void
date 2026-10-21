from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MeasurementResult:
    """The outcome of a distance, angle, or torsion measurement.

    kind is "distance", "angle", or "torsion". atom_indices are indices into
    whatever RenderModel.atoms list the measurement was taken against,
    in the order they were picked - for an angle, the middle index is
    the vertex.

    site_indices are the same picks expressed as base-structure site
    indices instead (RenderAtom.site_index) - the index a supercell or
    a completed periodic image was drawn *for*, not a position in the
    flat render list. A caller that wants to look up an atom's label or
    element needs this, not atom_indices: the same site can appear at
    several different render-list positions (once per drawn periodic
    image), and only the site index is meaningful outside the render
    model that happened to produce a particular view.
    """

    kind: str
    value: float
    atom_indices: tuple[int, ...]
    site_indices: tuple[int, ...] | None = None

    @property
    def label(self) -> str:
        if self.kind == "distance":
            return f"{self.value:.3f} \u00c5"

        return f"{self.value:.1f}\u00b0"


def distance(position_a, position_b) -> float:
    """Straight-line distance between two points, in whatever unit the
    positions are already in (angstrom, for every caller in this
    application).
    """
    return math.dist(position_a, position_b)


def nearest_periodic_image(
    anchor_position,
    base_position,
    lattice_matrix,
    *,
    search_range: int = 2,
):
    """Minimum-image convention: which periodic translation of
    base_position lands closest to anchor_position.

    Standard crystallographic behaviour for a distance measurement
    between two sites, and what "nearest-image convention" refers to as
    a term of art: the two atoms a person actually clicked on may be
    whichever periodic copies happened to be drawn (in a supercell, or
    because molecule completion pulled a particular image across a cell
    face) - not necessarily the closest copies to each other. This
    searches every integer lattice translation of base_position within
    search_range cells in every direction and returns whichever one is
    genuinely nearest to anchor_position, which is the physically
    meaningful "how far apart are these two atoms" answer regardless of
    which specific copies were on screen.

    Anchored on one fixed point (anchor_position) rather than searching
    both atoms' images independently: given two specific clicks, the
    first click's position is kept exactly where the person put it, and
    only the second atom's image is chosen to be nearest to that fixed
    point. This keeps the measurement line's other endpoint fixed at an
    actual clicked position rather than silently relocating both ends
    to some other pair of images the person didn't click.

    search_range=2 (125 candidate images) comfortably covers any
    reasonable pick; a atom more than two cells away from its nearest
    image of the other site would be a very unusual thing to click
    "as a pair" in the first place.

    Returns (position, image, distance):
      position  - the Cartesian position of the chosen image
      image     - the (i, j, k) integer translation applied, in whole
                  lattice-vector units
      distance  - the resulting distance from anchor_position
    """
    anchor = np.asarray(anchor_position, dtype=float)
    base = np.asarray(base_position, dtype=float)
    lattice = np.asarray(lattice_matrix, dtype=float)

    offsets = range(-search_range, search_range + 1)

    best_position = None
    best_image = None
    best_distance = math.inf

    for i, j, k in itertools.product(offsets, offsets, offsets):
        translation = np.array([i, j, k], dtype=float) @ lattice
        candidate = base + translation

        current = float(np.linalg.norm(candidate - anchor))

        if current < best_distance:
            best_distance = current
            best_position = tuple(float(v) for v in candidate)
            best_image = (i, j, k)

    return best_position, best_image, best_distance


def angle(position_a, position_vertex, position_c) -> float:
    """Angle in degrees at position_vertex, between the rays to
    position_a and position_c.

    Standard bond-angle convention: the middle argument is the vertex,
    not the first or last - a caller measuring the O-C-O angle in CO2
    passes (O, C, O), not (C, O, O).
    """
    vector_a = tuple(a - v for a, v in zip(position_a, position_vertex))
    vector_c = tuple(c - v for c, v in zip(position_c, position_vertex))

    magnitude_a = math.sqrt(sum(component**2 for component in vector_a))
    magnitude_c = math.sqrt(sum(component**2 for component in vector_c))

    if magnitude_a < 1e-9 or magnitude_c < 1e-9:
        raise ValueError(
            "Cannot measure an angle where the vertex coincides with "
            "one of the other two points."
        )

    dot = sum(a * c for a, c in zip(vector_a, vector_c))
    cosine = dot / (magnitude_a * magnitude_c)

    # Clamp against floating-point drift pushing |cosine| a hair past 1,
    # which would otherwise make acos raise for a genuinely straight
    # (180 degree) or coincident-direction (0 degree) angle.
    cosine = max(-1.0, min(1.0, cosine))

    return math.degrees(math.acos(cosine))


def dihedral(position_a, position_b, position_c, position_d) -> float:
    """Signed torsion (dihedral) angle A-B-C-D in degrees, in (-180, 180].

    The angle between the plane through A, B, C and the plane through
    B, C, D, looking down the B->C bond: positive when, viewed along
    B->C, A must turn clockwise to eclipse D - the IUPAC sign
    convention used by Mercury, Olex2, and PLATON, so values can be
    compared directly against their output.

    Raises ValueError when three consecutive points are collinear
    (A-B-C or B-C-D a straight line), where the torsion is undefined.
    """
    a = np.asarray(position_a, dtype=float)
    b = np.asarray(position_b, dtype=float)
    c = np.asarray(position_c, dtype=float)
    d = np.asarray(position_d, dtype=float)

    b1 = b - a
    b2 = c - b
    b3 = d - c

    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)

    norm_b2 = np.linalg.norm(b2)

    if (
        norm_b2 < 1e-9
        or np.linalg.norm(n1) < 1e-9
        or np.linalg.norm(n2) < 1e-9
    ):
        raise ValueError(
            "Cannot measure a torsion angle when three consecutive "
            "atoms are collinear."
        )

    m1 = np.cross(n1, b2 / norm_b2)

    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))

    value = -math.degrees(math.atan2(y, x))

    # Keep the result in (-180, 180]: a value at (or a rounding hair
    # past) -180 is the same anti-periplanar torsion as +180, and the
    # wrap itself must not push it a hair past +180 either.
    if value <= -180.0 + 1e-9:
        value += 360.0

    return min(value, 180.0)


def nearest_atom_index(
    picked_point,
    atom_positions,
    *,
    max_distance: float | None = None,
) -> int | None:
    """Find the atom whose centre is closest to a picked point.

    Picking in the 3D view returns a point on the surface of whichever
    glyphed sphere mesh was clicked - not the atom's centre, since many
    atoms of one element share a single batched mesh. This snaps that
    surface point back to the atom it actually belongs to.

    Returns None if atom_positions is empty, or if max_distance is
    given and even the nearest atom is farther than that (a click that
    landed on empty space or a bond rather than an atom).
    """
    if not atom_positions:
        return None

    best_index = None
    best_distance = math.inf

    for index, position in enumerate(atom_positions):
        current = distance(picked_point, position)

        if current < best_distance:
            best_distance = current
            best_index = index

    if max_distance is not None and best_distance > max_distance:
        return None

    return best_index
