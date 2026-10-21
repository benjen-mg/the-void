"""Geometric pore size distribution (PSD).

The definition is Gelb and Gubbins' (Langmuir 15, 305 (1999)), the one
PoreBlazer and most MOF papers use: V(D) is the volume that can be
covered by spheres of diameter D or more, each placed entirely in the
void (clear of every atom's van der Waals surface). V(0) is the whole
geometric void volume and V(D) = 0 above the largest cavity diameter.
The distribution is -dV/dD, normalised by V(0), so it integrates to 1
and has units of 1/A.

On a grid this reduces to a morphological opening. Let S(r) be the
grid points where a sphere of radius r fits, meaning the clearance to
the nearest atom surface is at least r. A void point is coverable at
radius r if it lies within r of some point of S(r). The sets shrink as
r grows, so each step only re-examines the points still covered at the
previous one.

Uses the same cell-centred grid and clearance field as pore analysis,
so the two agree. V(0) equals pore analysis's void volume at probe
radius 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from thevoid_core.services.pore_analysis import (
    build_periodic_atoms,
    compute_distance_field,
    major_occupancy_structure,
)
from thevoid_core.services.pore_geometry import periodic_images

DEFAULT_BIN_WIDTH = 0.2  # angstrom, in diameter


@dataclass(frozen=True, slots=True)
class PoreSizeDistribution:
    #: Diameters (A) at which V(D) is evaluated: 0, w, 2w, ...
    diameters: np.ndarray
    #: V(D): volume (A^3 per cell) coverable by spheres of diameter >= D.
    cumulative_volume: np.ndarray
    #: Bin centres (A) for `distribution`, between consecutive diameters.
    bin_centers: np.ndarray
    #: -dV/dD / V(0) per bin (1/A); integrates to 1 over the bins.
    distribution: np.ndarray
    #: V(0): total geometric void volume (A^3 per cell).
    pore_volume: float
    cell_volume: float
    bin_width: float
    grid_shape: tuple[int, int, int]

    @property
    def dominant_diameter(self) -> float | None:
        """Bin centre of the tallest peak, or None if there is no void."""
        if self.pore_volume <= 0 or not len(self.distribution):
            return None

        return float(self.bin_centers[int(np.argmax(self.distribution))])

    @property
    def void_fraction(self) -> float:
        return self.pore_volume / self.cell_volume if self.cell_volume else 0.0

    def rows(self):
        """(bin centre, distribution, V at the bin's lower edge) rows, for CSV."""
        return [
            (float(c), float(p), float(v))
            for c, p, v in zip(
                self.bin_centers, self.distribution, self.cumulative_volume[:-1]
            )
        ]


def compute_pore_size_distribution(
    structure,
    *,
    radius_provider,
    bin_width: float = DEFAULT_BIN_WIDTH,
    grid_spacing: float = 0.3,
    max_grid_points_per_axis: int = 60,
) -> PoreSizeDistribution:
    from scipy.spatial import cKDTree

    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive: {bin_width}")

    structure = major_occupancy_structure(structure)

    lattice = np.asarray(structure.lattice.matrix, dtype=float)
    lengths = np.linalg.norm(lattice, axis=1)
    cell_volume = float(abs(np.linalg.det(lattice)))

    shape = tuple(
        int(min(max_grid_points_per_axis, max(4, round(length / grid_spacing))))
        for length in lengths
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
    grid_points = grid_frac @ lattice
    n_total = len(grid_points)
    point_volume = cell_volume / n_total

    if len(structure) == 0:
        # No atoms: nothing bounds the pores.
        raise ValueError("A pore size distribution needs at least one atom.")

    symbols = [list(site.species.keys())[0].symbol for site in structure.sites]
    radii = np.array([radius_provider.get_vdw_radius(s) for s in symbols])
    positions, image_radii = build_periodic_atoms(
        structure, radii, float(radii.max()) + 2.0
    )
    clearance = compute_distance_field(grid_points, positions, image_radii)

    void = clearance > 0
    pore_volume = float(void.sum()) * point_volume

    max_radius = float(clearance.max()) if void.any() else 0.0
    n_steps = int(np.ceil(2.0 * max_radius / bin_width)) + 1
    diameters = np.arange(n_steps + 1) * bin_width

    cumulative = np.zeros(len(diameters))
    covered = np.nonzero(void)[0]
    cumulative[0] = pore_volume

    for step in range(1, len(diameters)):
        radius = diameters[step] / 2.0
        centres = np.nonzero(clearance >= radius)[0]

        if len(centres) == 0 or len(covered) == 0:
            covered = covered[:0]
            continue

        images, _r, _base, _off = periodic_images(
            lattice, grid_frac[centres], np.zeros(len(centres)), radius + 1e-9
        )
        tree = cKDTree(images)
        distance, _ = tree.query(
            grid_points[covered], k=1, distance_upper_bound=radius + 1e-9
        )
        covered = covered[np.isfinite(distance)]
        cumulative[step] = len(covered) * point_volume

    bin_centers = (diameters[:-1] + diameters[1:]) / 2.0

    if pore_volume > 0:
        distribution = (cumulative[:-1] - cumulative[1:]) / (
            bin_width * pore_volume
        )
    else:
        distribution = np.zeros(len(bin_centers))

    return PoreSizeDistribution(
        diameters=diameters,
        cumulative_volume=cumulative,
        bin_centers=bin_centers,
        distribution=distribution,
        pore_volume=pore_volume,
        cell_volume=cell_volume,
        bin_width=float(bin_width),
        grid_shape=shape,
    )
