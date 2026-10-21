"""Round 55: channel/pocket classification that sees windows the grid
cannot (pore_geometry.network_accessibility, pore_analysis.
network_assisted_sweep).

The reported bug: UiO-66 with N2 (probe radius 1.82 A) came out with
zero channel surface area - every square angstrom "pocket" - although
its PLD is 3.79 A, wider than the probe, and Zeo++ puts all 1106 A^2 in
channels. The grid decides two neighbouring nodes are connected when
both have room for the probe; near a window only slightly wider than
the probe, no pair of nodes sits close enough to its centre to see it
open. The fix joins the atoms' Voronoi network - whose edges pass
through window centres by construction - into the grid's percolation.

Checks, each with its answer known in advance:

* a simple-cubic lattice of Ar atoms, where the window and cavity radii
  are exact: a probe 0.03 A narrower than the window percolates (the
  grid alone says it does not), a probe 0.03 A wider does not (the
  network must not invent a connection);
* a bcc lattice shifted by -3e-10 of a cell, so that every Voronoi
  vertex sits a hair from a cell face: every network edge must have the
  exact length a*sqrt(2)/4. Before the fix, edges between vertices
  wrapped across a face were given the wrong cell offset, phantom
  edges across the cell that over-connected ZIF-8;
* real frameworks against Zeo++ (same Bondi radii), when RASPA2's
  structure files are available.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services import pore_geometry  # noqa: E402
from thevoid_core.services.pore_analysis import (  # noqa: E402
    analyze_pores,
    classify_void_grid,
    grid_shape_for,
    network_assisted_sweep,
    sweep_connectivity,
)
from thevoid_core.services.surface_area import analyze_surface_area  # noqa: E402

AR = 1.88  # Bondi
A = 10.2   # not a multiple of the 0.3 A grid spacing: window centres miss nodes
WINDOW = A / math.sqrt(2) - AR        # face-centre window radius
CAVITY = A * math.sqrt(3) / 2 - AR    # body-centre cavity radius

RADII = ElementRadiusProvider()


def _simple_cubic():
    return Structure(Lattice.cubic(A), ["Ar"], [[0, 0, 0]])


def _grid(structure, probe, use_network):
    return classify_void_grid(
        structure, np.array([AR] * len(structure)), probe_radius=probe,
        grid_shape=grid_shape_for(structure), use_network=use_network,
    )


# ----------------------------------------------------------------------
# Exact cases
# ----------------------------------------------------------------------


def test_a_window_just_wider_than_the_probe_is_open():
    """0.03 A of room: the grid misses it, the network does not."""
    structure = _simple_cubic()
    probe = WINDOW - 0.03

    grid_only = _grid(structure, probe, use_network=False)
    assert grid_only.accessibility_method == "grid"
    assert not grid_only.sweep.accessible.any()  # the bug, reproduced

    assisted = _grid(structure, probe, use_network=True)
    assert assisted.accessibility_method == "voronoi+grid"
    free = assisted.free_mask
    assert free.any() and assisted.sweep.accessible[free].all()
    assert set(assisted.sweep.dimensionality_by_root.values()) == {3}


def test_a_window_just_narrower_than_the_probe_stays_closed():
    """The network must never connect what a probe cannot pass."""
    structure = _simple_cubic()
    assisted = _grid(structure, WINDOW + 0.03, use_network=True)
    assert assisted.accessibility_method == "voronoi+grid"
    assert assisted.free_mask.any()          # the cavity is there...
    assert not assisted.sweep.accessible.any()  # ...but sealed


def test_pore_analysis_reports_the_method_and_exact_geometry():
    result = analyze_pores(_simple_cubic(), probe_radius=WINDOW - 0.03,
                           radius_provider=RADII)
    assert result.accessibility_method == "voronoi+grid"
    assert result.accessible_dimensionality == 3
    assert result.non_accessible_volume == 0
    assert result.pld == pytest.approx(2 * WINDOW, abs=0.01)
    assert result.lcd == pytest.approx(2 * CAVITY, abs=0.01)


def test_surface_area_moves_from_pocket_to_channel_across_the_window():
    """Either side of the window radius the total area is about the
    same; which side of the channel/pocket line it falls on flips.

    Known limit, kept visible: 0.03 A from the window, a few percent of
    the surface lies in the window's crevice, closer to it than the
    surface-area grid can resolve; those points are still counted as
    pocket (surface_area.classify_channel_points)."""
    structure = _simple_cubic()
    below = analyze_surface_area(structure, probe_radius=WINDOW - 0.03, probe_name="x",
                                 radius_provider=RADII, n_samples_per_atom=500)
    above = analyze_surface_area(structure, probe_radius=WINDOW + 0.03, probe_name="x",
                                 radius_provider=RADII, n_samples_per_atom=500)
    total_below = below.internal_accessible_area + below.pocket_accessible_area
    assert below.pocket_accessible_area < 0.1 * total_below
    assert above.internal_accessible_area == 0
    assert above.pocket_accessible_area > 0


def test_network_edges_are_right_for_vertices_on_a_cell_face():
    """bcc, shifted so that every Voronoi vertex wraps a hair inside or
    outside a cell face. The 24 edges of the network (the truncated
    octahedra's edges) are all a*sqrt(2)/4 long; an edge given the
    wrong cell offset would be a cell length longer or shorter."""
    a, eps = 10.0, -3e-10
    frac = [np.array([eps] * 3), np.array([0.5 + eps] * 3)]
    lattice = Lattice.cubic(a).matrix
    radii = np.array([AR, AR])
    clearance = pore_geometry.ClearanceFunction(lattice, frac, radii, 12)
    network = pore_geometry.build_voronoi_network(lattice, frac, radii, clearance, 25)

    lengths = [
        np.linalg.norm(network.vertex_positions[j] + np.array(image) @ lattice
                       - network.vertex_positions[i])
        for i, j, image, *_ in network.edges
    ]
    assert len(lengths) == 24
    assert np.allclose(lengths, a * math.sqrt(2) / 4, atol=1e-3)


def test_without_a_network_the_grid_answer_is_kept():
    """If the Voronoi network cannot be built, the sweep falls back to
    the grid's own answer, and says so."""
    structure = _simple_cubic()
    grid = _grid(structure, 1.0, use_network=False)
    sweep = sweep_connectivity(grid.distance_field, grid.grid_shape, 1.0)
    result, method, _ = network_assisted_sweep(
        structure.lattice.matrix, [s.frac_coords for s in structure], np.array([AR]),
        grid_frac=grid.grid_frac, distance_field=grid.distance_field,
        grid_shape=grid.grid_shape, probe_radius=1.0, grid_sweep=sweep,
        prepared=(None, None),
    )
    assert method == "grid" and result is sweep


# ----------------------------------------------------------------------
# Real frameworks, against Zeo++ (network_accessibility round 55)
# ----------------------------------------------------------------------


def _root():
    for candidate in (os.environ.get("RASPA_STRUCTURES"),
                      "/tmp/raspa/RASPA2-2.0.4/structures"):
        if candidate and Path(candidate, "mofs", "cif").is_dir():
            return Path(candidate, "mofs", "cif")
    return None


ROOT = _root()
real = pytest.mark.skipif(ROOT is None, reason="RASPA2 structures not available")


def _load(name):
    from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader
    from thevoid_core.services.pore_analysis import major_occupancy_structure

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return major_occupancy_structure(PymatgenCIFLoader().load(str(ROOT / f"{name}.cif")).structure)


@real
def test_uio66_is_a_channel_for_n2():
    """The reported case. Zeo++ 0.3 (Bondi radii, probe 1.82 A):
    ASA 1106 A^2, all channel, no pockets."""
    from thevoid_core.services.guest_accessibility import (
        AccessibilityStatus,
        analyze_guest_accessibility,
    )

    structure = _load("UIO-66")
    area = analyze_surface_area(structure, probe_radius=1.82, probe_name="N2",
                                radius_provider=RADII, n_samples_per_atom=300)
    assert area.pocket_accessible_area == 0
    assert area.internal_accessible_area == pytest.approx(1106, rel=0.02)

    guest = analyze_guest_accessibility(structure, guest_diameter=3.64, guest_name="N2",
                                        radius_provider=RADII)
    assert guest.status == AccessibilityStatus.ACCESSIBLE


@real
def test_zif8_stays_closed_to_n2():
    """The case that caught the edge-offset bug: ZIF-8's 3.3 A windows
    are narrower than N2, so none of its surface is channel."""
    structure = _load("ZIF-8")
    result = analyze_pores(structure, probe_radius=1.82, radius_provider=RADII)
    assert result.pld == pytest.approx(3.305, abs=0.01)
    assert result.accessible_volume == 0
    area = analyze_surface_area(structure, probe_radius=1.82, probe_name="N2",
                                radius_provider=RADII, n_samples_per_atom=300)
    assert area.internal_accessible_area == 0
