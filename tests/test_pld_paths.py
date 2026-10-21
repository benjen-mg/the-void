"""Round 56: the PLD refinement values actual paths
(pore_geometry.voronoi_pld, segment_minimum, _best_route).

Found through the CoRE MOF comparison: the refinement used to raise a
near-critical Voronoi edge to min(window centre, its two vertices)
without checking that a path joined them at that level, and kept a
vertex at its power centre when the true four-atom vertex did not
exist. The first overstated MOF-205's PLD by 0.115 A (8.433 against
Zeo++'s high-accuracy 8.318, same radii); the second understated a
CoRE MOF 2024 structure's by 0.38 A (5.57 against 5.95). Every refined
value now comes from a polyline whose clearance is computed exactly.

Checked here: the exact segment minimum against dense sampling; the PLD
never falls below the radical-Voronoi value; the CoRE case against a
rigorous lower bound (a 0.15 A grid path) and Zeo++; MOF-205 against
Zeo++ when RASPA2's structures are available.
"""

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from thevoid_core.data.element_radii import CCDCRadiusProvider, ElementRadiusProvider  # noqa: E402
from thevoid_core.services import pore_geometry as pg  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.pore_analysis import (  # noqa: E402
    analyze_pores,
    major_occupancy_structure,
)

FIXTURES = Path(__file__).parent / "fixtures"
CU_CASE = FIXTURES / "coremof" / "2015[Cu][nan]3[ASR]1.cif"


from tests.sample_structures import ZN_DIA  # noqa: E402

def _load(path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return major_occupancy_structure(PymatgenCIFLoader().load(str(path)).structure)


def _network(structure, provider):
    lattice = structure.lattice.matrix
    frac = [site.frac_coords for site in structure]
    radii = np.array([provider.get_vdw_radius(site.specie.symbol) for site in structure])
    clearance = pg.ClearanceFunction(lattice, frac, radii, 12.0)
    return clearance, pg.build_voronoi_network(lattice, frac, radii, clearance, 12.0)


def test_the_segment_minimum_is_exact():
    """Against 2001-point sampling on 500 random segments in a Zn MOF:
    the exact minimum is never above the
    sampled one, and never far below."""
    clearance, _ = _network(_load(ZN_DIA), ElementRadiusProvider())
    rng = np.random.default_rng(56)
    lattice = clearance.lattice
    for _ in range(500):
        start = rng.random(3) @ lattice
        direction = rng.normal(size=3)
        end = start + direction / np.linalg.norm(direction) * rng.uniform(0.0, 5.0)
        value, point = pg.segment_minimum(clearance, start, end)
        sampled = clearance(start + np.linspace(0, 1, 2001)[:, None] * (end - start)).min()
        assert value <= sampled + 1e-9
        assert value >= sampled - 1e-4
        assert clearance.at(point) == pytest.approx(value, abs=1e-9)


@pytest.mark.parametrize("path, provider", [
    (ZN_DIA, ElementRadiusProvider()),
    (CU_CASE, CCDCRadiusProvider()),
])
def test_the_pld_is_never_below_the_radical_voronoi_value(path, provider):
    clearance, network = _network(_load(path), provider)
    radical, _ = pg._percolation_capacity(
        len(network.vertex_positions), network.edges, [e[3] for e in network.edges])
    refined, point = pg.voronoi_pld(network, clearance)
    assert refined >= radical
    assert point.shape == (3,)


def test_the_coremof_case_is_no_longer_low():
    """2015[Cu][nan]3[ASR]1 with CoRE's (CCDC) radii. Radical Voronoi,
    the Void 0.8.5 and Zeo++'s standard mode: 5.5685. A 0.15 A grid
    finds a path at 5.8286 on its nodes, so the true PLD is at least
    5.8286 - 0.15 = 5.68 (adjacent nodes are 0.15 A apart, and the
    clearance changes by at most the distance moved). Zeo++'s high-
    accuracy mode, which CoRE MOF 2024 used: 5.94846."""
    result = analyze_pores(_load(CU_CASE), probe_radius=1.655,
                           radius_provider=CCDCRadiusProvider())
    assert result.pld >= 5.68
    assert result.pld == pytest.approx(5.94846, abs=0.02)


def test_a_layered_framework_is_at_least_what_the_grid_proves():
    """2023[Co][sql]2[ASR]1, a layered framework, CCDC radii. The radical
    Voronoi network says 4.388 A and 0.8.5 said 4.418. The 0.3 A pore
    grid percolates with every node 2.518 A from the atoms, which proves
    a 4.735 A probe passes (see refined_lcd_pld). The refined network
    reaches 5.016, and since round 57 the split-sphere network 5.151
    (test_split_network.py); Zeo++'s high-accuracy mode says 5.228."""
    result = analyze_pores(_load(FIXTURES / "coremof" / "2023[Co][sql]2[ASR]1.cif"),
                           probe_radius=1.2, radius_provider=CCDCRadiusProvider())
    assert result.geometry_method == "voronoi+split"
    assert 4.735 <= result.pld <= 5.228
    assert result.pld == pytest.approx(5.151, abs=0.01)


def _raspa_root():
    for candidate in (os.environ.get("RASPA_STRUCTURES"), "/tmp/raspa/RASPA2-2.0.4/structures"):
        if candidate and Path(candidate, "mofs", "cif").is_dir():
            return Path(candidate, "mofs", "cif")
    return None


@pytest.mark.skipif(_raspa_root() is None, reason="RASPA2 structures not available")
def test_mof205_is_no_longer_overstated():
    """MOF-205 with the Void's radii. 0.8.5 said 8.433; Zeo++ 0.3 in
    high-accuracy mode with the same radii says 8.31766, and a local
    0.06 A grid around the edge 0.8.5 relied on shows no path there at
    all above 8.38 (no 26-connected path even through nodes 0.052 A
    below that level)."""
    result = analyze_pores(_load(_raspa_root() / "MOF-205.cif"), probe_radius=1.2,
                           radius_provider=ElementRadiusProvider())
    assert result.pld == pytest.approx(8.31766, abs=0.005)
