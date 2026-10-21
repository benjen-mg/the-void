"""Round 57: the split-sphere Voronoi network, for channels the radical
network cannot represent (pore_geometry.split_network).

With a carbon (1.70 A) among hydrogens (1.09 A), the radical Voronoi
diagram - the one Zeo++ and the Void use - can run its edges through the
wrong places, and in layered frameworks miss the channel between the
layers. On 2020[Cd][hcb]2[ASR]1 (CoRE MOF 2024, CCDC radii) the 0.8.6
PLD was 2.341 A, where a 0.1 A grid path proves at least 2.48 A (2.581
on its nodes, less one spacing) and Zeo++'s high-accuracy mode gives
2.651. Splitting every larger atom into spheres of the smallest radius
makes the radical diagram follow the true channels; its edges are then
valued with the true clearance along them, so what it finds is a real
path.

Checked here: the edges that count are valued exactly; building once and
extending the exact range agrees with building afresh; equal radii are
left alone; the Cd case comes up to within 0.01 A of Zeo++ and never
below the proof; and accessibility agrees with the PLD there, which it
did not while only the PLD used the new network.

Round 58: the network used to be built only when the grid's PLD came
within 0.015 A of the radical network's. A live-fetched set of 36 CoRE
Zn frameworks, plotted with the new parity figure, showed three PLDs
0.18-0.34 A low that the signal missed (2017[ZnSi][nan]3 among them),
so it is now built whenever radii differ, with half the image cloud.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from thevoid_core.data.element_radii import CCDCRadiusProvider  # noqa: E402
from thevoid_core.services import pore_geometry as pg  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.pore_analysis import (  # noqa: E402
    analyze_pores,
    major_occupancy_structure,
)

FIXTURES = Path(__file__).parent / "fixtures"
CD_CASE = FIXTURES / "coremof" / "2020[Cd][hcb]2[ASR]1.cif"


from tests.sample_structures import ZN_DIA  # noqa: E402

def _load(path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return major_occupancy_structure(PymatgenCIFLoader().load(str(path)).structure)


def _inputs(path):
    structure = _load(path)
    lattice = structure.lattice.matrix
    frac = [site.frac_coords for site in structure]
    radii = np.array([CCDCRadiusProvider().get_vdw_radius(site.specie.symbol) for site in structure])
    return lattice, frac, radii, pg.ClearanceFunction(lattice, frac, radii, 10.0)


@pytest.fixture(scope="module")
def sample_split():
    lattice, frac, radii, clearance = _inputs(ZN_DIA)
    return lattice, clearance, pg.split_network(lattice, frac, radii, clearance, 9.0,
                                                exact_above=1.0, directions=12)


def test_edges_that_count_are_valued_exactly(sample_split):
    lattice, clearance, split = sample_split
    checked = 0
    for edge, raw in zip(split.edges, split.raw_edges):
        if max(raw[3], raw[8]) < 1.0:
            continue
        start = split.vertex_positions[edge[0]]
        end = split.vertex_positions[edge[1]] + np.asarray(edge[2], dtype=float) @ lattice
        assert edge[3] == pytest.approx(pg.segment_minimum(clearance, start, end)[0], abs=1e-12)
        checked += 1
    assert checked > 100
    # Vertex values are the true clearance, not the pieces'.
    assert np.allclose(split.vertex_clearance, clearance(split.vertex_positions))


def test_extending_the_exact_range_agrees_with_building_afresh(sample_split):
    lattice, clearance, split = sample_split
    _, frac, radii, _ = _inputs(ZN_DIA)
    fresh = pg.split_network(lattice, frac, radii, clearance, 9.0, exact_above=0.6,
                             directions=12)
    pg.make_exact(split, clearance, lattice, 0.6)
    assert split.exact_above == 0.6
    assert [e[3] for e in split.edges] == pytest.approx([e[3] for e in fresh.edges])


def test_equal_radii_are_not_split():
    """No atom is larger than the smallest: nothing to split, and the
    network is the radical one."""
    lattice = np.eye(3) * 10.2
    frac, radii = [np.zeros(3)], np.array([1.88])
    clearance = pg.ClearanceFunction(lattice, frac, radii, 12.0)
    split = pg.split_network(lattice, frac, radii, clearance, 12.0, exact_above=4.0)
    radical = pg.build_voronoi_network(lattice, frac, radii, clearance, 12.0)
    assert len(split.vertex_positions) == len(radical.vertex_positions)


def test_the_layered_case_now_finds_its_channel():
    result = analyze_pores(_load(CD_CASE), probe_radius=1.2,
                           radius_provider=CCDCRadiusProvider())
    assert result.geometry_method == "voronoi+split"
    assert 2.48 <= result.pld <= 2.651 + 0.01
    assert result.pld == pytest.approx(2.644, abs=0.01)
    # 1.2 A is below PLD / 2 = 1.32 A: the probe passes, and the
    # accessibility classification says so too.
    assert result.accessibility_method == "split+grid"
    assert result.accessible_dimensionality >= 1
    assert result.accessible_volume > 0


def test_structures_the_radical_network_handles_are_left_alone():
    """A Zn MOF: the split network is built
    (radii differ) but finds no wider path, so the radical network's
    value and label stand."""
    result = analyze_pores(_load(ZN_DIA), probe_radius=1.2,
                           radius_provider=CCDCRadiusProvider())
    assert result.geometry_method == "voronoi"
    assert result.accessibility_method == "voronoi+grid"


ZNSI_CASE = FIXTURES / "coremof" / "2017[ZnSi][nan]3[ASR]1.cif"


def test_a_case_the_round_57_trigger_missed():
    """2017[ZnSi][nan]3[ASR]1 (CoRE PLD 3.980 A, Zeo++ -ha): the radical
    network gives 3.640 A, and the grid's path (3.539 A) stayed 0.05 A
    (radius) below it - outside round 57's 0.015 A trigger, so 0.8.7
    reported 3.640. The split network finds 3.934 A."""
    result = analyze_pores(_load(ZNSI_CASE), probe_radius=1.655,
                           radius_provider=CCDCRadiusProvider())
    assert result.geometry_method == "voronoi+split"
    assert result.pld == pytest.approx(3.934, abs=0.01)
    assert abs(result.pld - 3.98017) < 0.15


def test_half_the_image_cloud_finds_the_same_path():
    """split_reach halves the reach. A thinner cloud can only lose an
    edge, never invent a wide one (edges that count are valued exactly),
    and on the Cd case it loses nothing."""
    lattice, frac, radii, clearance = _inputs(CD_CASE)
    reach = 2 * (2.0 + radii.max() + 0.5) + 1
    capacities = []
    for r in (reach, pg.split_reach(reach)):
        split = pg.split_network(lattice, frac, radii, clearance, r, exact_above=0.6)
        capacities.append(pg._percolation_capacity(
            len(split.vertex_positions), split.edges, [e[3] for e in split.edges])[0])
    assert pg.split_reach(reach) < reach
    assert capacities[1] == pytest.approx(capacities[0], abs=1e-9)
    assert 2 * capacities[0] == pytest.approx(2.644, abs=0.01)
