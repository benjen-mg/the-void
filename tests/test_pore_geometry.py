"""Round 50: grid-independent LCD / PLD (pore_geometry.py).

The reference values here are either exact geometry, derived by hand in
each docstring, or Zeo++ 0.2 (pyzeo) run once, offline, on the same
structure with the same van der Waals radii - Zeo++ is not a dependency
of this application and is not needed to run these tests.
"""

import math

import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("scipy")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.guest_accessibility import (  # noqa: E402
    AccessibilityStatus,
    analyze_guest_accessibility,
)
from thevoid_core.services.pore_analysis import analyze_pores  # noqa: E402
from thevoid_core.services.pore_geometry import (  # noqa: E402
    ClearanceFunction,
    _edge_bottleneck,
)


class FixedRadius:
    def __init__(self, radius):
        self.radius = radius

    def get_vdw_radius(self, symbol):
        return self.radius



def _simple_cubic(a):
    return Structure(Lattice.cubic(a), ["Co"], [[0, 0, 0]])


@pytest.mark.parametrize("a", [7.3, 10.0, 10.2, 12.77])
def test_simple_cubic_lcd_and_pld_are_exact_whatever_the_grid(a):
    """Simple cubic lattice of r = 2 A spheres. The largest cavity is
    at the body centre, a*sqrt(3)/2 from eight atoms; the window is the
    face centre, a/sqrt(2) from four. Before round 50 these came out
    0.4-0.7 A low whenever the cell edge put those points between grid
    nodes (a = 7.3, 10.2 - measured -0.52 / -0.42 A at 10.2).
    """
    r = 2.0
    result = analyze_pores(
        _simple_cubic(a), probe_radius=0.0, radius_provider=FixedRadius(r)
    )

    assert result.geometry_method == "voronoi"
    assert result.lcd == pytest.approx(2 * (a * math.sqrt(3) / 2 - r), abs=1e-3)
    assert result.pld == pytest.approx(2 * (a / math.sqrt(2) - r), abs=1e-3)


def test_zeo_plus_plus_agrees_on_simple_cubic():
    """Zeo++ (network -res) on the same sc a = 10.2 A, r = 2 A lattice:
    Di 13.66692, Df 10.42498."""
    result = analyze_pores(
        _simple_cubic(10.2), probe_radius=0.0, radius_provider=FixedRadius(2.0)
    )

    assert result.lcd == pytest.approx(13.66692, abs=1e-3)
    assert result.pld == pytest.approx(10.42498, abs=1e-3)


def test_body_centred_cubic_pld_is_the_triangle_window():
    """bcc, r = 2 A, a = 10.2 A. The Delaunay tetrahedra have isosceles
    triangular faces (sides a, a*sqrt(3)/2, a*sqrt(3)/2) with
    circumradius (3*sqrt(2)/8)*a = 0.5303a; the widest path squeezes
    through their circumcentres, so PLD = 2*(0.5303a - r). The largest
    cavity is the tetrahedral hole, a*sqrt(5)/4 from its four atoms.
    """
    a, r = 10.2, 2.0
    structure = Structure(
        Lattice.cubic(a), ["Co", "Co"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    result = analyze_pores(
        structure, probe_radius=0.0, radius_provider=FixedRadius(r)
    )

    assert result.lcd == pytest.approx(2 * (a * math.sqrt(5) / 4 - r), abs=1e-3)
    assert result.pld == pytest.approx(
        2 * (3 * math.sqrt(2) / 8 * a - r), abs=1e-3
    )


def test_refined_values_never_fall_below_the_grid():
    """LCD is a maximum: a refined value below what a grid node already
    showed would be a bug in the optimiser."""
    structure = _simple_cubic(10.2)
    grid = analyze_pores(
        structure, probe_radius=0.0, radius_provider=FixedRadius(2.0),
        refine_geometry=False,
    )
    refined = analyze_pores(
        structure, probe_radius=0.0, radius_provider=FixedRadius(2.0),
    )

    assert grid.geometry_method == "grid"
    assert refined.lcd >= grid.lcd
    assert refined.pld >= grid.pld


def test_pld_never_exceeds_lcd():
    result = analyze_pores(
        _simple_cubic(10.2), probe_radius=0.0, radius_provider=FixedRadius(2.0)
    )

    assert result.pld <= result.lcd


def test_guest_accessibility_uses_the_refined_pld():
    """The same sc a = 10.2 lattice: a 10.3 A guest fits through the
    exact 10.425 A window but not through the grid's 10.004 A one -
    exactly the kind of answer the grid used to flip.
    """
    result = analyze_guest_accessibility(
        _simple_cubic(10.2), guest_diameter=10.3, guest_name="Custom",
        radius_provider=FixedRadius(2.0),
    )

    assert result.minimum_window_diameter == pytest.approx(10.42498, abs=1e-3)
    assert result.status == AccessibilityStatus.ACCESSIBLE


def test_clearance_function_is_periodic():
    structure = _simple_cubic(10.0)
    clearance = ClearanceFunction(
        structure.lattice.matrix, structure.frac_coords, [2.0], 12.0
    )

    point = np.array([5.0, 5.0, 5.0])

    assert clearance.at(point) == pytest.approx(math.sqrt(75) - 2.0)
    assert clearance.at(point + [10.0, -20.0, 30.0]) == pytest.approx(
        clearance.at(point)
    )


def test_edge_bottleneck_is_the_closest_approach():
    """An atom of radius 1 at (0, 2, 0) beside the segment (-5,0,0) to
    (5,0,0): closest approach at the origin, clearance 2 - 1 = 1."""
    value, point = _edge_bottleneck(
        np.array([-5.0, 0, 0]), np.array([5.0, 0, 0]),
        np.array([[0.0, 2.0, 0.0]]), np.array([1.0]),
    )

    assert value == pytest.approx(1.0)
    assert point == pytest.approx([0.0, 0.0, 0.0])


def test_falls_back_to_the_grid_if_the_network_fails(monkeypatch):
    import thevoid_core.services.pore_geometry as geometry

    def boom(*args, **kwargs):
        raise RuntimeError("qhull says no")

    monkeypatch.setattr(geometry, "build_voronoi_network", boom)

    result = analyze_pores(
        _simple_cubic(10.2), probe_radius=0.0, radius_provider=FixedRadius(2.0)
    )

    assert result.geometry_method == "grid"
    # LCD is still refined (it does not need the network)...
    assert result.lcd == pytest.approx(13.66692, abs=1e-3)
    # ...PLD is the grid's.
    assert result.pld == pytest.approx(10.004, abs=0.01)


def test_accessible_verdict_is_never_shown_with_an_empty_network():
    """When the exact PLD admits the guest but the grid's does not, the
    3D view still gets the network to draw."""
    result = analyze_guest_accessibility(
        _simple_cubic(10.2), guest_diameter=10.3, guest_name="Custom",
        radius_provider=FixedRadius(2.0),
    )

    assert result.status == AccessibilityStatus.ACCESSIBLE
    assert len(result.accessible_points) > 0
