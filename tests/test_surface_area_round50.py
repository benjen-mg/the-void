"""Round 50: evenly spread surface sampling, vectorised exposure test,
and the corrected channel / pocket split. Zeo++ reference values were
computed once, offline, with the same radii (see test_pore_geometry.py).
"""


import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("scipy")

from pymatgen.core import Lattice, Structure  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services.surface_area import (  # noqa: E402
    analyze_surface_area,
    exposed_mask,
    fibonacci_sphere_points,
    random_rotation,
    sample_atom_surfaces,
)


from tests.sample_structures import ZN_DIA  # noqa: E402

class FixedRadius:
    def __init__(self, radius):
        self.radius = radius

    def get_vdw_radius(self, symbol):
        return self.radius


def _load(path):
    from thevoid_core.services.cif.pymatgen_cif_loader import (
        PymatgenCIFLoader,
    )

    return PymatgenCIFLoader().load(str(path)).structure


def test_fibonacci_points_lie_on_the_sphere_and_balance():
    rotation = random_rotation(np.random.default_rng(0))
    points = fibonacci_sphere_points(2000, 2.5, (1.0, -2.0, 3.0), rotation)

    offsets = points - np.array([1.0, -2.0, 3.0])
    assert np.allclose(np.linalg.norm(offsets, axis=1), 2.5)
    assert np.all(np.abs(offsets.mean(axis=0)) < 1e-2)


def test_random_rotation_is_a_proper_rotation():
    matrix = random_rotation(np.random.default_rng(7))

    assert np.allclose(matrix @ matrix.T, np.eye(3))
    assert np.linalg.det(matrix) == pytest.approx(1.0)


@pytest.mark.parametrize("seed", range(5))
def test_spherical_cap_is_accurate_with_few_samples(seed):
    """The 75%-exposed two-sphere case: with 2000 evenly spread samples
    the error stays under 0.5% for every seed - random sampling needed
    100,000 samples to be tested at 2%."""
    centres = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    radii = np.array([2.0, 2.0])

    _total, per_atom, _points, _exposed = sample_atom_surfaces(
        centres, radii, centres, radii, probe_radius=0.0,
        n_samples_per_atom=2000, rng=np.random.default_rng(seed),
    )

    expected = 0.75 * 4 * np.pi * 4.0
    assert per_atom[0] == pytest.approx(expected, rel=0.005)
    assert per_atom[1] == pytest.approx(expected, rel=0.005)


def test_vectorised_exposure_matches_brute_force():
    rng = np.random.default_rng(3)
    centres = rng.uniform(0, 8, size=(150, 3))
    radii = rng.uniform(0.8, 2.0, size=150)
    points = rng.uniform(0, 8, size=(3000, 3))

    fast = exposed_mask(points, cKDTree(centres), radii, float(radii.max()))

    distances = np.linalg.norm(points[:, None, :] - centres[None], axis=2)
    brute = ~np.any(distances < radii[None, :] - 1e-6, axis=1)

    assert np.array_equal(fast, brute)


def test_exposure_is_exact_even_with_more_than_k_neighbours():
    """More overlapping neighbours than the batched query looks at: the
    exhaustive fallback must still catch the one that buries the point.
    """
    centres = np.vstack(
        [np.full((80, 3), 10.0) + np.linspace(0, 0.1, 80)[:, None],
         [[0.0, 0.0, 0.0]]]
    )
    radii = np.concatenate([np.full(80, 20.0), [0.5]])
    # Inside the tiny atom at the origin, but also inside all 80 big
    # ones - buried either way.
    point = np.array([[0.1, 0.0, 0.0]])

    assert not exposed_mask(point, cKDTree(centres), radii, 20.0)[0]


def test_sealed_cavities_are_all_pocket_surface():
    """Simple cubic, r = 2 A, a = 10 A: body-centre cavities of radius
    6.66 A joined by windows of radius 5.07 A. A 5.5 A probe fits in
    the cavities but not through the windows - every bit of surface it
    touches is pocket. A 4 A probe goes through - all channel.
    """
    structure = Structure(Lattice.cubic(10.0), ["Co"], [[0, 0, 0]])

    sealed = analyze_surface_area(
        structure, probe_radius=5.5, probe_name="Custom",
        radius_provider=FixedRadius(2.0),
    )
    assert sealed.accessible_surface_area > 0
    assert sealed.internal_accessible_area == pytest.approx(0.0, abs=1e-9)
    assert sealed.pocket_accessible_area == pytest.approx(
        sealed.accessible_surface_area
    )

    open_ = analyze_surface_area(
        structure, probe_radius=4.0, probe_name="Custom",
        radius_provider=FixedRadius(2.0),
    )
    assert open_.pocket_accessible_area == pytest.approx(0.0, abs=1e-9)
    assert open_.internal_accessible_area == pytest.approx(
        open_.accessible_surface_area
    )


def test_channel_mask_is_a_subset_of_the_exposed_points():
    result = analyze_surface_area(
        _load(ZN_DIA), probe_radius=1.0, probe_name="Custom",
        radius_provider=ElementRadiusProvider(),
    )

    assert result.channel_mask.shape == result.accessible_mask.shape
    assert not np.any(result.channel_mask & ~result.accessible_mask)
