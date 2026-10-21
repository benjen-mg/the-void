"""analyze_surface_area(): the full pipeline, against real pymatgen
structures chosen so the expected ASA can be worked out by hand or
reasoned about with confidence.
"""

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services.surface_area import (  # noqa: E402
    AVOGADRO_NUMBER,
    PROBE_RADII,
    analyze_surface_area,
)


@pytest.fixture(scope="module")
def radius_provider():
    return ElementRadiusProvider()


def test_isolated_atom_asa_matches_the_sphere_formula(radius_provider):
    """Nothing nearby to block it: ASA must equal 4*pi*(vdw+probe)^2
    exactly, the same isolated-sphere case already verified at the
    sampling level, now checked all the way through the full pipeline.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    probe = PROBE_RADII["N2"]
    result = analyze_surface_area(
        structure, probe_radius=probe, probe_name="N2",
        radius_provider=radius_provider, n_samples_per_atom=5000,
    )

    vdw_li = radius_provider.get_vdw_radius("Li")
    expected = 4 * np.pi * (vdw_li + probe) ** 2

    assert result.accessible_surface_area == pytest.approx(expected)


def test_geometric_area_uses_zero_probe_radius(radius_provider):
    """The geometric pass must match the same isolated-sphere formula
    with no probe added at all - the true van der Waals surface.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    result = analyze_surface_area(
        structure, probe_radius=1.82, probe_name="N2",
        radius_provider=radius_provider, n_samples_per_atom=5000,
    )

    vdw_li = radius_provider.get_vdw_radius("Li")
    expected_geometric = 4 * np.pi * vdw_li**2

    assert result.geometric_surface_area == pytest.approx(expected_geometric)


def test_unit_conversion_to_square_meters_per_gram(radius_provider):
    """Hand-derived: ASA[m^2] / cell_mass[g], with ASA converted from
    A^2 via the exact 1 A^2 = 1e-20 m^2 factor and cell mass from
    pymatgen's own composition weight (amu) via Avogadro's number -
    checked independently here rather than trusting the function's own
    internal arithmetic to be self-consistent with itself.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    result = analyze_surface_area(
        structure, probe_radius=1.82, probe_name="N2",
        radius_provider=radius_provider, n_samples_per_atom=5000,
    )

    expected_mass_g = float(structure.composition.weight) / AVOGADRO_NUMBER
    expected_asa_per_mass = (
        result.accessible_surface_area * 1e-20
    ) / expected_mass_g

    assert result.cell_mass_grams == pytest.approx(expected_mass_g)
    assert result.asa_per_mass == pytest.approx(expected_asa_per_mass)


def test_unit_conversion_to_square_meters_per_cubic_centimeter(
    radius_provider,
):
    """Hand-derived: ASA[m^2] / cell_volume[cm^3], with cell volume
    converted from A^3 via the exact 1 A^3 = 1e-24 cm^3 factor.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    result = analyze_surface_area(
        structure, probe_radius=1.82, probe_name="N2",
        radius_provider=radius_provider, n_samples_per_atom=5000,
    )

    expected_volume_cm3 = result.cell_volume * 1e-24
    expected_asa_per_volume = (
        result.accessible_surface_area * 1e-20
    ) / expected_volume_cm3

    assert result.asa_per_volume == pytest.approx(expected_asa_per_volume)


def test_a_fully_open_cell_is_entirely_internal_accessible_area(
    radius_provider,
):
    """A single small atom in a large, essentially fully open cell: its
    entire exposed surface borders one single, fully connected,
    percolating void - internal_accessible_area must equal
    total_accessible_area (within the grid's own discretisation), not
    some arbitrary fraction of it.

    This is also the direct regression check for a real methodological
    bug found and fixed while building this: an earlier version
    classified each surface point by looking it up directly against a
    percolation grid, but every accessible surface point sits, by
    construction, *exactly* on the accessibility threshold - a
    boundary case a coarse grid lookup decides almost like a coin flip
    rather than a real measurement. That version reported roughly half
    of this same fully-open cell's surface as "not internal", which is
    not a plausible answer for a single atom floating in open space.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    result = analyze_surface_area(
        structure, probe_radius=PROBE_RADII["N2"], probe_name="N2",
        radius_provider=radius_provider, n_samples_per_atom=3000,
    )

    assert result.internal_accessible_area == pytest.approx(
        result.total_accessible_area, rel=0.05
    )


def test_total_accessible_area_equals_accessible_surface_area(
    radius_provider,
):
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, ["Zn"], [[0.5, 0.5, 0.5]])

    result = analyze_surface_area(
        structure, probe_radius=1.0, probe_name="Custom",
        radius_provider=radius_provider, n_samples_per_atom=1000,
    )

    assert result.total_accessible_area == result.accessible_surface_area


def test_empty_structure_reports_zero_area(radius_provider):
    lattice = Lattice.cubic(10.0)
    structure = Structure(lattice, [], [])

    result = analyze_surface_area(
        structure, probe_radius=1.0, probe_name="Custom",
        radius_provider=radius_provider,
    )

    assert result.accessible_surface_area == 0.0
    assert result.geometric_surface_area == 0.0
    assert result.asa_per_mass == 0.0
    assert result.asa_per_volume == 0.0
    assert result.internal_accessible_area == 0.0


def test_probe_radii_dictionary_has_the_required_probes():
    """CH4 was added to the shared, canonical probe table (data.
    probe_radii) when guest_accessibility.py needed it too - surface
    area analysis gains it as a natural, positive side effect of
    reusing one table rather than maintaining two that could drift
    apart, rather than being artificially excluded here.
    """
    assert {"N2", "CO2", "H2", "H2O", "CH4"} <= set(PROBE_RADII.keys())
    assert all(radius > 0 for radius in PROBE_RADII.values())


def test_custom_probe_radius_is_used_directly(radius_provider):
    """A "custom" probe is just any numeric radius passed straight
    through, not looked up from PROBE_RADII at all.
    """
    lattice = Lattice.cubic(15.0)
    structure = Structure(lattice, ["Li"], [[0.5, 0.5, 0.5]])

    custom_radius = 2.37
    result = analyze_surface_area(
        structure, probe_radius=custom_radius, probe_name="Custom",
        radius_provider=radius_provider, n_samples_per_atom=3000,
    )

    vdw_li = radius_provider.get_vdw_radius("Li")
    expected = 4 * np.pi * (vdw_li + custom_radius) ** 2

    assert result.accessible_surface_area == pytest.approx(expected)
    assert result.probe_radius == custom_radius
    assert result.probe_name == "Custom"


def test_sampled_points_are_returned_for_visualization(radius_provider):
    lattice = Lattice.cubic(12.0)
    structure = Structure(lattice, ["Zn"], [[0.5, 0.5, 0.5]])

    result = analyze_surface_area(
        structure, probe_radius=1.5, probe_name="Custom",
        radius_provider=radius_provider, n_samples_per_atom=500,
    )

    assert result.sampled_points.shape == (500, 3)
    assert result.accessible_mask.shape == (500,)
    assert np.all(np.isfinite(result.sampled_points))
