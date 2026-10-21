"""PowderXRDSimulator: radiation source override and range validation."""

import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.services.xrd.powder_xrd_simulator import (  # noqa: E402
    RADIATION_SOURCES,
    PowderXRDSimulator,
)


@pytest.fixture
def fcc_aluminium():
    lattice = Lattice.cubic(4.05)

    return Structure(
        lattice,
        ["Al", "Al", "Al", "Al"],
        [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
    )


def test_default_wavelength_is_cu_ka(fcc_aluminium):
    result = PowderXRDSimulator().simulate(fcc_aluminium)

    assert result.wavelength == pytest.approx(1.54184, abs=1e-4)


def test_wavelength_override_changes_the_pattern(fcc_aluminium):
    sim = PowderXRDSimulator()

    cu = sim.simulate(fcc_aluminium, two_theta_range=(10, 60))
    mo = sim.simulate(fcc_aluminium, two_theta_range=(10, 60), wavelength="MoKa")

    assert cu.wavelength != mo.wavelength
    # A shorter wavelength (Mo) fits more reflections into the same
    # 2-theta window than a longer one (Cu).
    assert mo.peak_count > cu.peak_count


def test_two_theta_range_is_recorded_on_the_result(fcc_aluminium):
    result = PowderXRDSimulator().simulate(fcc_aluminium, two_theta_range=(15, 65))

    assert result.two_theta_range == (15, 65)


def test_invalid_range_is_rejected(fcc_aluminium):
    with pytest.raises(ValueError):
        PowderXRDSimulator().simulate(fcc_aluminium, two_theta_range=(60, 15))


def test_calculators_are_cached_per_source(fcc_aluminium):
    sim = PowderXRDSimulator()

    sim.simulate(fcc_aluminium)
    sim.simulate(fcc_aluminium, wavelength="MoKa")
    sim.simulate(fcc_aluminium, wavelength="MoKa")

    assert set(sim._calculators.keys()) == {"CuKa", "MoKa"}


def test_radiation_source_table_values_are_valid_pymatgen_keys(fcc_aluminium):
    """Every label in the picker must actually simulate successfully."""
    sim = PowderXRDSimulator()

    for label, key in RADIATION_SOURCES.items():
        result = sim.simulate(fcc_aluminium, two_theta_range=(10, 40), wavelength=key)

        assert result.wavelength > 0, label
