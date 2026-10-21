"""Round 51: geometric pore size distribution."""

import math

import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("scipy")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.element_radii import ElementRadiusProvider  # noqa: E402
from thevoid_core.services.pore_analysis import analyze_pores  # noqa: E402
from thevoid_core.services.pore_size_distribution import (  # noqa: E402
    compute_pore_size_distribution,
)


from tests.sample_structures import ZN_DIA  # noqa: E402

class FixedRadius:
    def __init__(self, radius):
        self.radius = radius

    def get_vdw_radius(self, symbol):
        return self.radius


def _load(path):
    from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader

    return PymatgenCIFLoader().load(str(path)).structure


@pytest.fixture(scope="module")
def sc_psd():
    structure = Structure(Lattice.cubic(10.0), ["Co"], [[0, 0, 0]])
    return compute_pore_size_distribution(structure, radius_provider=FixedRadius(2.0))


def test_distribution_integrates_to_one(sc_psd):
    assert float(np.sum(sc_psd.distribution) * sc_psd.bin_width) == pytest.approx(1.0)


def test_cumulative_volume_starts_at_the_void_volume_and_never_rises(sc_psd):
    exact_void = 1000.0 - 4.0 / 3.0 * math.pi * 2.0**3
    assert sc_psd.pore_volume == pytest.approx(exact_void, rel=0.005)
    assert sc_psd.cumulative_volume[0] == sc_psd.pore_volume
    assert np.all(np.diff(sc_psd.cumulative_volume) <= 1e-9)
    assert sc_psd.cumulative_volume[-1] == 0.0


def test_largest_populated_bin_matches_the_largest_cavity(sc_psd):
    """Simple cubic, r = 2, a = 10: LCD 2*(5*sqrt(3) - 2) = 13.32 A.
    The PSD's last populated diameter must fall within a bin plus the
    grid's own resolution of it, and the dominant peak must be there
    too (most of the void sits inside the big body-centre sphere)."""
    lcd = 2 * (5 * math.sqrt(3) - 2)
    populated = sc_psd.diameters[np.nonzero(sc_psd.cumulative_volume)[0][-1]]

    assert lcd - 2 * sc_psd.bin_width - 0.3 <= populated <= lcd
    assert sc_psd.dominant_diameter == pytest.approx(lcd, abs=0.6)


def test_every_point_is_counted_once():
    """The whole void is covered at D = 0, and V(D) at the first bin edge
    is at most that - the distribution cannot create volume."""
    structure = _load(ZN_DIA)
    result = compute_pore_size_distribution(structure, radius_provider=ElementRadiusProvider())
    assert result.cumulative_volume[1] <= result.cumulative_volume[0]
    assert np.all(result.distribution >= 0)
    pores = analyze_pores(structure, probe_radius=0.0, radius_provider=ElementRadiusProvider())
    assert 100 * result.void_fraction == pytest.approx(pores.void_fraction_percent, abs=1e-6)


def test_rows_line_up_for_csv(sc_psd):
    rows = sc_psd.rows()
    assert len(rows) == len(sc_psd.bin_centers)
    assert rows[0][2] == pytest.approx(sc_psd.pore_volume)


def test_bad_arguments_are_rejected():
    structure = Structure(Lattice.cubic(10.0), ["Co"], [[0, 0, 0]])
    with pytest.raises(ValueError):
        compute_pore_size_distribution(
            structure, radius_provider=FixedRadius(2.0), bin_width=0
        )

    empty = Structure(Lattice.cubic(10.0), [], [])
    with pytest.raises(ValueError):
        compute_pore_size_distribution(empty, radius_provider=FixedRadius(2.0))
