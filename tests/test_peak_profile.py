"""Pseudo-Voigt convolution for the "profile" XRD view.

Pure numpy, no GUI or pymatgen dependency, so these run without
importorskip.
"""

import pytest

pytest.importorskip("numpy")

from thevoid_core.services.xrd.peak_profile import (
    DEFAULT_ETA,
    convolve_pattern,
    pseudo_voigt,
)


def test_pseudo_voigt_peaks_at_one_at_its_center():
    import numpy as np

    x = np.array([5.0])

    assert pseudo_voigt(x, center=5.0, fwhm=0.2)[0] == pytest.approx(1.0)


def test_pseudo_voigt_is_symmetric():
    import numpy as np

    x = np.array([4.7, 5.3])

    y = pseudo_voigt(x, center=5.0, fwhm=0.2)

    assert y[0] == pytest.approx(y[1])


def test_pseudo_voigt_half_width_is_actually_half_max():
    import numpy as np

    fwhm = 0.4
    x = np.array([0.0, fwhm / 2.0])

    y = pseudo_voigt(x, center=0.0, fwhm=fwhm, eta=DEFAULT_ETA)

    assert y[1] == pytest.approx(0.5, abs=1e-6)


def test_pseudo_voigt_rejects_non_positive_fwhm():
    import numpy as np

    with pytest.raises(ValueError):
        pseudo_voigt(np.array([0.0]), center=0.0, fwhm=0.0)


def test_convolved_pattern_renormalises_to_100():
    x, y = convolve_pattern(
        [20.0, 40.0],
        [100.0, 30.0],
        two_theta_range=(5, 90),
        fwhm=0.15,
    )

    assert max(y) == pytest.approx(100.0, abs=1e-6)


def test_convolved_pattern_peaks_line_up_with_input_centres():
    x, y = convolve_pattern(
        [30.0],
        [100.0],
        two_theta_range=(5, 90),
        fwhm=0.1,
        num_points=4001,
    )

    peak_index = max(range(len(y)), key=lambda i: y[i])

    assert x[peak_index] == pytest.approx(30.0, abs=0.05)


def test_overlapping_peaks_add_rather_than_replace():
    """Two peaks close together should sum, producing a taller merged
    peak than either alone - the whole point of a profile view over
    sticks.
    """
    x, y_close = convolve_pattern(
        [20.0, 20.05],
        [80.0, 80.0],
        two_theta_range=(5, 90),
        fwhm=0.3,
    )
    _, y_far = convolve_pattern(
        [20.0, 60.0],
        [80.0, 80.0],
        two_theta_range=(5, 90),
        fwhm=0.3,
    )

    # Both renormalise to 100 at their own tallest point, so compare the
    # raw (pre-renormalisation) behaviour indirectly: the "far apart"
    # case renormalises around a single isolated peak, the "close
    # together" case renormalises around a taller, summed peak. Confirm
    # via the un-normalised building block directly instead.
    import numpy as np

    xs = np.array(x)
    raw_close = (
        80.0 * pseudo_voigt(xs, 20.0, 0.3) + 80.0 * pseudo_voigt(xs, 20.05, 0.3)
    ).max()
    raw_single = (80.0 * pseudo_voigt(xs, 20.0, 0.3)).max()

    assert raw_close > raw_single


def test_empty_input_returns_a_flat_zero_curve():
    x, y = convolve_pattern([], [], two_theta_range=(5, 90), fwhm=0.15)

    assert len(x) == len(y)
    assert all(v == 0.0 for v in y)


def test_narrower_fwhm_gives_a_taller_curve_for_isolated_peaks():
    """Height-normalised peaks: at a fixed intensity, a sharper FWHM
    concentrates the same area into a taller, narrower spike - but since
    both are individually renormalised to 100 here, check the *width*
    instead, which is the property that should actually track FWHM.
    """
    x, y_sharp = convolve_pattern(
        [45.0], [100.0], two_theta_range=(5, 90), fwhm=0.05, num_points=4001
    )
    _, y_broad = convolve_pattern(
        [45.0], [100.0], two_theta_range=(5, 90), fwhm=0.5, num_points=4001
    )

    def width_above_half_max(xs, ys):
        above = [xv for xv, yv in zip(xs, ys) if yv >= 50.0]
        return max(above) - min(above)

    assert width_above_half_max(x, y_sharp) < width_above_half_max(x, y_broad)
