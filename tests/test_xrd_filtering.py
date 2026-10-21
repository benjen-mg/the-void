"""Peak reduction: merging unresolvable reflections, then thresholding.

A low-symmetry cell produces many hundreds of reflections, most of them
negligible and most of the rest too close together for any instrument to
separate. These tests pin the reduction behaviour.
"""

import pytest

from thevoid_core.services.xrd.xrd_result import (
    DEFAULT_MERGE_TOLERANCE,
    DEFAULT_MIN_RELATIVE_INTENSITY,
    XRDResult,
)


def make(two_theta, intensities, d_spacings=None, hkls=None):
    return XRDResult(
        two_theta=list(two_theta),
        intensities=list(intensities),
        hkls=hkls if hkls is not None else [],
        d_spacings=list(d_spacings or [1.0] * len(two_theta)),
        wavelength=1.5406,
    )


def test_threshold_keeps_peaks_at_or_above_the_cutoff():
    result = make([10, 20, 30, 40], [100, 20, 5, 1])

    kept = result.above(5.0)

    assert kept.two_theta == [10, 20, 30]
    assert kept.min_relative_intensity == 5.0


def test_threshold_is_relative_not_absolute():
    """Halving every intensity must not change which peaks survive."""
    a = make([10, 20, 30], [100, 20, 1]).above(5.0)
    b = make([10, 20, 30], [50, 10, 0.5]).above(5.0)

    assert a.two_theta == b.two_theta


def test_no_filtering_when_threshold_is_zero():
    result = make([10, 20, 30], [100, 1, 0.01])

    assert result.above(0).peak_count == 3


def test_unresolvable_reflections_are_summed():
    """Two reflections 0.01 deg apart are one observed peak."""
    result = make([20.00, 20.01, 40.0], [30, 30, 100])

    merged = result.merged(0.05)

    assert merged.peak_count == 2
    # 60 summed vs 100 renormalises to 60.
    assert merged.intensities[0] == pytest.approx(60.0)
    assert merged.two_theta[0] == pytest.approx(20.005)


def test_merge_uses_intensity_weighted_centroid():
    result = make([20.00, 20.04], [90, 10])

    merged = result.merged(0.05)

    assert merged.two_theta[0] == pytest.approx(20.004)


def test_merging_before_thresholding_rescues_a_real_peak():
    """Order matters.

    Five reflections at 4% each are individually below a 5% cutoff, but
    together they form a 20% peak that a diffractometer would record.
    """
    two_theta = [30.00, 30.01, 30.02, 30.03, 30.04, 50.0]
    intensities = [4, 4, 4, 4, 4, 100]

    threshold_first = make(two_theta, intensities).above(5.0)
    merge_first = make(two_theta, intensities).reduced(5.0, 0.05)

    assert threshold_first.peak_count == 1
    assert merge_first.peak_count == 2


def test_merge_tolerance_of_zero_is_a_no_op():
    result = make([20.00, 20.01], [50, 50])

    assert result.merged(0).peak_count == 2


def test_reduction_records_its_own_provenance():
    result = make(list(range(20)), [100] + [1] * 19)

    reduced = result.reduced(5.0, 0.05)

    assert reduced.unfiltered_peak_count == 20
    assert reduced.is_filtered
    assert reduced.min_relative_intensity == 5.0
    assert reduced.merge_tolerance == 0.05


def test_defaults_are_the_documented_conventions():
    assert DEFAULT_MIN_RELATIVE_INTENSITY == 5.0
    assert DEFAULT_MERGE_TOLERANCE == 0.05


def test_hkls_survive_a_merge():
    result = make(
        [20.00, 20.01],
        [50, 50],
        hkls=[[{"hkl": (1, 1, 1)}], [{"hkl": (2, 0, 0)}]],
    )

    merged = result.merged(0.05)

    assert len(merged.hkls[0]) == 2


def test_empty_pattern_is_handled():
    empty = make([], [])

    assert empty.reduced().peak_count == 0
    assert empty.max_intensity == 0.0


def test_two_theta_range_survives_filtering():
    result = make([10, 20, 30], [100, 5, 50])
    result.two_theta_range = (5.0, 90.0)

    reduced = result.reduced(10.0, 0.05)

    assert reduced.two_theta_range == (5.0, 90.0)


def test_effective_two_theta_range_falls_back_to_peak_span():
    result = make([12.0, 45.0, 8.0], [10, 10, 10])

    assert result.effective_two_theta_range == (8.0, 45.0)


def test_effective_two_theta_range_uses_explicit_value_when_set():
    result = make([12.0, 45.0], [10, 10])
    result.two_theta_range = (5.0, 90.0)

    assert result.effective_two_theta_range == (5.0, 90.0)


def test_effective_two_theta_range_handles_empty_pattern():
    empty = make([], [])

    assert empty.effective_two_theta_range == (0.0, 0.0)
