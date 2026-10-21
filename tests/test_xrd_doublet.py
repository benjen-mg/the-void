"""Round 50: K-alpha1/K-alpha2 doublet in the profile view."""

import math
import os

import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services.xrd.peak_profile import (  # noqa: E402
    KALPHA2_RATIO,
    convolve_pattern,
    kalpha_components,
    split_kalpha_doublet,
)

CU_KA = 1.54184
CU_KA1 = 1.54056
CU_KA2 = 1.54439


def test_averaged_kalpha_sources_have_a_doublet():
    assert kalpha_components(CU_KA) == (CU_KA1, CU_KA2)
    assert kalpha_components(0.71073) is not None  # Mo Ka


@pytest.mark.parametrize("wavelength", [CU_KA1, 1.39222, 1.2345])
def test_single_line_sources_have_none(wavelength):
    assert kalpha_components(wavelength) is None
    assert split_kalpha_doublet([30.0], [100.0], wavelength) == (
        [30.0], [100.0]
    )


def test_the_average_wavelength_is_the_2_to_1_weighted_mean():
    assert (2 * CU_KA1 + CU_KA2) / 3 == pytest.approx(CU_KA, abs=1e-5)


def test_split_positions_follow_braggs_law_and_conserve_intensity():
    two_theta, intensity = split_kalpha_doublet([90.0], [300.0], CU_KA)

    d = CU_KA / (2 * math.sin(math.radians(45.0)))
    expected1 = 2 * math.degrees(math.asin(CU_KA1 / (2 * d)))
    expected2 = 2 * math.degrees(math.asin(CU_KA2 / (2 * d)))

    assert two_theta == pytest.approx([expected1, expected2])
    assert intensity == pytest.approx([200.0, 100.0])
    assert sum(intensity) == pytest.approx(300.0)
    # 0.28 deg apart at 90 deg - about twice the default profile width.
    assert expected2 - expected1 == pytest.approx(0.2848, abs=5e-4)
    assert KALPHA2_RATIO == 0.5


def test_doublet_is_resolved_at_high_angle_in_the_trace():
    positions, heights = split_kalpha_doublet([90.0], [100.0], CU_KA)
    x, y = convolve_pattern(
        positions, heights, two_theta_range=(89.0, 91.0), fwhm=0.1,
        num_points=4001,
    )

    peaks = [
        i for i in range(1, len(y) - 1)
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > 10
    ]
    assert len(peaks) == 2


def test_dialog_offers_the_doublet_only_for_averaged_sources():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    from PySide6.QtWidgets import QApplication

    from thevoid_core.services.xrd.xrd_result import XRDResult
    from thevoid_core.ui.dialogs.xrd_dialog import XRDDialog

    QApplication.instance() or QApplication([])

    def result(wavelength):
        return XRDResult(
            two_theta=[30.0, 90.0], intensities=[100.0, 50.0],
            hkls=[[{"hkl": (1, 0, 0), "multiplicity": 6}]] * 2,
            d_spacings=[2.97, 1.09], wavelength=wavelength,
            two_theta_range=(5.0, 100.0),
        )

    averaged = XRDDialog(result(CU_KA))
    assert averaged.doublet_checkbox.isEnabled()
    assert averaged.doublet_checkbox.isChecked()

    single = XRDDialog(result(CU_KA1))
    assert not single.doublet_checkbox.isEnabled()
    assert not single.doublet_checkbox.isChecked()
