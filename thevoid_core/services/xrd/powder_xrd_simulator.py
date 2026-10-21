from __future__ import annotations

from pymatgen.analysis.diffraction.xrd import (
    WAVELENGTHS,
    XRDCalculator,
)

from thevoid_core.services.xrd.xrd_result import XRDResult

DEFAULT_WAVELENGTH = "CuKa"

DEFAULT_TWO_THETA_RANGE = (5.0, 90.0)

#: Common lab radiation sources, display label to pymatgen wavelength
#: key. Kα (unresolved Kα1/Kα2 average) is what most lab powder XRD
#: reports; the individually resolved Kα1/Kα2/Kβ lines are offered for
#: comparing against high-resolution data.
RADIATION_SOURCES: dict[str, str] = {
    "Cu K\u03b1 (1.5418 \u00c5)": "CuKa",
    "Cu K\u03b1\u2081 (1.5406 \u00c5)": "CuKa1",
    "Cu K\u03b2 (1.3922 \u00c5)": "CuKb1",
    "Mo K\u03b1 (0.7107 \u00c5)": "MoKa",
    "Co K\u03b1 (1.7903 \u00c5)": "CoKa",
    "Fe K\u03b1 (1.9373 \u00c5)": "FeKa",
    "Cr K\u03b1 (2.2910 \u00c5)": "CrKa",
    "Ag K\u03b1 (0.5609 \u00c5)": "AgKa",
}


class PowderXRDSimulator:
    """Powder XRD pattern simulation via pymatgen's XRDCalculator."""

    def __init__(self, wavelength: str = DEFAULT_WAVELENGTH) -> None:
        self.default_wavelength = wavelength
        self._calculators: dict[str, XRDCalculator] = {
            wavelength: XRDCalculator(wavelength=wavelength)
        }

    def _calculator(self, wavelength: str) -> XRDCalculator:
        """Cache one XRDCalculator per radiation source used so far.

        Construction is cheap, but simulate() may be called repeatedly
        (e.g. re-running with a different 2-theta range) with the same
        source, so there is no reason to rebuild it every time.
        """
        calculator = self._calculators.get(wavelength)

        if calculator is None:
            calculator = XRDCalculator(wavelength=wavelength)
            self._calculators[wavelength] = calculator

        return calculator

    @staticmethod
    def wavelength_to_angstrom(wavelength) -> float:
        """Resolve a wavelength name such as 'CuKa' to angstrom."""
        if isinstance(wavelength, str):
            return float(WAVELENGTHS[wavelength])

        return float(wavelength)

    def simulate(
        self,
        structure,
        two_theta_range: tuple[float, float] = DEFAULT_TWO_THETA_RANGE,
        wavelength: str | None = None,
    ) -> XRDResult:
        """Simulate a powder pattern.

        wavelength names a pymatgen radiation source key (e.g. 'CuKa',
        'MoKa'); None uses the source this simulator was constructed
        with. two_theta_range is (min, max) in degrees.
        """
        if two_theta_range[0] >= two_theta_range[1]:
            raise ValueError(
                f"two_theta_range must be (min, max) with min < max: "
                f"{two_theta_range}"
            )

        resolved_wavelength = wavelength or self.default_wavelength

        calculator = self._calculator(resolved_wavelength)

        pattern = calculator.get_pattern(
            structure,
            two_theta_range=two_theta_range,
        )

        return XRDResult(
            two_theta=list(pattern.x),
            intensities=list(pattern.y),
            hkls=list(pattern.hkls),
            d_spacings=list(pattern.d_hkls),
            wavelength=self.wavelength_to_angstrom(calculator.wavelength),
            two_theta_range=tuple(two_theta_range),
        )
