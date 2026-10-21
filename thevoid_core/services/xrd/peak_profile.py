from __future__ import annotations

import numpy as np

#: Default full width at half maximum, in degrees 2-theta. Roughly what
#: a well-aligned lab powder diffractometer resolves; a research
#: instrument might be sharper (0.05-0.1), a poorly aligned one broader
#: (0.3-0.5).
DEFAULT_FWHM = 0.15

#: Gaussian/Lorentzian mixing fraction. 0 is pure Gaussian, 1 is pure
#: Lorentzian. 0.5 is a reasonable general-purpose default; lab X-ray
#: line shapes are usually somewhere in the 0.3-0.7 range depending on
#: instrument and sample.
DEFAULT_ETA = 0.5

#: Points sampled across the requested 2-theta range. High enough that
#: a 0.05 deg FWHM peak (the sharpest offered) is still resolved by
#: several samples across its width over a typical 5-90 deg range.
DEFAULT_NUM_POINTS = 2001


def pseudo_voigt(
    x: np.ndarray,
    center: float,
    fwhm: float,
    eta: float = DEFAULT_ETA,
) -> np.ndarray:
    """Pseudo-Voigt line shape, normalised to a peak height of 1.

    Height-normalised rather than area-normalised: peak "intensity" in
    a calculated XRD pattern is conventionally a relative peak height
    (0-100), not an area, and stick plots and this profile need to
    agree on what "intensity" means when compared side by side.
    """
    if fwhm <= 0:
        raise ValueError(f"fwhm must be positive: {fwhm}")

    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    gamma = fwhm / 2.0

    gaussian = np.exp(-0.5 * ((x - center) / sigma) ** 2)
    lorentzian = gamma**2 / ((x - center) ** 2 + gamma**2)

    return eta * lorentzian + (1.0 - eta) * gaussian


def convolve_pattern(
    two_theta: list[float],
    intensities: list[float],
    *,
    two_theta_range: tuple[float, float],
    fwhm: float = DEFAULT_FWHM,
    eta: float = DEFAULT_ETA,
    num_points: int = DEFAULT_NUM_POINTS,
) -> tuple[list[float], list[float]]:
    """Convolve a calculated stick pattern into a continuous trace.

    Every reflection becomes a pseudo-Voigt peak of the given FWHM,
    summed together and renormalised so the tallest point is 100 -
    the same convention stick intensities use, so a profile and a stick
    plot of the same pattern are visually comparable. Overlapping peaks
    add, which is what happens on a real instrument and is exactly the
    effect a plain stick plot can't show.

    Pass the *full*, unfiltered reflection list, not a thresholded
    subset: a real trace's baseline and peak shoulders are built up from
    many small reflections that would individually fall below any
    sensible stick-plot cutoff, but still contribute to the curve.
    """
    x = np.linspace(two_theta_range[0], two_theta_range[1], num_points)

    if not two_theta:
        return list(x), [0.0] * num_points

    centers = np.asarray(two_theta, dtype=float)
    heights = np.asarray(intensities, dtype=float)

    # (num_points, num_peaks) - fine at these sizes (a few thousand
    # reflections and ~2000 points is a few million floats).
    profile = pseudo_voigt(x[:, None], centers[None, :], fwhm, eta)

    y = (profile * heights[None, :]).sum(axis=1)

    peak = y.max()

    if peak > 0:
        y = y * 100.0 / peak

    return list(x), list(y)


# ----------------------------------------------------------------------
# K-alpha doublet (round 50)
# ----------------------------------------------------------------------

#: Kalpha2 / Kalpha1 intensity ratio - the standard 1:2 from the 2p3/2
#: : 2p1/2 level occupancy, and what pymatgen's own weighted-average
#: "Ka" wavelengths assume ((2*Ka1 + Ka2) / 3).
KALPHA2_RATIO = 0.5


def kalpha_components(wavelength: float, *, tolerance: float = 5e-5):
    """If `wavelength` is one of the weighted-average K-alpha lines
    (pymatgen's "CuKa", "MoKa", ...), its (Ka1, Ka2) wavelengths;
    otherwise None - a Ka1-only or K-beta source has no doublet.
    """
    from pymatgen.analysis.diffraction.xrd import WAVELENGTHS

    for key, value in WAVELENGTHS.items():
        if key.endswith("Ka") and abs(value - wavelength) < tolerance:
            return WAVELENGTHS[key + "1"], WAVELENGTHS[key + "2"]

    return None


def split_kalpha_doublet(two_theta, intensities, wavelength: float):
    """Replace each reflection computed at the weighted-average K-alpha
    wavelength by its Ka1 and Ka2 components, at 2:1 intensity.

    A real Cu K-alpha tube emits both lines; every reflection therefore
    appears twice, split by 2*tan(theta)*(dlambda/lambda) - negligible
    at low angle, but 0.28 deg at 2-theta = 90 deg for Cu, about twice the
    default profile width. A trace drawn at the single averaged
    wavelength puts one peak midway between the two a diffractometer
    actually records. Returns the inputs unchanged when the wavelength
    has no doublet.
    """
    components = kalpha_components(wavelength)

    if components is None or not len(two_theta):
        return list(two_theta), list(intensities)

    lambda1, lambda2 = components
    theta = np.radians(np.asarray(two_theta, dtype=float) / 2.0)
    d = wavelength / (2.0 * np.sin(theta))

    out_two_theta = []
    out_intensity = []
    weight1 = 1.0 / (1.0 + KALPHA2_RATIO)
    weight2 = KALPHA2_RATIO / (1.0 + KALPHA2_RATIO)

    for spacing, intensity in zip(d, intensities):
        for lam, weight in ((lambda1, weight1), (lambda2, weight2)):
            s = lam / (2.0 * spacing)

            if s < 1.0:
                out_two_theta.append(float(np.degrees(2.0 * np.arcsin(s))))
                out_intensity.append(float(intensity) * weight)

    return out_two_theta, out_intensity
