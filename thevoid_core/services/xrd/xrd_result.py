from __future__ import annotations

from dataclasses import dataclass, field

#: Peaks weaker than this share of the strongest are dropped by default.
#:
#: Conventions, for reference:
#:   0.1 %  - everything the calculation produced; unusable as a table
#:   1   %  - the completeness cutoff used by ICDD PDF entries, and what
#:            a published "full" peak list normally reports
#:   5   %  - the dominant lines: what you would quote to identify a
#:            phase or annotate a plot
#:  10   %  - only the handful of defining reflections
#:
#: 5 % is the default because a low-symmetry cell (P2_1/c with 56 sites
#: generates many hundreds of reflections) is unreadable at 1 %, and
#: almost everything below 5 % is unresolvable on a real diffractogram
#: anyway.
DEFAULT_MIN_RELATIVE_INTENSITY = 5.0

#: Reflections closer than this in 2-theta are summed into one peak.
#:
#: Distinct hkl can be symmetry-inequivalent yet land within a few
#: thousandths of a degree. No powder diffractometer resolves them, so
#: listing them separately inflates the peak count without adding
#: information. 0.05 deg is a typical instrumental FWHM floor.
DEFAULT_MERGE_TOLERANCE = 0.05


@dataclass(slots=True)
class XRDResult:
    two_theta: list[float]
    intensities: list[float]

    hkls: list
    d_spacings: list[float]

    wavelength: float

    #: Provenance of any filtering already applied, so exports can say so.
    min_relative_intensity: float = 0.0
    merge_tolerance: float = 0.0
    unfiltered_peak_count: int = field(default=0)

    #: The 2-theta window the calculation was run over. Needed by
    #: anything that draws or recomputes over the full range rather than
    #: just the peak positions - the profile view in particular, since a
    #: continuous curve needs bounds, not just a list of peak centres.
    two_theta_range: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.unfiltered_peak_count:
            self.unfiltered_peak_count = len(self.two_theta)

    @property
    def peak_count(self) -> int:
        return len(self.two_theta)

    @property
    def max_intensity(self) -> float:
        return max(self.intensities) if self.intensities else 0.0

    @property
    def is_filtered(self) -> bool:
        return self.peak_count < self.unfiltered_peak_count

    @property
    def effective_two_theta_range(self) -> tuple[float, float]:
        """The calculation's 2-theta window, with a fallback.

        Falls back to the span of the peaks actually present, for
        results built by hand (e.g. in tests) without the field set.
        """
        if self.two_theta_range is not None:
            return self.two_theta_range

        if not self.two_theta:
            return (0.0, 0.0)

        return (min(self.two_theta), max(self.two_theta))

    # ------------------------------------------------------------------
    # Reduction
    # ------------------------------------------------------------------

    def reduced(
        self,
        min_relative_intensity: float = DEFAULT_MIN_RELATIVE_INTENSITY,
        merge_tolerance: float = DEFAULT_MERGE_TOLERANCE,
    ) -> "XRDResult":
        """Merge unresolvable reflections, then drop the weak ones.

        Order matters: merging first lets a cluster of individually
        negligible reflections survive as one observable peak, which is
        what a diffractometer would actually record.
        """
        return self.merged(merge_tolerance).above(min_relative_intensity)

    def merged(
        self,
        tolerance: float = DEFAULT_MERGE_TOLERANCE,
    ) -> "XRDResult":
        if tolerance <= 0 or self.peak_count < 2:
            return self._with(
                self.two_theta,
                self.intensities,
                self.hkls,
                self.d_spacings,
                merge_tolerance=tolerance,
            )

        order = sorted(range(self.peak_count), key=lambda i: self.two_theta[i])

        groups: list[list[int]] = [[order[0]]]

        for index in order[1:]:
            if (
                self.two_theta[index] - self.two_theta[groups[-1][-1]]
                <= tolerance
            ):
                groups[-1].append(index)
            else:
                groups.append([index])

        two_theta: list[float] = []
        intensities: list[float] = []
        hkls: list = []
        d_spacings: list[float] = []

        for group in groups:
            total = sum(self.intensities[i] for i in group)

            if total > 0:
                centroid = (
                    sum(self.two_theta[i] * self.intensities[i] for i in group)
                    / total
                )
            else:
                centroid = sum(self.two_theta[i] for i in group) / len(group)

            strongest = max(group, key=lambda i: self.intensities[i])

            two_theta.append(centroid)
            intensities.append(total)
            d_spacings.append(self.d_spacings[strongest])
            hkls.append(self._combine_hkls(group))

        return self._with(
            two_theta,
            self._renormalised(intensities),
            hkls,
            d_spacings,
            merge_tolerance=tolerance,
        )

    def above(self, min_relative_intensity: float) -> "XRDResult":
        """Keep peaks at or above a percentage of the strongest."""
        if min_relative_intensity <= 0 or not self.intensities:
            return self._with(
                self.two_theta,
                self.intensities,
                self.hkls,
                self.d_spacings,
                min_relative_intensity=min_relative_intensity,
            )

        threshold = self.max_intensity * min_relative_intensity / 100.0

        keep = [
            i
            for i, intensity in enumerate(self.intensities)
            if intensity >= threshold
        ]

        return self._with(
            [self.two_theta[i] for i in keep],
            [self.intensities[i] for i in keep],
            [self.hkls[i] for i in keep] if self.hkls else [],
            [self.d_spacings[i] for i in keep],
            min_relative_intensity=min_relative_intensity,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _with(
        self,
        two_theta,
        intensities,
        hkls,
        d_spacings,
        *,
        min_relative_intensity: float | None = None,
        merge_tolerance: float | None = None,
    ) -> "XRDResult":
        return XRDResult(
            two_theta=list(two_theta),
            intensities=list(intensities),
            hkls=list(hkls),
            d_spacings=list(d_spacings),
            wavelength=self.wavelength,
            min_relative_intensity=(
                self.min_relative_intensity
                if min_relative_intensity is None
                else min_relative_intensity
            ),
            merge_tolerance=(
                self.merge_tolerance
                if merge_tolerance is None
                else merge_tolerance
            ),
            unfiltered_peak_count=self.unfiltered_peak_count,
            two_theta_range=self.two_theta_range,
        )

    def _combine_hkls(self, group) -> list:
        combined: list = []

        for index in group:
            entry = self.hkls[index] if index < len(self.hkls) else None

            if entry is None:
                continue

            if isinstance(entry, (list, tuple)):
                combined.extend(entry)
            else:
                combined.append(entry)

        return combined

    @staticmethod
    def _renormalised(intensities: list[float]) -> list[float]:
        """Rescale so the strongest peak is 100 again after summing."""
        peak = max(intensities) if intensities else 0.0

        if peak <= 0:
            return list(intensities)

        return [value * 100.0 / peak for value in intensities]
