from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget


class XRDPlotWidget(QWidget):
    """Powder XRD stick pattern.

    The full calculated pattern is drawn faintly behind the retained
    peaks, so raising the intensity threshold never hides the fact that
    weaker reflections exist.

    Nothing is drawn inside the axes except data. The legend sits below
    the plot and the wavelength goes in the title, because the upper
    left of a diffractogram is exactly where the strongest peak and its
    label tend to be.
    """

    FULL_COLOUR = "#c3ccd8"
    PEAK_COLOUR = "#1f4e79"

    LABEL_COUNT = 8
    LABEL_FONTSIZE = 7

    #: How many candidates to lay out before pruning. Labels are pruned
    #: by measured extent, so this only bounds the work.
    LABEL_POOL = 40

    #: Fallback gap between labels as a fraction of the 2-theta range,
    #: used only when the renderer cannot be queried for text extents.
    LABEL_MIN_SEPARATION = 0.014

    #: Headroom above the tallest peak, leaving room for its label.
    HEADROOM_WITH_LABELS = 1.26
    HEADROOM_PLAIN = 1.06

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    PROFILE_COLOUR = "#1f4e79"
    PROFILE_FILL_ALPHA = 0.15

    def plot_pattern(self, result, full_result=None, label_peaks=True):
        self.figure.clear()

        axes = self.figure.add_subplot(111)

        show_full = full_result is not None and full_result is not result

        if show_full:
            axes.vlines(
                full_result.two_theta,
                0,
                full_result.intensities,
                linewidth=0.8,
                color=self.FULL_COLOUR,
                label=f"all {full_result.peak_count} reflections",
            )

        axes.vlines(
            result.two_theta,
            0,
            result.intensities,
            linewidth=1.4,
            color=self.PEAK_COLOUR,
            label=self._retained_label(result),
        )

        ceiling = max(
            result.max_intensity,
            full_result.max_intensity if show_full else 0.0,
            1.0,
        )

        wants_labels = bool(label_peaks and result.peak_count)

        axes.set_ylim(
            bottom=0,
            top=ceiling
            * (
                self.HEADROOM_WITH_LABELS
                if wants_labels
                else self.HEADROOM_PLAIN
            ),
        )

        axes.set_xlabel("2\u03b8 (\u00b0)")
        axes.set_ylabel("Relative Intensity")
        axes.set_title(self._title(result))

        # Below the axes, so a tall peak can never collide with it.
        axes.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            fontsize=8,
            frameon=False,
        )

        self.figure.tight_layout()

        if wants_labels:
            # Geometry must be settled before text extents mean
            # anything, so lay the labels out against the final axes.
            self.canvas.draw()
            self._prune_labels(axes, self._place_labels(axes, result))

        self.canvas.draw()

    def plot_profile(self, two_theta, intensities, *, wavelength=None, fwhm=None):
        """Draw a continuous pseudo-Voigt trace instead of sticks.

        Overlapping reflections sum here the way they would on a real
        instrument, which a stick plot can't show - the reason to offer
        this view at all. No peak labels or legend: with peaks free to
        merge into shoulders and humps, individually labelling "peaks"
        stops being well-defined the way it is for discrete sticks.
        """
        self.figure.clear()

        axes = self.figure.add_subplot(111)

        axes.plot(
            two_theta,
            intensities,
            linewidth=1.2,
            color=self.PROFILE_COLOUR,
        )
        axes.fill_between(
            two_theta,
            0,
            intensities,
            color=self.PROFILE_COLOUR,
            alpha=self.PROFILE_FILL_ALPHA,
            linewidth=0,
        )

        ceiling = max(intensities) if intensities else 1.0

        axes.set_ylim(bottom=0, top=ceiling * self.HEADROOM_PLAIN)

        if two_theta:
            axes.set_xlim(min(two_theta), max(two_theta))

        axes.set_xlabel("2\u03b8 (\u00b0)")
        axes.set_ylabel("Relative Intensity")

        title = "Powder XRD Pattern (simulated trace)"

        if wavelength:
            title += f"  (\u03bb = {wavelength:.4f} \u00c5"
            title += f", FWHM = {fwhm:g}\u00b0)" if fwhm else ")"

        axes.set_title(title)

        self.figure.tight_layout()
        self.canvas.draw()

    @staticmethod
    def _title(result) -> str:
        wavelength = getattr(result, "wavelength", None)

        if not wavelength:
            return "Powder XRD Pattern"

        return (
            f"Powder XRD Pattern  (\u03bb = {wavelength:.4f} \u00c5)"
        )

    @staticmethod
    def _retained_label(result) -> str:
        threshold = getattr(result, "min_relative_intensity", 0) or 0

        if threshold <= 0:
            return f"{result.peak_count} peaks"

        return f"{result.peak_count} peaks \u2265 {threshold:g}%"

    def _place_labels(self, axes, result) -> list:
        """Lay out candidate labels, strongest peak first."""
        candidates = sorted(
            range(result.peak_count),
            key=lambda i: result.intensities[i],
            reverse=True,
        )[: self.LABEL_POOL]

        return [
            axes.annotate(
                f"{result.two_theta[index]:.2f}",
                xy=(result.two_theta[index], result.intensities[index]),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=self.LABEL_FONTSIZE,
                rotation=90,
                color=self.PEAK_COLOUR,
            )
            for index in candidates
        ]

    def _prune_labels(self, axes, annotations) -> int:
        """Drop labels that overlap an already-kept one or the frame.

        Candidates arrive strongest-first, so when two peaks are too
        close to label separately the more intense one keeps its label.

        Overlap is tested on measured text extents rather than on a
        fixed 2-theta gap. That matters for a pair like 13.83 and 14.33:
        they are close enough in x that any reasonable gap rule would
        discard one, but their labels sit at very different heights
        because the peaks differ in intensity, so in practice they do
        not touch and both can be kept.
        """
        renderer = self._renderer()

        if renderer is None:
            return self._prune_by_gap(annotations)

        frame = axes.get_window_extent(renderer)

        kept: list = []

        for annotation in annotations:
            if len(kept) >= self.LABEL_COUNT:
                annotation.remove()
                continue

            box = annotation.get_window_extent(renderer).expanded(1.1, 1.1)

            if box.y1 > frame.y1 or any(box.overlaps(other) for other in kept):
                annotation.remove()
                continue

            kept.append(box)

        return len(kept)

    def _prune_by_gap(self, annotations) -> int:
        """Fallback when no renderer is available: fixed 2-theta gap."""
        positions = [a.xy[0] for a in annotations]

        if not positions:
            return 0

        span = (max(positions) - min(positions)) or 1.0
        minimum_gap = span * self.LABEL_MIN_SEPARATION

        kept: list[float] = []

        for annotation in annotations:
            position = annotation.xy[0]

            if len(kept) >= self.LABEL_COUNT or any(
                abs(position - other) < minimum_gap for other in kept
            ):
                annotation.remove()
                continue

            kept.append(position)

        return len(kept)

    def _renderer(self):
        try:
            return self.figure.canvas.get_renderer()
        except Exception:
            return None

    def export_png(self, filename: str, dpi: int = 300) -> None:
        self.figure.savefig(filename, dpi=dpi, bbox_inches="tight")
