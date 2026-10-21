"""Layout regressions for the XRD plot.

The legend used to sit at the axes' upper left, which is exactly where
the strongest peak and its label land on a typical diffractogram. These
tests assert on measured text extents, not on appearance.
"""

import os

import pytest

pytest.importorskip("matplotlib")
PySide6 = pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.xrd.xrd_result import XRDResult  # noqa: E402
from thevoid_core.ui.widgets.xrd_plot_widget import (  # noqa: E402
    XRDPlotWidget,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def pattern(two_theta, intensities):
    return XRDResult(
        two_theta=list(two_theta),
        intensities=list(intensities),
        hkls=[],
        d_spacings=[1.0] * len(two_theta),
        wavelength=1.5418,
    )


def reported_pattern():
    """The pattern from the bug report: a tall low-angle peak."""
    strong = [(11.03, 18), (13.83, 100), (14.33, 19), (20.39, 14), (33.60, 11)]

    two_theta = [t for t, _ in strong]
    intensities = [i for _, i in strong]

    position = 17.0
    while len(two_theta) < 110:
        two_theta.append(position)
        intensities.append(3.0)
        position += 0.7

    return pattern(two_theta, intensities)


def render(qt_app, result, full=None, size=(13, 5)):
    widget = XRDPlotWidget()
    widget.figure.set_size_inches(*size)
    widget.plot_pattern(result, full_result=full)

    return widget


def extents(widget):
    renderer = widget.figure.canvas.get_renderer()
    axes = widget.figure.axes[0]

    return (
        renderer,
        axes,
        [text.get_window_extent(renderer) for text in axes.texts],
    )


@pytest.mark.parametrize("size", [(13, 5), (7, 4), (18, 6)])
def test_no_label_overlaps_the_legend(qt_app, size):
    full = reported_pattern()
    widget = render(qt_app, full.reduced(10.0, 0.05), full, size)

    renderer, axes, boxes = extents(widget)
    legend = axes.get_legend().get_window_extent(renderer)

    assert not any(box.overlaps(legend) for box in boxes)


def test_legend_sits_below_the_axes(qt_app):
    full = reported_pattern()
    widget = render(qt_app, full.reduced(10.0, 0.05), full)

    renderer, axes, _ = extents(widget)

    assert (
        axes.get_legend().get_window_extent(renderer).y1
        <= axes.get_window_extent(renderer).y0
    )


@pytest.mark.parametrize("threshold", [0.0, 1.0, 5.0, 10.0])
def test_no_label_overlaps_another_label(qt_app, threshold):
    full = reported_pattern()
    widget = render(qt_app, full.reduced(threshold, 0.05), full)

    _, _, boxes = extents(widget)

    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            assert not box.overlaps(other)


def test_no_label_escapes_the_frame(qt_app):
    """The tallest peak's label must fit inside the headroom."""
    full = reported_pattern()
    widget = render(qt_app, full.reduced(10.0, 0.05), full)

    renderer, axes, boxes = extents(widget)
    frame = axes.get_window_extent(renderer)

    assert all(box.y1 <= frame.y1 + 0.5 for box in boxes)


def test_close_peaks_at_different_heights_both_keep_labels(qt_app):
    """13.83 and 14.33 are near in x but far apart in y.

    A fixed 2-theta gap rule would throw one away; measuring extents
    keeps both.
    """
    full = reported_pattern()
    widget = render(qt_app, full.reduced(10.0, 0.05), full)

    _, axes, _ = extents(widget)
    labels = {text.get_text() for text in axes.texts}

    assert {"13.83", "14.33"} <= labels


def test_adjacent_equal_peaks_lose_a_label_rather_than_collide(qt_app):
    """When peaks are close in both axes, the weaker label is dropped."""
    result = pattern([20.0, 20.02, 20.04], [100, 100, 100])

    widget = render(qt_app, result)

    _, _, boxes = extents(widget)

    for i, box in enumerate(boxes):
        for other in boxes[i + 1:]:
            assert not box.overlaps(other)


def test_wavelength_moved_into_the_title(qt_app):
    """It used to be an in-axes annotation at the upper right."""
    result = pattern([10.0, 20.0], [100, 50])

    widget = render(qt_app, result)

    assert "1.5418" in widget.figure.axes[0].get_title()


def test_label_count_is_capped(qt_app):
    result = pattern([10.0 + i for i in range(40)], [100 - i for i in range(40)])

    widget = render(qt_app, result)

    assert len(widget.figure.axes[0].texts) <= XRDPlotWidget.LABEL_COUNT
