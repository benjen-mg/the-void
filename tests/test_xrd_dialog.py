"""XRDDialog: the results window - style switching, filtering, the peak
table, and CSV/PNG export.

The export half is the part worth real coverage. The dialog decides
between two different exporter entry points depending on which style is
showing, and passes `full_result.wavelength` straight through to a
writer that formats it as a float - a mismatch there would only surface
when somebody clicked Export CSV on the profile view.
"""

import os

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QMessageBox,
)

from thevoid_core.services.xrd.peak_profile import DEFAULT_FWHM  # noqa: E402
from thevoid_core.services.xrd.xrd_result import XRDResult  # noqa: E402
from thevoid_core.ui.dialogs.xrd_dialog import (  # noqa: E402
    PROFILE,
    STICKS,
    XRDDialog,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def result():
    return XRDResult(
        two_theta=[10.0, 12.0, 20.5, 35.0, 60.0],
        intensities=[100.0, 3.0, 45.0, 0.4, 12.0],
        hkls=[
            [{"hkl": (1, 0, 0)}],
            [{"hkl": (1, 1, 0)}],
            [{"hkl": (2, 0, 0)}],
            [{"hkl": (2, 1, 1)}],
            [{"hkl": (3, 1, 0)}],
        ],
        d_spacings=[8.84, 7.37, 4.33, 2.56, 1.54],
        wavelength=1.54184,
        two_theta_range=(5.0, 90.0),
    )


@pytest.fixture
def dialog(qt_app, result):
    widget = XRDDialog(result)
    yield widget
    widget.close()


def _stub_save_dialog(monkeypatch, path):
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: (str(path), "")),
    )


def _select(combo, data):
    combo.setCurrentIndex(combo.findData(data))


def test_dialog_opens_on_the_stick_view(dialog):
    assert dialog.style_box.currentData() == STICKS
    assert not dialog.fwhm_box.isVisible()


def test_the_peak_table_is_populated(dialog):
    assert dialog.table.rowCount() == dialog.result.peak_count
    assert dialog.table.columnCount() == 4


def test_filtering_reduces_the_table_without_touching_the_source(
    dialog, result
):
    """Filtering is pure post-processing on an already-simulated
    pattern - the full result must survive it, since loosening the
    threshold again has to bring the peaks back without recalculating.
    """
    _select(dialog.threshold_box, 10.0)

    assert dialog.result.peak_count < result.peak_count
    assert dialog.full_result.peak_count == 5


def test_loosening_the_threshold_restores_every_peak(dialog):
    _select(dialog.threshold_box, 10.0)
    _select(dialog.threshold_box, 0.0)

    assert dialog.result.peak_count == 5


def test_switching_to_profile_computes_a_continuous_trace(dialog):
    _select(dialog.style_box, PROFILE)

    assert dialog._profile_two_theta is not None
    assert len(dialog._profile_two_theta) > dialog.full_result.peak_count
    assert dialog._profile_fwhm == pytest.approx(DEFAULT_FWHM)


def test_switching_back_to_sticks_discards_the_trace(dialog):
    """Stale profile state would make Export CSV write a trace while
    the person is looking at sticks.
    """
    _select(dialog.style_box, PROFILE)
    _select(dialog.style_box, STICKS)

    assert dialog._profile_two_theta is None
    assert dialog._profile_fwhm is None


def test_the_fwhm_control_appears_only_for_the_profile_view(dialog):
    _select(dialog.style_box, PROFILE)
    assert dialog.fwhm_label.isVisibleTo(dialog)

    _select(dialog.style_box, STICKS)
    assert not dialog.fwhm_label.isVisibleTo(dialog)


def test_changing_fwhm_recomputes_the_trace(dialog):
    _select(dialog.style_box, PROFILE)
    dialog.fwhm_box.setValue(0.5)

    assert dialog._profile_fwhm == pytest.approx(0.5)


def test_the_export_button_says_what_it_will_write(dialog):
    assert "peaks" in dialog.export_csv_button.text()

    _select(dialog.style_box, PROFILE)

    assert "profile points" in dialog.export_csv_button.text()


def test_stick_export_writes_the_peak_list(dialog, monkeypatch, tmp_path):
    out_path = tmp_path / "peaks.csv"
    _stub_save_dialog(monkeypatch, out_path)

    dialog.export_csv()

    lines = out_path.read_text().splitlines()

    assert any(line.startswith("# wavelength_A=") for line in lines)
    assert len(lines) > dialog.result.peak_count


def test_profile_export_writes_every_sampled_point(
    dialog, monkeypatch, tmp_path
):
    """The regression this file was written for: the profile branch
    hands the exporter full_result.wavelength, which is formatted with
    a float format code. A string there raises ValueError at write
    time, on a path no other test exercised.
    """
    out_path = tmp_path / "profile.csv"
    _select(dialog.style_box, PROFILE)
    _stub_save_dialog(monkeypatch, out_path)

    dialog.export_csv()

    text = out_path.read_text()

    assert "# wavelength_A=1.541840" in text
    assert "fwhm" in text.lower()
    assert len(text.splitlines()) > len(dialog._profile_two_theta)


def test_cancelling_the_save_dialog_writes_nothing(
    dialog, monkeypatch, tmp_path
):
    out_path = tmp_path / "unwanted.csv"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )

    dialog.export_csv()

    assert not out_path.exists()


def test_a_missing_extension_is_appended(dialog, monkeypatch, tmp_path):
    out_path = tmp_path / "noextension"
    _stub_save_dialog(monkeypatch, out_path)

    dialog.export_csv()

    assert (tmp_path / "noextension.csv").exists()


def test_a_dot_in_the_stem_is_not_mistaken_for_an_extension(
    dialog, monkeypatch, tmp_path
):
    """'sample.v2' must become 'sample.v2.csv', not 'sample.csv'."""
    out_path = tmp_path / "sample.v2"
    _stub_save_dialog(monkeypatch, out_path)

    dialog.export_csv()

    assert (tmp_path / "sample.v2.csv").exists()
    assert not (tmp_path / "sample.csv").exists()


def test_png_export_writes_an_image(dialog, monkeypatch, tmp_path):
    out_path = tmp_path / "pattern.png"
    _stub_save_dialog(monkeypatch, out_path)

    dialog.export_png()

    assert out_path.exists()
    assert out_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_an_export_failure_is_reported_rather_than_raised(
    dialog, monkeypatch, tmp_path
):
    shown = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        staticmethod(lambda *a, **k: shown.append(a)),
    )
    _stub_save_dialog(monkeypatch, tmp_path / "locked.png")

    def exploding_export(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dialog.plot, "export_png", exploding_export)

    dialog.export_png()

    assert shown
