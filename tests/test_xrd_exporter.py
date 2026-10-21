import csv
from dataclasses import dataclass

from thevoid_core.exports.xrd_exporter import XRDExporter


@dataclass
class FakeResult:
    two_theta: list
    intensities: list
    d_spacings: list
    hkls: list
    wavelength: float


def test_format_hkl_handles_pymatgen_payload():
    payload = [{"hkl": (1, 1, 1), "multiplicity": 8}]

    assert XRDExporter.format_hkl(payload) == "(1 1 1)"


def test_format_hkl_handles_overlapping_reflections():
    payload = [{"hkl": (3, 0, 0)}, {"hkl": (2, 2, 1)}]

    assert XRDExporter.format_hkl(payload) == "(3 0 0), (2 2 1)"


def test_format_hkl_handles_missing_data():
    assert XRDExporter.format_hkl(None) == ""


def test_export_csv_writes_expected_rows(tmp_path):
    result = FakeResult(
        two_theta=[28.44, 47.30],
        intensities=[100.0, 55.2],
        d_spacings=[3.1355, 1.9201],
        hkls=[[{"hkl": (1, 1, 1)}], [{"hkl": (2, 2, 0)}]],
        wavelength=1.54184,
    )

    target = tmp_path / "pattern.csv"
    XRDExporter.export_csv(target, result)

    rows = list(csv.reader(target.open(encoding="utf-8")))

    assert rows[0][0].startswith("# wavelength_A=")
    assert rows[1] == ["2theta_deg", "rel_intensity_pct", "d_spacing_A", "hkl"]
    assert rows[2][3] == "(1 1 1)"
    assert len(rows) == 4


def test_export_profile_csv_writes_all_points(tmp_path):
    two_theta = [10.0, 10.05, 10.1, 10.15, 10.2]
    intensities = [0.0, 40.0, 100.0, 40.0, 0.0]

    target = tmp_path / "profile.csv"
    XRDExporter.export_profile_csv(
        target, two_theta, intensities, wavelength=1.5406, fwhm=0.15
    )

    rows = list(csv.reader(target.open(encoding="utf-8")))

    assert rows[0][0] == "# wavelength_A=1.540600"
    assert rows[1][0] == "# fwhm_deg_2theta=0.15"
    assert rows[2][0] == "# points=5"
    assert rows[3] == ["2theta_deg", "rel_intensity_pct"]
    # One row per sampled point, not just peaks - 5 data rows for 5
    # points, unlike the stick exporter which only ever writes peaks.
    assert len(rows) == 4 + 5
    assert rows[4] == ["10.0", "0.0"]
    assert rows[6] == ["10.1", "100.0"]


def test_export_profile_csv_omits_absent_metadata(tmp_path):
    target = tmp_path / "profile_no_meta.csv"
    XRDExporter.export_profile_csv(target, [1.0, 2.0], [0.0, 1.0])

    rows = list(csv.reader(target.open(encoding="utf-8")))

    # No wavelength or fwhm rows when neither was supplied.
    assert rows[0][0] == "# points=2"
    assert rows[1] == ["2theta_deg", "rel_intensity_pct"]
