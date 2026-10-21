"""Round 54: loading, filtering and ranking a the-void-analyze results
folder, and finding a row's structure again
(services/batch_results.py).

A screening table that ranks a failed analysis at the top, or opens a
different file than the one a row's numbers came from, is worse than
the spreadsheet it replaces. So the checks here are about exactly
those: empty cells never pass a numeric filter and always sort last;
ties keep file order; a structure is only accepted when its SHA-256
matches the report.
"""

import csv
import json
import shutil
from pathlib import Path

import pytest

from thevoid_core.services.batch_results import (
    Filter,
    check_file,
    load_batch_results,
    locate_structure,
    report_markdown,
    view,
    write_view_csv,
)
from thevoid_core.services.report import file_sha256

from tests.sample_structures import ZN_DIA  # noqa: E402

FIXTURE = ZN_DIA  # an open Zn MOF

COLUMNS = ["file", "formula", "space_group", "topology", "lcd_A", "pld_A",
           "channel_asa_m2_per_g", "open_metal_sites", "guest_CO2", "errors"]
ROWS = [
    ["a.cif", "ZnC", "P1", "pcu 6-c", "7.5", "4.1", "1200", "0", "accessible", ""],
    ["b.cif", "CuC", "Fm-3m", "tbo (3,4)-c", "13.2", "6.5", "1800", "24", "accessible", ""],
    ["c.cif", "ZrC", "Fm-3m", "fcu 12-c", "8.0", "4.1", "", "0", "", "surface_area: Boom"],
    ["d.cif", "", "", "", "", "", "", "", "", "could not read structure: ValueError"],
    ["e.cif", "ZnN", "I-43m", "sod 4-c", "11.4", "3.4", "950", "0", "blocked", ""],
]


@pytest.fixture
def folder(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(ROWS)
    return out


def _names(rows):
    return [r.get("file") for r in rows]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


def test_columns_are_typed_from_their_contents(folder):
    results = load_batch_results(folder)
    assert results.numeric_columns == {"lcd_A", "pld_A", "channel_asa_m2_per_g",
                                       "open_metal_sites"}
    b = results.rows[1]
    assert b.get("lcd_A") == 13.2 and b.get("open_metal_sites") == 24
    assert isinstance(b.get("open_metal_sites"), int)
    assert results.rows[3].get("lcd_A") is None  # empty, not zero


def test_a_folder_or_its_csv_opens_and_anything_else_is_explained(folder, tmp_path):
    assert load_batch_results(folder / "summary.csv").folder == folder
    with pytest.raises(ValueError, match="No summary.csv"):
        load_batch_results(tmp_path)
    (tmp_path / "other.csv").write_text("x,y\n1,2\n")
    with pytest.raises(ValueError, match="not a the-void-analyze summary"):
        load_batch_results(tmp_path)
    with pytest.raises(ValueError, match="does not exist"):
        load_batch_results(tmp_path / "nowhere")


# ----------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------


def test_numeric_filters_never_pass_an_empty_cell(folder):
    """d.cif failed to load: it has no PLD. 'PLD below 5' must not
    include it, and neither must 'PLD not equal to 4.1'."""
    results = load_batch_results(folder)
    assert _names(view(results, filters=[Filter("pld_A", "<", "5")])) == ["a.cif", "c.cif", "e.cif"]
    assert "d.cif" not in _names(view(results, filters=[Filter("pld_A", "!=", "4.1")]))
    assert _names(view(results, filters=[Filter("pld_A", "is empty")])) == ["d.cif"]


def test_filters_combine_and_text_matches_ignore_case(folder):
    results = load_batch_results(folder)
    rows = view(results, filters=[Filter("pld_A", ">=", "3.3"),
                                  Filter("topology", "contains", "PCU")])
    assert _names(rows) == ["a.cif"]
    assert _names(view(results, filters=[Filter("errors", "is empty")])) == ["a.cif", "b.cif", "e.cif"]
    assert _names(view(results, filters=[Filter("guest_CO2", "=", "Accessible")])) == ["a.cif", "b.cif"]


def test_bad_filters_are_refused_with_a_reason(folder):
    results = load_batch_results(folder)
    with pytest.raises(ValueError, match="needs a number"):
        view(results, filters=[Filter("pld_A", ">=", "wide")])
    with pytest.raises(ValueError, match="does not apply"):
        view(results, filters=[Filter("topology", ">=", "3")])
    with pytest.raises(ValueError, match="No column"):
        view(results, filters=[Filter("nope", "=", "1")])


def test_search_looks_in_every_column(folder):
    results = load_batch_results(folder)
    assert _names(view(results, search="fm-3M")) == ["b.cif", "c.cif"]
    assert _names(view(results, search="boom")) == ["c.cif"]


# ----------------------------------------------------------------------
# Ranking
# ----------------------------------------------------------------------


def test_empty_cells_sort_last_in_both_directions(folder):
    """A failed analysis must not top a 'largest first' ranking."""
    results = load_batch_results(folder)
    assert _names(view(results, sort_column="channel_asa_m2_per_g", descending=True)) == \
        ["b.cif", "a.cif", "e.cif", "c.cif", "d.cif"]
    assert _names(view(results, sort_column="channel_asa_m2_per_g")) == \
        ["e.cif", "a.cif", "b.cif", "c.cif", "d.cif"]


def test_ties_keep_file_order_either_way(folder):
    """a and c share PLD 4.1: they stay a-then-c whichever way it sorts."""
    results = load_batch_results(folder)
    up = _names(view(results, sort_column="pld_A"))
    down = _names(view(results, sort_column="pld_A", descending=True))
    assert up.index("a.cif") < up.index("c.cif")
    assert down.index("a.cif") < down.index("c.cif")
    assert down[-1] == "d.cif"


def test_top_n_is_taken_after_filtering_and_ranking(folder, tmp_path):
    results = load_batch_results(folder)
    rows = view(results, filters=[Filter("errors", "is empty")],
                sort_column="lcd_A", descending=True, top_n=2)
    assert _names(rows) == ["b.cif", "e.cif"]

    path = tmp_path / "view.csv"
    write_view_csv(results, rows, path)
    written = list(csv.DictReader(open(path)))
    assert [(r["rank"], r["file"]) for r in written] == [("1", "b.cif"), ("2", "e.cif")]
    assert written[0]["channel_asa_m2_per_g"] == "1800.0"


# ----------------------------------------------------------------------
# Reports and structure files, from a real batch run
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_batch(tmp_path_factory):
    """the-void-analyze run from inside a project folder on a relative
    path, as people type it - then the results folder moved elsewhere."""
    pytest.importorskip("pymatgen")
    import os

    from thevoid_core import cli

    root = tmp_path_factory.mktemp("project")
    (root / "cifs").mkdir()
    shutil.copy(FIXTURE, root / "cifs" / "sample.cif")
    old = os.getcwd()
    os.chdir(root)
    try:
        cli.main(["cifs/sample.cif", "--out", "results", "-q", "--samples", "100",
                  "--no-surface", "--no-oms", "--no-topology"])
    finally:
        os.chdir(old)
    return root


def test_the_report_is_shown_as_written(real_batch):
    results = load_batch_results(real_batch / "results")
    row = results.rows[0]
    assert row.report_json.name == "sample.json" and row.report_md.name == "sample.md"
    assert report_markdown(row).startswith("# Analysis report")


def test_a_relative_path_is_found_and_verified(real_batch):
    results = load_batch_results(real_batch / "results")
    located = locate_structure(results.rows[0], results)
    assert located.status == "verified"
    assert located.path == (real_batch / "cifs" / "sample.cif").resolve()
    assert located.recorded_path == "cifs/sample.cif"


def test_an_edited_file_with_the_same_name_is_not_accepted(real_batch, tmp_path):
    """Same name, different contents: the row's numbers are not its own."""
    moved = tmp_path / "moved"
    shutil.copytree(real_batch / "results", moved / "results")
    (moved / "sample.cif").write_text(FIXTURE.read_text() + "\n# edited\n")
    results = load_batch_results(moved / "results")
    located = locate_structure(results.rows[0], results)
    assert located.status == "mismatch"
    assert "SHA-256 differs" in located.message


def test_a_moved_folder_is_found_through_a_folder_the_person_points_at(real_batch, tmp_path):
    moved = tmp_path / "elsewhere"
    shutil.copytree(real_batch / "results", moved / "results")
    results = load_batch_results(moved / "results")
    assert locate_structure(results.rows[0], results).status == "missing"
    located = locate_structure(results.rows[0], results, [real_batch / "cifs"])
    assert located.status == "verified"
    assert check_file(real_batch / "cifs" / "sample.cif", results.rows[0]).status == "verified"


def test_a_row_without_a_report_still_says_what_happened(folder):
    results = load_batch_results(folder)
    text = report_markdown(results.rows[3])
    assert "No report was written" in text and "could not read structure" in text
    assert locate_structure(results.rows[3], results).status == "missing"


def test_a_report_without_a_checksum_is_found_but_unverified(folder):
    (folder / "a.json").write_text(json.dumps({"input": {"file": "a.cif", "sha256": None}}))
    (folder / "a.cif").write_text("data_a\n")
    results = load_batch_results(folder)
    located = locate_structure(results.rows[0], results)
    assert located.status == "unverified"
    assert file_sha256(located.path)
