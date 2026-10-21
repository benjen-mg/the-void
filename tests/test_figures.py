"""Round 58: publication figures from a batch (services/figures.py,
the-void-figures, the-void-analyze --figures, Batch Results > Figure...).

A figure that goes into a paper has to be what it says it is. So the
checks here are about the things a reviewer or an editor would catch:

* the size is the journal's column width (85 / 175 mm) in every format,
  and the PNG is 600 dpi;
* the PDF embeds its fonts (TrueType, not Type 3) and the SVG keeps its
  text as text, so both can be edited;
* nothing is silently dropped: every row that is not plotted is listed
  in the JSON with its reason (a missing value, a value a log axis
  cannot show, a CoRE match that is not "identified");
* at most three categories get their own colour, each with its own
  marker as well (the third colour is too light on white to rely on
  alone), the rest are folded into an open grey "Other", and the
  colours follow category frequency, not the order rows arrive in;
* the parity plot's numbers are right on hand-checkable values.
"""

import csv
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from thevoid_core import figures_cli  # noqa: E402
from thevoid_core.services import figures  # noqa: E402
from thevoid_core.services.batch_results import Filter, load_batch_results, view  # noqa: E402

from tests.sample_structures import ZN_DIA  # noqa: E402

FIXTURE = ZN_DIA

COLUMNS = ["file", "topology", "pld_A", "lcd_A", "channel_asa_m2_per_g", "void_fraction_percent",
           "coremof_match", "coremof_pld_A", "ccdc_pld_A", "errors"]
# Topology counts: pcu 4 (one failed), dia 3, sql 2, hcb 1, kgd 1, one with none ->
# dia, pcu, sql shown; hcb, kgd and "(none)" folded into Other.
ROWS = [
    ["a.cif", "pcu", "4.0", "6.0", "1200", "55", "identified", "4.00", "3.95", ""],
    ["b.cif", "dia", "5.0", "7.5", "1500", "60", "identified", "5.00", "5.20", ""],
    ["c.cif", "pcu", "6.0", "8.0", "0", "40", "name only", "9.00", "6.00", ""],
    ["d.cif", "sql", "3.0", "5.0", "800", "35", "identified", "3.00", "2.90", ""],
    ["e.cif", "pcu", "", "", "", "", "none", "", "", "pore_analysis: Boom"],
    ["f.cif", "dia", "7.0", "9.0", "2100", "70", "identified", "7.00", "7.05", ""],
    ["g.cif", "hcb", "3.5", "5.5", "600", "30", "", "", "", ""],
    ["h.cif", "pcu", "4.5", "6.5", "1300", "50", "", "", "", ""],
    ["i.cif", "sql", "3.2", "4.8", "900", "33", "", "", "", ""],
    ["j.cif", "dia", "8.0", "11.0", "2500", "75", "", "", "", ""],
    ["k.cif", "kgd", "5.5", "7.0", "1700", "62", "", "", "", ""],
    ["l.cif", "", "4.2", "6.1", "1100", "48", "", "", "", ""],
]


@pytest.fixture
def results(tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(ROWS)
    return load_batch_results(out)


def _save(results, spec, tmp_path, style=None, rows=None, **kwargs):
    style = style or figures.FigureStyle()
    rows = list(results.rows) if rows is None else rows
    figure, data = figures.draw(results, rows, spec, style)
    written = figures.save(figure, data, spec, style, tmp_path / "fig", results=results, **kwargs)
    return figure, data, {p.suffix: p for p in written}


# ----------------------------------------------------------------------
# Size, fonts, formats
# ----------------------------------------------------------------------


@pytest.mark.parametrize("width, mm", [("single", 85.0), ("double", 175.0)])
def test_every_format_is_the_column_width(results, tmp_path, width, mm):
    from PIL import Image

    style = figures.FigureStyle(width=width)
    _, _, files = _save(results, figures.FigureSpec("scatter", "pld_A", "lcd_A"), tmp_path, style)
    assert set(files) == {".svg", ".pdf", ".png", ".json"}

    points = mm / 25.4 * 72
    svg = ET.parse(files[".svg"]).getroot()
    assert float(svg.get("width").removesuffix("pt")) == pytest.approx(points, abs=0.01)
    assert float(svg.get("height").removesuffix("pt")) == pytest.approx(points * 0.8, abs=0.01)

    pdf = files[".pdf"].read_bytes()
    box = pdf[pdf.index(b"/MediaBox"):].split(b"]")[0].split(b"[")[1].split()
    assert float(box[2]) == pytest.approx(points, abs=0.01)

    with Image.open(files[".png"]) as image:
        assert image.size[0] == pytest.approx(mm / 25.4 * 600, abs=1)
        assert image.info["dpi"][0] == pytest.approx(600, abs=0.5)

    record = json.loads(files[".json"].read_text())
    assert record["style"]["width_mm"] == mm and record["style"]["png_dpi"] == 600


@pytest.mark.skipif(shutil.which("pdffonts") is None, reason="needs poppler's pdffonts")
def test_the_pdf_embeds_truetype_fonts(results, tmp_path):
    """Type 3 fonts (matplotlib's default) are rejected by some journal
    systems and cannot be edited; pdf.fonttype 42 embeds TrueType."""
    _, _, files = _save(results, figures.FigureSpec("scatter", "pld_A", "lcd_A",
                                                    colour_by="topology"), tmp_path)
    listing = subprocess.run(["pdffonts", str(files[".pdf"])], capture_output=True, text=True,
                             check=True).stdout.splitlines()[2:]
    assert listing
    for line in listing:
        assert "Type 3" not in line
        assert "TrueType" in line or "CID" in line
        assert " yes " in line  # the "emb" column


def test_the_svg_keeps_its_text_as_text(results, tmp_path):
    """svg.fonttype none: labels stay editable text, not glyph paths."""
    _, _, files = _save(results, figures.FigureSpec("scatter", "pld_A", "lcd_A"), tmp_path)
    svg = files[".svg"].read_text()
    assert "<text" in svg
    assert figures.axis_label("pld_A") in svg and figures.axis_label("lcd_A") in svg


def test_the_font_size_is_seven_points(results):
    figure, _ = figures.draw(results, list(results.rows),
                             figures.FigureSpec("scatter", "pld_A", "lcd_A"))
    axes = figure.axes[0]
    assert axes.xaxis.label.get_size() == 7.0
    assert all(t.get_size() == 7.0 for t in axes.get_xticklabels())


# ----------------------------------------------------------------------
# What is plotted, and what is not
# ----------------------------------------------------------------------


def test_every_row_left_out_is_listed_with_its_reason(results, tmp_path):
    spec = figures.FigureSpec("scatter", "pld_A", "channel_asa_m2_per_g", log_y=True)
    _, data, files = _save(results, spec, tmp_path)
    excluded = dict(data.excluded)
    assert excluded["e.cif"].startswith("no ")          # the failed analysis
    assert "log axis" in excluded["c.cif"]              # channel ASA 0
    assert len(data.points) + len(data.excluded) == len(ROWS)

    record = json.loads(files[".json"].read_text())
    assert {e["file"] for e in record["excluded"]} == {"c.cif", "e.cif"}
    assert {p["file"] for p in record["points"]} == {r[0] for r in ROWS} - {"c.cif", "e.cif"}
    point = next(p for p in record["points"] if p["file"] == "b.cif")
    assert (point["x"], point["y"]) == (5.0, 1500.0)


def test_the_json_records_the_source_filters_and_style(results, tmp_path):
    import hashlib

    filters = [Filter("pld_A", ">=", "4")]
    rows = view(results, filters=filters, search="cif")
    _, data, files = _save(results, figures.FigureSpec("histogram", "pld_A"), tmp_path,
                           rows=rows, filters=filters, search="cif")
    record = json.loads(files[".json"].read_text())
    summary = results.folder / "summary.csv"
    assert record["source"]["sha256"] == hashlib.sha256(summary.read_bytes()).hexdigest()
    assert record["source"]["rows_in_batch"] == len(ROWS)
    assert record["filters"] == [filters[0].describe()] and record["search"] == "cif"
    assert all(p["x"] >= 4 for p in record["points"]) and len(record["points"]) == 8
    assert record["style"]["font"] and record["style"]["palette"]["categories"] == \
        figures.CATEGORY_COLOURS
    assert record["files"] == ["fig.svg", "fig.pdf", "fig.png"]


def test_three_categories_at_most_by_frequency_the_rest_are_other(results):
    spec = figures.FigureSpec("scatter", "pld_A", "lcd_A", colour_by="topology")
    figure, data = figures.draw(results, list(results.rows), spec)
    # Plotted: dia 3, pcu 3 (e.cif failed), sql 2 - a tie goes by name.
    assert data.categories == ["dia", "pcu", "sql"]
    assert sorted(data.folded) == ["(none)", "hcb", "kgd"]

    axes = figure.axes[0]
    legend = [t.get_text() for t in axes.get_legend().get_texts()]
    assert legend == ["dia (3)", "pcu (3)", "sql (2)", "Other (3)"]

    # Each named category has its own colour AND marker; Other is open grey.
    import matplotlib.colors as mcolors

    collections = axes.collections
    other = collections[0]  # drawn first, underneath
    assert len(other.get_offsets()) == 3
    assert mcolors.to_hex(other.get_edgecolor()[0]) == figures.OTHER_COLOUR
    assert other.get_facecolor().size == 0 or other.get_facecolor()[0][3] == 0
    shapes = set()
    for slot, collection in enumerate(collections[1:]):
        assert mcolors.to_hex(collection.get_facecolor()[0]) == figures.CATEGORY_COLOURS[slot]
        shapes.add(tuple(map(tuple, collection.get_paths()[0].vertices.round(3))))
    assert len(shapes) == 3


@pytest.mark.parametrize("summary, category", [
    ("sql 4-c", "sql"),
    ("sql 4-c, 2-fold interpenetrated", "sql"),
    ("kgd (3,6)-c", "kgd"),
    ("tbo (3,4)-c", "tbo"),
    ("sql", "sql"),                                     # coremof_topology: a bare code
    ("(3,4,6)-c net, not in the reference library", "not in library"),
    ("not identified: rod/layer SBU", "not identified"),
    ("no 2D/3D net (molecular or chain)", "not identified"),
    ("", "(none)"),
])
def test_topology_colours_by_the_net(summary, category):
    """Round 58's real CoRE batch: coloured by the full summary text, 36
    structures made 21 categories, and sql was split between "sql 4-c"
    and its interpenetrated form."""
    assert figures.category_of("topology", summary) == category
    assert figures.category_of("formula", "sql 4-c") == "sql 4-c"  # other columns as they are


def test_the_json_keeps_the_topology_text_behind_each_category(tmp_path):
    out = tmp_path / "r"
    out.mkdir()
    (out / "summary.csv").write_text(
        "file,topology,pld_A,lcd_A\n"
        "a.cif,\"sql 4-c, 2-fold interpenetrated\",4,6\nb.cif,sql 4-c,5,7\n")
    results = load_batch_results(out)
    _, data, files = _save(results, figures.FigureSpec("scatter", "pld_A", "lcd_A",
                                                       colour_by="topology"), tmp_path)
    assert data.categories == ["sql"]
    record = json.loads(files[".json"].read_text())
    assert record["points"][0]["colour_value"] == "sql 4-c, 2-fold interpenetrated"
    assert "colour_value" in record["points"][1]
    assert "RCSR code" in record["category_rule"]


def test_missing_or_unidentified_never_takes_a_colour(tmp_path):
    """Round 58's real batch: "not identified" (8) and "not in library"
    (8) outnumbered every net but sql (8); as categories they took two of
    the three colours and said nothing about the structures."""
    out = tmp_path / "r"
    out.mkdir()
    lines = ["file,topology,pld_A,lcd_A"]
    for i in range(5):
        lines.append(f"u{i}.cif,not identified: rod/layer SBU,4,6")
        lines.append(f"n{i}.cif,\"(3,4)-c net, not in the reference library\",4,6")
    lines += ["s1.cif,sql 4-c,4,6", "s2.cif,sql 4-c,4,6", "p1.cif,pcu 6-c,4,6"]
    (out / "summary.csv").write_text("\n".join(lines) + "\n")
    results = load_batch_results(out)
    _, data = figures.draw(results, list(results.rows), figures.FigureSpec(
        "scatter", "pld_A", "lcd_A", colour_by="topology"))
    assert data.categories == ["sql", "pcu"]
    assert set(data.folded) == {"not identified", "not in library"}


def test_colour_follows_the_category_not_the_row_order(results):
    """Reversing the rows must not repaint anything."""
    spec = figures.FigureSpec("scatter", "pld_A", "lcd_A", colour_by="topology")
    _, forward = figures.draw(results, list(results.rows), spec)
    _, backward = figures.draw(results, list(reversed(results.rows)), spec)
    assert forward.categories == backward.categories


def test_a_single_folded_category_keeps_its_name(results):
    rows = [r for r in results.rows if r.get("topology") in ("pcu", "dia", "sql", "hcb")]
    figure, data = figures.draw(results, rows, figures.FigureSpec(
        "scatter", "pld_A", "lcd_A", colour_by="topology"))
    assert data.folded == ["hcb"]
    assert "hcb (1)" in [t.get_text() for t in figure.axes[0].get_legend().get_texts()]


def test_one_series_has_no_legend(results):
    figure, _ = figures.draw(results, list(results.rows),
                             figures.FigureSpec("scatter", "pld_A", "lcd_A"))
    assert figure.axes[0].get_legend() is None


def test_histogram_counts_every_value(results):
    figure, data = figures.draw(results, list(results.rows),
                                figures.FigureSpec("histogram", "pld_A", bins=5))
    heights = [patch.get_height() for patch in figure.axes[0].patches]
    assert len(heights) == 5 and sum(heights) == len(data.points) == 11
    assert data.histogram["counts"] == [int(h) for h in heights]
    assert data.histogram["edges"][0] == 3.0 and data.histogram["edges"][-1] == 8.0


def test_histogram_bins_default_to_the_auto_rule_and_counts_are_whole(results, tmp_path):
    """Twenty fixed bins over eleven values is a comb of empty bars with
    a count axis reading 0.25, 0.50, ... (seen in the first render)."""
    import numpy as np

    _, data, files = _save(results, figures.FigureSpec("histogram", "pld_A"), tmp_path)
    values = [p[1] for p in data.points]
    assert len(data.histogram["counts"]) == len(np.histogram_bin_edges(values, "auto")) - 1
    record = json.loads(files[".json"].read_text())
    assert record["figure"]["bins"] == "auto" and record["histogram"] == data.histogram
    figure, _ = figures.draw(results, list(results.rows), figures.FigureSpec("histogram", "pld_A"))
    ticks = figure.axes[0].get_yticks()
    assert all(float(t).is_integer() for t in ticks)


def test_bad_columns_are_refused_with_a_reason(results):
    with pytest.raises(ValueError, match="No column"):
        figures.draw(results, [], figures.FigureSpec("scatter", "pld_A", "nope"))
    with pytest.raises(ValueError, match="not a numeric column"):
        figures.draw(results, [], figures.FigureSpec("scatter", "pld_A", "topology"))


# ----------------------------------------------------------------------
# Parity
# ----------------------------------------------------------------------


def test_parity_uses_identified_matches_only_and_its_statistics_are_right(results, tmp_path):
    spec = figures.FigureSpec("parity", "coremof_pld_A", "ccdc_pld_A")
    figure, data, files = _save(results, spec, tmp_path)
    # c.cif is a "name only" match: another structure's value, left out.
    assert dict(data.excluded)["c.cif"] == "CoRE MOF match: name only"
    assert [p[0] for p in data.points] == ["a.cif", "b.cif", "d.cif", "f.cif"]

    # Void - CoRE: -0.05, +0.20, -0.10, +0.05
    stats = json.loads(files[".json"].read_text())["statistics"]
    assert stats["n"] == 4
    assert stats["mean_difference"] == pytest.approx(0.025)
    assert stats["median_difference"] == pytest.approx(0.0)
    assert stats["mean_absolute_difference"] == pytest.approx(0.1)
    assert stats["tolerance"] == {"kind": "absolute", "value": 0.15}
    assert stats["outside_tolerance"] == 1  # b.cif

    axes = figure.axes[0]
    assert axes.get_xlim() == axes.get_ylim()
    text = "\n".join(t.get_text() for t in axes.texts)
    assert "n = 4" in text and "median Δ = +0.000 Å" in text
    assert "band ±0.15 Å: 1 outside" in text
    assert axes.get_ylabel() == "PLD, the Void (Å)"


def test_a_relative_tolerance_scales_with_the_value():
    points = [("a", 1000.0, 1040.0, None), ("b", 1000.0, 1060.0, None), ("c", 100.0, 106.0, None)]
    stats = figures.parity_statistics(points, "ccdc_asa_m2_per_g")
    assert stats["tolerance"] == {"kind": "relative", "value": 0.05}
    assert stats["outside_tolerance"] == 2


# ----------------------------------------------------------------------
# The default set and the command line
# ----------------------------------------------------------------------


def test_the_default_set_is_whatever_the_batch_has_columns_for(results):
    stems = [stem for stem, _ in figures.default_specs(results)]
    assert stems == ["pld_vs_channel_asa", "lcd_vs_pld", "histogram_pld_A",
                     "histogram_channel_asa_m2_per_g", "histogram_void_fraction_percent",
                     "parity_pld_A"]
    assert all(spec.colour_by == "topology" for stem, spec in figures.default_specs(results)
               if spec.kind == "scatter")


def test_figures_command_writes_the_default_set(results, capsys):
    assert figures_cli.main([str(results.folder)]) == 0
    out = results.folder / "figures"
    for stem, _ in figures.default_specs(results):
        for suffix in (".svg", ".pdf", ".png", ".json"):
            assert (out / f"{stem}{suffix}").is_file()


def test_figures_command_one_figure_with_a_filter(results, tmp_path):
    out = tmp_path / "custom"
    assert figures_cli.main([str(results.folder), "--scatter", "pld_A", "lcd_A", "--colour",
                             "topology", "--filter", "errors is empty", "--filter",
                             "pld_A >= 4", "--width", "double", "--formats", "svg",
                             "--out", str(out), "--name", "screen"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["screen.json", "screen.svg"]
    record = json.loads((out / "screen.json").read_text())
    assert record["style"]["width_mm"] == 175.0
    assert len(record["filters"]) == 2
    assert all(p["x"] >= 4 for p in record["points"])


def test_figures_command_errors(results, tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_:
        figures_cli.main([str(results.folder), "--scatter", "pld_A", "topology"])
    assert exit_.value.code == 2
    with pytest.raises(SystemExit):
        figures_cli.main([str(results.folder), "--hist", "pld_A", "--formats", "gif"])
    # Nothing survives the filter: exit 1, nothing written.
    out = tmp_path / "none"
    assert figures_cli.main([str(results.folder), "--hist", "pld_A", "--filter",
                             "pld_A > 100", "--out", str(out)]) == 1
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize("text, expected", [
    ("pld_A >= 3.3", ("pld_A", ">=", "3.3")),
    ("errors is empty", ("errors", "is empty", "")),
    ("errors is not empty", ("errors", "is not empty", "")),
    ("topology contains pcu", ("topology", "contains", "pcu")),
    ("topology does not contain sql", ("topology", "does not contain", "sql")),
    ("'formula' = Zn C", ("formula", "=", "Zn C")),
])
def test_filters_are_read_from_the_command_line(text, expected):
    f = figures_cli.parse_filter(text)
    assert (f.column, f.operator, f.value or "") == expected


def test_analyze_figures_writes_them_at_the_end_of_a_batch(tmp_path):
    """The real pipeline on a Zn MOF: the summary it writes is one the
    figure code can read, and --figures puts them in <out>/figures."""
    from thevoid_core import cli

    shutil.copy(FIXTURE, tmp_path / "zn_dia.cif")
    out = tmp_path / "results"
    code = cli.main([str(tmp_path / "zn_dia.cif"), "--out", str(out), "-q", "--no-oms",
                     "--no-topology", "--samples", "200", "--figures"])
    assert code == 0
    written = sorted(p.name for p in (out / "figures").iterdir())
    assert "pld_vs_channel_asa.pdf" in written and "histogram_pld_A.json" in written
    record = json.loads((out / "figures" / "pld_vs_channel_asa.json").read_text())
    assert record["points"][0]["file"] == "zn_dia.cif"
    assert record["points"][0]["x"] > 2.5  # its PLD, N2 radii (about 4.1 A)


# ----------------------------------------------------------------------
# The Batch Results window's Figure... dialog
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_dialog_plots_the_rows_as_shown_and_saves_them(qt_app, results, tmp_path):
    from thevoid_core.ui.windows.batch_results_window import BatchResultsWindow

    window = BatchResultsWindow()
    assert window.load(str(results.folder))
    window.search_edit.setText("cif")
    window.add_filter("pld_A", ">=", "4")
    window.refresh()
    dialog = window.open_figure()
    assert dialog is not None
    # Defaults: PLD against channel ASA, coloured by topology.
    assert dialog.spec().x == "pld_A" and dialog.spec().y == "channel_asa_m2_per_g"
    assert dialog.spec().colour_by == "topology"
    assert len(dialog.data.points) + len(dialog.data.excluded) == len(window.visible_rows)
    assert "Filters:" in dialog.status_label.text()

    dialog.kind_box.setCurrentIndex(dialog.kind_box.findData("parity"))
    assert (dialog.spec().x, dialog.spec().y) == ("coremof_pld_A", "ccdc_pld_A")
    dialog.kind_box.setCurrentIndex(dialog.kind_box.findData("histogram"))
    assert not dialog.y_box.isEnabled()

    written = dialog.save(str(tmp_path / "from_window"))
    assert {p.suffix for p in written} == {".svg", ".pdf", ".png", ".json"}
    record = json.loads((tmp_path / "from_window.json").read_text())
    assert record["search"] == "cif" and record["filters"] == ["pld_A >= 4"]
    assert all(p["x"] >= 4 for p in record["points"])
    dialog.close()
    window.close()


def test_the_dialog_explains_a_figure_it_cannot_draw(qt_app, results):
    from thevoid_core.ui.dialogs.figure_dialog import FigureDialog

    dialog = FigureDialog(results, [r for r in results.rows if r.get("file") == "e.cif"])
    assert not dialog.save_button.isEnabled()
    assert "0 of 1 rows plotted" in dialog.status_label.text()
    assert dialog.save("unused") == []
    dialog.close()


def test_the_parent_of_a_functionalised_batch_is_a_reference_not_a_category(tmp_path):
    """Round 59: coloured by fn_group, a the-void-functionalize batch has
    the parent as "(parent)". It must stand out, not be folded into
    Other with the smallest group: it is drawn as a black star and takes
    no colour."""
    out = tmp_path / "r"
    out.mkdir()
    lines = ["file,fn_group,pld_A,channel_asa_m2_per_g", "p.cif,(parent),8,3600"]
    for group, n in (("Br", 4), ("NH2", 4), ("NO2", 4), ("CH3", 3)):
        lines += [f"{group}{i}.cif,{group},{5 + i},{2000 + 100 * i}" for i in range(n)]
    (out / "summary.csv").write_text("\n".join(lines) + "\n")
    results = load_batch_results(out)
    figure, data = figures.draw(results, list(results.rows), figures.FigureSpec(
        "scatter", "pld_A", "channel_asa_m2_per_g", colour_by="fn_group"))
    assert data.categories == ["Br", "NH2", "NO2"] and data.folded == ["CH3"]
    legend = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]
    assert "parent" in legend and "CH3 (3)" in legend
