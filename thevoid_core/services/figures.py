"""Publication figures from a screening run (round 58, roadmap 3a).

Three kinds, all drawn from a the-void-analyze results folder
(services/batch_results.py) after whatever filters were applied:

    scatter    one numeric column against another, optionally coloured
               by a text column (topology, a metal, a guest verdict)
    histogram  the distribution of one numeric column
    parity     the Void's value against a reference database's for the
               same quantity (the --compare-coremof columns), with the
               y = x line, the tolerance band the comparison flags at,
               and the agreement statistics

Built to journal defaults, not screen defaults:

- **Size:** one column (85 mm) or two (175 mm) wide, drawn at that size
  in points so that 7 pt text is 7 pt on the page. Nothing is scaled
  afterwards.
- **Formats:** every figure is written as SVG (text kept as text),
  PDF (TrueType fonts embedded, fonttype 42) and 600 dpi PNG, with a
  JSON beside them. The JSON holds the exact data plotted (file, x, y,
  category), the rows left out and why, the filters, the style and the
  software version, so the figure can be checked, replotted or
  regenerated, and a reviewer's "which structure is that point?" has an
  answer.
- **Colour:** at most three categorical colours - the first three
  slots of a colour-blind-validated palette (all pairs pass CVD
  separation; checked with the dataviz validator, round 58). Every
  further category folds into "Other", in grey. Each category also has
  its own marker shape, so identity never rests on colour alone: that
  covers grayscale print, and the one slot (aqua) with less than 3:1
  contrast on white.
- **Axes:** one y-axis, thin recessive axes, no gridlines. Units are in
  the axis labels, and a log axis is used only when asked for.

Nothing here imports Qt; ui/dialogs/figure_dialog.py is the window
around it.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from thevoid_core.version import __version__

MM_PER_INCH = 25.4

#: Journal column widths, mm (the common single/double column of ACS,
#: RSC, Elsevier and Wiley journals, rounded down to fit all of them).
WIDTHS_MM = {"single": 85.0, "double": 175.0}

#: First three categorical slots (blue, orange, aqua), then "Other".
CATEGORY_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a"]
OTHER_COLOUR = "#898781"
#: A single-series plot's colour (the first slot).
SERIES_COLOUR = CATEGORY_COLOURS[0]
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
#: Marker per category, in the same order, so print and CVD readers can
#: tell them apart without colour; "Other" is a small open circle.
CATEGORY_MARKERS = ["o", "s", "^"]
OTHER_MARKER = "o"
MAX_CATEGORIES = len(CATEGORY_COLOURS)

FONT_FAMILIES = ["Arial", "Helvetica", "Liberation Sans", "FreeSans", "DejaVu Sans"]

FORMATS = ("svg", "pdf", "png")
PNG_DPI = 600

#: What the parity plot's tolerance band shows - the thresholds
#: services/reference/compare.py flags differences at.
PARITY_TOLERANCES = {
    "lcd_A": ("absolute", 0.15),
    "pld_A": ("absolute", 0.15),
    "asa_m2_per_g": ("relative", 0.05),
}


@dataclass
class FigureStyle:
    width: str = "single"
    #: Height as a fraction of the width (square-ish for one column).
    aspect: float = 0.8
    font_size_pt: float = 7.0
    marker_size_pt: float = 3.0
    line_width_pt: float = 0.6
    formats: tuple = FORMATS
    png_dpi: int = PNG_DPI

    @property
    def size_inches(self) -> tuple[float, float]:
        width = WIDTHS_MM[self.width] / MM_PER_INCH
        return width, width * self.aspect

    def as_dict(self) -> dict:
        width, height = self.size_inches
        return {"width": self.width, "width_mm": round(width * MM_PER_INCH, 1),
                "height_mm": round(height * MM_PER_INCH, 1), "font_size_pt": self.font_size_pt,
                "marker_size_pt": self.marker_size_pt, "line_width_pt": self.line_width_pt,
                "formats": list(self.formats), "png_dpi": self.png_dpi}


@dataclass
class FigureSpec:
    kind: str                       # "scatter" | "histogram" | "parity"
    x: str
    y: str | None = None
    colour_by: str | None = None
    log_x: bool = False
    log_y: bool = False
    #: None: numpy's "auto" rule (the larger of Sturges and
    #: Freedman-Diaconis), so a small batch does not get a comb of empty bins.
    bins: int | None = None
    title: str = ""

    def validate(self, results) -> None:
        from thevoid_core.services.batch_results import column_label

        if self.kind not in ("scatter", "histogram", "parity"):
            raise ValueError(f"Unknown figure kind {self.kind!r}")
        needed = [self.x] + ([self.y] if self.kind in ("scatter", "parity") else [])
        for column in needed:
            if column not in results.columns:
                raise ValueError(f"No column {column!r} in this batch")
            if not results.is_numeric(column):
                raise ValueError(f"{column_label(column)} is not a numeric column")
        if self.kind in ("scatter", "parity") and not self.y:
            raise ValueError(f"A {self.kind} plot needs a y column")
        if self.colour_by and self.colour_by not in results.columns:
            raise ValueError(f"No column {self.colour_by!r} to colour by")


@dataclass
class FigureData:
    """What is plotted, and what was left out."""

    points: list = field(default_factory=list)    # [(file, x, y, category)]
    excluded: list = field(default_factory=list)  # [(file, reason)]
    categories: list = field(default_factory=list)  # shown, in slot order
    folded: list = field(default_factory=list)    # folded into "Other"
    histogram: dict | None = None                 # {"edges": [...], "counts": [...]}
    colour_values: dict = field(default_factory=dict)  # file -> cell, where it differs


#: Axis labels that differ from the table's column labels. In the table
#: "PLD, CoRE convention" sits beside "CoRE PLD"; on a figure in a paper
#: the axis must say who computed the number. (Both use CoRE's radii and
#: probe; the JSON and the method notes say so.)
FIGURE_LABELS = {
    "ccdc_lcd_A": "LCD, the Void (Å)",
    "coremof_lcd_A": "LCD, CoRE MOF 2024 (Å)",
    "ccdc_pld_A": "PLD, the Void (Å)",
    "coremof_pld_A": "PLD, CoRE MOF 2024 (Å)",
    "ccdc_asa_m2_per_g": "ASA, the Void (m²/g)",
    "coremof_asa_m2_per_g": "ASA, CoRE MOF 2024 (m²/g)",
}


def axis_label(column: str) -> str:
    from thevoid_core.services.batch_results import column_label

    return FIGURE_LABELS.get(column) or column_label(column)


_RCSR_SUMMARY = re.compile(r"^([a-z][a-z0-9.\-]*) \(?[\d,]+\)?-c\b")

#: How a topology summary becomes a colour category (recorded in the JSON).
TOPOLOGY_RULE = ("the RCSR code of an identified net (interpenetration and "
                 "connectivity dropped); 'not in library' for a net not in the "
                 "reference library; 'not identified' otherwise")


#: Categories that say a value is missing, not what it is. They never
#: take one of the three colours: on the real CoRE batch, "not
#: identified" and "not in library" were the two largest groups and
#: pushed every net but sql into Other.
NON_CATEGORIES = frozenset({"(none)", "not identified", "not in library"})

#: A reference point, not a category (round 59: the parent in a
#: the-void-functionalize batch). It never takes a colour and is never
#: folded into Other: it is drawn in ink as a star, on top.
REFERENCE_CATEGORIES = frozenset({"(parent)"})
REFERENCE_MARKER = "*"


def category_of(column: str, raw) -> str:
    """The colour category of a cell. A topology summary is reduced to
    its net: "sql 4-c", "sql 4-c, 2-fold interpenetrated" and "sql" are
    all sql. Coloured by the full text, a 36-structure CoRE batch had 21
    categories, the three largest "not identified: rod/layer SBU" and
    two spellings of sql (round 58)."""
    if raw in (None, ""):
        return "(none)"
    text = str(raw).strip()
    if column == "topology" or column.endswith("_topology"):
        match = _RCSR_SUMMARY.match(text)
        if match:
            return match.group(1)
        if "not in the reference library" in text:
            return "not in library"
        if text.startswith(("not identified", "no ")):
            return "not identified"
        if re.fullmatch(r"[a-z][a-z0-9.\-]*", text):  # a bare code, e.g. coremof_topology
            return text
    return text


def collect(results, rows, spec: FigureSpec) -> FigureData:
    """The points of `spec` from `rows` (a batch_results view), with
    every row that cannot be plotted listed with its reason - a missing
    value, a value a log axis cannot show, or (parity against CoRE MOF)
    a match that is not "identified": a "possible" or "name only"
    entry's numbers are another structure's, not a test of the Void."""
    data = FigureData()
    columns = [spec.x] + ([spec.y] if spec.y and spec.kind != "histogram" else [])
    check_match = (spec.kind == "parity" and "coremof_match" in results.columns
                   and (spec.x.startswith("coremof_") or spec.y.startswith("coremof_")))
    for row in rows:
        name = row.get("file") or f"row {row.index + 1}"
        values = []
        reason = None
        if check_match and row.get("coremof_match") != "identified":
            data.excluded.append((name, f"CoRE MOF match: {row.get('coremof_match') or 'none'}"))
            continue
        for column in columns:
            value = row.get(column)
            if value is None:
                reason = f"no {axis_label(column)}"
                break
            if (column == spec.x and spec.log_x) or (column == spec.y and spec.log_y):
                if value <= 0:
                    reason = f"{axis_label(column)} = {value} cannot go on a log axis"
                    break
            values.append(float(value))
        if reason:
            data.excluded.append((name, reason))
            continue
        category = None
        if spec.colour_by:
            raw = row.get(spec.colour_by)
            category = category_of(spec.colour_by, raw)
            if raw not in (None, "") and str(raw) != category:
                data.colour_values[name] = str(raw)
        x = values[0]
        y = values[1] if len(values) > 1 else None
        data.points.append((name, x, y, category))

    if spec.colour_by:
        counts = {}
        for *_, category in data.points:
            counts[category] = counts.get(category, 0) + 1
        # Most frequent first; ties by name, so the order is stable.
        ranked = sorted(counts, key=lambda c: (c in NON_CATEGORIES, -counts[c], c))
        ranked = [c for c in ranked if c not in REFERENCE_CATEGORIES]
        named = [c for c in ranked if c not in NON_CATEGORIES]
        data.categories = named[:MAX_CATEGORIES]
        data.folded = [c for c in ranked if c not in data.categories]
    return data


def parity_statistics(points, column: str) -> dict:
    """n, mean and median difference (the Void minus reference), mean
    absolute difference, and how many fall outside the comparison's
    flag tolerance."""
    # points are (file, x = reference, y = the Void, category)
    differences = [p[2] - p[1] for p in points]
    if not differences:
        return {"n": 0}
    ordered = sorted(differences)
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    kind, tolerance = PARITY_TOLERANCES.get(_quantity(column), ("absolute", None))
    outside = None
    if tolerance is not None:
        outside = sum(
            1 for p in points
            if abs(p[2] - p[1]) > (tolerance * abs(p[1]) if kind == "relative" else tolerance)
        )
    return {
        "n": len(differences),
        "mean_difference": sum(differences) / len(differences),
        "median_difference": median,
        "mean_absolute_difference": sum(abs(d) for d in differences) / len(differences),
        "tolerance": None if tolerance is None else {"kind": kind, "value": tolerance},
        "outside_tolerance": outside,
    }


def _quantity(column: str) -> str:
    for prefix in ("coremof_", "ccdc_", "ref_"):
        if column.startswith(prefix):
            return column[len(prefix):]
    return column


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------


def _font_family():
    from matplotlib import font_manager

    available = {font.name for font in font_manager.fontManager.ttflist}
    for family in FONT_FAMILIES:
        if family in available:
            return family
    return "DejaVu Sans"


def _rc(style: FigureStyle) -> dict:
    return {
        "font.family": "sans-serif",
        "font.sans-serif": [_font_family()],
        "font.size": style.font_size_pt,
        "axes.labelsize": style.font_size_pt,
        "axes.titlesize": style.font_size_pt,
        "xtick.labelsize": style.font_size_pt,
        "ytick.labelsize": style.font_size_pt,
        "legend.fontsize": style.font_size_pt,
        "axes.linewidth": style.line_width_pt,
        "xtick.major.width": style.line_width_pt,
        "ytick.major.width": style.line_width_pt,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.edgecolor": SECONDARY_INK,
        "axes.labelcolor": INK,
        "xtick.color": SECONDARY_INK,
        "ytick.color": SECONDARY_INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 100,
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }


def draw(results, rows, spec: FigureSpec, style: FigureStyle | None = None, figure=None):
    """(matplotlib Figure, FigureData). A Figure can be passed in (the
    dialog's canvas); otherwise one is made, not attached to any window."""
    import matplotlib

    style = style or FigureStyle()
    spec.validate(results)
    data = collect(results, rows, spec)
    with matplotlib.rc_context(_rc(style)):
        if figure is None:
            from matplotlib.figure import Figure

            figure = Figure(figsize=style.size_inches)
        else:
            figure.clear()
            figure.set_size_inches(*style.size_inches, forward=False)
        axes = figure.add_subplot(111)
        {"scatter": _draw_scatter, "histogram": _draw_histogram,
         "parity": _draw_parity}[spec.kind](axes, data, spec, style)
        if spec.title:
            axes.set_title(spec.title, loc="left")
        figure.tight_layout(pad=0.4)
    figure._void_rc = _rc(style)
    return figure, data


def _scale(axes, spec):
    if spec.log_x:
        axes.set_xscale("log")
    if spec.log_y:
        axes.set_yscale("log")


def _scatter_groups(data):
    """[(label, colour, marker, filled, xs, ys)] in drawing order: "Other"
    first, underneath, then the named categories."""
    if not data.categories:
        xs = [p[1] for p in data.points]
        ys = [p[2] for p in data.points]
        return [(None, SERIES_COLOUR, "o", True, xs, ys)]
    groups = []
    other = [p for p in data.points
             if p[3] not in data.categories and p[3] not in REFERENCE_CATEGORIES]
    if other:
        label = "Other" if len(data.folded) != 1 else data.folded[0]
        groups.append((f"{label} ({len(other)})", OTHER_COLOUR, OTHER_MARKER, False,
                       [p[1] for p in other], [p[2] for p in other]))
    for slot, category in enumerate(data.categories):
        members = [p for p in data.points if p[3] == category]
        groups.append((f"{category} ({len(members)})", CATEGORY_COLOURS[slot],
                       CATEGORY_MARKERS[slot], True,
                       [p[1] for p in members], [p[2] for p in members]))
    for reference in sorted(REFERENCE_CATEGORIES):
        members = [p for p in data.points if p[3] == reference]
        if members:
            groups.append((reference.strip("()"), INK, REFERENCE_MARKER, True,
                           [p[1] for p in members], [p[2] for p in members]))
    return groups


def _plot_groups(axes, groups, style):
    """Filled markers get a thin white ring, so overlapping points stay
    distinct; "Other" is drawn open and underneath."""
    size = style.marker_size_pt ** 2
    named, other = [], []
    for label, colour, marker, filled, xs, ys in groups:
        reference = marker == REFERENCE_MARKER
        handle = axes.scatter(
            xs, ys, s=size * (3.0 if reference else 1.0), marker=marker, label=label,
            facecolors=colour if filled else "none",
            edgecolors="white" if filled else colour,
            linewidths=style.line_width_pt * (0.4 if filled else 0.8),
            zorder=3 if reference else 2 if filled else 1,
        )
        if label:
            (named if filled else other).append((handle, label))
    entries = named + other  # slot order, "Other" last
    if entries:
        axes.legend([h for h, _ in entries], [lab for _, lab in entries],
                    frameon=False, handletextpad=0.2, borderaxespad=0.2, loc="best")


def _draw_scatter(axes, data, spec, style):
    _plot_groups(axes, _scatter_groups(data), style)
    axes.set_xlabel(axis_label(spec.x))
    axes.set_ylabel(axis_label(spec.y))
    _scale(axes, spec)


def _draw_histogram(axes, data, spec, style):
    import numpy as np

    from matplotlib.ticker import MaxNLocator

    values = np.array([p[1] for p in data.points], dtype=float)
    if not len(values):
        edges = np.array([0.0, 1.0])
    elif spec.log_x:
        logs = np.log10(values)
        count = spec.bins or len(np.histogram_bin_edges(logs, bins="auto")) - 1
        edges = np.logspace(logs.min(), logs.max() if np.ptp(logs) else logs.min() + 1, count + 1)
    else:
        edges = np.histogram_bin_edges(values, bins=spec.bins or "auto")
    counts, edges, _ = axes.hist(values, bins=edges, color=SERIES_COLOUR, edgecolor="white",
                                 linewidth=style.line_width_pt)
    data.histogram = {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes.set_xlabel(axis_label(spec.x))
    axes.set_ylabel("Structures")
    _scale(axes, spec)
    if spec.log_y:
        axes.set_yscale("log")


def _draw_parity(axes, data, spec, style):
    import numpy as np

    groups = _scatter_groups(data)
    xs = [p[1] for p in data.points]
    ys = [p[2] for p in data.points]
    if xs:
        low = min(min(xs), min(ys))
        high = max(max(xs), max(ys))
        pad = 0.05 * (high - low or 1.0)
        low, high = low - pad, high + pad
        line = np.array([low, high])
        kind, tolerance = PARITY_TOLERANCES.get(_quantity(spec.y), ("absolute", None))
        if tolerance is not None:
            band = tolerance * line if kind == "relative" else np.array([tolerance, tolerance])
            axes.fill_between(line, line - band, line + band, color="#e1e0d9", linewidth=0,
                              zorder=0)
        axes.plot(line, line, color=MUTED, linewidth=style.line_width_pt, zorder=0.5)
        axes.set_xlim(low, high)
        axes.set_ylim(low, high)
        axes.set_aspect("equal", adjustable="box")
    _plot_groups(axes, groups, style)
    axes.set_xlabel(axis_label(spec.x))
    axes.set_ylabel(axis_label(spec.y))
    stats = parity_statistics(data.points, spec.y)
    if stats.get("n"):
        unit = "m²/g" if "m2" in spec.y else "Å"
        digits = 0 if unit == "m²/g" else 3
        text = (f"n = {stats['n']}\n"
                f"median Δ = {stats['median_difference']:+.{digits}f} {unit}\n"
                f"mean |Δ| = {stats['mean_absolute_difference']:.{digits}f} {unit}")
        tolerance = stats.get("tolerance")
        if tolerance:
            band = (f"±{tolerance['value']:g} {unit}" if tolerance["kind"] == "absolute"
                    else f"±{100 * tolerance['value']:g}%")
            text += f"\nband {band}: {stats['outside_tolerance']} outside"
        axes.text(0.03, 0.97, text, transform=axes.transAxes, va="top", ha="left",
                  color=SECONDARY_INK, linespacing=1.3)


# ----------------------------------------------------------------------
# Saving
# ----------------------------------------------------------------------


def save(figure, data: FigureData, spec: FigureSpec, style: FigureStyle, base_path, *,
         results=None, filters=(), search: str = "") -> list[Path]:
    """Write the figure in every format of `style` plus its JSON record,
    as base_path.<ext>. Returns the paths written."""
    import matplotlib

    base = Path(base_path)
    if base.suffix.lower() in (".svg", ".pdf", ".png", ".json"):
        base = base.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    written = []
    with matplotlib.rc_context(getattr(figure, "_void_rc", _rc(style))):
        for fmt in style.formats:
            path = base.with_suffix(f".{fmt}")
            options = {"dpi": style.png_dpi} if fmt == "png" else {}
            metadata = ({"Creator": f"the Void {__version__}"} if fmt == "pdf"
                        else {"Creator": f"the Void {__version__}"} if fmt == "svg" else None)
            figure.savefig(path, format=fmt, metadata=metadata, **options)
            written.append(path)
    record = figure_record(data, spec, style, results=results, filters=filters, search=search,
                           files=[p.name for p in written])
    json_path = base.with_suffix(".json")
    json_path.write_text(json.dumps(record, indent=1, allow_nan=False), encoding="utf-8")
    written.append(json_path)
    return written


def figure_record(data, spec, style, *, results=None, filters=(), search="", files=()) -> dict:
    source = None
    if results is not None:
        summary = Path(results.summary_path)
        source = {
            "summary": str(summary),
            "sha256": hashlib.sha256(summary.read_bytes()).hexdigest() if summary.is_file() else None,
            "rows_in_batch": len(results.rows),
        }
    record = {
        "figure": {"kind": spec.kind, "x": spec.x, "y": spec.y, "colour_by": spec.colour_by,
                   "log_x": spec.log_x, "log_y": spec.log_y,
                   "bins": (spec.bins or "auto") if spec.kind == "histogram" else None,
                   "x_label": axis_label(spec.x), "y_label": axis_label(spec.y) if spec.y else None},
        "style": {**style.as_dict(), "font": _font_family(),
                  "palette": {"categories": CATEGORY_COLOURS, "markers": CATEGORY_MARKERS,
                              "other": OTHER_COLOUR}},
        "source": source,
        "filters": [f.describe() if hasattr(f, "describe") else str(f) for f in filters],
        "search": search,
        "points": [{"file": name, "x": x, "y": y, "category": category,
                    **({"colour_value": data.colour_values[name]}
                       if name in data.colour_values else {})}
                   for name, x, y, category in data.points],
        "excluded": [{"file": name, "reason": reason} for name, reason in data.excluded],
        "categories_shown": data.categories,
        "categories_folded_into_other": data.folded,
        "files": list(files),
        "software": {"the_void": __version__},
        "generated": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    if spec.colour_by and (spec.colour_by == "topology" or spec.colour_by.endswith("_topology")):
        record["category_rule"] = TOPOLOGY_RULE
    if data.histogram is not None:
        record["histogram"] = data.histogram
    if spec.kind == "parity":
        record["statistics"] = parity_statistics(data.points, spec.y)
    return record


# ----------------------------------------------------------------------
# The default set for a batch
# ----------------------------------------------------------------------


def default_specs(results) -> list[tuple[str, FigureSpec]]:
    """(file stem, spec) for the figures `the-void-analyze --figures`
    writes: whatever of these the batch has the columns for."""
    have = set(results.numeric_columns)
    colour = "topology" if "topology" in results.columns else None
    specs = []
    if {"pld_A", "channel_asa_m2_per_g"} <= have:
        specs.append(("pld_vs_channel_asa", FigureSpec("scatter", "pld_A", "channel_asa_m2_per_g",
                                                       colour_by=colour)))
    if {"pld_A", "lcd_A"} <= have:
        specs.append(("lcd_vs_pld", FigureSpec("scatter", "pld_A", "lcd_A", colour_by=colour)))
    for column in ("pld_A", "channel_asa_m2_per_g", "void_fraction_percent"):
        if column in have:
            specs.append((f"histogram_{column}", FigureSpec("histogram", column)))
    for quantity in ("pld_A", "lcd_A", "asa_m2_per_g"):
        if {f"coremof_{quantity}", f"ccdc_{quantity}"} <= have:
            specs.append((f"parity_{quantity}", FigureSpec("parity", f"coremof_{quantity}",
                                                           f"ccdc_{quantity}")))
    return specs


def write_default_figures(results, folder, style: FigureStyle | None = None) -> list[Path]:
    """Every default figure of a batch into `folder`; returns the paths."""
    style = style or FigureStyle()
    written = []
    for stem, spec in default_specs(results):
        rows = list(results.rows)
        figure, data = draw(results, rows, spec, style)
        if not data.points:
            continue
        written += save(figure, data, spec, style, Path(folder) / stem, results=results)
    return written
