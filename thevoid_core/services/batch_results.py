"""Batch results: load, filter and rank a `the-void-analyze` results
folder, and find each row's structure again (round 54).

The command line writes one summary CSV per batch plus a JSON and a
Markdown report per structure (reports named after the input file's
stem, which the CLI guarantees unique). Screening means asking the
summary questions - "which have a PLD above 3.3 A and a pcu net, ranked
by channel ASA?" - and then looking closely at the few that answer
them. This module is that step without the spreadsheet: typed columns,
filters, ranking with missing values kept out of the way, and a
verified route from a row back to its structure file.

Nothing here imports Qt; ui/windows/batch_results_window.py is the view.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_SUMMARY_NAME = "summary.csv"

#: Friendly headers for the summary's columns. Unknown columns (a
#: guest_<name> per guest tested, any future addition) keep their own
#: name.
COLUMN_LABELS = {
    "file": "File",
    "formula": "Formula",
    "space_group": "Space group",
    "topology": "Topology",
    "density_g_cm3": "Density (g/cm³)",
    "cell_volume_A3": "Cell volume (Å³)",
    "lcd_A": "LCD (Å)",
    "pld_A": "PLD (Å)",
    "void_fraction_percent": "Void fraction (%)",
    "accessible_dimensionality": "Channel dim.",
    "channel_asa_m2_per_g": "Channel ASA (m²/g)",
    "channel_surface_area_A2": "Channel ASA (Å²)",
    "pocket_surface_area_A2": "Pocket area (Å²)",
    "open_metal_sites": "Open metal sites",
    "dominant_pore_diameter_A": "Dominant pore (Å)",
    "raspa_folder": "RASPA folder",
    # Round 56: --compare-coremof. ccdc_ columns are the Void's values
    # computed as CoRE MOF computes its own (CCDC radii, 1.655 A probe).
    "coremof_id": "CoRE MOF id",
    "coremof_match": "CoRE match",
    "ccdc_lcd_A": "LCD, CoRE convention (Å)",
    "coremof_lcd_A": "CoRE LCD (Å)",
    "ccdc_pld_A": "PLD, CoRE convention (Å)",
    "coremof_pld_A": "CoRE PLD (Å)",
    "ccdc_asa_m2_per_g": "ASA, CoRE convention (m²/g)",
    "coremof_asa_m2_per_g": "CoRE ASA (m²/g)",
    "coremof_topology": "CoRE topology",
    "coremof_differs": "Differs from CoRE",
    "matches_index": "File matches CoRE entry",
    # Round 59: the-void-functionalize's variants.csv, joined with
    # --reference-csv.
    "fn_parent": "Parent",
    "fn_group": "Group",
    "fn_orbit": "Linker orbit",
    "fn_per_linker": "Groups per linker",
    "fn_pattern": "Pattern",
    "fn_substituents": "Groups per cell",
    "fn_status": "Build check",
    "fn_closest_ratio": "Closest contact (× vdW)",
    "fn_closest_A": "Closest contact (Å)",
    "fn_space_group": "Variant space group",
    "fn_ring_turn_deg": "Ring turn (°)",
    "errors": "Errors",
}

NUMERIC_OPERATORS = (">=", "<=", ">", "<", "=", "!=")
TEXT_OPERATORS = ("=", "!=", "contains", "does not contain")
PRESENCE_OPERATORS = ("is empty", "is not empty")


def column_label(column: str) -> str:
    if column in COLUMN_LABELS:
        return COLUMN_LABELS[column]
    if column.startswith("guest_"):
        return f"Guest {column[6:]}"
    if column.startswith("ref_"):
        # --reference-csv: the database's own values, as fetched.
        return f"Ref: {column[4:]}"
    return column


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------


@dataclass
class BatchRow:
    #: Original position in the summary (0-based), the tie-breaker.
    index: int
    #: column -> float/int for numeric columns, str otherwise; None when
    #: the cell is empty (analysis skipped or failed).
    values: dict
    report_json: Path | None = None
    report_md: Path | None = None

    def get(self, column):
        return self.values.get(column)


@dataclass
class BatchResults:
    folder: Path
    summary_path: Path
    columns: list
    rows: list
    numeric_columns: set = field(default_factory=set)

    def is_numeric(self, column: str) -> bool:
        return column in self.numeric_columns

    def operators_for(self, column: str) -> tuple:
        base = NUMERIC_OPERATORS if self.is_numeric(column) else TEXT_OPERATORS
        return base + PRESENCE_OPERATORS


def _parse_number(text):
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def load_batch_results(path) -> BatchResults:
    """A results folder (containing summary.csv, or exactly one other
    CSV) or a summary CSV itself. Raises ValueError with a message fit
    to show a person when neither is found or the file is not a
    summary."""
    path = Path(path)

    if path.is_dir():
        summary = path / DEFAULT_SUMMARY_NAME
        if not summary.is_file():
            candidates = sorted(path.glob("*.csv"))
            if len(candidates) != 1:
                raise ValueError(
                    f"No {DEFAULT_SUMMARY_NAME} in {path}. Choose the folder "
                    "the-void-analyze wrote (its --out), or the summary CSV "
                    "itself."
                )
            summary = candidates[0]
    elif path.is_file():
        summary = path
    else:
        raise ValueError(f"{path} does not exist.")

    with open(summary, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        raw_rows = list(reader)

    if "file" not in columns:
        raise ValueError(
            f"{summary.name} has no 'file' column, so it is not a "
            "the-void-analyze summary."
        )

    numeric = set()
    for column in columns:
        cells = [row.get(column, "") for row in raw_rows]
        present = [c for c in cells if c not in ("", None)]
        if column not in ("file", "formula", "space_group", "topology", "errors") and present \
                and all(_parse_number(c) is not None for c in present):
            numeric.add(column)

    folder = summary.parent
    rows = []

    for index, raw in enumerate(raw_rows):
        values = {}
        for column in columns:
            cell = raw.get(column)
            if cell in ("", None):
                values[column] = None
            elif column in numeric:
                number = _parse_number(cell)
                values[column] = int(number) if number.is_integer() and \
                    column in ("accessible_dimensionality", "open_metal_sites") else number
            else:
                values[column] = cell
        stem = Path(values.get("file") or "").stem
        json_path = folder / f"{stem}.json"
        md_path = folder / f"{stem}.md"
        rows.append(BatchRow(
            index=index,
            values=values,
            report_json=json_path if stem and json_path.is_file() else None,
            report_md=md_path if stem and md_path.is_file() else None,
        ))

    return BatchResults(folder, summary, columns, rows, numeric)


# ----------------------------------------------------------------------
# Filtering and ranking
# ----------------------------------------------------------------------


@dataclass
class Filter:
    column: str
    operator: str
    value: str = ""

    def validate(self, results: BatchResults) -> None:
        if self.column not in results.columns:
            raise ValueError(f"No column {self.column!r}.")
        if self.operator not in results.operators_for(self.column):
            raise ValueError(
                f"{self.operator!r} does not apply to {column_label(self.column)}."
            )
        if (
            results.is_numeric(self.column)
            and self.operator in NUMERIC_OPERATORS
            and _parse_number(self.value) is None
        ):
            raise ValueError(
                f"{column_label(self.column)} {self.operator} needs a number, "
                f"not {self.value!r}."
            )

    def matches(self, row: BatchRow, results: BatchResults) -> bool:
        value = row.get(self.column)

        if self.operator == "is empty":
            return value is None
        if self.operator == "is not empty":
            return value is not None
        if value is None:
            # A missing number is not "below 3" or "above 3"; a row
            # whose analysis did not run should not pass for one that
            # did. Use "is empty" to find them.
            return False

        if results.is_numeric(self.column):
            target = _parse_number(self.value)
            return {
                ">=": value >= target, "<=": value <= target,
                ">": value > target, "<": value < target,
                "=": value == target, "!=": value != target,
            }[self.operator]

        text, target = str(value).casefold(), self.value.strip().casefold()
        return {
            "=": text == target,
            "!=": text != target,
            "contains": target in text,
            "does not contain": target not in text,
        }[self.operator]

    def describe(self) -> str:
        if self.operator in PRESENCE_OPERATORS:
            return f"{self.column} {self.operator}"
        return f"{self.column} {self.operator} {self.value}"


def view(
    results: BatchResults,
    *,
    filters=(),
    search: str = "",
    sort_column: str | None = None,
    descending: bool = False,
    top_n: int | None = None,
) -> list[BatchRow]:
    """The rows to show, in order.

    Filters combine with AND. The search matches any cell, case-
    insensitively. Sorting keeps rows with no value in the sort column
    at the bottom in both directions - a failed pore analysis must not
    top a "largest PLD first" ranking - and is stable, so equal values
    stay in file order. top_n applies last: the top N of what survives
    the filters.
    """
    for condition in filters:
        condition.validate(results)

    needle = search.strip().casefold()
    rows = [
        row for row in results.rows
        if all(condition.matches(row, results) for condition in filters)
        and (not needle or any(
            needle in str(v).casefold() for v in row.values.values() if v is not None
        ))
    ]

    if sort_column:
        present = [r for r in rows if r.get(sort_column) is not None]
        missing = [r for r in rows if r.get(sort_column) is None]
        numeric = results.is_numeric(sort_column)

        def key(row):
            value = row.get(sort_column)
            return value if numeric else str(value).casefold()

        # Python's sort is stable in both directions (reverse=True keeps
        # equal elements in their original order), so ties stay in file
        # order either way.
        present.sort(key=key, reverse=descending)
        rows = present + missing

    if top_n is not None and top_n > 0:
        rows = rows[:top_n]

    return rows


def write_view_csv(results: BatchResults, rows, path) -> None:
    """The rows as shown (with their rank), in the summary's columns."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["rank"] + results.columns)
        for rank, row in enumerate(rows, start=1):
            writer.writerow([rank] + [
                "" if row.get(c) is None else row.get(c) for c in results.columns
            ])


# ----------------------------------------------------------------------
# Reports and structures
# ----------------------------------------------------------------------


def read_report(row: BatchRow) -> dict | None:
    if row.report_json is None:
        return None
    try:
        return json.loads(row.report_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def report_markdown(row: BatchRow) -> str:
    """The structure's Markdown report as the CLI wrote it; rebuilt from
    the JSON if only that survives; otherwise what the summary knows."""
    if row.report_md is not None:
        try:
            return row.report_md.read_text(encoding="utf-8")
        except OSError:
            pass

    report = read_report(row)
    if report is not None:
        from thevoid_core.services.report import to_markdown

        try:
            return to_markdown(report)
        except Exception:
            pass

    name = row.get("file") or "this structure"
    error = row.get("errors")
    return (
        f"# {name}\n\nNo report was written for this structure."
        + (f"\n\nThe batch recorded: {error}" if error else "")
        + "\n"
    )


@dataclass
class Located:
    #: "verified" (checksum matches), "unverified" (the report recorded
    #: no checksum), "mismatch" (a file of that name exists but its
    #: contents differ), or "missing".
    status: str
    path: Path | None = None
    tried: list = field(default_factory=list)
    recorded_path: str | None = None

    @property
    def message(self) -> str:
        if self.status == "verified":
            return f"{self.path} (SHA-256 matches the report)"
        if self.status == "unverified":
            return f"{self.path} (the report has no checksum to compare)"
        if self.status == "mismatch":
            return (
                f"{self.path} exists, but its contents are not the file "
                "the report was made from (SHA-256 differs)."
            )
        return (
            f"Could not find {self.recorded_path or 'the structure file'}; "
            f"looked in {len(self.tried)} place(s)."
        )


def locate_structure(row: BatchRow, results: BatchResults, extra_dirs=()) -> Located:
    """Find the file a row was computed from.

    The report stores the path as it was given on the command line
    (often relative to wherever the command ran) and its SHA-256. Paths
    tried, in order: the stored path if absolute; the stored path
    relative to the results folder, its parent and grandparent; then
    the bare file name in any extra_dirs (folders the person pointed at
    earlier), the results folder and its parent. A candidate only counts
    when its SHA-256 matches the report's - a file of the same name that
    has since been edited, or is a different structure entirely, would
    otherwise load silently next to numbers that are not its own.
    """
    from thevoid_core.services.report import file_sha256

    report = read_report(row) or {}
    record = report.get("input", {}) or {}
    recorded = record.get("file") or row.get("file")
    checksum = record.get("sha256")

    if not recorded:
        return Located("missing")

    stored = Path(recorded)
    name = stored.name
    folder = results.folder
    candidates = []

    if stored.is_absolute():
        candidates.append(stored)
    else:
        for base in (folder, folder.parent, folder.parent.parent):
            candidates.append(base / stored)
    for base in list(extra_dirs) + [folder, folder.parent]:
        candidates.append(Path(base) / name)

    seen, tried, mismatch = set(), [], None

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        tried.append(resolved)
        if not resolved.is_file():
            continue
        if checksum is None:
            return Located("unverified", resolved, tried, recorded)
        if file_sha256(resolved) == checksum:
            return Located("verified", resolved, tried, recorded)
        mismatch = mismatch or resolved

    if mismatch is not None:
        return Located("mismatch", mismatch, tried, recorded)
    return Located("missing", None, tried, recorded)


def check_file(path, row: BatchRow) -> Located:
    """For a file the person picked by hand: does it match the report?"""
    from thevoid_core.services.report import file_sha256

    report = read_report(row) or {}
    checksum = (report.get("input") or {}).get("sha256")
    path = Path(path)
    if checksum is None:
        return Located("unverified", path, [path])
    status = "verified" if file_sha256(path) == checksum else "mismatch"
    return Located(status, path, [path], (report.get("input") or {}).get("file"))
