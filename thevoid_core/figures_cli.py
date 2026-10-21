"""`the-void-figures`: publication figures from a the-void-analyze results
folder (round 58; services/figures.py).

    the-void-figures results                       # the default set
    the-void-figures results --scatter pld_A channel_asa_m2_per_g --colour topology
    the-void-figures results --hist pld_A --filter "errors is empty"
    the-void-figures results --parity coremof_pld_A ccdc_pld_A --width double

Each figure is written as SVG, PDF and 600-dpi PNG, with a JSON of the
exact points, filters and style beside it, into --out (default:
<results>/figures). `the-void-analyze --figures` writes the default set
at the end of a batch.

Exit status: 0 on success, 1 if nothing could be plotted, 2 for bad
arguments.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from thevoid_core.version import __version__


def build_parser():
    parser = argparse.ArgumentParser(
        prog="the-void-figures",
        description=f"Publication figures from a batch (the Void {__version__}).")
    parser.add_argument("results", help="results folder (or its summary CSV)")
    parser.add_argument("--out", default=None, help="folder for the figures (default: <results>/figures)")
    parser.add_argument("--name", default=None, help="file name stem (default: from the figure)")
    kind = parser.add_mutually_exclusive_group()
    kind.add_argument("--scatter", nargs=2, metavar=("X", "Y"))
    kind.add_argument("--hist", metavar="COLUMN")
    kind.add_argument("--parity", nargs=2, metavar=("REFERENCE", "THE_VOID"),
                      help="e.g. coremof_pld_A ccdc_pld_A")
    parser.add_argument("--colour", "--color", dest="colour", default=None,
                        help="text column to colour points by (at most 3 categories + Other)")
    parser.add_argument("--log-x", action="store_true")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--bins", type=int, default=None,
                        help="histogram bins (default: numpy's auto rule)")
    parser.add_argument("--title", default="")
    parser.add_argument("--width", choices=["single", "double"], default="single",
                        help="85 mm or 175 mm (default: %(default)s)")
    parser.add_argument("--formats", default="svg,pdf,png", help="default: %(default)s")
    parser.add_argument("--filter", action="append", default=[], metavar='"COLUMN OP VALUE"',
                        help='e.g. "pld_A >= 3.3" or "errors is empty"; repeatable')
    parser.add_argument("--version", action="version", version=f"the Void {__version__}")
    return parser


def parse_filter(text: str):
    """'pld_A >= 3.3' / 'errors is empty' / 'topology contains pcu'."""
    from thevoid_core.services.batch_results import (
        NUMERIC_OPERATORS,
        PRESENCE_OPERATORS,
        TEXT_OPERATORS,
        Filter,
    )

    parts = shlex.split(text)
    if len(parts) >= 3 and " ".join(parts[1:]) in PRESENCE_OPERATORS:
        return Filter(parts[0], " ".join(parts[1:]))
    if len(parts) >= 5 and " ".join(parts[1:4]) == "does not contain":
        return Filter(parts[0], "does not contain", " ".join(parts[4:]))
    if len(parts) >= 3 and parts[1] in NUMERIC_OPERATORS + TEXT_OPERATORS:
        return Filter(parts[0], parts[1], " ".join(parts[2:]))
    raise ValueError(f"cannot read filter {text!r}: expected 'COLUMN OPERATOR VALUE'")


def main(argv=None) -> int:
    from thevoid_core.services import figures
    from thevoid_core.services.batch_results import load_batch_results, view

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        results = load_batch_results(args.results)
        filters = [parse_filter(text) for text in args.filter]
        rows = view(results, filters=filters)
    except ValueError as exc:
        parser.error(str(exc))
    formats = tuple(f.strip().lower() for f in args.formats.split(",") if f.strip())
    if not formats or any(f not in figures.FORMATS for f in formats):
        parser.error(f"--formats: choose from {', '.join(figures.FORMATS)}")
    style = figures.FigureStyle(width=args.width, formats=formats)
    out = Path(args.out) if args.out else results.folder / "figures"

    if args.scatter or args.hist or args.parity:
        if args.scatter:
            spec = figures.FigureSpec("scatter", *args.scatter, colour_by=args.colour)
            stem = f"{args.scatter[0]}_vs_{args.scatter[1]}"
        elif args.hist:
            spec = figures.FigureSpec("histogram", args.hist, bins=args.bins)
            stem = f"histogram_{args.hist}"
        else:
            spec = figures.FigureSpec("parity", *args.parity, colour_by=args.colour)
            stem = f"parity_{args.parity[1]}"
        spec.log_x, spec.log_y, spec.title = args.log_x, args.log_y, args.title
        specs = [(args.name or stem, spec)]
    else:
        specs = figures.default_specs(results)
        if not specs:
            print("This batch has none of the columns the default figures use "
                  "(pld_A, lcd_A, channel_asa_m2_per_g, ...).", file=sys.stderr)
            return 1

    written = []
    for stem, spec in specs:
        try:
            figure, data = figures.draw(results, rows, spec, style)
        except ValueError as exc:
            parser.error(str(exc))
        if not data.points:
            print(f"skipped {stem}: no row has the values it needs", file=sys.stderr)
            continue
        written += figures.save(figure, data, spec, style, out / stem, results=results,
                                filters=filters)
        if data.excluded:
            print(f"{stem}: {len(data.points)} plotted, {len(data.excluded)} left out "
                  "(listed in its JSON)")
    if not written:
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
