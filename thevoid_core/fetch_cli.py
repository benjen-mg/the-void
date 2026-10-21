"""`the-void-fetch`: download reference structures for screening
(round 56; services/reference/fetch.py).

    the-void-fetch coremof --out coremof --pld-min 3.3 --metals Cu --limit 50
    the-void-fetch coremof --pld-min 6 --topology pcu --dry-run
    the-void-fetch mofxdb --database hMOF --out hmof --pld-min 4 --limit 200

then

    the-void-analyze coremof/*.cif --out results --compare-coremof
    the-void-analyze hmof/*.cif --out results --reference-csv hmof/reference.csv

Exit status: 0 on success, 1 if the download failed, 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import sys

from thevoid_core.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="the-void-fetch",
        description=f"Download reference MOF structures (the Void {__version__}).",
    )
    sub = parser.add_subparsers(dest="source", required=True)

    core = sub.add_parser("coremof", help="CoRE MOF 2024 (openly licensed subset)",
                          description="Choose from the bundled CoRE MOF 2024 index and "
                          "download the matching structures CoRE MOF redistributes "
                          "(from papers' SI, CC-BY-4.0). CSD-derived matches are listed "
                          "in not_downloadable.csv.")
    core.add_argument("--out", default="coremof_structures", help="folder (default: %(default)s)")
    core.add_argument("--subset", choices=["ASR", "FSR", "Ion"],
                      help="ASR: all solvent removed; FSR: free solvent removed; Ion: charged")
    for name in ("pld", "lcd"):
        core.add_argument(f"--{name}-min", type=float, help=f"minimum {name.upper()}, A")
        core.add_argument(f"--{name}-max", type=float, help=f"maximum {name.upper()}, A")
    core.add_argument("--asa-min", type=float, help="minimum ASA, m2/g (CoRE's, 1.655 A probe)")
    core.add_argument("--metals", default="", help="comma-separated metals every hit must contain")
    core.add_argument("--topology", help="single-node net, e.g. pcu")
    core.add_argument("--oms", choices=["yes", "no"], help="with or without open metal sites")
    core.add_argument("--limit", type=int, default=None, help="at most this many structures")
    core.add_argument("--dry-run", action="store_true", help="only say what would be downloaded")
    core.add_argument("--refresh", action="store_true",
                      help="download the CoRE MOF archive again instead of using the cached copy")

    mofx = sub.add_parser("mofxdb", help="MOFX-DB (hMOF and others), online",
                          description="Query the MOFX-DB API "
                          "(https://mof.tech.northwestern.edu) and download the hits.")
    from thevoid_core.services.reference.fetch import MOFXDB_DATABASES

    mofx.add_argument("--database", default="hMOF", choices=MOFXDB_DATABASES)
    mofx.add_argument("--out", default="mofxdb_structures", help="folder (default: %(default)s)")
    for name in ("pld", "lcd", "vf", "sa-m2g"):
        mofx.add_argument(f"--{name}-min", type=float)
        mofx.add_argument(f"--{name}-max", type=float)
    mofx.add_argument("--name", help="name prefix, e.g. hMOF-1")
    mofx.add_argument("--elements", default="", help="comma-separated elements, e.g. Zn,O")
    mofx.add_argument("--limit", type=int, default=100, help="at most this many (default: %(default)s)")

    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"the Void {__version__}")
    return parser


def coremof_filters(args) -> dict:
    return {
        "subset": args.subset,
        "pld_min": args.pld_min, "pld_max": args.pld_max,
        "lcd_min": args.lcd_min, "lcd_max": args.lcd_max,
        "asa_min": args.asa_min,
        "metals": [m for m in args.metals.split(",") if m.strip()],
        "topology": args.topology,
        "has_oms": {"yes": True, "no": False}.get(args.oms),
    }


class ProgressPrinter:
    """progress(message, done, total) for the fetch functions: a live
    counter on a terminal; in a log, one line per step and per tenth of
    it (round 58: a fetch written to a file was 150 overwritten lines).
    Every "Writing <entry>" message counts as one step."""

    def __init__(self, stream, *, interactive: bool, quiet: bool = False):
        self.stream, self.interactive, self.quiet = stream, interactive, quiet
        self.last = {}

    def __call__(self, message, done, total):
        if self.quiet or not total:
            return None
        if self.interactive:
            print(f"\r{message}: {done}/{total}" + " " * 10, end="", file=self.stream, flush=True)
            return None
        step = "Writing" if message.startswith("Writing") else message
        tenth = int(10 * done / total)
        if self.last.get(step) != tenth:
            self.last[step] = tenth
            text = f"Writing structures: {done}/{total}" if step == "Writing" else f"{message}: {done}/{total}"
            print(text, file=self.stream, flush=True)
        return None


def main(argv=None) -> int:
    from thevoid_core.services.reference import fetch
    from thevoid_core.services.reference.net import Cancelled, DownloadError

    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "limit", None) is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    progress = ProgressPrinter(sys.stderr, interactive=sys.stderr.isatty(), quiet=args.quiet)

    try:
        if args.source == "coremof":
            filters = coremof_filters(args)
            if args.dry_run:
                downloadable, others = fetch.plan_coremof(filters, limit=args.limit)
                print(f"{len(downloadable)} downloadable, {len(others)} CSD-only "
                      "(listed, not downloaded)")
                for entry in downloadable[:20]:
                    print(f"  {entry.id}  PLD {entry.pld_A:.2f}  LCD {entry.lcd_A:.2f}  "
                          f"{entry.topology or '-'}")
                if len(downloadable) > 20:
                    print(f"  ... and {len(downloadable) - 20} more")
                return 0
            result = fetch.fetch_coremof(args.out, filters, limit=args.limit, progress=progress,
                                         refresh=args.refresh)
        else:
            filters = {
                "pld_min": args.pld_min, "pld_max": args.pld_max,
                "lcd_min": args.lcd_min, "lcd_max": args.lcd_max,
                "vf_min": args.vf_min, "vf_max": args.vf_max,
                "sa_m2g_min": args.sa_m2g_min, "sa_m2g_max": args.sa_m2g_max,
                "name": args.name,
                "elements": [e for e in args.elements.split(",") if e.strip()],
            }
            result = fetch.fetch_mofxdb(args.out, database=args.database, limit=args.limit,
                                        progress=progress, **filters)
    except (DownloadError, Cancelled) as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        print(f"\n{len(result.written)} structure(s) written to {result.folder}"
              + (f"; {len(result.not_downloadable)} CSD-only listed in not_downloadable.csv"
                 if result.not_downloadable else ""))
        for warning in result.warnings:
            print(f"warning: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
