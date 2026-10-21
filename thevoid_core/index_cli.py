"""`the-void-index`: a shared index of analyses (round 65;
docs/shared-index-design.md).

    the-void-index init MyLab-index --name "McKelvey Lab"
    the-void-index add MyLab-index results/ --contributor "B. Garmejani" --note "SI of ..."
    the-void-index build MyLab-index
    the-void-index find MyLab-index structure.cif
    the-void-index verify MyLab-index
    the-void-index withdraw MyLab-index 3f2a9c... --reason "wrong CIF" --by "B. G."
    the-void-index stale MyLab-index                    # numbers a later version changed
    the-void-index refresh MyLab-index --contributor "B. G." [-j 4] [--limit 20] [--dry-run]

`add` takes the-void-analyze results folders. Each report's structure
file must still be found, and match the report's SHA-256: the index's
identity is computed from the file. A structure file is copied into
the index only when its licence allows (services/index/licence.py); a
CSD-derived file never is. `build` writes build/, a results folder the
Batch Results window opens (File > Open Batch Results...). --structures
DIR (build, find, verify, add) points at folders holding structure
files that the index does not store, found by their SHA-256.

Exit status: 0 on success; 1 if anything was skipped (add), a problem
was found (verify, build), or nothing matched (find); 2 for bad
arguments or a folder that is not an index.
"""

from __future__ import annotations

import argparse
import sys

from thevoid_core.version import __version__


def build_parser():
    parser = argparse.ArgumentParser(prog="the-void-index",
                                     description=f"A shared index of analyses (the Void "
                                                 f"{__version__}).")
    parser.add_argument("--version", action="version", version=f"the Void {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="make a new, empty index")
    init.add_argument("index")
    init.add_argument("--name", required=True)
    init.add_argument("--results-licence", default="CC-BY-4.0",
                      help="the licence the results would carry in a public release "
                           "(default: %(default)s)")
    init.add_argument("--maintainer", action="append", default=[])

    add = commands.add_parser("add", help="add the reports of the-void-analyze results folders")
    add.add_argument("index")
    add.add_argument("results", nargs="+")
    add.add_argument("--contributor", required=True, help="a name, or initials")
    add.add_argument("--note", default="")
    add.add_argument("--origin", choices=["experimental", "hypothetical", "computed"],
                     default=None, help="where these structures come from, if known")
    add.add_argument("--licence", default=None,
                     help="SPDX licence of files nothing else decides (CC-BY-4.0, CC0-1.0, "
                          "...); never overrides a CSD header")
    add.add_argument("--licence-basis", default=None, help="why, e.g. 'our own structures'")
    add.add_argument("--structures", action="append", default=[], metavar="DIR")

    for name, text in (("build", "write build/ (summary.csv, reports, index.json.gz)"),
                       ("verify", "check entries, stored files and replicates")):
        sub = commands.add_parser(name, help=text)
        sub.add_argument("index")
        sub.add_argument("--structures", action="append", default=[], metavar="DIR")

    find = commands.add_parser("find", help="has anyone analysed this structure?")
    find.add_argument("index")
    find.add_argument("structure")
    find.add_argument("--structures", action="append", default=[], metavar="DIR")

    stale = commands.add_parser("stale", help="entries whose numbers a later version of the "
                                              "Void computes differently")
    stale.add_argument("index")
    refresh = commands.add_parser("refresh", help="analyse stale entries again with this "
                                                  "version (adds new entries)")
    refresh.add_argument("index")
    refresh.add_argument("--contributor", required=True, help="who ran the refresh")
    refresh.add_argument("--structures", action="append", default=[], metavar="DIR")
    refresh.add_argument("--limit", type=int, default=None, help="at most this many")
    refresh.add_argument("--dry-run", action="store_true", help="list what would be rerun")
    refresh.add_argument("-j", "--jobs", type=int, default=1,
                         help="analyse this many structures at once (default: %(default)s)")

    withdraw = commands.add_parser("withdraw", help="hide an entry from the build (kept on "
                                                    "record)")
    withdraw.add_argument("index")
    withdraw.add_argument("entry", help="an entry id, or a unique prefix of one")
    withdraw.add_argument("--reason", required=True)
    withdraw.add_argument("--by", required=True)
    return parser


def _progress(name, done, total):
    print(f"  {done}/{total} {name}", file=sys.stderr)


def main(argv=None) -> int:
    import logging
    import warnings

    from thevoid_core.services.index.store import Index, IndexProblem

    parser = build_parser()
    args = parser.parse_args(argv)
    # As the-void-analyze: pymatgen's CIF parser warns about every file.
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("pymatgen").setLevel(logging.ERROR)
    if args.command == "init":
        try:
            index = Index.init(args.index, name=args.name, results_licence=args.results_licence,
                               maintainers=args.maintainer)
        except IndexProblem as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"made {index.path} ({args.name}); add results with: the-void-index add "
              f"{args.index} RESULTS --contributor NAME")
        return 0
    try:
        index = Index.open(args.index)
    except IndexProblem as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "add":
        declared = ({"spdx": args.licence, "basis": args.licence_basis}
                    if args.licence or args.licence_basis else None)
        skipped = 0
        for folder in args.results:
            try:
                result = index.add_results(folder, contributor=args.contributor, note=args.note,
                                           origin=args.origin, declared_licence=declared,
                                           search_dirs=args.structures)
            except ValueError as exc:
                print(f"{folder}: {exc}", file=sys.stderr)
                skipped += 1
                continue
            print(f"{folder}: {len(result.added)} added, {len(result.present)} already present, "
                  f"{len(result.skipped)} skipped; {len(result.stored)} structure file(s) "
                  f"stored, {len(result.withheld)} withheld; {result.decisions} match "
                  "decision(s) recorded")
            for name, reason in result.skipped:
                print(f"  skipped {name}: {reason}")
            for name, basis in result.withheld:
                print(f"  not stored {name}: {basis}")
            skipped += len(result.skipped)
        return 1 if skipped else 0

    if args.command == "build":
        result = index.build(search_dirs=args.structures)
        print(f"{result.folder}: {result.rows} entries, {result.structures} structures, "
              f"{result.determinations} determinations ({result.fits} new match(es))")
        if result.undecided:
            print(f"  {len(result.undecided)} pair(s) not compared: a structure file is "
                  "neither stored nor under --structures; they count as different")
        for problem in result.problems:
            print(f"  problem: {problem}")
        print(f"  open it in File > Open Batch Results..., or read {result.folder}/summary.csv")
        return 1 if result.problems else 0

    if args.command == "find":
        found = index.find(args.structure, search_dirs=args.structures)
        if not found:
            print("not in the index")
            return 1
        for relation, entry in found:
            s, a = entry["structure"], entry["analysis"]
            summary = a["summary"]
            print(f"{relation:30s} {entry['entry_id'][:12]}  {s['file_name']}  "
                  f"the Void {a['software'].get('the_void', '?')}  "
                  f"{entry['contributor']['name']} {entry['contributor']['added']}  "
                  f"PLD {summary.get('pld_A')}  channel ASA {summary.get('channel_asa_m2_per_g')}")
        return 0

    if args.command == "verify":
        problems = index.verify(search_dirs=args.structures)
        for problem in problems:
            print(problem)
        print(f"{len(problems)} problem(s)")
        return 1 if problems else 0

    if args.command == "stale":
        stale = index.stale()
        for entry, verdict in stale:
            print(f"{entry['entry_id'][:12]}  {entry['structure']['file_name']}  the Void "
                  f"{verdict.version}: {', '.join(sorted(verdict.columns))}")
            for reason in verdict.reasons():
                print(f"    {reason}")
        print(f"{len(stale)} stale entr{'y' if len(stale) == 1 else 'ies'} "
              f"(the-void-index refresh {args.index} --contributor NAME analyses them again)")
        return 0

    if args.command == "refresh":
        if args.jobs < 1:
            parser.error("--jobs must be at least 1")
        result = index.refresh(contributor=args.contributor, search_dirs=args.structures,
                               limit=args.limit, dry_run=args.dry_run, progress=_progress,
                               jobs=args.jobs)
        if args.dry_run:
            print(f"would refresh {len(result.would_refresh)} of {result.planned} stale "
                  f"entr{'y' if result.planned == 1 else 'ies'}")
        else:
            print(f"refreshed {len(result.refreshed)} of {result.planned} stale "
                  f"entr{'y' if result.planned == 1 else 'ies'}")
        for entry_id, name, reason in result.skipped:
            print(f"  skipped {entry_id[:12]} {name}: {reason}")
        if result.refreshed:
            print(f"  then: the-void-index build {args.index}")
        return 1 if result.skipped else 0

    if args.command == "withdraw":
        try:
            path = index.withdraw(args.entry, reason=args.reason, by=args.by)
        except IndexProblem as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"withdrawn: {path.relative_to(index.path)}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
