"""`the-void-functionalize`: hypothetical variants of a MOF with its
linkers' ring hydrogens replaced by functional groups (round 59;
services/functionalize.py).

    the-void-functionalize IRMOF-1.cif --list
    the-void-functionalize IRMOF-1.cif --groups NH2,Br,CH3 --per-linker 1,2
    the-void-analyze IRMOF-1_functionalized/*.cif --out screen \\
        --reference-csv IRMOF-1_functionalized/variants.csv --figures

Each variant is a P1 CIF. Beside them: the parent, functionalization.json
(the method, every bond length and threshold used, and each variant's
closest contacts, bends and rotor angles) and variants.csv, whose fn_
columns `--reference-csv` joins into a batch summary.

Exit status: 0 if any variant was written, 1 if none could be, 2 for
bad arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thevoid_core.version import __version__


def build_parser():
    from thevoid_core.services.functionalize import GROUPS

    parser = argparse.ArgumentParser(
        prog="the-void-functionalize",
        description=f"Functionalised variants of a MOF (the Void {__version__}).")
    parser.add_argument("structure", help="parent structure (CIF or anything the Void reads)")
    parser.add_argument("--groups", default="NH2",
                        help=f"comma-separated, from {', '.join(GROUPS)} (default: %(default)s)")
    parser.add_argument("--per-linker", default="1",
                        help="groups per linker, comma-separated, e.g. 1,2 (default: %(default)s)")
    parser.add_argument("--orbit", type=int, action="append", default=None,
                        help="only this linker orbit (see --list); repeatable")
    parser.add_argument("--out", default=None,
                        help="output folder (default: <structure>_functionalized)")
    parser.add_argument("--no-ring-turn", action="store_true",
                        help="never turn a substituted ring to make room (by default a "
                             "tight or clashing variant's rings are turned about the linker "
                             "axis by the smallest angle that clears it)")
    parser.add_argument("--keep-clashes", action="store_true",
                        help="also write variants that fail the contact check")
    parser.add_argument("--list", action="store_true",
                        help="list the substitutable sites, linker orbits and patterns, and stop")
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--version", action="version", version=f"the Void {__version__}")
    return parser


def describe(parent) -> str:
    from thevoid_core.services import functionalize as fz

    lines = [f"{Path(parent.source).name if parent.source else 'structure'}: "
             f"{parent.structure.composition.formula.replace(' ', '')}, {parent.space_group} "
             f"({len(parent.operations)} symmetry operations used), "
             f"{len(parent.sites)} ring C-H on linkers"]
    for note in parent.notes:
        lines.append(f"  note: {note}")
    for orbit in parent.orbits:
        lines.append(f"  orbit {orbit.number}: {len(orbit.linkers)} x {orbit.formula}, "
                     f"{len(orbit.sites)} ring C-H each")
        for m, patterns in orbit.patterns.items():
            names = "; ".join(
                f"{d or 'p' + str(i)} "
                f"({' '.join(fz.site_name(parent.structure, h) for h in pattern)})"
                for i, (pattern, d) in enumerate(patterns, start=1))
            lines.append(f"    {m} per linker: {len(patterns)} pattern(s) ({names})")
    return "\n".join(lines)


def main(argv=None) -> int:
    from thevoid_core.services import functionalize as fz

    import warnings

    warnings.filterwarnings("ignore", module="pymatgen")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        groups = [fz.group(name) for name in args.groups.split(",") if name.strip()]
        per_linker = tuple(sorted({int(x) for x in args.per_linker.split(",") if x.strip()}))
    except ValueError as exc:
        parser.error(str(exc))
    if not groups or not per_linker or min(per_linker) < 1:
        parser.error("give at least one group and per-linker counts of 1 or more")

    path = Path(args.structure)
    if not path.is_file():
        parser.error(f"{path} not found")
    try:
        from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader

        structure = PymatgenCIFLoader().load(str(path)).structure
        parent = fz.analyze_parent(structure, source=str(path), max_per_linker=max(per_linker))
    except Exception as exc:  # a structure the Void cannot read or bond
        print(f"could not prepare {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.list or not args.quiet:
        print(describe(parent))
    if args.list:
        return 0
    if not parent.sites:
        return 1

    out = Path(args.out) if args.out else path.with_name(f"{path.stem}_functionalized")

    def progress(name, done, total):
        if not args.quiet:
            print(f"[{done + 1}/{total}] {name}", file=sys.stderr, flush=True)

    record = fz.generate(parent, groups, out, per_linker=per_linker, orbits=args.orbit,
                         keep_clashes=args.keep_clashes, progress=progress,
                         ring_turn=not args.no_ring_turn)
    written = 0
    for variant in record["_variants"]:
        closest = variant.closest[0] if variant.closest else None
        detail = (f"closest {closest[1]:.2f} A = {closest[0]:.2f} x vdW ({closest[2]})"
                  if closest else "")
        if variant.status == "clash" and not args.keep_clashes:
            line = f"rejected {variant.name}: {variant.reason}"
        else:
            written += 1
            line = f"{variant.status:5s} {variant.file}  {variant.space_group}  {detail}"
            if variant.ring_turn:
                line += f"  ring turned {abs(variant.ring_turn):.0f} deg"
        if not args.quiet or variant.status == "clash":
            print(line)
    if not args.quiet:
        print(f"{written} variant(s) in {out}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
