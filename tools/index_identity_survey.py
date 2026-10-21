"""Survey the shared index's structure identity over a folder of
structures (round 64): the measurements in docs/shared-index-design.md,
Appendix A, made with services/index/identity.py itself.

    python tools/index_identity_survey.py FOLDER [--jobs 2] [--out survey.json]
        [--no-variants] [--limit N]

For every structure:

- its framework_key;
- unless --no-variants, whether the key survives a pymatgen rewrite, a
  1x2x1 supercell (at most 300 atoms), and a rotation with origin shift
  and shuffled atoms.

Then, for every key held by two or more files, the structure_ids and
determination_ids assign_ids gives. The summary goes to stdout, and
every record and placement to --out.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _key_record(path, variants):
    import logging
    import warnings

    import numpy as np
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter

    from thevoid_core.services.index.identity import framework_key
    from thevoid_core.services.report import file_sha256

    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    record = {"file": Path(path).name, "path": str(path)}
    try:
        start = time.time()
        structure = Structure.from_file(path)
        key = framework_key(structure)
        record.update(key=key.key, formula=key.formula, n_atoms=key.n_atoms,
                      dimensionalities=list(key.dimensionalities), sha256=file_sha256(path),
                      seconds=round(time.time() - start, 3))
        if variants:
            record["rewrite"] = framework_key(
                Structure.from_str(str(CifWriter(structure)), fmt="cif")).key == key.key
            if len(structure) <= 300:
                record["supercell"] = framework_key(structure * (1, 2, 1)).key == key.key
            rng = np.random.default_rng(len(record["file"]))
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            order = rng.permutation(len(structure))
            shift = rng.uniform(0, 1, 3)
            moved = Structure(Lattice(structure.lattice.matrix @ q.T),
                              [structure[i].species for i in order],
                              [(structure[i].frac_coords + shift) % 1 for i in order])
            record["moved"] = framework_key(moved).key == key.key
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _load_structure(path):
    from pymatgen.core import Structure

    return Structure.from_file(path)


def _place_group(records):
    import functools
    import logging
    import warnings

    from thevoid_core.services.index.identity import MatchCache, Member, assign_ids

    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    cache = MatchCache()
    start = time.time()
    members = [Member(r["file"], r["sha256"], r["key"],
                      functools.partial(_load_structure, r["path"])) for r in records]
    placed = assign_ids(members, cache)
    return [p.__dict__ for p in placed], cache.fits, time.time() - start


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-variants", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    paths = sorted(Path(args.folder).rglob("*.cif"))[: args.limit]
    if not paths:
        print(f"no CIFs under {args.folder}", file=sys.stderr)
        return 2
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        records = list(pool.map(_key_record, paths, [not args.no_variants] * len(paths),
                                chunksize=8))
    ok = [r for r in records if "key" in r]
    print(f"{len(paths)} structures, {len(ok)} keyed, {len(paths) - len(ok)} failed")
    for name in ("rewrite", "supercell", "moved"):
        have = [r[name] for r in ok if name in r]
        if have:
            print(f"  key unchanged after {name}: {sum(have)}/{len(have)}")
    seconds = sorted(r["seconds"] for r in ok)
    print(f"  seconds per key: median {statistics.median(seconds):.3f}, "
          f"p99 {seconds[int(0.99 * (len(seconds) - 1))]:.2f}")

    by_key = defaultdict(list)
    for r in ok:
        by_key[r["key"]].append(r)
    shared = [g for g in by_key.values() if len(g) > 1]
    placements = [{"id": g[0]["file"], "key": g[0]["key"], "structure_id": g[0]["key"] + ".1",
                   "determination_id": g[0]["key"] + ".1.1"}
                  for g in by_key.values() if len(g) == 1]
    fits = 0
    group_seconds = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for placed, n_fits, sec in pool.map(_place_group, shared):
            placements += placed
            fits += n_fits
            group_seconds.append(sec)
    structures = {p["structure_id"] for p in placements}
    determinations = {p["determination_id"] for p in placements}
    split = sum(1 for g in shared
                if len({p["structure_id"] for p in placements if p["key"] == g[0]["key"]}) > 1)
    print(f"  keys: {len(by_key)} ({sum(1 for g in by_key.values() if len(g) == 1)} single, "
          f"{len(shared)} shared by {sum(len(g) for g in shared)} files)")
    print(f"  shared keys holding more than one structure: {split}")
    print(f"  structures: {len(structures)}; determinations: {len(determinations)}")
    if group_seconds:
        print(f"  matching: {fits} fits, median {statistics.median(group_seconds):.1f} s "
              f"per group, longest {max(group_seconds):.0f} s")
    errors = Counter(r["error"].split(":")[0] for r in records if "error" in r)
    if errors:
        print(f"  failures: {dict(errors)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"records": records, "placements": placements},
                                             indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
