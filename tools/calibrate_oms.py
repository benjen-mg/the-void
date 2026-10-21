"""Calibrate open-metal-site detection for metals it did not check
(round 69), and write thevoid_core/data/oms_calibration.json.

    python tools/calibrate_oms.py SI_FOLDER [--jobs 2] [--features features.jsonl]

SI_FOLDER is CoRE MOF 2024's SI structures as the-void-fetch coremof
downloads them (2,664 CIFs, CC-BY-4.0). The labels are CoRE's own:
a metal is open in a structure when it is in that entry's oms_metals.

Method
------
The geometric test is not changed (services/oms_detection.classify_site:
OPEN when the coordination number is below the metal's "saturated"
number and either asymmetry > 0.15 or the widest empty cone > 80
degrees). Only the saturated coordination number of each new metal is
set here, from the data:

1. Every metal site of every structure is classified with
   detect_open_metal_sites over all metals, keeping its coordination
   number, asymmetry and widest empty cone.
2. One item is one metal in one structure: open when any of its sites
   is open (CoRE labels metals per structure, not sites).
3. The structures are split in two by a hash of the paper they came
   from (DOI), so the ASR, FSR and ION versions of one structure, and
   a paper's related frameworks, stay on one side.
4. Metals are calibrated in groups: the lanthanides with Y (whose
   coordination chemistry is one family, and whose one-metal samples
   are too small to fit alone), the actinides Th-Pu, and every other
   metal on its own. A group is calibrated only when each half holds
   at least MIN_ITEMS items of it; the rest stay unchecked and are
   listed with their counts.
5. For a group, the saturated number is the one from 2 to 12 with the
   best balanced accuracy on the calibration half (ties: best accuracy,
   then the smallest number - the one that lets fewest sites be open).
6. Agreement is then reported on the other half, beside the same
   figure for the 16 metals checked before, with their own numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "thevoid_core" / "data" / "oms_calibration.json"
MIN_ITEMS = 10
CANDIDATES = range(2, 13)
LANTHANIDES_AND_Y = ("La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
                     "Tm", "Yb", "Lu", "Y")
ACTINIDES = ("Th", "U", "Np", "Pu")
GROUPS = {"lanthanides and Y": LANTHANIDES_AND_Y, "actinides": ACTINIDES}


def group_of(metal: str) -> str:
    for name, metals in GROUPS.items():
        if metal in metals:
            return name
    return metal


#: CoRE id -> the paper it came from (set by main from the index).
PAPERS: dict = {}


def paper(core_id: str) -> str:
    """The split key: the entry's DOI, so every structure of one paper
    (its ASR, FSR and ION versions, and its other frameworks) falls on
    one side; without a DOI, the id with its ASR/FSR/ION tag removed."""
    return PAPERS.get(core_id) or re.sub(r"\[(ASR|FSR|ION)\]", "", core_id,
                                         flags=re.IGNORECASE)


def half(core_id: str) -> int:
    """0: calibration half, 1: held-out half."""
    return int(hashlib.sha256(paper(core_id).encode()).hexdigest(), 16) % 2


def site_features(path) -> dict:
    """Every metal site's coordination number, asymmetry and widest
    empty cone, as detect_open_metal_sites computes them."""
    import logging
    import warnings

    from pymatgen.core import Structure

    from thevoid_core.data.metal_classification import is_metal
    from thevoid_core.services.oms_detection import detect_open_metal_sites
    from thevoid_core.services.pore_analysis import major_occupancy_structure
    from thevoid_core.services.site_labels import compute_site_labels

    logging.disable(logging.WARNING)
    warnings.simplefilter("ignore")
    record = {"id": Path(path).stem, "sites": []}
    try:
        structure = major_occupancy_structure(Structure.from_file(path))
        metals = {s.specie.symbol for s in structure.sites if is_metal(s.specie.symbol)}
        found = detect_open_metal_sites(structure, compute_site_labels(structure),
                                        supported_metals=metals)
        record["sites"] = [{"metal": s.metal_symbol, "cn": s.coordination_number,
                            "asym": float(s.asymmetry_score), "gap": float(s.angular_gap)}
                           for s in found.sites]
    except Exception as exc:  # recorded, not fatal
        record["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return record


def items_from(records, labels) -> list:
    """(core_id, metal, sites, open per CoRE) per metal per structure."""
    out = []
    for record in records:
        if "error" in record or record["id"] not in labels:
            continue
        by_metal = defaultdict(list)
        for site in record["sites"]:
            by_metal[site["metal"]].append(site)
        for metal, sites in sorted(by_metal.items()):
            out.append((record["id"], metal, sites, metal in labels[record["id"]]))
    return out


def predicted_open(sites, saturated_at) -> bool:
    from thevoid_core.services.oms_detection import OMSClassification, classify_site

    return any(classify_site(s["cn"], s["asym"], saturated_at, angular_gap=s["gap"])
               is OMSClassification.OPEN for s in sites)


def score(items, saturated_at) -> dict:
    """Agreement with CoRE; saturated_at is a number or {metal: number}."""
    tp = fn = tn = fp = 0
    for _, metal, sites, label in items:
        n = saturated_at[metal] if isinstance(saturated_at, dict) else saturated_at
        guess = predicted_open(sites, n)
        if label:
            tp, fn = tp + guess, fn + (not guess)
        else:
            tn, fp = tn + (not guess), fp + guess
    n = tp + fn + tn + fp
    return {"items": n, "agreement": round((tp + tn) / n, 4) if n else None,
            "balanced_accuracy": round(0.5 * ((tp / (tp + fn) if tp + fn else 1.0)
                                              + (tn / (tn + fp) if tn + fp else 1.0)), 4)
            if n else None,
            "open_agreed": f"{tp}/{tp + fn}", "saturated_agreed": f"{tn}/{tn + fp}"}


def calibrate(items) -> dict:
    from thevoid_core.data.metal_classification import CHECKED_BEFORE_ROUND_69
    from thevoid_core.services.oms_detection import (
        DEFAULT_ANGULAR_GAP_THRESHOLD,
        DEFAULT_ASYMMETRY_THRESHOLD,
        TYPICAL_MAX_COORDINATION,
    )

    by_group = defaultdict(list)
    for item in items:
        if item[1] not in CHECKED_BEFORE_ROUND_69:
            by_group[group_of(item[1])].append(item)
    groups, unchecked = {}, {}
    for name, group in sorted(by_group.items()):
        train = [i for i in group if half(i[0]) == 0]
        test = [i for i in group if half(i[0]) == 1]
        if len(train) < MIN_ITEMS or len(test) < MIN_ITEMS:
            unchecked[name] = {"items": len(group), "calibration_half": len(train),
                               "held_out_half": len(test)}
            continue
        curve = {n: score(train, n) for n in CANDIDATES}
        best = max(CANDIDATES, key=lambda n: (curve[n]["balanced_accuracy"],
                                              curve[n]["agreement"], -n))
        metals = sorted({i[1] for i in group})
        members = GROUPS.get(name, (name,))
        groups[name] = {
            "metals": list(members),
            "metals_seen": metals,
            "saturated_coordination": best,
            "calibration": curve[best],
            "held_out": score(test, best),
            "calibration_curve": {str(n): curve[n]["balanced_accuracy"] for n in CANDIDATES},
        }
    new_saturated = {m: g["saturated_coordination"] for g in groups.values() for m in g["metals"]}
    held_new = [i for i in items if i[1] in new_saturated and half(i[0]) == 1]
    held_old = [i for i in items if i[1] in CHECKED_BEFORE_ROUND_69 and half(i[0]) == 1]
    old_saturated = {m: TYPICAL_MAX_COORDINATION[m] for m in CHECKED_BEFORE_ROUND_69}
    per_old = {}
    for metal in sorted(CHECKED_BEFORE_ROUND_69):
        subset = [i for i in held_old if i[1] == metal]
        if subset:
            per_old[metal] = score(subset, old_saturated)
    return {
        "method": ("Saturated coordination number per metal group, chosen on half of "
                   "CoRE MOF 2024's SI structures (split by paper, DOI) to agree best "
                   "with CoRE's oms_metals (balanced accuracy; ties: accuracy, then the "
                   "smaller number). The geometric test is unchanged. Agreement is "
                   "reported on the other half. See tools/calibrate_oms.py."),
        "labels": "CoRE MOF 2024 oms_metals (CC-BY-4.0)",
        "thresholds": {"asymmetry": DEFAULT_ASYMMETRY_THRESHOLD,
                       "widest_empty_cone_deg": DEFAULT_ANGULAR_GAP_THRESHOLD},
        "min_items_per_half": MIN_ITEMS,
        "groups": groups,
        "held_out_new_metals": score(held_new, new_saturated),
        "held_out_metals_checked_before": score(held_old, old_saturated),
        "held_out_metals_checked_before_by_metal": per_old,
        "not_calibrated": unchecked,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--features", type=Path,
                        help="per-site features: read if it exists, else written")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)

    from thevoid_core.services.reference import coremof

    index = coremof.load_index().by_id
    if args.features and args.features.exists():
        records = [json.loads(line) for line in args.features.open()]
    else:
        paths = sorted(args.folder.glob("*.cif"))
        start = time.time()
        with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            records = list(pool.map(site_features, paths, chunksize=8))
        print(f"{len(records)} structures in {time.time() - start:.0f} s", file=sys.stderr)
        if args.features:
            args.features.write_text("".join(json.dumps(r) + "\n" for r in records))
    labels = {k: {m for m in (e.values.get("oms_metals") or "").split(",") if m}
              for k, e in index.items()}
    PAPERS.update({k: str(e.values["doi"]).strip().lower() for k, e in index.items()
                   if e.values.get("doi")})
    items = items_from(records, labels)
    result = calibrate(items)
    result["structures"] = len(records)
    result["structures_failed"] = sum("error" in r for r in records)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for name, g in result["groups"].items():
        h = g["held_out"]
        print(f"{name:18s} saturated at {g['saturated_coordination']:2d}  held-out "
              f"{h['items']:3d} items: agreement {h['agreement']:.3f}, balanced "
              f"{h['balanced_accuracy']:.3f} (open {h['open_agreed']}, saturated "
              f"{h['saturated_agreed']})")
    for key in ("held_out_new_metals", "held_out_metals_checked_before"):
        h = result[key]
        print(f"{key:32s} {h['items']:4d} items: agreement {h['agreement']:.3f}, balanced "
              f"{h['balanced_accuracy']:.3f}")
    print("not calibrated:", ", ".join(f"{k} ({v['items']})"
                                       for k, v in result["not_calibrated"].items()))


if __name__ == "__main__":
    main()
