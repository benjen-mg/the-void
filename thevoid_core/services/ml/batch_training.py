"""Training on your own data (round 61): a the-void-analyze batch plus a
column of labels, e.g. GCMC uptakes from MOFX-DB (labels_from_mofxdb)
or your own simulations, joined with --reference-csv.

The descriptors are "void-v1": the batch summary's own columns (LCD,
PLD, void fraction, channel ASA, density, channel dimensionality, open
metal sites, net) and composition features parsed from its formula.
Prediction for a new structure runs the same analyses and reads the
same columns, so a model sees at prediction exactly the kind of
numbers it was trained on. The analysis options of the batch are
recorded in the model card and reused.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

#: MOFX-DB's adsorbate names, by the formula a person types.
MOFXDB_ADSORBATES = {"CO2": "CarbonDioxide", "CH4": "Methane", "N2": "Nitrogen",
                     "H2": "Hydrogen", "Xe": "Xenon", "Kr": "Krypton", "Ar": "Argon",
                     "H2O": "Water"}


def uptake_at(points, pressure):
    """Linear interpolation of [(pressure, amount)] at `pressure`; None
    outside the measured range (never extrapolated)."""
    points = sorted((float(p), float(a)) for p, a in points if p is not None and a is not None)
    if not points or pressure < points[0][0] or pressure > points[-1][0]:
        return None
    for (p0, a0), (p1, a1) in zip(points, points[1:]):
        if p0 <= pressure <= p1:
            return a0 if p1 == p0 else a0 + (a1 - a0) * (pressure - p0) / (p1 - p0)
    return points[-1][1] if pressure == points[-1][0] else None


def labels_from_mofxdb(folder, adsorbate: str, pressure: float, *, temperature: float = 298.0,
                       tolerance_K: float = 1.0) -> tuple[list, dict]:
    """(rows, info) of single-component uptakes read from the
    <name>.isotherms.json files the-void-fetch mofxdb writes: one row
    {"file", column: value} per structure with a matching isotherm
    (that adsorbate alone, temperature within tolerance_K, pressure
    within the measured range). pressure is in the isotherm's own units
    (hMOF: bar); every unit found is reported in info."""
    name = MOFXDB_ADSORBATES.get(adsorbate, adsorbate)
    column = f"ref_uptake_{adsorbate}_{pressure:g}_{temperature:g}K"
    rows, units, skipped = [], set(), {}
    for path in sorted(Path(folder).glob("*.isotherms.json")):
        cif = path.name[: -len(".isotherms.json")] + ".cif"
        found = None
        for isotherm in json.loads(path.read_text(encoding="utf-8")):
            adsorbates = [a for a in isotherm.get("adsorbates") or [] if a]
            t = isotherm.get("temperature_K")
            if adsorbates != [name] or t is None or abs(float(t) - temperature) > tolerance_K:
                continue
            value = uptake_at(isotherm.get("points") or [], pressure)
            if value is None:
                skipped[cif] = "pressure outside the isotherm's range"
                continue
            found = value
            units.add(f"{isotherm.get('adsorption_units')} at {isotherm.get('pressure_units')}")
            break
        if found is None:
            skipped.setdefault(cif, f"no {name} isotherm at {temperature:g} K")
        else:
            rows.append({"file": cif, column: found})
    return rows, {"column": column, "adsorbate": name, "pressure": pressure,
                  "temperature_K": temperature, "units": sorted(units), "skipped": skipped}


def write_labels(rows, path, column) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", column])
        writer.writeheader()
        writer.writerows(rows)
    return path


def batch_nets(rows, keep: int = 15) -> list:
    from collections import Counter

    from thevoid_core.services.ml.descriptors import net_code

    counts = Counter(net_code(r.get("topology")) for r in rows)
    return [n for n, _ in counts.most_common() if n not in ("", "other", "unstable")][:keep]


def train_from_batch(results, target: str, *, log: bool = False, name: str | None = None,
                     label: str | None = None, unit: str = "", seed: int = 0, progress=None,
                     method_key: str | None = None):
    """A void-v1 Model predicting `target` (a column of the batch
    summary, e.g. ref_uptake_CO2_0.1_298K) from a the-void-analyze
    results folder, or (round 67) from a shared index.

    From an index, the model trains on its build: only current entries
    (not superseded by a newer analysis of the same file and options),
    and only entries computed one way - `method_key`, or by default the
    method most entries with the target share - because a void-v1 model
    must see at prediction the same kind of numbers it was trained on.
    Entries of one structure (its structure_id) are kept on one side of
    the train/test split, so replicates cannot inflate the test score."""
    from collections import Counter

    from thevoid_core.services.batch_results import load_batch_results
    from thevoid_core.services.ml import descriptors as D
    from thevoid_core.services.ml.model import train

    index_info = None
    if not hasattr(results, "rows") and (Path(results) / "INDEX.json").is_file():
        from thevoid_core.services.index.store import Index

        index = Index.open(results)
        built = index.build()
        if built.problems:
            raise ValueError("the index has problems: " + "; ".join(built.problems))
        index_info = {"index": index.info.get("name"), "path": str(index.path)}
        results = built.folder
    batch = load_batch_results(results) if not hasattr(results, "rows") else results
    if target not in batch.columns:
        raise ValueError(f"No column {target!r} in {batch.summary_path}; columns: "
                         f"{', '.join(c for c in batch.columns if c.startswith(('ref_', 'fn_')))}")
    kept = [r for r in batch.rows if not r.get("errors") and not r.get("superseded_by")]
    if method_key is not None or index_info is not None:
        if "method_key" not in batch.columns:
            raise ValueError("method_key: this summary has no method_key column (it is not "
                             "an index build)")
        if method_key is None:
            counts = Counter(r.get("method_key") for r in kept if r.get(target) not in (None, ""))
            if not counts:
                raise ValueError(f"no entry has a value of {target!r}")
            method_key = counts.most_common(1)[0][0]
        kept = [r for r in kept if r.get("method_key") == method_key]
        if index_info is not None:
            index_info.update(method_key=method_key, entries=len(kept),
                              stale_entries=sum(1 for r in kept if r.get("stale_columns")))
    rows = [dict(r.values) for r in kept]
    nets = batch_nets(rows)
    values = []
    for r in rows:
        v = r.get(target)
        v = float(v) if v not in (None, "") else math.nan
        if log:
            v = math.log10(v) if v > 0 else math.nan
        values.append(v)
    options = {}
    for r in kept:
        if r.report_json is not None:
            options = json.loads(r.report_json.read_text(encoding="utf-8")).get("config", {})
            break
    options = {k: v for k, v in options.items() if k in (
        "pore_probe_radius", "surface_probe", "surface_probe_radius", "surface_samples_per_atom",
        "grid_spacing", "rcsr_archive")}
    summary = Path(batch.summary_path)
    card = {"caveat": "", "analysis_options": options,
            "training_data": {"summary": str(summary),
                              "sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
                              "target_column": target},
            "convention": "the Void's own analyses (the-void-analyze), with the options above"}
    if index_info is not None:
        card["training_data"].update(index_info)
    groups = [r.get("structure_id") or Path(r.get("file") or str(i)).stem
              for i, r in enumerate(rows)]
    rows_d = [D.void_row(r, nets) for r in rows]
    return train(rows_d, values, groups,
                 name=name or target, label=label or target, unit=unit, target=target,
                 descriptor_set="void-v1", features=list(D.void_feature_names(nets)), nets=nets,
                 transform="log10" if log else "none", seed=seed, card=card, progress=progress)
