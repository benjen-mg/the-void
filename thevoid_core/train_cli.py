"""`the-void-train`: property models (round 61; services/ml).

    the-void-train list                                   # the bundled models and their test metrics
    the-void-train bundled --out models                   # rebuild the bundled models (CoRE MOF 2024)
    the-void-train mofxdb-labels hmof --adsorbate CO2 --pressure 0.1 --out labels.csv
    the-void-analyze hmof/*.cif --out results --reference-csv labels.csv
    the-void-train fit results --target ref_uptake_CO2_0.1_298K --out co2_model
    the-void-analyze new/*.cif --out screen --predict co2_model,water_stability

A model folder holds card.json (what it predicts, from what data, the
recipe, and its held-out test metrics) and model.joblib.

Exit status: 0 on success, 1 if a model could not be trained, 2 for
bad arguments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from thevoid_core.version import __version__


def build_parser():
    parser = argparse.ArgumentParser(prog="the-void-train",
                                     description=f"Property models (the Void {__version__}).")
    parser.add_argument("--version", action="version", version=f"the Void {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="the bundled models and their test metrics")
    bundled = commands.add_parser("bundled", help="retrain the bundled CoRE MOF 2024 models")
    bundled.add_argument("--out", required=True, help="folder to write the model folders into")
    bundled.add_argument("--seed", type=int, default=0)
    labels = commands.add_parser("mofxdb-labels",
                                 help="uptakes from a the-void-fetch mofxdb folder, as a CSV")
    labels.add_argument("folder")
    labels.add_argument("--adsorbate", required=True, help="CO2, CH4, N2, H2, Xe, Kr, ...")
    labels.add_argument("--pressure", type=float, required=True,
                        help="in the isotherms' own units (hMOF: bar)")
    labels.add_argument("--temperature", type=float, default=298.0)
    labels.add_argument("--out", required=True)
    fit = commands.add_parser("fit", help="train on a the-void-analyze results folder")
    fit.add_argument("results", help="a the-void-analyze results folder, or a shared index "
                                     "(the-void-index)")
    fit.add_argument("--method-key", default=None,
                     help="from an index: train only on entries computed this way (default: "
                          "the method most entries with the target share)")
    fit.add_argument("--target", required=True, help="a summary column, e.g. ref_uptake_CO2_0.1_298K")
    fit.add_argument("--log", action="store_true", help="model log10 of the target")
    fit.add_argument("--name", default=None)
    fit.add_argument("--label", default=None)
    fit.add_argument("--unit", default="")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--out", required=True, help="folder for card.json and model.joblib")
    return parser


def _metrics_line(card, transform=None) -> str:
    """card: a saved card.json, or a trained Model's card with its
    transform passed (a card gets "transform" only when saved)."""
    m = card.get("test_metrics", {})
    card = {**card, "transform": transform or card.get("transform")}
    scale = " (log10)" if card.get("transform") == "log10" else ""
    return (f"test n={m.get('n')}: MAE {m.get('mae'):.3g}{scale} (always-the-mean "
            f"{m.get('baseline_mae'):.3g}), R2 {m.get('r2'):.3f}, Spearman {m.get('spearman'):.3f}, "
            f"90% interval covers {100 * m.get('interval_coverage', 0):.0f}%")


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from thevoid_core.services.ml.model import _require_sklearn

        _require_sklearn()
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.command == "list":
        import json

        from thevoid_core.services.ml import bundled

        for name in bundled.bundled_names():
            card = json.loads((bundled.MODELS_DIR / name / "card.json").read_text(encoding="utf-8"))
            print(f"{name}: {card['label']}" + (f" [{card['unit']}]" if card.get("unit") else ""))
            print(f"  {_metrics_line(card)}")
            print(f"  {card.get('caveat', '')}")
        return 0

    if args.command == "bundled":
        from thevoid_core.services.ml import bundled

        entries, rows = bundled.training_rows()
        for name in bundled.bundled_names():
            model = bundled.train_bundled(name, entries, rows, seed=args.seed)
            model.save(Path(args.out) / name)
            print(f"{name}: {_metrics_line(model.card, model.transform)}")
        return 0

    if args.command == "mofxdb-labels":
        from thevoid_core.services.ml.batch_training import labels_from_mofxdb, write_labels

        if not Path(args.folder).is_dir():
            parser.error(f"{args.folder} is not a folder")
        rows, info = labels_from_mofxdb(args.folder, args.adsorbate, args.pressure,
                                        temperature=args.temperature)
        write_labels(rows, args.out, info["column"])
        print(f"{len(rows)} labels ({info['column']}; units: {', '.join(info['units']) or '-'}) "
              f"-> {args.out}; {len(info['skipped'])} structures without a matching isotherm")
        return 0 if rows else 1

    from thevoid_core.services.ml.batch_training import train_from_batch

    try:
        model = train_from_batch(args.results, args.target, log=args.log, name=args.name,
                                 label=args.label, unit=args.unit, seed=args.seed,
                                 method_key=args.method_key)
    except ValueError as exc:
        print(f"cannot train: {exc}", file=sys.stderr)
        return 1
    model.save(args.out)
    print(f"{model.name}: {_metrics_line(model.card, model.transform)}")
    top = ", ".join(f for f, _ in model.card.get("most_important_features", [])[:5])
    print(f"  most important: {top}")
    stale = model.card.get("training_data", {}).get("stale_entries")
    if stale:
        print(f"  note: {stale} of the entries it was trained on are stale (a later version "
              "computes some of their numbers differently); the-void-index refresh, then "
              "train again")
    print(f"  saved to {args.out}; use it with the-void-analyze --predict {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
