"""Rebuild thevoid_core/data/models/<name>/ (card.json, model.joblib),
the models bundled with the Void (round 61).

    python tools/train_bundled_models.py [name ...]

Deterministic for a given scikit-learn version: the same data, seed and
recipe (services/ml/model.py) give the same trees.
"""

import sys
import time

from thevoid_core.services.ml import bundled


def main(argv):
    names = argv[1:] or bundled.bundled_names()
    entries, rows = bundled.training_rows()
    for name in names:
        start = time.time()
        model = bundled.train_bundled(name, entries, rows)
        model.card["training_seconds"] = round(time.time() - start, 1)
        model.save(bundled.MODELS_DIR / name)
        m = model.card["test_metrics"]
        print(f"{name:18s} test n={m['n']} MAE {m['mae']:.3f} (baseline {m['baseline_mae']:.3f}) "
              f"R2 {m['r2']:.3f} rho {m['spearman']:.3f} coverage {m['interval_coverage']:.3f} "
              f"in-domain {m['inside_domain_fraction']:.2f} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main(sys.argv)
