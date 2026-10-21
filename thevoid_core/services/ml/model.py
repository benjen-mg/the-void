"""Property models: an ensemble of gradient-boosted trees on
descriptors, with a calibrated interval and an applicability domain
(round 61).

The training recipe, every choice recorded in the model card:

* Split by structure, not by row. CoRE lists the same framework in its
  ASR and FSR subsets; rows that share a structure (group key) are
  never split across training and test. The split is 70% train, 10%
  calibration, 20% test, seeded.
* Ensemble: ENSEMBLE_SIZE HistGradientBoostingRegressors, each fitted on
  a different 80% subsample of the training rows with its own seed.
  The prediction is their mean; their spread, sigma, measures how much
  the data pin the answer down.
* Interval: sigma alone is not a calibrated error bar. On the
  calibration rows, the 90th percentile of |error| / (sigma + s0) gives
  a factor k (split conformal, normalised by the ensemble spread; s0 is
  the median sigma, so a near-zero spread cannot make the interval
  vanish). The 90% interval is prediction ± k (sigma + s0). Its coverage
  is measured again on the test rows and recorded.
* Applicability domain: descriptors are standardised by the training
  rows. A structure's mean distance to its DOMAIN_K nearest training
  rows is compared with the 95th percentile of that distance among the
  training rows themselves (each left out of its own neighbours). Above
  it, the structure is "outside the domain": its prediction is an
  extrapolation, and the card reports how much worse such predictions
  are on the test rows.
* Test metrics: MAE, RMSE, R², Spearman ρ and interval coverage, with
  the MAE of always predicting the training mean alongside.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ENSEMBLE_SIZE = 5
SUBSAMPLE = 0.8
COVERAGE = 0.90
DOMAIN_K = 5
DOMAIN_PERCENTILE = 95
#: Training rows kept for the domain check (a seeded sample of this many;
#: the threshold is measured on the same sample, so the two agree).
DOMAIN_ROWS = 4000
#: 200 trees at 0.1 fit as well as 400 at 0.05 on the CoRE targets
#: (water KH MAE 1.079 vs 1.077 log units) at half the size.
HYPERPARAMETERS = {"max_iter": 200, "learning_rate": 0.1, "max_leaf_nodes": 31,
                   "min_samples_leaf": 20, "l2_regularization": 0.0}


def _require_sklearn():
    import importlib.util

    if importlib.util.find_spec("sklearn") is None:
        raise ImportError(
            "The property models need scikit-learn, which the Void does not require: "
            "pip install scikit-learn (or pip install -e \".[ml]\")") from None


@dataclass
class Prediction:
    model: str
    label: str
    unit: str
    value: float
    low: float
    high: float
    sigma: float
    in_domain: bool
    domain_distance: float
    note: str = ""

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Model:
    name: str
    label: str
    unit: str
    target: str
    #: "log10" when the model works on log10 of the target.
    transform: str
    descriptor_set: str
    features: list
    #: The coremof nets or the batch's nets, for void-v1.
    nets: list = field(default_factory=list)
    estimators: list = field(default_factory=list)
    k: float = 1.0
    s0: float = 0.0
    domain_mean: list = field(default_factory=list)
    domain_scale: list = field(default_factory=list)
    domain_rows: object = None
    domain_threshold: float = math.inf
    card: dict = field(default_factory=dict)
    #: The k-d tree of domain_rows, built on first use (not saved).
    _tree: object = field(default=None, repr=False, compare=False)

    # -- predicting ---------------------------------------------------

    def _matrix(self, rows):
        return np.array([[row.get(f, np.nan) for f in self.features] for row in rows], dtype=float)

    def _domain(self, X):
        from scipy.spatial import cKDTree

        Z = (X - np.array(self.domain_mean)) / np.array(self.domain_scale)
        Z = np.where(np.isnan(Z), 0.0, Z)  # a missing value sits at the training mean
        if self._tree is None:
            self._tree = cKDTree(self.domain_rows)
        distances, _ = self._tree.query(Z, k=DOMAIN_K)
        return distances.mean(axis=1)

    def predict_rows(self, rows) -> list:
        from threadpoolctl import threadpool_limits  # a scikit-learn dependency

        X = self._matrix(rows)
        # One OpenMP thread: for a few rows threads only cost, and with
        # the cores busy (a parallel batch) they cost 10x (round 61).
        with threadpool_limits(1, user_api="openmp"):
            members = np.array([e.predict(X) for e in self.estimators])
        mean, sigma = members.mean(axis=0), members.std(axis=0)
        half = self.k * (sigma + self.s0)
        distance = self._domain(X)
        out = []
        for i in range(len(rows)):
            value, low, high = mean[i], mean[i] - half[i], mean[i] + half[i]
            if self.transform == "log10":
                value, low, high = 10 ** value, 10 ** low, 10 ** high
            elif self.card.get("clip"):
                lo, hi = self.card["clip"]
                value, low, high = (float(np.clip(v, lo, hi)) for v in (value, low, high))
            out.append(Prediction(self.name, self.label, self.unit, float(value), float(low),
                                  float(high), float(sigma[i]),
                                  bool(distance[i] <= self.domain_threshold),
                                  float(distance[i]), self.card.get("caveat", "")))
        return out

    # -- saving ---------------------------------------------------------

    def save(self, folder) -> Path:
        """card.json (readable) and model.joblib (the fitted trees and
        domain rows)."""
        import joblib
        import sklearn

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        card = dict(self.card)
        card.update({"name": self.name, "label": self.label, "unit": self.unit,
                     "target": self.target, "transform": self.transform,
                     "descriptor_set": self.descriptor_set, "features": list(self.features),
                     "nets": list(self.nets), "k": self.k, "s0": self.s0,
                     "domain_threshold": self.domain_threshold,
                     "sklearn_version": sklearn.__version__})
        (folder / "card.json").write_text(json.dumps(card, indent=1, allow_nan=False),
                                          encoding="utf-8")
        joblib.dump({"estimators": self.estimators, "domain_mean": self.domain_mean,
                     "domain_scale": self.domain_scale,
                     "domain_rows": np.asarray(self.domain_rows, dtype=np.float32)},
                    folder / "model.joblib", compress=3)
        return folder

    @classmethod
    def load(cls, folder):
        """A saved model. Only load models you trust: the trees are
        stored with joblib (pickle)."""
        import joblib

        _require_sklearn()
        folder = Path(folder)
        card = json.loads((folder / "card.json").read_text(encoding="utf-8"))
        state = joblib.load(folder / "model.joblib")
        return cls(card["name"], card["label"], card["unit"], card["target"], card["transform"],
                   card["descriptor_set"], card["features"], card.get("nets", []),
                   state["estimators"], card["k"], card["s0"], state["domain_mean"],
                   state["domain_scale"], np.asarray(state["domain_rows"], dtype=float),
                   card["domain_threshold"], card)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------


def group_split(groups, seed=0, fractions=(0.7, 0.1, 0.2)):
    """(train, calibration, test) row indices, whole groups at a time."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups)))
    rng.shuffle(unique)
    cut1 = int(round(fractions[0] * len(unique)))
    cut2 = int(round((fractions[0] + fractions[1]) * len(unique)))
    which = {g: 0 for g in unique[:cut1]}
    which.update({g: 1 for g in unique[cut1:cut2]})
    which.update({g: 2 for g in unique[cut2:]})
    codes = np.array([which[g] for g in groups])
    return tuple(np.nonzero(codes == c)[0] for c in (0, 1, 2))


def _metrics(y, p, low=None, high=None):
    from scipy.stats import spearmanr

    residual = p - y
    out = {"n": int(len(y)), "mae": float(np.mean(np.abs(residual))),
           "rmse": float(np.sqrt(np.mean(residual ** 2))),
           "r2": float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else None,
           "spearman": float(spearmanr(y, p)[0]) if len(y) > 2 else None}
    if low is not None:
        out["interval_coverage"] = float(np.mean((y >= low) & (y <= high)))
    return out


def train(rows, y, groups, *, name, label, unit, target, descriptor_set, features, nets=(),
          transform="none", seed=0, card=None, progress=None) -> Model:
    """Fit a Model on descriptor rows (dicts) and targets y (already
    log10-transformed when transform="log10"), with the recipe above."""
    _require_sklearn()
    from scipy.spatial import cKDTree
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = np.array([[row.get(f, np.nan) for f in features] for row in rows], dtype=float)
    y = np.asarray(y, dtype=float)
    keep = ~np.isnan(y)
    X, y, groups = X[keep], y[keep], [g for g, k in zip(groups, keep) if k]
    if len(y) < 30:
        raise ValueError(f"Only {len(y)} rows with a {target} value: too few to train on")
    # Round 67: a feature with no value in any row (open metal sites in a
    # batch run with --no-oms) made scikit-learn's binning fail ("window
    # shape cannot be larger than input array shape"). It carries no
    # information, so it is held at 0 in training; at prediction it is
    # whatever the structure has, and no tree splits on it.
    empty = [f for j, f in enumerate(features) if np.all(np.isnan(X[:, j]))]
    for f in empty:
        X[:, list(features).index(f)] = 0.0
    train_i, cal_i, test_i = group_split(groups, seed)
    rng = np.random.default_rng(seed)
    estimators = []
    for member in range(ENSEMBLE_SIZE):
        if progress is not None:
            progress(f"{name}: tree ensemble member {member + 1}", member, ENSEMBLE_SIZE)
        chosen = rng.choice(train_i, size=int(SUBSAMPLE * len(train_i)), replace=False)
        estimator = HistGradientBoostingRegressor(random_state=seed + member, **HYPERPARAMETERS)
        estimators.append(estimator.fit(X[chosen], y[chosen]))

    def ensemble(indices):
        members = np.array([e.predict(X[indices]) for e in estimators])
        return members.mean(axis=0), members.std(axis=0)

    cal_mean, cal_sigma = ensemble(cal_i)
    s0 = float(np.median(cal_sigma))
    scores = np.abs(cal_mean - y[cal_i]) / (cal_sigma + s0)
    level = min(1.0, math.ceil((len(cal_i) + 1) * COVERAGE) / len(cal_i))
    k = float(np.quantile(scores, level))

    mean = np.nanmean(X[train_i], axis=0)
    scale = np.nanstd(X[train_i], axis=0)
    mean = np.where(np.isnan(mean), 0.0, mean)
    scale = np.where((scale > 0) & ~np.isnan(scale), scale, 1.0)
    Z = np.where(np.isnan((X[train_i] - mean) / scale), 0.0, (X[train_i] - mean) / scale)
    if len(Z) > DOMAIN_ROWS:
        Z = Z[rng.choice(len(Z), DOMAIN_ROWS, replace=False)]
    Z = Z.astype(np.float32).astype(float)  # stored as float32; threshold measured on the same
    distances, _ = cKDTree(Z).query(Z, k=DOMAIN_K + 1)
    threshold = float(np.percentile(distances[:, 1:].mean(axis=1), DOMAIN_PERCENTILE))

    model = Model(name, label, unit, target, transform, descriptor_set, list(features),
                  list(nets), estimators, k, s0, mean.tolist(), scale.tolist(), Z, threshold,
                  dict(card or {}))
    test_mean, test_sigma = ensemble(test_i)
    half = k * (test_sigma + s0)
    test_distance = model._domain(X[test_i])
    inside = test_distance <= threshold
    metrics = _metrics(y[test_i], test_mean, test_mean - half, test_mean + half)
    metrics["baseline_mae"] = float(np.mean(np.abs(y[test_i] - y[train_i].mean())))
    metrics["inside_domain_fraction"] = float(inside.mean())
    if inside.any():
        metrics["mae_inside_domain"] = float(np.mean(np.abs(test_mean - y[test_i])[inside]))
    if (~inside).any():
        metrics["mae_outside_domain"] = float(np.mean(np.abs(test_mean - y[test_i])[~inside]))
    importances = _importance(estimators, X[test_i], y[test_i], features, seed)
    model.card.update({
        "trained": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "rows": {"train": int(len(train_i)), "calibration": int(len(cal_i)),
                 "test": int(len(test_i)), "groups": int(len(set(groups)))},
        "recipe": {"ensemble_size": ENSEMBLE_SIZE, "subsample": SUBSAMPLE,
                   "hyperparameters": HYPERPARAMETERS, "split": "70/10/20 by group",
                   "seed": seed, "interval": f"{int(COVERAGE * 100)}% split conformal, "
                   "normalised by the ensemble spread",
                   "domain": f"mean distance to {DOMAIN_K} nearest training rows (standardised) "
                             f"above their {DOMAIN_PERCENTILE}th percentile"},
        "test_metrics": metrics,
        "features_without_values": empty,
        "metrics_scale": "log10 of the target" if transform == "log10" else "the target",
        "most_important_features": importances[:8],
    })
    return model


def _importance(estimators, X, y, features, seed):
    """Permutation importance on the test rows (increase in MAE), top
    features first."""
    rng = np.random.default_rng(seed)

    def mae(M):
        return float(np.mean(np.abs(np.mean([e.predict(M) for e in estimators], axis=0) - y)))

    base = mae(X)
    scores = []
    for j, feature in enumerate(features):
        column = X[:, j].copy()
        if np.all(np.isnan(column)) or np.nanstd(column) == 0:
            continue
        M = X.copy()
        M[:, j] = rng.permutation(column)
        scores.append((feature, round(mae(M) - base, 6)))
    return sorted(scores, key=lambda s: -s[1])
