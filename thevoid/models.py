"""Property models (round 61): predictions with an interval and an
applicability-domain flag, from models bundled with the Void (trained
on CoRE MOF 2024) or trained on your own batch.

    thevoid.models()                                   # bundled names -> cards
    fw.predict()                                       # every bundled model
    fw.predict(["water_kh"])["water_kh"].value

    model = thevoid.train("results", "ref_uptake_CO2_0.1_298K", log=True)
    model.save("co2_model")
    thevoid.load_model("co2_model").predict(fw)

Every prediction is a model's estimate, not a measurement or a
simulation: read `caveat`, and treat `in_domain=False` as "this
structure is unlike anything the model was trained on".
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from thevoid.results import _Result


@dataclass(frozen=True, repr=False)
class PredictionResult(_Result):
    """One model's prediction for one structure.

    value, with low-high its 90% interval (split-conformal on held-out
    structures, so about 90% of test structures fell inside theirs), in
    `unit`. in_domain: the structure's descriptors are about as close
    to the training structures as the training structures are to each
    other (mean distance to the 5 nearest, under the 95th percentile of
    the training set's); domain_distance is that mean distance. caveat
    says what the training values are."""

    model: str
    label: str
    unit: str
    value: float
    low: float
    high: float
    in_domain: bool
    domain_distance: float
    caveat: str
    details: dict = field(default_factory=dict, compare=False)
    raw: object = field(default=None, compare=False)


def _result(prediction) -> PredictionResult:
    d = prediction.as_dict()
    return PredictionResult(prediction.model, prediction.label, prediction.unit, prediction.value,
                            prediction.low, prediction.high, prediction.in_domain,
                            prediction.domain_distance, prediction.note, d, prediction)


class Predictions:
    """The predictions for one structure: index by model name, or
    iterate. descriptors: the inputs each descriptor set computed;
    notes: anything that could not be computed (it enters the model as
    missing)."""

    def __init__(self, raw: dict, name: str = "framework"):
        self.raw = raw
        self.name = name
        self.results = [_result(p) for p in raw["predictions"]]
        self.descriptors = {k: dict(v) for k, v in raw["descriptors"].items()}
        self.notes = dict(raw["notes"])

    def __getitem__(self, model: str) -> PredictionResult:
        for r in self.results:
            if r.model == model:
                return r
        raise KeyError(f"No prediction from {model!r}; have {[r.model for r in self.results]}")

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def to_dict(self) -> dict:
        """As the JSON report's "predictions" section."""
        from thevoid_core.services.ml.predict import predictions_to_dict

        return predictions_to_dict(self.raw)

    def rows(self) -> list:
        return [{"structure": self.name, **{k: getattr(r, k) for k in (
            "model", "label", "unit", "value", "low", "high", "in_domain", "domain_distance")}}
            for r in self.results]

    def to_dataframe(self):
        """One row per model (needs pandas)."""
        from thevoid._tables import dataframe

        return dataframe(self.rows())

    def __repr__(self):
        shown = ", ".join(f"{r.model}={r.value:.3g}" + ("" if r.in_domain else " (out of domain)")
                          for r in self.results)
        return f"Predictions({self.name!r}: {shown})"


class Model:
    """A trained property model: card (what it predicts, the training
    data, the recipe, held-out test metrics) and the fitted trees."""

    def __init__(self, raw):
        self.raw = raw

    @property
    def name(self) -> str:
        return self.raw.name

    @property
    def label(self) -> str:
        return self.raw.label

    @property
    def unit(self) -> str:
        return self.raw.unit

    @property
    def descriptor_set(self) -> str:
        """"coremof-v1" (CoRE MOF 2024's descriptors, computed its way)
        or "void-v1" (a the-void-analyze summary row)."""
        return self.raw.descriptor_set

    @property
    def card(self) -> dict:
        return dict(self.raw.card)

    @property
    def test_metrics(self) -> dict:
        """MAE (log10 units for a log model), always-the-mean MAE, R²,
        Spearman, interval coverage and in-domain fraction on the
        held-out structures."""
        return dict(self.raw.card.get("test_metrics", {}))

    def predict(self, framework, *, samples_per_atom: int = 1000) -> PredictionResult:
        """This model's prediction for a Framework."""
        return framework.predict([self], samples_per_atom=samples_per_atom)[self.name]

    def save(self, folder) -> Path:
        """card.json and model.joblib, for load_model() and
        the-void-analyze --predict FOLDER."""
        return self.raw.save(folder)

    def __repr__(self):
        m = self.test_metrics
        return (f"Model({self.name!r}, {self.descriptor_set}, test R2={m.get('r2', float('nan')):.3f}, "
                f"MAE={m.get('mae', float('nan')):.3g})")


def models() -> dict:
    """The bundled models: name -> card (read from disk; does not need
    scikit-learn)."""
    from thevoid_core.services.ml import bundled

    return {name: json.loads((bundled.MODELS_DIR / name / "card.json").read_text(encoding="utf-8"))
            for name in bundled.bundled_names()}


def load_model(source) -> Model:
    """A bundled model by name, or a folder the-void-train (or
    Model.save) wrote. Only load folders you trust: the trees are
    stored with joblib (pickle)."""
    from thevoid_core.services.ml.predict import resolve

    return Model(resolve([str(source) if isinstance(source, Path) else source])[0])


def train(results, target: str, *, log: bool = False, name: str | None = None,
          label: str | None = None, unit: str = "", seed: int = 0,
          method_key: str | None = None) -> Model:
    """Train a void-v1 model on a batch: `results` is a the-void-analyze
    results folder, a BatchTable written to one, or (round 67) a shared
    index (a thevoid.Index or its folder), and `target` a column of its
    summary (e.g. one joined with reference_csv). log: model log10 of
    the target (for quantities spanning decades). From an index, only
    current entries of one method are used: method_key, or by default
    the one most entries with the target share."""
    from thevoid_core.services.ml.batch_training import train_from_batch

    from thevoid.batch import BatchTable
    from thevoid.index import Index

    folder = results
    if isinstance(results, Index):
        folder = results.path
    elif isinstance(results, BatchTable):
        if results.folder is None:
            raise ValueError("Write the BatchTable to a folder first (table.write(folder)): "
                             "the model card records the summary it was trained on")
        folder = results.folder
    return Model(train_from_batch(folder, target, log=log, name=name, label=label, unit=unit,
                                  seed=seed, method_key=method_key))


def mofxdb_labels(folder, adsorbate: str, pressure: float, *, temperature: float = 298.0,
                  out=None) -> tuple[list, dict]:
    """(rows, info): uptakes at `pressure` (the isotherms' own units;
    hMOF: bar) from a fetch_mofxdb folder, interpolated, never
    extrapolated. out: also write them as a CSV for
    analyze_batch(reference_csv=...)."""
    from thevoid_core.services.ml.batch_training import labels_from_mofxdb, write_labels

    rows, info = labels_from_mofxdb(folder, adsorbate, pressure, temperature=temperature)
    if out is not None:
        write_labels(rows, out, info["column"])
    return rows, info
