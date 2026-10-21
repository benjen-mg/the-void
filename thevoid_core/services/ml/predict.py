"""Predictions for a structure or a batch (round 61)."""

from __future__ import annotations

from pathlib import Path


def resolve(models=None) -> list:
    """Model objects from names of bundled models, folders of saved
    ones, or Model objects; None: every bundled model."""
    from thevoid_core.services.ml import bundled
    from thevoid_core.services.ml.model import Model, _require_sklearn

    _require_sklearn()
    if models is None:
        models = bundled.bundled_names()
    if isinstance(models, (str, Path)) or isinstance(models, Model):
        models = [models]
    out = []
    for m in models:
        if isinstance(m, Model):
            out.append(m)
        elif isinstance(m, str) and m in bundled.SPECS:
            out.append(bundled.load_bundled(m))
        elif Path(m).is_dir() and (Path(m) / "card.json").is_file():
            out.append(Model.load(m))
        else:
            raise ValueError(f"No model {m!r}: bundled ones are {bundled.bundled_names()}, "
                             "or give a folder the-void-train wrote")
    return out


def predict_structure(structure, models=None, *, summary_row=None, samples_per_atom=1000,
                      cached=None) -> dict:
    """{"predictions": [Prediction...], "descriptors": {set: row},
    "notes": {...}} for one structure.

    coremof-v1 models get descriptors computed as CoRE MOF 2024 computes
    its own; void-v1 models need the structure's the-void-analyze
    summary row (summary_row), computed here if not given. cached:
    CoRE-convention results already computed (pores, surface, oms,
    topology), passed to coremof_descriptors."""
    from thevoid_core.services.ml import descriptors as D

    chosen = resolve(models)
    rows, notes = {}, {}
    if any(m.descriptor_set == "coremof-v1" for m in chosen):
        rows["coremof-v1"], notes = D.coremof_descriptors(
            structure, samples_per_atom=samples_per_atom, **(cached or {}))
    predictions = []
    for model in chosen:
        if model.descriptor_set == "coremof-v1":
            row = rows["coremof-v1"]
        elif model.descriptor_set == "void-v1":
            if summary_row is None:
                from thevoid_core.services.report import AnalysisConfig, run_analysis, summary_row as sr

                summary_row = sr(run_analysis(structure, AnalysisConfig(**model.card.get(
                    "analysis_options", {}))))
            row = D.void_row(summary_row, model.nets)
            rows["void-v1"] = row
        else:
            raise ValueError(f"Model {model.name} uses unknown descriptors {model.descriptor_set!r}")
        predictions.extend(model.predict_rows([row]))
    return {"predictions": predictions, "descriptors": rows, "notes": notes}


def predictions_to_dict(result: dict) -> dict:
    """The report's form: {model: {...}} plus the descriptors used."""
    from thevoid_core.services.report import _clean

    return _clean({
        "models": {p.model: p.as_dict() for p in result["predictions"]},
        "descriptors": result["descriptors"],
        "notes": result["notes"],
    })
