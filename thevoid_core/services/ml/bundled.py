"""The models that ship with the Void (round 61), trained on CoRE MOF
2024 (tools/train_bundled_models.py rebuilds them).

What each predicts, and what its training values are:

water_kh
    The Henry coefficient of water at 298 K (mmol/g/Pa, modelled as
    log10): the first Widom value in CoRE MOF 2024. These are GCMC
    Widom-insertion simulations (UFF4MOF / TIP4P), so the labels are
    simulations, not predictions. That the value is water's is inferred
    from the data (see tools/build_coremof_index.py): the file does not
    say.
thermal_stability
    The decomposition temperature (°C) CoRE MOF 2024 lists. That value
    is itself a machine-learned prediction (Nandy et al.'s models, as
    CoRE MOF Tools applies them), so this model is a surrogate of a
    surrogate.
solvent_stability, water_stability
    The probability of staying intact on solvent removal, and in water,
    as CoRE lists them. Also model predictions, not measurements.

The trees are stored with joblib, tied to the scikit-learn version
that fitted them. With a different version the models are retrained
from the bundled data, deterministically, into the user cache
(net.cache_dir()/models), and the card says so.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

from thevoid_core.services.ml import descriptors as D

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
LINKERS = Path(__file__).resolve().parents[2] / "data" / "coremof2024_linkers.json.gz"

SPECS = {
    "water_kh": dict(label="Water Henry coefficient (298 K)", unit="mmol/g/Pa",
                     target="water_kh_mmol_g_pa", transform="log10",
                     caveat="Trained on CoRE MOF 2024's Widom-insertion values (UFF4MOF/TIP4P); "
                            "that the first Widom value is water's is inferred from the data."),
    "thermal_stability": dict(label="Thermal decomposition temperature", unit="°C",
                              target="stability_thermal_C", transform="none",
                              caveat="Trained on CoRE MOF 2024's thermal stabilities, which are "
                                     "themselves ML predictions, not measurements."),
    "solvent_stability": dict(label="Stable on solvent removal (probability)", unit="",
                              target="stability_solvent", transform="none", clip=(0.0, 1.0),
                              caveat="Trained on CoRE MOF 2024's solvent-removal stabilities, "
                                     "which are ML predictions, not measurements."),
    "water_stability": dict(label="Stable in water (probability)", unit="",
                            target="stability_water", transform="none", clip=(0.0, 1.0),
                            caveat="Trained on CoRE MOF 2024's water stabilities, which are ML "
                                   "predictions, not measurements."),
}


def group_key(entry) -> str:
    """Rows that are the same framework: CoRE's ASR and FSR copies of one
    structure share a refcode apart from the subset tag."""
    refcode = entry.values.get("refcode") or entry.id
    for tag in ("_ASR", "_FSR", "_Ion"):
        refcode = refcode.replace(tag, "")
    return refcode


def training_rows():
    """(entries, descriptor rows) for every CoRE MOF 2024 index entry."""
    from thevoid_core.services.reference import coremof

    with gzip.open(LINKERS, "rt", encoding="utf-8") as handle:
        linkers = json.load(handle)["linkers"]
    entries = coremof.search()
    rows = [D.coremof_row(e.values, {"fragments": linkers[e.id]} if e.id in linkers else None)
            for e in entries]
    return entries, rows


def train_bundled(name, entries=None, rows=None, seed=0, progress=None):
    from thevoid_core.services.ml.model import train
    from thevoid_core.services.reference.coremof import index_info

    spec = SPECS[name]
    if entries is None:
        entries, rows = training_rows()
    values = []
    for e in entries:
        v = e.values.get(spec["target"])
        v = float(v) if v not in (None, "") else math.nan
        if spec["transform"] == "log10":
            v = math.log10(v) if v > 0 else math.nan
        values.append(v)
    info = index_info()
    card = {"caveat": spec["caveat"],
            "training_data": {"database": info.get("name"), "citation": info.get("citation"),
                              "doi": info.get("doi"), "licence": info.get("licence"),
                              "index_sha256": info.get("sha256"),
                              "linker_table": LINKERS.name},
            "convention": "descriptors computed as CoRE MOF 2024: Zeo++'s CCDC radii, "
                          "1.655 Å probe (the Void: radii='ccdc')"}
    if "clip" in spec:
        card["clip"] = list(spec["clip"])
    return train(rows, values, [group_key(e) for e in entries], name=name, label=spec["label"],
                 unit=spec["unit"], target=spec["target"], descriptor_set="coremof-v1",
                 features=list(D.COREMOF_V1), nets=list(D.COREMOF_NETS),
                 transform=spec["transform"], seed=seed, card=card, progress=progress)


_LOADED = {}


def load_bundled(name):
    """The bundled model, retrained into the cache if this
    scikit-learn is not the one that fitted it."""
    import sklearn

    from thevoid_core.services.ml.model import Model

    if name not in SPECS:
        raise ValueError(f"Unknown model {name!r}; bundled: {sorted(SPECS)}")
    if name in _LOADED:
        return _LOADED[name]
    folder = MODELS_DIR / name
    card = json.loads((folder / "card.json").read_text(encoding="utf-8"))
    if card.get("sklearn_version") != sklearn.__version__:
        from thevoid_core.services.reference.net import cache_dir

        cached = cache_dir() / "models" / sklearn.__version__ / name
        if (cached / "card.json").is_file():
            model = Model.load(cached)
        else:
            model = train_bundled(name)
            model.card["retrained_because"] = (
                f"bundled trees were fitted with scikit-learn {card.get('sklearn_version')}, "
                f"this is {sklearn.__version__}")
            model.save(cached)
    else:
        model = Model.load(folder)
    _LOADED[name] = model
    return model


def bundled_names():
    return list(SPECS)
