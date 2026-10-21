"""Round 61 (roadmap 4a): property models - descriptors, the training
recipe, the bundled CoRE MOF 2024 models, and training on a batch.

The checks follow what would make a prediction wrong without anyone
noticing: descriptors computed for a structure that are not the ones
the model was trained on; an interval that does not hold what it says;
a structure unlike the training set not being flagged; models silently
unusable after a scikit-learn upgrade; uptake labels extrapolated past
an isotherm's end. Fixtures are CoRE MOF 2024 SI structures and the
recorded MOFX-DB pages (tests/fixtures).
"""

import csv
import gzip
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("sklearn")

from pymatgen.core import Structure  # noqa: E402

from thevoid_core.services.ml import bundled  # noqa: E402
from thevoid_core.services.ml import descriptors as D  # noqa: E402
from thevoid_core.services.ml.batch_training import (  # noqa: E402
    labels_from_mofxdb,
    train_from_batch,
    uptake_at,
)
from thevoid_core.services.ml.model import Model, group_split, train  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"  # MIL-53(Cr)/MIL-47 type


@pytest.fixture(scope="module")
def linker_table():
    with gzip.open(bundled.LINKERS, "rt", encoding="utf-8") as handle:
        return json.load(handle)["linkers"]


@pytest.fixture(scope="module")
def cr_descriptors():
    """coremof-v1 descriptors of the Cr-BDC fixture, computed once."""
    return D.coremof_descriptors(Structure.from_file(CR_BDC), samples_per_atom=500)


# ----------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------


@pytest.mark.parametrize("text, code", [
    ("pcu", "pcu"),                                              # CoRE's own code
    ("pcu 6-c, 2-fold interpenetrated", "pcu"),                  # the Void's summary
    ("(3,4)-c net, not in the reference library", "other"),
    ("unlisted", "other"),
    ("not identified: unstable net", "unstable"),
    ("not identified: no periodic net", ""),
    ("", ""), (None, ""),
])
def test_net_codes_read_coremof_and_the_void_alike(text, code):
    """The same net must land in the same one-hot column whether the
    text came from the CoRE index (training) or the Void (prediction)."""
    assert D.net_code(text) == code


def test_net_features_are_one_hot_with_other_and_unstable():
    row = D.net_features("dia")
    assert row["net_dia"] == 1.0 and sum(row.values()) == 1.0
    assert D.net_features("xyz")["net_other"] == 1.0
    assert D.net_features("unstable")["net_unstable"] == 1.0
    assert sum(D.net_features("").values()) == 0.0


@pytest.mark.parametrize("stem", [
    "2010[Cr][bpq]3[ASR]1", "2017[ZnSi][nan]3[ASR]1", "2019[V][bpq]3[ASR]1",
    "2020[Cd][hcb]2[ASR]1", "2023[Ni][dia]3[ASR]1",
])
def test_linkers_from_the_bond_graph_match_mofid(stem, linker_table):
    """The training side's linker chemistry comes from CoRE's MOFid
    strings, the prediction side's from the bond graph: the heavy-atom
    formulas of the distinct linkers must be the same (118 of 124 CoRE
    SI structures agree; these five fixtures are among them)."""
    def heavy(fragments):
        return sorted(tuple(sorted((k, v) for k, v in f.items() if k not in ("H", "D")))
                      for f in fragments)

    mine = D.organic_fragments(Structure.from_file(FIXTURES / "coremof" / f"{stem}.cif"))
    assert heavy(mine) == heavy(linker_table[stem])


def test_a_ligand_on_one_metal_is_not_a_linker():
    """A carbon fragment that touches a single metal (coordinated
    solvent, a terminal formate) links nothing: MOFid does not list it."""
    from pymatgen.core import Lattice

    # One Zn, bridged along a by a two-carbon "linker" (C bonded to Zn
    # at both ends through the periodic boundary), plus a CO on one Zn.
    lattice = Lattice.orthorhombic(5.0, 12.0, 12.0)
    structure = Structure(lattice, ["Zn", "C", "C", "C", "O"],
                          [[0.0, 0.5, 0.5], [0.39, 0.5, 0.5], [0.61, 0.5, 0.5],
                           [0.0, 0.66, 0.5], [0.0, 0.755, 0.5]])
    fragments = D.organic_fragments(structure)
    assert fragments == [{"C": 2}]


def test_linker_features_average_distinct_linkers():
    row = D.linker_features([{"C": 8, "O": 4, "H": 4}, {"C": 6, "N": 2, "H": 4}])
    assert row["linker_types"] == 2.0 and row["linker_heavy_atoms_max"] == 12.0
    assert row["linker_O_fraction"] == pytest.approx((4 / 12) / 2)
    assert row["linker_N_fraction"] == pytest.approx((2 / 8) / 2)
    empty = D.linker_features([])
    assert empty["linker_types"] == 0.0 and math.isnan(empty["linker_N_fraction"])


def test_a_structure_gets_the_descriptors_its_index_entry_has(cr_descriptors):
    """The core check of the bridge: the Void's coremof-v1 row for a CoRE
    structure against the row built from CoRE's own index values, which
    is what the models were trained on. Geometry within CoRE-comparison
    tolerances; chemistry, dimension, catenation and open metal sites
    exactly. The net was the one known gap until round 70: CoRE names
    MIL-53's rod-SBU net bpq, and the Void assigned none; now both say
    bpq."""
    from thevoid_core.services.reference import coremof

    entry = coremof.load_index().by_id["2010[Cr][bpq]3[ASR]1"]
    with gzip.open(bundled.LINKERS, "rt", encoding="utf-8") as handle:
        linkers = json.load(handle)["linkers"][entry.id]
    reference = D.coremof_row(entry.values, {"fragments": linkers})
    mine, notes = cr_descriptors
    assert set(mine) == set(D.COREMOF_V1)
    assert mine["lcd_A"] == pytest.approx(reference["lcd_A"], abs=0.15)
    assert mine["pld_A"] == pytest.approx(reference["pld_A"], abs=0.15)
    assert mine["asa_m2_g"] == pytest.approx(reference["asa_m2_g"], rel=0.05)
    assert mine["density_g_cm3"] == pytest.approx(reference["density_g_cm3"], rel=0.01)
    assert mine["mean_atomic_mass"] == pytest.approx(reference["mean_atomic_mass"], rel=0.01)
    exact = [f for f in D.COREMOF_V1 if f.startswith(("metal_", "n_metal", "has_l", "has_a",
                                                        "linker_"))]
    exact += ["dimension", "catenation", "has_oms"]
    assert {f: mine[f] for f in exact} == {f: reference[f] for f in exact}
    assert reference["net_bpq"] == 1.0
    # Round 70: the rods are split at their metals, and the Void's net
    # is CoRE's (before, it assigned none).
    nets = [f for f in D.COREMOF_V1 if f.startswith("net_")]
    assert {f: mine[f] for f in nets} == {f: reference[f] for f in nets}


def test_open_metal_sites_are_missing_where_the_void_cannot_check_them():
    """Round 61's transfer check: the Void's OMS detection covers 16
    metals, and reporting "no open metal sites" for a lanthanide MOF it
    never checked had moved predicted water Henry coefficients by 9-11
    decades (CoRE flags most as having them). Now has_oms is missing,
    on the training side too, whenever a metal is outside that set.
    (Round 69 calibrated the lanthanides, so they now count; Pb, with
    too few labelled structures to calibrate, still does not.)"""
    assert D.oms_checkable(["Zn", "Cu"]) and not D.oms_checkable(["Zn", "Pb"])
    assert D.oms_checkable(["Eu"])
    row = D.coremof_row({"metals": "Pb", "has_oms": True}, None)
    assert math.isnan(row["has_oms"])
    assert D.coremof_row({"metals": "Cu", "has_oms": True}, None)["has_oms"] == 1.0
    assert D.coremof_row({"metals": "Eu", "has_oms": True}, None)["has_oms"] == 1.0
    summary = {"formula": "Pb2 C24 H12 O12", "open_metal_sites": 0, "topology": ""}
    assert math.isnan(D.void_row(summary, [])["has_oms"])
    summary["formula"] = "Zn4 C24 H12 O13"
    assert D.void_row(summary, [])["has_oms"] == 0.0


def test_composition_features_from_a_formula():
    row = D.composition_features("Zn4 C24 H12 O13")
    assert row["n_metal_species"] == 1.0 and row["metal_Z_max"] == 30.0
    assert row["O_fraction"] == pytest.approx(13 / 41)
    bad = D.composition_features("not a formula")
    assert math.isnan(bad["mean_atomic_mass"])


# ----------------------------------------------------------------------
# The recipe, on synthetic data where the truth is known
# ----------------------------------------------------------------------


def _synthetic(n=900, seed=1, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = 2.0 * X[:, 0] + np.sin(2 * X[:, 1]) + noise * rng.normal(size=n)
    rows = [{f"x{j}": float(X[i, j]) for j in range(4)} for i in range(n)]
    groups = [f"g{i // 2}" for i in range(n)]  # pairs, like ASR/FSR copies
    return rows, y, groups


def test_the_split_keeps_each_group_on_one_side():
    groups = [f"g{i // 3}" for i in range(300)]
    parts = group_split(groups, seed=0)
    assert sum(len(p) for p in parts) == 300
    sides = [{groups[i] for i in p} for p in parts]
    assert not (sides[0] & sides[1] or sides[0] & sides[2] or sides[1] & sides[2])
    assert [len(s) for s in sides] == [70, 10, 20]


def test_the_interval_covers_about_ninety_percent_of_new_data():
    """Split-conformal: on fresh draws from the same distribution the 90%
    intervals must hold close to 90% of the true values (and the model
    must beat always-the-mean)."""
    rows, y, groups = _synthetic()
    model = train(rows, y, groups, name="toy", label="toy", unit="", target="y",
                  descriptor_set="void-v1", features=["x0", "x1", "x2", "x3"])
    m = model.card["test_metrics"]
    assert m["mae"] < 0.5 * m["baseline_mae"] and m["r2"] > 0.8
    fresh_rows, fresh_y, _ = _synthetic(n=1500, seed=7)
    predictions = model.predict_rows(fresh_rows)
    inside = np.mean([p.low <= t <= p.high for p, t in zip(predictions, fresh_y)])
    assert 0.85 <= inside <= 0.95
    assert 0.85 <= m["interval_coverage"] <= 0.95


def test_a_structure_unlike_the_training_set_is_flagged():
    rows, y, groups = _synthetic()
    model = train(rows, y, groups, name="toy", label="toy", unit="", target="y",
                  descriptor_set="void-v1", features=["x0", "x1", "x2", "x3"])
    typical, far = model.predict_rows([{"x0": 0.1, "x1": -0.2, "x2": 0.0, "x3": 0.3},
                                       {"x0": 9.0, "x1": 9.0, "x2": -9.0, "x3": 9.0}])
    assert typical.in_domain and not far.in_domain
    assert far.domain_distance > 3 * typical.domain_distance
    assert 0.9 <= model.card["test_metrics"]["inside_domain_fraction"] <= 1.0


def test_log_models_and_clipping_transform_back():
    rows, y, groups = _synthetic()
    log_model = train(rows, y, groups, name="toy", label="toy", unit="", target="y",
                      descriptor_set="void-v1", features=["x0", "x1", "x2", "x3"],
                      transform="log10")
    p = log_model.predict_rows([{"x0": 0.0, "x1": 0.0, "x2": 0.0, "x3": 0.0}])[0]
    assert p.low < p.value < p.high and p.value == pytest.approx(10 ** np.log10(p.value))
    assert 0.1 < p.value < 10  # 10**(about 0)

    clipped = train(rows, y, groups, name="toy", label="toy", unit="", target="y",
                    descriptor_set="void-v1", features=["x0", "x1", "x2", "x3"],
                    card={"clip": [0.0, 1.0]})
    q = clipped.predict_rows([{"x0": 3.0, "x1": 0.0, "x2": 0.0, "x3": 0.0}])[0]
    assert 0.0 <= q.low <= q.value <= q.high <= 1.0


def test_too_few_labels_is_refused():
    rows, y, groups = _synthetic(n=20)
    with pytest.raises(ValueError, match="too few"):
        train(rows, y, groups, name="toy", label="toy", unit="", target="y",
              descriptor_set="void-v1", features=["x0"])


def test_a_saved_model_predicts_the_same(tmp_path):
    rows, y, groups = _synthetic(n=300)
    model = train(rows, y, groups, name="toy", label="Toy", unit="u", target="y",
                  descriptor_set="void-v1", features=["x0", "x1", "x2", "x3"])
    model.save(tmp_path / "toy")
    card = json.loads((tmp_path / "toy" / "card.json").read_text())
    assert card["name"] == "toy" and card["recipe"]["ensemble_size"] == 5
    again = Model.load(tmp_path / "toy")
    probe = rows[:5]
    assert [p.value for p in again.predict_rows(probe)] == \
        pytest.approx([p.value for p in model.predict_rows(probe)])
    assert [p.in_domain for p in again.predict_rows(probe)] == \
        [p.in_domain for p in model.predict_rows(probe)]


# ----------------------------------------------------------------------
# The bundled models
# ----------------------------------------------------------------------


def test_every_bundled_model_has_a_card_that_says_what_it_is():
    for name in bundled.bundled_names():
        card = json.loads((bundled.MODELS_DIR / name / "card.json").read_text(encoding="utf-8"))
        assert card["descriptor_set"] == "coremof-v1" and card["features"] == list(D.COREMOF_V1)
        assert card["caveat"] and card["training_data"]["doi"]
        m = card["test_metrics"]
        # Better than always-the-mean, and the interval holds what it says.
        assert m["mae"] < 0.7 * m["baseline_mae"], name
        assert 0.87 <= m["interval_coverage"] <= 0.93, name
    stability = json.loads((bundled.MODELS_DIR / "thermal_stability" / "card.json").read_text())
    assert "ML predictions" in stability["caveat"]


def test_bundled_models_predict_for_a_structure(cr_descriptors):
    row, _notes = cr_descriptors
    predictions = {}
    for name in bundled.bundled_names():
        predictions[name] = bundled.load_bundled(name).predict_rows([row])[0]
    kh = predictions["water_kh"]
    assert kh.unit == "mmol/g/Pa" and kh.low < kh.value < kh.high and kh.in_domain
    assert 1e-8 < kh.value < 1e-1
    assert 200 < predictions["thermal_stability"].value < 600
    for name in ("solvent_stability", "water_stability"):
        assert 0.0 <= predictions[name].low <= predictions[name].value <= \
            predictions[name].high <= 1.0


def test_predict_structure_reuses_given_results(monkeypatch):
    """Framework.predict and the report pass the results they already
    have; coremof_descriptors must then not recompute them."""
    from thevoid_core.services import pore_analysis, surface_area
    from thevoid_core.services.ml.predict import predict_structure

    structure = Structure.from_file(CR_BDC)
    from thevoid_core.data.element_radii import CCDCRadiusProvider

    pores = pore_analysis.analyze_pores(structure, probe_radius=1.655,
                                        radius_provider=CCDCRadiusProvider())
    surface = surface_area.analyze_surface_area(
        structure, probe_radius=1.655, probe_name="CoRE", radius_provider=CCDCRadiusProvider(),
        n_samples_per_atom=200)

    def refuse(*a, **k):
        raise AssertionError("recomputed")

    monkeypatch.setattr(pore_analysis, "analyze_pores", refuse)
    monkeypatch.setattr(surface_area, "analyze_surface_area", refuse)
    result = predict_structure(structure, ["water_stability"],
                               cached={"pores": pores, "surface": surface})
    assert [p.model for p in result["predictions"]] == ["water_stability"]
    assert result["descriptors"]["coremof-v1"]["pld_A"] == pytest.approx(pores.pld)


def test_another_scikit_learn_retrains_into_the_cache(tmp_path, monkeypatch):
    """joblib trees are only safe to load with the scikit-learn that
    wrote them: with another version the model is retrained (here
    stood in for by loading the bundled one) into the cache, the card
    says why, and the next load reads the cache."""
    import sklearn

    from thevoid_core.services.reference import net

    monkeypatch.setattr(sklearn, "__version__", "0.0.test")
    monkeypatch.setattr(net, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(bundled, "_LOADED", {})
    calls = []

    def retrain(name, *a, **k):
        calls.append(name)
        return Model.load(bundled.MODELS_DIR / name)

    monkeypatch.setattr(bundled, "train_bundled", retrain)
    model = bundled.load_bundled("water_stability")
    assert calls == ["water_stability"]
    assert "0.0.test" in model.card["retrained_because"]
    cached = tmp_path / "models" / "0.0.test" / "water_stability"
    assert (cached / "card.json").is_file()
    monkeypatch.setattr(bundled, "_LOADED", {})
    bundled.load_bundled("water_stability")
    assert calls == ["water_stability"]  # read from the cache, not retrained again


def test_an_unknown_model_says_what_there_is():
    from thevoid_core.services.ml.predict import resolve

    with pytest.raises(ValueError, match="water_kh"):
        resolve(["no_such_model"])


def test_missing_scikit_learn_says_how_to_install_it(monkeypatch):
    import importlib.util

    from thevoid_core.services.ml import model as model_module

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "sklearn" else real(name, *a))
    with pytest.raises(ImportError, match=r"\.\[ml\]"):
        model_module._require_sklearn()


# ----------------------------------------------------------------------
# Reports and the command lines
# ----------------------------------------------------------------------


def test_the_report_and_summary_carry_predictions():
    from thevoid_core.services.report import (
        AnalysisConfig,
        run_analysis,
        summary_row,
        to_markdown,
    )

    config = AnalysisConfig(pore=False, surface_area=False, topology=False,
                            surface_samples_per_atom=300, predict=["water_stability"])
    report = run_analysis(Structure.from_file(CR_BDC), config, source=str(CR_BDC))
    assert "predictions" not in report.get("errors", {})
    q = report["results"]["predictions"]["models"]["water_stability"]
    assert q["low"] <= q["value"] <= q["high"] and "ML predictions" in q["note"]
    row = summary_row(report)
    assert row["pred_water_stability"] == pytest.approx(q["value"])
    assert row["pred_water_stability_in_domain"] in (True, 1, "True", "yes")
    assert "## Predicted properties (models, not measurements)" in to_markdown(report)


def test_train_cli_lists_the_bundled_models(capsys):
    from thevoid_core import train_cli

    assert train_cli.main(["list"]) == 0
    out = capsys.readouterr().out
    for name in bundled.bundled_names():
        assert f"{name}:" in out
    assert "ML predictions" in out and "90% interval covers" in out


# ----------------------------------------------------------------------
# Training on your own batch
# ----------------------------------------------------------------------


def test_uptake_is_interpolated_never_extrapolated():
    points = [(0.1, 1.0), (1.0, 2.0), (10.0, 4.0)]
    assert uptake_at(points, 0.1) == 1.0
    assert uptake_at(points, 0.55) == pytest.approx(1.5)
    assert uptake_at(points, 10.0) == 4.0
    assert uptake_at(points, 0.05) is None and uptake_at(points, 20.0) is None
    assert uptake_at([], 1.0) is None


def _mofxdb_folder(tmp_path):
    from thevoid_core.services.reference import fetch

    def opener(url):
        page = parse_qs(urlparse(url).query)["page"][0]
        return (FIXTURES / f"mofxdb_page{page}.json").read_bytes()

    fetch.fetch_mofxdb(tmp_path, database="hMOF", limit=10, opener=opener)
    return tmp_path


def test_mofxdb_labels_from_a_fetched_folder(tmp_path):
    folder = _mofxdb_folder(tmp_path / "hmof")
    rows, info = labels_from_mofxdb(folder, "CO2", 0.55)
    assert info["column"] == "ref_uptake_CO2_0.55_298K"
    assert info["adsorbate"] == "CarbonDioxide"
    by_file = {r["file"]: r[info["column"]] for r in rows}
    assert by_file["hMOF-1.cif"] == pytest.approx(1.5)  # between (0.1, 1.0) and (1.0, 2.0)
    assert all(f.endswith(".cif") for f in info["skipped"])

    # Out of the isotherm's range, or another temperature: no label.
    rows, info = labels_from_mofxdb(folder, "CO2", 50.0)
    assert rows == [] and "outside" in next(iter(info["skipped"].values()))
    rows, _ = labels_from_mofxdb(folder, "CO2", 0.55, temperature=77)
    assert rows == []


def test_the_train_cli_writes_labels(tmp_path, capsys):
    from thevoid_core import train_cli

    folder = _mofxdb_folder(tmp_path / "hmof")
    out = tmp_path / "labels.csv"
    assert train_cli.main(["mofxdb-labels", str(folder), "--adsorbate", "CO2",
                           "--pressure", "0.1", "--out", str(out)]) == 0
    with open(out, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and set(rows[0]) == {"file", "ref_uptake_CO2_0.1_298K"}
    assert "labels" in capsys.readouterr().out


def _synthetic_batch(folder, n=120, seed=3):
    """A summary.csv with the Void's columns, and a target that depends
    on PLD and surface area, as an uptake would."""
    rng = np.random.default_rng(seed)
    folder.mkdir(parents=True)
    nets = ["pcu", "dia", "sql"]
    columns = ["file", "formula", "lcd_A", "pld_A", "void_fraction_percent",
               "channel_asa_m2_per_g", "density_g_cm3", "accessible_dimensionality",
               "open_metal_sites", "topology", "errors", "ref_uptake_CO2_0.1_298K"]
    with open(folder / "summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for i in range(n):
            pld = rng.uniform(3, 12)
            asa = rng.uniform(200, 3500)
            uptake = 0.5 + 0.0008 * asa + 0.15 * pld + 0.1 * rng.normal()
            writer.writerow([f"m{i}.cif", "Zn4 C24 H12 O13", pld + 2, pld, rng.uniform(20, 80), asa,
                             rng.uniform(0.4, 1.6), 3, int(rng.integers(0, 3)),
                             f"{nets[i % 3]} 6-c", "", uptake])
    return folder


def test_train_from_a_batch_and_predict_from_a_summary_row(tmp_path):
    folder = _synthetic_batch(tmp_path / "results")
    model = train_from_batch(folder, "ref_uptake_CO2_0.1_298K", unit="mol/kg")
    assert model.descriptor_set == "void-v1" and model.nets == ["pcu", "dia", "sql"]
    assert model.card["training_data"]["target_column"] == "ref_uptake_CO2_0.1_298K"
    assert len(model.card["training_data"]["sha256"]) == 64
    m = model.card["test_metrics"]
    assert m["r2"] > 0.7
    top = [f for f, _ in model.card["most_important_features"][:3]]
    # lcd_A is pld_A + 2 here, so either can carry the pore-size signal.
    assert "channel_asa_m2_per_g" in top and {"pld_A", "lcd_A"} & set(top)

    row = {"lcd_A": 10, "pld_A": 8, "void_fraction_percent": 50, "channel_asa_m2_per_g": 2000,
           "density_g_cm3": 1.0, "accessible_dimensionality": 3, "open_metal_sites": 0,
           "topology": "pcu 6-c", "formula": "Zn4 C24 H12 O13"}
    p = model.predict_rows([D.void_row(row, model.nets)])[0]
    assert p.value == pytest.approx(0.5 + 1.6 + 1.2, abs=0.4) and p.in_domain

    with pytest.raises(ValueError, match="No column"):
        train_from_batch(folder, "ref_nothing")


def test_the_train_cli_fits_and_saves(tmp_path, capsys):
    from thevoid_core import train_cli

    folder = _synthetic_batch(tmp_path / "results")
    out = tmp_path / "co2_model"
    assert train_cli.main(["fit", str(folder), "--target", "ref_uptake_CO2_0.1_298K",
                           "--log", "--unit", "mol/kg", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "(log10)" in text and "--predict" in text
    card = json.loads((out / "card.json").read_text())
    assert card["transform"] == "log10" and card["unit"] == "mol/kg"
    assert train_cli.main(["fit", str(folder), "--target", "nope", "--out", str(out)]) == 1


def test_the_analyze_cli_writes_prediction_columns(tmp_path):
    from thevoid_core import cli

    cli.main([str(CR_BDC), "--out", str(tmp_path), "-q", "--samples", "300", "--no-pore",
              "--no-surface", "--no-topology", "--predict", "water_stability"])
    with open(tmp_path / "summary.csv", newline="") as handle:
        row = next(csv.DictReader(handle))
    low, value, high = (float(row[f"pred_water_stability{s}"]) for s in ("_low", "", "_high"))
    assert 0 <= low <= value <= high <= 1
    assert "pred_water_kh" not in row  # only the model asked for


# ----------------------------------------------------------------------
# The Python API
# ----------------------------------------------------------------------


def test_framework_predict_and_the_model_objects():
    import thevoid

    fw = thevoid.load(CR_BDC)
    predictions = fw.predict(["water_stability", "thermal_stability"], samples_per_atom=300)
    assert len(predictions) == 2 and "out of domain" not in repr(predictions)
    p = predictions["water_stability"]
    assert isinstance(p, thevoid.PredictionResult) and p.low <= p.value <= p.high
    assert "ML predictions" in p.caveat and p.to_dict()["value"] == p.value
    assert predictions.descriptors["coremof-v1"]["pld_A"] == pytest.approx(
        fw.pores(1.655, radii="ccdc").pld_A)  # the cached result, reused
    with pytest.raises(KeyError, match="water_kh"):
        predictions["water_kh"]
    assert set(predictions.to_dict()["models"]) == {"water_stability", "thermal_stability"}

    model = thevoid.load_model("water_stability")
    assert model.descriptor_set == "coremof-v1" and model.test_metrics["r2"] > 0.5
    assert model.predict(fw, samples_per_atom=300).value == pytest.approx(p.value)
    cards = thevoid.models()
    assert set(cards) == set(bundled.bundled_names()) and cards["water_kh"]["unit"] == "mmol/g/Pa"


def test_train_and_labels_from_python(tmp_path):
    import thevoid

    folder = _synthetic_batch(tmp_path / "results")
    model = thevoid.train(folder, "ref_uptake_CO2_0.1_298K", log=True, unit="mol/kg")
    assert isinstance(model, thevoid.Model) and model.descriptor_set == "void-v1"
    saved = model.save(tmp_path / "co2")
    again = thevoid.load_model(saved)
    assert again.name == "ref_uptake_CO2_0.1_298K" and again.card["transform"] == "log10"
    with pytest.raises(ValueError, match="Write the BatchTable"):
        thevoid.train(thevoid.BatchTable([], [], {}), "x")

    hmof = _mofxdb_folder(tmp_path / "hmof")
    rows, info = thevoid.mofxdb_labels(hmof, "CO2", 1.0, out=tmp_path / "labels.csv")
    assert rows and (tmp_path / "labels.csv").read_text().startswith("file,ref_uptake_CO2_1_298K")


# ----------------------------------------------------------------------
# The GUI
# ----------------------------------------------------------------------


def _fake_result():
    from thevoid_core.services.ml.model import Prediction

    return {"predictions": [
        Prediction("water_kh", "Water Henry coefficient (298 K)", "mmol/g/Pa", 8.7e-5, 2.8e-6,
                   2.8e-3, 0.2, True, 1.1, "inferred from the data"),
        Prediction("thermal_stability", "Thermal decomposition temperature", "°C", 411.2, 343.0,
                   478.9, 10.0, False, 9.5, "ML predictions, not measurements"),
    ], "descriptors": {"coremof-v1": {"pld_A": 7.1}}, "notes": {"net": "no net assigned"}}


def test_the_prediction_window_shows_intervals_and_the_domain():
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from thevoid_core.ui.dialogs.prediction_dialog import PredictionDialog, format_value

    QApplication.instance() or QApplication([])
    dialog = PredictionDialog()
    dialog.set_result(_fake_result())
    cells = [[dialog.table.item(r, c).text() for c in range(5)] for r in range(2)]
    assert cells[0] == ["Water Henry coefficient (298 K)", "8.70e-05", "2.80e-06 – 2.80e-03",
                        "mmol/g/Pa", "inside"]
    assert cells[1][1:3] == ["411", "343 – 479"] and cells[1][4] == "outside"
    notes = dialog.notes_label.text()
    assert "Outside the applicability domain" in notes and "ML predictions" in notes
    assert "no net assigned" in notes
    assert "not measurements" in dialog.header_label.text()
    assert dialog.selection_text().splitlines()[0] == "Property\tPredicted\t90% interval\tUnit\tDomain"
    asked = []
    dialog.models_requested.connect(asked.append)
    dialog.add_model_folder("/some/model")
    assert asked == [["/some/model"]]
    assert format_value(float("nan")) == "–" and format_value(0.642) == "0.642"


def test_the_menu_runs_predictions_into_the_report(monkeypatch):
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication([])
    from thevoid_core.ui.windows.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    window = MainWindow()
    monkeypatch.setattr(window._pool, "start", lambda worker: None)
    window._on_structure_ready(window._load_and_analyse(str(CR_BDC)))
    assert window.predict_action.text() == "Predict Properties..."
    assert "not measurements" in window.predict_action.toolTip()
    from thevoid_core.services.ml import predict as predict_module

    seen = {}

    def quick(structure, models):
        seen["models"] = models
        return _fake_result()

    monkeypatch.setattr(predict_module, "predict_structure", quick)
    window._on_predictions_ready(window._run_predictions(window.current_structure, ["/m"]))
    assert seen["models"] == bundled.bundled_names() + ["/m"]
    assert window._prediction_dialog.isVisible()
    assert "1 outside the applicability domain" in window.statusBar().currentMessage()
    report = window.build_analysis_report()
    assert report["results"]["predictions"]["models"]["water_kh"]["value"] == 8.7e-5
    from tests.sample_structures import ZN_DIA

    window._on_structure_ready(window._load_and_analyse(str(ZN_DIA)))
    assert window.current_predictions is None and not window._prediction_dialog.isVisible()
    window.close()


def test_a_feature_no_row_has_does_not_break_training():
    """Round 67: a batch run with --no-oms has no open-metal-site value
    anywhere, and scikit-learn's binning failed on the all-missing
    column ("window shape cannot be larger than input array shape").
    It is now held constant, recorded on the card, and ignored."""
    rows, y, groups = _synthetic()
    for row in rows:
        row["x_missing"] = float("nan")
    model = train(rows, y, groups, name="toy", label="toy", unit="", target="y",
                  descriptor_set="void-v1", features=["x0", "x1", "x2", "x3", "x_missing"])
    assert model.card["features_without_values"] == ["x_missing"]
    assert model.card["test_metrics"]["r2"] > 0.8
    a, b = model.predict_rows([{**rows[0], "x_missing": float("nan")},
                               {**rows[0], "x_missing": 7.0}])
    assert a.value == pytest.approx(b.value)
