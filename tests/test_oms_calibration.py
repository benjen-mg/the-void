"""Round 69: open-metal-site detection for the lanthanides, Y, the
actinides Th-Pu, In and Ag.

These metals were not checked before, because no textbook "saturated"
coordination number fits them. The geometric test is unchanged; each
new metal's saturated number was fitted to CoRE MOF 2024's own labels
on half of its SI structures (tools/calibrate_oms.py), and the file it
wrote (data/oms_calibration.json) records the method, the fitted
numbers and the agreement on the other half. These tests hold the code
to that file, check the fitting rule on data whose answer is known, and
check real CoRE structures come out as CoRE labels them.
"""

import importlib.util
import json
import logging
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "oms"
SM_DESOLVATED = FIXTURES / "2019[Sm][sqc]3[ASR]1.cif"
SM_SOLVATED = FIXTURES / "2019[Sm][sqc]3[FSR]1.cif"
IN_MOF = FIXTURES / "2020[In][hcb]2[ASR]1.cif"
PB_MOF = FIXTURES / "2021[Pb][llj]2[ASR]1.cif"


def _tool():
    spec = importlib.util.spec_from_file_location("calibrate_oms", ROOT / "tools" / "calibrate_oms.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _calibration():
    from thevoid_core.data.metal_classification import OMS_CALIBRATION_FILE

    return json.loads(OMS_CALIBRATION_FILE.read_text())


def _detect(path):
    from pymatgen.core import Structure

    from thevoid_core.services.oms_detection import detect_open_metal_sites
    from thevoid_core.services.site_labels import compute_site_labels

    structure = Structure.from_file(path)
    return detect_open_metal_sites(structure, compute_site_labels(structure))


# ----------------------------------------------------------------------
# The calibration record and the code agree
# ----------------------------------------------------------------------


def test_the_calibration_records_its_method_and_held_out_agreement():
    """What was fitted, on what, and how well it did on the half it was
    not fitted on - the numbers the calibration reported."""
    record = _calibration()
    assert "split by paper" in record["method"] and "CC-BY-4.0" in record["labels"]
    assert record["thresholds"] == {"asymmetry": 0.15, "widest_empty_cone_deg": 80.0}
    assert set(record["groups"]) == {"lanthanides and Y", "actinides", "In", "Ag"}
    for group in record["groups"].values():
        assert 2 <= group["saturated_coordination"] <= 12
        for half in ("calibration", "held_out"):
            assert group[half]["items"] >= record["min_items_per_half"]
            assert 0 <= group[half]["agreement"] <= 1
    assert "Eu" in record["groups"]["lanthanides and Y"]["metals"]
    assert "Y" in record["groups"]["lanthanides and Y"]["metals"]
    assert record["held_out_new_metals"]["items"] > 100
    assert record["held_out_metals_checked_before"]["items"] > 500
    # Too few labelled structures: left unchecked, and said so.
    assert "Pb" in record["not_calibrated"] and "K" in record["not_calibrated"]
    assert record["structures"] == 2664 and record["structures_failed"] == 0


def test_the_checked_metals_are_the_old_sixteen_plus_the_calibrated():
    from thevoid_core.data.metal_classification import (
        CHECKED_BEFORE_ROUND_69,
        OMS_CALIBRATED_SATURATION,
        OMS_SUPPORTED_METALS,
    )
    from thevoid_core.services.oms_detection import TYPICAL_MAX_COORDINATION

    record = _calibration()
    fitted = {m: g["saturated_coordination"] for g in record["groups"].values() for m in g["metals"]}
    assert OMS_CALIBRATED_SATURATION == fitted
    assert OMS_SUPPORTED_METALS == CHECKED_BEFORE_ROUND_69 | set(fitted)
    assert len(CHECKED_BEFORE_ROUND_69) == 16 and not CHECKED_BEFORE_ROUND_69 & set(fitted)
    for metal, number in fitted.items():
        assert TYPICAL_MAX_COORDINATION[metal] == number
    # The metals checked before keep their numbers.
    assert TYPICAL_MAX_COORDINATION["Zr"] == TYPICAL_MAX_COORDINATION["Hf"] == 8
    assert all(TYPICAL_MAX_COORDINATION[m] == 6 for m in CHECKED_BEFORE_ROUND_69 - {"Zr", "Hf"})
    for metal in ("Pb", "Bi", "Na", "K", "Ca"):
        assert metal not in OMS_SUPPORTED_METALS


def test_without_the_calibration_only_the_old_metals_are_checked(tmp_path, caplog):
    """A missing or broken file must not stop OMS detection: it falls
    back to the 16 metals, and says so."""
    from thevoid_core.data.metal_classification import _calibrated_saturation

    with caplog.at_level(logging.WARNING):
        assert _calibrated_saturation(tmp_path / "missing.json") == {}
    assert "only the 16 metals" in caplog.text
    (tmp_path / "broken.json").write_text("{")
    assert _calibrated_saturation(tmp_path / "broken.json") == {}


# ----------------------------------------------------------------------
# Real CoRE structures
# ----------------------------------------------------------------------


def test_desolvated_sm_is_open_and_solvated_sm_is_saturated():
    """One Sm framework from one paper, as CoRE gives it twice: with its
    solvent removed (CoRE: Sm open) and with it kept (CoRE: none). The
    solvated Sm is 10-coordinate; without the solvent it is 8, below the
    lanthanides' calibrated saturated number."""
    saturated = _calibration()["groups"]["lanthanides and Y"]["saturated_coordination"]
    assert 8 < saturated <= 10
    desolvated, solvated = _detect(SM_DESOLVATED), _detect(SM_SOLVATED)
    assert desolvated.unsupported_metals == solvated.unsupported_metals == ()
    assert {s.coordination_number for s in desolvated.sites} == {8}
    assert desolvated.oms_count == len(desolvated.sites) > 0
    assert {s.coordination_number for s in solvated.sites} == {10}
    assert solvated.oms_count == 0
    assert {s.saturated_at for s in desolvated.sites} == {saturated}


def test_an_in_framework_is_checked_and_saturated():
    result = _detect(IN_MOF)
    assert result.unsupported_metals == () and result.sites
    assert result.oms_count == 0


def test_pb_is_still_not_checked():
    from thevoid_core.services.ml.descriptors import oms_checkable

    result = _detect(PB_MOF)
    assert result.sites == () and result.unsupported_metals == ("Pb",)
    assert not oms_checkable(["Pb"]) and oms_checkable(["Sm"]) and oms_checkable(["In", "Zn"])


def test_the_report_records_the_number_used_and_where_it_came_from():
    from thevoid_core.services.report import oms_to_dict

    saturated = _calibration()["groups"]["lanthanides and Y"]["saturated_coordination"]
    d = oms_to_dict(_detect(SM_DESOLVATED))
    assert {s["saturated_coordination"] for s in d["sites"]} == {saturated}
    sm = d["calibrated_metals"]["Sm"]
    assert sm["group"] == "lanthanides and Y" and sm["saturated_coordination"] == saturated
    assert 0 < sm["held_out_agreement"] <= 1 and sm["held_out_items"] > 100
    # A metal checked before has no calibration entry.
    from thevoid_core.services.oms_detection import calibration_of

    assert calibration_of("Cu") is None and calibration_of("Pb") is None


def test_the_tool_sees_the_sites_detection_sees():
    """The calibration's features are the detector's own numbers."""
    record = _tool().site_features(SM_DESOLVATED)
    result = _detect(SM_DESOLVATED)
    assert "error" not in record
    assert [(s["metal"], s["cn"]) for s in record["sites"]] == [
        (s.metal_symbol, s.coordination_number) for s in result.sites]
    assert [s["asym"] for s in record["sites"]] == pytest.approx(
        [s.asymmetry_score for s in result.sites])


# ----------------------------------------------------------------------
# The fitting rule, on data whose answer is known
# ----------------------------------------------------------------------


def _site(cn, open_geometry=True):
    return {"metal": "X", "cn": cn, "asym": 0.3 if open_geometry else 0.0, "gap": 50.0}


def _items(tool, metal, spec):
    """spec: [(cn, CoRE open?, count)]; ids chosen to land in each half."""
    out, n = [], 0
    for cn, label, count in spec:
        for which in (0, 1):
            made = 0
            while made < count:
                core_id = f"2020[{metal}][abc]{n}[ASR]1"
                n += 1
                if tool.half(core_id) == which:
                    out.append((core_id, metal, [_site(cn)], label))
                    made += 1
    return out


def test_the_number_chosen_is_the_one_that_agrees_best():
    """Sites below 8 are open per the labels, 8 and above not: the best
    saturated number is 8, and it agrees exactly on the held-out half."""
    tool = _tool()
    items = _items(tool, "In", [(6, True, 6), (7, True, 6), (8, False, 6), (9, False, 6)])
    group = tool.calibrate(items)["groups"]["In"]
    assert group["saturated_coordination"] == 8
    assert group["held_out"]["agreement"] == 1.0
    assert group["calibration_curve"]["7"] < 1.0 and group["calibration_curve"]["9"] < 1.0


def test_ties_go_to_the_smaller_number_and_small_samples_are_left_out():
    tool = _tool()
    # All labelled open, all at CN 5: any number above 5 agrees fully;
    # 6 is the one that lets fewest sites be open.
    items = _items(tool, "In", [(5, True, 12)])
    assert tool.calibrate(items)["groups"]["In"]["saturated_coordination"] == 6
    few = _items(tool, "Pb", [(5, True, 4)])
    result = tool.calibrate(few)
    assert "Pb" not in result["groups"] and result["not_calibrated"]["Pb"]["items"] == 8
    # Metals checked before are never refitted.
    assert "Cu" not in tool.calibrate(_items(tool, "Cu", [(5, True, 12)]))["groups"]


def test_one_papers_structures_stay_on_one_side():
    """The ASR/FSR/ION versions of a structure - and the other
    frameworks of the same paper - must not straddle the split, or the
    held-out figure would be measured partly on what was fitted."""
    tool = _tool()
    ids = ["2019[Sm][sqc]3[ASR]1", "2019[Sm][sqc]3[FSR]1", "2019[Sm][sqc]3[ION]1"]
    assert len({tool.half(i) for i in ids}) == 1
    tool.PAPERS.update({"2001[Eu][a]1[ASR]1": "10.1/x", "2001[Eu][b]7[ASR]2": "10.1/x"})
    assert tool.half("2001[Eu][a]1[ASR]1") == tool.half("2001[Eu][b]7[ASR]2")
    assert tool.group_of("Eu") == tool.group_of("Y") == "lanthanides and Y"
    assert tool.group_of("U") == "actinides" and tool.group_of("In") == "In"


# ----------------------------------------------------------------------
# Old index entries
# ----------------------------------------------------------------------


def test_old_entries_with_a_newly_checked_metal_are_stale_in_oms_only():
    from thevoid_core.data.metal_classification import OMS_CALIBRATED_SATURATION
    from thevoid_core.services.index import staleness

    changes = [c for c in staleness.load_changes() if c.get("condition") == "has_metal"]
    assert len(changes) == 1 and changes[0]["columns"] == ["open_metal_sites"]
    assert set(changes[0]["metals"]) == set(OMS_CALIBRATED_SATURATION)

    def entry(metals):
        return {"entry_id": "e", "structure": {"file_name": "a.cif", "file_sha256": "s",
                                                "metals": metals},
                "analysis": {"software": {"the_void": "0.9.8"},
                             "summary": {"open_metal_sites": 0, "pld_A": 5.0}, "config": {}}}

    now = changes[0]["version"]
    assert set(staleness.staleness_of(entry(["Sm"]), changes, now).columns) == {"open_metal_sites"}
    assert not staleness.staleness_of(entry(["Cu"]), changes, now).stale
    assert not staleness.staleness_of(entry(["Pb"]), changes, now).stale
    # An entry that does not list its metals is assumed affected.
    old = entry(None)
    del old["structure"]["metals"]
    assert staleness.staleness_of(old, changes, now).stale
