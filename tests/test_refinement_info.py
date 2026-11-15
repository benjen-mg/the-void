"""Version 1.1.0: the refinement details a file reports about itself
(services/refinement.py), in the Structure panel, reports and the API.

The Void shows these numbers so that an analysis says what quality of
structure it was run on; a wrong R1 beside a pore size would mislead.
The check is that a SHELXL .res and the CIF SHELXL wrote from the same
refinement give the same numbers, and that a file that reports nothing
shows nothing rather than zeros.
"""

import os
import warnings
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from tests.sample_structures import ZN_DIA  # noqa: E402
from thevoid_core.services.refinement import (  # noqa: E402
    refinement_for_file,
    refinement_from_cif_data,
    refinement_rows,
)

FIXTURES = Path(__file__).parent / "fixtures" / "shelx"
I43D_RES, I43D_CIF, P21C_RES = (FIXTURES / n for n in ("I-43d.res", "I-43d.cif", "p21c.res"))


def test_the_res_and_the_cif_of_one_refinement_agree():
    res, cif = refinement_for_file(I43D_RES), refinement_for_file(I43D_CIF)
    for key in ("r1", "r1_all", "reflections_observed", "reflections_all", "parameters",
                "restraints", "peak", "hole", "wavelength_A"):
        assert res[key] == cif[key], key
    assert (cif["r1"], cif["wr2"], cif["goof"]) == (0.0321, 0.086, 1.095)
    assert cif["temperature_K"] == 100 and res["temperature_K"] == pytest.approx(100.15)
    assert (res["source"], cif["source"]) == ("SHELXL .res", "CIF")
    assert res["difference_peaks"] == 3
    # SHELXL-2012 did not print wR2 in the .res; nothing is invented.
    assert "wr2" not in res


def test_a_newer_res_has_wr2_and_goof():
    info = refinement_for_file(P21C_RES)
    assert (info["r1"], info["wr2"], info["goof"]) == (0.04, 0.1005, 1.016)
    assert (info["reflections_all"], info["restraints"], info["parameters"]) == (10786, 1842, 945)
    assert (info["peak"], info["hole"]) == (0.407, -0.691)


def test_a_file_that_reports_no_refinement_shows_none(tmp_path):
    assert refinement_for_file(ZN_DIA) == {}          # a database CIF
    assert refinement_for_file(None) == {}
    assert refinement_for_file(tmp_path / "POSCAR") == {}
    assert refinement_rows({}) == []
    # Conditions alone (a temperature, no R factor) are not a refinement.
    assert refinement_from_cif_data({"_diffrn_ambient_temperature": "100(2)"}) == {}


def test_cif_values_with_uncertainties_old_names_masks_and_twins():
    info = refinement_from_cif_data({
        "_refine_ls_R_factor_obs": "0.0412",            # the older name
        "_refine_ls_wR_factor_ref": "0.1100",
        "_diffrn_ambient_temperature": "150(2)",
        "_refine_ls_number_parameters": "?",
        "_platon_squeeze_details": "disordered DMF was removed",
        "_twin_individual_id": ["1", "2"],
        "_smtbx_masks_special_details": "?",
    })
    assert info == {"r1": 0.0412, "wr2": 0.11, "temperature_K": 150.0,
                    "solvent_mask": True, "twinned": True, "source": "CIF"}


def test_the_rows_read_as_a_crystallographer_expects():
    rows = dict(refinement_rows(refinement_for_file(I43D_CIF)))
    assert rows["R1 (I > 2σ(I))"] == "0.0321 (2483 reflections)"
    assert rows["wR2 (all data)"] == "0.0860"
    assert rows["Goodness of fit"] == "1.095"
    assert rows["Data / restraints / parameters"] == "2743 / 194 / 247"
    assert rows["Largest peak / hole (e Å⁻³)"] == "0.207 / -0.267"
    assert rows["Temperature (K)"] == "100"
    notes = dict(refinement_rows({"r1": 0.05, "solvent_mask": True, "twinned": True}))["Notes"]
    assert "solvent mask" in notes and "twin" in notes


def test_reports_and_the_api_carry_it(tmp_path):
    thevoid = pytest.importorskip("thevoid")
    from thevoid_core.services.report import new_report, to_markdown

    fw = thevoid.load(I43D_CIF)
    assert fw.refinement["r1"] == 0.0321 and fw.refinement["goof"] == 1.095
    assert thevoid.load(ZN_DIA).refinement == {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = new_report(fw.structure, source=str(I43D_CIF))
    assert report["refinement"]["wr2"] == 0.086
    text = to_markdown(report)
    assert "## Refinement (as reported by the input file)" in text
    assert "| wR2 (all data) | 0.0860 |" in text

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = new_report(thevoid.load(ZN_DIA).structure, source=str(ZN_DIA))
    assert "refinement" not in plain and "## Refinement" not in to_markdown(plain)


def test_the_structure_panel_shows_it_only_when_there_is_some():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from thevoid_core.ui.widgets.structure_info_panel import StructureInfoPanel

    app = QApplication.instance() or QApplication([])  # noqa: F841
    panel = StructureInfoPanel()
    assert panel.refinement_group.isHidden()

    panel.set_refinement(refinement_for_file(I43D_CIF))
    assert not panel.refinement_group.isHidden()
    text = panel.refinement_label.text()
    assert "0.0321" in text and "2743 / 194 / 247" in text

    panel.set_refinement({})
    assert panel.refinement_group.isHidden()
