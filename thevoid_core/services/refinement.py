"""What a structure file says about its own refinement (version 1.1.0).

A CIF written by a refinement program records how well the model fits
the diffraction data (R1, wR2, the goodness of fit), how much data and
how many parameters there were, and the largest features left in the
difference map. A SHELX .res file records the same in the REM lines
SHELXL writes after HKLF. The Void shows these beside the structure and
copies them into reports, so that an analysis says what quality of
structure it was run on. It does not compute or check any of them.

One dictionary for both sources, with these keys where the file has
them:

    source                 "CIF" or "SHELXL .res"
    r1                     R1 for the observed data (I > 2 sigma(I))
    r1_all                 R1 for all data
    wr2                    wR2 for all data
    goof                   goodness of fit, S
    reflections_observed   reflections with I > 2 sigma(I)
    reflections_all        unique reflections used
    parameters, restraints
    peak, hole             largest difference peak and hole, e/A^3
    temperature_K, wavelength_A
    twinned                True when the model has twin components
    solvent_mask           True when a solvent mask (SQUEEZE, Olex2's
                           mask) was used: the pores then hold
                           electron density that the atom list does not
    difference_peaks       how many Q peaks a .res lists
"""

from __future__ import annotations

import re
from pathlib import Path

#: Our key -> the CIF data names that carry it, current name first.
CIF_NAMES = {
    "r1": ("_refine_ls_R_factor_gt", "_refine_ls_R_factor_obs"),
    "r1_all": ("_refine_ls_R_factor_all",),
    "wr2": ("_refine_ls_wR_factor_ref", "_refine_ls_wR_factor_all"),
    "goof": ("_refine_ls_goodness_of_fit_ref", "_refine_ls_goodness_of_fit_all"),
    "reflections_observed": ("_reflns_number_gt", "_reflns_number_observed"),
    "reflections_all": ("_refine_ls_number_reflns", "_reflns_number_total"),
    "parameters": ("_refine_ls_number_parameters",),
    "restraints": ("_refine_ls_number_restraints",),
    "peak": ("_refine_diff_density_max",),
    "hole": ("_refine_diff_density_min",),
    "temperature_K": ("_diffrn_ambient_temperature", "_cell_measurement_temperature"),
    "wavelength_A": ("_diffrn_radiation_wavelength",),
}
_COUNTS = {"reflections_observed", "reflections_all", "parameters", "restraints"}
_MASK_PREFIXES = ("_platon_squeeze", "_smtbx_masks", "_shelx_fab_file")
_NUMBER = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _number(text):
    """'100(2)' -> 100.0; '?' or '.' -> None."""
    if isinstance(text, (list, tuple)):
        text = text[0] if text else None
    match = _NUMBER.match(str(text)) if text is not None else None
    return float(match.group(1)) if match else None


def refinement_from_cif_data(data: dict) -> dict:
    """From one CIF block's data-name -> value mapping."""
    info = {}
    for key, names in CIF_NAMES.items():
        for name in names:
            value = _number(data.get(name))
            if value is not None:
                info[key] = int(round(value)) if key in _COUNTS else value
                break
    quality = any(k in info for k in ("r1", "wr2", "goof"))
    if not quality:
        return {}
    for name, value in data.items():
        lowered = name.lower()
        filled = str(value).strip() not in ("", "?", ".", "[]")
        if lowered.startswith(_MASK_PREFIXES) and filled:
            info["solvent_mask"] = True
        if lowered.startswith("_twin_") and filled:
            info["twinned"] = True
    info["source"] = "CIF"
    return info


def refinement_from_cif(path) -> dict:
    """From the first block of a CIF that reports a refinement; {} when
    none does (a database or hand-made CIF) or the file cannot be read."""
    try:
        from pymatgen.io.cif import CifFile

        blocks = CifFile.from_file(str(path)).data
    except Exception:
        return {}
    for block in blocks.values():
        info = refinement_from_cif_data(block.data)
        if info:
            return info
    return {}


def refinement_from_shelx(model) -> dict:
    """From a ShelxModel (services/shelx). A .ins, which has not been
    refined, gives only the conditions; a .res gives the fit as well."""
    info = dict(model.refinement)
    if model.temperature_C is not None:
        info["temperature_K"] = round(model.temperature_C + 273.15, 2)
    if model.wavelength:
        info["wavelength_A"] = model.wavelength
    if model.twinned:
        info["twinned"] = True
    if model.solvent_mask:
        info["solvent_mask"] = True
    if model.difference_peaks:
        info["difference_peaks"] = model.difference_peaks
    info["source"] = "SHELXL .res" if model.refinement else "SHELX .ins"
    return info


def refinement_for_file(path) -> dict:
    """The refinement details of a structure file; {} when it has none."""
    if path is None:
        return {}
    suffix = Path(path).suffix.lower()
    if suffix == ".cif":
        return refinement_from_cif(path)
    if suffix in (".res", ".ins"):
        try:
            from thevoid_core.services.shelx import read_shelx

            return refinement_from_shelx(read_shelx(path))
        except Exception:
            return {}
    return {}


def refinement_rows(info: dict) -> list:
    """[(label, text)] for a panel or a report, in reading order."""
    if not info:
        return []
    rows = []

    def number(key, digits):
        return f"{info[key]:.{digits}f}" if key in info else None

    r1 = number("r1", 4)
    if r1:
        observed = f" ({info['reflections_observed']} reflections)" \
            if "reflections_observed" in info else ""
        rows.append(("R1 (I > 2σ(I))", r1 + observed))
    if "r1_all" in info:
        rows.append(("R1 (all data)", number("r1_all", 4)))
    if "wr2" in info:
        rows.append(("wR2 (all data)", number("wr2", 4)))
    if "goof" in info:
        rows.append(("Goodness of fit", number("goof", 3)))
    counts = [str(info[k]) if k in info else "–"
              for k in ("reflections_all", "restraints", "parameters")]
    if any(c != "–" for c in counts):
        rows.append(("Data / restraints / parameters", " / ".join(counts)))
    if "peak" in info or "hole" in info:
        rows.append(("Largest peak / hole (e Å⁻³)",
                     f"{number('peak', 3) or '–'} / {number('hole', 3) or '–'}"))
    if "temperature_K" in info:
        rows.append(("Temperature (K)", f"{info['temperature_K']:.0f}"))
    if "wavelength_A" in info:
        rows.append(("Wavelength (Å)", f"{info['wavelength_A']:.5f}"))
    notes = []
    if info.get("solvent_mask"):
        notes.append("solvent mask used: the pores hold density that is not in the atom list")
    if info.get("twinned"):
        notes.append("refined as a twin")
    if info.get("source") == "SHELX .ins":
        notes.append("not refined yet (.ins)")
    if notes:
        rows.append(("Notes", "; ".join(notes)))
    return rows
