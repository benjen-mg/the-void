from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Common metal-organic framework node metals, called out for testing
#: and documentation. is_metal() itself is general - every metal in the
#: periodic table, not just this list - this set exists only because it
#: was specifically requested and deserves direct test coverage rather
#: than assuming pymatgen's general classification happens to be right
#: for exactly these.
MOF_NODE_METALS = frozenset({"Zn", "Cu", "Co", "Ni", "Fe", "Zr", "Ti", "Al"})

#: Metals Open Metal Site detection is explicitly scoped to - its own
#: requirement, listed separately from MOF_NODE_METALS above rather
#: than reusing it silently, since the two lists exist for different
#: reasons and have already diverged.
#:
#: The original nine (Zn, Cu, Co, Ni, Fe, Zr, Ti, Al, Cr) were extended
#: in round 49 with the other metals that routinely form MOF nodes and
#: whose open sites are well documented in the literature: Mg and Mn
#: (M-MOF-74 / CPO-27, the textbook open-metal-site family, and Mn
#: carboxylate frameworks such as MUF-16(Mn)), Cd, V (MIL-47/MIL-100(V)),
#: Sc (MIL-100/88(Sc)), Hf (the UiO-66(Hf) analogue of Zr), and Ga
#: (MIL-53(Ga)). Before this, a Mn or Mg framework reported "0 open
#: metal sites (0 metal centres analyzed)" - silently indistinguishable
#: from a structure with no metals at all. Alkali and alkaline-earth
#: metals other than Mg, and the lanthanides, were still left out: their
#: coordination numbers vary too widely for a textbook reference maximum
#: to mean anything, which the whole classification depends on.
CHECKED_BEFORE_ROUND_69 = frozenset(
    {
        "Zn", "Cu", "Co", "Ni", "Fe", "Zr", "Ti", "Al", "Cr",
        "Mg", "Mn", "Cd", "V", "Sc", "Hf", "Ga",
    }
)

#: Round 69: metals whose saturated coordination number was calibrated
#: against CoRE MOF 2024's open-metal labels (tools/calibrate_oms.py,
#: recorded in oms_calibration.json with its held-out agreement):
#: {metal: saturated coordination number}. The lanthanides with Y, the
#: actinides Th-Pu, In and Ag at the time of writing. Metals with too
#: few labelled structures to calibrate (the alkali and alkaline-earth
#: metals, Pb, Bi, Mo, Pd, ...) stay unchecked.
OMS_CALIBRATION_FILE = Path(__file__).resolve().parent / "oms_calibration.json"


def _calibrated_saturation(path=OMS_CALIBRATION_FILE) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            groups = json.load(handle)["groups"]
    except (OSError, ValueError, KeyError):
        logger.warning("No OMS calibration at %s; only the %d metals checked before "
                       "round 69 are checked.", path, len(CHECKED_BEFORE_ROUND_69))
        return {}
    return {metal: int(group["saturated_coordination"])
            for group in groups.values() for metal in group["metals"]}


OMS_CALIBRATED_SATURATION: dict[str, int] = _calibrated_saturation()

OMS_SUPPORTED_METALS = CHECKED_BEFORE_ROUND_69 | frozenset(OMS_CALIBRATED_SATURATION)


def is_metal(symbol: str) -> bool:
    """Whether an element symbol is a metal, for MOF node/linker
    partitioning and coordination analysis alike.

    Delegates to pymatgen's own classification (Element.is_metal)
    rather than hand-maintaining a list: pymatgen's table already
    covers the whole periodic table correctly, including elements nobody
    thought to add to a hand-written set. An unrecognised symbol (not a
    real element, or a disorder/oxidation-state artifact that slipped
    through) is treated as non-metal - the safer default for a boundary
    decision, since misclassifying a real linker atom as a metal would
    silently cut it out of its own linker.
    """
    try:
        from pymatgen.core import Element

        return bool(Element(symbol).is_metal)
    except Exception:
        logger.debug(
            "Could not classify %r as metal or not; treating as "
            "non-metal.",
            symbol,
            exc_info=True,
        )
        return False
