"""Structures shared by many tests.

ZN_DIA is the general-purpose sample: a small, ordered, three-periodic Zn
MOF from CoRE MOF 2024's openly licensed SI set (CC-BY-4.0; see
fixtures/identity/README.txt), with PLD 4.08 A, LCD 5.11 A, the dia net
and the elements C, H, N, O and Zn. Tests that only need "a MOF" use it.
"""

from pathlib import Path

ZN_DIA = Path(__file__).parent / "fixtures" / "identity" / "2020[Zn][dia]3[ASR]1.cif"

#: ZN_DIA without its CIF charges (fixtures/open/README.txt).
ZN_DIA_UNCHARGED = Path(__file__).parent / "fixtures" / "open" / "zn_dia_uncharged.cif"
