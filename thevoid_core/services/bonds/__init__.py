"""Bond detection.

CovalentBondDetector - bonding by summed covalent radii - is the only
detector. An earlier version also offered CrystalNN (a coordination-
environment model from pymatgen), selectable from a menu. It was
removed: covalent-radius bonding handles the molecular and coordination
structures this application targets correctly and predictably, while
CrystalNN was tuned for inorganic frameworks, missed hydrogens, ran
slower, and was the source of an early duplicate-bond defect. Offering
a choice between a detector that works and one with known problems
added a decision the person opening a file shouldn't have to make.
"""

from thevoid_core.services.bonds.bond_model import (
    Bond,
    BondingDiagnostics,
    ShortContact,
    canonical_key,
    site_occupancy,
    statistics,
)
from thevoid_core.services.bonds.covalent_bond_detector import (
    CovalentBondDetector,
)

__all__ = [
    "Bond",
    "BondingDiagnostics",
    "CovalentBondDetector",
    "ShortContact",
    "canonical_key",
    "site_occupancy",
    "statistics",
]
