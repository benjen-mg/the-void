from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ElementRadiusProvider:
    """Atomic radii in angstrom.

    Bond detection and ball-and-stick sizing both want covalent radii, so
    those are the primary table (Cordero et al., Dalton Trans. 2008,
    2832-2838). The previous implementation used
    ``atomic_radius or average_ionic_radius`` from pymatgen, which mixes
    two different physical quantities and gave alkali metals balls
    roughly three times the size of carbon.

    pymatgen is consulted only as a fallback for symbols outside the
    table, so nothing is imported at module scope and the provider stays
    cheap to construct on a worker thread.
    """

    DEFAULT_RADIUS = 1.0

    COVALENT_RADII: dict[str, float] = {
        "H": 0.31, "He": 0.28,
        "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
        "O": 0.66, "F": 0.57, "Ne": 0.58,
        "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
        "S": 1.05, "Cl": 1.02, "Ar": 1.06,
        "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53,
        "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
        "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19,
        "Se": 1.20, "Br": 1.20, "Kr": 1.16,
        "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64,
        "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39,
        "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39, "Sb": 1.39,
        "Te": 1.38, "I": 1.39, "Xe": 1.40,
        "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03,
        "Nd": 2.01, "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96,
        "Tb": 1.94, "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90,
        "Yb": 1.87, "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W": 1.62,
        "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
        "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40,
        "At": 1.50, "Rn": 1.50,
        "Fr": 2.60, "Ra": 2.21, "Ac": 2.15, "Th": 2.06, "Pa": 2.00,
        "U": 1.96, "Np": 1.90, "Pu": 1.87, "Am": 1.80, "Cm": 1.69,
        # Bk (97) through Lr (103): no well-established experimental
        # covalent radius was available to draw from for this table.
        # Rather than let these seven fall through to the pymatgen
        # mix of atomic_radius/average_ionic_radius - the exact
        # inconsistency this table exists to avoid (see the class
        # docstring) - they are linearly interpolated across the
        # gentle actinide-contraction trend already visible from Ac
        # (2.15) through Cm (1.69), continued at the same per-element
        # step out to an estimated Lr of 1.61. These seven values are
        # therefore approximations, not measurements; treat bonding
        # results for these elements as a starting point to verify,
        # not a settled answer.
        "Bk": 1.68, "Cf": 1.67, "Es": 1.66, "Fm": 1.64,
        "Md": 1.63, "No": 1.62, "Lr": 1.61,
    }

    #: Deuterium and tritium are hydrogen for every purpose here.
    ALIASES = {"D": "H", "T": "H"}

    #: Van der Waals radii in angstrom (Bondi, J. Phys. Chem. 1964,
    #: 68, 441; Batsanov, Inorg. Mater. 2001, 37, 871 for elements
    #: Bondi didn't cover). Used only for space-filling display - bond
    #: detection and ball-and-stick sizing use the covalent table above.
    #: Deliberately not merged into one table: covalent and van der
    #: Waals radii answer different physical questions, and an element
    #: missing from this table falls back to a multiple of its covalent
    #: radius (see get_vdw_radius) rather than silently borrowing the
    #: wrong kind of radius.
    VDW_RADII: dict[str, float] = {
        "H": 1.20, "He": 1.40,
        "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70, "N": 1.55,
        "O": 1.52, "F": 1.47, "Ne": 1.54,
        "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80,
        "S": 1.80, "Cl": 1.75, "Ar": 1.88,
        "K": 2.75, "Ca": 2.31, "Sc": 2.15, "Ti": 2.11, "V": 2.07,
        "Cr": 2.06, "Mn": 2.05, "Fe": 2.04, "Co": 2.00, "Ni": 1.97,
        "Cu": 1.96, "Zn": 2.01, "Ga": 1.87, "Ge": 2.11, "As": 1.85,
        "Se": 1.90, "Br": 1.85, "Kr": 2.02,
        "Rb": 3.03, "Sr": 2.49, "Y": 2.32, "Zr": 2.23, "Nb": 2.18,
        "Mo": 2.17, "Tc": 2.16, "Ru": 2.13, "Rh": 2.10, "Pd": 2.10,
        "Ag": 2.11, "Cd": 2.18, "In": 1.93, "Sn": 2.17, "Sb": 2.06,
        "Te": 2.06, "I": 1.98, "Xe": 2.16,
        "Cs": 3.43, "Ba": 2.68, "La": 2.43, "Ce": 2.42, "Pr": 2.40,
        "Nd": 2.39, "Pm": 2.38, "Sm": 2.36, "Eu": 2.35, "Gd": 2.34,
        "Tb": 2.33, "Dy": 2.31, "Ho": 2.30, "Er": 2.29, "Tm": 2.27,
        "Yb": 2.26, "Lu": 2.24, "Hf": 2.23, "Ta": 2.22, "W": 2.18,
        "Re": 2.16, "Os": 2.16, "Ir": 2.13, "Pt": 2.13, "Au": 2.14,
        "Hg": 2.23, "Tl": 1.96, "Pb": 2.02, "Bi": 2.07, "Po": 1.97,
        "At": 2.02, "Rn": 2.20,
        "Fr": 3.48, "Ra": 2.83, "Ac": 2.47, "Th": 2.45, "Pa": 2.43,
        "U": 2.41, "Np": 2.39, "Pu": 2.43, "Am": 2.44, "Cm": 2.45,
    }

    #: For elements missing from VDW_RADII (mainly the late actinides),
    #: approximate as covalent radius times this factor - roughly the
    #: covalent-to-van-der-Waals ratio typical of the surrounding rows,
    #: better than silently falling through to a whole different table.
    VDW_FROM_COVALENT_FACTOR = 1.4

    def get_vdw_radius(self, symbol: str) -> float:
        key = self._normalise(symbol)

        radius = self.VDW_RADII.get(key)

        if radius is not None:
            return radius

        return self.get_covalent_radius(symbol) * self.VDW_FROM_COVALENT_FACTOR

    def get_covalent_radius(self, symbol: str) -> float:
        key = self._normalise(symbol)

        radius = self.COVALENT_RADII.get(key)

        if radius is not None:
            return radius

        return self._from_pymatgen(key)

    def get_radius(self, symbol: str) -> float:
        """General-purpose accessor used by the viewer."""
        return self.get_covalent_radius(symbol)

    def _from_pymatgen(self, symbol: str) -> float:
        try:
            from pymatgen.core import Element

            element = Element(symbol)

            radius = element.atomic_radius or element.average_ionic_radius

            if radius is None:
                return self.DEFAULT_RADIUS

            return float(radius)
        except Exception:
            logger.debug("No radius for %r; using default.", symbol)
            return self.DEFAULT_RADIUS

    def _normalise(self, symbol: str) -> str:
        if not symbol:
            return ""

        letters = "".join(
            char for char in str(symbol).strip() if char.isalpha()
        )

        key = letters[:1].upper() + letters[1:2].lower()

        return self.ALIASES.get(key, key)


class CCDCRadiusProvider(ElementRadiusProvider):
    """Van der Waals radii as Zeo++ uses them by default (round 56) - the
    CCDC's set: Bondi (1964), hydrogen 1.09 A (Rowland & Taylor, J.
    Phys. Chem. 1996, 100, 7384), and 2.00 A for every element Bondi
    gives no value for.

    CoRE MOF, and most screening databases computed with Zeo++, use
    these; the Void's own analyses use ElementRadiusProvider's table,
    which gives transition metals radii of about 2 A (Cu 1.96, Zn 2.01)
    where Bondi's are 1.40 and 1.39, and H 1.20. Comparing the Void's
    numbers with such a database is only a test of the method when both
    use the same radii, so services/reference/compare.py computes with
    this table. The values are Zeo++ 0.3's built-in ones, read back
    from it (through pyzeo) for every element H-Cm and kept in
    tests/fixtures/zeopp_default_radii.json, which
    tests/test_reference_coremof.py checks this table against.
    """

    DEFAULT_VDW = 2.00

    VDW_RADII: dict[str, float] = {
        "H": 1.09, "He": 1.40, "Li": 1.82, "C": 1.70, "N": 1.55, "O": 1.52,
        "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73, "Si": 2.10,
        "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88, "K": 2.75,
        "Ni": 1.63, "Cu": 1.40, "Zn": 1.39, "Ga": 1.87, "As": 1.85,
        "Se": 1.90, "Br": 1.85, "Kr": 2.02, "Pd": 1.63, "Ag": 1.72,
        "Cd": 1.58, "In": 1.93, "Sn": 2.17, "Te": 2.06, "I": 1.98,
        "Xe": 2.16, "Pt": 1.72, "Au": 1.66, "Hg": 1.55, "Tl": 1.96,
        "Pb": 2.02, "U": 1.86,
    }

    def get_vdw_radius(self, symbol: str) -> float:
        return self.VDW_RADII.get(self._normalise(symbol), self.DEFAULT_VDW)
