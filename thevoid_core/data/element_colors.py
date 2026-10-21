from __future__ import annotations

from typing import Dict, Tuple

RGB = Tuple[float, float, float]


class ElementColorProvider:
    """Jmol element colours.

    Stored internally as RGB 0-255 and returned as PyVista-compatible
    floats in the range 0-1. Coverage runs H (Z=1) through Lr (Z=103);
    anything outside that, or an unrecognised symbol, falls back to
    magenta so it is obvious on screen.
    """

    UNKNOWN_COLOR: RGB = (1.0, 0.0, 1.0)

    def __init__(self) -> None:
        #: Per-instance colour overrides, checked before falling back
        #: to the shared JMOL_COLORS class table - deliberately never
        #: written to JMOL_COLORS itself, since that dict is shared by
        #: every instance (every viewer, every window) and a colour
        #: change made in one window's "customize colours" dialog must
        #: not silently repaint every other window's atoms too.
        self._overrides: Dict[str, RGB] = {}

    def set_color(self, symbol: str, rgb: RGB) -> None:
        """Override this element's colour for this provider instance
        only. rgb is an (r, g, b) tuple of floats in [0, 1] - the same
        convention get_color already returns, so a caller reading a
        colour with get_color and writing it back with set_color needs
        no conversion.
        """
        self._overrides[self._normalise(symbol)] = rgb

    def reset_color(self, symbol: str) -> None:
        """Remove this element's override, reverting it to its default
        Jmol colour. Does nothing (not an error) if it had no override
        to begin with.
        """
        self._overrides.pop(self._normalise(symbol), None)

    def reset_all_colors(self) -> None:
        """Remove every override at once, reverting every element to
        its default Jmol colour.
        """
        self._overrides.clear()

    def is_overridden(self, symbol: str) -> bool:
        return self._normalise(symbol) in self._overrides

    JMOL_COLORS: Dict[str, Tuple[int, int, int]] = {
        "H": (255, 255, 255),
        "He": (217, 255, 255),
        "Li": (204, 128, 255),
        "Be": (194, 255, 0),
        "B": (255, 181, 181),
        "C": (144, 144, 144),
        "N": (48, 80, 248),
        "O": (255, 13, 13),
        "F": (144, 224, 80),
        "Ne": (179, 227, 245),
        "Na": (171, 92, 242),
        "Mg": (138, 255, 0),
        "Al": (191, 166, 166),
        "Si": (240, 200, 160),
        "P": (255, 128, 0),
        "S": (255, 255, 48),
        "Cl": (31, 240, 31),
        "Ar": (128, 209, 227),
        "K": (143, 64, 212),
        "Ca": (61, 255, 0),
        "Sc": (230, 230, 230),
        "Ti": (191, 194, 199),
        "V": (166, 166, 171),
        "Cr": (138, 153, 199),
        "Mn": (156, 122, 199),
        "Fe": (224, 102, 51),
        "Co": (240, 144, 160),
        "Ni": (80, 208, 80),
        "Cu": (200, 128, 51),
        "Zn": (125, 128, 176),
        "Ga": (194, 143, 143),
        "Ge": (102, 143, 143),
        "As": (189, 128, 227),
        "Se": (255, 161, 0),
        "Br": (166, 41, 41),
        "Kr": (92, 184, 209),
        "Rb": (112, 46, 176),
        "Sr": (0, 255, 0),
        "Y": (148, 255, 255),
        "Zr": (148, 224, 224),
        "Nb": (115, 194, 201),
        "Mo": (84, 181, 181),
        "Tc": (59, 158, 158),
        "Ru": (36, 143, 143),
        "Rh": (10, 125, 140),
        "Pd": (0, 105, 133),
        "Ag": (192, 192, 192),
        "Cd": (255, 217, 143),
        "In": (166, 117, 115),
        "Sn": (102, 128, 128),
        "Sb": (158, 99, 181),
        "Te": (212, 122, 0),
        "I": (148, 0, 148),
        "Xe": (66, 158, 176),
        "Cs": (87, 23, 143),
        "Ba": (0, 201, 0),
        "La": (112, 212, 255),
        "Ce": (255, 255, 199),
        "Pr": (217, 255, 199),
        "Nd": (199, 255, 199),
        "Pm": (163, 255, 199),
        "Sm": (143, 255, 199),
        "Eu": (97, 255, 199),
        "Gd": (69, 255, 199),
        "Tb": (48, 255, 199),
        "Dy": (31, 255, 199),
        "Ho": (0, 255, 156),
        "Er": (0, 230, 117),
        "Tm": (0, 212, 82),
        "Yb": (0, 191, 56),
        "Lu": (0, 171, 36),
        "Hf": (77, 194, 255),
        "Ta": (77, 166, 255),
        "W": (33, 148, 214),
        "Re": (38, 125, 171),
        "Os": (38, 102, 150),
        "Ir": (23, 84, 135),
        "Pt": (208, 208, 224),
        "Au": (255, 209, 35),
        "Hg": (184, 184, 208),
        "Tl": (166, 84, 77),
        "Pb": (87, 89, 97),
        "Bi": (158, 79, 181),
        # Previously missing: Po onwards. Actinide structures used to
        # render entirely magenta.
        "Po": (171, 92, 0),
        "At": (117, 79, 69),
        "Rn": (66, 130, 150),
        "Fr": (66, 0, 102),
        "Ra": (0, 125, 0),
        "Ac": (112, 171, 250),
        "Th": (0, 186, 255),
        "Pa": (0, 161, 255),
        "U": (0, 143, 255),
        "Np": (0, 128, 255),
        "Pu": (0, 107, 255),
        "Am": (84, 92, 242),
        "Cm": (120, 92, 227),
        "Bk": (138, 79, 227),
        "Cf": (161, 54, 212),
        "Es": (179, 31, 212),
        "Fm": (179, 31, 186),
        "Md": (179, 13, 166),
        "No": (189, 13, 135),
        "Lr": (199, 0, 102),
    }

    def get_color(self, symbol: str) -> RGB:
        normalised = self._normalise(symbol)

        if normalised in self._overrides:
            return self._overrides[normalised]

        rgb = self.JMOL_COLORS.get(normalised)

        if rgb is None:
            return self.UNKNOWN_COLOR

        return (
            rgb[0] / 255.0,
            rgb[1] / 255.0,
            rgb[2] / 255.0,
        )

    def has_color(self, symbol: str) -> bool:
        return self._normalise(symbol) in self.JMOL_COLORS

    @staticmethod
    def _normalise(symbol: str) -> str:
        """Strip oxidation state and whitespace: 'Fe3+' -> 'Fe'."""
        if not symbol:
            return ""

        cleaned = "".join(
            char for char in str(symbol).strip() if char.isalpha()
        )

        return cleaned[:1].upper() + cleaned[1:2].lower()

    def startup_report(self) -> str:
        examples = ["H", "C", "N", "O", "Zn", "Si", "U"]

        lines = ["=== the Void Jmol Color Report ==="]

        for element in examples:
            lines.append(f"{element:>2} -> {self.get_color(element)}")

        lines.append(f"Unknown -> {self.get_color('UnknownElement')}")

        return "\n".join(lines)
