from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from thevoid_core.services.bonds.bond_model import site_occupancy

logger = logging.getLogger(__name__)


@dataclass
class CIFInfo:
    formula: str
    space_group: str

    #: Number of crystallographic sites in the cell. Not the same as the
    #: number of atoms when any site is partially occupied, which is why
    #: both are reported.
    site_count: int

    partial_occupancy_sites: int

    a: float
    b: float
    c: float

    alpha: float
    beta: float
    gamma: float

    file_path: str
    structure: Any

    #: Site label -> disorder group, read directly from the CIF's
    #: _atom_site_disorder_group column when present. Empty for files
    #: with no such column (most files, and every non-CIF format this
    #: loader accepts) or where a site's group is unset ('.' or '?').
    #: See PymatgenCIFLoader._read_disorder_groups for why this has to
    #: be read from the raw file rather than from the parsed Structure.
    disorder_groups: dict[str, str] = field(default_factory=dict)

    @property
    def atom_count(self) -> int:
        """Retained for callers written against the old attribute name."""
        return self.site_count

    @property
    def is_ordered(self) -> bool:
        return self.partial_occupancy_sites == 0

    @property
    def has_explicit_disorder_groups(self) -> bool:
        return bool(self.disorder_groups)


class PymatgenCIFLoader:
    def load(self, file_path: str) -> CIFInfo:
        structure = Structure.from_file(file_path)

        lattice = structure.lattice

        try:
            space_group = SpacegroupAnalyzer(
                structure,
                symprec=0.01,
            ).get_space_group_symbol()
        except Exception:
            space_group = "unknown"

        return CIFInfo(
            formula=structure.composition.reduced_formula,
            space_group=space_group,
            site_count=len(structure),
            partial_occupancy_sites=sum(
                1 for site in structure.sites if site_occupancy(site) < 0.99
            ),
            a=lattice.a,
            b=lattice.b,
            c=lattice.c,
            alpha=lattice.alpha,
            beta=lattice.beta,
            gamma=lattice.gamma,
            file_path=str(Path(file_path)),
            structure=structure,
            disorder_groups=self._read_disorder_groups(file_path),
        )

    @staticmethod
    def _read_disorder_groups(file_path: str) -> dict[str, str]:
        """Read _atom_site_disorder_group directly from the raw CIF.

        pymatgen's parsed Structure doesn't carry this column through as
        a site property - every site's `.properties` dict comes back
        empty regardless of what the CIF contains - so the only way to
        recover it is to read the raw CIF block pymatgen already parsed
        internally for its own purposes, keyed by _atom_site_label,
        which pymatgen does preserve as Site.label.

        Silently returns {} for anything that isn't a CIF (a POSCAR,
        VASP file, or other format Structure.from_file also accepts
        has no such column at all), or a CIF that doesn't have the tag,
        or if the label/group columns don't line up for any reason -
        this is supplementary metadata, not required for the structure
        to load.
        """
        if Path(file_path).suffix.lower() not in (".cif",):
            return {}

        try:
            from pymatgen.io.cif import CifParser

            parser = CifParser(file_path)

            groups: dict[str, str] = {}

            for block in parser._cif.data.values():
                labels = block.data.get("_atom_site_label")
                disorder = block.data.get("_atom_site_disorder_group")

                if not labels or not disorder:
                    continue

                for label, group in zip(labels, disorder):
                    # '.' and '?' are CIF's "not applicable" / "unknown"
                    # markers, not real group identifiers.
                    if group not in (".", "?", ""):
                        groups[label] = group

            return groups
        except Exception:
            logger.debug(
                "Could not read disorder groups from %r.",
                file_path,
                exc_info=True,
            )
            return {}
