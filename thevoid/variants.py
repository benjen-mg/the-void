"""Functionalised variants (Build > Functionalise Linkers...,
the-void-functionalize)."""

from __future__ import annotations

from pathlib import Path


class Sites:
    """What a structure offers for functionalisation."""

    def __init__(self, parent):
        self._parent = parent

    @property
    def count(self) -> int:
        """Ring C-H on linkers, in the cell."""
        return len(self._parent.sites)

    @property
    def space_group(self) -> str:
        return self._parent.space_group

    @property
    def orbits(self) -> list:
        """[{"orbit", "linkers", "formula", "ring_CH_per_linker",
        "patterns": {count: [descriptor, ...]}}]"""
        return [{"orbit": o.number, "linkers": len(o.linkers), "formula": o.formula,
                 "ring_CH_per_linker": len(o.sites),
                 "patterns": {m: [d for _, d in ps] for m, ps in o.patterns.items()}}
                for o in self._parent.orbits]

    @property
    def notes(self) -> list:
        return list(self._parent.notes)

    def __str__(self):
        from thevoid_core.functionalize_cli import describe

        return describe(self._parent)

    __repr__ = __str__


class Variant:
    """One functionalised variant."""

    def __init__(self, raw, parent_name):
        self.raw = raw
        self.name = raw.name
        self.group = raw.group
        self.orbit = raw.orbit
        self.per_linker = raw.per_linker
        self.pattern = raw.descriptor
        self.substituents = len(raw.replaced)
        #: "ok", "tight" (written, closest contact under 80% of the vdW
        #: sum) or "clash" (rejected; see reason).
        self.status = raw.status
        self.reason = raw.reason
        self.space_group = raw.space_group
        closest = raw.closest[0] if raw.closest else None
        self.closest_contact_A = closest[1] if closest else None
        self.closest_contact_ratio = closest[0] if closest else None
        self.closest_contact = closest[2] if closest else ""
        #: Degrees the rings were turned to make room (0: built flat).
        self.ring_turn_deg = getattr(raw, "ring_turn", 0.0)
        self._parent_name = parent_name
        self._framework = None

    @property
    def framework(self):
        """The variant as a Framework, ready for any analysis."""
        from thevoid.framework import Framework

        if self._framework is None:
            self._framework = Framework(self.raw.structure, name=self.name)
        return self._framework

    def row(self) -> dict:
        return {"name": self.name, "parent": self._parent_name, "group": self.group,
                "orbit": self.orbit, "per_linker": self.per_linker, "pattern": self.pattern,
                "substituents": self.substituents, "status": self.status,
                "closest_contact_A": self.closest_contact_A,
                "closest_contact_ratio": self.closest_contact_ratio,
                "space_group": self.space_group, "reason": self.reason,
                "ring_turn_deg": self.ring_turn_deg}

    def __repr__(self):
        return (f"Variant({self.name!r}, {self.status}, closest {self.closest_contact_A} Å"
                f" = {self.closest_contact_ratio} x vdW)")


class Variants(list):
    """A list of Variant, with the context to write them."""

    def __init__(self, parent, built, groups, per_linker, *, parent_name="", ring_turn=True):
        super().__init__(Variant(v, parent_name) for v in built)
        self._parent, self._groups, self._per_linker = parent, groups, per_linker
        self._ring_turn = ring_turn

    def accepted(self) -> list:
        """The variants that pass the check (status ok or tight)."""
        return [v for v in self if v.status != "clash"]

    def write(self, folder, *, keep_clashes: bool = False) -> Path:
        """A CIF per variant, the parent, functionalization.json and
        variants.csv - the-void-functionalize's folder."""
        from thevoid_core.services import functionalize as fz

        fz.write_variants(self._parent, [v.raw for v in self], self._groups, folder,
                          per_linker=self._per_linker, keep_clashes=keep_clashes,
                          ring_turn=self._ring_turn)
        return Path(folder)

    @property
    def rows(self) -> list:
        return [v.row() for v in self]

    def to_dataframe(self):
        from thevoid._tables import dataframe

        return dataframe(self.rows)
