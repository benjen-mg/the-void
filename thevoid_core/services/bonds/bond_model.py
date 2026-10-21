from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True, frozen=True)
class Bond:
    """A contact between two sites of a periodic structure.

    ``image`` is the lattice translation applied to ``atom2`` relative to
    ``atom1``: the bond runs from site ``atom1`` in the reference cell to
    site ``atom2`` shifted by ``image`` cells. (0, 0, 0) means both ends
    lie in the same cell.

    Cartesian endpoints are deliberately not stored. They are derived by
    the render model, which is the only component that knows which
    periodic images are actually being drawn.
    """

    atom1: int
    atom2: int
    image: tuple[int, int, int]
    length: float


def canonical_key(
    index_a: int,
    index_b: int,
    image: tuple[int, int, int],
) -> tuple[int, int, tuple[int, int, int]]:
    """Collapse the two directions of one contact onto a single key.

    A neighbour search reports i -> j across image I and j -> i across
    image -I. Ordering the indices without negating the image leaves both
    copies in the result, which double-counts and double-draws every bond
    that crosses a cell boundary.
    """
    if index_a <= index_b:
        return (index_a, index_b, tuple(image))

    return (index_b, index_a, tuple(-v for v in image))


def statistics(bonds) -> dict[str, float]:
    if not bonds:
        return {"count": 0, "min": 0.0, "max": 0.0, "avg": 0.0}

    lengths = [bond.length for bond in bonds]

    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
    }


@dataclass(slots=True)
class ShortContact:
    """A pair too close to be a bond, recorded rather than silently dropped.

    Almost always two alternative positions of one disordered atom. They
    must not be bonded: doing so draws a ~0.8 A stick between what is
    physically a single atom.
    """

    atom1: int
    atom2: int
    symbol1: str
    symbol2: str
    distance: float
    both_partial: bool


@dataclass(slots=True)
class BondingDiagnostics:
    """What the detector rejected, and why, so the UI can say so."""

    short_contacts: list = field(default_factory=list)
    partial_occupancy_sites: int = 0

    @property
    def short_contact_count(self) -> int:
        return len(self.short_contacts)

    @property
    def shortest_rejected(self) -> float | None:
        if not self.short_contacts:
            return None

        return min(c.distance for c in self.short_contacts)

    @property
    def suggests_disorder(self) -> bool:
        return self.partial_occupancy_sites > 0 or any(
            c.both_partial for c in self.short_contacts
        )


def site_occupancy(site) -> float:
    """Total occupancy of a site, tolerant of plain mappings."""
    species = site.species

    total = getattr(species, "num_atoms", None)

    if total is not None:
        return float(total)

    try:
        return float(sum(species.values()))
    except Exception:
        return 1.0
