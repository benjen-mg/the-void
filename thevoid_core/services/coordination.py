from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Neighbor:
    """One neighbouring atom in a coordination environment."""

    site_index: int
    symbol: str
    label: str
    distance: float
    image: tuple[int, int, int]

    #: Cartesian offset from the centre site to this neighbour (not
    #: normalised - divide by `distance` for a unit vector), used by
    #: OMS detection's coordination-geometry analysis. Optional and
    #: defaulted to None so every existing caller that builds a
    #: Neighbor without it (largely test fixtures checking other
    #: behaviour) keeps working unchanged.
    direction: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class CoordinationEnvironment:
    """The coordination environment of one site: everything CrystalNN
    considers bonded to it, sorted nearest first.
    """

    center_site_index: int
    center_label: str
    center_symbol: str
    neighbors: tuple[Neighbor, ...]

    @property
    def coordination_number(self) -> int:
        return len(self.neighbors)

    @property
    def average_bond_length(self) -> float:
        if not self.neighbors:
            return 0.0

        return sum(n.distance for n in self.neighbors) / len(self.neighbors)

    @property
    def min_bond_length(self) -> float:
        return min((n.distance for n in self.neighbors), default=0.0)

    @property
    def max_bond_length(self) -> float:
        return max((n.distance for n in self.neighbors), default=0.0)


def analyze_coordination_environment(
    structure,
    site_index: int,
    labels,
) -> CoordinationEnvironment:
    """Coordination environment of one site, via pymatgen's CrystalNN.

    A dedicated, on-demand analysis tool - not the application's
    bonding model for rendering, which is a separate, deliberately
    single, fixed covalent-radius rule (see services/bonds/__init__.py
    for why a second bonding *model* was rejected there). Coordination
    number, bond lengths, and neighbour identity are a different
    question from "should a stick be drawn here": CrystalNN's
    coordination-environment-aware algorithm is the right tool
    specifically for that question, particularly for the transition-
    and post-transition-metal framework nodes (Zn, Cu, Co, Ni, Fe, Zr,
    Ti, Al) this was written to handle well, where a fixed distance
    cutoff can both under- and over-count depending on the metal's
    oxidation state and coordination geometry.

    labels is the same index-aligned site-label list used everywhere
    else in the application (services.site_labels.compute_site_labels),
    so a neighbour's label always matches what the structure tree and
    the Selected Atom panel would show for that same site.
    """
    from pymatgen.analysis.local_env import CrystalNN

    nn = CrystalNN(
        distance_cutoffs=None,
        x_diff_weight=0,
        porous_adjustment=True,
    )

    center_site = structure[site_index]

    try:
        nn_info = nn.get_nn_info(structure, site_index)
    except Exception:
        logger.warning(
            "CrystalNN failed to compute the coordination environment "
            "for site %d; reporting zero neighbours.",
            site_index,
            exc_info=True,
        )
        nn_info = []

    neighbors = []

    for entry in nn_info:
        neighbor_index = entry.get("site_index")

        if neighbor_index is None:
            continue

        image = tuple(int(round(v)) for v in entry.get("image", (0, 0, 0)))

        distance = float(center_site.distance(entry["site"]))
        offset = entry["site"].coords - center_site.coords

        neighbors.append(
            Neighbor(
                site_index=neighbor_index,
                symbol=_symbol(structure[neighbor_index]),
                label=_label(labels, neighbor_index),
                distance=distance,
                image=image,
                direction=tuple(float(v) for v in offset),
            )
        )

    neighbors.sort(key=lambda neighbor: neighbor.distance)

    return CoordinationEnvironment(
        center_site_index=site_index,
        center_label=_label(labels, site_index),
        center_symbol=_symbol(center_site),
        neighbors=tuple(neighbors),
    )


def _symbol(site) -> str:
    species = list(site.species.keys())

    return species[0].symbol if species else "?"


def _label(labels, index: int) -> str:
    if 0 <= index < len(labels):
        return labels[index]

    return f"?{index + 1}"
