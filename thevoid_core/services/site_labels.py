from __future__ import annotations

from collections import defaultdict


def compute_site_labels(structure) -> list[str]:
    """Per-element display labels for every site, in site order.

    Counters restart per element, so a quartz cell reads Si1 Si2 Si3 O1
    O2 ... rather than numbering every site globally - what a
    crystallographer coming from any standard viewer expects, and what
    the structure tree has always shown.

    This is the single source of truth for "what is site N called" - it
    exists so the structure tree, the 3D-picked atom panel, and
    anything else that names a site by index all agree with each other.
    Previously the tree computed this inline and nothing else needed
    it; now that a 3D pick has to report the same label a tree row
    shows, duplicating the loop risked the two drifting apart.
    """
    counters: dict[str, int] = defaultdict(int)
    labels: list[str] = []

    for site in structure.sites:
        species = list(site.species.keys())
        element = species[0].symbol if species else "?"

        counters[element] += 1
        labels.append(f"{element}{counters[element]}")

    return labels
