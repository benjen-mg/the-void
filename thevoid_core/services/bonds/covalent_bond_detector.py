from __future__ import annotations

import logging

import numpy as np

from thevoid_core.data.element_radii import ElementRadiusProvider
from thevoid_core.services.bonds.bond_model import (
    Bond,
    BondingDiagnostics,
    ShortContact,
    canonical_key,
    site_occupancy,
)

logger = logging.getLogger(__name__)


class CovalentBondDetector:
    """Bond by summed covalent radii.

    Two atoms are bonded when their separation is no greater than the sum
    of their covalent radii plus a tolerance. This is the sole bonding
    model the application uses: it is deterministic, fast, and correct
    for the molecular and coordination-compound crystals this viewer
    targets, including hydrogens, which a coordination-environment model
    tends to miss.
    """

    #: Slack added to the summed radii, in angstrom.
    TOLERANCE = 0.4

    #: Lower bound as a fraction of the summed covalent radii.
    #:
    #: A flat floor does not work here. 0.4 A rejects only exact
    #: duplicates, so a disordered oxygen split over two sites 0.79 A
    #: apart sails through the O-O cutoff of 1.72 A and gets drawn as a
    #: bond roughly half the length of a real one. Scaling with the pair
    #: puts the floor at 0.92 A for O-O and 0.43 A for H-H, which
    #: rejects disorder alternatives while clearing every real bond:
    #: the shortest common ones are H-H in H2 at 0.74 A (floor 0.43),
    #: O-H at 0.96 A (floor 0.68) and C-H at 1.09 A (floor 0.75).
    MIN_BOND_FACTOR = 0.7

    #: Below this, two sites are the same atom written twice.
    ABSOLUTE_MIN_DISTANCE = 0.1

    #: Occupancy under this marks a site as partially occupied.
    FULL_OCCUPANCY = 0.99

    HYDROGEN = frozenset({"H", "D", "T"})

    def __init__(
        self,
        tolerance: float | None = None,
        radius_provider: ElementRadiusProvider | None = None,
    ) -> None:
        self.tolerance = self.TOLERANCE if tolerance is None else tolerance
        self.radius_provider = radius_provider or ElementRadiusProvider()

    def detect(self, structure) -> list[Bond]:
        return self.detect_with_diagnostics(structure)[0]

    def detect_with_diagnostics(
        self,
        structure,
    ) -> tuple[list[Bond], BondingDiagnostics]:
        symbols = [self._symbol(site) for site in structure.sites]
        occupancies = [site_occupancy(site) for site in structure.sites]

        diagnostics = BondingDiagnostics(
            partial_occupancy_sites=sum(
                1 for occ in occupancies if occ < self.FULL_OCCUPANCY
            )
        )

        radii = np.array(
            [self.radius_provider.get_covalent_radius(s) for s in symbols]
        )

        if not len(radii):
            return [], diagnostics

        search_radius = float(2.0 * radii.max() + self.tolerance)

        centers, neighbors, offsets, distances = structure.get_neighbor_list(
            search_radius
        )

        bonds: dict[tuple, Bond] = {}
        rejected: dict[tuple[int, int], ShortContact] = {}

        for i, j, offset, distance in zip(
            centers, neighbors, offsets, distances
        ):
            i = int(i)
            j = int(j)
            distance = float(distance)

            if distance < self.ABSOLUTE_MIN_DISTANCE:
                continue

            summed = radii[i] + radii[j]

            if distance > summed + self.tolerance:
                continue

            if distance < summed * self.MIN_BOND_FACTOR:
                pair = (i, j) if i <= j else (j, i)

                if pair not in rejected:
                    rejected[pair] = ShortContact(
                        atom1=pair[0],
                        atom2=pair[1],
                        symbol1=symbols[pair[0]],
                        symbol2=symbols[pair[1]],
                        distance=distance,
                        both_partial=(
                            occupancies[i] < self.FULL_OCCUPANCY
                            and occupancies[j] < self.FULL_OCCUPANCY
                        ),
                    )
                continue

            # Two hydrogens within range are geminal, not bonded;
            # drawing them puts a spurious edge across every CH2.
            if symbols[i] in self.HYDROGEN and symbols[j] in self.HYDROGEN:
                continue

            key = canonical_key(i, j, tuple(int(round(v)) for v in offset))

            if key in bonds:
                continue

            bonds[key] = Bond(
                atom1=i,
                atom2=j,
                image=tuple(int(round(v)) for v in offset),
                length=distance,
            )

        diagnostics.short_contacts = list(rejected.values())

        if diagnostics.short_contacts:
            logger.info(
                "Rejected %d contact(s) below the minimum bond distance; "
                "shortest %.3f A. This usually means disorder.",
                diagnostics.short_contact_count,
                diagnostics.shortest_rejected,
            )

        logger.info(
            "Covalent detector found %d bonds over %d sites.",
            len(bonds),
            len(structure),
        )

        return list(bonds.values()), diagnostics

    @staticmethod
    def _symbol(site) -> str:
        species = list(site.species.keys())

        return species[0].symbol if species else "X"
