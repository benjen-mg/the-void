from __future__ import annotations

import itertools
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from thevoid_core.data.metal_classification import is_metal
from thevoid_core.services.bonds.bond_model import site_occupancy

logger = logging.getLogger(__name__)

HYDROGEN_SYMBOLS = frozenset({"H", "D", "T"})

Image = tuple[int, int, int]
Node = tuple[int, Image]


class CompletionMode(str, Enum):
    """How far past the reference cell to draw."""

    #: Only sites in the reference cell; bonds leaving it are hidden.
    CELL_ONLY = "cell_only"

    #: Follow bonds outward until every molecule is whole. The only
    #: mode that guarantees every drawn bond terminates in an atom.
    COMPLETE_MOLECULES = "complete_molecules"

    #: One shell of neighbours past the cell. The fallback for extended
    #: solids, where "complete the molecule" would never terminate.
    COORDINATION_SHELL = "coordination_shell"


@dataclass(frozen=True, slots=True)
class RenderAtom:
    symbol: str
    position: tuple[float, float, float]
    site_index: int
    image: Image

    @property
    def in_reference_cell(self) -> bool:
        return self.image == (0, 0, 0)


@dataclass(frozen=True, slots=True)
class RenderBond:
    atom_a: int
    atom_b: int
    length: float


@dataclass(frozen=True, slots=True)
class Fragment:
    """A connected component of the drawn graph."""

    formula: str
    atom_count: int
    occurrences: int

    @property
    def is_isolated_ion_or_solvent(self) -> bool:
        """Small and self-contained: a counter-ion or solvent molecule.

        Useful for answering "should this be floating unattached?" - a
        nitrate or a water legitimately is, a 20-atom fragment probably
        means the bonding model missed something.
        """
        return self.atom_count <= 6


@dataclass
class RenderModel:
    atoms: list[RenderAtom] = field(default_factory=list)
    bonds: list[RenderBond] = field(default_factory=list)
    mode: CompletionMode = CompletionMode.COMPLETE_MOLECULES

    #: True when molecule completion was abandoned because the structure
    #: is extended (a framework, a salt, a polymer) rather than molecular.
    fell_back: bool = False

    fragments: list[Fragment] = field(default_factory=list)

    #: Sites dropped because their occupancy fell below the threshold.
    hidden_partial_sites: int = 0

    #: Number of unit cells requested along a, b, c. (1, 1, 1) is a
    #: single cell.
    supercell: tuple[int, int, int] = (1, 1, 1)

    #: Decorative copies added at the outer faces of the displayed
    #: region for atoms that sit exactly on a cell boundary (see
    #: RenderModelBuilder.build(show_boundary_replicas=...)). These
    #: atoms have no bonds of their own - they exist purely so the
    #: displayed box looks complete at its edges, the way a printed
    #: packing diagram does.
    boundary_replica_count: int = 0

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def image_atom_count(self) -> int:
        return sum(1 for atom in self.atoms if not atom.in_reference_cell)

    @property
    def outside_supercell_count(self) -> int:
        """Atoms added by molecule completion, beyond the requested tiling.

        Distinct from image_atom_count, which only ever checks against
        the single cell (0, 0, 0): with a 2x2x2 supercell requested,
        most drawn atoms have a non-zero image and are not "extra" in
        any meaningful sense - they were asked for. This counts only
        the atoms that completion pulled in beyond that request.
        """
        nx, ny, nz = self.supercell

        return sum(
            1
            for atom in self.atoms
            if not (
                0 <= atom.image[0] < nx
                and 0 <= atom.image[1] < ny
                and 0 <= atom.image[2] < nz
            )
        )

    @property
    def fragment_count(self) -> int:
        return sum(f.occurrences for f in self.fragments)

    @property
    def bonded_atom_indices(self) -> set[int]:
        indices: set[int] = set()

        for bond in self.bonds:
            indices.add(bond.atom_a)
            indices.add(bond.atom_b)

        return indices

    @property
    def isolated_atom_indices(self) -> set[int]:
        """Atoms with no bond at all - a lone counter-ion, for instance.

        Used by the wireframe display style: wireframe draws bonds as
        thin lines and skips atom spheres entirely, but an atom with no
        bonds would then be completely invisible, so isolated atoms get
        a small sphere as a deliberate exception.
        """
        return set(range(len(self.atoms))) - self.bonded_atom_indices


class RenderModelBuilder:
    """Turn a structure plus its bond list into drawable geometry.

    The viewer previously drew spheres for the reference cell only, while
    drawing bonds all the way out to periodic-image neighbours. Every bond
    crossing a cell face therefore ended in empty space, and molecules
    straddling a boundary appeared as fragments. This builder decides
    which periodic images to materialise so that each drawn bond has an
    atom at both ends.
    """

    #: Ceiling on the completion walk. A molecular crystal settles well
    #: below this; an extended solid runs away and is caught by it.
    MAX_ATOMS = 20_000

    def build(
        self,
        structure,
        bonds,
        *,
        mode: CompletionMode = CompletionMode.COMPLETE_MOLECULES,
        show_hydrogens: bool = True,
        min_occupancy: float = 0.0,
        supercell: tuple[int, int, int] = (1, 1, 1),
        show_boundary_replicas: bool = False,
    ) -> RenderModel:
        nx, ny, nz = supercell

        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError(f"supercell dimensions must be >= 1: {supercell}")

        symbols = [self._symbol(site) for site in structure.sites]
        occupancies = [site_occupancy(site) for site in structure.sites]

        hidden = 0
        visible = []

        for symbol, occupancy in zip(symbols, occupancies):
            if not show_hydrogens and symbol in HYDROGEN_SYMBOLS:
                visible.append(False)
                continue

            if occupancy < min_occupancy:
                visible.append(False)
                hidden += 1
                continue

            visible.append(True)

        adjacency = self._adjacency(bonds, visible)

        # A supercell request seeds every cell in the tiling, not just
        # (0, 0, 0). Completion then runs from all of them at once, so a
        # molecule that straddles the boundary between tile (0,0,0) and
        # tile (1,0,0) is discovered once and shared rather than drawn
        # twice - the walk simply reaches the same node from either side
        # and _grow's "if neighbor in seen" check absorbs the overlap.
        seeds: list[Node] = [
            (index, (i, j, k))
            for index in range(len(structure))
            if visible[index]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
        ]

        fell_back = False

        if mode is CompletionMode.CELL_ONLY:
            nodes = list(seeds)
        elif mode is CompletionMode.COORDINATION_SHELL:
            nodes = self._grow(seeds, adjacency, max_depth=1)
        else:
            nodes = self._grow(seeds, adjacency, max_depth=None)

            if nodes is None:
                # Unbounded completion didn't terminate - the bond
                # graph percolates through the lattice with no finite
                # "whole molecule" to find, the situation a genuinely
                # 3-periodic coordination framework (a MOF) is always
                # in: every linker connects to a metal, which connects
                # to another linker, forever. Before dropping all the
                # way to a one-shell view, try once more with metal
                # atoms treated as walk boundaries - a metal atom's own
                # bonds are still fully drawn, it just never becomes a
                # launching point for discovering a new periodic image
                # nobody has reached yet. Organic linkers are finite
                # between their metal-coordination points even in an
                # infinite framework, so this still completes every
                # ring and chain in full while actually terminating.
                # Confirmed directly against a real 3-periodic MOF
                # file that this unbounded attempt cannot complete on
                # its own.
                is_metal_site = [
                    is_metal(symbol) for symbol in symbols
                ]

                metal_bounded_nodes = self._grow(
                    seeds,
                    adjacency,
                    max_depth=None,
                    stop_expansion_at=lambda site_index: is_metal_site[
                        site_index
                    ],
                )

                # Only trust this if it actually grew something beyond
                # the bare seeds. A purely metallic or ionic extended
                # solid - where the percolating bonds are themselves
                # metal-to-metal or metal-to-non-bridging-non-metal,
                # not organic linkers bridging metal nodes - degenerates
                # under this rule to just the seeds and no bonds at
                # all, which is a worse result than the coordination
                # shell this would otherwise fall back to, not a
                # genuine completion.
                if (
                    metal_bounded_nodes is not None
                    and len(metal_bounded_nodes) > len(seeds)
                ):
                    nodes = metal_bounded_nodes
                else:
                    nodes = None

                if nodes is None:
                    logger.info(
                        "Structure is extended, not molecular, and "
                        "stopping at metal centres still did not "
                        "terminate; falling back to a single "
                        "coordination shell."
                    )
                    nodes = self._grow(seeds, adjacency, max_depth=1)
                    fell_back = True

        if nodes is None:
            # Even one shell blew the budget: a very large cell with dense
            # coordination. Draw the requested cells and nothing more.
            logger.warning(
                "Coordination shell exceeded %d atoms; "
                "drawing the requested cell(s) only.",
                self.MAX_ATOMS,
            )
            nodes = list(seeds)
            fell_back = True

        model = self._assemble(
            structure,
            symbols,
            adjacency,
            nodes,
            mode,
            fell_back,
            hidden,
            supercell,
        )

        if show_boundary_replicas:
            self._add_boundary_replicas(structure, model, supercell)

        return model

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    @staticmethod
    def _adjacency(bonds, visible) -> dict[int, list[tuple[int, Image, float]]]:
        """Both directions of every bond, keyed by originating site.

        A bond stored as atom1 -> atom2 across image I is also traversable
        as atom2 -> atom1 across image -I.
        """
        adjacency: dict[int, list[tuple[int, Image, float]]] = defaultdict(list)

        for bond in bonds:
            if not (visible[bond.atom1] and visible[bond.atom2]):
                continue

            image = tuple(int(v) for v in bond.image)
            reverse = tuple(-v for v in image)

            adjacency[bond.atom1].append((bond.atom2, image, bond.length))
            adjacency[bond.atom2].append((bond.atom1, reverse, bond.length))

        return adjacency

    def _grow(
        self,
        seeds: list[Node],
        adjacency,
        *,
        max_depth: int | None,
        stop_expansion_at=None,
    ) -> list[Node] | None:
        """Breadth-first walk outward from the reference cell.

        Returns None when the walk exceeds MAX_ATOMS, which means the
        bond graph percolates through the lattice and no finite set of
        images will ever complete a "molecule".

        stop_expansion_at, if given, is a predicate on a site index: a
        node whose site it accepts is still fully included (all of its
        bonds to already-known atoms are drawn - see _assemble, which
        includes a bond whenever both endpoints ended up in the final
        node set, regardless of which node "discovered" the other), it
        is just never used as a launching point to pull in a periodic
        image of some other atom that nothing has reached yet. This is
        what lets a genuinely 3-periodic MOF's organic linkers complete
        in full (every seed - including every ligand atom - still
        expands normally) while the walk as a whole terminates instead
        of chasing the infinite lattice outward through its metal
        centres forever.
        """
        seen: dict[Node, int] = {node: 0 for node in seeds}
        queue: deque[Node] = deque(seeds)

        while queue:
            site_index, image = queue.popleft()
            depth = seen[(site_index, image)]

            if max_depth is not None and depth >= max_depth:
                continue

            if stop_expansion_at is not None and stop_expansion_at(
                site_index
            ):
                continue

            for neighbor_index, offset, _ in adjacency.get(site_index, ()):
                neighbor: Node = (
                    neighbor_index,
                    (
                        image[0] + offset[0],
                        image[1] + offset[1],
                        image[2] + offset[2],
                    ),
                )

                if neighbor in seen:
                    continue

                if len(seen) >= self.MAX_ATOMS:
                    return None

                seen[neighbor] = depth + 1
                queue.append(neighbor)

        return list(seen)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _assemble(
        self,
        structure,
        symbols,
        adjacency,
        nodes: list[Node],
        mode: CompletionMode,
        fell_back: bool,
        hidden_partial_sites: int,
        supercell: tuple[int, int, int],
    ) -> RenderModel:
        lattice = np.asarray(structure.lattice.matrix, dtype=float)

        index_of: dict[Node, int] = {}
        atoms: list[RenderAtom] = []

        for node in nodes:
            site_index, image = node

            position = (
                np.asarray(structure[site_index].coords, dtype=float)
                + np.asarray(image, dtype=float) @ lattice
            )

            index_of[node] = len(atoms)

            atoms.append(
                RenderAtom(
                    symbol=symbols[site_index],
                    position=tuple(float(v) for v in position),
                    site_index=site_index,
                    image=image,
                )
            )

        bonds: list[RenderBond] = []
        emitted: set[tuple[int, int]] = set()

        for node in nodes:
            site_index, image = node
            a = index_of[node]

            for neighbor_index, offset, length in adjacency.get(site_index, ()):
                other: Node = (
                    neighbor_index,
                    (
                        image[0] + offset[0],
                        image[1] + offset[1],
                        image[2] + offset[2],
                    ),
                )

                b = index_of.get(other)

                # A bond is drawn only when both ends were materialised.
                # This is what keeps sticks from running off into space
                # at the edge of whatever was drawn.
                if b is None:
                    continue

                key = (a, b) if a <= b else (b, a)

                if key in emitted:
                    continue

                emitted.add(key)
                bonds.append(RenderBond(atom_a=a, atom_b=b, length=length))

        return RenderModel(
            atoms=atoms,
            bonds=bonds,
            mode=mode,
            fell_back=fell_back,
            fragments=self._fragments(atoms, bonds),
            hidden_partial_sites=hidden_partial_sites,
            supercell=supercell,
        )

    #: Fractional-coordinate distance from 0 or 1 within which a site
    #: is treated as sitting exactly on a cell face.
    BOUNDARY_TOLERANCE = 1e-3

    def _add_boundary_replicas(
        self,
        structure,
        model: RenderModel,
        supercell: tuple[int, int, int],
    ) -> None:
        """Decorate cell-boundary atoms with copies at the outer faces.

        A crystallographic unit cell is conventionally drawn as a closed
        box: an atom sitting exactly on a face appears once at that
        face, in the *next* cell over. A single displayed cell, viewed
        alone, then looks like it is missing atoms at whichever faces
        weren't the "owning" side - a corner atom shows once instead of
        at all eight corners of the box, a common surprise for anyone
        used to a printed packing diagram.

        This does not change chemistry: no atom is added or removed,
        only extra draw copies of atoms already in the model. No bonds
        are drawn to these copies - correctly threading bonds through
        every combination of mirrored replicas is substantially more
        work than the decorative value justifies, and every real bond
        is already fully represented by the un-mirrored atom the
        replica copies. Mutates model in place, appending to model.atoms
        (safe: existing RenderBond indices point at earlier positions in
        the list and are unaffected by appending).
        """
        nx, ny, nz = supercell
        dims = (nx, ny, nz)

        lattice = np.asarray(structure.lattice.matrix, dtype=float)
        tol = self.BOUNDARY_TOLERANCE

        existing: set[tuple[int, Image]] = {
            (atom.site_index, atom.image) for atom in model.atoms
        }

        replicas: list[RenderAtom] = []

        for atom in model.atoms:
            image = atom.image

            # Only decorate atoms that are part of the requested tiling.
            # Atoms added purely by molecule completion beyond it sit at
            # arbitrary images with no well-defined "outer face" of
            # their own to decorate.
            if not all(0 <= image[axis] < dims[axis] for axis in range(3)):
                continue

            frac = structure[atom.site_index].frac_coords

            # Axes where this atom sits on a face, and which face: a
            # low-face atom (frac ~ 0) only needs a far-side replica if
            # it's already in the last tile along that axis - an
            # earlier tile's own low-face atom is exactly where the
            # previous tile's high-face replica would land anyway, so
            # it needs no replica of its own (see the module-level note
            # in RenderModelBuilder's docstring on avoiding double
            # counting). Symmetrically for a high-face atom in the
            # first tile.
            axis_targets: list[tuple[int, int]] = []

            for axis in range(3):
                value = float(frac[axis]) % 1.0

                if value < tol and image[axis] == dims[axis] - 1:
                    axis_targets.append((axis, dims[axis]))
                elif value > 1.0 - tol and image[axis] == 0:
                    axis_targets.append((axis, -1))

            if not axis_targets:
                continue

            # Every non-empty combination: a face atom needs one
            # replica, an edge atom needs three (two faces plus the
            # edge itself), a corner atom needs seven.
            for size in range(1, len(axis_targets) + 1):
                for combo in itertools.combinations(axis_targets, size):
                    new_image = list(image)

                    for axis, target in combo:
                        new_image[axis] = target

                    new_image = tuple(new_image)
                    key = (atom.site_index, new_image)

                    if key in existing:
                        continue

                    existing.add(key)

                    position = (
                        np.asarray(
                            structure[atom.site_index].coords, dtype=float
                        )
                        + np.asarray(new_image, dtype=float) @ lattice
                    )

                    replicas.append(
                        RenderAtom(
                            symbol=atom.symbol,
                            position=tuple(float(v) for v in position),
                            site_index=atom.site_index,
                            image=new_image,
                        )
                    )

        model.atoms.extend(replicas)
        model.boundary_replica_count = len(replicas)

    @staticmethod
    def _symbol(site) -> str:
        species = list(site.species.keys())

        return species[0].symbol if species else "X"

    @staticmethod
    def _fragments(
        atoms: list[RenderAtom],
        bonds: list[RenderBond],
    ) -> list[Fragment]:
        """Group the drawn atoms into connected components.

        This is what answers "should that bit be floating on its own?".
        A nitrate or a water sitting unattached is chemically correct; a
        large fragment adrift usually means the bonding model missed
        something. Counting them lets the UI say which case it is
        instead of leaving it to the eye.
        """
        if not atoms:
            return []

        neighbors: dict[int, list[int]] = defaultdict(list)

        for bond in bonds:
            neighbors[bond.atom_a].append(bond.atom_b)
            neighbors[bond.atom_b].append(bond.atom_a)

        seen = [False] * len(atoms)
        tally: dict[tuple[str, int], int] = defaultdict(int)

        for start in range(len(atoms)):
            if seen[start]:
                continue

            counts: dict[str, int] = defaultdict(int)
            stack = [start]
            seen[start] = True
            size = 0

            while stack:
                current = stack.pop()
                counts[atoms[current].symbol] += 1
                size += 1

                for neighbor in neighbors.get(current, ()):
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        stack.append(neighbor)

            tally[(_hill_formula(counts), size)] += 1

        return [
            Fragment(formula=formula, atom_count=size, occurrences=count)
            for (formula, size), count in sorted(
                tally.items(), key=lambda item: (-item[0][1], item[0][0])
            )
        ]


def _hill_formula(counts: dict[str, int]) -> str:
    """Hill notation: C first, H second, everything else alphabetical."""
    ordered: list[str] = []

    if "C" in counts:
        ordered.append("C")

        if "H" in counts:
            ordered.append("H")

        ordered += sorted(s for s in counts if s not in ("C", "H"))
    else:
        ordered += sorted(counts)

    return "".join(
        f"{symbol}{counts[symbol] if counts[symbol] > 1 else ''}"
        for symbol in ordered
    )

