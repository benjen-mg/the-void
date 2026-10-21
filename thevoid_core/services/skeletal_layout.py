from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FragmentDiagram:
    """One chemically meaningful fragment, ready to render: an organic
    linker shown in full, or a metal centre shown together with its
    directly-coordinating atoms (its local coordination environment).

    occurrence_count is how many symmetry-equivalent copies of this
    exact fragment exist in the unit cell - a real MOF unit cell very
    commonly contains the same linker or the same metal coordination
    environment repeated many times over (CALF-20, for instance, has
    17 chemically identical triazolate linkers), and showing every
    single copy as its own separate picture would mean a long diagram
    of the same few pictures repeated, not a more informative one. One
    representative per distinct fragment is shown instead, with this
    count carried through so a caller can note how many copies it
    actually represents.
    """

    label: str
    mol: object  # rdkit.Chem.Mol, with 2D coordinates already computed
    occurrence_count: int = 1


@dataclass(frozen=True, slots=True)
class SkeletalDiagramResult:
    fragments: tuple[FragmentDiagram, ...]

    #: Fragments that raised during sanitisation or layout and were
    #: skipped rather than failing the whole diagram - counted, not
    #: silently dropped, so a caller can report "N fragments could not
    #: be drawn" rather than just showing fewer pictures than expected
    #: with no explanation.
    skipped_count: int


def build_fragment_mol(symbols, bonds, atom_indices):
    """Build an RDKit RWMol from a subset of a larger structure's atoms
    (atom_indices) and whichever bonds connect two atoms both inside
    that subset.

    Every bond that crosses the subset's boundary - one endpoint
    inside atom_indices, the other outside it - gets capped with a
    dummy atom (RDKit atomic number 0, drawn as "*") on the inside
    endpoint, one dummy per such bond, rather than being silently
    dropped. This matters for chemical honesty: RDKit's own
    sanitisation fills in whatever valence is "missing" on an atom
    with implicit hydrogens, and for a nitrogen that in the real
    structure coordinates to a metal centre but has that bond simply
    omitted here, the result is a nitrogen that looks like a plain
    amine (an extra, invented N-H) rather than what it actually is - a
    ring nitrogen whose real fourth connection continues outside this
    fragment. Confirmed directly, not assumed: building a triazolate
    linker's own fragment without capping produced exactly this -
    coordinating nitrogens rendered as if freely N-H terminated. A
    dummy-atom cap says "something real continues here, not drawn",
    which is both correct and the standard cheminformatics convention
    for a truncated attachment point.

    Every boundary crossed by these fragments (linkers and metal nodes
    are split apart at exactly the metal/non-metal edge - see
    mof_analysis.partition_framework) is a metal-ligand coordination
    bond on the far side, never an ordinary organic bond. Capping it
    with an ordinary single bond, as if it were ready to continue as
    plain covalent connectivity, overfills the donor atom's valence
    the moment that atom's in-fragment bonds already account for its
    full ordinary valence - confirmed directly against a real file: a
    pyridyl-type nitrogen with two ring bonds plus two such single-bond
    caps (one per coordinating metal centre) raised RDKit's "explicit
    valence ... greater than permitted", silently discarding that
    entire linker (and every node+linkers cluster built from it) from
    the diagram rather than just the one metal-only "node" fragment
    that never carries this cap at all. A zero-order bond
    (Chem.BondType.ZERO) is RDKit's own convention for exactly this
    case - a real connection that must not count toward valence - and
    is used here whenever the atom on the far side of the cut is a
    metal; every other boundary (an ordinary organic bond truncated by
    the fragment's own extent) keeps the plain single bond it always
    had.
    """
    from rdkit import Chem

    from thevoid_core.data.metal_classification import is_metal

    index_set = set(atom_indices)
    local_index = {global_i: i for i, global_i in enumerate(atom_indices)}

    mol = Chem.RWMol()

    for global_i in atom_indices:
        mol.AddAtom(Chem.Atom(symbols[global_i]))

    seen_internal_bonds = set()

    for bond in bonds:
        a_inside = bond.atom_a in index_set
        b_inside = bond.atom_b in index_set

        if a_inside and b_inside:
            a, b = local_index[bond.atom_a], local_index[bond.atom_b]
            if a == b:
                continue
            key = (min(a, b), max(a, b))
            if key in seen_internal_bonds:
                continue
            mol.AddBond(a, b, Chem.BondType.SINGLE)
            seen_internal_bonds.add(key)

        elif a_inside or b_inside:
            inside_global = bond.atom_a if a_inside else bond.atom_b
            outside_global = bond.atom_b if a_inside else bond.atom_a
            inside_local = local_index[inside_global]
            dummy = mol.AddAtom(Chem.Atom(0))
            cap_bond_type = (
                Chem.BondType.ZERO
                if is_metal(symbols[outside_global])
                else Chem.BondType.SINGLE
            )
            mol.AddBond(inside_local, dummy, cap_bond_type)

    return mol


def expand_metal_node_atom_indices(node_atom_indices, bonds) -> tuple[int, ...]:
    """A metal node's own atom_indices (the metal itself, and any
    bridging atom for a multi-metal cluster) plus every atom directly
    bonded to any of them - the metal's immediate coordination sphere.

    Showing just the bare metal (or metal cluster) alone would not be
    a meaningful "coordination environment" diagram - the whole point
    is seeing what it is actually bonded to, the same way a real
    published figure of a metal centre's coordination environment
    always shows the directly-coordinating atoms around it, not the
    metal in isolation.
    """
    own = set(node_atom_indices)
    expanded = set(own)

    for bond in bonds:
        if bond.atom_a in own:
            expanded.add(bond.atom_b)
        if bond.atom_b in own:
            expanded.add(bond.atom_a)

    return tuple(sorted(expanded))


def build_node_cluster_atom_indices(
    node_atom_indices, linkers, bonds
) -> tuple[int, ...]:
    """A metal node's own atoms, plus every linker directly bonded to
    it, included in full - not just the linker's first, connecting
    atom the way expand_metal_node_atom_indices's single-shell
    expansion does, but the whole linker, so the diagram shows real
    connectivity between a metal centre and its complete surrounding
    linkers: a genuine "node with its linkers attached" picture, not a
    metal with a ring of isolated stubs around it.

    Any bond leading further still - from one of these linkers onward
    to a *different* metal node not included here - is left for
    build_fragment_mol's own dummy-atom capping to mark, the same as
    everywhere else a fragment's boundary is drawn: this is one node's
    local neighbourhood, not the whole periodic network.
    """
    node_atoms = set(node_atom_indices)
    combined = set(node_atoms)

    for linker in linkers:
        linker_atoms = set(linker.atom_indices)

        directly_connected = any(
            (bond.atom_a in node_atoms and bond.atom_b in linker_atoms)
            or (bond.atom_b in node_atoms and bond.atom_a in linker_atoms)
            for bond in bonds
        )

        if directly_connected:
            combined |= linker_atoms

    return tuple(sorted(combined))


def _build_and_sanitize_fragment(symbols, bonds, atom_indices, label):
    """Build and sanitise one fragment, without yet computing 2D
    coordinates - split out from full layout so duplicate fragments
    (very common in a MOF unit cell - see FragmentDiagram's own
    occurrence_count docstring) can be detected and collapsed before
    paying for layout on every single copy.

    Returns None (rather than raising) if sanitisation fails, so one
    problematic fragment does not prevent every other fragment in the
    diagram from being generated.
    """
    from rdkit import Chem

    try:
        mol = build_fragment_mol(symbols, bonds, atom_indices)
        Chem.SanitizeMol(mol)
    except Exception:
        logger.warning(
            "Could not build fragment %r; skipping just this fragment "
            "rather than the whole diagram.",
            label,
            exc_info=True,
        )
        return None

    return mol


def _canonical_key(mol) -> str:
    """A structural identity key for deduplication - two fragments with
    this same key are the same molecule (same atoms, same bonds,
    including the dummy-atom caps marking where each one is cut off),
    not merely the same formula.
    """
    from rdkit import Chem

    return Chem.MolToSmiles(mol, canonical=True)


def compute_skeletal_diagram(symbols, bonds, linkers, metal_nodes) -> SkeletalDiagramResult:
    """Build a standardised 2D skeletal diagram for one MOF unit cell's
    full nodes and linkers - as separate, individually clean fragment
    diagrams, not one flattened whole-framework graph, and with
    symmetry-equivalent copies of the same fragment collapsed into one
    representative picture rather than repeated.

    Three kinds of fragment are produced, all through the same
    deduplication: each distinct linker on its own; each distinct
    metal centre with just its immediate coordination sphere (one bond
    out); and each distinct metal centre together with every linker
    directly bonded to it, each shown in full - a genuine "node with
    its linkers attached" picture (build_node_cluster_atom_indices),
    not just a metal ringed by isolated stubs. Confirmed directly
    against a real structure that this connected view lays out cleanly
    (a Zn centre with its three coordinating linkers rendered as one
    clear, readable diagram) even though the *full* periodic network
    does not - it is small enough, and simple enough in topology (a
    star around one metal, not a many-times-bridged cycle), to be
    exactly the scale a standard 2D layout algorithm can still handle.

    The fragment-based design is deliberate, confirmed necessary rather
    than assumed: a MOF's periodic framework, even truncated to one
    cell, is typically one densely cross-linked graph once every
    metal-ligand coordination bond is included (many linkers all
    bridging between several shared metal centres) - a topology that
    does not have a clean planar embedding, and no 2D layout algorithm
    tried while building this (a distance-preserving graph layout,
    RDKit's default depiction, RDKit's CoordGen engine specifically
    built for macrocycles) produced anything readable for it as a
    single connected diagram; all three were tried directly against a
    real structure before this design was settled on. Individual
    fragments - one organic linker, or one metal centre with its
    immediate coordination sphere or its full connected linkers - are
    exactly the scale at which a real, standard, readable skeletal
    diagram is achievable, and are also what real MOF literature
    actually shows when depicting these structures in 2D: the linker
    on its own, and the metal's local coordination geometry, not the
    whole infinite network flattened.

    The deduplication is equally deliberate: a real unit cell very
    often contains the same linker or the same metal coordination
    environment repeated many times by symmetry (confirmed directly
    against CALF-20, whose real unit cell contains 17 chemically
    identical triazolate linkers) - showing every single copy as its
    own picture produces a long diagram of the same few pictures
    repeated, not a more informative one. Two fragments are treated as
    the same one if they produce the same canonical SMILES (including
    the dummy-atom caps marking where each is cut off, so "the same
    linker cut at the same points" is required, not merely "the same
    formula").

    symbols, bonds: the periodic graph's own symbol list and bond list
    (as build_periodic_graph_for_mof_analysis and partition_framework
    already produce and consume elsewhere in this application).
    linkers, metal_nodes: mof_analysis.FrameworkPartition's own output -
    reused directly rather than re-deriving a different decomposition,
    so this diagram's fragments are exactly the same linkers and nodes
    MOF Analysis itself reports.
    """
    from rdkit.Chem import rdDepictor

    groups: dict[str, list] = {}
    group_order: list[str] = []
    skipped = 0

    for linker in linkers:
        mol = _build_and_sanitize_fragment(
            symbols, bonds, linker.atom_indices, f"Linker ({linker.formula})"
        )
        if mol is None:
            skipped += 1
            continue

        key = _canonical_key(mol)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append((mol, f"Linker ({linker.formula})"))

    for node in metal_nodes:
        expanded = expand_metal_node_atom_indices(node.atom_indices, bonds)
        mol = _build_and_sanitize_fragment(
            symbols, bonds, expanded, f"{node.formula} node"
        )
        if mol is None:
            skipped += 1
            continue

        key = _canonical_key(mol)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append((mol, f"{node.formula} node"))

    for node in metal_nodes:
        cluster_atoms = build_node_cluster_atom_indices(
            node.atom_indices, linkers, bonds
        )
        label = f"{node.formula} node + linkers"
        mol = _build_and_sanitize_fragment(symbols, bonds, cluster_atoms, label)
        if mol is None:
            skipped += 1
            continue

        key = _canonical_key(mol)
        if key not in groups:
            groups[key] = []
            group_order.append(key)
        groups[key].append((mol, label))

    fragments = []

    for key in group_order:
        members = groups[key]
        representative_mol, label = members[0]

        try:
            rdDepictor.Compute2DCoords(representative_mol)
        except Exception:
            logger.warning(
                "Could not lay out fragment %r; skipping just this "
                "fragment rather than the whole diagram.",
                label,
                exc_info=True,
            )
            skipped += len(members)
            continue

        fragments.append(
            FragmentDiagram(
                label=label,
                mol=representative_mol,
                occurrence_count=len(members),
            )
        )

    return SkeletalDiagramResult(
        fragments=tuple(fragments), skipped_count=skipped
    )
