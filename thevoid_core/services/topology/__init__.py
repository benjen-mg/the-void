"""Topology: the underlying net of a framework and its RCSR name
(round 53).

    from thevoid_core.services.topology import analyze_topology
    result = analyze_topology(structure, bonds)
    result.rcsr         # "tbo", or None
    result.summary      # "tbo (3,4-c)", "pcu, 2-fold interpenetrated", ...

net.py holds periodic nets and their exact canonical comparison,
library.py the reference nets, deconstruct.py the structure -> net
step. This module ties them together and is what the Structure panel,
reports and the command line call. Nothing here imports Qt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

METHOD_NOTE = (
    "single-node deconstruction (SBUs = metals + mu-bridging atoms + "
    "metal-bound carboxylate groups; linkers with >= 3 connections are "
    "nodes, 2-connected pieces become edges; zeolite-like frameworks as "
    "T-atom nets; infinite SBUs split at their metals as CrystalNets does), "
    "then exact net identification: barycentric placement "
    "in rational arithmetic, reduction to the minimal cell, and a "
    "Systre-style canonical isomorphism test against RCSR reference nets"
)


@dataclass
class NetComponent:
    """One connected piece of the underlying net (several for an
    interpenetrated framework)."""

    dimensionality: int
    rcsr: str | None
    #: Vertices in the net's primitive cell.
    vertices: int
    #: Sorted distinct coordination numbers, e.g. (3, 4).
    connectivity: tuple
    #: Distinct coordination sequences (first 10 shells).
    coordination_sequences: list
    status: str  # "identified", "not_in_library", "unstable", "too_complex", "chain"
    description: str = ""


@dataclass
class TopologyResult:
    #: "identified", "not_in_library", "rod_sbu", "periodic_linker",
    #: "no_metal", "no_net", "unstable", "failed"
    status: str
    rcsr: str | None = None
    representation: str = "single-node"
    interpenetration: int = 0
    components: list = field(default_factory=list)
    #: [{"kind", "formula", "connectivity", "count"}]
    vertex_kinds: list = field(default_factory=list)
    edge_formulas: dict = field(default_factory=dict)
    dropped_formulas: dict = field(default_factory=dict)
    merged_edges: int = 0
    library: str = ""
    message: str = ""
    notes: list = field(default_factory=list)
    #: Round 70: infinite SBUs split at their metals,
    #: [{"formula", "dimensionality", "metals", "count"}], and for such a
    #: framework its all-node net too ({"status", "rcsr", "summary",
    #: "message"}; CoRE MOF 2024's topology_all_node).
    rod_sbus: list = field(default_factory=list)
    all_node: dict | None = None

    @property
    def connectivity_label(self) -> str:
        values = sorted({c for comp in self.components for c in comp.connectivity})
        if not values:
            return ""
        if len(values) == 1:
            return f"{values[0]}-c"
        return "(" + ",".join(str(v) for v in values) + ")-c"

    @property
    def summary(self) -> str:
        """One line for the Structure panel and the summary CSV."""
        if self.status == "identified":
            text = f"{self.rcsr} {self.connectivity_label}".strip()
            if self.interpenetration > 1:
                text += f", {self.interpenetration}-fold interpenetrated"
            return text
        if self.status == "not_in_library":
            if self.rod_sbus:
                return (f"{self.connectivity_label} net of split rod SBUs, "
                        "not in the reference library").strip()
            return f"{self.connectivity_label} net, not in the reference library".strip()
        return {
            "rod_sbu": "not identified: rod/layer SBU",
            "periodic_linker": "not identified: periodic organic part",
            "no_metal": "not identified: metal-free framework",
            "no_net": "no 2D/3D net (molecular or chain)",
            "unstable": "not identified: unstable net",
            "failed": "not identified: analysis failed",
        }.get(self.status, self.status)


def analyze_topology(
    structure,
    bonds=None,
    *,
    library=None,
    is_metal=None,
    bond_detector=None,
) -> TopologyResult:
    """The underlying net of structure, named by RCSR code where the
    reference library has it. Never raises for a structure it cannot
    handle: the result's status and message say why."""
    from thevoid_core.services.raspa_export import prepare_framework
    from thevoid_core.services.topology.deconstruct import ALL_NODE, SHAPES, deconstruct
    from thevoid_core.services.topology.library import default_library

    if is_metal is None:
        from thevoid_core.data.metal_classification import is_metal

    library = library or default_library()

    try:
        prepared = prepare_framework(
            structure, bonds, remove_free_molecules=True, bond_detector=bond_detector
        )
    except ValueError as exc:
        return TopologyResult(status="failed", message=str(exc), library=library.description)

    notes = list(prepared.notes)
    if prepared.removed:
        removed = ", ".join(f"{m.count} x {m.formula}" for m in prepared.removed)
        notes.append(f"Free (non-bonded) molecules ignored: {removed}.")

    decon = deconstruct(prepared.structure, prepared.bonds, is_metal)
    result = _identify(decon, library, notes)

    if decon.rods:
        shapes = ", ".join(f"{r['formula']} ({SHAPES[r['dimensionality']]})" for r in decon.rods)
        result.rod_sbus = [dict(r) for r in decon.rods]
        result.notes.append(
            f"Infinite SBU split at its metals, as CrystalNets and CoRE MOF 2024 do: {shapes}. "
            "Each metal is a vertex, joined along the rod through the atoms it shares with "
            "the next; carboxylates and other organic parts of the rod go with the linkers.")
        # The all-node net as well (CoRE's topology_all_node): every
        # carboxylate group of a linker is its own vertex.
        try:
            other = _identify(deconstruct(prepared.structure, prepared.bonds, is_metal,
                                          representation=ALL_NODE), library, [])
            result.all_node = {"status": other.status, "rcsr": other.rcsr,
                               "summary": other.summary, "message": other.message}
        except Exception as exc:  # the single-node result stands on its own
            logger.debug("All-node net failed.", exc_info=True)
            result.all_node = {"status": "failed", "rcsr": None,
                               "summary": "not identified: analysis failed",
                               "message": f"{type(exc).__name__}: {exc}"}

    return result


def _identify(decon, library, notes) -> TopologyResult:
    """The TopologyResult of one deconstruction: its net's components
    identified against library."""
    from thevoid_core.services.topology.net import (
        UnstableNetError,
        connected_components,
        coordination_sequences,
        prepare,
    )

    result = TopologyResult(
        status=decon.status,
        representation=decon.representation,
        edge_formulas=decon.edge_formulas,
        dropped_formulas=decon.dropped_formulas,
        merged_edges=decon.merged_edges,
        library=library.description,
        message=decon.message,
        notes=notes,
    )

    if decon.status != "ok":
        return result

    degrees = decon.net.degrees()
    result.vertex_kinds = [
        {"kind": k.kind, "formula": k.formula, "connectivity": k.connectivity, "count": k.count}
        for k in decon.vertex_kinds(degrees)
    ]

    if decon.merged_edges:
        result.notes.append(
            f"{decon.merged_edges} parallel edge(s) merged: two linkers "
            "joining the same pair of nodes count as one edge of the net."
        )

    periodic = []

    for component, order in connected_components(decon.net):
        if component.dimension == 0:
            continue
        if component.dimension == 1:
            result.components.append(NetComponent(
                1, None, component.n_vertices, tuple(sorted(set(component.degrees()))),
                [], "chain",
            ))
            continue
        periodic.append((component, order))

    if not periodic:
        result.status = "no_net"
        result.message = "The net has no 2- or 3-periodic component (chains only)."
        return result

    names = set()

    for component, order in periodic:
        entry = NetComponent(
            component.dimension, None, component.n_vertices,
            tuple(sorted(set(component.degrees()))), [], "not_in_library",
        )
        try:
            query = prepare(component)
            entry.vertices = query.net.n_vertices
            entry.coordination_sequences = sorted(set(query.sequences))
            matches = library.identify(query)
            if matches:
                entry.rcsr = matches[0].name
                entry.description = matches[0].description
                entry.status = "identified"
                if len(matches) > 1:
                    result.notes.append(
                        "Several library entries match: "
                        + ", ".join(m.name for m in matches)
                    )
        except UnstableNetError as exc:
            entry.status = "unstable"
            entry.coordination_sequences = sorted(set(coordination_sequences(component)))
            kinds = sorted({
                decon.vertex_labels[order[i]][1]
                for group in exc.collisions for i in group
            })
            result.notes.append(
                f"{sum(len(g) for g in exc.collisions)} vertices fall on top of "
                f"another in the barycentric placement ({', '.join(kinds)}): "
                "nodes with exactly the same neighbours, such as the pairs of "
                "paddlewheels in rht-type MOFs."
            )
        except ValueError as exc:
            entry.status = "too_complex"
            result.notes.append(f"Net comparison stopped: {exc}")
        result.components.append(entry)
        names.add(entry.rcsr)

    top_dimension = max(c.dimension for c, _ in periodic)
    main = [c for c in result.components if c.dimensionality == top_dimension]
    result.interpenetration = len(main)
    statuses = {c.status for c in main}

    if statuses == {"identified"} and len({c.rcsr for c in main}) == 1:
        result.status = "identified"
        result.rcsr = main[0].rcsr
    elif "unstable" in statuses:
        result.status = "unstable"
        result.message = (
            "The underlying net is unstable (two vertices coincide in its "
            "barycentric placement), which exact identification cannot "
            "handle."
        )
    elif "identified" in statuses:
        result.status = "identified"
        result.rcsr = " + ".join(sorted(c.rcsr for c in main if c.rcsr))
        result.notes.append("The interpenetrating nets are not all the same.")
    else:
        result.status = "not_in_library"
        result.message = (
            "The underlying net was built but matches no net in the "
            f"{library.description}."
        )

    return result


def topology_to_dict(result: TopologyResult) -> dict:
    """JSON-safe form for reports."""
    return {
        "status": result.status,
        "rcsr": result.rcsr,
        "summary": result.summary,
        "representation": result.representation,
        "interpenetration": result.interpenetration,
        "components": [
            {
                "dimensionality": c.dimensionality,
                "rcsr": c.rcsr,
                "status": c.status,
                "vertices_per_primitive_cell": c.vertices,
                "connectivity": list(c.connectivity),
                "coordination_sequences": [list(s) for s in c.coordination_sequences],
            }
            for c in result.components
        ],
        "nodes": result.vertex_kinds,
        "edges_from": result.edge_formulas,
        "removed_terminal": result.dropped_formulas,
        "merged_parallel_edges": result.merged_edges,
        "library": result.library,
        "message": result.message,
        "notes": result.notes,
        "rod_sbus": result.rod_sbus,
        "all_node": result.all_node,
    }
