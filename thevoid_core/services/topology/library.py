"""Reference nets, and naming a net by RCSR code (round 53).

The bundled library (data/rcsr_nets.json, built by
tools/build_topology_library.py) holds a curated set of RCSR nets: the
common MOF, ZIF and zeolite nets and the 2D layers, each as its
quotient graph. A full Systre archive of RCSR (Systre's RCSR.arc, or
MOFid's Resources/RCSRnets.arc: 2,930 nets) can be used as well by
pointing the THE_VOID_RCSR_ARCHIVE environment variable at it, or with
the command line's --rcsr-archive. It is not bundled, for licensing
reasons.

Matching is exact. Candidates are narrowed by dimension, vertex count
and the multiset of coordination numbers, then by coordination
sequences, and a name is only given when net.isomorphic() confirms the
two nets are the same net. Coordination sequences alone are not
enough: lta and rho share theirs to at least six shells.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from thevoid_core.services.topology.net import (
    PreparedNet,
    isomorphic,
    parse_systre_key,
    prepare,
)

logger = logging.getLogger(__name__)

BUNDLED_LIBRARY = Path(__file__).resolve().parent.parent.parent / "data" / "rcsr_nets.json"
ARCHIVE_ENVIRONMENT_VARIABLE = "THE_VOID_RCSR_ARCHIVE"


def read_systre_archive(path) -> dict[str, str]:
    """name -> Systre key, from a Systre .arc file (blocks of `key`,
    `id`, ... `end`)."""
    nets = {}
    current: dict[str, str] = {}

    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            word, _, rest = line.strip().partition(" ")
            if word == "key":
                current["key"] = rest.strip()
            elif word == "id":
                current["id"] = rest.strip()
            elif word == "end":
                if "key" in current and "id" in current:
                    nets[current["id"]] = current["key"]
                current = {}

    return nets


@dataclass
class ReferenceNet:
    name: str
    key: str
    description: str = ""
    source: str = "bundled"
    _prepared: PreparedNet | None = field(default=None, repr=False)
    _quick: tuple | None = field(default=None, repr=False)

    def quick_invariant(self) -> tuple:
        """(dimension, vertex count, sorted degrees), from the key alone.
        Archive keys are already minimal, so this is comparable with a
        prepared (minimal) query net."""
        if self._quick is None:
            net = parse_systre_key(self.key)
            degrees = sorted(net.degrees())
            self._quick = (net.dimension, net.n_vertices, tuple(degrees))
        return self._quick

    def prepared(self) -> PreparedNet:
        if self._prepared is None:
            self._prepared = prepare(parse_systre_key(self.key))
        return self._prepared


class NetLibrary:
    def __init__(self, nets: list[ReferenceNet], *, description: str = "") -> None:
        self.nets = nets
        self.description = description
        self._by_quick: dict[tuple, list[ReferenceNet]] = {}

        for net in nets:
            try:
                self._by_quick.setdefault(net.quick_invariant(), []).append(net)
            except Exception:
                logger.debug("Skipping unreadable reference net %s", net.name)

    def __len__(self) -> int:
        return len(self.nets)

    def names(self) -> list[str]:
        return [net.name for net in self.nets]

    def identify(self, query: PreparedNet) -> list[ReferenceNet]:
        """Every reference net isomorphic to query (normally none or one;
        more than one would mean duplicates in the library)."""
        quick = (
            query.net.dimension,
            query.net.n_vertices,
            tuple(sorted(len(v) for v in query.vectors)),
        )
        matches = []

        for candidate in self._by_quick.get(quick, []):
            try:
                reference = candidate.prepared()
                if isomorphic(query, reference):
                    matches.append(candidate)
            except ValueError:
                logger.debug("Could not compare with %s", candidate.name, exc_info=True)

        return matches


def bundled_library() -> NetLibrary:
    data = json.loads(BUNDLED_LIBRARY.read_text(encoding="utf-8"))
    nets = [
        ReferenceNet(entry["name"], entry["key"], entry.get("description", ""))
        for entry in data["nets"]
    ]
    return NetLibrary(nets, description=f"bundled curated RCSR set ({len(nets)} nets)")


def archive_library(path) -> NetLibrary:
    nets = [
        ReferenceNet(name, key, source="archive")
        for name, key in read_systre_archive(path).items()
    ]
    return NetLibrary(nets, description=f"Systre archive {Path(path).name} ({len(nets)} nets)")


_CACHE: dict[str, NetLibrary] = {}


def default_library(archive_path=None) -> NetLibrary:
    """The full archive when one is configured (argument, then the
    environment variable), otherwise the bundled set. Cached per
    process: preparing reference nets is the slow part."""
    path = archive_path or os.environ.get(ARCHIVE_ENVIRONMENT_VARIABLE)
    cache_key = str(path) if path else "bundled"

    if cache_key not in _CACHE:
        if path:
            try:
                library = archive_library(path)
                bundled = bundled_library()
                # Keep the bundled descriptions for nets both contain.
                notes = {net.name: net.description for net in bundled.nets}
                for net in library.nets:
                    net.description = notes.get(net.name, "")
                _CACHE[cache_key] = library
            except OSError:
                logger.warning("Could not read RCSR archive %s; using the bundled set.", path)
                _CACHE[cache_key] = bundled_library()
        else:
            _CACHE[cache_key] = bundled_library()

    return _CACHE[cache_key]
