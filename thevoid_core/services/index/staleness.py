"""Which index entries are out of date (round 67; design section 6).

An entry's numbers are those of the version of the Void that computed
them. When a later version changes how a number is computed, the entry
keeps its old number, and the index should say so rather than let it
be ranked beside new ones as if nothing had changed.

The method changelog (data/method_changes.json) records which
versions changed which summary columns. An entry is
**stale** in a column when a version newer than its own, but not newer
than this one, changed that column and the entry has a value there.

An entry is **superseded** when a newer version has analysed the same
file with the same options: the newer entry is the current one, and the
old one needs no refresh. Only current stale entries are refreshed.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass, field
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parents[2] / "data" / "method_changes.json"


def version_tuple(text) -> tuple:
    """"0.9.3" -> (0, 9, 3); anything unreadable sorts first."""
    parts = []
    for part in str(text or "").split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def load_changes(path=CHANGELOG) -> list:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)["changes"]


@dataclass
class Staleness:
    entry_id: str
    version: str
    #: Columns whose method changed since, each with the versions and
    #: what changed.
    columns: dict = field(default_factory=dict)
    superseded_by: str | None = None

    @property
    def stale(self) -> bool:
        return bool(self.columns)

    def reasons(self) -> list:
        out, seen = [], set()
        for changes in self.columns.values():
            for change in changes:
                key = (change["version"], change["what"])
                if key not in seen:
                    seen.add(key)
                    out.append(f"{change['version']}: {change['what']}")
        return out


def _applies(change, entry) -> bool:
    if change.get("condition") == "not_cif":
        return not entry["structure"]["file_name"].lower().endswith(".cif")
    if change.get("condition") == "value_matches":
        # Only entries whose value in a column matches one of the
        # patterns (fnmatch), e.g. a net the old version could not name.
        summary = entry["analysis"].get("summary") or {}
        return any(fnmatch.fnmatchcase(str(summary.get(column) or ""), pattern)
                   for column, patterns in change["values"].items() for pattern in patterns)
    if change.get("condition") == "has_metal":
        # An entry that does not list its metals is assumed affected.
        metals = entry["structure"].get("metals")
        return metals is None or bool(set(metals) & set(change["metals"]))
    return True


def staleness_of(entry: dict, changes: list, current_version: str) -> Staleness:
    version = entry["analysis"]["software"].get("the_void", "")
    own, now = version_tuple(version), version_tuple(current_version)
    summary = entry["analysis"].get("summary") or {}
    present = [c for c, v in summary.items() if v not in (None, "")]
    result = Staleness(entry["entry_id"], version)
    for change in changes:
        if not (own < version_tuple(change["version"]) <= now) or not _applies(change, entry):
            continue
        for pattern in change["columns"]:
            for column in fnmatch.filter(present, pattern):
                result.columns.setdefault(column, []).append(change)
    return result


def _config_without_version(entry) -> str:
    from thevoid_core.services.index.entries import canonical

    return canonical(entry["analysis"].get("config") or {})


def assess(entries: dict, *, current_version: str | None = None, changes=None) -> dict:
    """{entry_id: Staleness} for every entry: its stale columns, and the
    newest entry of the same file and options when that is newer."""
    from thevoid_core.version import __version__

    current_version = current_version or __version__
    changes = load_changes() if changes is None else changes
    out = {k: staleness_of(e, changes, current_version) for k, e in entries.items()}
    newest = {}
    for entry_id, entry in entries.items():
        key = (entry["structure"]["file_sha256"], _config_without_version(entry))
        version = version_tuple(entry["analysis"]["software"].get("the_void", ""))
        best = newest.get(key)
        if best is None or (version, entry_id) > best[0]:
            newest[key] = ((version, entry_id), entry_id)
    for entry_id, entry in entries.items():
        key = (entry["structure"]["file_sha256"], _config_without_version(entry))
        (best_version, _), best_id = newest[key]
        own = version_tuple(entry["analysis"]["software"].get("the_void", ""))
        if best_id != entry_id and best_version > own:
            out[entry_id].superseded_by = best_id
    return out
