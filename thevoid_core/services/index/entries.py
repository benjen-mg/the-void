"""Index entries (round 65; design section 4).

An entry is one analysis of one structure file, by one version of the
Void, with one configuration: the report the-void-analyze wrote, with
local paths removed and identity, licence and contributor added.

Entries are immutable and named by the SHA-256 of their canonical JSON
(`entry_id`): sorted keys, no spaces, UTF-8, no NaN. The same analysis
added twice is one file, two people adding entries never conflict, and
a hand-edited entry no longer matches its name (Index.verify).

What an entry leaves out:

- the report's absolute input path (`input.file`), and any other
  absolute path in the results (a RASPA folder, say), which becomes
  its base name;
- the report's generation time, so the same analysis gives the same
  entry.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path

ENTRY_FORMAT_VERSION = 1


def canonical(obj) -> str:
    """The text an id is computed from."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def compute_entry_id(entry: dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in entry.items() if k != "entry_id"})
                          .encode("utf-8")).hexdigest()


def analysis_id(entry: dict) -> str:
    """What makes two entries the same analysis: the same file, analysed
    to the same results with the same software. Unlike entry_id it
    leaves out who added it, when, and what they said. Adding the same
    analysis again is therefore recognised however it is annotated, and
    two members who add it separately (two entry files, no conflict)
    are shown as one row with both contributors."""
    return hashlib.sha256(canonical({"file_sha256": entry["structure"]["file_sha256"],
                                     "analysis": entry["analysis"]}).encode("utf-8")
                          ).hexdigest()


def method_key(version: str, config: dict) -> str:
    """Entries with the same method_key were computed the same way: the
    same version of the Void with the same options (design section 6)."""
    return hashlib.sha256(canonical({"the_void": version, "config": config})
                          .encode("utf-8")).hexdigest()[:16]


def _strip_paths(obj):
    """Absolute paths become base names: they give away user names and
    folder layouts, and mean nothing on another machine."""
    if isinstance(obj, dict):
        return {k: _strip_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_paths(v) for v in obj]
    if isinstance(obj, str) and "\n" not in obj and _looks_absolute(obj):
        return obj.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return obj


def _looks_absolute(text: str) -> bool:
    """A POSIX, home-relative or Windows absolute path with at least one
    folder in it."""
    body = text.replace("\\", "/")
    if body.startswith(("/", "~/")):
        return body.rstrip("/").count("/") >= 1 and len(body) > 1
    return len(body) > 3 and body[0].isalpha() and body[1:3] == ":/"


#: Result fields that are settings, not measurements, by analysis: what a
#: report without a config (one from the GUI) records of how it was run.
_PARAMETER_FIELDS = ("probe", "probe_radius_A", "samples_per_atom", "guest",
                     "guest_diameter_A", "grid_shape", "radii", "wavelength",
                     "two_theta_range_deg", "bin_width_A", "lcd_pld_method",
                     "accessibility_method", "library")


def inferred_config(report: dict) -> dict:
    """A config for a report that has none (Analysis menu results, round
    66): which analyses ran and the settings their results record. So
    GUI entries computed with different probes or samples get different
    method_keys, and are never compared as replicates."""
    results = report.get("results") or {}

    def parameters(section):
        if not isinstance(section, dict):
            return {}
        return {k: section[k] for k in _PARAMETER_FIELDS if k in section}

    out = {}
    for name, section in sorted(results.items()):
        if name == "guests" and isinstance(section, dict):
            out[name] = {g: parameters(v) for g, v in sorted(section.items())}
        else:
            out[name] = parameters(section)
    return {"inferred_from_results": True, "analyses": out}


def aliases_for(structure, path, cif_text: str | None = None) -> list[dict]:
    """Names to search by, never identity (design 3.4): the file's stem,
    its data blocks and CSD codes, and CoRE MOF 2024 entries identified
    as this structure, with their refcodes and DOIs."""
    from thevoid_core.services.reference.coremof import find_matches, names_from_cif_text

    out = [{"kind": "name", "value": Path(path).stem}]
    text = cif_text if cif_text is not None else Path(path).read_text(
        encoding="utf-8", errors="replace")
    for name in names_from_cif_text(text):
        out.append({"kind": "data_block", "value": name})
    try:
        matches = find_matches(structure, path=path)
    except Exception:
        matches = []
    for match in matches:
        if match.confidence != "identified":
            continue
        out.append({"kind": "coremof", "value": match.entry.id})
        if match.entry.values.get("csd_refcode"):
            out.append({"kind": "csd", "value": match.entry.values["csd_refcode"]})
        if match.entry.values.get("doi"):
            out.append({"kind": "doi", "value": match.entry.values["doi"]})
    seen, unique = set(), []
    for alias in out:
        key = (alias["kind"], alias["value"])
        if key not in seen:
            seen.add(key)
            unique.append(alias)
    return unique


def make_entry(report: dict, structure_path, *, contributor: str, note: str = "",
               origin: str | None = None, declared_licence: dict | None = None,
               joined: dict | None = None, added: str | None = None, structure=None,
               previous: dict | None = None) -> dict:
    """An entry from a the-void-analyze report and the file it was
    computed from. Raises ValueError if the file is not the one the
    report was computed from (SHA-256).

    previous: an older entry of the same file (a refresh, round 67).
    What the file alone no longer shows is carried over from it: its
    aliases, its origin and parent, and its licence record when the file
    has no licence of its own where it now is (a CSD header still wins).
    """
    from pymatgen.core import Structure

    from thevoid_core.services.index import licence
    from thevoid_core.services.index.identity import KEY_VERSION, framework_key
    from thevoid_core.services.reference.coremof import StructureFacts
    from thevoid_core.services.report import _clean, file_sha256, summary_row

    path = Path(structure_path)
    sha = file_sha256(path)
    recorded = (report.get("input") or {}).get("sha256")
    if recorded and sha != recorded:
        raise ValueError(f"{path.name} has changed since it was analysed (SHA-256 "
                         f"{sha[:12]}... against the report's {recorded[:12]}...)")
    structure = structure if structure is not None else Structure.from_file(path)
    key = framework_key(structure)
    text = path.read_text(encoding="utf-8", errors="replace")
    parent = licence._parent_of(path)
    software = dict((report.get("provenance") or {}).get("software") or {})
    config = dict(report.get("config") or {}) or inferred_config(report)
    summary = summary_row(report)
    summary["file"] = path.name
    entry = {
        "entry_format_version": ENTRY_FORMAT_VERSION,
        "structure": {
            "file_sha256": sha,
            "file_name": path.name,
            "framework_key": key.key,
            "key_version": KEY_VERSION,
            "dimensionalities": list(key.dimensionalities),
            "formula": key.formula,
            "n_atoms": key.n_atoms,
            "metals": sorted(StructureFacts.of(structure).metals),
            "aliases": aliases_for(structure, path, text),
            "origin": "hypothetical" if parent is not None else (origin or "unspecified"),
            "parent": ({"file_name": Path(parent.get("file") or "").name,
                        "file_sha256": parent.get("sha256")} if parent is not None else None),
            "licence": licence.detect(path, declared_licence),
            "report": report.get("structure") or {},
        },
        "analysis": {
            "software": software,
            "platform": (report.get("provenance") or {}).get("platform"),
            "report_format_version": (report.get("provenance") or {}).get(
                "report_format_version"),
            "config": config,
            "method_key": method_key(software.get("the_void", ""), config),
            "methods": report.get("methods") or {},
            "results": report.get("results") or {},
            "errors": report.get("errors") or {},
            "summary": summary,
            "joined": dict(joined or {}),
        },
        "contributor": {
            "name": contributor,
            "added": added or datetime.date.today().isoformat(),
            "note": note,
        },
    }
    if previous is not None:
        s, old = entry["structure"], previous["structure"]
        seen = {(a["kind"], a["value"]) for a in s["aliases"]}
        s["aliases"] += [a for a in old.get("aliases") or []
                         if (a["kind"], a["value"]) not in seen]
        if s["parent"] is None and old.get("parent") is not None:
            s["origin"], s["parent"] = old["origin"], old["parent"]
        elif s["origin"] == "unspecified":
            s["origin"] = old.get("origin", "unspecified")
        if s["licence"]["basis"] == "unknown" and old.get("licence"):
            s["licence"] = dict(old["licence"])
    entry = _clean(_strip_paths(entry))
    entry["entry_id"] = compute_entry_id(entry)
    return entry


def entry_to_report(entry: dict, input_file: str) -> dict:
    """The entry as a the-void-analyze report, for build/: the Batch
    Results window reads it as it reads any results folder. input_file
    is where the build expects the structure (a stored copy, relative
    to the index, or just the file name)."""
    s, a = entry["structure"], entry["analysis"]
    return {
        "provenance": {"software": a["software"], "platform": a["platform"],
                       "report_format_version": a["report_format_version"],
                       "generated": entry["contributor"]["added"]},
        "input": {"file": input_file, "sha256": s["file_sha256"]},
        "structure": s["report"],
        "methods": a["methods"],
        "results": a["results"],
        "errors": a["errors"],
        "config": a["config"],
        "index": {
            "entry_id": entry["entry_id"],
            "framework_key": s["framework_key"],
            "method_key": a["method_key"],
            "contributor": entry["contributor"],
            "licence": s["licence"],
            "origin": s["origin"],
            "parent": s["parent"],
            "aliases": s["aliases"],
        },
    }
