"""May a structure file be redistributed? (round 65; design section 5)

Default-deny. A CIF does not state its own licence: neither CoRE MOF's
SI files nor the CSD's do. So the answer comes from where the file came
from, in this order:

1. **A CSD header in the file.** "Cambridge Structural Database", or a
   CCDC deposition code with the CCDC's download notice. The file is
   CSD-derived and is never stored, whatever else is said about it.
2. **A reference.csv beside the file, written by the-void-fetch.**
   Since 0.9.5 it has `ref_licence` and `ref_licence_basis` columns
   per file. Older CoRE folders are recognised by `ref_source` = "SI"
   and the CoRE citation in README.txt.
3. **A functionalised variant** (functionalization.json beside it)
   takes its parent's answer, found the same way from the parent's
   copy in the folder. A variant of a CSD structure is therefore not
   stored.
4. **What the contributor declared** (`declared`), for files nothing
   above decides.
5. **Otherwise unknown:** results only.

Only these licences allow storing: CC-BY-4.0, CC-BY-SA-4.0, CC0-1.0 and
PDDL-1.0, plus "public-domain". Anything else is recorded but not
stored.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REDISTRIBUTABLE = {"CC-BY-4.0", "CC-BY-SA-4.0", "CC0-1.0", "PDDL-1.0", "public-domain"}

_CSD_PATTERNS = (
    re.compile(r"Cambridge Structural Database", re.IGNORECASE),
    re.compile(r"Cambridge Crystallographic Data Centre", re.IGNORECASE),
)


def verdict(spdx, basis, *, redistributable=None) -> dict:
    """The licence record an entry carries."""
    if redistributable is None:
        redistributable = spdx in REDISTRIBUTABLE
    return {"spdx": spdx or None, "redistributable": bool(redistributable), "basis": basis}


def has_csd_header(text: str) -> bool:
    """The CCDC's download notice, which every CIF from the CSD (and
    from CCDC's Access Structures) carries."""
    return any(p.search(text) for p in _CSD_PATTERNS)


def _reference_row(path: Path) -> dict | None:
    table = path.parent / "reference.csv"
    if not table.is_file():
        return None
    try:
        with open(table, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("file") == path.name:
                    return row
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    return None


def _from_fetch(path: Path) -> dict | None:
    row = _reference_row(path)
    if row is None:
        return None
    if row.get("ref_licence") or row.get("ref_licence_basis"):
        return verdict(row.get("ref_licence") or None,
                       row.get("ref_licence_basis") or "the-void-fetch reference.csv")
    # Folders fetched before 0.9.5: CoRE's SI subset, told by its own
    # source column and the README's citation.
    readme = path.parent / "README.txt"
    if row.get("ref_source") == "SI" and readme.is_file() and \
            "CoRE MOF" in readme.read_text(encoding="utf-8", errors="replace"):
        return verdict("CC-BY-4.0", "fetched from CoRE MOF 2024's SI archive (README.txt)")
    return None


def _parent_of(path: Path) -> dict | None:
    """The functionalization.json record naming this file as a variant."""
    record_path = path.parent / "functionalization.json"
    if not record_path.is_file():
        return None
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for variant in record.get("variants") or []:
        if Path(variant.get("file") or "").name == path.name:
            return record.get("parent") or {}
    return None


def detect(path, declared: dict | None = None) -> dict:
    """The licence record for a structure file (see the module
    docstring). declared: {"spdx": ..., "basis": ...} from the
    contributor, used only when nothing else decides."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    if has_csd_header(text):
        return verdict("CSD", "CSD header in file", redistributable=False)
    fetched = _from_fetch(path)
    if fetched is not None:
        return fetched
    parent = _parent_of(path)
    if parent is not None:
        copy = path.parent / Path(parent.get("file") or "").name
        if parent.get("file") and copy.is_file() and copy != path:
            inherited = detect(copy)
        elif parent.get("file") and Path(parent["file"]).is_file():
            inherited = detect(parent["file"])
        else:
            inherited = verdict(None, "parent file not found", redistributable=False)
        return {**inherited, "basis": f"functionalised variant; parent: {inherited['basis']}"}
    if declared and (declared.get("spdx") or declared.get("basis")):
        return verdict(declared.get("spdx"),
                       f"declared by contributor: {declared.get('basis') or declared.get('spdx')}")
    return verdict(None, "unknown", redistributable=False)
