"""The shared index from Python (round 66): the-void-index, as
objects.

    index = thevoid.Index.create("MyLab-index", name="McKelvey Lab")
    index.add(table, contributor="B. G.")      # a BatchTable written to a folder, or a folder
    index.add(fw.analyze(), fw, contributor="B. G.")   # one Report, and its Framework
    index.build()
    index.table().to_dataframe()
    index.find(fw)                             # [Match(relation, entry_id, ...)]
    index.verify()                             # [] when nothing is wrong

docs/shared-index-design.md is the design, with what was measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AddSummary:
    """What an add did. added / present: entry ids; skipped: (file,
    reason); stored: files copied into the index; withheld: (file,
    licence basis) for files kept out of it."""

    added: tuple
    present: tuple
    skipped: tuple
    stored: tuple
    withheld: tuple

    def __repr__(self):
        return (f"AddSummary({len(self.added)} added, {len(self.present)} already present, "
                f"{len(self.skipped)} skipped, {len(self.stored)} stored, "
                f"{len(self.withheld)} withheld)")


@dataclass(frozen=True)
class Match:
    """An entry related to a structure. relation: "same file", "same
    determination", "same structure", "same key, not compared" or "same
    key, different structure"."""

    relation: str
    entry_id: str
    file_name: str
    the_void_version: str
    contributor: str
    added: str
    summary: dict


def _summary(result) -> AddSummary:
    return AddSummary(tuple(result.added), tuple(result.present), tuple(result.skipped),
                      tuple(result.stored), tuple(result.withheld))


class Index:
    """A shared index folder (see the-void-index)."""

    def __init__(self, path):
        from thevoid_core.services.index.store import Index as _Index

        self.raw = _Index.open(path)

    @classmethod
    def create(cls, path, *, name: str, results_licence: str = "CC-BY-4.0") -> "Index":
        """A new, empty index at path (the-void-index init)."""
        from thevoid_core.services.index.store import Index as _Index

        _Index.init(path, name=name, results_licence=results_licence)
        return cls(path)

    @property
    def path(self) -> Path:
        return self.raw.path

    @property
    def name(self) -> str:
        return self.raw.info.get("name", "")

    def __len__(self):
        return len(self.raw.current_entries())

    def __repr__(self):
        return f"Index({self.name!r}, {len(self)} entries, {self.path})"

    def add(self, what, framework=None, *, contributor: str, note: str = "",
            licence: str | None = None, licence_basis: str | None = None,
            structures=()) -> AddSummary:
        """Add analyses:

        - a the-void-analyze results folder, or a BatchTable written to
          one (table.write(folder), or analyze_batch(out=...));
        - a Report with the Framework it came from (fw.analyze()), whose
          structure must come from a file.

        licence / licence_basis: declared for files nothing else decides
        (never overriding a CSD header). structures: folders holding
        structure files that were moved since the analysis."""
        from thevoid.batch import BatchTable
        from thevoid.report import Report

        declared = ({"spdx": licence, "basis": licence_basis}
                    if licence or licence_basis else None)
        if isinstance(what, Report):
            source = getattr(framework, "source", None)
            if not source:
                raise ValueError("add(report, framework): the framework must come from a file "
                                 "(thevoid.load(path)), so the index can check it")
            return _summary(self.raw.add_report(what.data, source, contributor=contributor,
                                                note=note, declared_licence=declared,
                                                search_dirs=structures))
        if isinstance(what, BatchTable):
            if what.folder is None:
                raise ValueError("write the BatchTable to a folder first (table.write(folder))")
            what = what.folder
        return _summary(self.raw.add_results(what, contributor=contributor, note=note,
                                             declared_licence=declared,
                                             search_dirs=structures))

    def build(self, *, structures=()) -> Path:
        """Write build/ (the-void-index build) and return it. Raises
        RuntimeError listing the problems if any entry could not be read."""
        result = self.raw.build(search_dirs=structures)
        if result.problems:
            raise RuntimeError("; ".join(result.problems))
        return result.folder

    def table(self):
        """The built index as a BatchTable-like view: .rows, .to_dataframe()
        (builds first if build/ is missing or older than the entries)."""
        folder = self.raw.path / "build"
        if not (folder / "summary.csv").is_file() or self._stale(folder):
            self.build()
        return IndexTable(folder)

    def _stale(self, folder) -> bool:
        built = (folder / "summary.csv").stat().st_mtime
        newest = max((f.stat().st_mtime for sub in ("entries", "withdrawn", "matches")
                      for f in (self.raw.path / sub).glob("**/*.json")), default=0.0)
        return newest > built

    def find(self, structure, *, structures=()) -> list:
        """[Match] for a Framework (from a file) or a structure file path,
        closest first."""
        source = getattr(structure, "source", structure)
        if source is None:
            raise ValueError("find(): the framework must come from a file")
        out = []
        for relation, entry in self.raw.find(source, search_dirs=structures):
            a = entry["analysis"]
            out.append(Match(relation, entry["entry_id"], entry["structure"]["file_name"],
                             a["software"].get("the_void", ""),
                             entry["contributor"]["name"], entry["contributor"]["added"],
                             dict(a["summary"])))
        return out

    def verify(self, *, structures=()) -> list:
        """Everything wrong with the index, as messages (the-void-index
        verify); [] when nothing is."""
        return self.raw.verify(search_dirs=structures)

    def stale(self) -> list:
        """[{"entry_id", "file_name", "the_void_version", "columns",
        "reasons"}] for current entries whose numbers a later version of
        the Void computes differently (the-void-index stale)."""
        return [{"entry_id": entry["entry_id"], "file_name": entry["structure"]["file_name"],
                 "the_void_version": verdict.version, "columns": sorted(verdict.columns),
                 "reasons": verdict.reasons()} for entry, verdict in self.raw.stale()]

    def refresh(self, *, contributor: str, jobs: int = 1, limit: int | None = None,
                structures=()) -> dict:
        """Analyse the stale entries again with this version (the-void-index
        refresh), jobs at a time. {"refreshed": [(old id, new id)],
        "skipped": [(id, file, reason)]}."""
        result = self.raw.refresh(contributor=contributor, jobs=jobs, limit=limit,
                                  search_dirs=structures)
        return {"refreshed": list(result.refreshed), "skipped": list(result.skipped)}

    def withdraw(self, entry_id: str, *, reason: str, by: str) -> None:
        """Hide an entry from the build; it stays on record."""
        self.raw.withdraw(entry_id, reason=reason, by=by)


class IndexTable:
    """The rows of an index's build/summary.csv."""

    def __init__(self, folder):
        from thevoid_core.services.batch_results import load_batch_results

        self.folder = Path(folder)
        self.raw = load_batch_results(self.folder)

    @property
    def columns(self) -> list:
        return list(self.raw.columns)

    @property
    def rows(self) -> list:
        return [dict(r.values) for r in self.raw.rows]

    def to_dataframe(self):
        from thevoid._tables import dataframe

        return dataframe(self.rows, columns=self.columns)

    def __len__(self):
        return len(self.raw.rows)

    def __repr__(self):
        return f"IndexTable({len(self)} rows, {self.folder})"
