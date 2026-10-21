"""Reports: everything the-void-analyze writes for one structure."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path


def analysis_options() -> dict:
    """{option: default} for Framework.analyze and analyze_batch: the
    fields of the-void-analyze's AnalysisConfig."""
    from thevoid_core.services.report import AnalysisConfig

    defaults = AnalysisConfig()
    return {f.name: getattr(defaults, f.name) for f in fields(AnalysisConfig)}


def analysis_config(options: dict):
    """An AnalysisConfig from keyword options, refusing unknown ones
    (a misspelt option silently ignored would be a wrong result)."""
    from thevoid_core.services.report import AnalysisConfig, _guest_diameter

    valid = analysis_options()
    unknown = sorted(set(options) - set(valid))
    if unknown:
        raise TypeError(f"Unknown analysis option(s) {unknown}; valid: {sorted(valid)}")
    options = dict(options)
    if "guests" in options:
        if isinstance(options["guests"], str):
            options["guests"] = [g for g in options["guests"].split(",") if g.strip()]
        options["guests"] = list(options["guests"])
        for entry in options["guests"]:
            _guest_diameter(entry)  # raises ValueError for an unknown guest
    if "xrd_two_theta_range" in options:
        options["xrd_two_theta_range"] = tuple(options["xrd_two_theta_range"])
    return AnalysisConfig(**options)


class Report:
    """One structure's report, as the-void-analyze writes it."""

    def __init__(self, data: dict, *, name: str = "framework"):
        self.data = data
        self.name = name

    @property
    def errors(self) -> dict:
        """{analysis: message} for every analysis that failed."""
        return dict(self.data.get("errors", {}))

    @property
    def ok(self) -> bool:
        return not self.data.get("errors")

    @property
    def row(self) -> dict:
        """The structure's row of the batch summary CSV."""
        from thevoid_core.services.report import summary_row

        return summary_row(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def to_json(self) -> str:
        from thevoid_core.services.report import to_json

        return to_json(self.data)

    def to_markdown(self) -> str:
        from thevoid_core.services.report import to_markdown

        return to_markdown(self.data)

    def write(self, path) -> Path:
        """Write as JSON (.json) or Markdown (.md), by the suffix."""
        path = Path(path)
        if path.suffix.lower() == ".md":
            path.write_text(self.to_markdown(), encoding="utf-8")
        elif path.suffix.lower() == ".json":
            path.write_text(self.to_json(), encoding="utf-8")
        else:
            raise ValueError("Give a .json or .md path")
        return path

    def __repr__(self):
        done = ", ".join(sorted(self.data.get("results", {})))
        return f"Report({self.name!r}: {done or 'nothing'}" + \
            (f"; errors: {', '.join(self.errors)})" if self.errors else ")")

    def _repr_json_(self):  # Jupyter
        return json.loads(self.to_json())
