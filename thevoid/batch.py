"""Many structures at once: the-void-analyze from Python."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _analyse(structure, source, space_group, config):
    """Worker (module level, so it pickles): one structure's report."""
    import logging
    import warnings

    from thevoid_core.services.report import run_analysis

    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            report = run_analysis(structure, config, source=source)
    finally:
        logging.disable(previous)
    if space_group:
        report["structure"]["space_group"] = space_group
    return report


def _unique(names):
    seen, out = {}, []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        out.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return out


class BatchTable:
    """The result of analyze_batch: a summary row and a report per
    structure, in input order."""

    def __init__(self, names, reports, errors, folder=None):
        self.names = names
        #: Report per structure (None where the input could not be read).
        self.reports = reports
        #: {name: message} for inputs that failed to load or any
        #: analysis that failed.
        self.errors = errors
        #: Where it was written, if it was.
        self.folder = Path(folder) if folder else None
        self._extra = {}

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def rows(self) -> list:
        """The summary CSV's rows (dicts, the CLI's columns)."""
        rows = []
        for name, report in zip(self.names, self.reports):
            if report is None:
                rows.append({"file": name, "errors": self.errors.get(name, "")})
                continue
            row = report.row
            if not row.get("file"):
                row["file"] = name
            row.update(self._extra.get(row["file"], {}))
            rows.append(row)
        return rows

    def to_dataframe(self):
        """A pandas DataFrame of rows (needs pandas)."""
        from thevoid_core.services.report import SUMMARY_COLUMNS
        from thevoid._tables import dataframe

        rows = self.rows
        columns = list(SUMMARY_COLUMNS) + [k for r in rows for k in r if k not in SUMMARY_COLUMNS]
        return dataframe(rows, columns=list(dict.fromkeys(columns)))

    def to_csv(self, path) -> Path:
        from thevoid_core.services.report import summary_csv

        path = Path(path)
        path.write_text(summary_csv(self.rows), encoding="utf-8")
        return path

    def join(self, csv_path) -> "BatchTable":
        """Join the ref_ / fn_ columns of a the-void-fetch reference.csv
        or a the-void-functionalize variants.csv (the-void-analyze
        --reference-csv), by file name."""
        from thevoid_core.cli import read_reference_csv

        for file, row in read_reference_csv(csv_path).items():
            self._extra[file] = {k: v for k, v in row.items() if k != "file"}
        if self.folder is not None:
            self.to_csv(self.folder / "summary.csv")
        return self

    def results(self):
        """The batch as the Batch Results window and the figures see it
        (writes it to a temporary folder first if it was not written)."""
        from thevoid_core.services.batch_results import load_batch_results

        if self.folder is None:
            import tempfile

            self._temp = tempfile.TemporaryDirectory(prefix="thevoid-batch-")
            self.write(self._temp.name, reports=False)
            folder = Path(self._temp.name)
            self.folder = None
        else:
            folder = self.folder
        return load_batch_results(folder)

    def write(self, folder, *, reports: bool = True) -> Path:
        """summary.csv, plus a JSON and Markdown report per structure
        (as the-void-analyze --out writes them)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if reports:
            for name, report in zip(self.names, self.reports):
                if report is not None:
                    stem = Path(name).stem
                    report.write(folder / f"{stem}.json")
                    report.write(folder / f"{stem}.md")
        self.to_csv(folder / "summary.csv")
        self.folder = folder
        return folder

    def __len__(self):
        return len(self.names)

    def __repr__(self):
        return f"BatchTable({len(self)} structures, {len(self.errors)} with errors)"


def analyze_batch(inputs, *, jobs: int = 1, out=None, reference_csv=None, figures: bool = False,
                  progress=None, **options) -> BatchTable:
    """Analyse many structures as the-void-analyze does.

    inputs: file paths, Frameworks, pymatgen Structures or ASE Atoms,
    mixed. options: the analysis options (thevoid.analysis_options()).
    jobs > 1 runs structures in parallel processes. out: a folder to
    write summary.csv and per-structure reports into, as --out does;
    structures that did not come from a file are also written there as
    CIFs, so the batch can be reproduced and every row opened.
    reference_csv joins ref_/fn_ columns; figures writes the default
    figures into out/figures. progress(name, done, total) is called
    after each structure.
    """
    from thevoid._inputs import to_structure
    from thevoid.framework import Framework
    from thevoid.report import Report, analysis_config

    config = analysis_config(options)
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    items, errors = [], {}
    for index, item in enumerate(inputs):
        try:
            if isinstance(item, Framework):
                structure, source, space_group, name = (item._structure, item.source,
                                                        item.space_group, item.name)
            else:
                structure, source, space_group = to_structure(item)
                name = Path(source).stem if source else structure.composition.reduced_formula
            items.append([name, structure, source, space_group])
        except Exception as exc:
            name = Path(str(item)).stem if isinstance(item, (str, Path)) else f"input {index + 1}"
            items.append([name, None, None, None])
            errors[name] = f"could not read structure: {type(exc).__name__}: {exc}"
    file_stems = [i[0] for i in items if i[2]]
    shared = sorted({n for n in file_stems if file_stems.count(n) > 1})
    if shared:
        # As the-void-analyze: two files with one name would overwrite
        # each other's reports.
        raise ValueError(f"input files share a name: {', '.join(shared)}")
    names = _unique([i[0] for i in items])
    for item, name in zip(items, names):
        if item[0] in errors and name != item[0]:
            errors[name] = errors.pop(item[0])
        item[0] = name

    if out is not None:
        from pymatgen.io.cif import CifWriter

        Path(out).mkdir(parents=True, exist_ok=True)
        for item in items:
            name, structure, source = item[0], item[1], item[2]
            if structure is not None and source is None:
                path = Path(out) / f"{name}.cif"
                CifWriter(structure).write_file(str(path))
                item[2] = str(path)

    reports = [None] * len(items)
    todo = [k for k, item in enumerate(items) if item[1] is not None]
    done = 0

    def record(k, report):
        nonlocal done
        reports[k] = Report(report, name=items[k][0])
        if report.get("errors"):
            errors[items[k][0]] = "; ".join(f"{a}: {m}" for a, m in report["errors"].items())
        done += 1
        if progress is not None:
            progress(items[k][0], done, len(todo))

    if jobs == 1 or len(todo) <= 1:
        for k in todo:
            record(k, _analyse(items[k][1], items[k][2], items[k][3], config))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {k: pool.submit(_analyse, items[k][1], items[k][2], items[k][3], config)
                       for k in todo}
            for k in todo:
                record(k, futures[k].result())

    table = BatchTable([i[0] for i in items], reports, errors)
    if reference_csv:
        table.join(reference_csv)
    if out is not None:
        table.write(out)
        if figures:
            from thevoid_core.services.batch_results import load_batch_results
            from thevoid_core.services.figures import write_default_figures

            write_default_figures(load_batch_results(Path(out)), Path(out) / "figures")
    elif figures:
        raise ValueError("figures=True needs out= (a folder to write them into)")
    return table
