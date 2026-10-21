"""Command-line batch analysis: `the-void-analyze` (or
`python -m thevoid_core.cli`).

Runs the same analyses as the application, with the same code and
defaults, on one or many structure files without opening a window.
Each input gets a JSON report and a Markdown report, and the batch gets
a one-row-per-structure summary CSV for screening.

    the-void-analyze MOFs/*.cif --out results --psd --guests CO2,N2
    the-void-analyze CALF-20.cif --surface-probe-radius 1.65 --xrd
    the-void-analyze MOFs/*.cif --raspa --raspa-adsorbate CO2 --raspa-pressures 0.1,1,10
    the-void-analyze coremof/*.cif --compare-coremof
    the-void-analyze hmof/*.cif --reference-csv hmof/reference.csv
    the-void-analyze MOFs/*.cif --figures

--compare-coremof (round 56) finds each structure in the bundled CoRE
MOF 2024 index and compares, recomputing the Void's values under CoRE's
convention (services/reference/compare.py); --reference-csv joins a
reference.csv written by `the-void-fetch` into the summary, by file
name, as its ref_ columns.

--raspa also writes a RASPA2 GCMC input folder per structure
(<out>/<name>_RASPA_<adsorbate>/, see services/raspa_export.py); what it
assumed is recorded in that structure's report.

Exit status: 0 if every file was read and every requested analysis
ran, 1 if anything failed (the failure is recorded in that file's
report and in the summary's "errors" column), 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from thevoid_core.version import __version__


def build_parser() -> argparse.ArgumentParser:
    from thevoid_core.data.probe_radii import PROBE_RADII

    parser = argparse.ArgumentParser(
        prog="the-void-analyze",
        description="Batch pore, surface-area, open-metal-site, topology, "
        "PSD and XRD analysis of crystal structures (the Void "
        f"{__version__}).",
    )
    parser.add_argument("files", nargs="+", help="CIF / POSCAR / JSON structure files")
    parser.add_argument("-o", "--out", default="the_void_results",
                        help="output directory (default: %(default)s)")
    parser.add_argument("--summary", default="summary.csv",
                        help="summary CSV name inside --out (default: %(default)s)")
    parser.add_argument("--format", choices=["json", "md", "both"], default="both",
                        help="per-structure report format (default: %(default)s)")
    parser.add_argument("--pore-probe-radius", type=float, default=1.2,
                        help="probe radius for void fraction / accessible volume, A (default: %(default)s)")
    parser.add_argument("--surface-probe", default="N2", choices=sorted(PROBE_RADII),
                        help="named surface-area probe (default: %(default)s)")
    parser.add_argument("--surface-probe-radius", type=float, default=None,
                        help="custom surface-area probe radius, A (overrides --surface-probe)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="surface samples per atom (default: %(default)s)")
    parser.add_argument("--grid-spacing", type=float, default=0.3,
                        help="void-space grid spacing, A (default: %(default)s)")
    parser.add_argument("--guests", default="",
                        help="comma-separated guests to test, by name "
                        f"({', '.join(sorted(PROBE_RADII))}) or name=diameter")
    parser.add_argument("--psd", action="store_true", help="also compute the pore size distribution")
    parser.add_argument("--psd-bin", type=float, default=0.2, help="PSD bin width, A (default: %(default)s)")
    parser.add_argument("--xrd", action="store_true", help="also simulate powder XRD")
    parser.add_argument("--xrd-wavelength", default="CuKa", help="pymatgen wavelength key (default: %(default)s)")
    parser.add_argument("--no-pore", action="store_true", help="skip pore analysis")
    parser.add_argument("--no-surface", action="store_true", help="skip surface area")
    parser.add_argument("--no-oms", action="store_true", help="skip open-metal-site detection")
    parser.add_argument("--no-topology", action="store_true",
                        help="skip net / RCSR topology identification")
    parser.add_argument("--rcsr-archive", default=None,
                        help="Systre .arc file of RCSR nets to match against "
                        "(default: the bundled curated set)")
    reference = parser.add_argument_group(
        "Reference databases",
        "Compare with CoRE MOF 2024 (bundled index) or join values from a "
        "folder written by the-void-fetch.",
    )
    reference.add_argument("--compare-coremof", action="store_true",
                           help="find each structure in CoRE MOF 2024 and compare LCD, PLD, "
                           "ASA, topology and open metal sites (the Void's values "
                           "recomputed with CoRE's radii and probe)")
    reference.add_argument("--predict", nargs="?", const="all", default=None, metavar="MODELS",
                           help="apply property models (round 61): 'all' bundled ones "
                           "(water_kh, thermal_stability, solvent_stability, "
                           "water_stability), a comma-separated list, or folders "
                           "the-void-train wrote; adds pred_ columns (needs scikit-learn)")
    reference.add_argument("--reference-csv", default=None,
                           help="reference.csv from the-void-fetch: its ref_ columns are "
                           "added to the summary, matched by file name")
    raspa = parser.add_argument_group(
        "RASPA2 export",
        "Write a ready-to-run RASPA2 GCMC input set for each structure. "
        "A documented starting point: UFF framework + TraPPE adsorbate, "
        "every assumption listed in the folder's README.txt.",
    )
    raspa.add_argument("--raspa", action="store_true",
                       help="write <out>/<name>_RASPA_<adsorbate>/ per structure")
    raspa.add_argument("--raspa-adsorbate", default="CO2", choices=["CO2", "N2", "CH4"],
                       help="TraPPE adsorbate (default: %(default)s)")
    raspa.add_argument("--raspa-temperature", type=float, default=298.0,
                       help="temperature, K (default: %(default)s)")
    raspa.add_argument("--raspa-pressures", default="1",
                       help="comma-separated pressures in bar, one isotherm point each (default: %(default)s)")
    raspa.add_argument("--raspa-cycles", type=int, default=10000,
                       help="production cycles (default: %(default)s)")
    raspa.add_argument("--raspa-init-cycles", type=int, default=5000,
                       help="initialisation cycles (default: %(default)s)")
    raspa.add_argument("--raspa-cutoff", type=float, default=12.8,
                       help="LJ and Ewald real-space cutoff, A (default: %(default)s)")
    raspa.add_argument("--raspa-keep-free-molecules", action="store_true",
                       help="keep non-bonded solvent/counter-ions instead of removing them")
    raspa.add_argument("--raspa-no-blocking", action="store_true",
                       help="do not block pockets the adsorbate cannot reach")
    raspa.add_argument("--raspa-blocking-probe", type=float, default=None,
                       help="probe radius for pocket blocking, A (default: the adsorbate's usual probe)")
    raspa.add_argument("--raspa-charge-equilibration", action="store_true",
                       help="let RASPA assign framework charges when the CIF has none")
    parser.add_argument("--figures", action="store_true",
                        help="also write the default publication figures (SVG, PDF, PNG + "
                        "JSON) into <out>/figures; see the-void-figures for others")
    parser.add_argument("-j", "--jobs", type=int, default=1,
                        help="structures to analyse in parallel (default: %(default)s)")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    parser.add_argument("--version", action="version", version=f"the Void {__version__}")
    return parser


def config_from_args(args):
    from thevoid_core.services.report import AnalysisConfig

    return AnalysisConfig(
        pore=not args.no_pore,
        pore_probe_radius=args.pore_probe_radius,
        surface_area=not args.no_surface,
        surface_probe=args.surface_probe,
        surface_probe_radius=args.surface_probe_radius,
        surface_samples_per_atom=args.samples,
        oms=not args.no_oms,
        psd=args.psd,
        psd_bin_width=args.psd_bin,
        guests=[g for g in (s.strip() for s in args.guests.split(",")) if g],
        xrd=args.xrd,
        xrd_wavelength=args.xrd_wavelength,
        grid_spacing=args.grid_spacing,
        topology=not args.no_topology,
        rcsr_archive=args.rcsr_archive,
        compare_coremof=args.compare_coremof,
        predict=[m for m in (args.predict or "").split(",") if m.strip()],
    )


def raspa_settings_from_args(args):
    """RaspaExportSettings for --raspa, or None. Raises ValueError for
    bad values (reported as an argument error by main)."""
    if not args.raspa:
        return None

    from thevoid_core.services.raspa_export import RaspaExportSettings

    try:
        pressures = tuple(
            float(p) * 1.0e5 for p in args.raspa_pressures.split(",") if p.strip()
        )
    except ValueError:
        raise ValueError("--raspa-pressures must be numbers in bar, comma-separated") from None

    settings = RaspaExportSettings(
        adsorbate=args.raspa_adsorbate,
        temperature_K=args.raspa_temperature,
        pressures_Pa=pressures,
        cycles=args.raspa_cycles,
        initialization_cycles=args.raspa_init_cycles,
        cutoff_A=args.raspa_cutoff,
        charge_equilibration=args.raspa_charge_equilibration,
        remove_free_molecules=not args.raspa_keep_free_molecules,
        block_pockets=not args.raspa_no_blocking,
        blocking_probe_radius_A=args.raspa_blocking_probe,
        grid_spacing_A=args.grid_spacing,
    )
    settings.validate()
    return settings


def export_raspa(structure, path: str, settings, out_dir: str, report: dict) -> None:
    """Write one structure's RASPA folder and record it in its report -
    or record why it could not be written. Never raises."""
    from thevoid_core.services.raspa_export import (
        build_raspa_export,
        summary_for_report,
        write_raspa_export,
    )

    directory = Path(out_dir) / f"{Path(path).stem}_RASPA_{settings.adsorbate}"

    try:
        export = build_raspa_export(structure, None, settings, source=path)
        write_raspa_export(export, directory)
    except Exception as exc:  # recorded, not raised
        report["errors"]["raspa_export"] = f"{type(exc).__name__}: {exc}"
        return

    report["results"]["raspa_export"] = summary_for_report(export, directory)


def analyse_file(path: str, config, out_dir: str, report_format: str,
                 raspa_settings=None):
    """Worker: returns (path, summary row, ok, message).

    Library warnings are silenced while this runs and the previous
    logging state restored afterwards. With -j 1 this runs in the
    calling process, and the old unconditional logging.disable() left
    logging switched off for whatever that process did next (round 52:
    it made a Help-menu logging test fail whenever a CLI test ran
    before it).
    """
    import logging
    import warnings

    previous = logging.root.manager.disable
    logging.disable(logging.WARNING)

    try:
        # pymatgen's CIF parser issues a UserWarning per oddity (a P1
        # file, missing formula keys, every author line it cannot
        # parse): dozens per file, which buried the done/FAILED lines of
        # a batch (round 54). Real failures are still reported.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return _analyse_file(path, config, out_dir, report_format, raspa_settings)
    finally:
        logging.disable(previous)


def _analyse_file(path, config, out_dir, report_format, raspa_settings):
    from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader
    from thevoid_core.services.report import (
        run_analysis,
        summary_row,
        to_json,
        to_markdown,
    )

    try:
        info = PymatgenCIFLoader().load(path)
    except Exception as exc:
        message = f"could not read structure: {type(exc).__name__}: {exc}"
        return path, {"file": Path(path).name, "errors": message}, False, message

    report = run_analysis(info.structure, config, source=path)
    report["structure"]["space_group"] = info.space_group

    if raspa_settings is not None:
        export_raspa(info.structure, path, raspa_settings, out_dir, report)

    stem = Path(path).stem
    out = Path(out_dir)

    if report_format in ("json", "both"):
        (out / f"{stem}.json").write_text(to_json(report), encoding="utf-8")
    if report_format in ("md", "both"):
        (out / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")

    ok = not report["errors"]
    message = "ok" if ok else "; ".join(f"{k}: {v}" for k, v in report["errors"].items())
    return path, summary_row(report), ok, message


def read_reference_csv(path) -> dict:
    """{file name: row} from a the-void-fetch reference.csv, or a
    the-void-functionalize variants.csv (round 59). Only their ref_ and
    fn_ columns (and matches_index) are joined, so the summary's own
    columns can never be overwritten."""
    import csv

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file" not in reader.fieldnames:
            raise ValueError(f"{path} has no 'file' column; is it a the-void-fetch reference.csv "
                             "or a the-void-functionalize variants.csv?")
        rows = {}
        for row in reader:
            rows[row["file"]] = {k: v for k, v in row.items()
                                 if k == "file" or k.startswith(("ref_", "fn_"))
                                 or k == "matches_index"}
    return rows


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    config = config_from_args(args)

    try:
        from thevoid_core.services.report import _guest_diameter

        for entry in config.guests:
            _guest_diameter(entry)
    except ValueError as exc:
        parser.error(str(exc))

    if args.rcsr_archive and not Path(args.rcsr_archive).is_file():
        parser.error(f"--rcsr-archive: no such file {args.rcsr_archive}")

    reference_rows = {}
    if args.reference_csv:
        try:
            reference_rows = read_reference_csv(args.reference_csv)
        except (OSError, ValueError) as exc:
            parser.error(f"--reference-csv: {exc}")

    try:
        raspa_settings = raspa_settings_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    files = [str(Path(f)) for f in args.files]
    missing = [f for f in files if not Path(f).is_file()]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Two inputs with the same name would overwrite each other's reports.
    stems = [Path(f).stem for f in files]
    duplicates = {s for s in stems if stems.count(s) > 1}
    if duplicates:
        parser.error(f"input files share a name: {', '.join(sorted(duplicates))}")

    results = {}
    rows_ok = True

    def record(path, row, ok, message):
        nonlocal rows_ok
        results[path] = row
        rows_ok = rows_ok and ok
        if not ok:
            print(f"FAILED {path}: {message}", file=sys.stderr)
        elif not args.quiet:
            print(f"done   {path}")

    for path in missing:
        record(path, {"file": Path(path).name, "errors": "file not found"}, False, "file not found")

    todo = [f for f in files if f not in missing]

    if args.jobs == 1 or len(todo) <= 1:
        for path in todo:
            record(*analyse_file(path, config, str(out_dir), args.format,
                                 raspa_settings))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(analyse_file, path, config, str(out_dir),
                            args.format, raspa_settings)
                for path in todo
            ]
            for future in futures:
                record(*future.result())

    from thevoid_core.services.report import summary_csv

    ordered = [results[f] for f in files]
    for row in ordered:
        extra = reference_rows.get(row.get("file", ""))
        if extra:
            row.update({k: v for k, v in extra.items() if k != "file"})
    (out_dir / args.summary).write_text(summary_csv(ordered), encoding="utf-8")

    if not args.quiet:
        print(f"summary: {out_dir / args.summary}")

    if args.figures:
        from thevoid_core.services.batch_results import load_batch_results
        from thevoid_core.services.figures import write_default_figures

        try:
            written = write_default_figures(load_batch_results(out_dir / args.summary),
                                            out_dir / "figures")
        except Exception as exc:  # the batch itself is done; say why
            print(f"FAILED figures: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows_ok = False
        else:
            if not args.quiet:
                print(f"figures: {len([p for p in written if p.suffix == '.json'])} "
                      f"in {out_dir / 'figures'}")

    return 0 if rows_ok else 1


if __name__ == "__main__":
    sys.exit(main())
