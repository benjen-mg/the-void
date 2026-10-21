"""Download reference structures into a folder ready for batch analysis
(round 56).

Two sources:

CoRE MOF 2024 (`fetch_coremof`)
    Structures are chosen from the bundled index (coremof.search), so a
    query is answered offline and says up front how many of its hits
    can be downloaded: CoRE MOF redistributes only the 2,664 structures
    it took from papers' supporting information (CC-BY-4.0, one 9.5 MB
    archive from the CoRE MOF Tools repository, cached after the first
    download); the other 14,538 come from the CSD and need a CSD
    licence. Those are listed, with refcodes, in not_downloadable.csv.
    Each downloaded CIF is checked against its index entry (atom count
    and density), because the archive on GitHub can change after the
    index was bundled.

hMOF, or any MOFX-DB database (`fetch_mofxdb`)
    The MOFX-DB API (Bobbitt et al., J. Chem. Eng. Data 68, 483 (2023);
    https://mof.tech.northwestern.edu), paged 200 at a time. hMOF itself
    is Wilmer et al., Nat. Chem. 4, 83 (2012). Each structure's CIF is
    written with its simulated isotherms beside it (<name>.isotherms.json)
    and MOFX-DB's own geometry in reference.csv. Round 56 could not
    reach MOFX-DB from where it was built: this client follows the
    documented API and was tested on responses of that form, not on the
    live service.

Either way the folder gets reference.csv (one row per structure file,
the database's values, columns prefixed ref_, and since round 65 each
file's ref_licence and ref_licence_basis) and README.txt (source,
citation, licence, query, date, what was skipped and why). Feed the
folder to `the-void-analyze`, and --reference-csv joins reference.csv
into the summary.
"""

from __future__ import annotations

import csv
import datetime
import io
import json
import re
import urllib.parse
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from thevoid_core.services.reference import coremof
from thevoid_core.services.reference.net import cached_download, get_bytes

MOFXDB_URL = "https://mof.tech.northwestern.edu/mofs.json"
MOFXDB_CITATION = ("N. S. Bobbitt, K. Shi, B. J. Bucior, H. Chen, N. Tracy-Amoroso, Z. Li "
                   "et al., MOFX-DB: An Online Database of Computational Adsorption Data "
                   "for Nanoporous Materials, J. Chem. Eng. Data 68, 483 (2023), "
                   "doi:10.1021/acs.jced.2c00583")
HMOF_CITATION = ("C. E. Wilmer, M. Leaf, C. Y. Lee, O. K. Farha, B. G. Hauser, J. T. Hupp, "
                 "R. Q. Snurr, Large-scale screening of hypothetical metal-organic "
                 "frameworks, Nat. Chem. 4, 83 (2012)")
MOFXDB_DATABASES = ("hMOF", "CoREMOF 2014", "CoREMOF 2019", "CSD", "IZA", "PCOD-syn", "Tobacco")

SI_ARCHIVE_NAME = "coremof2024_CR_SI.zip"


@dataclass
class FetchResult:
    source: str
    folder: Path
    written: list = field(default_factory=list)       # [Path]
    skipped: list = field(default_factory=list)       # [(name, reason)]
    not_downloadable: list = field(default_factory=list)  # [CoreMofEntry]
    warnings: list = field(default_factory=list)
    matched: int = 0

    @property
    def reference_csv(self) -> Path:
        return self.folder / "reference.csv"


def safe_file_stem(name: str) -> str:
    """A file name from a database name: path separators and characters
    Windows refuses become "_"; CoRE MOF's brackets are kept, since the
    stem is how a structure is found in the index again."""
    stem = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name)).strip("._")
    return stem or "structure"


def _unique_path(folder: Path, stem: str, suffix: str) -> Path:
    path = folder / f"{stem}{suffix}"
    count = 2
    while path.exists():
        path = folder / f"{stem}_{count}{suffix}"
        count += 1
    return path


def _write_csv(path: Path, rows: list[dict]):
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or ["file"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in columns})


def _describe_filters(filters: dict) -> str:
    parts = [f"{k}={v}" for k, v in filters.items() if v not in (None, "", [], False)]
    return ", ".join(parts) or "none"


# ----------------------------------------------------------------------
# CoRE MOF 2024
# ----------------------------------------------------------------------


def plan_coremof(filters: dict | None = None, *, limit=None):
    """(downloadable, not_downloadable) entries for a query, offline -
    what fetch_coremof would do. limit caps the downloadable list."""
    entries = coremof.search(**(filters or {}))
    downloadable = [e for e in entries if e.downloadable]
    others = [e for e in entries if not e.downloadable]
    if limit:
        downloadable = downloadable[:limit]
    return downloadable, others


def fetch_coremof(folder, filters: dict | None = None, *, limit=None, opener=None,
                  progress=None, verify=True, refresh=False) -> FetchResult:
    """Write the downloadable CoRE MOF 2024 structures matching
    `filters` (coremof.search's keywords) into `folder`.

    The archive is downloaded once and cached (net.cache_dir());
    refresh=True downloads it again. progress(message, done, total) is
    called along the way; returning False cancels (net.Cancelled). Raises net.DownloadError if the
    archive cannot be downloaded - nothing is written then."""
    folder = Path(folder)
    filters = dict(filters or {})
    downloadable, others = plan_coremof(filters, limit=limit)
    result = FetchResult("CoRE MOF 2024", folder, not_downloadable=others,
                         matched=len(downloadable) + len(others))
    source = coremof.index_info()

    def report(message, done=0, total=0):
        if progress is not None:
            return progress(message, done, total)
        return None

    if downloadable:
        path, digest, same = cached_download(
            source["si_zip_url"], SI_ARCHIVE_NAME,
            expected_sha256=source.get("si_zip_sha256"), opener=opener, refresh=refresh,
            progress=lambda done, total: report("Downloading the CoRE MOF archive", done, total or 0),
        )
        if same is False:
            result.warnings.append(
                "The CoRE MOF archive on GitHub has changed since this version of the "
                f"Void was released (SHA-256 {digest[:12]}..., expected "
                f"{source['si_zip_sha256'][:12]}...). Each structure was checked "
                "against its index entry; see the matches_index column."
            )
        archive = zipfile.ZipFile(path)
        members = {Path(name).stem: name for name in archive.namelist() if name.endswith(".cif")}
    else:
        archive, members = None, {}

    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for number, entry in enumerate(downloadable, start=1):
        if report(f"Writing {entry.id}", number, len(downloadable)) is False:
            from thevoid_core.services.reference.net import Cancelled

            raise Cancelled("fetch")
        member = members.get(entry.id)
        if member is None:
            result.skipped.append((entry.id, "not in the downloaded archive"))
            continue
        target = _unique_path(folder, safe_file_stem(entry.id), ".cif")
        data = archive.read(member)
        target.write_bytes(data)
        result.written.append(target)
        row = {"file": target.name}
        row.update({f"ref_{k}": v for k, v in entry.values.items()})
        row["matches_index"] = _check_file(target, entry) if verify else ""
        # Round 65: each file's licence, machine-readable, for the
        # shared index (services/index/licence.py).
        row["ref_licence"] = source["licence"]
        row["ref_licence_basis"] = (f"{source['name']}: supporting-information archive "
                                    f"({source['si_zip_url']})")
        rows.append(row)

    _write_csv(folder / "reference.csv", rows)
    if others:
        _write_csv(folder / "not_downloadable.csv", [
            {"id": e.id, "csd_refcode": e.csd_refcode, "doi": e.doi, "year": e.year,
             "lcd_A": e.lcd_A, "pld_A": e.pld_A, "asa_m2_g": e.asa_m2_g,
             "topology": e.topology}
            for e in others
        ])
    _write_readme(folder, result, [
        f"Source: {source['name']}, bundled index retrieved {source.get('retrieved')} "
        f"(SHA-256 {source.get('sha256', '')[:12]}...).",
        f"Structures: {source['si_zip_url']}",
        f"Citation: {source['citation']}, doi:{source['doi']}.",
        f"Licence: {source['licence']}. Keep this attribution with the files.",
        f"Query: {_describe_filters(filters)}; limit {limit or 'none'}.",
        f"Matched {result.matched} entries: {len(downloadable)} downloadable "
        f"(from papers' supporting information), {len(others)} from the CSD.",
        "CSD-derived structures are not redistributed by CoRE MOF; they are listed "
        "in not_downloadable.csv with their refcodes, for anyone with a CSD licence.",
        "Values in reference.csv are CoRE MOF's: Zeo++ with its default (CCDC) radii, "
        "surface areas with a 1.655 A probe. The Void's own analyses use other radii "
        "and the N2 probe; use the-void-analyze --compare-coremof for a like-for-like "
        "comparison.",
    ])
    return result


def _check_file(path: Path, entry) -> str:
    """'yes' if the file is the entry's structure (coremof.check), 'no:
    <what differs>' if not, 'unreadable: <why>'."""
    try:
        import warnings

        from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = PymatgenCIFLoader().load(str(path)).structure
        checks = coremof.check(entry, coremof.StructureFacts.of(structure))
    except Exception as exc:
        return f"unreadable: {type(exc).__name__}"
    failed = [k for k, (_, _, ok) in checks.items() if not ok]
    return "yes" if not failed else "no: " + ", ".join(failed)


# ----------------------------------------------------------------------
# MOFX-DB (hMOF and others)
# ----------------------------------------------------------------------


def mofxdb_query_url(database="hMOF", *, page=1, pld_min=None, pld_max=None,
                     lcd_min=None, lcd_max=None, vf_min=None, vf_max=None,
                     sa_m2g_min=None, sa_m2g_max=None, name=None, elements=None) -> str:
    """The documented /mofs.json query (https://mof.tech.northwestern.edu/api)."""
    params = [("database", database), ("page", page)]
    for key, value in (("pld_min", pld_min), ("pld_max", pld_max), ("lcd_min", lcd_min),
                       ("lcd_max", lcd_max), ("vf_min", vf_min), ("vf_max", vf_max),
                       ("sa_m2g_min", sa_m2g_min), ("sa_m2g_max", sa_m2g_max),
                       ("name", name)):
        if value not in (None, ""):
            params.append((key, value))
    for element in elements or []:
        params.append(("elements[]", element))
    return MOFXDB_URL + "?" + urllib.parse.urlencode(params)


def isotherm_summary(isotherm: dict) -> dict:
    """The parts of one MOFX-DB isotherm worth keeping: what, at what
    temperature, with which force fields, in which units, and the
    points (pressure, total adsorption)."""
    return {
        "id": isotherm.get("id"),
        "adsorbates": [a.get("name") for a in isotherm.get("adsorbates") or []],
        "temperature_K": isotherm.get("temperature"),
        "adsorbent_forcefield": isotherm.get("adsorbent_forcefield"),
        "molecule_forcefield": isotherm.get("molecule_forcefield"),
        "pressure_units": isotherm.get("pressureUnits"),
        "adsorption_units": isotherm.get("adsorptionUnits"),
        "doi": isotherm.get("DOI"),
        "points": [[p.get("pressure"), p.get("total_adsorption")]
                   for p in isotherm.get("isotherm_data") or []],
    }


def fetch_mofxdb(folder, *, database="hMOF", limit=100, opener=None, progress=None,
                 **filters) -> FetchResult:
    """Write up to `limit` structures of a MOFX-DB database matching
    `filters` (mofxdb_query_url's keywords) into `folder`. A page that
    fails to download raises net.DownloadError; what was written before
    it is kept and listed in README.txt."""
    if database not in MOFXDB_DATABASES:
        raise ValueError(f"Unknown MOFX-DB database {database!r}; one of {MOFXDB_DATABASES}")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    result = FetchResult(f"MOFX-DB {database}", folder)
    rows = []
    page, pages = 1, 1
    try:
        while page <= pages and (not limit or len(result.written) < limit):
            url = mofxdb_query_url(database, page=page, **filters)
            if progress is not None and progress(
                    f"MOFX-DB page {page} of {pages}", len(result.written), limit or 0) is False:
                from thevoid_core.services.reference.net import Cancelled

                raise Cancelled("fetch")
            payload = json.loads(get_bytes(url, opener=opener).decode("utf-8"))
            pages = int(payload.get("pages") or 1)
            results = payload.get("results") or []
            if page == 1:
                result.matched = pages * 200 if pages > 1 else len(results)
            for record in results:
                if limit and len(result.written) >= limit:
                    break
                _write_mofxdb_record(folder, record, rows, result)
            if not results:
                break
            page += 1
    finally:
        _write_csv(folder / "reference.csv", rows)
        citation = [MOFXDB_CITATION + "."]
        if database == "hMOF":
            citation.append("hMOF: " + HMOF_CITATION + ".")
        _write_readme(folder, result, [
            f"Source: MOFX-DB ({MOFXDB_URL}), database {database}.",
            *[f"Citation: {c}" for c in citation],
            "Licence: see https://mof.tech.northwestern.edu - cite the papers above.",
            f"Query: {_describe_filters(filters)}; limit {limit or 'none'}; "
            f"pages read {page - 1 if page > 1 else 0} of {pages}.",
            "Values in reference.csv are MOFX-DB's own (Zeo++ geometry, simulated "
            "isotherms); the radii and probe behind them are MOFX-DB's, not the Void's.",
            "Each <name>.isotherms.json holds that structure's simulated isotherms "
            "(adsorbate, temperature, force fields, units, points).",
        ])
    return result


def _write_mofxdb_record(folder, record, rows, result):
    name = record.get("name") or f"mofxdb_{record.get('id')}"
    cif = record.get("cif")
    if not cif or "_cell_length_a" not in cif:
        result.skipped.append((name, "no CIF in the response"))
        return
    target = _unique_path(folder, safe_file_stem(name), ".cif")
    target.write_text(cif, encoding="utf-8")
    result.written.append(target)
    isotherms = [isotherm_summary(i) for i in record.get("isotherms") or []]
    if isotherms:
        target.with_suffix(".isotherms.json").write_text(
            json.dumps(isotherms, indent=1), encoding="utf-8")
    elements = record.get("elements") or []
    rows.append({
        "file": target.name,
        "ref_database": record.get("database"),
        "ref_id": record.get("id"),
        "ref_name": name,
        "ref_lcd_A": record.get("lcd"),
        "ref_pld_A": record.get("pld"),
        "ref_void_fraction": record.get("void_fraction"),
        "ref_asa_m2_g": record.get("surface_area_m2g"),
        "ref_asa_m2_cm3": record.get("surface_area_m2cm3"),
        "ref_elements": ",".join(e.get("symbol", "") if isinstance(e, dict) else str(e)
                                 for e in elements),
        "ref_mofid": record.get("mofid"),
        "ref_mofkey": record.get("mofkey"),
        "ref_hashkey": record.get("hashkey"),
        "ref_mofdb_version": record.get("mofdb_version"),
        "ref_isotherms": len(isotherms),
        "ref_adsorbates": ",".join(sorted({a for i in isotherms for a in i["adsorbates"] if a})),
        # Round 65: MOFX-DB states no licence for its structures, so the
        # shared index does not store them (results only).
        "ref_licence": "",
        "ref_licence_basis": "MOFX-DB: no licence stated",
    })


# ----------------------------------------------------------------------


def _write_readme(folder: Path, result: FetchResult, lines: list[str]):
    text = io.StringIO()
    text.write(f"Reference structures: {result.source}\n")
    text.write(f"Written by the Void on {datetime.date.today().isoformat()}.\n\n")
    for line in lines:
        text.write(line + "\n")
    text.write(f"\n{len(result.written)} structure file(s) written.\n")
    for name, reason in result.skipped:
        text.write(f"Skipped {name}: {reason}\n")
    for warning in result.warnings:
        text.write(f"Warning: {warning}\n")
    text.write("\nAnalyse them with:\n    the-void-analyze *.cif --out results "
               "--reference-csv reference.csv\n")
    (folder / "README.txt").write_text(text.getvalue(), encoding="utf-8")
