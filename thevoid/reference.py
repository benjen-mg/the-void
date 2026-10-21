"""CoRE MOF 2024 and MOFX-DB (the-void-fetch)."""

from __future__ import annotations


def search_coremof(*, subset=None, pld_min=None, pld_max=None, lcd_min=None, lcd_max=None,
                   asa_min=None, metals=None, topology=None, has_oms=None,
                   downloadable_only=False, limit=None) -> list:
    """Entries of the bundled CoRE MOF 2024 index (17,202, CC-BY-4.0)
    passing every filter, as dicts of the index's columns (lcd_A,
    pld_A, asa_m2_g, topology, doi, ...). Offline."""
    from thevoid_core.services.reference import coremof

    if isinstance(metals, str):
        metals = [m.strip() for m in metals.split(",") if m.strip()]
    entries = coremof.search(subset=subset, pld_min=pld_min, pld_max=pld_max, lcd_min=lcd_min,
                             lcd_max=lcd_max, asa_min=asa_min, metals=metals, topology=topology,
                             has_oms=has_oms, downloadable_only=downloadable_only, limit=limit)
    return [dict(e.values) for e in entries]


def fetch_coremof(folder, *, limit=None, progress=None, **filters):
    """Download CoRE MOF 2024's openly licensed structures matching
    `filters` (search_coremof's) into `folder`, with reference.csv and
    README.txt. Returns the fetch result (.written, .not_downloadable)."""
    from thevoid_core.services.reference import fetch

    if isinstance(filters.get("metals"), str):
        filters["metals"] = [m.strip() for m in filters["metals"].split(",") if m.strip()]
    return fetch.fetch_coremof(folder, filters, limit=limit, progress=progress)


def fetch_mofxdb(folder, *, database: str = "hMOF", limit: int = 100, progress=None, **filters):
    """Download MOFX-DB structures (hMOF by default) into `folder`."""
    from thevoid_core.services.reference import fetch

    return fetch.fetch_mofxdb(folder, database=database, limit=limit, progress=progress,
                              **filters)
