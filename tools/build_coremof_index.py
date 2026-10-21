"""Build thevoid_core/data/coremof2024.json.gz, the bundled CoRE MOF 2024
property index (round 56).

    python tools/build_coremof_index.py CR.json [CR.zip]

CR.json is the computation-ready (CR) property file of the CoRE MOF 2024
database, as distributed with CoRE MOF Tools:
https://raw.githubusercontent.com/Chung-Research-Group/CoRE-MOF-Tools/main/CoREMOF/data/info/CR.json
(42 MB, 17,202 entries in three subsets: ASR, all solvent removed;
FSR, free solvent removed; Ion, charged frameworks with counter-ions).
CR.zip (optional) is the same repository's CoREMOF/data/SI/CR.zip, the
2,664 structures taken from papers' supporting information; the rest
come from the CSD and are not redistributable. When CR.zip is given,
its SHA-256 is recorded so the fetch service can say whether the
structures it downloads are the ones these values were computed for.

Database and data: G. Zhao, L. Brabson, S. Chheda, J. Huang, H. Kim,
K. Liu et al., "CoRE MOF DB: a curated experimental metal-organic
framework database with machine-learned properties for integrated
material-process screening", Matter 8, 102140 (2025),
doi:10.1016/j.matt.2025.102140. Licence CC-BY-4.0, which permits this
trimmed copy with attribution; the attribution travels inside the file
and is shown wherever its values are.

Kept: identifiers, the Zeo++ pore geometry (LCD, PLD, LFPD, surface
areas, pore volume, void fraction), density, atom count, mean atomic
mass, space group, metals, topology, open metal sites, stability and
water class; and (round 61) the first Widom value, as
water_kh_mmol_g_pa. Dropped: MOFid strings, heat capacity, the second
Widom value, and the GEMC isotherms.

Widom: the file gives two numbers per entry, in mmol/g/Pa, without
saying for which gas. The first is taken to be water's Henry
coefficient because it ranks exactly as CoRE's own water classes do.
Median log10 by class: superweak -7.45, weak -4.66, strong -2.51,
superstrong_high_loading +0.30. The classes are defined from water
adsorption, and the values fit water's scale. The second value, a
median 14% of the first, looks like its uncertainty and is not kept.
The stability values are CoRE's own model predictions (Nandy et al.'s
models: thermal decomposition in degrees C, and the probability of
staying stable on solvent removal and in water), not measurements.
"""

import datetime
import gzip
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "thevoid_core" / "data" / "coremof2024.json.gz"

SOURCE_URL = ("https://raw.githubusercontent.com/Chung-Research-Group/CoRE-MOF-Tools/"
              "main/CoREMOF/data/info/CR.json")
SI_ZIP_URL = ("https://raw.githubusercontent.com/Chung-Research-Group/CoRE-MOF-Tools/"
              "main/CoREMOF/data/SI/CR.zip")

COLUMNS = [
    "id", "subset", "refcode", "csd_refcode", "source", "doi", "year",
    "metals", "lcd_A", "pld_A", "lfpd_A", "density_g_cm3",
    "asa_A2", "asa_m2_g", "nasa_A2", "nasa_m2_g", "void_fraction",
    "pv_A3", "pv_cm3_g", "dimension", "topology", "topology_all_node",
    "catenation", "has_oms", "oms_metals", "n_atoms", "mean_atomic_mass",
    "space_group_number", "water_class", "stability_thermal_C",
    "stability_solvent", "stability_water", "unmodified", "memo",
    "water_kh_mmol_g_pa",
]


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN -> None


def _text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("nan", "none", "unknown") else text


def _widom_first(record):
    """The first Widom value; a few entries store it as {id: value}."""
    values = record.get("Widom")
    if not isinstance(values, list) or not values:
        return None
    first = values[0]
    if isinstance(first, dict):
        first = next(iter(first.values()), None)
    return _number(first)


def row(entry_id, subset, record):
    topology = record.get("Topology") or {}
    stability = record.get("Stability") or {}
    source = _text(record.get("Source"))
    refcode = _text(record.get("refcode"))
    csd = refcode.split("_")[0] if source == "CSD" else ""
    return [
        entry_id, subset, refcode, csd, source, _text(record.get("DOI")),
        _text(record.get("Year")), _text(record.get("Metal")),
        _number(record.get("LCD")), _number(record.get("PLD")), _number(record.get("LFPD")),
        _number(record.get("Density")),
        _number(record.get("ASA")), _number(record.get("GSA")),
        _number(record.get("NASA")), _number(record.get("NGSA")),
        _number(record.get("VF")), _number(record.get("PV")), _number(record.get("GPV")),
        record.get("Dimension"),
        _text(topology.get("SingleNode")), _text(topology.get("AllNode")),
        topology.get("Catenation"),
        {"Yes": True, "No": False}.get(record.get("hasOMS")),
        _text(record.get("TypeOMS")), record.get("NAtoms"),
        _number(record.get("AtomicMass")),
        (record.get("SpaceGroup") or {}).get("number"),
        _text(record.get("WaterClass")),
        _number(stability.get("Thermal")), _number(stability.get("Solvent")),
        _number(stability.get("Water")),
        bool(record.get("unmodified")) if record.get("unmodified") is not None else None,
        _text(record.get("memo")),
        _widom_first(record),
    ]


def main(argv):
    cr_path = Path(argv[1])
    raw = cr_path.read_bytes()
    data = json.loads(raw)
    rows = []
    for subset in ("ASR", "FSR", "Ion"):
        for entry_id, record in data[subset].items():
            rows.append(row(entry_id, subset, record))
    source = {
        "name": "CoRE MOF 2024 (computation-ready subset)",
        "citation": ("G. Zhao, L. Brabson, S. Chheda, J. Huang, H. Kim, K. Liu et al., "
                     "CoRE MOF DB: a curated experimental metal-organic framework "
                     "database with machine-learned properties for integrated "
                     "material-process screening, Matter 8, 102140 (2025)"),
        "doi": "10.1016/j.matt.2025.102140",
        "licence": "CC-BY-4.0",
        "url": SOURCE_URL,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "retrieved": datetime.date.today().isoformat(),
        "units": data.get("unit", {}),
        "method": ("Zeo++ with its default atomic radii and high-accuracy "
                   "Voronoi decomposition; surface areas and pore volumes with a "
                   "1.655 A probe (checked against Zeo++ 0.3 in round 56)"),
        "si_zip_url": SI_ZIP_URL,
    }
    if len(argv) > 2:
        source["si_zip_sha256"] = hashlib.sha256(Path(argv[2]).read_bytes()).hexdigest()
    if OUT.is_file():
        # Rebuilt from the same file (round 61 added a column): the data
        # were retrieved when that file was, not today.
        with gzip.open(OUT, "rt", encoding="utf-8") as handle:
            previous = json.load(handle)["source"]
        if previous.get("sha256") == source["sha256"]:
            source["retrieved"] = previous.get("retrieved", source["retrieved"])
    payload = {"source": source, "columns": COLUMNS, "rows": rows}
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as handle:
        handle.write(text)
    print(f"{len(rows)} entries -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv)
