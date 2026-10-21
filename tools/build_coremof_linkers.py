"""Build thevoid_core/data/coremof2024_linkers.json.gz: the heavy-atom
make-up of each CoRE MOF 2024 entry's linkers, from the MOFid strings in
CR.json (round 61; the training data of the coremof-v1 models).

    python tools/build_coremof_linkers.py CR.json

For every entry with a MOFid v1 string: each distinct carbon-containing
building block of its SMILES part, as {element: count} of heavy atoms
(SMILES leave hydrogens implicit; the descriptors use heavy atoms only).
Metal nodes carry no carbon and are skipped, as the Void's
organic_fragments skips them. Same source, licence and citation as the
index (CC-BY-4.0, Zhao et al., Matter 2025); the SHA-256 of CR.json is
recorded.
"""

import gzip
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "thevoid_core" / "data" / "coremof2024_linkers.json.gz"


def fragments(mofid_v1: str) -> list | None:
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    if not isinstance(mofid_v1, str) or " MOFid" not in mofid_v1:
        return None
    seen, out = set(), []
    for part in mofid_v1.split(" MOFid")[0].split("."):
        if part in seen:
            continue
        seen.add(part)
        molecule = Chem.MolFromSmiles(part, sanitize=False)
        if molecule is None:
            continue
        counts = {}
        for atom in molecule.GetAtoms():
            counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
        if counts.get("C"):
            out.append(counts)
    return out


def main(argv):
    raw = Path(argv[1]).read_bytes()
    data = json.loads(raw)
    table = {}
    for subset in ("ASR", "FSR", "Ion"):
        for entry_id, record in data[subset].items():
            found = fragments((record.get("mofid") or {}).get("v1"))
            if found is not None:
                table[entry_id] = found
    payload = {
        "source": {"file": "CR.json (CoRE MOF 2024, computation-ready)",
                   "sha256": hashlib.sha256(raw).hexdigest(), "licence": "CC-BY-4.0",
                   "citation": "G. Zhao et al., Matter 8, 102140 (2025)",
                   "what": "heavy-atom counts of each distinct carbon-containing MOFid v1 block"},
        "linkers": table,
    }
    with gzip.open(OUT, "wt", encoding="utf-8", compresslevel=9) as handle:
        handle.write(json.dumps(payload, separators=(",", ":")))
    print(f"{len(table)} entries -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main(sys.argv)
