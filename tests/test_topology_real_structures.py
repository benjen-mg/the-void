"""Round 53: topology of real, published frameworks.

RASPA2's source distribution ships ~108 MOF and 54 zeolite CIFs whose
nets are known from the literature (Park et al., PNAS 103, 10186 (2006)
and Banerjee et al., Science 319, 939 (2008) for the ZIFs; the
original papers for the MOFs). They have no stated licence, so they are
not copied into this repository: these tests run when the files are
available - RASPA_STRUCTURES pointing at RASPA2-2.0.4/structures, as
the CI workflow sets - and are skipped otherwise.

Names are compared as RCSR codes. RCSR's name for a zeolite framework
is usually the IZA code in lower case, but not always: the IZA ATS
framework is RCSR "rad" (RCSR's "ats" is an unrelated (3,6,9)-c net).
"""

import os
import warnings
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.topology import analyze_topology  # noqa: E402


def _root():
    for candidate in (os.environ.get("RASPA_STRUCTURES"),
                      "/tmp/raspa/RASPA2-2.0.4/structures"):
        if candidate and Path(candidate, "mofs", "cif").is_dir():
            return Path(candidate)
    return None


ROOT = _root()
pytestmark = pytest.mark.skipif(ROOT is None, reason="RASPA2 structures not available")

#: file -> (summary the Void must report). Every entry was checked
#: against the literature assignment when this table was written.
EXPECTED = {
    # MOFs
    "IRMOF-1": "pcu 6-c",
    "IRMOF-10": "pcu 6-c",
    "IRMOF-16": "pcu 6-c",
    "IRMOF-9": "pcu 6-c, 2-fold interpenetrated",
    "IRMOF-11": "pcu 6-c, 2-fold interpenetrated",
    "IRMOF-13": "pcu 6-c, 2-fold interpenetrated",
    "IRMOF-15": "pcu 6-c, 2-fold interpenetrated",
    "ZnBDCdabco": "pcu 6-c",
    "Cu-BTC": "tbo (3,4)-c",
    "UIO-66": "fcu 12-c",
    "Zr-TPDC": "fcu 12-c",
    "MIL-88A-open": "acs 6-c",
    "MIL-88B-dry": "acs 6-c",
    "MOF-177_ortho": "qom (3,6)-c",
    "MOF-200": "qom (3,6)-c",
    "MOF-205": "ith-d (3,6)-c",
    "MOF-210-primitive": "toz (3,6)-c",
    "UMCM-1": "muo (3,6)-c",
    "MIL-100-primitive": "moo (3,6)-c",
    "MIL-101-primitive": "mtn-e 6-c",
    "ZnBDC": "sql 4-c",
    "CoBDP": "pts 4-c",
    # ZIFs
    "ZIF-1": "crb 4-c",
    "ZIF-2": "crb 4-c",
    "ZIF-3": "dft 4-c",
    "ZIF-4": "cag 4-c",
    "ZIF-5": "gar (4,6)-c",
    "ZIF-6": "gis 4-c",
    "ZIF-7": "sod 4-c",
    "ZIF-8": "sod 4-c",
    "ZIF-9": "sod 4-c",
    "ZIF-10": "mer 4-c",
    "ZIF-11": "rho 4-c",
    "ZIF-12": "rho 4-c",
    "ZIF-68": "gme 4-c",
    "ZIF-69": "gme 4-c",
    "ZIF-71": "rho 4-c",
    "ZIF-77": "frl 4-c",
    "ZIF-80": "gme 4-c",
    "ZIF-90": "sod 4-c",
    # zeolites (T-atom nets)
    "CHA_SI": "cha 4-c",
    "GME": "gme 4-c",
    "ITQ-29": "lta 4-c",
    "SOD": "sod 4-c",
    "AFI_SI": "afi 4-c",
    "AlPO-5": "afi 4-c",
    "LEV": "lev 4-c",
    "ERI_SI": "eri 4-c",
    "KFI": "kfi 4-c",
    "AFX_SI": "afx 4-c",
    "ATS": "rad 4-c",
    # Round 70: rod SBUs split at their metals, as CrystalNets and CoRE
    # MOF 2024 do. MIL-53 and MIL-47 are bpq, as CoRE labels them.
    # CoRE assigns MOF-74 no net; nor does RCSR have this one.
    "MIL-53(Cr)lt": "bpq (4,6)-c",
    "MIL-47": "bpq (4,6)-c",
    "MgMOF-74": "(6,8)-c net of split rod SBUs, not in the reference library",
    # reported, not identified
    "NU-100SP-primitive": "not identified: unstable net",
    "PCN-68-primitive": "not identified: unstable net",
}


def _load(name):
    path = next(ROOT.glob(f"*/cif/{name}.cif"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PymatgenCIFLoader().load(str(path)).structure


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_published_topology_is_reproduced(name):
    assert analyze_topology(_load(name)).summary == EXPECTED[name]


def test_rht_type_mofs_explain_why_they_are_unstable():
    """Single-node rht MOFs have pairs of paddlewheels bonded to exactly
    the same four linkers; the note names them."""
    result = analyze_topology(_load("NU-100SP-primitive"))
    assert any("C4Cu2O8" in note and "rht" in note for note in result.notes)


def test_a_bad_bond_length_in_the_file_is_surfaced():
    """RASPA2's PCN-61 model has a 1.042 A C-C triple bond (real: 1.20),
    below the bond detector's floor, so the linker falls apart. The
    result must not claim a net; the framework note says why to look."""
    result = analyze_topology(_load("PCN-61"))
    assert result.rcsr is None
    assert any("No bonded component extends periodically" in n for n in result.notes)
