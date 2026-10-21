"""Build thevoid_core/data/rcsr_nets.json, the curated reference nets
topology identification matches against (round 53).

    python tools/build_topology_library.py RCSRnets.arc

The input is a Systre archive of RCSR nets (Systre's own RCSR.arc, or
the copy MOFid ships as Resources/RCSRnets.arc). For every net named
below this checks that the net is connected, stable and can be keyed by
thevoid_core.services.topology.net, then stores its quotient graph (the
archive's key, unchanged), its vertex count and its coordination
numbers. The RCSR database: M. O'Keeffe, M. A. Peskov, S. J. Ramsden,
O. M. Yaghi, Acc. Chem. Res. 41, 1782 (2008), http://rcsr.net.

Only the nets listed are bundled. A full archive can still be used at
run time (Settings / --rcsr-archive), see services/topology/library.py.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thevoid_core.services.topology.library import read_systre_archive  # noqa: E402
from thevoid_core.services.topology.net import parse_systre_key, prepare  # noqa: E402

#: name -> examples, only where the assignment is standard in the MOF
#: or zeolite literature. Every example marked * was identified from a real
#: structure file (RASPA2's bundled CIFs) in round 53; see
#: tests/test_topology_real_structures.py.
CURATED = {
    # 3-periodic, common in MOFs
    "pcu": "primitive cubic; MOF-5 / IRMOF series*, DMOF-1*",
    "dia": "diamond",
    "srs": "SrSi2",
    "bcu": "body-centred cubic",
    "fcu": "face-centred cubic; UiO-66*, UiO-68 (Zr-TPDC)*",
    "nbo": "NbO; MOF-505",
    "sod": "sodalite; ZIF-8*, ZIF-7*, ZIF-90*",
    "acs": "MIL-88A-D*",
    "cds": "CdSO4",
    "lvt": "",
    "qtz": "quartz",
    "ths": "ThSi2",
    "pts": "PtS; Co(BDP)*",
    "tbo": "twisted boracite; HKUST-1*",
    "pto": "MOF-14",
    "rtl": "rutile",
    "ftw": "",
    "soc": "",
    "csq": "NU-1000",
    "she": "",
    "stp": "",
    "the": "",
    "ssa": "",
    "ssb": "",
    "fof": "",
    # rht-type MOFs (PCN-61, NU-100) are not rht in the single-node
    # representation: pairs of paddlewheels share all four neighbours,
    # so that net is unstable and is reported as such (round 53).
    "rht": "",
    "qom": "MOF-177*, MOF-200*",
    "muo": "UMCM-1*",
    "ith": "",
    "ith-d": "MOF-205*",
    "mtn": "",
    "mtn-e": "MIL-101*",
    "moo": "MIL-100*",
    "kgd": "",
    "toz": "MOF-210*",
    "tfz-d": "",
    "bor": "boracite",
    "ctn": "C3N4",
    "flu": "fluorite",
    "scu": "",
    "sqc": "",
    "toc": "",
    "pth": "",
    "crs": "",
    "nia": "NiAs",
    "stb": "",
    "lon": "lonsdaleite",
    "unc": "",
    "bnn": "",
    "hms": "",
    "dmd": "",
    "tsi": "",
    # CALF-20 (the bundled fixture) with oxalate as a pillar between
    # single Zn nodes, round 55's carboxylate-cluster rule.
    "dmc": "CALF-20*",
    # zeolite and ZIF nets
    "lta": "Linde type A*; ZIF-20*",
    "rho": "RHO; ZIF-11*, ZIF-12*, ZIF-71*, ZIF-93*",
    "gis": "gismondine; ZIF-6*",
    "cag": "CaGa2O4; ZIF-4*",
    "dft": "DAF-2; ZIF-3*",
    "crb": "CrB4; ZIF-1*, ZIF-2*",
    "mer": "merlinoite; ZIF-10*",
    "gme": "gmelinite; ZIF-68*, ZIF-69*, ZIF-80*",
    "fau": "faujasite",
    "ana": "analcime",
    "cha": "chabazite*",
    "lcs": "",
    "zni": "Zn(imidazolate)2",
    "coi": "",
    "nog": "",
    "frl": "ZIF-77*",
    "gar": "garnet; ZIF-5*",
    "nou": "",
    "afi": "AFI; AlPO-5*",
    "lev": "levyne*",
    "eri": "erionite*",
    "kfi": "KFI*",
    "afx": "AFX*",
    # RCSR's "ats" is a different, (3,6,9)-c net; the ATS zeolite
    # framework is RCSR "rad" (checked against RASPA2's ATS.cif).
    "rad": "zeolite ATS*",
    # rod nets (only reached if a rod SBU is ever reduced to points)
    "sra": "SrAl2",
    "etb": "",
    "msf": "",
    # Round 70: nets of rod-SBU frameworks with their rods split at the
    # metals, as CrystalNets and CoRE MOF 2024 do. * here: identified by
    # the Void on CoRE MOF 2024 SI structures with that CoRE label.
    "bpq": "MIL-53*, MIL-47*, MIL-53(Sc/Fe/In/Al)* (single node)",
    "rna": "MIL-53*, MIL-47* (all-node)",
    "pts-x": "",
    "sea": "",
    "fgl": "",
    "ggl": "",
    "lcv": "",
    "tfz": "",
    "nbo-a": "",
    # 2-periodic
    "sql": "square lattice; MOF-2 layers (ZnBDC)*",
    "hcb": "honeycomb",
    "hxl": "hexagonal (triangular) lattice",
    "kgm": "kagome",
    "fes": "",
    "htb": "",
}


def main(path):
    archive = read_systre_archive(path)
    nets = []

    for name, description in CURATED.items():
        key = archive[name]
        prepared = prepare(parse_systre_key(key))
        nets.append({
            "name": name,
            "key": key,
            "dimension": prepared.net.dimension,
            "vertices": prepared.net.n_vertices,
            "degrees": sorted(len(v) for v in prepared.vectors),
            # The * marks are for this file's readers (verified against a
            # real structure); they are not shown in the application.
            "description": description.replace("*", ""),
        })
        print(f"{name:6} d={prepared.net.dimension} V={prepared.net.n_vertices}")

    out = Path(__file__).resolve().parent.parent / "thevoid_core" / "data" / "rcsr_nets.json"
    out.write_text(json.dumps({
        "source": "RCSR (rcsr.net; O'Keeffe, Peskov, Ramsden & Yaghi, Acc. "
                  "Chem. Res. 41, 1782 (2008)), quotient graphs as keyed by "
                  "Systre (Delgado-Friedrichs & O'Keeffe, Acta Cryst. A59, 351 "
                  "(2003)); built by tools/build_topology_library.py",
        "nets": nets,
    }, indent=1) + "\n")
    print(f"{len(nets)} nets -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
