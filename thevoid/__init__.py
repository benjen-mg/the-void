"""The Void's Python API (round 60).

    import thevoid

    fw = thevoid.load("CALF-20.cif")          # a path, pymatgen Structure or ASE Atoms
    fw.pores().pld_A                          # 2.799
    fw.surface_area("N2").channel_asa_m2_per_g
    fw.guest("CO2").status
    fw.topology().summary                     # "dmc 3-c" style summaries
    fw.compare_coremof().flagged              # rows that differ from CoRE MOF 2024

    table = thevoid.analyze_batch(paths, jobs=4, out="results", guests=["CO2"])
    table.to_dataframe()                      # with pandas installed
    thevoid.figure(table, "scatter", "pld_A", "channel_asa_m2_per_g", colour_by="topology")

    variants = fw.functionalize(["NH2", "Br"], per_linker=[1, 2])
    variants.write("CALF-20_functionalized")

    fw.predict()["water_stability"].value     # bundled models (needs scikit-learn)
    model = thevoid.train("results", "ref_uptake_CO2_0.1_298K", log=True)

    index = thevoid.Index.create("MyLab-index", name="McKelvey Lab")   # round 66
    index.add(table, contributor="B. G.")
    index.find(fw)

Every call runs the same code, with the same defaults, as the GUI and
the command-line tools. The same file therefore gives the same numbers
by any route, under the same names: result fields are named as the
JSON reports and summary CSV name them, units included.

Stability: the names in __all__, and the fields and methods documented
on the objects they return, are the public API. They change only with
a major version.
The `thevoid_core` package is internal, and so is any result's `raw`.
"""

from __future__ import annotations

from thevoid_core.version import __version__
from thevoid.batch import BatchTable, analyze_batch
from thevoid.figures import Figure, figure
from thevoid.framework import RADII, Framework, probe_radius
from thevoid.index import AddSummary, Index, IndexTable, Match
from thevoid.models import Model, PredictionResult, Predictions, load_model, mofxdb_labels, models, train
from thevoid.reference import fetch_coremof, fetch_mofxdb, search_coremof
from thevoid.report import Report, analysis_options
from thevoid.results import (
    ComparisonResult,
    GuestResult,
    OpenMetalSitesResult,
    PartitionResult,
    PoreResult,
    PoreSizeDistributionResult,
    SurfaceAreaResult,
    TopologyResult,
    XRDResult,
)
from thevoid.variants import Sites, Variant, Variants


def load(source, *, name: str | None = None) -> Framework:
    """A Framework from a file path (CIF, or any format pymatgen reads),
    a pymatgen Structure, or an ASE Atoms (converted; needs ASE)."""
    from thevoid._inputs import to_structure

    structure, path, space_group = to_structure(source)
    return Framework(structure, source=path, space_group=space_group, name=name)


def _probes():
    from thevoid_core.data.probe_radii import PROBE_RADII

    return dict(PROBE_RADII)


def _groups():
    from thevoid_core.services.functionalize import GROUPS

    return tuple(GROUPS)


#: Probe/guest name -> radius (Å), as every analysis uses them.
PROBES = _probes()

#: Functional groups functionalize() can place.
GROUPS = _groups()

__all__ = [
    "__version__",
    "load",
    "Framework",
    "analyze_batch",
    "BatchTable",
    "analysis_options",
    "Report",
    "figure",
    "Figure",
    "search_coremof",
    "fetch_coremof",
    "fetch_mofxdb",
    "probe_radius",
    "PROBES",
    "GROUPS",
    "RADII",
    "Sites",
    "Variant",
    "Variants",
    "models",
    "load_model",
    "train",
    "mofxdb_labels",
    "Model",
    "Predictions",
    "PredictionResult",
    "Index",
    "IndexTable",
    "AddSummary",
    "Match",
    "PoreResult",
    "SurfaceAreaResult",
    "GuestResult",
    "OpenMetalSitesResult",
    "TopologyResult",
    "PartitionResult",
    "PoreSizeDistributionResult",
    "XRDResult",
    "ComparisonResult",
]
