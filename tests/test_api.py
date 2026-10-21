"""Round 60: the Python API (`import thevoid`).

An API is a promise. The checks here keep it:

* same numbers: a file gives exactly what the-void-analyze reports for
  it, and the same crystal read through ASE gives the same numbers as
  its CIF, surface area included;
* same names: result fields are the report's keys, and to_dict() is the
  report's dictionary;
* no silent mistakes: an unknown option, probe, guest, group or input
  type is an error that says what is valid;
* the batch writes the folder the Batch Results window and
  the-void-figures read, and parallel equals serial;
* the public surface is pinned (test_the_public_api_is_what_it_says), so
  it cannot change without this test and the docs changing with it.
"""

import inspect
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

import thevoid  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
from tests.sample_structures import ZN_DIA  # noqa: E402

CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"
CU = FIXTURES / "coremof" / "2015[Cu][nan]3[ASR]1.cif"


@pytest.fixture(scope="module")
def sample():
    """An openly licensed Zn MOF, the general sample since round 75."""
    return thevoid.load(ZN_DIA)


@pytest.fixture(scope="module")
def cr():
    return thevoid.load(CR_BDC)


# ----------------------------------------------------------------------
# The public surface
# ----------------------------------------------------------------------

PUBLIC = {
    "load": ["source", "name"],
    "analyze_batch": ["inputs", "jobs", "out", "reference_csv", "figures", "progress", "options"],
    "figure": ["batch", "kind", "x", "y", "colour_by", "log_x", "log_y", "bins", "title",
               "width", "filters", "save"],
    "search_coremof": ["subset", "pld_min", "pld_max", "lcd_min", "lcd_max", "asa_min", "metals",
                       "topology", "has_oms", "downloadable_only", "limit"],
    "fetch_coremof": ["folder", "limit", "progress", "filters"],
    "fetch_mofxdb": ["folder", "database", "limit", "progress", "filters"],
    "analysis_options": [],
    "probe_radius": ["probe"],
    "models": [],
    "load_model": ["source"],
    "train": ["results", "target", "log", "name", "label", "unit", "seed", "method_key"],
    "mofxdb_labels": ["folder", "adsorbate", "pressure", "temperature", "out"],
}

FRAMEWORK_METHODS = {
    "pores": ["probe", "grid_spacing", "radii"],
    "surface_area": ["probe", "samples_per_atom", "seed", "radii"],
    "guest": ["guest", "diameter", "grid_spacing", "radii"],
    "open_metal_sites": [],
    "topology": ["rcsr_archive"],
    "partition": [],
    "pore_size_distribution": ["bin_width", "grid_spacing", "radii"],
    "xrd": ["wavelength", "two_theta_range", "min_relative_intensity", "merge_tolerance"],
    "coremof_matches": [],
    "compare_coremof": ["entry", "samples_per_atom"],
    "analyze": ["options"],
    "functionalization_sites": ["max_per_linker"],
    "functionalize": ["groups", "per_linker", "orbits", "ring_turn"],
    "write_cif": ["path"],
    "raspa_input": ["folder", "settings"],
    "to_pymatgen": [],
    "to_ase": [],
    "summary": [],
    "clear_cache": [],
    "predict": ["models", "samples_per_atom"],
}


def _parameters(function):
    return [p for p in inspect.signature(function).parameters if p != "self"]


def test_the_public_api_is_what_it_says():
    """If this fails, the public API has changed: update
    docs/python-api.md and this list together (a major version)."""
    assert sorted(thevoid.__all__) == sorted([
        "__version__", "load", "Framework", "analyze_batch", "BatchTable", "analysis_options",
        "Report", "figure", "Figure", "search_coremof", "fetch_coremof", "fetch_mofxdb",
        "probe_radius", "PROBES", "GROUPS", "RADII", "Sites", "Variant", "Variants",
        "PoreResult", "SurfaceAreaResult", "GuestResult", "OpenMetalSitesResult",
        "TopologyResult", "PartitionResult", "PoreSizeDistributionResult", "XRDResult",
        "ComparisonResult", "models", "load_model", "train", "mofxdb_labels", "Model",
        "Predictions", "PredictionResult", "Index", "IndexTable", "AddSummary", "Match"])
    for name in thevoid.__all__:
        assert hasattr(thevoid, name), name
    for name, parameters in PUBLIC.items():
        assert _parameters(getattr(thevoid, name)) == parameters, name
    for name, parameters in FRAMEWORK_METHODS.items():
        assert _parameters(getattr(thevoid.Framework, name)) == parameters, name
    for name in ("structure", "formula", "n_sites", "space_group", "density_g_cm3",
                 "cell_volume_A3", "bonds"):
        assert isinstance(inspect.getattr_static(thevoid.Framework, name), property), name
    from thevoid_core.version import __version__

    assert thevoid.__version__ == __version__
    assert set(thevoid.PROBES) == {"N2", "CO2", "H2", "H2O", "CH4"}
    assert "NH2" in thevoid.GROUPS and len(thevoid.GROUPS) == 11


def test_every_public_thing_is_documented():
    for name in thevoid.__all__:
        obj = getattr(thevoid, name)
        if callable(obj):
            assert inspect.getdoc(obj), name
    for name in FRAMEWORK_METHODS:
        assert inspect.getdoc(getattr(thevoid.Framework, name)), name


# ----------------------------------------------------------------------
# Same numbers, same names
# ----------------------------------------------------------------------


def test_the_api_gives_exactly_what_the_cli_reports(sample, tmp_path):
    from thevoid_core import cli

    cli.main([str(ZN_DIA), "--out", str(tmp_path), "-q", "--samples", "300",
              "--guests", "CO2", "--psd"])
    report = json.loads((tmp_path / f"{ZN_DIA.stem}.json").read_text())["results"]
    assert sample.pores().to_dict() == report["pore"]
    assert sample.surface_area(samples_per_atom=300).to_dict() == report["surface_area"]
    assert sample.guest("CO2").to_dict() == report["guests"]["CO2"]
    assert sample.open_metal_sites().to_dict() == report["oms"]
    assert sample.pore_size_distribution().to_dict() == report["psd"]
    assert json.loads(json.dumps(sample.topology().to_dict())) == report["topology"]


def test_fields_are_the_reports_keys(sample):
    pores = sample.pores()
    for key, value in pores.to_dict().items():
        if key == "grid_shape":
            assert tuple(value) == pores.grid_shape
        else:
            assert getattr(pores, key) == value, key
    assert pores.lcd_A == pytest.approx(5.1096, abs=1e-3)
    assert pores.pld_A == pytest.approx(4.0778, abs=1e-3)
    assert "raw" not in repr(pores) and "details" not in repr(pores)


def test_results_are_cached_and_frozen(sample):
    assert sample.pores() is sample.pores()
    assert sample.pores(1.2) is sample.pores()        # the default is 1.2 A
    assert sample.pores("N2") is not sample.pores()
    with pytest.raises(AttributeError):
        sample.pores().pld_A = 3.0


def test_the_same_crystal_through_ase_gives_the_same_numbers(cr):
    """Round 60 found that surface sampling depended on how the cell's
    axes lie in space (ASE puts a along x; a CIF is read in pymatgen's
    standard orientation): 1450 vs 1505 m2/g for the same Cr-BDC and
    seed. The sampling is now drawn in the cell's own frame, so every
    number agrees; CIF-read structures sample exactly as before."""
    ase_io = pytest.importorskip("ase.io")
    atoms = ase_io.read(str(CR_BDC))
    other = thevoid.load(atoms, name="from ASE")
    assert other.pores().pld_A == pytest.approx(cr.pores().pld_A, abs=1e-6)
    assert other.surface_area(samples_per_atom=300).channel_asa_m2_per_g == pytest.approx(
        cr.surface_area(samples_per_atom=300).channel_asa_m2_per_g, rel=1e-9)
    back = cr.to_ase()
    assert len(back) == cr.n_sites and back.pbc.all()


def test_a_rotated_cell_samples_the_same_surface(cr):
    """The same point, without ASE: rotate the whole crystal."""
    from pymatgen.core import Lattice, Structure
    from scipy.spatial.transform import Rotation

    s = cr.structure
    rotation = Rotation.from_euler("xyz", [23, -41, 67], degrees=True).as_matrix()
    turned = Structure(Lattice(s.lattice.matrix @ rotation.T), s.species, s.frac_coords)
    a = cr.surface_area(samples_per_atom=200, seed=3)
    b = thevoid.load(turned).surface_area(samples_per_atom=200, seed=3)
    assert b.accessible_surface_area_A2 == pytest.approx(a.accessible_surface_area_A2, rel=1e-9)


def test_the_framework_does_not_share_the_callers_structure(cr):
    s = cr.structure
    fw = thevoid.load(s)
    s.remove_sites([0])
    assert fw.n_sites == cr.n_sites


def test_ccdc_radii_reproduce_the_coremof_convention(sample):
    """radii="ccdc": the numbers the CoRE comparison computes. CoRE MOF
    2024 (Zeo++, -ha) lists PLD 4.149 and LCD 5.158 A for this structure;
    the Void's exact values sit a few hundredths below (round 56)."""
    pores = sample.pores(1.655, radii="ccdc")
    assert pores.pld_A == pytest.approx(4.1155, abs=2e-3)
    assert -0.06 < pores.pld_A - 4.14913 < 0 and -0.06 < pores.lcd_A - 5.15766 < 0
    with pytest.raises(ValueError, match="radii"):
        sample.pores(radii="bondi")


# ----------------------------------------------------------------------
# Errors that say what is valid
# ----------------------------------------------------------------------


def test_mistakes_are_errors_not_silence(sample, tmp_path):
    with pytest.raises(ValueError, match="Unknown probe"):
        sample.pores("argon")
    with pytest.raises(ValueError, match="negative"):
        sample.pores(-1.0)
    with pytest.raises(ValueError, match="Unknown guest"):
        sample.guest("argon")
    assert sample.guest("argon", diameter=3.4).guest == "argon"
    with pytest.raises(TypeError, match="Unknown analysis option"):
        sample.analyze(pore_probe=1.0)
    with pytest.raises(ValueError, match="Unknown group"):
        sample.functionalize("SH")
    with pytest.raises(TypeError, match="give a file path"):
        thevoid.load(42)
    with pytest.raises(FileNotFoundError):
        thevoid.load(tmp_path / "missing.cif")
    with pytest.raises(TypeError, match="Unknown RASPA setting"):
        sample.raspa_input(tmp_path / "r", temperature=300)


# ----------------------------------------------------------------------
# Reports and batches
# ----------------------------------------------------------------------


def test_analyze_is_run_analysis(sample, tmp_path):
    report = sample.analyze(surface_samples_per_atom=200, guests="CO2,N2", oms=False)
    assert report.ok and set(report["results"]) >= {"pore", "surface_area", "guests", "topology"}
    assert report.row["guest_CO2"] == sample.guest("CO2").status == "accessible"
    assert report.row["pld_A"] == sample.pores().pld_A
    assert json.loads(report.write(tmp_path / "r.json").read_text())["input"]["sha256"]
    assert report.write(tmp_path / "r.md").read_text().startswith("#")
    with pytest.raises(ValueError):
        report.write(tmp_path / "r.txt")
    assert thevoid.analysis_options()["pore_probe_radius"] == 1.2


def test_batch_writes_the_folder_the_rest_of_the_void_reads(cr, tmp_path):
    from thevoid_core.services.batch_results import load_batch_results

    inputs = [CR_BDC, thevoid.load(CU), cr.structure, "no-such-file.cif"]
    options = dict(surface_samples_per_atom=100, oms=False, topology=False)
    table = thevoid.analyze_batch(inputs, out=tmp_path / "b", **options)
    assert len(table) == 4 and set(table.errors) == {"no-such-file"}
    rows = table.rows
    assert rows[0]["file"] == CR_BDC.name
    assert rows[2]["file"].endswith(".cif")
    assert (tmp_path / "b" / rows[2]["file"]).is_file()      # in-memory: written as a CIF
    assert "could not read" in rows[3]["errors"]
    results = load_batch_results(tmp_path / "b")
    assert [r.get("file") for r in results.rows] == [r["file"] for r in rows]
    assert results.rows[0].report_json is not None
    assert rows[0]["pld_A"] == pytest.approx(cr.pores().pld_A)


def test_parallel_equals_serial(tmp_path):
    options = dict(surface_samples_per_atom=100, oms=False, topology=False)
    serial = thevoid.analyze_batch([CR_BDC, CU], **options)
    parallel = thevoid.analyze_batch([CR_BDC, CU], jobs=2, **options)
    assert serial.rows == parallel.rows


def test_names_that_would_collide_are_refused(tmp_path):
    other = tmp_path / "x"
    other.mkdir()
    shutil.copy(CR_BDC, other / CR_BDC.name)
    with pytest.raises(ValueError, match="share a name"):
        thevoid.analyze_batch([CR_BDC, other / CR_BDC.name], pore=False, surface_area=False)


def test_pandas_is_optional(monkeypatch, tmp_path):
    import builtins

    table = thevoid.analyze_batch([CR_BDC], pore=False, surface_area=False, oms=False,
                                  topology=False)
    real_import = builtins.__import__

    def no_pandas(name, *args, **kwargs):
        if name == "pandas":
            raise ImportError("No module named 'pandas'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pandas)
    with pytest.raises(ImportError, match="pip install pandas"):
        table.to_dataframe()
    monkeypatch.undo()
    assert table.to_csv(tmp_path / "s.csv").read_text().startswith("file,")
    pd = pytest.importorskip("pandas")
    frame = table.to_dataframe()
    assert isinstance(frame, pd.DataFrame) and list(frame.columns)[0] == "file"


def test_figure_from_a_table_and_from_a_folder(tmp_path):
    table = thevoid.analyze_batch([CR_BDC, CU], out=tmp_path / "b", surface_samples_per_atom=100,
                                  oms=False, topology=False)
    fig = thevoid.figure(table, "scatter", "pld_A", "channel_asa_m2_per_g",
                         save=tmp_path / "f")
    assert len(fig.points) == 2
    assert sorted(p.suffix for p in tmp_path.glob("f.*")) == [".json", ".pdf", ".png", ".svg"]
    again = thevoid.figure(tmp_path / "b", "histogram", "pld_A", filters=["pld_A >= 6"])
    assert len(again.points) == 1 and again._repr_png_()[:4] == b"\x89PNG"
    memory = thevoid.figure(thevoid.analyze_batch([CR_BDC], surface_samples_per_atom=100,
                                                  oms=False, topology=False), "histogram", "pld_A")
    assert len(memory.points) == 1


# ----------------------------------------------------------------------
# Building, and the rest
# ----------------------------------------------------------------------


def test_functionalize_from_python(cr, tmp_path):
    sites = cr.functionalization_sites()
    assert sites.count == 8 and sites.orbits[0]["patterns"][2] == ["ortho", "para", "meta"]
    assert "Imma" in str(sites)
    variants = cr.functionalize(["NH2", "F"], per_linker=[1, 2])
    assert len(variants) == 8
    assert {v.status for v in variants} <= {"ok", "tight", "clash"}
    accepted = variants.accepted()
    assert accepted and all(v.status != "clash" for v in accepted)
    fw = accepted[0].framework
    assert isinstance(fw, thevoid.Framework) and "N" in fw.formula
    assert fw.pores().pld_A < cr.pores().pld_A      # -NH2 narrows the channel
    folder = variants.write(tmp_path / "v")
    assert (folder / "variants.csv").is_file() and (folder / "functionalization.json").is_file()
    assert len(list(folder.glob("*_L1_*.cif"))) == len(accepted)
    assert variants.rows[0]["group"] == "NH2"


def test_search_coremof_offline():
    hits = thevoid.search_coremof(metals="Cu", pld_min=6, downloadable_only=True, limit=3)
    assert len(hits) == 3
    assert all("Cu" in h["metals"] and h["pld_A"] >= 6 and h["source"] == "SI" for h in hits)


def test_coremof_matches_and_raspa_and_cif(sample, tmp_path):
    best = sample.coremof_matches()[0]
    assert best[:2] == ("2020[Zn][dia]3[ASR]1", "identified")
    folder = sample.raspa_input(tmp_path / "raspa", adsorbate="CO2", cycles=10,
                              initialization_cycles=10, pressures_Pa=[1e4, 1e5])
    assert (folder / "simulation.input").is_file()
    path = sample.write_cif(tmp_path / "c.cif")
    assert thevoid.load(path).formula == sample.formula


def test_xrd_and_psd_are_arrays_of_plain_numbers(sample):
    xrd = sample.xrd()
    assert xrd.relative_intensity and max(xrd.relative_intensity) == pytest.approx(100.0)
    assert all(type(v) is float for v in xrd.two_theta_deg)
    strongest = xrd.strongest(3)
    assert len(strongest) == 3 and strongest == sorted(strongest)
    psd = sample.pore_size_distribution()
    assert len(psd.diameters_A) == len(psd.distribution_per_A) == len(psd.cumulative_volume_A3)
    assert psd.dominant_diameter_A == pytest.approx(4.1, abs=0.2)  # cages below the 5.11 A LCD
    assert not math.isnan(psd.void_fraction)
    assert np.all(np.diff(psd.diameters_A) > 0)
