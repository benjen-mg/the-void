"""Round 66: the shared index from Python (thevoid.Index) and from a
report without a config (the GUI's).

The Python API must do exactly what the-void-index does, with the same
rules: a Report can be added only with the file it came from, a
BatchTable only once written to a folder, and the numbers read back are
the ones analysed. A GUI report has no config, so the index works one
out from the results: two GUI analyses with different probes must not
be compared as replicates.
"""

import json
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

import thevoid  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"
CU2 = FIXTURES / "identity" / "2022[Cu][sql]2[ASR]2.cif"
CU4 = FIXTURES / "identity" / "2022[Cu][sql]2[ASR]4.cif"

FAST = dict(surface_samples_per_atom=100, oms=False, topology=False)


@pytest.fixture(scope="module")
def table(tmp_path_factory):
    out = tmp_path_factory.mktemp("batch") / "results"
    return thevoid.analyze_batch([CU2, CU4, CR_BDC], out=out, **FAST)


def test_create_add_build_and_read_back(tmp_path, table):
    index = thevoid.Index.create(tmp_path / "idx", name="API Lab")
    summary = index.add(table, contributor="A. B.")
    assert isinstance(summary, thevoid.AddSummary)
    assert (len(summary.added), len(summary.present), len(summary.stored)) == (3, 0, 0)
    assert "3 withheld" in repr(summary)  # nothing says these copies may be shared
    assert len(index) == 3 and "API Lab" in repr(index)
    view = index.table()
    assert isinstance(view, thevoid.IndexTable) and len(view) == 3
    rows = {r["name"]: r for r in view.rows}
    assert rows[CR_BDC.name]["pld_A"] == pytest.approx(
        next(r for r in table.rows if r["file"] == CR_BDC.name)["pld_A"])
    assert rows[CU2.name]["structure_id"] == rows[CU4.name]["structure_id"]
    assert index.verify() == []
    again = index.add(table.folder, contributor="C. D.")
    assert len(again.added) == 0 and len(again.present) == 3


def test_the_table_rebuilds_when_entries_change(tmp_path, table):
    index = thevoid.Index.create(tmp_path / "idx", name="API Lab")
    index.add(table, contributor="A. B.")
    assert len(index.table()) == 3
    entry_id = index.table().rows[0]["entry_id"]
    index.withdraw(entry_id, reason="test", by="A. B.")
    assert len(index.table()) == 2


def test_find_a_framework(tmp_path, table):
    index = thevoid.Index.create(tmp_path / "idx", name="API Lab")
    index.add(table, contributor="A. B.")
    matches = index.find(thevoid.load(CU4))
    assert [(m.relation, m.file_name) for m in matches] == [
        ("same file", CU4.name), ("same structure", CU2.name)]
    assert isinstance(matches[0], thevoid.Match) and matches[0].contributor == "A. B."
    assert matches[0].summary["pld_A"] > 0
    from tests.sample_structures import ZN_DIA

    assert index.find(ZN_DIA) == []


def test_add_a_report_with_its_framework(tmp_path):
    index = thevoid.Index.create(tmp_path / "idx", name="API Lab")
    fw = thevoid.load(CR_BDC)
    report = fw.analyze(**FAST)
    summary = index.add(report, fw, contributor="A. B.", licence="CC-BY-4.0",
                        licence_basis="CoRE MOF 2024 SI")
    assert len(summary.added) == 1 and summary.stored == (CR_BDC.name,)
    with pytest.raises(ValueError, match="must come from a file"):
        index.add(report, thevoid.Framework(fw.structure), contributor="A. B.")


def test_an_unwritten_table_is_refused(tmp_path):
    index = thevoid.Index.create(tmp_path / "idx", name="API Lab")
    with pytest.raises(ValueError, match="write the BatchTable"):
        index.add(thevoid.BatchTable([], [], {}), contributor="A. B.")


def test_opening_something_else_fails_clearly(tmp_path):
    from thevoid_core.services.index.store import IndexProblem

    with pytest.raises(IndexProblem, match="not an index"):
        thevoid.Index(tmp_path)


# ----------------------------------------------------------------------
# Reports without a config (the GUI's)
# ----------------------------------------------------------------------


def _gui_report(probe):
    """A report as the main window builds it: new_report plus results,
    no config."""
    from pymatgen.core import Structure

    from thevoid_core.data.element_radii import ElementRadiusProvider
    from thevoid_core.services import report as reporting
    from thevoid_core.services.surface_area import analyze_surface_area

    structure = Structure.from_file(CR_BDC)
    report = reporting.new_report(structure, source=str(CR_BDC))
    radius = {"N2": 1.86, "CO2": 1.65}[probe]
    report["results"]["surface_area"] = reporting.surface_area_to_dict(analyze_surface_area(
        structure, probe_radius=radius, probe_name=probe, radius_provider=ElementRadiusProvider(),
        n_samples_per_atom=100))
    return report


def test_a_gui_report_gets_a_config_from_its_results(tmp_path):
    from thevoid_core.services.index.entries import inferred_config, make_entry

    n2, co2 = _gui_report("N2"), _gui_report("CO2")
    assert "config" not in n2
    config = inferred_config(n2)
    assert config["inferred_from_results"] is True
    assert config["analyses"]["surface_area"]["probe"] == "N2"
    assert config["analyses"]["surface_area"]["samples_per_atom"] == 100
    a = make_entry(n2, CR_BDC, contributor="A. B.")
    b = make_entry(co2, CR_BDC, contributor="A. B.")
    assert a["analysis"]["method_key"] != b["analysis"]["method_key"]
    index = thevoid.Index.create(tmp_path / "idx", name="GUI Lab")
    index.raw.add_report(n2, CR_BDC, contributor="A. B.")
    index.raw.add_report(co2, CR_BDC, contributor="A. B.")
    # Same file, different settings: not replicates, so no disagreement.
    assert index.verify() == []
    entries = [json.loads(f.read_text()) for f in index.raw.entry_files()]
    assert all(e["analysis"]["summary"]["channel_asa_m2_per_g"] > 0 for e in entries)


def test_a_report_with_no_results_is_refused(tmp_path):
    from pymatgen.core import Structure

    from thevoid_core.services import report as reporting

    index = thevoid.Index.create(tmp_path / "idx", name="GUI Lab")
    empty = reporting.new_report(Structure.from_file(CR_BDC), source=str(CR_BDC))
    with pytest.raises(ValueError, match="run an analysis first"):
        index.raw.add_report(empty, CR_BDC, contributor="A. B.")
