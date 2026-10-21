"""Round 59: linker functionalisation (services/functionalize.py,
the-void-functionalize, Build > Functionalise Linkers...).

A hypothetical variant is only useful if it is a structure a chemist
would accept. The checks here are therefore about chemistry and
bookkeeping, not only that a file appears:

* each group has the bond lengths and angles it claims, and a conjugated
  group is planar when there is room;
* only ring C-H on linkers are offered: not methyl H, not solvent, and
  none on an oxalate;
* patterns are counted once up to symmetry (terephthalate: 1 / 3 / 1 / 1
  for 1-4 groups), and "uniform" means every linker gets the pattern;
* the strip-and-rebuild check on real structures (CoRE MOF 2024): a
  real -Br (MIL-47(V)-Br) and -NH2 (a Ni dia MOF) are replaced by H and
  rebuilt within 0.15 A of where they really are, and pass the check;
* contacts: overlaps are rejected, hydrogen bonds are not clashes, and
  the bond detector sees exactly the bonds each group was placed with;
* the files: the CIF reads back, variants.csv (with the parent) joins
  into a batch with --reference-csv, and the JSON records the method.
"""

import json
import math
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from thevoid_core.services import functionalize as fz  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
CR_BDC = FIXTURES / "coremof" / "2010[Cr][bpq]3[ASR]1.cif"
V_BR_BDC = FIXTURES / "coremof" / "2019[V][bpq]3[ASR]1.cif"
NI_NH2 = FIXTURES / "coremof" / "2023[Ni][dia]3[ASR]1.cif"
RASPA_STRUCTURES = os.environ.get("RASPA_STRUCTURES")


def _load(path):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PymatgenCIFLoader().load(str(path)).structure


@pytest.fixture(scope="module")
def cr_parent():
    return fz.analyze_parent(_load(CR_BDC), source=str(CR_BDC), max_per_linker=4)


# ----------------------------------------------------------------------
# Group geometry
# ----------------------------------------------------------------------


def _angle(a, b, c):
    u, v = a - b, c - b
    return math.degrees(math.acos(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v)))


def test_groups_have_the_geometry_they_record():
    ring_c = np.zeros(3)
    nh2 = fz.GROUPS["NH2"].build(0.0)
    n, h1, h2 = (a[2] for a in nh2)
    assert np.linalg.norm(n) == pytest.approx(1.380)
    assert np.linalg.norm(h1 - n) == pytest.approx(1.01)
    assert _angle(h1, n, h2) == pytest.approx(120.0)
    assert _angle(ring_c, n, h1) == pytest.approx(120.0)
    assert abs(h1[2]) < 1e-12 and abs(h2[2]) < 1e-12  # phi 0: in the ring plane (w = 0)

    no2 = fz.GROUPS["NO2"].build(90.0)
    n, o1, o2 = (a[2] for a in no2)
    assert _angle(o1, n, o2) == pytest.approx(124.0)
    assert np.linalg.norm(o1 - n) == pytest.approx(1.217)

    ch3 = fz.GROUPS["CH3"].build(37.0)
    c, *hs = (a[2] for a in ch3)
    assert all(_angle(ring_c, c, h) == pytest.approx(109.4712, abs=1e-3) for h in hs)
    assert _angle(hs[0], c, hs[1]) == pytest.approx(109.4712, abs=1e-3)

    ome = fz.GROUPS["OCH3"].build(0.0)
    o, cm, *hm = (a[2] for a in ome)
    assert _angle(ring_c, o, cm) == pytest.approx(117.5)
    assert all(np.linalg.norm(h - cm) == pytest.approx(1.09) for h in hm)
    assert all(_angle(o, cm, h) == pytest.approx(109.4712, abs=1e-3) for h in hm)

    cn = fz.GROUPS["CN"].build(0.0)
    assert np.allclose(cn[1][2], [1.443 + 1.136, 0, 0])
    # Every atom hangs from the ring carbon or an earlier atom, one bond deeper.
    for name, grp in fz.GROUPS.items():
        atoms = grp.build(0.0)
        for element, depth, _pos, host in atoms:
            assert depth == (1 if host < 0 else atoms[host][1] + 1), name


def test_group_names_are_found_either_way():
    assert fz.group("nh2").name == "NH2"
    assert fz.group("amino").name == "NH2"
    with pytest.raises(ValueError, match="Unknown group"):
        fz.group("SH")


# ----------------------------------------------------------------------
# Sites, orbits, patterns
# ----------------------------------------------------------------------


def test_terephthalate_patterns_up_to_symmetry(cr_parent):
    """Cr-BDC (Imma): two BDC linkers per cell, one orbit, four ring C-H
    each; up to symmetry, one pattern for one group, ortho / meta / para
    for two, one each for three and four."""
    assert cr_parent.space_group == "Imma"
    assert len(cr_parent.orbits) == 1
    orbit = cr_parent.orbits[0]
    assert orbit.formula == "C8H4O4" and len(orbit.linkers) == 2 and len(orbit.sites) == 4
    assert [len(orbit.patterns[m]) for m in (1, 2, 3, 4)] == [1, 3, 1, 1]
    assert sorted(d for _, d in orbit.patterns[2]) == ["meta", "ortho", "para"]


def test_uniform_means_every_linker_gets_the_pattern(cr_parent):
    orbit = cr_parent.orbits[0]
    for m, patterns in orbit.patterns.items():
        for pattern, _ in patterns:
            replaced = fz.pattern_sites(cr_parent, orbit, pattern)
            per_linker = {}
            for h in replaced:
                per_linker[cr_parent.sites[h].linker] = per_linker.get(cr_parent.sites[h].linker, 0) + 1
            assert per_linker == {lid: m for lid in orbit.linkers}


def test_methyl_hydrogens_are_not_ring_hydrogens():
    """The Ni dia MOF's -NH2 hydrogens and the V-Br's are not offered,
    and a CH3 put on Cr-BDC does not make its own H substitutable."""
    parent = fz.analyze_parent(_load(NI_NH2), max_per_linker=1)
    symbols = [s.specie.symbol for s in parent.structure.sites]
    for h, site in parent.sites.items():
        assert symbols[site.carbon] == "C" and site.ring_size == 6
    cr = fz.analyze_parent(_load(CR_BDC), max_per_linker=1)
    variant = fz.build_variant(cr, fz.GROUPS["CH3"], cr.orbits[0], cr.orbits[0].patterns[1][0][0])
    again = fz.analyze_parent(variant.structure, max_per_linker=1)
    assert len(again.sites) == len(cr.sites) - len(variant.replaced)


# ----------------------------------------------------------------------
# Placement against real structures
# ----------------------------------------------------------------------


def _strip_and_rebuild(path, name, anchor):
    """Replace every real ring-bound `anchor` group by an H on the ring
    bisector, rebuild it here, and return (variant, deviations of the
    rebuilt first atom from the real one)."""
    from pymatgen.core import Structure

    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

    s = _load(path)
    s = Structure(s.lattice, [x.species for x in s], s.frac_coords % 1.0,
                  labels=[x.label for x in s])
    adjacency = fz._adjacency(len(s), CovalentBondDetector().detect(s))
    symbols = [x.specie.symbol for x in s]
    lattice = s.lattice.matrix
    remove, real, new_h = set(), [], []
    species, coords = [], []
    for x, element in enumerate(symbols):
        if element != anchor:
            continue
        ring = [(w, img) for w, img in adjacency[x] if symbols[w] == "C"
                and fz._smallest_ring(w, adjacency, symbols) == 6]
        if len(ring) != 1:
            continue
        c, image = ring[0]
        hangers = [w for w, _ in adjacency[x] if w != c]
        remove.update([x] + hangers)
        centre = s[c].coords
        neighbours = [(s[w].frac_coords + np.array(img)) @ lattice - centre
                      for w, img in adjacency[c] if w != x]
        u = -sum(v / np.linalg.norm(v) for v in neighbours)
        u /= np.linalg.norm(u)
        real.append((centre + 0.95 * u, (s[x].frac_coords - np.array(image)) @ lattice))
    keep = [i for i in range(len(s)) if i not in remove]
    species = [s[i].species for i in keep] + ["H"] * len(real)
    coords = [s[i].coords for i in keep] + [h for h, _ in real]
    stripped = Structure(s.lattice, species, coords, coords_are_cartesian=True)
    parent = fz.analyze_parent(stripped, max_per_linker=1)
    hydrogens = list(range(len(keep), len(keep) + len(real)))
    assert set(hydrogens) <= set(parent.sites)
    variant = fz.build_variant(parent, fz.GROUPS[name], None, (), hydrogens=hydrogens)
    stride = len(fz.GROUPS[name].build(0.0))
    first = len(keep)  # new atoms follow the kept sites, one group per hydrogen, in order
    deviations = []
    for k, (_h, x_real) in enumerate(real):
        delta = variant.structure[first + k * stride].coords - x_real
        delta -= np.rint(delta @ np.linalg.inv(lattice)) @ lattice
        deviations.append(float(np.linalg.norm(delta)))
    return variant, deviations


@pytest.mark.parametrize("path, name, anchor", [(V_BR_BDC, "Br", "Br"), (NI_NH2, "NH2", "N")])
def test_real_groups_are_rebuilt_where_they_are(path, name, anchor):
    """MIL-47(V)-Br (doi:10.1039/C8DT04214D) and a Ni -NH2 dia MOF
    (doi:10.1039/D2TA06863J): each real group replaced by H and rebuilt.
    On 160 such CoRE structures the median deviation is 0.08 A; these
    two are 0.08 A each."""
    variant, deviations = _strip_and_rebuild(path, name, anchor)
    assert variant.status == "ok", variant.closest
    assert max(deviations) < 0.15


def test_a_conjugated_group_leaves_the_plane_only_when_it_must(cr_parent):
    """-NH2 on Cr-BDC: planar, one N-H would sit 1.55 A from the H of the
    linker facing it across the channel (65% of the vdW sum), so the
    amine is turned out of the plane instead; on IRMOF-1, with room, it
    stays planar (test_irmof1_gives_irmof3_and_irmof2)."""
    orbit = cr_parent.orbits[0]
    nh2 = fz.build_variant(cr_parent, fz.GROUPS["NH2"], orbit, orbit.patterns[1][0][0])
    assert nh2.status == "ok"
    assert all(r not in (0.0, 180.0) for r in nh2.rotors)
    assert nh2.closest[0][0] >= fz.TIGHT_RATIO


def test_facing_groups_are_rejected(cr_parent):
    """Cr-BDC's narrow channel puts the meta positions of neighbouring
    linkers face to face: two -CN there overlap, and the flat build says
    so rather than writing the file. (Round 71: with the rings turned,
    25 degrees each, they clear - see test_ring_turns.py.)"""
    orbit = cr_parent.orbits[0]
    meta = [p for p, d in orbit.patterns[2] if d == "meta"][0]
    variant = fz.build_variant(cr_parent, fz.GROUPS["CN"], orbit, meta, ring_turn=False)
    assert variant.status == "clash"
    assert "vdW sum" in variant.reason or "bond" in variant.reason


def test_the_bond_detector_sees_exactly_the_bonds_placed(cr_parent):
    from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector

    orbit = cr_parent.orbits[0]
    for name in ("NH2", "OCH3", "F"):
        variant = fz.build_variant(cr_parent, fz.GROUPS[name], orbit, orbit.patterns[1][0][0])
        assert variant.status in ("ok", "tight")
        n_parent = len(cr_parent.structure)
        n_new = len(variant.structure) - (n_parent - len(variant.replaced))
        assert n_new == len(variant.replaced) * len(fz.GROUPS[name].build(0.0))
        bonds = CovalentBondDetector().detect(variant.structure)
        before = CovalentBondDetector().detect(cr_parent.structure)
        # Each replaced C-H bond becomes one C-X bond, plus the group's own.
        internal = len(fz.GROUPS[name].build(0.0)) - 1
        assert len(bonds) == len(before) + len(variant.replaced) * internal


def test_hydrogen_bonds_are_not_clashes():
    class Env:
        def vdw(self, element):
            return {"H": 1.2, "O": 1.52, "N": 1.55, "C": 1.7}[element]

    env = Env()
    # N-H...O at 2.0 A is 74% of the vdW sum, but a hydrogen bond.
    assert fz._judge(2.0, "H", "N", False, "O", None, False, env, "")[0] == 1.0
    # C-H...O at 2.0 A is a clash.
    assert fz._judge(2.0, "H", "C", False, "O", None, False, env, "")[0] < fz.CLASH_RATIO
    # N...O with the N carrying H: fine at 2.6, a clash at 2.2.
    assert fz._judge(2.6, "N", "C", True, "O", None, False, env, "")[0] == 1.0
    assert fz._judge(2.2, "N", "C", True, "O", None, False, env, "")[0] < fz.CLASH_RATIO


def test_sites_on_the_cell_edge_are_handled_like_any_other(cr_parent):
    """Found by the strip-and-rebuild check: a site stored at -1e-17 was
    wrapped to 1.0 for its bonds but to 0.0 for its periodic copies, so
    a group's own ring carbon looked like a neighbour 1.5 A away. The
    same structure with every site pushed a whole cell over, or to
    -1e-17, must give the same variant."""
    from pymatgen.core import Structure

    s = cr_parent.structure
    orbit = cr_parent.orbits[0]
    pattern = orbit.patterns[1][0][0]
    reference = fz.build_variant(cr_parent, fz.GROUPS["NH2"], orbit, pattern)
    frac = s.frac_coords.copy()
    frac[::2] += 1.0
    frac[frac == frac.min()] = -1e-17
    moved = Structure(s.lattice, [x.species for x in s], frac, labels=[x.label for x in s])
    parent = fz.analyze_parent(moved, max_per_linker=1)
    variant = fz.build_variant(parent, fz.GROUPS["NH2"], parent.orbits[0],
                               parent.orbits[0].patterns[1][0][0])
    assert variant.status == reference.status
    assert variant.closest[0][:2] == pytest.approx(reference.closest[0][:2], abs=1e-3)


# ----------------------------------------------------------------------
# Files, batch and command line
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def generated(tmp_path_factory, cr_parent):
    out = tmp_path_factory.mktemp("variants")
    source = out.parent / CR_BDC.name
    shutil.copy(CR_BDC, source)
    parent = fz.analyze_parent(_load(source), source=str(source), max_per_linker=2)
    record = fz.generate(parent, [fz.GROUPS["NH2"], fz.GROUPS["CN"]], out, per_linker=(1, 2))
    return out, record


def test_files_and_provenance(generated):
    out, record = generated
    variants = record["_variants"]
    written = [v for v in variants if v.file]
    rejected = [v for v in variants if not v.file]
    assert written and all(v.status == "clash" for v in rejected)
    # Round 71: -CN meta overlapped flat and was rejected; turned 25
    # degrees, its rings clear, and it is written with the turn recorded.
    meta_cn = next(v for v in variants if v.group == "CN" and v.descriptor == "meta")
    assert meta_cn.file and meta_cn.ring_turn and "turned" in meta_cn.ring_note
    for v in written:
        back = _load(out / v.file)
        assert back.composition.reduced_formula == v.structure.composition.reduced_formula
    assert (out / CR_BDC.name).is_file()  # the parent, beside its variants

    data = json.loads((out / "functionalization.json").read_text())
    assert data["parent"]["space_group"] == "Imma" and data["parent"]["sha256"]
    assert data["method"]["clash_ratio"] == fz.CLASH_RATIO
    assert data["method"]["groups"]["NH2"]["C-N"] == 1.380
    assert len(data["variants"]) == len(variants)
    entry = next(e for e in data["variants"] if e["file"])
    assert entry["closest_contacts"] and entry["replaced_hydrogens"][0].startswith("H")
    assert data["method"]["ring_turn"]["max_deg"] == fz.MAX_RING_TURN
    turned = next(e for e in data["variants"] if e["name"] == meta_cn.name)
    assert turned["ring_turn_deg"] == meta_cn.ring_turn and turned["ring_turns_deg"]
    import csv

    rows = list(csv.DictReader((out / "variants.csv").open()))
    assert next(r for r in rows if r["file"] == meta_cn.file)["fn_ring_turn_deg"] == \
        str(meta_cn.ring_turn)
    assert {"rotor_angles_deg", "bends_deg", "space_group", "formula"} <= set(entry)

    with open(out / "variants.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["fn_group"] == "(parent)" and rows[0]["file"] == CR_BDC.name
    assert {r["file"] for r in rows[1:]} == {v.file for v in written}


def test_variants_join_a_batch(generated, tmp_path):
    """the-void-analyze --reference-csv variants.csv carries the fn_
    columns into the summary, so a figure can colour by group."""
    from thevoid_core import cli
    from thevoid_core.services.batch_results import column_label, load_batch_results

    out, record = generated
    files = [str(out / CR_BDC.name)] + [str(out / v.file) for v in record["_variants"] if v.file][:2]
    code = cli.main(files + ["--out", str(tmp_path / "screen"), "-q", "--no-oms", "--no-topology",
                             "--no-surface", "--reference-csv", str(out / "variants.csv")])
    assert code == 0
    results = load_batch_results(tmp_path / "screen")
    groups = {row.get("file"): row.get("fn_group") for row in results.rows}
    assert groups[CR_BDC.name] == "(parent)"
    assert "NH2" in groups.values()
    assert column_label("fn_group") == "Group"


def test_command_line(tmp_path, capsys):
    from thevoid_core import functionalize_cli

    source = tmp_path / CR_BDC.name
    shutil.copy(CR_BDC, source)
    assert functionalize_cli.main([str(source), "--list", "--per-linker", "1,2"]) == 0
    listing = capsys.readouterr().out
    assert "Imma" in listing and "ortho" in listing and "C8H4O4" in listing

    out = tmp_path / "v"
    assert functionalize_cli.main([str(source), "--groups", "F,NH2", "--out", str(out), "-q"]) == 0
    assert sorted(p.name for p in out.glob("*_L1_m1.cif")) == [
        "2010[Cr][bpq]3[ASR]1_F_L1_m1.cif", "2010[Cr][bpq]3[ASR]1_NH2_L1_m1.cif"]

    with pytest.raises(SystemExit) as exit_:
        functionalize_cli.main([str(source), "--groups", "SH"])
    assert exit_.value.code == 2


def test_nothing_to_substitute_exits_1(tmp_path, capsys):
    from pymatgen.core import Lattice, Structure

    from thevoid_core import functionalize_cli

    salt = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    path = tmp_path / "salt.cif"
    salt.to(filename=str(path))
    assert functionalize_cli.main([str(path)]) == 1
    assert "nothing to functionalise" in capsys.readouterr().out


@pytest.mark.skipif(not RASPA_STRUCTURES, reason="needs RASPA2's structures (RASPA_STRUCTURES)")
def test_irmof1_gives_irmof3_and_irmof2():
    """IRMOF-1 + one -NH2 / -Br per BDC has IRMOF-3's / IRMOF-2's
    composition (RASPA2's models of Eddaoudi et al., Science 2002).
    -NH2 fits flat. -Br is tight flat; the model twists its rings 25
    degrees against the carboxylates, and since round 71 the build turns
    them too, by the smallest angle that clears it (15 degrees)."""
    root = Path(RASPA_STRUCTURES) / "mofs" / "cif"
    parent = fz.analyze_parent(_load(root / "IRMOF-1.cif"), max_per_linker=4)
    orbit = parent.orbits[0]
    assert len(orbit.linkers) == 24 and [len(orbit.patterns[m]) for m in (1, 2, 3, 4)] == [1, 3, 1, 1]
    mono = orbit.patterns[1][0][0]
    assert fz.build_variant(parent, fz.GROUPS["Br"], orbit, mono, ring_turn=False).status == \
        "tight"
    for name, model, turn in (("NH2", "IRMOF-3", 0.0), ("Br", "IRMOF-2", 15.0)):
        variant = fz.build_variant(parent, fz.GROUPS[name], orbit, mono)
        assert variant.status == "ok" and abs(variant.ring_turn) == turn
        if name == "NH2":
            assert set(variant.rotors) == {0.0}   # planar, with room
        assert variant.structure.composition.reduced_formula == \
            _load(root / f"{model}.cif").composition.reduced_formula


# ----------------------------------------------------------------------
# The dialog
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_dialog_reads_plans_generates_and_opens(qt_app, tmp_path):
    from PySide6.QtCore import QThreadPool

    from thevoid_core.ui.dialogs.functionalize_dialog import FunctionalizeDialog

    source = tmp_path / CR_BDC.name
    shutil.copy(CR_BDC, source)
    opened = []
    dialog = FunctionalizeDialog(_load(source), source=str(source), open_file=opened.append)
    QThreadPool.globalInstance().waitForDone(60000)
    qt_app.processEvents()
    assert dialog.parent_analysis is not None
    assert "Imma" in dialog.summary_label.text()
    assert "1 variant(s)" in dialog.plan_label.text()      # NH2, one per linker
    dialog.count_boxes[1].setChecked(True)                 # and two per linker
    assert "4 variant(s)" in dialog.plan_label.text()
    assert dialog.count_boxes[3].isEnabled() and dialog.generate_button.isEnabled()

    dialog.folder_edit.setText(str(tmp_path / "out"))
    assert dialog.start(synchronous=True)
    qt_app.processEvents()
    assert dialog.table.rowCount() == 4
    assert "the-void-analyze" in dialog.status_label.text()
    path = dialog.open_selected(row=0)
    assert opened == [path] and Path(path).is_file()
    dialog.close()


def test_the_main_window_has_the_menu(qt_app, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    from thevoid_core.ui.windows.main_window import MainWindow

    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args, **kwargs: shown.append(args))
    window = MainWindow()
    assert window.functionalize_action.text() == "Functionalise Linkers..."
    assert window.functionalize_linkers() is None  # nothing loaded: a message, no dialog
    assert shown
    window.close()
