"""Round 55: the MOF Analysis panel's linkers and nodes, one unit cell's
worth, each exactly once (mof_analysis.partition_structure).

Before, the panel partitioned build_periodic_graph_for_mof_analysis's
expanded patch, which completes every linker but never expands a metal,
so clusters at the patch edge came out in pieces: IRMOF-1 listed its 8
Zn4O nodes next to 48 lone "Zn" (24 "Zn2" once paddlewheel grouping
existed) - bits of the Zn4O clusters of the next cells. Every check below has its answer fixed by the
formula and Z of the structure, or by how the structure was built.
"""

import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure  # noqa: E402

from thevoid_core.data.metal_classification import is_metal  # noqa: E402
from thevoid_core.services.bonds.bond_model import Bond  # noqa: E402
from thevoid_core.services.bonds.covalent_bond_detector import CovalentBondDetector  # noqa: E402
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader  # noqa: E402
from thevoid_core.services.mof_analysis import partition_structure  # noqa: E402


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return PymatgenCIFLoader().load(str(path)).structure


def _partition(structure, bonds=None, **kwargs):
    if bonds is None:
        bonds = CovalentBondDetector().detect(structure)
    return partition_structure(structure, bonds, is_metal, **kwargs)


def _formulas(items):
    return Counter(item.formula for item in items)


def _extent(structure, site_images, indices):
    """Largest distance between two atoms of one component, as placed."""
    lattice = structure.lattice.matrix
    points = np.array([
        (structure[site].frac_coords + np.array(image)) @ lattice
        for site, image in (site_images[i] for i in indices)
    ])
    return float(np.max(np.linalg.norm(points[:, None] - points[None], axis=2)))


def _bonds(pairs):
    return [Bond(i, j, tuple(image), 1.5) for i, j, image in pairs]


# ----------------------------------------------------------------------
# Built structures
# ----------------------------------------------------------------------


def _scatter(symbols, a=10.0):
    rng = np.random.default_rng(0)
    return Structure(Lattice.cubic(a), symbols, rng.random((len(symbols), 3)))


def test_a_paddlewheel_is_one_cu2_node():
    """Cu2(O2C)4 square grid, bonds given; the two Cu are 2.6 A apart
    and held together only through the carboxylates. The carboxylates
    stay with the linkers (the panel's convention); the node is Cu2."""
    pairs = []
    carbons = []
    index = 2
    for _ in range(4):
        c, oa, ob = index, index + 1, index + 2
        index += 3
        carbons.append(c)
        pairs += [(c, oa, (0, 0, 0)), (c, ob, (0, 0, 0)),
                  (0, oa, (0, 0, 0)), (1, ob, (0, 0, 0))]
    lx, ly = index, index + 1
    pairs += [(carbons[0], lx, (0, 0, 0)), (lx, carbons[1], (1, 0, 0)),
              (carbons[2], ly, (0, 0, 0)), (ly, carbons[3], (0, 1, 0))]
    structure = _scatter(["Cu", "Cu"] + ["C", "O", "O"] * 4 + ["C", "C"])
    structure.replace(0, "Cu", [0.5, 0.5, 0.5])
    structure.replace(1, "Cu", [0.5, 0.5, 0.76])

    partition, _ = _partition(structure, _bonds(pairs))
    assert _formulas(partition.metal_nodes) == {"Cu2": 1}
    assert _formulas(partition.linkers) == {"C3O4": 2}


def test_a_rod_is_one_node_flagged_as_a_rod():
    """Zn-O-Zn-O... along a (MOF-74, MIL-53), linked along b and c:
    one node per cell, dimensionality 1."""
    pairs = [(0, 1, (0, 0, 0)), (1, 0, (1, 0, 0)),
             (0, 2, (0, 0, 0)), (2, 3, (0, 0, 0)), (3, 0, (0, 1, 0)),
             (0, 4, (0, 0, 0)), (4, 5, (0, 0, 0)), (5, 0, (0, 0, 1))]
    structure = _scatter(["Zn", "O", "C", "C", "C", "C"])
    partition, _ = _partition(structure, _bonds(pairs))

    (node,) = partition.metal_nodes
    assert node.formula == "OZn" and node.dimensionality == 1
    assert all(linker.dimensionality == 0 for linker in partition.linkers)


def test_a_terminal_ligand_belongs_to_its_node():
    """A Cl on one Zn bridges nothing: it is part of the node (ClZn),
    not a one-atom "linker"."""
    pairs = [(0, 1, (0, 0, 0)), (1, 2, (0, 0, 0)), (2, 0, (1, 0, 0)),
             (0, 3, (0, 0, 0))]
    structure = _scatter(["Zn", "C", "C", "Cl"])
    partition, _ = _partition(structure, _bonds(pairs))
    assert _formulas(partition.metal_nodes) == {"ClZn": 1}
    assert _formulas(partition.linkers) == {"C2": 1}


# ----------------------------------------------------------------------
# Real frameworks (RASPA2's structure files, when available)
# ----------------------------------------------------------------------


def _root():
    for candidate in (os.environ.get("RASPA_STRUCTURES"),
                      "/tmp/raspa/RASPA2-2.0.4/structures"):
        if candidate and Path(candidate, "mofs", "cif").is_dir():
            return Path(candidate, "mofs", "cif")
    return None


ROOT = _root()
real = pytest.mark.skipif(ROOT is None, reason="RASPA2 structures not available")

#: file -> (nodes, linkers, node dimensionality), per conventional cell.
EXPECTED = {
    # Zn4O(BDC)3, Z = 8
    "IRMOF-1": ({"OZn4": 8}, {"C8H4O4": 24}, 0),
    # Cu3(BTC)2, Z = 16: 24 paddlewheels, 32 BTC
    "Cu-BTC": ({"Cu2": 24}, {"C9H3O6": 32}, 0),
    # Cr3O(F/OH)(H2O)2(BDC)3, terminal O folded into the trimer
    "MIL-88B-open": ({"Cr3O4": 2}, {"C8H4O4": 6}, 0),
    # Cr(OH)(BDC): infinite trans-OH-bridged chains
    "MIL-53(Cr)lt": ({"Cr2H2O2": 2}, {"C8H4O4": 4}, 1),
    # Zn2(BDC)2(dabco): paddlewheels pillared by dabco
    "ZnBDCdabco": ({"Zn2": 1}, {"C8H4O4": 2, "C6H12N2": 1}, 0),
}


@real
@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_real_frameworks_come_out_as_their_formula_units(name):
    structure = _load(ROOT / f"{name}.cif")
    partition, site_images = _partition(structure)
    nodes, linkers, dimensionality = EXPECTED[name]

    assert _formulas(partition.metal_nodes) == nodes
    assert _formulas(partition.linkers) == linkers
    assert {n.dimensionality for n in partition.metal_nodes} == {dimensionality}
    assert len(site_images) == len(structure)  # every site, once
    assert len({site for site, _ in site_images}) == len(structure)
    if dimensionality == 0:
        for node in partition.metal_nodes:
            assert _extent(structure, site_images, node.atom_indices) < 8.0


# ----------------------------------------------------------------------
# The panel
# ----------------------------------------------------------------------


def test_the_panel_says_which_nodes_are_rods_and_what_was_left_out():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from thevoid_core.services.mof_analysis import FrameworkPartition, MetalNode
    from thevoid_core.ui.widgets.mof_analysis_panel import MOFAnalysisPanel

    QApplication.instance() or QApplication([])
    panel = MOFAnalysisPanel()
    partition = FrameworkPartition(
        linkers=[],
        metal_nodes=[MetalNode(1, (0, 1), ("Mg", "Mg", "Mg"), "Mg3O6", 1),
                     MetalNode(2, (2,), ("Zn",), "Zn", 0)],
        excluded_disorder_sites=3,
    )
    panel.set_partition(partition)

    text = panel.summary_label.text()
    assert "2 metal node(s) per unit cell" in text
    assert "1 node(s) are infinite (rod (1D))" in text
    assert "3 minor-disorder site(s) left out" in text
    assert panel.node_table.horizontalHeaderItem(3).text() == "Metal Atoms"
    assert panel.node_table.item(0, 4).text() == "rod (1D)"
    assert panel.node_table.item(1, 4).text() == "cluster"
