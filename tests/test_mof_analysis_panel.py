"""MOFAnalysisPanel: linker/node tables and click-to-highlight, through
the real Qt widget.
"""

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.mof_analysis import (  # noqa: E402
    FrameworkPartition,
    Linker,
    MetalNode,
)
from thevoid_core.ui.widgets.mof_analysis_panel import (  # noqa: E402
    MOFAnalysisPanel,
)


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def sample_partition():
    return FrameworkPartition(
        linkers=[
            Linker(linker_id=1, atom_indices=(3, 4, 5), formula="C6H4O2"),
            Linker(linker_id=2, atom_indices=(6, 7, 8), formula="C6H4O2"),
        ],
        metal_nodes=[
            MetalNode(
                node_id=1,
                atom_indices=(0, 1, 2),
                metal_symbols=("Zn", "Zn"),
                formula="OZn2",
            ),
        ],
    )


def test_starts_with_no_data(qt_app):
    panel = MOFAnalysisPanel()

    assert panel.linker_table.rowCount() == 0
    assert panel.node_table.rowCount() == 0
    assert "No MOF analysis" in panel.summary_label.text()


def test_set_partition_populates_the_summary(qt_app, sample_partition):
    panel = MOFAnalysisPanel()

    panel.set_partition(sample_partition)

    assert "2 linker" in panel.summary_label.text()
    assert "1 metal node" in panel.summary_label.text()


def test_set_partition_populates_the_linker_table(qt_app, sample_partition):
    panel = MOFAnalysisPanel()

    panel.set_partition(sample_partition)

    assert panel.linker_table.rowCount() == 2
    assert panel.linker_table.item(0, 0).text() == "1"
    assert panel.linker_table.item(0, 1).text() == "C6H4O2"
    assert panel.linker_table.item(0, 2).text() == "3"


def test_set_partition_populates_the_node_table(qt_app, sample_partition):
    panel = MOFAnalysisPanel()

    panel.set_partition(sample_partition)

    assert panel.node_table.rowCount() == 1
    assert panel.node_table.item(0, 0).text() == "1"
    assert panel.node_table.item(0, 1).text() == "Zn"
    assert panel.node_table.item(0, 2).text() == "OZn2"
    assert panel.node_table.item(0, 3).text() == "2"


def test_clear_empties_both_tables(qt_app, sample_partition):
    panel = MOFAnalysisPanel()
    panel.set_partition(sample_partition)

    panel.clear()

    assert panel.linker_table.rowCount() == 0
    assert panel.node_table.rowCount() == 0
    assert "No MOF analysis" in panel.summary_label.text()


def test_selecting_a_linker_row_emits_its_atom_indices_and_colour(
    qt_app, sample_partition
):
    panel = MOFAnalysisPanel()
    panel.set_partition(sample_partition)

    received = []
    panel.highlight_requested.connect(
        lambda indices, color: received.append((indices, color))
    )

    panel.linker_table.selectRow(1)

    assert received == [([6, 7, 8], MOFAnalysisPanel.LINKER_COLOR)]


def test_selecting_a_node_row_emits_its_atom_indices_and_colour(
    qt_app, sample_partition
):
    panel = MOFAnalysisPanel()
    panel.set_partition(sample_partition)

    received = []
    panel.highlight_requested.connect(
        lambda indices, color: received.append((indices, color))
    )

    panel.node_table.selectRow(0)

    assert received == [([0, 1, 2], MOFAnalysisPanel.NODE_COLOR)]


def test_linker_and_node_highlight_colours_are_different(qt_app):
    assert MOFAnalysisPanel.LINKER_COLOR != MOFAnalysisPanel.NODE_COLOR


def test_node_with_multiple_of_the_same_metal_lists_the_type_once(
    qt_app,
):
    """metal_symbols carries one entry per metal atom (Zn, Zn for a
    two-zinc node) - the displayed "Metal Type(s)" column should show
    the distinct element, not repeat it.
    """
    panel = MOFAnalysisPanel()

    partition = FrameworkPartition(
        linkers=[],
        metal_nodes=[
            MetalNode(
                node_id=1,
                atom_indices=(0, 1, 2, 3),
                metal_symbols=("Zn", "Zn", "Zn", "Zn"),
                formula="OZn4",
            ),
        ],
    )
    panel.set_partition(partition)

    assert panel.node_table.item(0, 1).text() == "Zn"
