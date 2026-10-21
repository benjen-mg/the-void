"""StructureInfoPanel: the crystallographic summary, plus the two
things it is there to warn about.

The warnings are the substance. A disordered file and a rejected short
contact both look like ordinary structures in the 3D view; this panel
is the only place either is stated, so the highlighting and the tooltip
detail are functional rather than decorative.
"""

import os
from dataclasses import dataclass

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from thevoid_core.services.bonds.bond_model import (  # noqa: E402
    BondingDiagnostics,
    ShortContact,
)
from thevoid_core.ui.widgets.structure_info_panel import (  # noqa: E402
    WARNING_STYLE,
    StructureInfoPanel,
)


@dataclass
class FakeInfo:
    formula: str = "Zn(C2H2N3)2"
    space_group: str = "P2_1/c"
    site_count: int = 56
    partial_occupancy_sites: int = 0
    a: float = 8.9142
    b: float = 9.6938
    c: float = 9.4841
    alpha: float = 90.0
    beta: float = 115.87
    gamma: float = 90.0


@dataclass
class FakeFragment:
    formula: str
    occurrences: int


@dataclass
class FakeModel:
    fragments: list
    fragment_count: int


STATS = {"count": 70, "min": 0.964, "max": 2.187, "avg": 1.402}


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qt_app):
    widget = StructureInfoPanel()
    yield widget
    widget.deleteLater()


def test_a_fresh_panel_shows_placeholders(panel):
    assert panel.formula_label.text() == "-"
    assert panel.bondcount_label.text() == "-"


def test_the_crystallographic_summary_is_filled_in(panel):
    panel.set_structure(FakeInfo(), STATS)

    assert panel.formula_label.text() == "Zn(C2H2N3)2"
    assert panel.spacegroup_label.text() == "P2_1/c"
    assert panel.sitecount_label.text() == "56"


def test_cell_lengths_and_angles_are_shown_at_useful_precision(panel):
    panel.set_structure(FakeInfo(), STATS)

    assert panel.a_label.text() == "8.9142"
    assert panel.beta_label.text() == "115.87"


def test_bond_statistics_are_shown(panel):
    panel.set_structure(FakeInfo(), STATS)

    assert panel.bondcount_label.text() == "70"
    assert panel.bond_min_label.text() == "0.964"
    assert panel.bond_avg_label.text() == "1.402"


def test_an_ordered_structure_reports_no_disorder(panel):
    panel.set_structure(FakeInfo(), STATS)

    assert panel.disorder_label.text() == "none"
    assert panel.disorder_label.styleSheet() == ""


def test_disorder_is_counted_and_highlighted(panel):
    panel.set_structure(FakeInfo(partial_occupancy_sites=12), STATS)

    assert "12" in panel.disorder_label.text()
    assert panel.disorder_label.styleSheet() == WARNING_STYLE


def test_disorder_is_picked_up_from_diagnostics_when_info_is_silent(panel):
    """Not every loader path fills partial_occupancy_sites on the info
    object, so the panel falls back to the bond detector's count rather
    than reporting a disordered file as clean.
    """
    diagnostics = BondingDiagnostics(partial_occupancy_sites=4)

    panel.set_structure(FakeInfo(), STATS, diagnostics=diagnostics)

    assert "4" in panel.disorder_label.text()
    assert panel.disorder_label.styleSheet() == WARNING_STYLE


def test_no_short_contacts_reports_none(panel):
    panel.set_structure(FakeInfo(), STATS, diagnostics=BondingDiagnostics())

    assert panel.short_contact_label.text() == "none"
    assert panel.short_contact_label.toolTip() == ""


def test_short_contacts_report_the_count_and_the_shortest(panel):
    diagnostics = BondingDiagnostics(
        short_contacts=[
            ShortContact(
                atom1=52,
                atom2=54,
                symbol1="O",
                symbol2="O",
                distance=0.794,
                both_partial=True,
            ),
            ShortContact(
                atom1=1,
                atom2=2,
                symbol1="C",
                symbol2="C",
                distance=1.102,
                both_partial=False,
            ),
        ]
    )

    panel.set_structure(FakeInfo(), STATS, diagnostics=diagnostics)

    assert "2 rejected" in panel.short_contact_label.text()
    assert "0.794" in panel.short_contact_label.text()
    assert panel.short_contact_label.styleSheet() == WARNING_STYLE


def test_the_short_contact_tooltip_lists_the_pairs_shortest_first(panel):
    diagnostics = BondingDiagnostics(
        short_contacts=[
            ShortContact(
                atom1=1,
                atom2=2,
                symbol1="C",
                symbol2="C",
                distance=1.102,
                both_partial=False,
            ),
            ShortContact(
                atom1=52,
                atom2=54,
                symbol1="O",
                symbol2="O",
                distance=0.794,
                both_partial=True,
            ),
        ]
    )

    panel.set_structure(FakeInfo(), STATS, diagnostics=diagnostics)

    tooltip = panel.short_contact_label.toolTip()

    assert tooltip.index("O53") < tooltip.index("C2")


def test_fragments_are_listed_with_their_counts(panel):
    model = FakeModel(
        fragments=[
            FakeFragment("C2H2N3", 8),
            FakeFragment("Zn", 4),
        ],
        fragment_count=12,
    )

    panel.set_structure(FakeInfo(), STATS, model=model)

    assert panel.fragment_label.text() == "12"
    assert "8 \u00d7 C2H2N3" in panel.fragment_detail_label.text()


def test_the_fragment_list_is_capped(panel):
    model = FakeModel(
        fragments=[FakeFragment(f"C{i}", 1) for i in range(20)],
        fragment_count=20,
    )

    panel.set_structure(FakeInfo(), STATS, model=model)

    assert panel.fragment_detail_label.text().count("\u00d7") == 8


def test_no_model_leaves_the_fragment_rows_blank(panel):
    panel.set_structure(FakeInfo(), STATS)

    assert panel.fragment_label.text() == "-"


def test_reloading_a_clean_structure_clears_a_previous_warning(panel):
    """The panel is reused across loads, so a warning style left behind
    from the previous file would mark a clean structure as disordered.
    """
    panel.set_structure(FakeInfo(partial_occupancy_sites=12), STATS)
    panel.set_structure(FakeInfo(), STATS)

    assert panel.disorder_label.text() == "none"
    assert panel.disorder_label.styleSheet() == ""
