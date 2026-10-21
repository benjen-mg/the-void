"""Regression test for duplicated periodic-image bonds.

A neighbour search reports i -> j across image I and j -> i across
image -I. Both must collapse onto one key, or every bond crossing a cell
boundary is counted and drawn twice. This is shared by every bond
detector, currently just CovalentBondDetector.
"""

from thevoid_core.services.bonds.bond_model import canonical_key, statistics


def test_reverse_direction_collapses_to_one_key():
    assert canonical_key(0, 3, (1, 0, 0)) == canonical_key(3, 0, (-1, 0, 0))


def test_intracell_bond_is_stable():
    assert canonical_key(2, 5, (0, 0, 0)) == canonical_key(5, 2, (0, 0, 0))


def test_distinct_images_stay_distinct():
    assert canonical_key(0, 3, (1, 0, 0)) != canonical_key(0, 3, (0, 1, 0))


def test_self_bond_across_a_cell_face_is_kept():
    """A site bonded to its own translate is legitimate in a framework."""
    assert canonical_key(0, 0, (1, 0, 0)) == canonical_key(0, 0, (1, 0, 0))


def test_statistics_on_empty_input():
    assert statistics([])["count"] == 0
