"""is_metal(): delegates to pymatgen, but the special-attention list
from the request gets its own direct coverage rather than trusting
that delegation is correct by assumption.
"""

import pytest

pytest.importorskip("pymatgen")

from thevoid_core.data.metal_classification import (  # noqa: E402
    MOF_NODE_METALS,
    is_metal,
)


@pytest.mark.parametrize("symbol", sorted(MOF_NODE_METALS))
def test_special_attention_metals_are_classified_as_metal(symbol):
    assert is_metal(symbol) is True


@pytest.mark.parametrize("symbol", ["C", "H", "O", "N", "S", "P", "F", "Cl"])
def test_common_organic_linker_elements_are_not_metals(symbol):
    assert is_metal(symbol) is False


@pytest.mark.parametrize("symbol", ["Si", "B", "Ge", "As"])
def test_metalloids_are_not_classified_as_metals(symbol):
    """Metalloids sit on the boundary; pymatgen (and this function)
    treats them as non-metal, which matters for MOF partitioning since
    a metalloid in a linker (e.g. a boronic-acid-based linker) must not
    be cut away from its own linker.
    """
    assert is_metal(symbol) is False


def test_special_attention_set_matches_what_was_requested():
    assert MOF_NODE_METALS == {
        "Zn", "Cu", "Co", "Ni", "Fe", "Zr", "Ti", "Al",
    }


def test_unrecognised_symbol_is_treated_as_non_metal():
    assert is_metal("Qq") is False


def test_empty_symbol_is_treated_as_non_metal():
    assert is_metal("") is False
