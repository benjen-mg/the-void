"""ElementRadiusProvider: covalent radii table coverage."""

from thevoid_core.data.element_radii import ElementRadiusProvider


def test_common_elements_use_the_table_not_the_fallback():
    provider = ElementRadiusProvider()

    assert provider.get_covalent_radius("C") == 0.76
    assert provider.get_covalent_radius("O") == 0.66
    assert provider.get_covalent_radius("H") == 0.31


def test_alkali_metal_no_longer_dwarfs_carbon():
    """Regression: the old atomic_radius/average_ionic_radius fallback
    gave Na a radius roughly three times carbon's.
    """
    provider = ElementRadiusProvider()

    sodium = provider.get_covalent_radius("Na")
    carbon = provider.get_covalent_radius("C")

    assert sodium / carbon < 2.5


def test_late_actinides_are_covered_directly_not_via_pymatgen():
    """Bk through Lr previously fell through to the pymatgen fallback
    this table exists to avoid. They should now come from the table.
    """
    provider = ElementRadiusProvider()

    for symbol in ("Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"):
        assert symbol in provider.COVALENT_RADII, symbol


def test_late_actinide_radii_continue_the_actinide_contraction_trend():
    """Each successive element should be no larger than the last -
    interpolated values must stay monotonic with the established trend
    they extend (Ac through Cm already decreases with atomic number).
    """
    provider = ElementRadiusProvider()

    sequence = ["Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
    radii = [provider.get_covalent_radius(symbol) for symbol in sequence]

    assert radii == sorted(radii, reverse=True)


def test_deuterium_and_tritium_are_hydrogen():
    provider = ElementRadiusProvider()

    hydrogen = provider.get_covalent_radius("H")

    assert provider.get_covalent_radius("D") == hydrogen
    assert provider.get_covalent_radius("T") == hydrogen


def test_oxidation_state_suffix_is_ignored():
    provider = ElementRadiusProvider()

    assert provider.get_covalent_radius("Fe3+") == provider.get_covalent_radius(
        "Fe"
    )


def test_unknown_symbol_falls_back_without_raising():
    provider = ElementRadiusProvider()

    # Not a real element; must not raise, and must return something
    # usable rather than zero or negative.
    assert provider.get_covalent_radius("Zz") > 0
