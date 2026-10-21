from thevoid_core.data.element_colors import ElementColorProvider


def test_known_element_is_scaled_to_unit_range():
    provider = ElementColorProvider()

    assert provider.get_color("C") == (144 / 255, 144 / 255, 144 / 255)


def test_actinides_are_covered():
    provider = ElementColorProvider()

    for symbol in ("U", "Th", "Pu", "Am", "Lr"):
        assert provider.has_color(symbol), symbol
        assert provider.get_color(symbol) != provider.UNKNOWN_COLOR


def test_oxidation_state_is_stripped():
    provider = ElementColorProvider()

    assert provider.get_color("Fe3+") == provider.get_color("Fe")


def test_unknown_symbol_falls_back():
    provider = ElementColorProvider()

    assert provider.get_color("Qq") == provider.UNKNOWN_COLOR


# --- per-instance color overrides ---


def test_set_color_overrides_the_default():
    provider = ElementColorProvider()

    provider.set_color("C", (1.0, 0.0, 0.0))

    assert provider.get_color("C") == (1.0, 0.0, 0.0)


def test_set_color_does_not_affect_other_instances():
    a = ElementColorProvider()
    b = ElementColorProvider()

    a.set_color("C", (1.0, 0.0, 0.0))

    assert b.get_color("C") == (144 / 255, 144 / 255, 144 / 255)


def test_set_color_does_not_affect_other_elements():
    provider = ElementColorProvider()

    provider.set_color("C", (1.0, 0.0, 0.0))

    assert provider.get_color("O") == (255 / 255, 13 / 255, 13 / 255)


def test_reset_color_reverts_to_the_default():
    provider = ElementColorProvider()
    provider.set_color("C", (1.0, 0.0, 0.0))

    provider.reset_color("C")

    assert provider.get_color("C") == (144 / 255, 144 / 255, 144 / 255)


def test_reset_color_on_a_non_overridden_element_does_not_raise():
    provider = ElementColorProvider()

    provider.reset_color("C")  # should not raise


def test_reset_all_colors_clears_every_override():
    provider = ElementColorProvider()
    provider.set_color("C", (1.0, 0.0, 0.0))
    provider.set_color("O", (0.0, 1.0, 0.0))

    provider.reset_all_colors()

    assert provider.get_color("C") == (144 / 255, 144 / 255, 144 / 255)
    assert provider.get_color("O") == (255 / 255, 13 / 255, 13 / 255)


def test_is_overridden_reflects_override_state():
    provider = ElementColorProvider()

    assert not provider.is_overridden("C")

    provider.set_color("C", (1.0, 0.0, 0.0))
    assert provider.is_overridden("C")

    provider.reset_color("C")
    assert not provider.is_overridden("C")


def test_set_color_respects_oxidation_state_normalisation():
    """Setting a colour for 'Fe3+' should apply to plain 'Fe' too,
    consistent with get_color's own oxidation-state stripping.
    """
    provider = ElementColorProvider()

    provider.set_color("Fe3+", (0.1, 0.2, 0.3))

    assert provider.get_color("Fe") == (0.1, 0.2, 0.3)
