"""Commonly cited approximate probe/guest radii (angstrom), used
throughout the computational
porous-materials literature for accessible-surface-area, accessible-
volume, and guest-accessibility calculations (the same convention
Zeo++-style tools use). These are not a single universally agreed
physical constant - different sources report kinetic diameters that
differ by a few tenths of an angstrom depending on whether the
underlying measurement is gas viscosity, molecular sieving, or a
computed collision cross-section - so treat these as reasonable,
documented defaults, not definitive values. Anyone needing a specific
literature value for a specific paper's convention should use a custom
probe radius/diameter instead.

Single canonical location for this table - surface_area.py and
guest_accessibility.py both need the same probe set (N2/CO2/H2/H2O,
plus CH4 for guest accessibility) and must not drift apart into two
slightly different tables for what is chemically the same gas.
"""

#: Where each value comes from (round 50 - the module previously said
#: "kinetic diameter / 2" for all of them, which is true only of N2 and
#: CH4):
#:   N2   1.82  - kinetic diameter 3.64 A / 2
#:   CO2  1.525 - Lennard-Jones sigma of the TraPPE O site, 3.05 A / 2
#:                (kinetic diameter 3.3 A would give 1.65)
#:   H2   1.48  - Lennard-Jones sigma 2.96 A / 2 (kinetic 2.89 A: 1.445)
#:   H2O  1.58  - SPC/TIP3P oxygen sigma 3.166 A / 2 (kinetic 2.65 A:
#:                1.325)
#:   CH4  1.90  - kinetic diameter 3.8 A / 2
#: Values unchanged so existing results stay comparable; use a custom
#: probe to follow a different convention.
PROBE_RADII: dict[str, float] = {
    "N2": 1.82,
    "CO2": 1.525,
    "H2": 1.48,
    "H2O": 1.58,
    "CH4": 1.90,
}
