"""Round 53: periodic nets and their exact comparison
(services/topology/net.py, services/topology/library.py).

A net's name is only as trustworthy as the test that two nets are the
same, so that test is checked from both sides: every reference net is
recognised again after being disguised (a supercell, a different basis,
shuffled labels), and nets that agree on every cheap invariant but are
different nets are told apart. Coordination sequences are checked
against the published values.
"""

import random

import pytest

from thevoid_core.services.topology.library import (
    bundled_library,
    read_systre_archive,
)
from thevoid_core.services.topology.net import (
    PeriodicNet,
    UnstableNetError,
    barycentric_placement,
    connected_components,
    coordination_sequences,
    integer_row_basis,
    isomorphic,
    minimal_net,
    parse_systre_key,
    prepare,
)


def scramble(net, seed=0, axis=0):
    """The same net, disguised: a 2x supercell along `axis`, then a
    random unimodular change of basis, then shuffled vertex labels."""
    rnd = random.Random(seed)
    d, n = net.dimension, net.n_vertices
    edges = []
    for i, j, t in net.edges:
        for c in range(2):
            shifted = list(t)
            total = c + shifted[axis]
            shifted[axis] = total // 2
            edges.append((i + n * c, j + n * (total % 2), tuple(shifted)))
    basis = [[int(a == b) for b in range(d)] for a in range(d)]
    for _ in range(3):
        a, b = rnd.sample(range(d), 2)
        k = rnd.choice([-1, 1])
        basis[a] = [x + k * y for x, y in zip(basis[a], basis[b])]
    edges = [
        (i, j, tuple(sum(t[k] * basis[k][m] for k in range(d)) for m in range(d)))
        for i, j, t in edges
    ]
    perm = list(range(2 * n))
    rnd.shuffle(perm)
    return PeriodicNet.build(2 * n, [(perm[i], perm[j], t) for i, j, t in edges], d)


@pytest.fixture(scope="module")
def library():
    return bundled_library()


def _key(library, name):
    return next(net.key for net in library.nets if net.name == name)


# ----------------------------------------------------------------------
# Invariants against published values
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name, sequence", [
    ("pcu", (6, 18, 38, 66, 102, 146)),
    ("dia", (4, 12, 24, 42, 64, 92)),
    ("fcu", (12, 42, 92, 162, 252, 362)),
    ("bcu", (8, 26, 56, 98, 152, 218)),
    ("sod", (4, 10, 20, 34, 52, 74)),
    ("srs", (3, 6, 12, 24, 35, 48)),
    ("nbo", (4, 12, 28, 50, 76, 110)),
    ("sql", (4, 8, 12, 16, 20, 24)),
    ("hcb", (3, 6, 9, 12, 15, 18)),
])
def test_coordination_sequences_match_published_values(library, name, sequence):
    net = parse_systre_key(_key(library, name))
    assert set(coordination_sequences(net, 6)) == {sequence}


# ----------------------------------------------------------------------
# The isomorphism test
# ----------------------------------------------------------------------


def test_every_bundled_net_is_recognised_in_disguise(library):
    """The property naming depends on: a supercell, a new basis and new
    labels change the quotient graph completely, but not the net."""
    for k, reference in enumerate(library.nets):
        net = parse_systre_key(reference.key)
        disguised = scramble(net, seed=k, axis=k % net.dimension)
        prepared = prepare(disguised)
        assert prepared.net.n_vertices == net.n_vertices, reference.name
        matches = library.identify(prepared)
        assert [m.name for m in matches] == [reference.name]


def test_nets_with_identical_invariants_are_still_told_apart(library):
    """bcu and hex are both one-vertex 8-connected nets with the same
    coordination sequence through ten shells - every cheap invariant
    agrees - yet they are different nets. (lta and rho agree through
    six shells.) This is why a name is only given after an exact
    isomorphism test."""
    hex_key = "3 1 1 -1 -1 0 1 1 -1 0 0 1 1 0 -1 0 1 1 0 0 -1"  # RCSR hex
    bcu = prepare(parse_systre_key(_key(library, "bcu")))
    hexagonal = prepare(parse_systre_key(hex_key))
    assert bcu.invariant == hexagonal.invariant
    assert not isomorphic(bcu, hexagonal)
    assert not isomorphic(hexagonal, bcu)

    lta = prepare(parse_systre_key(_key(library, "lta")))
    rho = prepare(parse_systre_key(_key(library, "rho")))
    assert lta.sequences[0][:6] == rho.sequences[0][:6]
    assert not isomorphic(lta, rho)


def test_the_bundled_library_has_no_duplicates_and_minimal_keys(library):
    prepared = {net.name: net.prepared() for net in library.nets}
    for net in library.nets:
        # Systre keys are already primitive; quick filtering relies on it.
        assert prepared[net.name].net.n_vertices == parse_systre_key(net.key).n_vertices
    names = list(prepared)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if prepared[a].invariant == prepared[b].invariant:
                assert not isomorphic(prepared[a], prepared[b]), (a, b)


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------


def test_supercells_reduce_to_the_primitive_cell():
    pcu = parse_systre_key("3 1 1 1 0 0 1 1 0 1 0 1 1 0 0 1")
    big = pcu
    for axis in range(3):
        big = scramble(big, seed=axis, axis=axis)
    assert big.n_vertices == 8
    reduced, _, index = minimal_net(big)
    assert reduced.n_vertices == 1 and index == 8


def test_barycentric_placement_is_exact():
    """dia's second vertex sits a quarter of the way along the body
    diagonal in its primitive (fcc) basis - exactly, as Fractions. Its
    four edges from vertex 1 reach cells 0, a, b and c, whose average is
    (1/4, 1/4, 1/4); so p1 + (1/4, 1/4, 1/4) = p0 = 0."""
    from fractions import Fraction

    dia = parse_systre_key("3 1 2 0 0 0 1 2 0 0 1 1 2 0 1 0 1 2 1 0 0")
    positions = barycentric_placement(dia)
    assert positions[0] == (0, 0, 0)
    assert positions[1] == (Fraction(-1, 4),) * 3
    assert all(isinstance(x, Fraction) for x in positions[1])


def test_an_unstable_net_is_refused_not_guessed():
    """Two vertices with exactly the same neighbours land on the same
    point; the method cannot tell them apart, so it must say so."""
    net = PeriodicNet.build(3, [
        (0, 0, (0, 1, 0)), (0, 0, (0, 0, 1)),
        (0, 1, (0, 0, 0)), (1, 0, (1, 0, 0)),
        (0, 2, (0, 0, 0)), (2, 0, (1, 0, 0)),
    ], 3)
    with pytest.raises(UnstableNetError) as info:
        prepare(net)
    assert info.value.collisions == [[1, 2]]


def test_layers_in_a_3d_cell_are_found_as_2d_nets(library):
    """sql layers stacked in a 3D cell, not bonded to each other: each
    layer is its own 2-periodic component, and is named sql."""
    net = PeriodicNet.build(2, [
        (0, 0, (1, 0, 0)), (0, 0, (0, 1, 0)),
        (1, 1, (1, 0, 0)), (1, 1, (0, 1, 0)),
    ], 3)
    components = connected_components(net)
    assert [c.dimension for c, _ in components] == [2, 2]
    assert [m.name for m in library.identify(prepare(components[0][0]))] == ["sql"]


def test_interpenetrated_copies_span_a_sublattice():
    """Two pcu nets offset by half a cell, written in the doubled cell's
    basis: each component's own lattice is a sublattice, and re-expressed
    in it, each is plain pcu with one vertex."""
    edges = []
    for v in (0, 1):
        edges += [(v, v, (1, 0, 0)), (v, v, (0, 1, 0)), (v, v, (0, 0, 1))]
    components = connected_components(PeriodicNet.build(2, edges, 3))
    assert len(components) == 2
    assert all(c.n_vertices == 1 and c.dimension == 3 for c, _ in components)


def test_integer_basis_keeps_the_lattice():
    assert integer_row_basis([(2, 0, 0), (0, 2, 0), (1, 1, 0)]) == [[1, 1, 0], [0, 2, 0]]
    assert integer_row_basis([(0, 0, 0)]) == []


def test_systre_archives_are_read(tmp_path):
    path = tmp_path / "x.arc"
    path.write_text(
        "key      3 1 1 -1 0 0 1 1 0 -1 0 1 1 0 0 -1\nversion  1.0\nid       pcu\nend\n\n"
        "key      2 1 1 -1 0 1 1 0 -1\nid       sql\nend\n"
    )
    nets = read_systre_archive(path)
    assert set(nets) == {"pcu", "sql"}
    assert parse_systre_key(nets["sql"]).dimension == 2
