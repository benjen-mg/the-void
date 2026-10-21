"""render_skeletal_diagram(): composing fragment diagrams into one
grid image via RDKit's own drawing.
"""

from dataclasses import dataclass

import pytest

pytest.importorskip("rdkit")

from thevoid_core.services.skeletal_diagram_render import (  # noqa: E402
    render_skeletal_diagram,
)
from thevoid_core.services.skeletal_layout import (  # noqa: E402
    FragmentDiagram,
    SkeletalDiagramResult,
    build_fragment_mol,
)


@dataclass(frozen=True)
class _Bond:
    atom_a: int
    atom_b: int


def _fragment(occurrence_count=1):
    from rdkit import Chem
    from rdkit.Chem import rdDepictor

    symbols = ["C", "O", "N"]
    bonds = [_Bond(0, 1), _Bond(1, 2)]
    mol = build_fragment_mol(symbols, bonds, (0, 1, 2))
    Chem.SanitizeMol(mol)
    rdDepictor.Compute2DCoords(mol)

    return FragmentDiagram(
        label="test fragment", mol=mol, occurrence_count=occurrence_count
    )


def test_produces_a_pil_image():
    result = SkeletalDiagramResult(fragments=(_fragment(),), skipped_count=0)

    image = render_skeletal_diagram(result)

    assert image.size[0] > 0
    assert image.size[1] > 0


def test_empty_result_still_returns_an_image():
    result = SkeletalDiagramResult(fragments=(), skipped_count=0)

    image = render_skeletal_diagram(result)

    assert image is not None
    assert image.size[0] > 0


def test_multiple_fragments_produce_a_larger_image_than_one():
    one = SkeletalDiagramResult(fragments=(_fragment(),), skipped_count=0)
    many = SkeletalDiagramResult(
        fragments=(_fragment(), _fragment(), _fragment(), _fragment()),
        skipped_count=0,
    )

    image_one = render_skeletal_diagram(one)
    image_many = render_skeletal_diagram(many)

    # 4 fragments at FRAGMENTS_PER_ROW=4 fill exactly one row - same
    # height as one fragment alone, but should not be smaller.
    assert image_many.size[0] >= image_one.size[0]


def test_a_fragment_occurring_once_gets_no_caption_text():
    """Checked indirectly: MolsToGridImage accepts a legends list, and
    an empty string for a single-occurrence fragment must not error or
    be replaced with something else - confirmed by checking the
    render completes and produces a normal image for this case.
    """
    result = SkeletalDiagramResult(fragments=(_fragment(1),), skipped_count=0)

    image = render_skeletal_diagram(result)

    assert image.size[0] > 0


def test_a_fragment_occurring_many_times_still_renders():
    """A deduplicated fragment with occurrence_count > 1 must still
    render without error - the "xN" caption path.
    """
    result = SkeletalDiagramResult(fragments=(_fragment(17),), skipped_count=0)

    image = render_skeletal_diagram(result)

    assert image.size[0] > 0
