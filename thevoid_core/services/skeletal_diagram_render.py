from __future__ import annotations

#: Fragments per row in the composed grid image.
FRAGMENTS_PER_ROW = 4

#: Pixel size of each individual fragment's own sub-image within the
#: grid.
FRAGMENT_IMAGE_SIZE = (300, 300)


def render_skeletal_diagram(result):
    """Compose every fragment in a SkeletalDiagramResult into one grid
    image, via RDKit's own Draw.MolsToGridImage - deliberately not a
    custom renderer, since RDKit's drawing already implements the
    standard skeletal-formula convention this feature is specifically
    meant to follow: carbon and hydrogen stay implicit (no vertex
    label), every other element - heteroatoms, and metal centres - is
    labelled with its symbol, exactly the convention real published
    structure diagrams use.

    The one caption this adds - "x N" beneath a fragment that occurs
    more than once in the unit cell - is not an extra annotation
    layered on top of an otherwise plain diagram; it is the only way
    the deduplication (see compute_skeletal_diagram's own docstring for
    why symmetry-equivalent copies are collapsed to one picture) stays
    honest about how many real copies one picture actually stands for,
    rather than silently implying there is only one. A fragment that
    occurs exactly once gets no caption at all.

    Returns a PIL Image. Callers decide whether to embed it in a Qt
    widget, save it to a file, or both - the same separation of
    "compute the diagram" from "do something with it" this
    application already uses for every other rendered result.
    """
    from rdkit.Chem import Draw

    mols = [fragment.mol for fragment in result.fragments]

    if not mols:
        return _blank_image()

    legends = [
        f"x{fragment.occurrence_count}"
        if fragment.occurrence_count > 1
        else ""
        for fragment in result.fragments
    ]

    return Draw.MolsToGridImage(
        mols,
        molsPerRow=FRAGMENTS_PER_ROW,
        subImgSize=FRAGMENT_IMAGE_SIZE,
        legends=legends,
    )


def _blank_image():
    from PIL import Image

    return Image.new("RGB", FRAGMENT_IMAGE_SIZE, color="white")
