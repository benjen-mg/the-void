"""Turning what a caller has - a file, a pymatgen Structure, an ASE
Atoms - into the pymatgen Structure every analysis works on."""

from __future__ import annotations

from pathlib import Path


def is_ase_atoms(obj) -> bool:
    return type(obj).__name__ == "Atoms" and type(obj).__module__.startswith("ase")


def to_structure(obj):
    """(pymatgen Structure, source path or None, space group or None).

    A path is read exactly as File > Open and the-void-analyze read it
    (PymatgenCIFLoader), so the same file gives the same numbers in all
    three. A pymatgen Structure is copied, so later changes to the
    caller's object cannot change a Framework's results. ASE Atoms are
    converted with pymatgen's own adaptor (ASE is imported only then)."""
    from pymatgen.core import Structure

    if isinstance(obj, (str, Path)):
        path = Path(obj)
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        import warnings

        from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            info = PymatgenCIFLoader().load(str(path))
        return info.structure, str(path), info.space_group
    if isinstance(obj, Structure):
        return obj.copy(), None, None
    if is_ase_atoms(obj):
        if not all(obj.pbc):
            raise ValueError("ASE Atoms must be periodic in all three directions "
                             "(atoms.pbc = True) to be a framework")
        from pymatgen.io.ase import AseAtomsAdaptor

        return AseAtomsAdaptor.get_structure(obj), None, None
    raise TypeError(
        f"Cannot make a framework from {type(obj).__name__}: give a file path, "
        "a pymatgen Structure or an ASE Atoms")
