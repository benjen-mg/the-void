"""SHELX .res / .ins files: reader.py reads the model."""

from thevoid_core.services.shelx.reader import (  # noqa: F401
    ShelxAtom,
    ShelxError,
    ShelxModel,
    parse_shelx,
    read_shelx,
    shelx_to_cif,
)

#: File suffixes read as SHELX instruction files.
SHELX_SUFFIXES = (".res", ".ins")
