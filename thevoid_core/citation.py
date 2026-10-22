"""How to cite the Void, and its licence (round 75).

One source for what Help > About, the README and CITATION.cff say, so the
three cannot drift apart (tests/test_publishing.py checks that they agree).
"""

from __future__ import annotations

from thevoid_core.version import __version__

TITLE = "the Void"
AUTHORS = (("Benyamin M.", "Garmejani", "Victoria University of Wellington"),)
YEAR = 2026

LICENCE_SPDX = "CC-BY-NC-ND-4.0"
LICENCE_NAME = ("Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 "
                "International (CC BY-NC-ND 4.0)")
LICENCE_URL = "https://creativecommons.org/licenses/by-nc-nd/4.0/"
LICENCE_SUMMARY = (
    "Free for research, teaching and other non-commercial use, and to "
    "share unchanged, with credit. Commercial use needs the author's "
    "permission. Modified versions may not be distributed."
)

REPOSITORY_URL = "https://github.com/benjen-mg/the-void"

#: Zenodo's DOI for all versions of the Void; it resolves to the latest.
DOI = "10.5281/zenodo.23578711"


def author_list() -> str:
    """'B. M. Garmejani', or 'A. One and B. Two', or 'A. One, B. Two and C. Three'."""
    names = [" ".join(f"{part[0]}." for part in given.split()) + f" {family}"
             for given, family, _ in AUTHORS]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def citation_text(version: str = __version__) -> str:
    """The reference to put in a paper's reference list."""
    text = f"{author_list()}, {TITLE}, version {version} ({YEAR}). Software"
    if DOI:
        text += f", doi:{DOI}"
    elif REPOSITORY_URL:
        text += f", {REPOSITORY_URL}"
    return text + "."


def bibtex(version: str = __version__) -> str:
    authors = " and ".join(f"{family}, {given}" for given, family, _ in AUTHORS)
    lines = [
        "@software{the_void,",
        f"  author  = {{{authors}}},",
        f"  title   = {{{TITLE}}},",
        f"  version = {{{version}}},",
        f"  year    = {{{YEAR}}},",
    ]
    if DOI:
        lines.append(f"  doi     = {{{DOI}}},")
    if REPOSITORY_URL:
        lines.append(f"  url     = {{{REPOSITORY_URL}}},")
    lines.append(f"  note    = {{Licensed {LICENCE_SPDX}}}")
    lines.append("}")
    return "\n".join(lines)
