"""Round 75: the repository can be published.

The Void goes public under CC BY-NC-ND 4.0 (research and other
non-commercial use only). What would go wrong unnoticed,
and so is tested here:

- a structure from the Cambridge Structural Database (which may not be
  redistributed) committed with the code, either as a file or pasted
  into another fixture;
- a test structure whose source and licence nobody wrote down;
- the version, licence or citation saying one thing in LICENSE,
  CITATION.cff, pyproject.toml, the README or Help > About and another
  elsewhere;
- bundled data that NOTICE.md does not account for.
"""

import gzip
import re
import subprocess
import tomllib
from pathlib import Path


from thevoid_core import citation
from thevoid_core.services.index.licence import has_csd_header
from thevoid_core.version import __version__

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "build", "dist"}

#: Files that hold structures or data, as opposed to code and prose
#: (licence.py's own patterns, and this file, mention the CSD by name).
DATA = {".cif", ".mcif", ".cssr", ".res", ".xyz", ".pdb", ".mol2", ".vasp",
        ".json", ".csv", ".txt", ".gz"}

#: CIF data items only the CCDC writes.
CCDC_ITEMS = re.compile(r"_database_code_depnum_ccdc|_database_code_CSD", re.IGNORECASE)


def _published_files():
    """What Git would publish: the tracked files in a checkout (not the
    output folders a test run or CI's smoke test writes beside them),
    otherwise every file outside caches and build folders."""
    if (ROOT / ".git").exists():
        listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True,
                                check=True).stdout.decode("utf-8").split("\0")
        yield from (ROOT / name for name in listed if name and (ROOT / name).is_file())
        return
    for path in ROOT.rglob("*"):
        if not path.is_file() or SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        yield path


def _text(path: Path) -> str:
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _from_the_csd(text: str) -> bool:
    return has_csd_header(text) or bool(CCDC_ITEMS.search(text))


def test_the_check_recognises_a_csd_file():
    """The guard below is only as good as this."""
    notice = ("# This file contains crystal structure data downloaded from the\n"
              "# Cambridge Structural Database (CSD) hosted by the Cambridge\n"
              "# Crystallographic Data Centre (CCDC).\n")
    assert _from_the_csd(notice + "data_x\n")
    assert _from_the_csd("data_x\n_database_code_depnum_ccdc_archive 'CCDC 1234567'\n")
    open_file = ROOT / "tests" / "fixtures" / "identity" / "2020[Zn][dia]3[ASR]1.cif"
    assert not _from_the_csd(open_file.read_text())


def test_no_structure_from_the_csd_is_published():
    found = [str(p.relative_to(ROOT)) for p in _published_files()
             if p.suffix in DATA and _from_the_csd(_text(p))]
    assert found == []


def test_every_test_structure_says_where_it_came_from():
    """Each fixture folder's README.txt names its structure files, with
    their source and licence."""
    unnamed = []
    for path in _published_files():
        if path.suffix != ".cif":
            continue
        readme = path.parent / "README.txt"
        if not readme.is_file() or path.name not in readme.read_text(encoding="utf-8"):
            unnamed.append(str(path.relative_to(ROOT)))
    assert unnamed == []


def _cff():
    """CITATION.cff's top-level scalars and its authors (the file is
    simple enough not to need a YAML library)."""
    fields, authors = {}, []
    for line in (ROOT / "CITATION.cff").read_text(encoding="utf-8").splitlines():
        if line.startswith("  - given-names:"):
            authors.append({"given-names": line.split(":", 1)[1].strip()})
        elif authors and line.startswith("    ") and ":" in line:
            key, value = line.strip().split(":", 1)
            authors[-1][key] = value.strip().strip('"')
        elif re.match(r"^[a-z-]+:", line):
            key, value = line.split(":", 1)
            fields[key] = value.strip().strip('"')
    return fields, authors


def test_licence_version_and_authors_agree_everywhere():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    fields, authors = _cff()

    assert project["version"] == fields["version"] == __version__
    assert project["license"] == citation.LICENCE_SPDX == fields["license"]
    assert project["license-files"] == ["LICENSE", "NOTICE.md"]
    assert fields["doi"] == citation.DOI
    assert [(a["given-names"], a["family-names"], a["affiliation"]) for a in authors] \
        == list(citation.AUTHORS)
    assert [a["name"] for a in project["authors"]] == [f"{g} {f}" for g, f, _ in citation.AUTHORS]

    licence = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "CC BY-NC-ND 4.0" in licence and citation.LICENCE_URL + "legalcode" in licence
    assert "non-commercial" in licence and "commercial purposes" in licence
    assert "NOTICE.md" in licence


def test_the_readme_and_about_cite_the_current_version():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"**Version {__version__}**" in readme
    assert f"> {citation.citation_text()}" in readme
    assert f"```bibtex\n{citation.bibtex()}\n```" in readme
    assert "CC BY-NC-ND 4.0" in readme and "NOTICE.md" in readme
    assert "CC BY-ND " not in readme  # the licence before round 75's change

    from thevoid_core.ui.dialogs.about_dialog import cite_html

    html = cite_html()
    assert citation.citation_text() in html
    assert citation.LICENCE_URL in html and "NOTICE.md" in html


def test_notice_accounts_for_the_bundled_data():
    """Every data file shipped in the package that is not wholly the
    author's own is named in NOTICE.md."""
    notice = (ROOT / "NOTICE.md").read_text(encoding="utf-8")
    own = {"method_changes.json"}  # the Void's own change log
    data = ROOT / "thevoid_core" / "data"
    shipped = [p for p in data.iterdir() if p.is_file() and p.suffix != ".py"]
    assert shipped
    for path in shipped:
        if path.name not in own:
            assert f"thevoid_core/data/{path.name}" in notice, path.name
    assert "thevoid_core/data/models/" in notice
    assert all((m / "card.json").is_file() and '"licence"' in (m / "card.json").read_text()
               for m in (data / "models").iterdir() if m.is_dir())
