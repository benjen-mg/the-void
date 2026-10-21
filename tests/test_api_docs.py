"""Round 60: every example in docs/python-api.md runs, as written.

The examples are run in order in one namespace, as a reader would type
them, in a scratch folder where tests/fixtures is available at the same
relative path. Documentation that has stopped working fails the suite,
rather than a reader's first attempt.
"""

import os
import re
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "docs" / "python-api.md"


def _blocks():
    return re.findall(r"```python\n(.*?)```", DOC.read_text(encoding="utf-8"), flags=re.S)


def test_the_page_has_examples():
    assert len(_blocks()) >= 8


def test_every_example_runs(tmp_path, monkeypatch, capsys):
    pytest.importorskip("ase")
    (tmp_path / "tests").mkdir()
    os.symlink(ROOT / "tests" / "fixtures", tmp_path / "tests" / "fixtures")
    monkeypatch.chdir(tmp_path)
    namespace = {}
    for number, block in enumerate(_blocks(), start=1):
        try:
            exec(compile(block, f"docs/python-api.md, example {number}", "exec"), namespace)
        except Exception as exc:
            raise AssertionError(f"example {number} failed: {type(exc).__name__}: {exc}\n{block}")
    out = capsys.readouterr().out
    # The numbers the page quotes are the numbers it prints.
    assert "5.11 4.078" in out
    assert "dia 4-c" in out
    assert "{'C6H5N2O2': 8}" in out
    assert "\n4.12\n" in out
    assert "BatchTable(3 structures, 0 with errors)" in out
    assert (tmp_path / "screen" / "summary.csv").is_file()
    assert (tmp_path / "screen" / "figures" / "pld_vs_asa.pdf").is_file()
    assert (tmp_path / "cr_variants" / "variants.csv").is_file()
    assert (tmp_path / "zn_dia.json").is_file()
    assert "AddSummary(3 added, 0 already present, 0 skipped, 0 stored, 3 withheld)" in out
    assert "same file 2010[Cr][bpq]3[ASR]1.cif" in out
    assert (tmp_path / "lab-index" / "build" / "summary.csv").is_file()
