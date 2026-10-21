"""Round 74: one name. The application was "the Void" in its window
title, menus, reports and commands, but its package (and the command
that started the desktop application, and the pip distribution) was
still called crystalstudio, an earlier working name. Everything is now
the Void: the internal package is thevoid_core, the desktop command is
the-void, and the distribution is the-void.
"""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = re.compile(r"crystal[\s_-]?studio", re.IGNORECASE)
TEXT = {".py", ".md", ".toml", ".txt", ".json", ".yml", ".yaml", ".cfg", ".ini", ".sh"}


def test_the_old_name_is_gone():
    found = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT or "__pycache__" in path.parts:
            continue
        if path == Path(__file__).resolve():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace")
                                      .splitlines(), start=1):
            if OLD.search(line):
                found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert not (ROOT / "crystalstudio").exists()
    assert found == []


def test_the_commands_and_distribution_are_the_void():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["name"] == "the-void"
    scripts = project["scripts"]
    assert scripts["the-void"] == "thevoid_core.app.application:main"
    assert all(name.startswith("the-void") for name in scripts)
    assert all(target.startswith("thevoid_core.") for target in scripts.values())


def test_what_the_user_sees_says_the_void():
    from thevoid_core.app.application import TheVoidApplication  # noqa: F401
    from thevoid_core.infrastructure.settings import Settings
    from thevoid_core.services.export import export_cp2k

    assert Settings.ORGANIZATION == Settings.APPLICATION == "the Void"
    import inspect

    assert inspect.signature(export_cp2k).parameters["project_name"].default == "the_void"
