# PyInstaller recipe for the desktop application (one folder, no console).
#
#     pyinstaller packaging/the_void.spec --noconfirm
#
# writes dist/the Void/, with "the Void.exe" (on Windows) inside. The
# release workflow (.github/workflows/release.yml) runs it on Windows, checks
# the result with "the Void.exe --self-test", and wraps the folder into an
# installer with packaging/the_void.iss.
#
# Most packages are found by following imports. Those listed below are not:
# they read data files at run time (pymatgen, RDKit, spglib), load compiled
# modules by name (VTK), or are needed only to unpickle the bundled property
# models (scikit-learn).

import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent  # noqa: F821 - defined by PyInstaller
APP = ROOT / "thevoid_core"
VERSION = re.search(r'"(.+)"', (APP / "version.py").read_text()).group(1)


def windows_version_file() -> str | None:
    """The version shown in the .exe's Properties > Details (Windows only)."""
    if sys.platform != "win32":
        return None
    numbers = tuple(int(n) for n in (VERSION.split(".") + ["0", "0", "0"])[:4])
    path = Path(workpath) / "version_info.txt"  # noqa: F821 - defined by PyInstaller
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(filevers={numbers}, prodvers={numbers}),
  kids=[StringFileInfo([StringTable('040904B0', [
    StringStruct('CompanyName', 'Benyamin M. Garmejani'),
    StringStruct('FileDescription', 'the Void'),
    StringStruct('FileVersion', '{VERSION}'),
    StringStruct('InternalName', 'the Void'),
    StringStruct('LegalCopyright', 'Copyright (c) 2026 Benyamin M. Garmejani. CC BY-NC-ND 4.0'),
    StringStruct('OriginalFilename', 'the Void.exe'),
    StringStruct('ProductName', 'the Void'),
    StringStruct('ProductVersion', '{VERSION}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])])
""", encoding="utf-8")
    return str(path)

datas = [
    (str(APP / "data"), "thevoid_core/data"),
    (str(APP / "assets"), "thevoid_core/assets"),
    # An example structure for new users (CoRE MOF 2024, CC-BY-4.0), and
    # the documents the licence requires to travel with every copy.
    (str(ROOT / "tests" / "fixtures" / "identity" / "2020[Zn][dia]3[ASR]1.cif"), "examples"),
    (str(ROOT / "packaging" / "EXAMPLES.txt"), "examples"),
    (str(ROOT / "LICENSE"), "."),
    (str(ROOT / "NOTICE.md"), "."),
    (str(ROOT / "CITATION.cff"), "."),
    (str(ROOT / "README.md"), "."),
]
binaries = []
hiddenimports = []

for package in ("pymatgen", "rdkit", "spglib", "pyvista", "pyvistaqt"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

datas += collect_data_files("matplotlib")
hiddenimports += collect_submodules("vtkmodules")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("thevoid_core")
hiddenimports += collect_submodules("thevoid")
hiddenimports += ["joblib"]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "the_void.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    # mypy: pyvista imports its plugin only when mypy is present, and a
    # frozen copy of mypy cannot load its compiled parts.
    excludes=["tkinter", "pytest", "IPython", "jupyter", "notebook", "PyQt5", "PyQt6",
              "mypy", "black", "ruff"],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="the Void",
    console=False,
    icon=str(APP / "assets" / "icon.ico"),
    version=windows_version_file(),
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    name="the Void",
)
