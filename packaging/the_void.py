"""Entry point of the Windows build (packaging/the_void.spec).

Starts the desktop application. Two extras that only the frozen build
needs:

- multiprocessing.freeze_support(), so that the worker processes the
  shared index's parallel refresh starts run their work instead of
  opening another copy of the application;
- ``--self-test OUT.json``: loads the bundled example structure, runs
  the main analyses and opens the main window off-screen, then writes
  what happened to OUT.json and exits (0 when everything worked). The
  release workflow runs it on the built application before making the
  installer, because a frozen build can lack a module or a data file
  that a normal Python install has.
"""

import json
import multiprocessing
import os
import sys
import traceback
from pathlib import Path


def _bundle_dir() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def example_structure() -> Path:
    return _bundle_dir() / "examples" / "2020[Zn][dia]3[ASR]1.cif"


def self_test(out: Path) -> int:
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    checks = {}

    def check(name, function):
        try:
            checks[name] = {"ok": True, "value": function()}
        except Exception:
            checks[name] = {"ok": False, "error": traceback.format_exc()}

    import thevoid

    example = example_structure()
    fw = thevoid.load(example)
    check("version", lambda: thevoid.__version__)
    check("load", lambda: repr(fw))
    check("pores", lambda: round(fw.pores().pld_A, 3))
    check("surface_area", lambda: round(fw.surface_area("N2", samples_per_atom=100)
                                        .channel_asa_m2_per_g))
    check("topology", lambda: fw.topology().summary)
    check("open_metal_sites", lambda: fw.open_metal_sites().open_metal_sites)
    check("xrd", lambda: len(fw.xrd().two_theta_deg))
    check("coremof", lambda: fw.coremof_matches()[0][0])
    check("predict", lambda: {r.model: r.value for r in fw.predict(samples_per_atom=100).results})

    def gui():
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication([])
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        from thevoid_core.ui.windows.main_window import MainWindow

        window = MainWindow()
        window._on_structure_ready(window._load_and_analyse(str(example)))
        title = window.windowTitle()
        window.close()
        app.processEvents()
        return title

    check("main_window", gui)

    ok = all(item["ok"] for item in checks.values())
    out.write_text(json.dumps({"ok": ok, "checks": checks}, indent=2, default=str),
                   encoding="utf-8")
    return 0 if ok else 1


def main() -> int:
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "--self-test":
        return self_test(Path(sys.argv[2]))
    from thevoid_core.app.application import main as run_application

    return run_application()


if __name__ == "__main__":
    sys.exit(main())
