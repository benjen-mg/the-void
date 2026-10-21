"""Round 52: the RASPA2 export as a person reaches it - the parameter
dialog, File > Export Structure / Simulate in the main window, the
analysis report, and `the-void-analyze --raspa`.

The service itself is tested in test_raspa_export.py; these check the
wiring: that what the dialog shows is what gets exported, that the
window writes a folder of its own and records it in the report, and
that a batch export failure is recorded rather than fatal.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from tests.sample_structures import ZN_DIA_UNCHARGED  # noqa: E402
from thevoid_core import cli  # noqa: E402
from thevoid_core.services.raspa_export import (  # noqa: E402
    MANIFEST_NAME,
    RaspaExportSettings,
)
from thevoid_core.ui.dialogs.raspa_export_dialog import (  # noqa: E402
    RaspaExportDialog,
    parse_pressures_bar,
)

# The open Zn-dia sample without its charges: the uncharged case.
FIXTURE = ZN_DIA_UNCHARGED
NAME = FIXTURE.stem


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------
# Dialog
# ----------------------------------------------------------------------


def test_dialog_defaults_are_the_documented_defaults(qt_app):
    settings = RaspaExportDialog().settings()
    assert settings == RaspaExportSettings()


def test_dialog_round_trips_what_it_is_given(qt_app):
    given = RaspaExportSettings(
        adsorbate="N2", temperature_K=77.0, pressures_Pa=(1e3, 1e5),
        initialization_cycles=100, cycles=200, cutoff_A=14.0,
        remove_free_molecules=False, block_pockets=False,
        blocking_probe_radius_A=1.4,
    )
    assert RaspaExportDialog(settings=given).settings() == given


def test_pressures_are_entered_in_bar_and_exported_in_pascal():
    assert parse_pressures_bar("0.1, 1 ,10") == pytest.approx((1e4, 1e5, 1e6))
    for bad in ("", "one", "1, -2"):
        with pytest.raises(ValueError):
            parse_pressures_bar(bad)


def test_bad_pressures_keep_the_dialog_open(qt_app):
    dialog = RaspaExportDialog()
    dialog.pressure_edit.setText("1, fish")
    dialog._on_accept()
    assert dialog.result() == 0
    assert "bar" in dialog.error_label.text()


def test_blocking_probe_follows_the_adsorbate_until_edited(qt_app):
    """Switching CO2 -> CH4 should switch the probe to CH4's radius, but
    never overwrite a value the person typed."""
    dialog = RaspaExportDialog()
    assert dialog.block_probe_box.value() == pytest.approx(1.525)
    dialog.adsorbate_box.setCurrentIndex(dialog.adsorbate_box.findData("CH4"))
    assert dialog.block_probe_box.value() == pytest.approx(1.90)
    assert dialog.settings().blocking_probe_radius_A is None

    dialog.block_probe_box.setValue(1.3)
    dialog.adsorbate_box.setCurrentIndex(dialog.adsorbate_box.findData("N2"))
    assert dialog.block_probe_box.value() == pytest.approx(1.3)
    assert dialog.settings().blocking_probe_radius_A == pytest.approx(1.3)


def test_dialog_says_the_framework_is_uncharged_before_export(qt_app):
    """The single assumption most likely to change a number is stated
    up front, not only in a README found afterwards."""
    dialog = RaspaExportDialog()
    assert "UNCHARGED" in dialog.charge_label.text()
    dialog.adsorbate_box.setCurrentIndex(dialog.adsorbate_box.findData("CH4"))
    assert "UNCHARGED" not in dialog.charge_label.text()

    charged = RaspaExportDialog(has_cif_charges=True)
    assert "_atom_site_charge" in charged.charge_label.text()
    assert not charged.qeq_box.isEnabled()
    charged.qeq_box.setChecked(True)
    assert charged.settings().charge_equilibration is False


# ----------------------------------------------------------------------
# Main window
# ----------------------------------------------------------------------


@pytest.fixture
def window(qt_app, monkeypatch):
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")
    from thevoid_core.ui.windows.main_window import MainWindow

    messages = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: messages.append(a[2])))
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: messages.append(a[2])))

    win = MainWindow()
    win._messages = messages
    win._on_structure_ready(win._load_and_analyse(str(FIXTURE)))
    # Run the worker synchronously, as the other window tests do.
    monkeypatch.setattr(win, "_run", lambda fn, done, *a: done(fn(*a)))
    yield win
    win.close()


def test_both_menu_entries_exist_and_explain_themselves(window):
    assert window.export_raspa_action.text() == "RASPA2 GCMC Input Set..."
    assert window.simulate_raspa_action.text() == "Export RASPA2 GCMC Input Set..."
    assert window.export_raspa_action.toolTip()
    titles = [a.text() for a in window.menuBar().actions()]
    assert titles == ["File", "View", "Analysis", "Build", "Simulate", "Measure", "Help"]


def test_window_export_writes_the_folder_and_records_it(window, tmp_path):
    window.raspa_settings = RaspaExportSettings(cycles=10, initialization_cycles=10)
    target = tmp_path / f"{NAME}_RASPA_CO2"
    window.export_raspa_input_set(directory=str(target))

    assert (target / "simulation.input").is_file()
    assert (target / f"{NAME}.cif").is_file()
    manifest = json.loads((target / MANIFEST_NAME).read_text())
    assert manifest["input"]["file"].endswith(f"{NAME}.cif")
    assert not window._busy

    # The summary dialog shows the warnings (uncharged; not blocked).
    assert any("UNCHARGED" in m for m in window._messages)

    report = window.build_analysis_report()
    assert report["results"]["raspa_export"]["unit_cells"] == [3, 2, 2]
    md = tmp_path / "r.md"
    assert window.export_analysis_report(str(md))
    assert "## RASPA2 export" in md.read_text(encoding="utf-8")


def test_existing_files_are_not_replaced_without_asking(window, tmp_path, monkeypatch):
    window.raspa_settings = RaspaExportSettings(cycles=10, initialization_cycles=10)
    target = tmp_path / "set"
    target.mkdir()
    (target / "simulation.input").write_text("mine")

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    window.export_raspa_input_set(directory=str(target))
    assert (target / "simulation.input").read_text() == "mine"
    assert window.current_raspa_export_summary is None

    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    window.export_raspa_input_set(directory=str(target))
    assert (target / "simulation.input").read_text() != "mine"


def test_loading_a_structure_forgets_the_last_export(window, tmp_path):
    window.raspa_settings = RaspaExportSettings(cycles=10, initialization_cycles=10)
    window.export_raspa_input_set(directory=str(tmp_path / "a"))
    assert window.current_raspa_export_summary is not None
    window._on_structure_ready(window._load_and_analyse(str(FIXTURE)))
    assert window.current_raspa_export_summary is None


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

FAST = ["--samples", "200", "--no-surface", "--no-oms"]


def test_cli_writes_a_raspa_folder_per_structure(tmp_path):
    out = tmp_path / "out"
    code = cli.main([str(FIXTURE), "--out", str(out), "-q", *FAST, "--raspa",
                     "--raspa-adsorbate", "N2", "--raspa-pressures", "0.1,1",
                     "--raspa-cycles", "100"])

    folder = out / f"{NAME}_RASPA_N2"
    text = (folder / "simulation.input").read_text()
    assert "ExternalPressure              10000 100000" in text
    assert "NumberOfCycles                100" in text
    report = json.loads((out / f"{NAME}.json").read_text())
    assert report["results"]["raspa_export"]["adsorbate"] == "N2"
    assert "## RASPA2 export" in (out / f"{NAME}.md").read_text()
    assert "raspa_folder" in (out / "summary.csv").read_text().splitlines()[0]
    assert code == 0


def test_cli_records_a_failed_export_and_carries_on(tmp_path, monkeypatch):
    from thevoid_core.services import raspa_export

    def broken(*a, **k):
        raise ValueError("no framework")

    monkeypatch.setattr(raspa_export, "build_raspa_export", broken)
    out = tmp_path / "out"
    code = cli.main([str(FIXTURE), "--out", str(out), "-q", *FAST, "--raspa"])

    report = json.loads((out / f"{NAME}.json").read_text())
    assert report["errors"]["raspa_export"] == "ValueError: no framework"
    assert "pore" in report["results"]
    assert code == 1


def test_cli_rejects_bad_raspa_arguments(tmp_path):
    with pytest.raises(SystemExit) as info:
        cli.main([str(FIXTURE), "--out", str(tmp_path), "--raspa",
                  "--raspa-pressures", "one"])
    assert info.value.code == 2
    with pytest.raises(SystemExit) as info:
        cli.main([str(FIXTURE), "--out", str(tmp_path), "--raspa",
                  "--raspa-temperature", "-5"])
    assert info.value.code == 2


def test_cli_leaves_logging_as_it_found_it(tmp_path):
    """With -j 1 the worker runs in the calling process. It used to call
    logging.disable() and never undo it, switching logging off for
    everything that ran afterwards in the same process - which is how
    this round found it: a Help-menu logging test failed whenever a CLI
    test happened to run first."""
    import logging

    before = logging.root.manager.disable
    cli.main([str(FIXTURE), "--out", str(tmp_path / "out"), "-q", *FAST, "--no-pore"])
    assert logging.root.manager.disable == before


def test_the_simulate_menu_entry_asks_and_writes(window, tmp_path, monkeypatch):
    """Round 72 bug: clicked from the menu, Qt passed the action's
    checked state (False) as `directory`, so the dialogs were skipped
    and the export tried to write to the folder False ("TypeError:
    argument should be a str or an os.PathLike object ..., not 'bool'").
    The tests above called the method directly and never saw it. This
    goes through the menu action, as a click does."""
    from PySide6.QtWidgets import QDialog, QFileDialog

    from thevoid_core.ui.dialogs import raspa_export_dialog

    asked = []

    def accept(dialog):
        asked.append("settings")
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(raspa_export_dialog.RaspaExportDialog, "exec", accept)
    for number, action in enumerate((window.simulate_raspa_action, window.export_raspa_action)):
        parent = tmp_path / str(number)
        parent.mkdir()
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            staticmethod(lambda *a, _p=parent, **k: asked.append("folder") or str(_p)))
        asked.clear()
        window.raspa_settings = RaspaExportSettings(cycles=10, initialization_cycles=10)
        action.trigger()
        assert asked == ["settings", "folder"]
        assert (parent / f"{NAME}_RASPA_CO2" / "simulation.input").is_file()
        assert window.current_raspa_export_summary is not None
