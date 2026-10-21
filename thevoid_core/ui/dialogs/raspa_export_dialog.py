"""Parameters for File > Export Structure > RASPA2 GCMC Input Set...
(round 52).

Every field here changes what RASPA will simulate, not how anything is
displayed, so they are gathered before the export rather than left as
hidden defaults. The dialog also states the assumptions a person cannot
change here (the force field, and that the framework is uncharged when
the file has no charges) before anything is written, so nobody finds
out from a README after the fact.
"""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from thevoid_core.data.probe_radii import PROBE_RADII
from thevoid_core.services.raspa_export import (
    ADSORBATES,
    RaspaExportSettings,
)


def parse_pressures_bar(text: str) -> tuple[float, ...]:
    """'0.1, 1, 10' (bar) -> pressures in Pa. Raises ValueError with a
    message fit to show a person."""
    parts = [p for p in text.replace(";", ",").replace(" ", ",").split(",") if p]

    if not parts:
        raise ValueError("Enter at least one pressure.")

    try:
        values = [float(p) for p in parts]
    except ValueError:
        raise ValueError(
            "Pressures must be numbers in bar, separated by commas."
        ) from None

    if any(v <= 0 for v in values):
        raise ValueError("Pressures must be positive.")

    return tuple(v * 1.0e5 for v in values)


class RaspaExportDialog(QDialog):
    def __init__(
        self,
        parent=None,
        *,
        settings: RaspaExportSettings | None = None,
        has_cif_charges: bool = False,
    ) -> None:
        super().__init__(parent)

        settings = settings or RaspaExportSettings()
        #: Fields the dialog doesn't show (print interval, grid spacing,
        #: name) are carried through from here unchanged.
        self._base_settings = settings
        self._has_cif_charges = has_cif_charges

        self.setWindowTitle("Export RASPA2 GCMC Input Set")

        layout = QVBoxLayout(self)

        intro = QLabel(
            "Writes a self-contained RASPA2 input folder for a rigid-"
            "framework GCMC adsorption run: framework CIF, force field, "
            "adsorbate model, pocket blocking and simulation.input. It is "
            "a documented starting point; every assumption is listed in "
            "the folder's README.txt."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()

        self.adsorbate_box = QComboBox()
        for key, model in ADSORBATES.items():
            self.adsorbate_box.addItem(f"{key} - {model.description}", key)
        self.adsorbate_box.setCurrentIndex(
            max(0, self.adsorbate_box.findData(settings.adsorbate))
        )
        self.adsorbate_box.setToolTip(
            "TraPPE adsorbate model. CO2 and N2 carry partial charges; "
            "CH4 is a single uncharged site."
        )
        form.addRow("Adsorbate:", self.adsorbate_box)

        self.temperature_box = QDoubleSpinBox()
        self.temperature_box.setRange(1.0, 2000.0)
        self.temperature_box.setDecimals(2)
        self.temperature_box.setSuffix(" K")
        self.temperature_box.setValue(settings.temperature_K)
        form.addRow("Temperature:", self.temperature_box)

        self.pressure_edit = QLineEdit(
            ", ".join(f"{p / 1e5:g}" for p in settings.pressures_Pa)
        )
        self.pressure_edit.setToolTip(
            "One or more pressures in bar, separated by commas. RASPA runs "
            "one isotherm point per pressure, in order. Written to "
            "simulation.input in pascal."
        )
        form.addRow("Pressures (bar):", self.pressure_edit)

        self.init_cycles_box = QSpinBox()
        self.init_cycles_box.setRange(0, 10_000_000)
        self.init_cycles_box.setSingleStep(1000)
        self.init_cycles_box.setValue(settings.initialization_cycles)
        form.addRow("Initialisation cycles:", self.init_cycles_box)

        self.cycles_box = QSpinBox()
        self.cycles_box.setRange(1, 10_000_000)
        self.cycles_box.setSingleStep(1000)
        self.cycles_box.setValue(settings.cycles)
        form.addRow("Production cycles:", self.cycles_box)

        self.cutoff_box = QDoubleSpinBox()
        self.cutoff_box.setRange(4.0, 30.0)
        self.cutoff_box.setDecimals(2)
        self.cutoff_box.setSuffix(" Å")
        self.cutoff_box.setValue(settings.cutoff_A)
        self.cutoff_box.setToolTip(
            "Lennard-Jones and Ewald real-space cutoff. The number of unit "
            "cells is chosen so every perpendicular width exceeds twice "
            "this."
        )
        form.addRow("Cutoff:", self.cutoff_box)

        layout.addLayout(form)

        self.remove_free_box = QCheckBox(
            "Remove free (non-bonded) molecules - lattice solvent, "
            "counter-ions"
        )
        self.remove_free_box.setChecked(settings.remove_free_molecules)
        self.remove_free_box.setToolTip(
            "Molecules not bonded periodically into the framework are "
            "removed so the activated framework is simulated. Solvent "
            "coordinated to a metal is part of the framework and always "
            "kept. The README lists everything removed."
        )
        layout.addWidget(self.remove_free_box)

        self.block_box = QCheckBox(
            "Block pockets the adsorbate cannot reach"
        )
        self.block_box.setChecked(settings.block_pockets)
        self.block_box.setToolTip(
            "Void the adsorbate fits into but cannot reach through any "
            "channel is covered by RASPA blocking spheres, classified with "
            "the Pore Analysis grid at the probe radius below. If no "
            "channel is open to that probe at all, nothing is blocked and "
            "the export warns instead."
        )
        layout.addWidget(self.block_box)

        probe_form = QFormLayout()
        self.block_probe_box = QDoubleSpinBox()
        self.block_probe_box.setRange(0.5, 5.0)
        self.block_probe_box.setDecimals(3)
        self.block_probe_box.setSuffix(" Å")
        self.block_probe_box.setToolTip(
            "Probe radius used to find unreachable pockets. Defaults to "
            "the radius the rest of the application uses for this gas."
        )
        probe_form.addRow("Blocking probe radius:", self.block_probe_box)
        layout.addLayout(probe_form)

        self.qeq_box = QCheckBox(
            "Let RASPA assign framework charges (charge equilibration)"
        )
        self.qeq_box.setChecked(settings.charge_equilibration)
        self.qeq_box.setToolTip(
            "Only used when the input CIF has no _atom_site_charge column "
            "and the adsorbate is charged. RASPA computes the charges in a "
            "quick first step on one unit cell; the folder's run_raspa.sh "
            "runs that step and then the GCMC."
        )
        layout.addWidget(self.qeq_box)

        self.charge_label = QLabel("")
        self.charge_label.setWordWrap(True)
        layout.addWidget(self.charge_label)

        self.error_label = QLabel("")
        self.error_label.setStyleSheet("color: #c0392b;")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Choose Folder..."
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.button_box = buttons

        self._probe_follows_adsorbate = settings.blocking_probe_radius_A is None
        self.block_probe_box.setValue(
            settings.blocking_probe_radius_A
            if settings.blocking_probe_radius_A is not None
            else PROBE_RADII[ADSORBATES[settings.adsorbate].probe_radius_key]
        )
        self.block_probe_box.valueChanged.connect(self._on_probe_edited)
        self.adsorbate_box.currentIndexChanged.connect(self._on_adsorbate_changed)
        self.block_box.toggled.connect(self.block_probe_box.setEnabled)
        self.qeq_box.toggled.connect(lambda _: self._update_charge_label())
        self.block_probe_box.setEnabled(self.block_box.isChecked())
        self._update_charge_label()

    # ------------------------------------------------------------------

    def _adsorbate(self):
        return ADSORBATES[self.adsorbate_box.currentData()]

    def _on_probe_edited(self, _value) -> None:
        if not getattr(self, "_setting_probe", False):
            self._probe_follows_adsorbate = False

    def _on_adsorbate_changed(self, _index) -> None:
        if self._probe_follows_adsorbate:
            self._setting_probe = True
            self.block_probe_box.setValue(
                PROBE_RADII[self._adsorbate().probe_radius_key]
            )
            self._setting_probe = False

        self._update_charge_label()

    def _update_charge_label(self) -> None:
        model = self._adsorbate()
        self.qeq_box.setEnabled(not self._has_cif_charges)

        if self._has_cif_charges:
            text = (
                "Framework charges: taken from the file's _atom_site_charge "
                "column."
            )
            style = ""
        elif self.qeq_box.isChecked():
            text = (
                "Framework charges: computed by RASPA's charge equilibration "
                "in a first step - run the folder's run_raspa.sh."
            )
            style = ""
        elif model.is_charged:
            text = (
                f"Framework charges: none in this file, so the framework is "
                f"exported UNCHARGED and {model.name}'s electrostatic "
                "interaction with it is missing. This usually underestimates "
                "uptake."
            )
            style = "color: #b9770e;"
        else:
            text = (
                f"Framework charges: none; {model.name} is uncharged, so "
                "none are needed."
            )
            style = ""

        self.charge_label.setText(
            text + " Force field: UFF (framework) + TraPPE, Lorentz-Berthelot."
        )
        self.charge_label.setStyleSheet(style)

    def _on_accept(self) -> None:
        try:
            parse_pressures_bar(self.pressure_edit.text())
        except ValueError as exc:
            self.pressure_edit.setStyleSheet("border: 1px solid #c0392b;")
            self.error_label.setText(str(exc))
            self.error_label.setVisible(True)
            return

        self.pressure_edit.setStyleSheet("")
        self.error_label.setVisible(False)
        self.accept()

    def settings(self) -> RaspaExportSettings:
        model = self._adsorbate()
        default_probe = PROBE_RADII[model.probe_radius_key]
        probe = self.block_probe_box.value()

        return replace(
            self._base_settings,
            adsorbate=model.name,
            temperature_K=self.temperature_box.value(),
            pressures_Pa=parse_pressures_bar(self.pressure_edit.text()),
            initialization_cycles=self.init_cycles_box.value(),
            cycles=self.cycles_box.value(),
            cutoff_A=self.cutoff_box.value(),
            charge_equilibration=(
                self.qeq_box.isChecked() and not self._has_cif_charges
            ),
            remove_free_molecules=self.remove_free_box.isChecked(),
            block_pockets=self.block_box.isChecked(),
            blocking_probe_radius_A=(
                None if abs(probe - default_probe) < 5e-4 else probe
            ),
        )
