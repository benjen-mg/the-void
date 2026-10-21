"""File > Fetch Reference Structures... (round 56): download CoRE MOF
2024 or MOFX-DB structures into a folder, ready for the-void-analyze
(services/reference/fetch.py).

For CoRE MOF the query is answered from the bundled index as it is
typed - "41 downloadable, 257 from the CSD (listed, not downloaded)" -
so a person knows before downloading anything what the licence allows.
MOFX-DB is queried live. The download runs off the GUI thread with a
progress bar and can be cancelled; the folder gets reference.csv and a
README with the citation and licence, and the dialog ends by saying how
to analyse it.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

SOURCES = [("CoRE MOF 2024 (openly licensed structures)", "coremof"),
           ("MOFX-DB (hMOF and others; online)", "mofxdb")]


def parse_number(text: str, name: str):
    """'' -> None; a number -> float; anything else -> ValueError with a
    message fit to show."""
    text = text.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"{name}: '{text}' is not a number") from None


class _Signals(QObject):
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)


class _FetchJob(QRunnable):
    def __init__(self, function, **kwargs):
        super().__init__()
        self.function, self.kwargs = function, kwargs
        self.signals = _Signals()
        self.cancelled = False

    def _progress(self, message, done, total):
        self.signals.progress.emit(message, int(done), int(total))
        return not self.cancelled

    @Slot()
    def run(self):
        try:
            result = self.function(progress=self._progress, **self.kwargs)
        except Exception as exc:  # reported in the dialog
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result)


class FetchReferenceDialog(QDialog):
    def __init__(self, parent=None, *, default_folder: str | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fetch Reference Structures")
        self.resize(560, 520)
        self.job = None
        self.result = None
        self._pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Download reference MOF structures into a folder, with the "
            "database's own values in reference.csv and its citation and "
            "licence in README.txt, ready for the-void-analyze."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.source_box = QComboBox()
        for label, key in SOURCES:
            self.source_box.addItem(label, key)
        form.addRow("Source:", self.source_box)

        self.pld_min, self.pld_max = QLineEdit(), QLineEdit()
        self.lcd_min, self.lcd_max = QLineEdit(), QLineEdit()
        for edit in (self.pld_min, self.pld_max, self.lcd_min, self.lcd_max):
            edit.setPlaceholderText("any")
        form.addRow("PLD (Å), min / max:", self._pair(self.pld_min, self.pld_max))
        form.addRow("LCD (Å), min / max:", self._pair(self.lcd_min, self.lcd_max))
        layout.addLayout(form)

        self.stack = QStackedWidget()
        # CoRE MOF page
        core = QWidget()
        core_form = QFormLayout(core)
        core_form.setContentsMargins(0, 0, 0, 0)
        self.subset_box = QComboBox()
        for label, value in (("any", None), ("ASR: all solvent removed", "ASR"),
                             ("FSR: free solvent removed", "FSR"), ("Ion: charged frameworks", "Ion")):
            self.subset_box.addItem(label, value)
        core_form.addRow("Subset:", self.subset_box)
        self.metals_edit = QLineEdit()
        self.metals_edit.setPlaceholderText("e.g. Cu, Zn (all must be present)")
        core_form.addRow("Metals:", self.metals_edit)
        self.topology_edit = QLineEdit()
        self.topology_edit.setPlaceholderText("e.g. pcu")
        core_form.addRow("Topology:", self.topology_edit)
        self.oms_box = QComboBox()
        for label, value in (("any", None), ("with open metal sites", True),
                             ("without open metal sites", False)):
            self.oms_box.addItem(label, value)
        core_form.addRow("Open metal sites:", self.oms_box)
        self.asa_min = QLineEdit()
        self.asa_min.setPlaceholderText("any (CoRE's value, 1.655 Å probe)")
        core_form.addRow("ASA min (m²/g):", self.asa_min)
        self.stack.addWidget(core)
        # MOFX-DB page
        mofx = QWidget()
        mofx_form = QFormLayout(mofx)
        mofx_form.setContentsMargins(0, 0, 0, 0)
        from thevoid_core.services.reference.fetch import MOFXDB_DATABASES

        self.database_box = QComboBox()
        self.database_box.addItems(list(MOFXDB_DATABASES))
        mofx_form.addRow("Database:", self.database_box)
        self.elements_edit = QLineEdit()
        self.elements_edit.setPlaceholderText("e.g. Zn, O")
        mofx_form.addRow("Elements:", self.elements_edit)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("name prefix, e.g. hMOF-1")
        mofx_form.addRow("Name:", self.name_edit)
        self.vf_min = QLineEdit()
        self.vf_min.setPlaceholderText("any")
        mofx_form.addRow("Void fraction min:", self.vf_min)
        self.stack.addWidget(mofx)
        self.stack.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout.addWidget(self.stack)

        tail = QFormLayout()
        self.limit_box = QSpinBox()
        self.limit_box.setRange(1, 100000)
        self.limit_box.setValue(100)
        tail.addRow("At most:", self.limit_box)
        self.folder_edit = QLineEdit(default_folder or str(Path.home() / "reference_structures"))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        tail.addRow("Folder:", self._pair(self.folder_edit, browse))
        layout.addLayout(tail)

        self.plan_label = QLabel()
        self.plan_label.setWordWrap(True)
        layout.addWidget(self.plan_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

        buttons = QHBoxLayout()
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.setEnabled(False)
        self.open_folder_button.clicked.connect(self._open_folder)
        buttons.addWidget(self.open_folder_button)
        buttons.addStretch(1)
        self.fetch_button = QPushButton("Fetch")
        self.fetch_button.setDefault(True)
        self.fetch_button.clicked.connect(self.start)
        buttons.addWidget(self.fetch_button)
        self.cancel_button = QPushButton("Close")
        self.cancel_button.clicked.connect(self._cancel_or_close)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)

        self.source_box.currentIndexChanged.connect(self._source_changed)
        for edit in (self.pld_min, self.pld_max, self.lcd_min, self.lcd_max,
                     self.metals_edit, self.topology_edit, self.asa_min):
            edit.textChanged.connect(self.update_plan)
        for box in (self.subset_box, self.oms_box):
            box.currentIndexChanged.connect(self.update_plan)
        self.limit_box.valueChanged.connect(self.update_plan)
        self._source_changed()

    @staticmethod
    def _pair(first, second):
        widget = QWidget()
        row = QHBoxLayout(widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(first)
        row.addWidget(second)
        return widget

    @property
    def source(self) -> str:
        return self.source_box.currentData()

    def _source_changed(self):
        self.stack.setCurrentIndex(self.source_box.currentIndex())
        self.update_plan()

    # ------------------------------------------------------------------

    def coremof_filters(self) -> dict:
        return {
            "subset": self.subset_box.currentData(),
            "pld_min": parse_number(self.pld_min.text(), "PLD min"),
            "pld_max": parse_number(self.pld_max.text(), "PLD max"),
            "lcd_min": parse_number(self.lcd_min.text(), "LCD min"),
            "lcd_max": parse_number(self.lcd_max.text(), "LCD max"),
            "asa_min": parse_number(self.asa_min.text(), "ASA min"),
            "metals": [m.strip() for m in self.metals_edit.text().split(",") if m.strip()],
            "topology": self.topology_edit.text().strip() or None,
            "has_oms": self.oms_box.currentData(),
        }

    def mofxdb_filters(self) -> dict:
        return {
            "pld_min": parse_number(self.pld_min.text(), "PLD min"),
            "pld_max": parse_number(self.pld_max.text(), "PLD max"),
            "lcd_min": parse_number(self.lcd_min.text(), "LCD min"),
            "lcd_max": parse_number(self.lcd_max.text(), "LCD max"),
            "vf_min": parse_number(self.vf_min.text(), "Void fraction min"),
            "name": self.name_edit.text().strip() or None,
            "elements": [e.strip() for e in self.elements_edit.text().split(",") if e.strip()],
        }

    def update_plan(self) -> None:
        if self.source == "mofxdb":
            self.plan_label.setText("MOFX-DB is queried online when you press Fetch "
                                    "(https://mof.tech.northwestern.edu).")
            self.fetch_button.setEnabled(self.job is None)
            return
        try:
            filters = self.coremof_filters()
        except ValueError as exc:
            self.plan_label.setText(str(exc))
            self.fetch_button.setEnabled(False)
            return
        from thevoid_core.services.reference.fetch import plan_coremof

        downloadable, others = plan_coremof(filters, limit=self.limit_box.value())
        self.plan_label.setText(
            f"{len(downloadable)} structure(s) to download; {len(others)} more match but "
            "come from the CSD, which CoRE MOF cannot redistribute - they will be listed, "
            "with refcodes, in not_downloadable.csv."
        )
        self.fetch_button.setEnabled(self.job is None and bool(downloadable or others))

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder for the structures",
                                                  self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def start(self) -> bool:
        from thevoid_core.services.reference import fetch

        folder = self.folder_edit.text().strip()
        if not folder:
            self.status_label.setText("Choose a folder first.")
            return False
        try:
            if self.source == "coremof":
                job = _FetchJob(fetch.fetch_coremof, folder=folder, filters=self.coremof_filters(),
                                limit=self.limit_box.value())
            else:
                job = _FetchJob(fetch.fetch_mofxdb, folder=folder,
                                database=self.database_box.currentText(),
                                limit=self.limit_box.value(), **self.mofxdb_filters())
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return False
        self.job = job
        job.signals.progress.connect(self._on_progress)
        job.signals.finished.connect(self._on_finished)
        job.signals.failed.connect(self._on_failed)
        self.fetch_button.setEnabled(False)
        self.cancel_button.setText("Cancel")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("Starting…")
        self._pool.start(job)
        return True

    def _on_progress(self, message, done, total):
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
        self.status_label.setText(message)

    def _finish(self):
        self.job = None
        self.progress.setVisible(False)
        self.cancel_button.setText("Close")
        self.update_plan()

    def _on_finished(self, result):
        self.result = result
        self._finish()
        lines = [f"{len(result.written)} structure(s) written to {result.folder}."]
        if result.not_downloadable:
            lines.append(f"{len(result.not_downloadable)} CSD-only match(es) listed in "
                         "not_downloadable.csv.")
        if result.skipped:
            lines.append(f"{len(result.skipped)} skipped (see README.txt).")
        lines += result.warnings
        compare = " --compare-coremof" if result.source.startswith("CoRE") else ""
        lines.append("Analyse them with:  the-void-analyze *.cif --out results "
                     f"--reference-csv reference.csv{compare}")
        self.status_label.setText("\n".join(lines))
        self.open_folder_button.setEnabled(True)

    def _on_failed(self, message):
        self._finish()
        self.status_label.setText(
            "Cancelled." if message.startswith("Cancelled") else f"Failed: {message}")
        self.open_folder_button.setEnabled(Path(self.folder_edit.text()).is_dir())

    def _cancel_or_close(self):
        if self.job is not None:
            self.job.cancelled = True
            self.status_label.setText("Cancelling…")
        else:
            self.close()

    def _open_folder(self):
        folder = self.result.folder if self.result else Path(self.folder_edit.text())
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))
