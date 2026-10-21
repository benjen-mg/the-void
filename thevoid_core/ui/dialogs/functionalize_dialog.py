"""Build > Functionalise Linkers... (round 59): hypothetical variants of
the loaded MOF with ring hydrogens of its linkers replaced by functional
groups (services/functionalize.py).

The dialog first reads the structure off the GUI thread. It finds the
linker orbits, the ring C-H sites on each, and the distinct patterns,
and says what it found. As groups, counts per linker and orbits are
ticked, it says how many variants that makes. Generating runs off the
GUI thread with a progress bar and can be cancelled. Each variant is
listed with its build check (ok / tight / rejected) and closest
contact; any of them can be opened in the main window. The folder gets
the parent, functionalization.json and variants.csv, and the dialog
ends by giving the command that screens them.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

STATUS_TEXT = {"ok": "ok", "tight": "tight", "clash": "rejected"}


class _Signals(QObject):
    progress = Signal(str, int, int)
    finished = Signal(object)
    failed = Signal(str)


class _Job(QRunnable):
    def __init__(self, function, *args, **kwargs):
        super().__init__()
        self.function, self.args, self.kwargs = function, args, kwargs
        self.signals = _Signals()
        self.cancelled = False

    def _progress(self, message, done, total):
        self.signals.progress.emit(message, int(done), int(total))
        return not self.cancelled

    @Slot()
    def run(self):
        try:
            if self.kwargs.pop("_with_progress", False):
                self.kwargs["progress"] = self._progress
            result = self.function(*self.args, **self.kwargs)
        except Exception as exc:  # reported in the dialog
            self.signals.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.signals.finished.emit(result)


class FunctionalizeDialog(QDialog):
    def __init__(self, structure, bonds=None, *, source=None, parent=None,
                 open_file=None) -> None:
        from thevoid_core.services.functionalize import GROUPS

        super().__init__(parent)
        self.setWindowTitle("Functionalise Linkers")
        self.resize(760, 640)
        self.source = source
        self.open_file = open_file
        self.parent_analysis = None
        self.record = None
        self.job = None
        self._pool = QThreadPool.globalInstance()

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Replace ring hydrogens of the linkers with functional groups. Groups are "
            "placed rigidly from standard geometry, and the framework is not moved. Every "
            "variant is checked for close contacts. The same pattern is used on every "
            "linker of an orbit.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.summary_label = QLabel("Reading the structure...")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        lists = QHBoxLayout()
        self.group_list = QListWidget()
        for name, group in GROUPS.items():
            item = QListWidgetItem(f"-{name}  ({group.description})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if name == "NH2" else Qt.CheckState.Unchecked)
            self.group_list.addItem(item)
        lists.addLayout(self._titled("Groups", self.group_list))
        self.orbit_list = QListWidget()
        lists.addLayout(self._titled("Linker orbits", self.orbit_list))
        counts = QVBoxLayout()
        counts.addWidget(QLabel("Groups per linker"))
        self.count_boxes = []
        for m in range(1, 5):
            box = QCheckBox(str(m))
            box.setChecked(m == 1)
            box.toggled.connect(self.update_plan)
            self.count_boxes.append(box)
            counts.addWidget(box)
        counts.addStretch(1)
        self.ring_turn_box = QCheckBox("Turn rings to make room")
        self.ring_turn_box.setChecked(True)
        self.ring_turn_box.setToolTip(
            "If a variant is tight or clashes, turn its substituted rings about the "
            "linker axis by the smallest angle that clears it, as real MOFs twist a "
            "crowded ring against its carboxylates. The same turn on every linker of "
            "the pattern, by symmetry.")
        counts.addWidget(self.ring_turn_box)
        lists.addLayout(counts)
        layout.addLayout(lists)

        folder_row = QHBoxLayout()
        stem = Path(source).stem if source else "framework"
        default = (Path(source).with_name(f"{stem}_functionalized") if source
                   else Path.home() / f"{stem}_functionalized")
        self.folder_edit = QLineEdit(str(default))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        folder_row.addWidget(QLabel("Folder:"))
        folder_row.addWidget(self.folder_edit, 1)
        folder_row.addWidget(browse)
        layout.addLayout(folder_row)

        self.plan_label = QLabel()
        layout.addWidget(self.plan_label)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Variant", "Check", "Closest contact", "Space group"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setWordWrap(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(lambda *_: self.open_selected())
        layout.addWidget(self.table, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.open_button = QPushButton("Open Selected")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(lambda: self.open_selected())
        buttons.addWidget(self.open_button)
        self.folder_button = QPushButton("Open Folder")
        self.folder_button.setEnabled(False)
        self.folder_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.folder_edit.text())))
        buttons.addWidget(self.folder_button)
        buttons.addStretch(1)
        self.generate_button = QPushButton("Generate")
        self.generate_button.setDefault(True)
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(lambda: self.start())
        buttons.addWidget(self.generate_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self._cancel_or_close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.group_list.itemChanged.connect(self.update_plan)
        self.orbit_list.itemChanged.connect(self.update_plan)

        from thevoid_core.services.functionalize import analyze_parent

        self.job = _Job(analyze_parent, structure, bonds, source=source, max_per_linker=4)
        self.job.signals.finished.connect(self._on_parent)
        self.job.signals.failed.connect(self._on_parent_failed)
        self._pool.start(self.job)

    @staticmethod
    def _titled(title, widget):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        box.addWidget(widget)
        return box

    # -- reading the parent ---------------------------------------------

    @Slot(object)
    def _on_parent(self, parent):
        self.job = None
        self.parent_analysis = parent
        text = (f"{parent.structure.composition.formula.replace(' ', '')}, "
                f"{parent.space_group}: {len(parent.sites)} ring C-H on "
                f"{len(parent.linker_sites)} linkers.")
        if parent.notes:
            text += " " + " ".join(parent.notes)
        self.summary_label.setText(text)
        self.orbit_list.blockSignals(True)
        for orbit in parent.orbits:
            item = QListWidgetItem(f"{orbit.number}: {len(orbit.linkers)} x {orbit.formula}, "
                                   f"{len(orbit.sites)} C-H each")
            item.setData(Qt.ItemDataRole.UserRole, orbit.number)
            if orbit.sites:
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked)
            else:
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.orbit_list.addItem(item)
        self.orbit_list.blockSignals(False)
        most = max((len(o.sites) for o in parent.orbits), default=0)
        for m, box in enumerate(self.count_boxes, start=1):
            box.setEnabled(m <= most)
        self.update_plan()

    @Slot(str)
    def _on_parent_failed(self, message):
        self.job = None
        self.summary_label.setText(f"This structure cannot be functionalised: {message}")

    # -- the plan ---------------------------------------------------------

    def chosen_groups(self):
        from thevoid_core.services.functionalize import group

        return [group(self.group_list.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.group_list.count())
                if self.group_list.item(i).checkState() == Qt.CheckState.Checked]

    def chosen_orbits(self):
        return [self.orbit_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.orbit_list.count())
                if self.orbit_list.item(i).checkState() == Qt.CheckState.Checked]

    def chosen_counts(self):
        return tuple(m for m, box in enumerate(self.count_boxes, start=1)
                     if box.isChecked() and box.isEnabled())

    def jobs(self):
        from thevoid_core.services.functionalize import plan

        if self.parent_analysis is None:
            return []
        orbits = self.chosen_orbits()
        if not orbits:
            return []
        return plan(self.parent_analysis, self.chosen_groups(), per_linker=self.chosen_counts(),
                    orbits=orbits)

    def update_plan(self, *_):
        if self.parent_analysis is None:
            return
        n = len(self.jobs())
        if not self.parent_analysis.sites:
            self.plan_label.setText("No ring C-H on a linker: nothing to substitute.")
        else:
            self.plan_label.setText(f"{n} variant(s) will be built." if n else
                                    "Tick at least one group, orbit and count.")
        self.generate_button.setEnabled(bool(n) and self.job is None)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder for the variants",
                                                  str(Path(self.folder_edit.text()).parent))
        if folder:
            self.folder_edit.setText(folder)

    # -- generating -------------------------------------------------------

    def start(self, synchronous=False) -> bool:
        from thevoid_core.services.functionalize import generate

        if self.parent_analysis is None or self.job is not None or not self.jobs():
            return False
        self.table.setRowCount(0)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.generate_button.setEnabled(False)
        self.close_button.setText("Cancel")
        self.job = _Job(generate, self.parent_analysis, self.chosen_groups(),
                        self.folder_edit.text(), per_linker=self.chosen_counts(),
                        orbits=self.chosen_orbits(), ring_turn=self.ring_turn_box.isChecked(),
                        _with_progress=True)
        self.job.signals.progress.connect(self._on_progress)
        self.job.signals.finished.connect(self._on_generated)
        self.job.signals.failed.connect(self._on_failed)
        if synchronous:
            self.job.run()
        else:
            self._pool.start(self.job)
        return True

    @Slot(str, int, int)
    def _on_progress(self, name, done, total):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status_label.setText(f"Building {name}...")

    @Slot(object)
    def _on_generated(self, record):
        cancelled = self.job is not None and self.job.cancelled
        self.job = None
        self.record = record
        self.progress.setVisible(False)
        self.close_button.setText("Close")
        variants = record["_variants"]
        self.table.setRowCount(len(variants))
        for row, variant in enumerate(variants):
            closest = variant.closest[0] if variant.closest else None
            check = STATUS_TEXT.get(variant.status, variant.status)
            if variant.ring_turn:
                check += f", ring {abs(variant.ring_turn):.0f}°"
            cells = [variant.file or variant.name, check,
                     f"{closest[1]:.2f} Å ({closest[0]:.2f} × vdW)" if closest else "",
                     variant.space_group]
            tips = ["", "\n".join(t for t in (variant.reason, variant.ring_note) if t),
                    closest[2] if closest else "", ""]
            for column, (text, tip) in enumerate(zip(cells, tips)):
                item = QTableWidgetItem(text)
                if tip:
                    item.setToolTip(tip)
                self.table.setItem(row, column, item)
        written = [v for v in variants if v.file]
        rejected = len(variants) - len(written)
        folder = self.folder_edit.text()
        text = f"{len(written)} written to {folder}"
        if rejected:
            text += (f", {rejected} rejected (a contact below 75% of the vdW sum, or a bond "
                     "the group was not placed to make; the Check column's tooltip says which)")
        if cancelled:
            text += " (cancelled)"
        text += (f". To screen them: the-void-analyze \"{folder}\"/*.cif --out screen "
                 f"--reference-csv \"{folder}/variants.csv\" --figures")
        self.status_label.setText(text)
        self.open_button.setEnabled(bool(written) and self.open_file is not None)
        self.folder_button.setEnabled(True)
        self.update_plan()

    @Slot(str)
    def _on_failed(self, message):
        self.job = None
        self.progress.setVisible(False)
        self.close_button.setText("Close")
        self.status_label.setText(f"Failed: {message}")
        self.update_plan()

    def open_selected(self, row=None):
        if self.record is None or self.open_file is None:
            return None
        if row is None:
            rows = self.table.selectionModel().selectedRows()
            row = rows[0].row() if rows else 0
        variant = self.record["_variants"][row]
        if not variant.file:
            self.status_label.setText(f"{variant.name} was rejected: {variant.reason}")
            return None
        path = str(Path(self.folder_edit.text()) / variant.file)
        self.open_file(path)
        return path

    def _cancel_or_close(self):
        if self.job is not None and self.parent_analysis is not None:
            self.job.cancelled = True
            self.status_label.setText("Cancelling after the current variant...")
        else:
            self.close()
