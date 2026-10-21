"""File > Shared Index > Add... and Batch Results > Add to Shared Index...
(round 66): where to add, as whom, and under what licence.

The licence choice defaults to deciding from where each file came from
(services/index/licence.py): CoRE MOF files from the-void-fetch are
stored with their CC-BY-4.0 attribution, a file with the CSD's header
never is, and anything else keeps only its results. Choosing a licence
here declares it for the files nothing else decides - it can never
override a CSD header - and asks why, because the reason is recorded in
every entry.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

LICENCES = [
    ("Decide from where each file came from", None),
    ("CC-BY-4.0", "CC-BY-4.0"),
    ("CC0-1.0", "CC0-1.0"),
    ("CC-BY-SA-4.0", "CC-BY-SA-4.0"),
]


def is_index(folder) -> bool:
    return bool(folder) and (Path(folder) / "INDEX.json").is_file()


def create_index(parent, start: str = "") -> str | None:
    """Ask for a folder and a name and make a new index there; the
    folder, or None if cancelled."""
    from thevoid_core.services.index.store import Index, IndexProblem

    folder = QFileDialog.getExistingDirectory(parent, "Folder for the New Index", start)
    if not folder:
        return None
    name, ok = QInputDialog.getText(parent, "New Shared Index",
                                    "Name (e.g. the group's):", text=Path(folder).name)
    if not ok or not name.strip():
        return None
    try:
        Index.init(folder, name=name.strip())
    except IndexProblem as exc:
        QMessageBox.warning(parent, "New Shared Index", str(exc))
        return None
    return folder


class SharedIndexAddDialog(QDialog):
    def __init__(self, parent=None, *, what: str = "", index_path: str = "",
                 contributor: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle("Add to Shared Index")
        self.setModal(True)
        self.resize(560, 280)

        layout = QVBoxLayout(self)
        self.what_label = QLabel(what)
        self.what_label.setWordWrap(True)
        layout.addWidget(self.what_label)

        form = QFormLayout()
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        self.path_edit = QLineEdit(index_path)
        self.path_edit.setPlaceholderText("an index folder (INDEX.json)")
        self.path_edit.textChanged.connect(self._validate)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        new = QPushButton("New...")
        new.setToolTip("Make a new, empty index (the-void-index init).")
        new.clicked.connect(self._new)
        row_layout.addWidget(self.path_edit, 1)
        row_layout.addWidget(browse)
        row_layout.addWidget(new)
        form.addRow("Index:", row)

        self.contributor_edit = QLineEdit(contributor)
        self.contributor_edit.setPlaceholderText("a name, or initials")
        self.contributor_edit.textChanged.connect(self._validate)
        form.addRow("Contributor:", self.contributor_edit)
        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText("optional, e.g. 'SI of Smith 2025, as deposited'")
        form.addRow("Note:", self.note_edit)

        self.licence_box = QComboBox()
        for label, spdx in LICENCES:
            self.licence_box.addItem(label, spdx)
        self.licence_box.currentIndexChanged.connect(self._validate)
        form.addRow("Structure licence:", self.licence_box)
        self.basis_edit = QLineEdit()
        self.basis_edit.setPlaceholderText("why these files may be shared, e.g. 'our own "
                                           "structures'")
        self.basis_edit.textChanged.connect(self._validate)
        form.addRow("Because:", self.basis_edit)
        layout.addLayout(form)

        notes = QLabel(
            "Results are always added. A structure file is copied into the index only "
            "when its licence allows it: CoRE MOF files from Fetch Reference Structures "
            "carry CC-BY-4.0; a file with the CSD's header is never copied, whatever is "
            "chosen here."
        )
        notes.setWordWrap(True)
        notes.setStyleSheet("color: gray;")
        layout.addWidget(notes)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                        | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._validate()

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Shared Index Folder",
                                                  self.path_edit.text())
        if folder:
            self.path_edit.setText(folder)

    def _new(self) -> None:
        folder = create_index(self, self.path_edit.text())
        if folder:
            self.path_edit.setText(folder)

    def _validate(self, *_) -> None:
        declared = self.licence_box.currentData() is not None
        self.basis_edit.setEnabled(declared)
        ok = (is_index(self.path_edit.text().strip())
              and bool(self.contributor_edit.text().strip())
              and (not declared or bool(self.basis_edit.text().strip())))
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(ok)

    def values(self) -> dict:
        spdx = self.licence_box.currentData()
        return {
            "index": self.path_edit.text().strip(),
            "contributor": self.contributor_edit.text().strip(),
            "note": self.note_edit.text().strip(),
            "declared_licence": ({"spdx": spdx, "basis": self.basis_edit.text().strip()}
                                 if spdx else None),
        }


def describe_add(result) -> str:
    """The message shown after an add."""
    lines = [f"{len(result.added)} added, {len(result.present)} already in the index, "
             f"{len(result.skipped)} skipped."]
    if result.stored:
        lines.append(f"{len(result.stored)} structure file(s) copied into the index.")
    if result.withheld:
        lines.append(f"{len(result.withheld)} structure file(s) not copied (results only):")
        lines += [f"  {name}: {basis}" for name, basis in result.withheld[:10]]
    if result.skipped:
        lines.append("Skipped:")
        lines += [f"  {name}: {reason}" for name, reason in result.skipped[:10]]
    return "\n".join(lines)
