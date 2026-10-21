"""Help > About, and Help > Methods & References."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QLabel, QTextBrowser, QVBoxLayout

from html import escape

from thevoid_core import citation
from thevoid_core.version import __version__

#: The literature each analysis rests on - what to cite alongside this
#: program when its numbers appear in a paper.
REFERENCES = [
    ("Structure handling, CrystalNN, XRD", "S. P. Ong et al., Comput. Mater. Sci. 68, 314 (2013) - pymatgen; "
     "H. Pan et al., Inorg. Chem. 60, 1590 (2021) - CrystalNN"),
    ("Pore limiting diameter (Voronoi network)", "T. F. Willems, C. H. Rycroft, M. Kazi, J. C. Meza, M. Haranczyk, "
     "Microporous Mesoporous Mater. 149, 134 (2012) - Zeo++"),
    ("Accessible surface area", "T. Düren, F. Millange, G. Férey, K. S. Walton, R. Q. Snurr, "
     "J. Phys. Chem. C 111, 15350 (2007)"),
    ("Pore size distribution", "L. D. Gelb, K. E. Gubbins, Langmuir 15, 305 (1999)"),
    ("Van der Waals radii", "A. Bondi, J. Phys. Chem. 68, 441 (1964); S. S. Batsanov, Inorg. Mater. 37, 871 (2001)"),
    ("Covalent radii (bonding)", "B. Cordero et al., Dalton Trans. 2832 (2008)"),
    ("Fibonacci sphere sampling", "Á. González, Math. Geosci. 42, 49 (2010)"),
    ("2D skeletal diagrams", "RDKit: Open-source cheminformatics, https://www.rdkit.org"),
    ("Topology: reference nets", "M. O'Keeffe, M. A. Peskov, S. J. Ramsden, O. M. Yaghi, "
     "Acc. Chem. Res. 41, 1782 (2008) - RCSR"),
    ("Topology: net identification", "O. Delgado-Friedrichs, M. O'Keeffe, Acta Cryst. A59, "
     "351 (2003) - Systre's method"),
    ("RASPA2 export: simulation code", "D. Dubbeldam, S. Calero, D. E. Ellis, R. Q. Snurr, "
     "Mol. Simul. 42, 81 (2016) - RASPA"),
    ("RASPA2 export: framework force field", "A. K. Rappé, C. J. Casewit, K. S. Colwell, "
     "W. A. Goddard III, W. M. Skiff, J. Am. Chem. Soc. 114, 10024 (1992) - UFF"),
    ("RASPA2 export: adsorbates", "J. J. Potoff, J. I. Siepmann, AIChE J. 47, 1676 (2001) - "
     "TraPPE CO2, N2; M. G. Martin, J. I. Siepmann, J. Phys. Chem. B 102, 2569 (1998) - TraPPE CH4"),
    ("RASPA2 export: charge equilibration (opt-in)", "C. E. Wilmer, K. C. Kim, R. Q. Snurr, "
     "J. Phys. Chem. Lett. 3, 2506 (2012) - EQeq, RASPA's default method"),
    ("Reference data: CoRE MOF 2024 (bundled index, CC-BY-4.0)", "G. Zhao, L. Brabson, "
     "S. Chheda, J. Huang, H. Kim, K. Liu et al., Matter 8, 102140 (2025)"),
    ("Reference data: MOFX-DB", "N. S. Bobbitt et al., J. Chem. Eng. Data 68, 483 (2023); "
     "hMOF: C. E. Wilmer et al., Nat. Chem. 4, 83 (2012)"),
    ("CoRE comparison radii (CCDC set)", "A. Bondi (1964); R. S. Rowland, R. Taylor, "
     "J. Phys. Chem. 100, 7384 (1996) - H"),
]


def versions_text() -> str:
    from thevoid_core.services.report import provenance

    software = provenance()["software"]
    try:
        from PySide6 import __version__ as qt_version
    except Exception:
        qt_version = "unknown"
    parts = [f"the Void {software['the_void']}", f"Python {software['python']}",
             f"PySide6 {qt_version}", f"numpy {software['numpy']}",
             f"scipy {software['scipy']}", f"pymatgen {software['pymatgen']}"]
    return ", ".join(parts)


def cite_html() -> str:
    """Round 75: how to cite the Void itself, and its licence."""
    return (
        "<h3>How to cite the Void</h3>"
        "<p>If the Void, or numbers it produced, appear in a publication, "
        "please cite it:</p>"
        f"<p style='margin-left:12px'>{escape(citation.citation_text())}</p>"
        "<p>and the methods the numbers rest on (below).</p>"
        "<p><b>Licence:</b> "
        f"<a href='{citation.LICENCE_URL}'>{escape(citation.LICENCE_NAME)}</a>. "
        f"{escape(citation.LICENCE_SUMMARY)} Bundled third-party data keep their "
        "own licences (NOTICE.md).</p>"
    )


def methods_html() -> str:
    from thevoid_core.services.report import METHOD_NOTES

    rows = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in METHOD_NOTES.items())
    refs = "".join(f"<li><b>{topic}</b>: {cite}</li>" for topic, cite in REFERENCES)
    return (
        cite_html()
        + "<h3>Methods</h3><ul>" + rows + "</ul>"
        "<h3>References</h3><p>If results from this program appear in a "
        "publication, please also cite the methods they rest on:</p><ul>" + refs + "</ul>"
    )


class AboutDialog(QDialog):
    def __init__(self, parent=None, *, log_path=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("About the Void")
        self.resize(620, 520)

        header = QLabel(
            f"<h2>the Void {__version__}</h2>"
            "<p>Crystallography and porous-materials software: "
            "3D viewing, measurement, pore and surface analysis, open metal "
            "sites, powder XRD.</p>"
        )
        header.setWordWrap(True)

        self.versions_label = QLabel(versions_text())
        self.versions_label.setWordWrap(True)
        self.versions_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.methods = QTextBrowser()
        self.methods.setOpenExternalLinks(True)
        self.methods.setHtml(methods_html())

        layout = QVBoxLayout(self)
        layout.addWidget(header)
        layout.addWidget(self.versions_label)
        if log_path:
            log_label = QLabel(f"Log file: {log_path}")
            log_label.setWordWrap(True)
            log_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(log_label)
        layout.addWidget(self.methods, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
