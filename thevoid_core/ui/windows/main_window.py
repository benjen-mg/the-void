from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    Qt,
    QThreadPool,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QActionGroup, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtWidgets import QTabWidget

from thevoid_core.data.metal_classification import is_metal
from thevoid_core.infrastructure.settings import Settings
from thevoid_core.services.bonds import statistics
from thevoid_core.services.bonds.bond_model import site_occupancy
from thevoid_core.services.bonds.covalent_bond_detector import (
    CovalentBondDetector,
)
from thevoid_core.services.coordination import analyze_coordination_environment
from thevoid_core.services.cif.pymatgen_cif_loader import PymatgenCIFLoader
from thevoid_core.services.measurement import (
    MeasurementResult,
    nearest_periodic_image,
)
from thevoid_core.services.mof_analysis import (
    build_periodic_graph_for_mof_analysis,
    partition_framework,
    partition_structure,
)
from thevoid_core.services.guest_accessibility import (
    analyze_guest_accessibility,
)
from thevoid_core.services.oms_detection import detect_open_metal_sites
from thevoid_core.services.skeletal_diagram_render import (
    render_skeletal_diagram,
)
from thevoid_core.services.skeletal_layout import compute_skeletal_diagram
from thevoid_core.services.export import (
    FORMATS_REQUIRING_ORDERED_STRUCTURE,
    export_cif,
    export_cp2k,
    export_gaussian,
    export_poscar,
    export_quantum_espresso,
    export_xyz,
    ordered_structure_for_export,
)
from thevoid_core.services.pore_analysis import analyze_pores
from thevoid_core.services.surface_area import analyze_surface_area
from thevoid_core.services.site_labels import compute_site_labels
from thevoid_core.services.structure.render_model import (
    CompletionMode,
    RenderModelBuilder,
)
from thevoid_core.services.xrd.powder_xrd_simulator import (
    DEFAULT_TWO_THETA_RANGE,
    DEFAULT_WAVELENGTH,
    PowderXRDSimulator,
)
from thevoid_core.ui.dialogs.bonding_tolerance_dialog import (
    BondingToleranceDialog,
)
from thevoid_core.ui.dialogs.about_dialog import AboutDialog
from thevoid_core.ui.dialogs.psd_dialog import PoreSizeDistributionDialog
from thevoid_core.ui.dialogs.supercell_dialog import SupercellDialog
from thevoid_core.ui.dialogs.xrd_dialog import XRDDialog
from thevoid_core.ui.dialogs.xrd_setup_dialog import XRDSetupDialog
from thevoid_core.ui.widgets.coordination_panel import (
    CoordinationEnvironmentPanel,
)
from thevoid_core.ui.widgets.measurements_panel import MeasurementsPanel
from thevoid_core.ui.widgets.mof_analysis_panel import MOFAnalysisPanel
from thevoid_core.ui.widgets.oms_panel import OMSPanel
from thevoid_core.ui.widgets.color_customization_dialog import (
    ColorCustomizationDialog,
)
from thevoid_core.ui.widgets.element_legend import ElementLegend, element_counts
from thevoid_core.ui.widgets.skeletal_diagram_dialog import (
    SkeletalDiagramDialog,
)
from thevoid_core.ui.widgets.pore_analysis_panel import PoreAnalysisPanel
from thevoid_core.ui.widgets.guest_accessibility_panel import (
    GuestAccessibilityPanel,
)
from thevoid_core.ui.widgets.selected_atom_panel import SelectedAtomPanel
from thevoid_core.ui.widgets.surface_area_panel import SurfaceAreaPanel
from thevoid_core.ui.widgets.structure_info_panel import StructureInfoPanel
from thevoid_core.ui.widgets.structure_tree import StructureTree
from thevoid_core.ui.widgets.viewer_widget import (
    CrystalViewer,
    DisplayStyle,
    tile_points_for_supercell,
)

logger = logging.getLogger(__name__)


class _WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _Worker(QRunnable):
    """Run a plain callable off the GUI thread.

    Bond detection and the XRD calculation both take seconds to minutes
    on a large cell. The callable must not touch Qt widgets; results come
    back through finished, which is delivered on the GUI thread.
    """

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()

        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as exc:
            logger.error("Background task failed", exc_info=True)
            self.signals.failed.emit(str(exc))
        else:
            self.signals.finished.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.loader = PymatgenCIFLoader()
        self.xrd_simulator = PowderXRDSimulator()
        self.render_builder = RenderModelBuilder()

        # Covalent-radius bonding is the only detector offered: a fixed
        # geometric rule that handles the molecular and coordination
        # compounds this application targets correctly and predictably.
        # See services/bonds/__init__.py for why a second option
        # (CrystalNN) was removed rather than left selectable.
        self.bond_detector = CovalentBondDetector()

        self.current_info = None
        self.current_structure = None
        self.current_bonds: list = []
        self.current_site_labels: list[str] = []
        self.current_render_model = None
        self.current_selected_site_index: int | None = None
        self._mof_site_images: list = []

        #: Cached so _on_oms_site_selected can look an OMSSite back up
        #: from the panel table's row index without re-running
        #: detection.
        self.current_oms_result = None

        #: Every point-cloud analysis's most recent result, cached so
        #: its 3D overlay can be re-tiled and redrawn after a rebuild
        #: (a supercell change, a display style change, ...) without
        #: needing to re-run the analysis - render_model() resets every
        #: overlay on every redraw, by design (see its own docstring),
        #: so something has to be responsible for putting the current
        #: one back afterwards. current_analysis_visualization names
        #: which of these (if any) is the one currently on screen -
        #: "pore" / "surface_area" / "guest_accessibility" / "oms" /
        #: None - since only one of them is ever shown at a time.
        self.current_pore_result = None
        self.current_surface_area_result = None
        self.current_guest_accessibility_result = None
        self.current_analysis_visualization: str | None = None

        #: Round 51: kept for File > Export Analysis Report..., which
        #: gathers every result computed for the current structure.
        self.current_psd_result = None
        self.current_xrd_result = None

        #: Round 52: the record of the last RASPA export of this
        #: structure (raspa_export.summary_for_report), for the report.
        self.current_raspa_export_summary = None

        #: Round 53: the underlying net of the current structure, worked
        #: out in the background after every load (and bond re-detection,
        #: which can change it). The token discards a result that arrives
        #: after a newer structure has been loaded.
        self.current_topology_result = None
        self._topology_token = 0
        self._topology_worker = None
        #: Round 56: the last CoRE MOF 2024 comparison of this structure,
        #: the candidates it chose from, and the window showing them.
        self.current_reference_comparison = None
        self.current_reference_matches = []
        self._reference_dialog = None
        self._fetch_dialog = None
        self.current_predictions = None
        self._prediction_dialog = None
        # Round 66: the shared index (File > Shared Index).
        self._index_token = 0
        self.current_index_matches = None
        #: Remembered so a second export starts from the last choices.
        self.raspa_settings = None
        self._raspa_target_directory = None

        self.current_diagnostics = None

        self.completion_mode = CompletionMode.COMPLETE_MOLECULES
        self.show_hydrogens = True

        # 0.0 draws every site; 0.5 drops the minor components of a
        # disorder group. Sites at exactly 0.5 are kept, because with a
        # 50/50 split neither alternative is the minor one.
        self.min_occupancy = 0.0

        # (1, 1, 1) is a single unit cell.
        self.supercell = (1, 1, 1)

        # Off by default: molecule completion already gives a
        # chemically complete view; this only decorates the box edges.
        self.show_boundary_replicas = False

        # Remembered so re-running XRD starts from what was used last,
        # rather than resetting to the defaults every time.
        self.xrd_two_theta_range = DEFAULT_TWO_THETA_RANGE
        self.xrd_wavelength_key = DEFAULT_WAVELENGTH

        # None, "distance", or "angle" - which measurement tool (if any)
        # is currently active in the 3D view.
        self.measurement_mode = None

        self._pool = QThreadPool.globalInstance()
        self._worker: _Worker | None = None
        self._busy = False

        self.setWindowTitle("the Void")
        self.setAcceptDrops(True)

        icon_path = (
            Path(__file__).resolve().parent.parent.parent
            / "assets"
            / "icon.png"
        )
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        self.resize(1600, 1000)

        self._create_menu()
        self._create_viewer()
        self._create_docks()
        self._create_status_bar()

        self._load_recent_files()
        self._rebuild_recent_files_menu()
        self._restore_layout()

        self.statusBar().showMessage(
            "Ready — open a structure to begin (File > Open Structure..."
            ", Ctrl+O, or drag a file onto the window)."
        )

    @staticmethod
    def _settings() -> QSettings:
        """A QSettings instance that works whether or not the running
        QApplication has had setOrganizationName/setApplicationName
        called on it (the real application does, via
        TheVoidApplication.run(); a test that constructs
        MainWindow directly generally does not) - passing both
        explicitly here means recent-files persistence works either
        way, rather than silently landing in a different, anonymous
        settings location under test.
        """
        return QSettings(Settings.ORGANIZATION, Settings.APPLICATION)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _create_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        self.open_action = file_menu.addAction("Open Structure...")
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setToolTip(
            "Open a CIF, VASP POSCAR/CONTCAR, or pymatgen JSON structure "
            "file. You can also just drag a file onto the window."
        )
        self.open_action.triggered.connect(self.open_structure)

        self._recent_files_menu = file_menu.addMenu("Open Recent")
        self._recent_files_menu.setToolTip(
            "Structure files opened recently - click one to reopen it."
        )
        # Populated by _rebuild_recent_files_menu() once __init__ has
        # loaded the saved list; left empty here since nothing has been
        # read from settings yet at this point in construction.

        # Round 54: rank and filter a the-void-analyze results folder.
        self.open_batch_action = file_menu.addAction("Open Batch Results...")
        self.open_batch_action.setShortcut("Ctrl+Shift+O")
        self.open_batch_action.setToolTip(
            "Open a folder written by the-void-analyze: sort, filter and "
            "rank its summary, read each structure's report, and open any "
            "row's structure here in 3D."
        )
        self.open_batch_action.triggered.connect(lambda: self.open_batch_results())

        # Round 56: reference structures from CoRE MOF 2024 / MOFX-DB.
        self.fetch_reference_action = file_menu.addAction("Fetch Reference Structures...")
        self.fetch_reference_action.setToolTip(
            "Download CoRE MOF 2024 (openly licensed subset) or MOFX-DB "
            "(hMOF and others) structures into a folder, with the "
            "database's values and citation, ready for the-void-analyze."
        )
        self.fetch_reference_action.triggered.connect(self.fetch_reference_structures)

        # Round 66: the group's shared index (the-void-index).
        index_menu = file_menu.addMenu("Shared Index")
        self.shared_index_menu = index_menu
        self.index_add_action = index_menu.addAction("Add Current Structure...")
        self.index_add_action.setToolTip(
            "Add the analyses run on this structure so far to the shared index, "
            "with who ran them and how. The structure file is copied in only if "
            "its licence allows (never a CSD-derived file).")
        self.index_add_action.triggered.connect(lambda: self.add_current_to_index())
        self.index_open_action = index_menu.addAction("Open Index...")
        self.index_open_action.setToolTip(
            "Build the shared index and open it in the Batch Results window: every "
            "analysis the group has added, with structure and determination ids.")
        self.index_open_action.triggered.connect(lambda: self.open_shared_index())
        index_menu.addSeparator()
        self.index_set_action = index_menu.addAction("Set Index Folder...")
        self.index_set_action.setToolTip(
            "Choose the group's index folder (or make a new one). Every structure "
            "you open is then looked up in it: see Previously Analysed in the "
            "Structure panel.")
        self.index_set_action.triggered.connect(lambda: self.choose_shared_index())

        file_menu.addSeparator()

        self.export_view_action = file_menu.addAction("Export View as PNG...")
        self.export_view_action.setShortcut("Ctrl+Shift+S")
        self.export_view_action.setToolTip(
            "Save exactly what the 3D view currently shows as a PNG "
            "image, at the viewport's own resolution."
        )
        self.export_view_action.triggered.connect(self.export_view)

        export_structure_menu = file_menu.addMenu("Export Structure")
        export_structure_menu.setToolTip(
            "Write the currently loaded unit cell out in another "
            "format - always the crystallographic cell itself, not "
            "whatever supercell or display completion the 3D view "
            "happens to be showing."
        )

        self.export_poscar_action = export_structure_menu.addAction(
            "VASP POSCAR..."
        )
        self.export_poscar_action.setToolTip(
            "VASP's structure format. Cannot store partial "
            "occupancies - you'll be offered an ordered approximation "
            "if the structure is disordered."
        )
        self.export_poscar_action.triggered.connect(
            lambda: self.export_structure_as("poscar")
        )

        self.export_cif_action = export_structure_menu.addAction("CIF...")
        self.export_cif_action.setToolTip(
            "Crystallographic Information File - preserves occupancies "
            "exactly, including any disorder, unlike every other export "
            "format here."
        )
        self.export_cif_action.triggered.connect(
            lambda: self.export_structure_as("cif")
        )

        self.export_xyz_action = export_structure_menu.addAction(
            "XYZ (extended)..."
        )
        self.export_xyz_action.setToolTip(
            "Plain-text atom list with Cartesian coordinates, plus the "
            "lattice and periodic-boundary flags in the comment line "
            "(the \"extended XYZ\" convention ASE, OVITO, and VMD all "
            "read)."
        )
        self.export_xyz_action.triggered.connect(
            lambda: self.export_structure_as("xyz")
        )

        self.export_gaussian_action = export_structure_menu.addAction(
            "Gaussian Input (.gjf)..."
        )
        self.export_gaussian_action.setToolTip(
            "A Gaussian .gjf input deck for this structure, ready to "
            "have its route section and job type filled in."
        )
        self.export_gaussian_action.triggered.connect(
            lambda: self.export_structure_as("gaussian")
        )

        self.export_cp2k_action = export_structure_menu.addAction(
            "CP2K Input..."
        )
        self.export_cp2k_action.setToolTip(
            "A CP2K input file with this structure's cell and "
            "coordinates filled in, ready for the rest of the "
            "calculation setup."
        )
        self.export_cp2k_action.triggered.connect(
            lambda: self.export_structure_as("cp2k")
        )

        self.export_qe_action = export_structure_menu.addAction(
            "Quantum ESPRESSO Input..."
        )
        self.export_qe_action.setToolTip(
            "A Quantum ESPRESSO pw.x input file with this structure's "
            "cell and atomic positions filled in."
        )
        self.export_qe_action.triggered.connect(
            lambda: self.export_structure_as("quantum_espresso")
        )

        export_structure_menu.addSeparator()

        # Round 52: unlike the formats above, this is a whole input set
        # for a simulation, with parameters of its own - so it asks for
        # them in a dialog and writes a folder rather than one file.
        # The same action also lives in its own Simulate menu.
        self.export_raspa_action = export_structure_menu.addAction(
            "RASPA2 GCMC Input Set..."
        )
        self.export_raspa_action.setToolTip(
            "A self-contained RASPA2 folder for a rigid-framework GCMC "
            "adsorption run: P1 framework CIF, UFF + TraPPE force field, "
            "adsorbate model, pocket blocking and simulation.input, with "
            "every assumption documented in a README."
        )
        self.export_raspa_action.triggered.connect(lambda: self.export_raspa_input_set())

        self.export_report_action = file_menu.addAction(
            "Export Analysis Report..."
        )
        self.export_report_action.setShortcut("Ctrl+E")
        self.export_report_action.setToolTip(
            "Save the structure summary and every analysis run on it so "
            "far (pore, surface area, guest, OMS, PSD, XRD) as JSON or "
            "Markdown, with the software versions, parameters, methods "
            "and a checksum of the input file - for a lab notebook or a "
            "paper's supporting information."
        )
        self.export_report_action.triggered.connect(lambda: self.export_analysis_report())

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)

        view_menu = menu_bar.addMenu("View")

        # Select/View cursor mode: toolbar only, not also a View menu
        # entry - the two were originally added as alternative routes
        # to the same toggle, but having both turned out to just be a
        # confusing duplicate of the same two buttons in two places.
        # The toolbar is the one kept: this is a control meant to be
        # flipped constantly while reviewing an analysis result, and a
        # toolbar button is faster to reach than a menu item.
        self.cursor_mode_group = QActionGroup(self)
        self.cursor_mode_group.setExclusive(True)

        self.select_mode_action = QAction("Select", self)
        self.select_mode_action.setCheckable(True)
        self.select_mode_action.setChecked(True)
        self.select_mode_action.setToolTip(
            "Clicking an atom in the 3D view selects it (or picks it "
            "for a measurement, if one is in progress). The default."
        )
        self.cursor_mode_group.addAction(self.select_mode_action)

        self.view_mode_action = QAction("View (Rotate Only)", self)
        self.view_mode_action.setCheckable(True)
        self.view_mode_action.setToolTip(
            "Clicking in the 3D view only rotates/pans/zooms - nothing "
            "gets selected or measured. Use this while looking around "
            "an analysis result (Pore Analysis, Surface Area, ...) "
            "without a click-and-drag being mistaken for a click to "
            "select."
        )
        self.cursor_mode_group.addAction(self.view_mode_action)

        self.cursor_mode_group.triggered.connect(
            self._on_cursor_mode_changed
        )

        cursor_toolbar = self.addToolBar("Cursor Mode")
        cursor_toolbar.setObjectName("cursor_mode_toolbar")
        cursor_toolbar.addAction(self.select_mode_action)
        cursor_toolbar.addAction(self.view_mode_action)

        self.complete_action = view_menu.addAction("Complete Molecules")
        self.complete_action.setCheckable(True)
        self.complete_action.setChecked(True)
        self.complete_action.setToolTip(
            "Draw the periodic images needed to make every molecule whole, "
            "instead of clipping them at the cell boundary."
        )
        self.complete_action.toggled.connect(self._on_completion_toggled)

        self.cell_action = view_menu.addAction("Show Unit Cell")
        self.cell_action.setCheckable(True)
        self.cell_action.setChecked(True)
        self.cell_action.setToolTip(
            "Draw the crystallographic unit cell as a wireframe box "
            "around the structure."
        )
        self.cell_action.toggled.connect(self._on_cell_toggled)

        self.hydrogen_action = view_menu.addAction("Show Hydrogens")
        self.hydrogen_action.setCheckable(True)
        self.hydrogen_action.setChecked(True)
        self.hydrogen_action.setToolTip(
            "Uncheck to hide hydrogen atoms - useful for seeing a "
            "framework's heavy-atom connectivity without the clutter "
            "of every C-H and O-H."
        )
        self.hydrogen_action.toggled.connect(self._on_hydrogens_toggled)

        self.disorder_action = view_menu.addAction(
            "Show Minor Disorder Components"
        )
        self.disorder_action.setCheckable(True)
        self.disorder_action.setChecked(True)
        self.disorder_action.setToolTip(
            "Uncheck to hide sites with occupancy below 0.5, which "
            "removes the duplicate atoms that disorder introduces."
        )
        self.disorder_action.toggled.connect(self._on_disorder_toggled)

        view_menu.addSeparator()

        style_menu = view_menu.addMenu("Display Style")
        style_menu.setToolTip("How atoms and bonds are drawn in the 3D view.")

        self.style_group = QActionGroup(self)
        self.style_group.setExclusive(True)

        style_tooltips = {
            DisplayStyle.BALL_AND_STICK: (
                "Atoms as small spheres connected by sticks - the "
                "clearest view of connectivity, and the default."
            ),
            DisplayStyle.SPACE_FILLING: (
                "Atoms as full van der Waals spheres, touching where "
                "they are in contact - shows how densely packed the "
                "structure really is, at the cost of hiding bonds "
                "behind the spheres."
            ),
            DisplayStyle.WIREFRAME: (
                "Bonds only, as thin lines with no atom spheres at all "
                "- the least visually cluttered option for a large or "
                "densely-bonded structure."
            ),
        }

        for label, style, default in (
            ("Ball-and-Stick", DisplayStyle.BALL_AND_STICK, True),
            ("Space-Filling", DisplayStyle.SPACE_FILLING, False),
            ("Wireframe", DisplayStyle.WIREFRAME, False),
        ):
            action = style_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(default)
            action.setData(style)
            action.setToolTip(style_tooltips[style])
            self.style_group.addAction(action)

        self.style_group.triggered.connect(self._on_display_style_changed)

        view_menu.addSeparator()

        supercell_menu = view_menu.addMenu("Supercell")
        supercell_menu.setToolTip(
            "Display multiple repeats of the unit cell side by side, to "
            "see the extended framework rather than just one cell. "
            "Display only - every analysis still runs on the single "
            "crystallographic cell."
        )

        self.supercell_group = QActionGroup(self)
        self.supercell_group.setExclusive(True)

        for label, dims, default in (
            ("1 \u00d7 1 \u00d7 1 (single cell)", (1, 1, 1), True),
            ("2 \u00d7 2 \u00d7 2", (2, 2, 2), False),
            ("3 \u00d7 3 \u00d7 3", (3, 3, 3), False),
        ):
            action = supercell_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(default)
            action.setData(dims)
            self.supercell_group.addAction(action)

        supercell_menu.addSeparator()

        self.supercell_custom_action = supercell_menu.addAction("Custom...")
        self.supercell_custom_action.setCheckable(True)
        self.supercell_custom_action.setToolTip(
            "Choose a different repeat count along each of the three "
            "cell axes independently, rather than the equal-on-every-"
            "axis presets above."
        )
        # Sentinel: None means "open the dialog", distinguishing it from
        # the fixed presets above sharing the same action group.
        self.supercell_custom_action.setData(None)
        self.supercell_group.addAction(self.supercell_custom_action)

        self.supercell_group.triggered.connect(self._on_supercell_changed)

        self.boundary_action = view_menu.addAction("Show Cell-Edge Atom Copies")
        self.boundary_action.setCheckable(True)
        self.boundary_action.setChecked(False)
        self.boundary_action.setToolTip(
            "Draw extra copies of atoms sitting exactly on a cell face, "
            "so the displayed box looks complete at its edges and "
            "corners - the way a printed packing diagram does. "
            "Decorative only: these copies have no bonds of their own."
        )
        self.boundary_action.toggled.connect(self._on_boundary_replicas_toggled)

        view_menu.addSeparator()

        background_menu = view_menu.addMenu("Background")
        background_menu.setToolTip(
            "3D view background colour - Dark can make light-coloured "
            "elements easier to pick out."
        )

        self.background_group = QActionGroup(self)
        self.background_group.setExclusive(True)

        for label, color, default in (
            ("Light", CrystalViewer.BACKGROUND_LIGHT, True),
            ("Dark", CrystalViewer.BACKGROUND_DARK, False),
        ):
            action = background_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(default)
            action.setData(color)
            self.background_group.addAction(action)

        self.background_group.triggered.connect(self._on_background_changed)

        opacity_menu = view_menu.addMenu("Opacity")
        opacity_menu.setToolTip(
            "Make atoms and bonds partially transparent - useful for "
            "seeing an analysis overlay (pores, surface points, ...) "
            "through the framework rather than hidden behind it."
        )

        self.opacity_group = QActionGroup(self)
        self.opacity_group.setExclusive(True)

        for label, value, default in (
            ("100%", 1.0, True),
            ("75%", 0.75, False),
            ("50%", 0.5, False),
            ("25%", 0.25, False),
        ):
            action = opacity_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(default)
            action.setData(value)
            self.opacity_group.addAction(action)

        self.opacity_group.triggered.connect(self._on_opacity_changed)

        view_menu.addSeparator()

        reset_action = view_menu.addAction("Reset View")
        reset_action.setShortcut("Ctrl+R")
        reset_action.setToolTip(
            "Re-centre and re-zoom the camera to frame the whole "
            "structure, undoing any rotating/panning/zooming you've "
            "done - does not change any display setting."
        )
        reset_action.triggered.connect(self._reset_view)

        view_menu.addSeparator()

        self.skeletal_diagram_action = view_menu.addAction(
            "2D Skeletal Diagram..."
        )
        self.skeletal_diagram_action.setToolTip(
            "Generate a clean, idealised 2D line diagram of one unit "
            "cell's full nodes and linkers - not a literal projection "
            "(no single 2D view can preserve every real 3D angle and "
            "distance at once), but a layout that respects real bond "
            "lengths as closely as an idealised, non-overlapping 2D "
            "diagram can."
        )
        self.skeletal_diagram_action.triggered.connect(
            self._on_skeletal_diagram_menu_action
        )

        self.customize_colors_action = view_menu.addAction(
            "Customize Atom Colors..."
        )
        self.customize_colors_action.setToolTip(
            "Change the render colour of any element, or reset them "
            "back to the default Jmol colours."
        )
        self.customize_colors_action.triggered.connect(
            self._on_customize_colors_menu_action
        )

        analysis_menu = menu_bar.addMenu("Analysis")

        self.xrd_action = analysis_menu.addAction("Powder XRD...")
        self.xrd_action.setToolTip(
            "Simulate a powder X-ray diffraction pattern for the "
            "currently loaded structure, for a chosen radiation source "
            "and 2θ range."
        )
        self.xrd_action.triggered.connect(self.show_xrd)

        analysis_menu.addSeparator()

        self.bonding_tolerance_action = analysis_menu.addAction(
            "Bonding Tolerance..."
        )
        self.bonding_tolerance_action.setToolTip(
            "Adjust how much longer than the sum of covalent radii a "
            "contact can be and still count as a bond, then re-detect "
            "bonds with the new tolerance."
        )
        self.bonding_tolerance_action.triggered.connect(
            self._on_bonding_tolerance
        )

        analysis_menu.addSeparator()

        self.mof_analysis_action = analysis_menu.addAction(
            "Analyze MOF Structure"
        )
        self.mof_analysis_action.setToolTip(
            "Separate organic linkers from metal-containing nodes in "
            "the currently displayed structure."
        )
        self.mof_analysis_action.triggered.connect(self.analyze_mof_structure)

        self.coordination_analysis_action = analysis_menu.addAction(
            "Coordination Environment"
        )
        self.coordination_analysis_action.setToolTip(
            "Detect the coordination environment (neighbour list, bond "
            "lengths) of the currently selected atom using CrystalNN."
        )
        self.coordination_analysis_action.triggered.connect(
            self.analyze_coordination_environment
        )

        self.pore_analysis_action = analysis_menu.addAction(
            "Pore Analysis..."
        )
        self.pore_analysis_action.setToolTip(
            "Compute largest cavity diameter, pore limiting diameter, "
            "void fraction, and accessible/non-accessible volume for "
            "the currently loaded structure."
        )
        self.pore_analysis_action.triggered.connect(self._on_pore_menu_action)

        self.psd_action = analysis_menu.addAction("Pore Size Distribution...")
        self.psd_action.setToolTip(
            "Geometric pore size distribution: how the pore volume "
            "divides by the largest sphere that can reach each point, "
            "plotted with the cumulative curve and exportable as CSV."
        )
        self.psd_action.triggered.connect(self.compute_psd)

        self.surface_area_action = analysis_menu.addAction(
            "Surface Area Analysis..."
        )
        self.surface_area_action.setToolTip(
            "Compute accessible and geometric surface area via Monte "
            "Carlo surface sampling, for a chosen probe molecule."
        )
        self.surface_area_action.triggered.connect(
            self._on_surface_area_menu_action
        )

        self.guest_accessibility_action = analysis_menu.addAction(
            "Guest Accessibility Analysis..."
        )
        self.guest_accessibility_action.setToolTip(
            "Determine whether a guest molecule can diffuse through "
            "this framework - accessible, partially accessible, or "
            "inaccessible - and find the limiting pore size."
        )
        self.guest_accessibility_action.triggered.connect(
            self._on_guest_accessibility_menu_action
        )

        self.oms_action = analysis_menu.addAction(
            "Detect Open Metal Sites..."
        )
        self.oms_action.setToolTip(
            "Find undercoordinated, unsaturated metal centres - "
            "potential adsorption sites - among the framework's metal "
            "atoms."
        )
        self.oms_action.triggered.connect(self._on_oms_menu_action)

        analysis_menu.addSeparator()

        # Round 56.
        self.compare_coremof_action = analysis_menu.addAction(
            "Compare with CoRE MOF 2024..."
        )
        self.compare_coremof_action.setToolTip(
            "Find this structure in the bundled CoRE MOF 2024 index and set "
            "the database's LCD, PLD, surface areas, topology and open metal "
            "sites beside the Void's - computed the way CoRE MOF computes "
            "them (Zeo++'s radii, 1.655 A probe), so the comparison tests "
            "the method, not the convention."
        )
        self.compare_coremof_action.triggered.connect(lambda: self.compare_with_coremof())

        # Round 61.
        self.predict_action = analysis_menu.addAction("Predict Properties...")
        self.predict_action.setToolTip(
            "Estimate the water Henry coefficient and the thermal, solvent-removal "
            "and water stabilities with models trained on CoRE MOF 2024 "
            "(gradient boosting on pore, surface, chemistry and topology "
            "descriptors), each with a 90% interval and an applicability-domain "
            "flag. Model estimates, not measurements."
        )
        self.predict_action.triggered.connect(lambda: self.predict_properties())

        analysis_menu.addSeparator()

        self.clear_analysis_visualization_action = analysis_menu.addAction(
            "Clear Analysis Visualization"
        )
        self.clear_analysis_visualization_action.setToolTip(
            "Remove every analysis result currently overlaid on the 3D "
            "view (MOF/coordination/OMS highlights, pore points, "
            "surface area points and mesh, guest accessibility "
            "channels/blocked regions/bottlenecks) - manual measurements "
            "and the current atom selection are left alone."
        )
        self.clear_analysis_visualization_action.triggered.connect(
            self.clear_analysis_visualization
        )

        # Round 59: hypothetical variants of the loaded framework.
        build_menu = menu_bar.addMenu("Build")
        build_menu.setToolTipsVisible(True)
        self.functionalize_action = build_menu.addAction("Functionalise Linkers...")
        self.functionalize_action.setToolTip(
            "Make variants of this MOF with ring hydrogens of its linkers replaced by "
            "-NH2, -CH3, -F, -Br, -NO2 and other groups, each checked for close "
            "contacts and written as a CIF ready for the-void-analyze."
        )
        self.functionalize_action.triggered.connect(self.functionalize_linkers)

        # Round 52: bridges to simulation codes. The Void does geometry;
        # this is where a characterized structure leaves for a program
        # that does physics.
        simulate_menu = menu_bar.addMenu("Simulate")
        simulate_menu.setToolTipsVisible(True)
        self.simulate_raspa_action = simulate_menu.addAction(
            "Export RASPA2 GCMC Input Set..."
        )
        self.simulate_raspa_action.setToolTip(
            self.export_raspa_action.toolTip()
        )
        self.simulate_raspa_action.triggered.connect(
            lambda: self.export_raspa_input_set()
        )

        measure_menu = menu_bar.addMenu("Measure")

        self.measure_group = QActionGroup(self)
        self.measure_group.setExclusive(True)

        self.measure_off_action = None

        measure_tooltips = {
            None: (
                "Clicking in the 3D view selects atoms as usual - no "
                "measurement in progress."
            ),
            "distance": (
                "Click two atoms in turn to measure the distance "
                "between them, resolved across periodic boundaries so "
                "a measurement into the next cell gives the real "
                "distance."
            ),
            "angle": (
                "Click three atoms in turn (the middle click is the "
                "vertex) to measure the angle at the vertex."
            ),
            "torsion": (
                "Click four atoms A-B-C-D in turn to measure the signed "
                "torsion (dihedral) angle about the B-C bond, IUPAC sign "
                "convention (-180\u00b0 to 180\u00b0)."
            ),
        }

        for label, mode, default in (
            ("Off", None, True),
            ("Distance", "distance", False),
            ("Angle", "angle", False),
            ("Torsion", "torsion", False),
        ):
            action = measure_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(default)
            action.setData(mode)
            action.setToolTip(measure_tooltips[mode])
            self.measure_group.addAction(action)

            if mode is None:
                self.measure_off_action = action

        self.measure_group.triggered.connect(self._on_measure_mode_changed)

        measure_menu.addSeparator()

        clear_measurements_action = measure_menu.addAction(
            "Clear Measurements"
        )
        clear_measurements_action.setToolTip(
            "Remove every measurement - distances, angles, and "
            "torsions alike."
        )
        clear_measurements_action.triggered.connect(
            self._on_clear_measurements
        )

        self._create_help_menu(menu_bar)

    def _create_viewer(self) -> None:
        self.viewer = CrystalViewer()
        self.viewer.on_measurement = self._on_measurement_result
        self.viewer.on_measurement_failed = self._on_measurement_failed
        self.viewer.on_atom_selected = self._on_atom_selected_in_3d
        self.viewer.resolve_distance = self._resolve_nearest_image_distance

        # A first-run "nothing loaded yet" page, shown instead of a
        # blank/black 3D viewport until a structure is actually opened -
        # confirmed directly to be the more welcoming default: an empty
        # pyvista viewport gives no hint at all that anything needs to
        # happen before it becomes useful, whereas this page says
        # exactly what to do (three independent ways: menu, shortcut,
        # or drag-and-drop) and is swapped out for the real viewer the
        # moment a structure loads successfully.
        self._welcome_widget = self._build_welcome_widget()

        self._view_stack = QStackedWidget()
        self._view_stack.addWidget(self._welcome_widget)
        self._view_stack.addWidget(self.viewer)
        self._view_stack.setCurrentWidget(self._welcome_widget)

        self.setCentralWidget(self._view_stack)

    def _build_welcome_widget(self) -> QWidget:
        widget = QWidget()

        title = QLabel("No structure loaded")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        subtitle = QLabel(
            "Open a CIF, VASP POSCAR/CONTCAR, or pymatgen JSON file to "
            "look at it in 3D and run analyses on it.\n\n"
            "File > Open Structure..., Ctrl+O, or just drag a file onto "
            "this window."
        )
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: palette(mid);")

        open_button = QPushButton("Open Structure...")
        open_button.clicked.connect(self.open_structure)

        layout = QVBoxLayout(widget)
        layout.addStretch()
        layout.addWidget(title)
        layout.addSpacing(8)
        layout.addWidget(subtitle)
        layout.addSpacing(16)
        layout.addWidget(open_button, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch()

        return widget

    def _create_status_bar(self) -> None:
        """A permanent, indeterminate progress indicator in the status
        bar, shown only while a background analysis is running.

        Before this, the only feedback during a potentially slow
        operation (bond re-detection, pore/surface-area/guest-
        accessibility analysis, XRD simulation, ...) was the wait
        cursor and a status bar message - easy to miss, and giving no
        visual sense that the application is actually working rather
        than stuck. _set_busy shows/hides this alongside everything
        else it already does when a task starts and finishes.
        """
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # indeterminate: no known duration
        self._progress_bar.setMaximumWidth(160)
        self._progress_bar.setMaximumHeight(14)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._progress_bar)

    def _create_docks(self) -> None:
        """Three docks, not eight.

        Earlier versions gave every panel its own separate QDockWidget
        and tabified some of them together as an afterthought. Two
        real problems came from that: Qt's tabifyDockWidget() does not
        raise the newly grouped dock, so a panel that had just been
        tabified behind another one required a manual click - or, in
        at least one reported case, a window minimize/maximize - to
        even become visible; and eight separate title bars, borders,
        and drag handles is a lot of visual clutter for what are
        really three logical groups of panels.

        Consolidated here into Structure Tree (its own dock - it's a
        primary navigation panel, not secondary information), an
        Inspector dock on the right holding Structure Information /
        Selected Atom / Coordination Environment / Element Legend as
        tabs, and a Results dock on the bottom holding Measurements /
        MOF Structure as tabs. Whenever an action produces a result
        that belongs on one of these tabs, the handler now explicitly
        switches to it and raises the dock - see _show_inspector_tab
        and _show_results_tab - so the result is always immediately
        visible rather than something the person has to go hunting for
        behind another tab.
        """
        self.tree = StructureTree()
        self.tree.atom_selected.connect(self._on_atom_selected_in_tree)

        tree_dock = QDockWidget("Structure Tree", self)
        tree_dock.setObjectName("structure_tree_dock")  # for saveState
        tree_dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, tree_dock)

        # --- Inspector (right): per-structure and per-atom detail ---

        self.info_panel = StructureInfoPanel()
        self.selected_atom_panel = SelectedAtomPanel()
        self.coordination_panel = CoordinationEnvironmentPanel()
        self.pore_analysis_panel = PoreAnalysisPanel()
        self.pore_analysis_panel.run_requested.connect(
            self.analyze_pore_structure
        )
        self.pore_analysis_panel.visualization_toggled.connect(
            self.viewer.set_pore_visibility
        )
        self.surface_area_panel = SurfaceAreaPanel()
        self.surface_area_panel.run_requested.connect(
            self.analyze_surface_area_structure
        )
        self.surface_area_panel.points_visualization_toggled.connect(
            self.viewer.set_surface_points_visibility
        )
        self.surface_area_panel.mesh_visualization_toggled.connect(
            self._on_surface_mesh_toggled
        )
        self.guest_accessibility_panel = GuestAccessibilityPanel()
        self.guest_accessibility_panel.run_requested.connect(
            self.analyze_guest_accessibility_structure
        )
        self.guest_accessibility_panel.accessible_visualization_toggled.connect(
            self.viewer.set_guest_accessible_visibility
        )
        self.guest_accessibility_panel.blocked_visualization_toggled.connect(
            self.viewer.set_guest_blocked_visibility
        )
        self.guest_accessibility_panel.bottleneck_visualization_toggled.connect(
            self.viewer.set_guest_bottleneck_visibility
        )
        self.oms_panel = OMSPanel()
        self.oms_panel.run_requested.connect(self.detect_oms_structure)
        self.oms_panel.site_selected.connect(self._on_oms_site_selected)
        self.oms_panel.highlight_all_toggled.connect(
            self._on_oms_highlight_all_toggled
        )
        # Round 73: the element key sits at the top of the Structure tab,
        # four to a row, instead of on a Legend tab of its own that had
        # to be found first (and was scrolled off the tab bar).
        self.element_legend = ElementLegend(self.viewer.color_provider, columns=4,
                                            title=False)
        self.info_panel.set_element_legend(self.element_legend)

        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.addTab(self.info_panel, "Structure")
        self.inspector_tabs.addTab(self.selected_atom_panel, "Selected Atom")
        self.inspector_tabs.addTab(
            self.coordination_panel, "Coordination"
        )
        self.inspector_tabs.addTab(self.pore_analysis_panel, "Pores")
        self.inspector_tabs.addTab(self.surface_area_panel, "Surface Area")
        self.inspector_tabs.addTab(
            self.guest_accessibility_panel, "Guest Accessibility"
        )
        self.inspector_tabs.addTab(self.oms_panel, "Open Metal Sites")

        inspector_dock = QDockWidget("Inspector", self)
        inspector_dock.setObjectName("inspector_dock")
        inspector_dock.setWidget(self.inspector_tabs)
        self.addDockWidget(
            Qt.DockWidgetArea.RightDockWidgetArea, inspector_dock
        )
        self._inspector_dock = inspector_dock

        # --- Results (bottom): output of an explicit analysis action ---

        self.measurements_panel = MeasurementsPanel()

        self.mof_analysis_panel = MOFAnalysisPanel()
        self.mof_analysis_panel.highlight_requested.connect(
            self._on_mof_highlight_requested
        )

        self.results_tabs = QTabWidget()
        self.results_tabs.addTab(self.measurements_panel, "Measurements")
        self.results_tabs.addTab(self.mof_analysis_panel, "MOF Structure")

        results_dock = QDockWidget("Results", self)
        results_dock.setObjectName("results_dock")
        results_dock.setWidget(self.results_tabs)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, results_dock)
        self._results_dock = results_dock
        self._tree_dock = tree_dock

        self._apply_default_dock_sizes()

    #: Round 73: the Results dock opened about 100 px tall, one table row
    #: or none. It now starts at this share of the window's height, and
    #: is grown back to at least RESULTS_MIN_HEIGHT whenever a result is
    #: shown in it.
    RESULTS_HEIGHT_FRACTION = 0.3
    RESULTS_MIN_HEIGHT = 260
    #: Bumped when the dock layout changes, so an old saved layout is
    #: not restored onto a different set of docks.
    LAYOUT_VERSION = 1

    def _apply_default_dock_sizes(self) -> None:
        # Widths generous enough that the tree's columns and the
        # inspector's tab labels don't truncate on first launch - all
        # still freely resizable afterwards.
        self.resizeDocks(
            [self._tree_dock, self._inspector_dock],
            [300, 420],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._results_dock],
            [max(self.RESULTS_MIN_HEIGHT, int(self.height() * self.RESULTS_HEIGHT_FRACTION))],
            Qt.Orientation.Vertical,
        )

    def _ensure_results_room(self) -> None:
        """Grow the Results dock to RESULTS_MIN_HEIGHT if it has been
        left (or restored) shorter, so a result shown there has rows to
        show. A taller dock is left as it is."""
        dock = self._results_dock
        if dock.isFloating() or dock.height() >= self.RESULTS_MIN_HEIGHT:
            return
        self.resizeDocks([dock], [self.RESULTS_MIN_HEIGHT], Qt.Orientation.Vertical)

    def _save_layout(self) -> None:
        settings = self._settings()
        settings.setValue("layout/version", self.LAYOUT_VERSION)
        settings.setValue("layout/geometry", self.saveGeometry())
        settings.setValue("layout/state", self.saveState(self.LAYOUT_VERSION))

    def _restore_layout(self) -> None:
        """The window size and the dock sizes from the last session, if
        they were saved by this layout version. Every dock is shown
        afterwards: one closed by accident would otherwise stay gone."""
        settings = self._settings()
        try:
            version = int(settings.value("layout/version", 0) or 0)
        except (TypeError, ValueError):
            version = 0
        if version != self.LAYOUT_VERSION:
            return
        geometry = settings.value("layout/geometry")
        state = settings.value("layout/state")
        if geometry is not None:
            self.restoreGeometry(geometry)
        if state is not None:
            self.restoreState(state, self.LAYOUT_VERSION)
        for dock in (self._tree_dock, self._inspector_dock, self._results_dock):
            dock.show()

    def _show_inspector_tab(self, widget) -> None:
        """Make an Inspector tab the active, visible one - always
        called when that tab's content has just been updated, so a
        fresh result is never left sitting unseen behind another tab.
        """
        self._inspector_dock.show()
        self.inspector_tabs.setCurrentWidget(widget)
        self._inspector_dock.raise_()

    def _show_results_tab(self, widget) -> None:
        self._results_dock.show()
        self.results_tabs.setCurrentWidget(widget)
        self._results_dock.raise_()
        self._ensure_results_room()

    # ------------------------------------------------------------------
    # Busy handling
    # ------------------------------------------------------------------

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy

        self.open_action.setEnabled(not busy)
        self.xrd_action.setEnabled(not busy)
        self.mof_analysis_action.setEnabled(not busy)
        self.coordination_analysis_action.setEnabled(not busy)
        self.pore_analysis_action.setEnabled(not busy)
        self.surface_area_action.setEnabled(not busy)
        self.guest_accessibility_action.setEnabled(not busy)
        self.oms_action.setEnabled(not busy)
        self.compare_coremof_action.setEnabled(not busy)
        self.predict_action.setEnabled(not busy)
        self.index_add_action.setEnabled(not busy)
        self.index_open_action.setEnabled(not busy)
        self.skeletal_diagram_action.setEnabled(not busy)
        self.export_raspa_action.setEnabled(not busy)
        self.simulate_raspa_action.setEnabled(not busy)

        self._progress_bar.setVisible(busy)

        if busy:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        else:
            QApplication.restoreOverrideCursor()

        if message:
            self.statusBar().showMessage(message)

    def _run(self, fn, on_success, *args, **kwargs) -> None:
        worker = _Worker(fn, *args, **kwargs)

        worker.signals.finished.connect(on_success)
        worker.signals.failed.connect(self._on_task_failed)

        # Held so the signals QObject survives the run.
        self._worker = worker

        self._pool.start(worker)

    @Slot(str)
    def _on_task_failed(self, message: str) -> None:
        self._set_busy(False, "Task failed")
        if self._prediction_dialog is not None:
            self._prediction_dialog.set_busy(False)
        QMessageBox.critical(self, "Error", message)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def open_structure(self) -> None:
        if self._busy:
            return

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            "",
            "Structure Files (*.cif *.CIF *.json *.vasp POSCAR CONTCAR);;"
            "CIF Files (*.cif *.CIF);;All Files (*)",
        )

        if not filename:
            return

        self._open_file(filename)

    def _open_file(self, filename: str) -> None:
        """Shared by the Open Structure... dialog, a click on a File >
        Open Recent entry, and a file dropped onto the window - every
        route into loading a structure funnels through here so they
        stay in sync (the same busy-guard, the same status message, the
        same worker) rather than each reimplementing it slightly
        differently.
        """
        if self._busy:
            return

        self._set_busy(True, "Loading structure and detecting bonds...")
        self._run(self._load_and_analyse, self._on_structure_ready, filename)

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    #: How many recently-opened files to remember - enough to be useful
    #: (switching between a handful of structures under active
    #: comparison) without the menu turning into an unscannable list.
    _MAX_RECENT_FILES = 8

    def _load_recent_files(self) -> list[str]:
        try:
            stored = self._settings().value("recentFiles", [])
        except Exception:
            logger.debug(
                "Could not read recent files from settings; starting "
                "with an empty list.",
                exc_info=True,
            )
            stored = []

        # QSettings hands back a bare string instead of a list when
        # exactly one value was ever stored - a well-known Qt quirk,
        # not a corrupted setting - so it is normalised back to a list
        # here rather than surfacing as "Open Recent has one weirdly-
        # named entry per character" the first time this happens.
        if isinstance(stored, str):
            stored = [stored] if stored else []

        self._recent_files = [str(path) for path in stored]
        return self._recent_files

    def _save_recent_files(self) -> None:
        try:
            self._settings().setValue("recentFiles", self._recent_files)
        except Exception:
            logger.debug(
                "Could not persist recent files to settings.",
                exc_info=True,
            )

    def _add_recent_file(self, filename: str) -> None:
        path = str(Path(filename).resolve())

        self._recent_files = [p for p in self._recent_files if p != path]
        self._recent_files.insert(0, path)
        self._recent_files = self._recent_files[: self._MAX_RECENT_FILES]

        self._save_recent_files()
        self._rebuild_recent_files_menu()

    def _rebuild_recent_files_menu(self) -> None:
        menu = self._recent_files_menu
        menu.clear()

        if not self._recent_files:
            empty_action = menu.addAction("No recent files")
            empty_action.setEnabled(False)
            return

        for path in self._recent_files:
            action = menu.addAction(path)
            action.setToolTip(path)
            # A missing/moved file is a normal thing to have in this
            # list (the file itself is never touched to check, since
            # doing that for every entry every time the menu is
            # rebuilt would mean a filesystem stat per rename) - Open
            # already reports a clear error if it turns out to be
            # gone, the same as opening it by hand would.
            action.triggered.connect(
                lambda checked=False, p=path: self._open_file(p)
            )

        menu.addSeparator()

        clear_action = menu.addAction("Clear Recent Files")
        clear_action.triggered.connect(self._clear_recent_files)

    def _clear_recent_files(self) -> None:
        self._recent_files = []
        self._save_recent_files()
        self._rebuild_recent_files_menu()

    # ------------------------------------------------------------------
    # Drag and drop
    # ------------------------------------------------------------------

    #: Kept in sync with the Open Structure File dialog's own filter -
    #: a dropped file is only accepted if it looks like something this
    #: application can actually load.
    _DROPPABLE_SUFFIXES = {".cif", ".json", ".vasp"}
    _DROPPABLE_NAMES = {"poscar", "contcar"}

    def _is_droppable(self, filename: str) -> bool:
        path = Path(filename)
        return (
            path.suffix.lower() in self._DROPPABLE_SUFFIXES
            or path.name.lower() in self._DROPPABLE_NAMES
        )

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        local_files = [url.toLocalFile() for url in urls if url.isLocalFile()]

        if len(local_files) == 1 and self._is_droppable(local_files[0]):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        local_files = [url.toLocalFile() for url in urls if url.isLocalFile()]

        if len(local_files) == 1 and self._is_droppable(local_files[0]):
            event.acceptProposedAction()
            self._open_file(local_files[0])
        else:
            event.ignore()

    def _load_and_analyse(self, filename: str):
        """Worker thread. Must not touch any Qt widget."""
        info = self.loader.load(filename)

        bonds, diagnostics = self.bond_detector.detect_with_diagnostics(
            info.structure
        )

        return info, bonds, diagnostics

    def _redetect_bonds(self):
        """Worker thread. Re-run bond detection on the structure already
        loaded - used after the bonding tolerance changes, where the
        structure itself hasn't, only which contacts count as bonds.
        """
        bonds, diagnostics = self.bond_detector.detect_with_diagnostics(
            self.current_structure
        )

        return self.current_info, bonds, diagnostics

    def export_view(self) -> None:
        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before exporting the view.",
            )
            return

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export View as PNG",
            "",
            "PNG Files (*.png)",
        )

        if not filename:
            return

        filename = self._ensure_png_suffix(filename)

        try:
            self.viewer.export_image(filename)
        except (OSError, ValueError, RuntimeError) as error:
            QMessageBox.critical(
                self,
                "Export failed",
                f"The image could not be saved.\n\nFile: {filename}\n"
                f"Reason: {error}",
            )
        else:
            self.statusBar().showMessage(f"Saved {filename}", 5000)

    @staticmethod
    def _ensure_png_suffix(filename: str) -> str:
        """Append .png when the user omitted it.

        Appends rather than replacing any existing suffix, so a
        legitimate dot in the filename (e.g. 'structure.v2') isn't
        mistaken for an extension and overwritten.
        """
        path = Path(filename)

        if path.suffix.lower() == ".png":
            return str(path)

        return str(path.with_name(path.name + ".png"))

    #: One entry per supported quantum-chemistry export format:
    #: (export function, dialog title, suggested default filename,
    #: file-dialog filter string). POSCAR's suggested name is the bare
    #: word "POSCAR" - VASP's own convention is an extension-less
    #: filename, not "structure.POSCAR" - so it is deliberately not
    #: paired with an extension the way every other format here is.
    EXPORT_FORMATS = {
        "poscar": (
            export_poscar, "Export VASP POSCAR", "POSCAR",
            "POSCAR Files (*);;All Files (*)",
        ),
        "cif": (
            export_cif, "Export CIF", "structure.cif", "CIF Files (*.cif)",
        ),
        "xyz": (
            export_xyz, "Export Extended XYZ", "structure.xyz",
            "XYZ Files (*.xyz)",
        ),
        "gaussian": (
            export_gaussian, "Export Gaussian Input", "structure.gjf",
            "Gaussian Input Files (*.gjf)",
        ),
        "cp2k": (
            export_cp2k, "Export CP2K Input", "structure.inp",
            "CP2K Input Files (*.inp)",
        ),
        "quantum_espresso": (
            export_quantum_espresso, "Export Quantum ESPRESSO Input",
            "structure.in", "Quantum ESPRESSO Files (*.in)",
        ),
    }

    def export_structure_as(self, format_key: str) -> None:
        """Write the currently loaded structure out in one of the
        supported quantum-chemistry formats (File > Export Structure).

        Exports the base structure directly (current_structure) rather
        than whatever supercell or display completion the 3D view
        happens to be showing - a POSCAR or CIF is meant to describe
        the crystallographic unit cell, not an enlarged viewing copy of
        it, and every export function's whole job is preserving that
        cell, its periodicity, and its atomic positions exactly.
        """
        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before exporting.",
            )
            return

        export_function, dialog_title, default_name, file_filter = (
            self.EXPORT_FORMATS[format_key]
        )

        structure = self._structure_for_export(format_key)

        if structure is None:
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, dialog_title, default_name, file_filter,
        )

        if not filename:
            return

        try:
            content = export_function(structure)
            Path(filename).write_text(content)
        except Exception as error:  # noqa: BLE001 - see below
            # Deliberately broad, unlike the rest of this file's narrow
            # catches. Every analysis runs on a _Worker, whose run()
            # already funnels any exception into a failure dialog;
            # export is the one thing that runs synchronously on the GUI
            # thread, so an exception type nobody predicted propagates
            # out of a Qt slot instead of being reported. That is not
            # hypothetical: exporting a disordered structure to Quantum
            # ESPRESSO raised AttributeError, which the previous
            # (OSError, ValueError, RuntimeError) tuple did not cover.
            # The specific cause is fixed above, but third-party writers
            # get to raise whatever they like on input nobody has tried
            # yet, and a failed export should always be a dialog.
            logger.error("Structure export failed", exc_info=True)

            QMessageBox.critical(
                self,
                "Export failed",
                f"The structure could not be exported.\n\n"
                f"File: {filename}\nReason: {error}",
            )
        else:
            self.statusBar().showMessage(f"Saved {filename}", 5000)

    def _structure_for_export(self, format_key: str):
        """The structure to hand the writer, or None to abandon the
        export.

        Only POSCAR and Quantum ESPRESSO need anything done here, and
        only for a disordered structure: their file formats cannot
        express fractional occupancy at all. Rather than let the writer
        fail on it, the minor disorder components are dropped - the
        same major-occupancy filter the pore and surface-area analyses
        already use - but never silently, because that discards real
        atoms from the person's file. They are told exactly how many
        sites are involved and asked first, since an ordered
        approximation is a reasonable thing to want for a DFT
        calculation and also a reasonable thing to refuse.
        """
        structure = self.current_structure

        if (
            format_key not in FORMATS_REQUIRING_ORDERED_STRUCTURE
            or structure.is_ordered
        ):
            return structure

        ordered = ordered_structure_for_export(structure)

        if not ordered.is_ordered:
            # The 0.5 threshold keeps a site split evenly between two
            # species, so filtering is not guaranteed to produce an
            # ordered cell. Say so plainly instead of handing the
            # writer something it will still reject.
            QMessageBox.critical(
                self,
                "Cannot export a disordered structure",
                "This structure has sites shared between two or more "
                "species, which this format has no way to represent.\n\n"
                "Export to CIF instead, which records occupancies, or "
                "resolve the disorder before exporting.",
            )
            return None

        dropped = len(structure) - len(ordered)

        answer = QMessageBox.question(
            self,
            "Disordered structure",
            f"This format cannot store partial occupancies.\n\n"
            f"Export an ordered approximation instead, keeping the "
            f"major component of each site? {len(ordered)} of "
            f"{len(structure)} sites would be written and {dropped} "
            f"minor disorder component(s) dropped.\n\n"
            f"This is the same filter the pore and surface area "
            f"analyses apply. Export to CIF to keep the occupancies.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return None

        return ordered

    # ------------------------------------------------------------------
    # Batch results (round 54)
    # ------------------------------------------------------------------

    def open_batch_results(self, folder: str | None = None):
        """Show the Batch Results window (one per main window, reused),
        asking for a folder if none is given. Returns the window."""
        from thevoid_core.ui.windows.batch_results_window import (
            BatchResultsWindow,
        )

        if getattr(self, "batch_window", None) is None:
            self.batch_window = BatchResultsWindow(main_window=self)

        if folder is None:
            folder = QFileDialog.getExistingDirectory(
                self, "Open Batch Results Folder", ""
            )
            if not folder:
                return None

        if self.batch_window.load(folder):
            self.batch_window.show()
            self.batch_window.raise_()
        return self.batch_window

    # ------------------------------------------------------------------
    # Topology (round 53)
    # ------------------------------------------------------------------

    def _start_topology(self) -> None:
        """Work out the underlying net off the GUI thread. It does not
        use the busy state: it is a background detail of the Structure
        panel, and every other action stays available meanwhile."""
        self._topology_token += 1
        self.current_topology_result = None

        if self.current_structure is None:
            self.info_panel.set_topology(None)
            return

        self.info_panel.set_topology_pending()
        worker = _Worker(
            self._compute_topology,
            self.current_structure,
            self.current_bonds,
            self._topology_token,
        )
        worker.signals.finished.connect(self._on_topology_ready)
        worker.signals.failed.connect(self._on_topology_failed)
        self._topology_worker = worker
        self._pool.start(worker)

    def _compute_topology(self, structure, bonds, token):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.topology import analyze_topology

        return token, analyze_topology(
            structure, bonds, bond_detector=CovalentBondDetector(
                tolerance=self.bond_detector.tolerance
            ),
        )

    @Slot(object)
    def _on_topology_ready(self, payload) -> None:
        token, result = payload

        if token != self._topology_token:
            return  # a newer structure has been loaded since

        self.current_topology_result = result
        self.info_panel.set_topology(result)

    @Slot(str)
    def _on_topology_failed(self, message: str) -> None:
        # analyze_topology records failures in its result; anything that
        # still escapes is a bug, logged by _Worker. The panel says so
        # rather than showing an error dialog for a background detail.
        self.info_panel.topology_label.setText("not identified: analysis failed")
        self.info_panel.topology_label.setToolTip(message)

    # ------------------------------------------------------------------
    # RASPA2 export (round 52)
    # ------------------------------------------------------------------

    def export_raspa_input_set(self, directory: str | None = None) -> None:
        """Ask for simulation parameters and a folder, build the input
        set on the worker thread (pocket classification is a full pore-
        analysis grid), then write it.

        directory, when given, skips the folder dialog (tests).
        """
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before exporting a RASPA input set.",
            )
            return

        from thevoid_core.services.raspa_export import read_cif_charges
        from thevoid_core.ui.dialogs.raspa_export_dialog import (
            RaspaExportDialog,
        )

        source = self.current_info.file_path if self.current_info else None
        dialog = RaspaExportDialog(
            self,
            settings=self.raspa_settings,
            has_cif_charges=read_cif_charges(source) is not None,
        )

        if directory is None and dialog.exec() != QDialog.DialogCode.Accepted:
            return

        settings = dialog.settings()

        if directory is None:
            parent = QFileDialog.getExistingDirectory(
                self,
                "Where to create the RASPA2 input folder",
                str(Path(source).parent) if source else "",
            )

            if not parent:
                return

            # A folder of its own, so a dozen RASPA files never land
            # loose beside the person's CIFs.
            stem = Path(source).stem if source else "framework"
            directory = str(Path(parent) / f"{stem}_RASPA_{settings.adsorbate}")

        self.raspa_settings = settings
        self._raspa_target_directory = directory

        self._set_busy(True, "Building RASPA2 input set (finding pockets)...")
        self._run(
            self._build_raspa_export,
            self._on_raspa_export_ready,
            self.current_structure,
            self.current_bonds,
            settings,
            source,
        )

    def _build_raspa_export(self, structure, bonds, settings, source):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.raspa_export import build_raspa_export

        return build_raspa_export(
            structure,
            bonds,
            settings,
            source=source,
            bond_detector=self.bond_detector,
        )

    @Slot(object)
    def _on_raspa_export_ready(self, export) -> None:
        from thevoid_core.services.raspa_export import (
            existing_export_files,
            summary_for_report,
            write_raspa_export,
        )

        self._set_busy(False)
        directory = self._raspa_target_directory
        clashes = existing_export_files(export, directory)

        if clashes and not self._confirm_overwrite(directory, clashes):
            self.statusBar().showMessage("RASPA export cancelled.", 5000)
            return

        try:
            write_raspa_export(export, directory)
        except OSError as exc:
            QMessageBox.critical(
                self, "Export failed",
                f"The RASPA input set could not be written.\n\n{exc}",
            )
            return

        self.current_raspa_export_summary = summary_for_report(export, directory)
        self.statusBar().showMessage(
            f"RASPA2 input set written to {directory}", 8000
        )

        summary = export.summary
        blocking = summary["blocking"]
        lines = [
            f"Wrote {len(export.files)} files to:\n{directory}",
            "",
            f"{summary['adsorbate']['name']} at "
            f"{summary['settings']['temperature_K']} K; unit cells "
            f"{' x '.join(map(str, summary['unit_cells']))}; "
            f"charges: {summary['charges']['source']}.",
        ]

        if summary["framework"]["removed_molecules"]:
            removed = ", ".join(
                f"{m['count']} x {m['formula']}"
                for m in summary["framework"]["removed_molecules"]
            )
            lines.append(f"Removed as free molecules: {removed}.")

        if blocking is not None and blocking["status"] == "blocked":
            lines.append(f"{blocking['spheres']} pocket-blocking sphere(s).")

        if export.warnings:
            lines += ["", "Warnings:"] + [f"\u2022 {w}" for w in export.warnings]

        if summary["charges"]["source"] == "charge_equilibration":
            lines += ["", "Run run_raspa.sh: it computes the framework "
                      "charges first, then runs the GCMC."]

        lines += ["", "README.txt in the folder lists every assumption."]

        box = QMessageBox.warning if export.warnings else QMessageBox.information
        box(self, "RASPA2 Input Set Written", "\n".join(lines))

    def _confirm_overwrite(self, directory, clashes) -> bool:
        answer = QMessageBox.question(
            self,
            "Replace existing files?",
            f"{len(clashes)} file(s) already exist in {directory} "
            f"({', '.join(clashes[:5])}{'...' if len(clashes) > 5 else ''}). "
            "Replace them?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    @Slot(object)
    def _on_structure_ready(self, payload) -> None:
        info, bonds, diagnostics = payload

        try:
            self.current_info = info
            self.current_structure = info.structure
            self.current_bonds = bonds
            self.current_diagnostics = diagnostics
            self.current_site_labels = compute_site_labels(info.structure)

            self.tree.populate(info.structure)

            self.element_legend.update_elements(element_counts(info.structure))

            self._rebuild_view(reset_camera=True)

            # Swap the first-run welcome page out for the real 3D view,
            # the first time a structure loads successfully - a no-op
            # if it's already showing (every subsequent load, or a
            # bond-tolerance re-detection re-using this same handler).
            self._view_stack.setCurrentWidget(self.viewer)

            if info.file_path:
                self._add_recent_file(info.file_path)

            # A previous selection (a row highlighted, the panel showing
            # an atom's details) no longer refers to anything - the
            # tree was just repopulated from scratch and the viewer's
            # own selection state was already reset inside
            # _rebuild_view -> render_model.
            self.selected_atom_panel.clear()

            # Same reasoning for a previous MOF analysis: its atom
            # indices were only ever valid against the render model
            # that was current when it ran, which _rebuild_view above
            # has already replaced.
            self.mof_analysis_panel.clear()

            # And for a previous coordination environment analysis:
            # the selected site it was computed for no longer applies
            # to a freshly loaded structure.
            self.current_selected_site_index = None
            self.coordination_panel.clear()

            # And for a previous pore analysis: its accessible_points
            # were computed for a structure that no longer applies. The
            # viewer's own pore visualization is already cleared
            # automatically inside _rebuild_view -> render_model, same
            # as the selection highlight and group highlight above.
            self.pore_analysis_panel.clear()

            # And for a previous surface area analysis: same reasoning,
            # same automatic viewer-side reset inside render_model.
            self.surface_area_panel.clear()

            # And for a previous guest accessibility analysis: same
            # reasoning, same automatic viewer-side reset.
            self.guest_accessibility_panel.clear()

            # And for a previous OMS detection: its site indices no
            # longer apply to a freshly loaded structure.
            self.current_oms_result = None
            self.oms_panel.clear()

            # And for every point-cloud analysis's cached result and
            # the tracker that says which one to redisplay after a
            # rebuild - all invalid for a freshly loaded structure.
            self.current_pore_result = None
            self.current_surface_area_result = None
            self.current_guest_accessibility_result = None
            self.current_analysis_visualization = None
            self.current_psd_result = None
            self.current_xrd_result = None
            self.current_raspa_export_summary = None
            self.current_reference_comparison = None
            self.current_reference_matches = []
            if self._reference_dialog is not None:
                self._reference_dialog.hide()
            self.current_predictions = None
            if self._prediction_dialog is not None:
                self._prediction_dialog.hide()

            # A freshly loaded structure is the one moment it makes
            # sense to force the Inspector to a specific tab - the new
            # structure's own summary - rather than wherever the person
            # happened to leave it. Every other update (selecting an
            # atom, running an analysis) only raises its own tab, never
            # yanking the view away from what's already being read.
            self._show_inspector_tab(self.info_panel)

            self._start_topology()
            self._start_index_lookup()

            self._warn_about_disorder(diagnostics, info)
        except Exception as exc:
            logger.error("Failed to display structure", exc_info=True)

            self._set_busy(False, "Failed to display structure")
            QMessageBox.critical(self, "Display Error", str(exc))

    # ------------------------------------------------------------------
    # View assembly
    # ------------------------------------------------------------------

    def _rebuild_view(self, *, reset_camera: bool = False) -> None:
        """Rebuild the render model and redraw.

        Cheap enough to run on the GUI thread: it is a breadth-first walk
        over an already-computed bond graph, not a neighbour search.
        """
        if self.current_structure is None:
            return

        model = self.render_builder.build(
            self.current_structure,
            self.current_bonds,
            mode=self.completion_mode,
            show_hydrogens=self.show_hydrogens,
            min_occupancy=self.min_occupancy,
            supercell=self.supercell,
            show_boundary_replicas=self.show_boundary_replicas,
        )

        # Cached so other features (MOF analysis, coordination
        # environment) that need "the atoms and bonds currently drawn"
        # can use exactly this model rather than reaching into the
        # viewer's private state or rebuilding a second, possibly
        # inconsistent copy under different settings.
        self.current_render_model = model

        self.viewer.render_model(
            model,
            lattice_matrix=self.current_structure.lattice.matrix,
            reset_camera=reset_camera,
        )

        self.info_panel.set_structure(
            self.current_info,
            statistics(self.current_bonds),
            diagnostics=self.current_diagnostics,
            model=model,
        )

        self._set_busy(False, self._describe(model))

        # render_model() above just reset every analysis overlay (by
        # design - see clear_all_analysis_visualizations's docstring),
        # so whichever one was actually showing needs to be redrawn now,
        # re-tiled for whatever supercell this rebuild just applied -
        # otherwise a supercell change would silently wipe an in-
        # progress analysis's overlay instead of extending it to match.
        self._redisplay_current_analysis_visualization()

    def _redisplay_current_analysis_visualization(self) -> None:
        if self.current_analysis_visualization == "pore":
            if self.current_pore_result is not None:
                self._display_pore_visualization(self.current_pore_result)
        elif self.current_analysis_visualization == "surface_area":
            if self.current_surface_area_result is not None:
                self._display_surface_area_visualization(
                    self.current_surface_area_result
                )
        elif self.current_analysis_visualization == "guest_accessibility":
            if self.current_guest_accessibility_result is not None:
                self._display_guest_accessibility_visualization(
                    self.current_guest_accessibility_result
                )
        elif self.current_analysis_visualization == "oms":
            if self.current_oms_result is not None:
                self._highlight_all_oms(self.current_oms_result)

    def _describe(self, model) -> str:
        info = self.current_info

        formula = info.formula if info else "structure"

        nx, ny, nz = model.supercell

        if (nx, ny, nz) == (1, 1, 1):
            extra = f"{model.image_atom_count} from neighbouring cells"
        else:
            extra = (
                f"{nx}\u00d7{ny}\u00d7{nz} cells, "
                f"{model.outside_supercell_count} completing molecules "
                f"at the edge"
            )

        message = (
            f"{formula}: {len(model.atoms)} atoms drawn ({extra}), "
            f"{len(model.bonds)} bonds"
        )

        if model.boundary_replica_count:
            message += (
                f", {model.boundary_replica_count} decorative edge copies"
            )

        if model.fell_back:
            message += " - extended structure, showing one coordination shell"

        return message

    # ------------------------------------------------------------------
    # View actions
    # ------------------------------------------------------------------

    @Slot(bool)
    def _on_cursor_mode_changed(self, action) -> None:
        """Select/View toggle: only ever called by the exclusive
        cursor_mode_group, so 'action' is always exactly the one that
        just became checked.
        """
        want_selection = action is self.select_mode_action

        if not self.viewer.set_picking_enabled(want_selection):
            # Picking isn't available on this system at all (no
            # working VTK interactor) - View mode is trivially
            # satisfiable regardless (there is nothing to disable),
            # but Select mode genuinely can't be honoured. Revert the
            # menu/toolbar to View rather than leaving Select checked
            # on a mode that silently does nothing.
            self.view_mode_action.setChecked(True)
            QMessageBox.warning(
                self,
                "Selection unavailable",
                "Atom picking isn't available in this 3D view - your "
                "graphics driver or display setup may not support it.",
            )
            return

        self.statusBar().showMessage(
            "Select mode: click an atom to select it."
            if want_selection
            else "View mode: click and drag only rotates - nothing "
            "gets selected.",
            4000,
        )

    def _on_completion_toggled(self, checked: bool) -> None:
        self.completion_mode = (
            CompletionMode.COMPLETE_MOLECULES
            if checked
            else CompletionMode.CELL_ONLY
        )
        self._rebuild_view()

    @Slot(bool)
    def _on_cell_toggled(self, checked: bool) -> None:
        self.viewer.show_unit_cell = checked
        self._rebuild_view()

    @Slot(bool)
    def _on_hydrogens_toggled(self, checked: bool) -> None:
        self.show_hydrogens = checked
        self._rebuild_view()

    @Slot(object)
    def _on_supercell_changed(self, action) -> None:
        dims = action.data()

        if dims is None:
            self._prompt_custom_supercell(action)
            return

        self.supercell = dims
        self._rebuild_view()

    def _prompt_custom_supercell(self, custom_action) -> None:
        dialog = SupercellDialog(self.supercell, self)

        if dialog.exec() != SupercellDialog.DialogCode.Accepted:
            # Cancelled: put the checkmark back on whichever option
            # actually describes the current supercell, without
            # re-triggering _on_supercell_changed.
            self._sync_supercell_menu_selection()
            return

        self.supercell = dialog.dimensions()

        nx, ny, nz = self.supercell
        custom_action.setText(f"Custom ({nx} \u00d7 {ny} \u00d7 {nz})")

        self._rebuild_view()

    def _sync_supercell_menu_selection(self) -> None:
        """Check whichever Supercell menu entry matches self.supercell.

        If self.supercell doesn't match any fixed preset, it must be a
        previously-chosen custom value, so Custom stays checked rather
        than the group ending up with nothing checked at all.
        """
        matched_preset = False

        for action in self.supercell_group.actions():
            is_current = action.data() == self.supercell

            if is_current:
                matched_preset = True

            action.blockSignals(True)
            action.setChecked(is_current)
            action.blockSignals(False)

        if not matched_preset:
            self.supercell_custom_action.blockSignals(True)
            self.supercell_custom_action.setChecked(True)
            self.supercell_custom_action.blockSignals(False)

    @Slot(bool)
    def _on_disorder_toggled(self, checked: bool) -> None:
        self.min_occupancy = 0.0 if checked else 0.5
        self._rebuild_view()

    @Slot(object)
    def _on_display_style_changed(self, action) -> None:
        # DisplayStyle(...) rather than assigning action.data() directly:
        # Qt's QAction data round-trip silently unwraps a str-Enum back
        # to a plain str (confirmed directly), so without this
        # conversion self.viewer.display_style would hold the right
        # *value* but not a real DisplayStyle instance - harmless where
        # comparisons use ==, but worth pinning at the boundary rather
        # than depending on every future comparison getting that right.
        self.viewer.display_style = DisplayStyle(action.data())
        # Style is a pure redraw of the existing render model - no need
        # to recompute it - but _rebuild_view is cheap (a graph walk,
        # not a neighbour search) so reusing it is simpler than adding
        # a separate "redraw only" path.
        self._rebuild_view()

    @Slot(bool)
    def _on_boundary_replicas_toggled(self, checked: bool) -> None:
        self.show_boundary_replicas = checked
        self._rebuild_view()

    def _warn_about_disorder(self, diagnostics, info=None) -> None:
        """Say plainly when the file is disordered.

        A short contact that gets rejected is invisible otherwise, and
        the user is left wondering why two atoms sit on top of each
        other with nothing joining them.

        Rejected short contacts are always inferred from geometry: two
        sites too close together to both be real. When the CIF also
        carries an explicit _atom_site_disorder_group column, that is
        an independent, higher-confidence source for the same
        conclusion (the crystallographer who solved the structure said
        so directly, rather than us guessing from a distance cutoff),
        so the message says which situation applies.
        """
        if diagnostics is None or not diagnostics.short_contact_count:
            return

        shortest = diagnostics.shortest_rejected

        examples = "\n".join(
            f"  {c.symbol1}{c.atom1 + 1}-{c.symbol2}{c.atom2 + 1}: "
            f"{c.distance:.3f} A"
            for c in sorted(
                diagnostics.short_contacts, key=lambda c: c.distance
            )[:6]
        )

        if info is not None and info.has_explicit_disorder_groups:
            confirmation = (
                "The CIF file confirms this directly: it labels "
                f"{len(info.disorder_groups)} site(s) with explicit "
                "disorder group numbers.\n\n"
            )
        else:
            confirmation = (
                "This normally means the file is disordered: the same "
                "atom is modelled in two or more alternative positions, "
                "each with partial occupancy.\n\n"
            )

        QMessageBox.information(
            self,
            "Disorder detected",
            (
                f"{diagnostics.short_contact_count} pair(s) of sites are "
                f"too close to be bonded (shortest {shortest:.3f} A). "
                "They have not been drawn as bonds.\n\n"
                f"{confirmation}"
                f"{examples}\n\n"
                "Use View > Show Minor Disorder Components to hide the "
                "alternatives with occupancy below 0.5."
            ),
        )

    def _reset_view(self) -> None:
        self.viewer.reset_view()

    def _on_background_changed(self, action) -> None:
        self.viewer.set_background_color(action.data())

    @Slot(object)
    def _on_opacity_changed(self, action) -> None:
        self.viewer.set_opacity(action.data())
        self._rebuild_view()

    def _on_bonding_tolerance(self) -> None:
        dialog = BondingToleranceDialog(self.bond_detector.tolerance, self)

        if dialog.exec() != BondingToleranceDialog.DialogCode.Accepted:
            return

        new_tolerance = dialog.tolerance()

        if new_tolerance == self.bond_detector.tolerance:
            return

        self.bond_detector.tolerance = new_tolerance

        if self.current_structure is None:
            return

        self._set_busy(True, "Re-detecting bonds with new tolerance...")
        self._run(self._redetect_bonds, self._on_structure_ready)

    # ------------------------------------------------------------------
    # MOF analysis
    # ------------------------------------------------------------------

    def analyze_mof_structure(self) -> None:
        """Separate the currently loaded structure into organic linkers
        and metal-containing nodes.

        Builds its own periodicity-aware bond graph (since round 55
        mof_analysis.partition_structure, on the periodic bond graph
        itself) from the base structure
        and detected bonds, rather than reusing self.current_render_
        model. That matters for a real reason, not just consistency:
        the render model exists to draw a bounded, fast-to-render 3D
        view, and its "complete every molecule" walk has no way to
        terminate on a genuinely 3-periodic MOF - confirmed directly
        against a real Zn-triazolate-oxalate structure, which produced
        RenderModel.fell_back=True and linkers truncated wherever the
        resulting coordination-shell boundary happened to cut through
        a ring. The dedicated graph builder solves that by treating
        metal atoms as expansion boundaries, so organic linkers always
        complete fully while the walk still terminates.
        """
        if self._busy:
            return

        if self.current_structure is None or not self.current_bonds:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before running MOF analysis.",
            )
            return

        self._set_busy(True, "Analyzing linkers and metal nodes...")

        self._run(
            self._run_mof_partition,
            self._on_mof_partition_ready,
            self.current_structure,
            self.current_bonds,
        )

    def _run_mof_partition(self, structure, bonds):
        """Worker thread. Must not touch any Qt widget.

        Round 55: partitions the periodic bond graph of one unit cell
        (mof_analysis.partition_structure) instead of the expanded
        patch build_periodic_graph_for_mof_analysis builds, so each SBU
        and linker of the cell is listed once and whole - IRMOF-1 is 8
        Zn4O nodes and 24 BDC, not 8 Zn4O plus 48 boundary "Zn"
        fragments - paddlewheels are one Cu2 node, rod SBUs are flagged
        as rods, and minor disorder components are left out as in every
        other analysis. The skeletal diagram keeps its own expanded
        graph (it needs the metal neighbours of each linker).
        """
        return partition_structure(structure, bonds, is_metal)

    @Slot(object)
    def _on_mof_partition_ready(self, payload) -> None:
        partition, site_images = payload

        # Clear whatever a previous, different analysis left on screen
        # - MOF Analysis itself doesn't draw anything until a linker or
        # node row is clicked, but a stale Pore/Surface/Guest/OMS
        # overlay from an earlier analysis should not still be sitting
        # there once attention has moved on to a new one.
        self.viewer.clear_all_analysis_visualizations()
        # MOF's own highlight (drawn on row click, via highlight_group)
        # has no supercell-tiling redisplay support of its own yet - see
        # _highlight_all_oms's docstring for the same limitation - so
        # there is nothing here to re-tile after a rebuild; leaving the
        # tracker at None means a later supercell change simply leaves
        # the view clean rather than incorrectly restoring whatever
        # point-cloud overlay was showing before this analysis ran.
        self.current_analysis_visualization = None

        # Kept so a result row's atom_indices (positions in the
        # periodic-graph node list build_periodic_graph_for_mof_
        # analysis produced) can be translated into (site_index,
        # image) pairs, and from there into whichever render-model
        # atom actually represents that instance - see
        # _on_mof_highlight_requested.
        self._mof_site_images = site_images

        self.mof_analysis_panel.set_partition(partition)
        self._show_results_tab(self.mof_analysis_panel)

        self._set_busy(
            False,
            f"MOF analysis: {partition.linker_count} linker(s), "
            f"{partition.metal_node_count} metal node(s)",
        )

    @Slot(object, str)
    def _on_mof_highlight_requested(self, graph_node_indices, color: str) -> None:
        """graph_node_indices are positions in the site_images list
        partition_structure returned - not indices into
        the render model the 3D view currently shows, which is a
        different graph (built for display, with its own completion
        and supercell settings). Translate each one to a (site_index,
        image) pair via self._mof_site_images, then to whichever
        render-model atom actually represents that instance, if any.
        """
        site_images = [
            self._mof_site_images[index]
            for index in graph_node_indices
            if index < len(self._mof_site_images)
        ]

        atom_indices = self._render_atom_indices_for_site_images(site_images)

        self.viewer.highlight_group(atom_indices, color)

    # ------------------------------------------------------------------
    # Coordination environment analysis
    # ------------------------------------------------------------------

    #: Distinct from every other group-highlight colour already in use
    #: (linkers, metal nodes, atom selection, measurement lines), so a
    #: coordination sphere is never confused with any of them.
    COORDINATION_HIGHLIGHT_COLOR = "#cc33ff"

    def analyze_coordination_environment(self) -> None:
        """Detect the coordination environment of the currently
        selected atom (via the 3D view or the structure tree - either
        one sets current_selected_site_index) using CrystalNN.
        """
        if self._busy:
            return

        if (
            self.current_structure is None
            or self.current_selected_site_index is None
        ):
            QMessageBox.information(
                self,
                "No Atom Selected",
                "Select an atom - click it in the 3D view or in the "
                "structure tree - before analyzing its coordination "
                "environment.",
            )
            return

        self._set_busy(True, "Analyzing coordination environment...")

        self._run(
            self._run_coordination_analysis,
            self._on_coordination_environment_ready,
            self.current_structure,
            self.current_selected_site_index,
            self.current_site_labels,
        )

    def _run_coordination_analysis(self, structure, site_index, labels):
        """Worker thread. Must not touch any Qt widget. CrystalNN can
        take real time on a large or unusual structure, which is the
        whole reason this runs off the GUI thread rather than inline.
        """
        return analyze_coordination_environment(structure, site_index, labels)

    @Slot(object)
    def _on_coordination_environment_ready(self, environment) -> None:
        self.coordination_panel.set_environment(environment)
        self._show_inspector_tab(self.coordination_panel)

        self.viewer.clear_all_analysis_visualizations()
        # See _on_mof_partition_ready's comment: highlight_group-based
        # highlights have no supercell-tiling redisplay support yet.
        self.current_analysis_visualization = None

        site_indices = [environment.center_site_index] + [
            neighbor.site_index for neighbor in environment.neighbors
        ]
        atom_indices = self._render_atom_indices_for_sites(site_indices)

        self.viewer.highlight_group(
            atom_indices, self.COORDINATION_HIGHLIGHT_COLOR
        )

        self._set_busy(
            False,
            f"{environment.center_label} is "
            f"{environment.coordination_number}-coordinate "
            f"(avg. bond length {environment.average_bond_length:.3f} \u00c5)"
            if environment.coordination_number
            else f"{environment.center_label}: no coordination "
            "environment found",
        )

    # ------------------------------------------------------------------
    # Pore analysis
    # ------------------------------------------------------------------

    def _on_pore_menu_action(self) -> None:
        """Analysis > Pore Analysis...: just switches to the panel and
        runs with whatever probe radius is already set there (or the
        default, the first time), rather than opening a separate
        dialog for a single number the panel already asks for directly.
        """
        self._show_inspector_tab(self.pore_analysis_panel)
        self.analyze_pore_structure(self.pore_analysis_panel.probe_radius())

    def analyze_pore_structure(self, probe_radius: float) -> None:
        """Run pore analysis on the currently loaded structure.

        Connected to PoreAnalysisPanel.run_requested, which is what
        actually supplies probe_radius (the panel's own spin box) - this
        method does not read the panel itself, so it works the same way
        whether triggered from the panel's button or the menu action.
        """
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before running pore analysis.",
            )
            return

        self._set_busy(True, "Analyzing pore structure...")

        self._run(
            self._run_pore_analysis,
            self._on_pore_analysis_ready,
            self.current_structure,
            probe_radius,
            self.viewer.radius_provider,
        )

    def _run_pore_analysis(self, structure, probe_radius, radius_provider):
        """Worker thread. Must not touch any Qt widget. The grid-based
        distance-field and connectivity computation this runs can take
        real time on a fine grid or a large cell, which is the whole
        reason this happens off the GUI thread rather than inline.
        """
        return analyze_pores(
            structure,
            probe_radius=probe_radius,
            radius_provider=radius_provider,
        )

    @Slot(object)
    def _on_pore_analysis_ready(self, result) -> None:
        self.pore_analysis_panel.set_result(result)
        self._show_inspector_tab(self.pore_analysis_panel)

        self.viewer.clear_all_analysis_visualizations()

        self.current_pore_result = result
        self.current_analysis_visualization = "pore"
        self._display_pore_visualization(result)

        if result.pld is not None:
            message = (
                f"Pore analysis: LCD {result.lcd:.2f} \u00c5, PLD "
                f"{result.pld:.2f} \u00c5, void fraction "
                f"{result.void_fraction_percent:.1f}%"
            )
        else:
            message = (
                f"Pore analysis: LCD {result.lcd:.2f} \u00c5, no "
                "percolating pathway found at this probe radius"
            )

        self._set_busy(False, message)

    def _display_pore_visualization(self, result) -> None:
        """Draw a pore-analysis result's overlay, tiled to match
        whatever supercell is currently shown. Shared by the handler
        that runs right after analysis completes and by
        _redisplay_current_analysis_visualization(), which calls this
        same method again after a rebuild (a supercell change, a
        display style change, ...) puts the correct points back -
        render_model() clears every overlay on every redraw by design,
        so something has to restore whichever one was showing.
        """
        tiled_points = tile_points_for_supercell(
            result.accessible_points,
            self.current_structure.lattice.matrix,
            self.supercell,
        )
        self.viewer.show_pore_points(tiled_points)

        if not self.pore_analysis_panel.show_visualization_checkbox.isChecked():
            # Computed and cached on the viewer either way; just not
            # drawn until the checkbox is turned on. Toggling it
            # afterwards uses these same cached points rather than
            # needing to re-run the analysis.
            self.viewer.set_pore_visibility(False)

    # ------------------------------------------------------------------
    # Surface area analysis
    # ------------------------------------------------------------------

    def _on_surface_area_menu_action(self) -> None:
        """Analysis > Surface Area Analysis...: switches to the panel
        and runs with whatever probe is already selected there, the
        same "the panel is the dialog" pattern used for pore analysis.
        """
        self._show_inspector_tab(self.surface_area_panel)
        self.analyze_surface_area_structure(
            self.surface_area_panel.probe_radius(),
            self.surface_area_panel.probe_combo.currentText(),
        )

    def analyze_surface_area_structure(
        self, probe_radius: float, probe_name: str
    ) -> None:
        """Run accessible surface area analysis on the currently loaded
        structure. Connected to SurfaceAreaPanel.run_requested, which
        supplies both arguments from its own combo box / spin box, so
        this works identically whether triggered from the panel's
        button or the menu action.
        """
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before running surface area "
                "analysis.",
            )
            return

        self._set_busy(True, "Analyzing accessible surface area...")

        self._run(
            self._run_surface_area_analysis,
            self._on_surface_area_analysis_ready,
            self.current_structure,
            probe_radius,
            probe_name,
            self.viewer.radius_provider,
        )

    def _run_surface_area_analysis(
        self, structure, probe_radius, probe_name, radius_provider
    ):
        """Worker thread. Must not touch any Qt widget. Monte Carlo
        surface sampling over every atom (plus the internal/total
        percolation classification's own grid pass) is the whole
        reason this happens off the GUI thread.
        """
        return analyze_surface_area(
            structure,
            probe_radius=probe_radius,
            probe_name=probe_name,
            radius_provider=radius_provider,
        )

    @Slot(object)
    def _on_surface_area_analysis_ready(self, result) -> None:
        self.surface_area_panel.set_result(result)
        self._show_inspector_tab(self.surface_area_panel)

        self.viewer.clear_all_analysis_visualizations()

        self.current_surface_area_result = result
        self.current_analysis_visualization = "surface_area"
        self._display_surface_area_visualization(result)

        self._set_busy(
            False,
            f"Surface area ({result.probe_name}): ASA "
            f"{result.accessible_surface_area:.1f} \u00c5\u00b2 "
            f"({result.asa_per_mass:.1f} m\u00b2/g)",
        )

    def _display_surface_area_visualization(self, result) -> None:
        """Draw a surface-area result's overlay, tiled to match
        whatever supercell is currently shown - see
        _display_pore_visualization's docstring for why this needs to
        be its own reusable method rather than only running once, right
        after analysis completes.
        """
        nx, ny, nz = self.supercell
        n_images = nx * ny * nz

        tiled_points = tile_points_for_supercell(
            result.sampled_points,
            self.current_structure.lattice.matrix,
            self.supercell,
        )
        # accessible_mask is index-aligned with sampled_points, one
        # entry per original point - each point's n_images tiled copies
        # all need that same original point's accessible/blocked
        # classification, so the mask is repeated (not tiled the way
        # positions are), each value consecutively n_images times to
        # match tile_points_for_supercell's own point ordering (all of
        # one point's images together before the next point's).
        tiled_mask = np.repeat(result.accessible_mask, n_images)

        self.viewer.show_surface_points(tiled_points, tiled_mask)

        if not self.surface_area_panel.show_points_checkbox.isChecked():
            self.viewer.set_surface_points_visibility(False)

        if self.surface_area_panel.show_mesh_checkbox.isChecked():
            tiled_accessible_points = tiled_points[tiled_mask]
            self.viewer.show_surface_mesh(tiled_accessible_points)

    def _on_surface_mesh_toggled(self, visible: bool) -> None:
        """The mesh overlay is built lazily, on first request, from
        whichever accessible points the most recent analysis cached on
        the viewer - it is off by default (see SurfaceAreaPanel), so
        the first time it is turned on there is no existing mesh actor
        yet to simply make visible.
        """
        if visible and self.viewer._surface_mesh_actor is None:
            points = self.viewer._last_surface_points
            mask = self.viewer._last_surface_accessible_mask

            if points is not None and mask is not None and len(points):
                self.viewer.show_surface_mesh(points[mask])

            return

        self.viewer.set_surface_mesh_visibility(visible)

    # ------------------------------------------------------------------
    # Guest accessibility analysis
    # ------------------------------------------------------------------

    def _on_guest_accessibility_menu_action(self) -> None:
        """Analysis > Guest Accessibility Analysis...: switches to the
        panel and runs with whatever guest is already selected there,
        the same "the panel is the dialog" pattern used for pore and
        surface area analysis.
        """
        self._show_inspector_tab(self.guest_accessibility_panel)
        self.analyze_guest_accessibility_structure(
            self.guest_accessibility_panel.guest_diameter(),
            self.guest_accessibility_panel.guest_combo.currentText(),
        )

    def analyze_guest_accessibility_structure(
        self, guest_diameter: float, guest_name: str
    ) -> None:
        """Run guest accessibility analysis on the currently loaded
        structure. Connected to GuestAccessibilityPanel.run_requested,
        which supplies both arguments from its own combo box / spin
        box, so this works identically whether triggered from the
        panel's button or the menu action.
        """
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before running guest "
                "accessibility analysis.",
            )
            return

        self._set_busy(True, "Analyzing guest accessibility...")

        self._run(
            self._run_guest_accessibility_analysis,
            self._on_guest_accessibility_analysis_ready,
            self.current_structure,
            guest_diameter,
            guest_name,
            self.viewer.radius_provider,
        )

    def _run_guest_accessibility_analysis(
        self, structure, guest_diameter, guest_name, radius_provider
    ):
        """Worker thread. Must not touch any Qt widget."""
        return analyze_guest_accessibility(
            structure,
            guest_diameter=guest_diameter,
            guest_name=guest_name,
            radius_provider=radius_provider,
        )

    @Slot(object)
    def _on_guest_accessibility_analysis_ready(self, result) -> None:
        self.guest_accessibility_panel.set_result(result)
        self._show_inspector_tab(self.guest_accessibility_panel)

        self.viewer.clear_all_analysis_visualizations()

        self.current_guest_accessibility_result = result
        self.current_analysis_visualization = "guest_accessibility"
        self._display_guest_accessibility_visualization(result)

        status_text = result.status.value.replace("_", " ")
        self._set_busy(
            False,
            f"Guest accessibility ({result.guest_name}): {status_text}",
        )

    def _display_guest_accessibility_visualization(self, result) -> None:
        """Draw a guest-accessibility result's three overlays, tiled to
        match whatever supercell is currently shown - see
        _display_pore_visualization's docstring for why this needs to
        be its own reusable method rather than only running once, right
        after analysis completes.
        """
        panel = self.guest_accessibility_panel
        lattice = self.current_structure.lattice.matrix

        tiled_accessible = tile_points_for_supercell(
            result.accessible_points, lattice, self.supercell
        )
        self.viewer.show_guest_accessible_channels(tiled_accessible)
        if not panel.show_accessible_checkbox.isChecked():
            self.viewer.set_guest_accessible_visibility(False)

        tiled_blocked = tile_points_for_supercell(
            result.blocked_points, lattice, self.supercell
        )
        self.viewer.show_guest_blocked_regions(tiled_blocked)
        if not panel.show_blocked_checkbox.isChecked():
            self.viewer.set_guest_blocked_visibility(False)

        bottleneck_positions = (
            np.array([b.position for b in result.bottlenecks])
            if result.bottlenecks
            else np.zeros((0, 3))
        )
        tiled_bottlenecks = tile_points_for_supercell(
            bottleneck_positions, lattice, self.supercell
        )
        self.viewer.show_guest_bottlenecks(tiled_bottlenecks)
        if not panel.show_bottlenecks_checkbox.isChecked():
            self.viewer.set_guest_bottleneck_visibility(False)

    # ------------------------------------------------------------------
    # Open Metal Site (OMS) detection
    # ------------------------------------------------------------------

    #: How far out from the metal centre to place a missing-position
    #: marker - a representative bond-length-scale offset, not derived
    #: from any specific ligand (there isn't one; that is the point).
    OMS_MARKER_DISTANCE = 2.2

    def clear_analysis_visualization(self) -> None:
        """Analysis > Clear Analysis Visualization: the manual "reset
        the 3D view" action, for whenever the automatic clearing that
        now happens before each new analysis result isn't enough - e.g.
        clearing everything without necessarily running another
        analysis right away.
        """
        self.viewer.clear_all_analysis_visualizations()
        self.current_analysis_visualization = None
        self.statusBar().showMessage("Analysis visualization cleared.", 3000)

    def _on_customize_colors_menu_action(self) -> None:
        """View > Customize Atom Colors...: opens a dialog listing
        every element currently in the loaded structure, letting the
        person change or reset each one's render colour. Colours apply
        immediately as they're changed (not held back for an OK button
        the way the dialog's own docstring explains), so the 3D view
        and the legend overlay both need to be refreshed live as
        colors_changed fires, not just once when the dialog closes.
        """
        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before customizing atom colors.",
            )
            return

        elements = {
            next(iter(site.species)).symbol
            for site in self.current_structure.sites
        }

        dialog = ColorCustomizationDialog(
            self.viewer.color_provider, elements, self
        )
        dialog.colors_changed.connect(self._on_atom_colors_changed)
        dialog.exec()

    def _on_atom_colors_changed(self) -> None:
        self.element_legend.update_elements(element_counts(self.current_structure))
        self._rebuild_view()

    def _on_skeletal_diagram_menu_action(self) -> None:
        """View > 2D Skeletal Diagram...: a clean, standardised 2D
        skeletal-formula diagram of one unit cell's full nodes and
        linkers - as individual fragment diagrams (one per distinct
        linker or metal coordination environment, symmetry-equivalent
        copies collapsed together), not one flattened whole-framework
        graph. See skeletal_layout.compute_skeletal_diagram's own
        docstring for why: a MOF's periodic framework, even truncated
        to one cell, is normally one densely cross-linked graph with no
        clean planar embedding, confirmed directly (not assumed) by
        trying three different 2D layout approaches against a real
        structure and finding all three produced an unreadable tangle
        for it as a single connected diagram.

        Reuses the same periodic graph and linker/metal-node
        partitioning MOF Analysis itself already computes and reports,
        rather than a separately-derived decomposition, so this
        diagram's fragments are exactly the same linkers and nodes MOF
        Analysis shows.
        """
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before generating a 2D "
                "skeletal diagram.",
            )
            return

        import importlib.util

        if importlib.util.find_spec("rdkit") is None:
            QMessageBox.critical(
                self,
                "Missing dependency: rdkit",
                "The 2D Skeletal Diagram feature needs the 'rdkit' "
                "package, which isn't installed in this Python "
                "environment.\n\n"
                "Install it with:\n"
                "    pip install rdkit\n\n"
                "or, to install every dependency this application "
                "needs at once:\n"
                "    pip install -r requirements.txt",
            )
            return

        self._set_busy(True, "Generating 2D skeletal diagram...")

        self._run(
            self._generate_skeletal_diagram,
            self._on_skeletal_diagram_ready,
            self.current_structure,
            self.current_bonds,
        )

    def _generate_skeletal_diagram(self, structure, bonds):
        """Worker thread. Must not touch any Qt widget. RDKit's mol-
        building and sanitisation (the genuinely slow part for a large
        unit cell with many fragments) happens here; rendering the
        final composed image happens back on the main thread in
        _on_skeletal_diagram_ready, kept separate on principle even
        though RDKit's own drawing is not known to have the same
        thread-affinity concerns matplotlib's Qt-integrated backend
        does - consistent with how every other worker/main-thread split
        in this application is drawn, rather than a special case here.
        """
        graph = build_periodic_graph_for_mof_analysis(
            structure, bonds, is_metal
        )
        partition = partition_framework(
            graph.symbols, graph.bonds, is_metal, positions=graph.positions
        )

        return compute_skeletal_diagram(
            graph.symbols, graph.bonds, partition.linkers, partition.metal_nodes
        )

    @Slot(object)
    def _on_skeletal_diagram_ready(self, result) -> None:
        image = render_skeletal_diagram(result)

        message = (
            f"2D skeletal diagram: {len(result.fragments)} distinct "
            f"fragment(s)."
        )
        if result.skipped_count:
            message += (
                f" {result.skipped_count} fragment(s) could not be "
                "drawn and were skipped."
            )
        self._set_busy(False, message)

        dialog = SkeletalDiagramDialog(image, self)
        dialog.exec()

    def _on_oms_menu_action(self) -> None:
        """Analysis > Detect Open Metal Sites...: switches to the panel
        and runs immediately - unlike pore/surface/guest analysis, OMS
        detection has no per-run parameter to collect from the panel
        first, so there is nothing to wait on.
        """
        self._show_inspector_tab(self.oms_panel)
        self.detect_oms_structure()

    def detect_oms_structure(self) -> None:
        """Run Open Metal Site detection on the currently loaded
        structure. Connected to OMSPanel.run_requested.
        """
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before running OMS detection.",
            )
            return

        self._set_busy(True, "Detecting open metal sites...")

        self._run(
            self._run_oms_detection,
            self._on_oms_detection_ready,
            self.current_structure,
            self.current_site_labels,
        )

    def _run_oms_detection(self, structure, labels):
        """Worker thread. Must not touch any Qt widget."""
        return detect_open_metal_sites(structure, labels)

    @Slot(object)
    def _on_oms_detection_ready(self, result) -> None:
        self.current_oms_result = result
        self.oms_panel.set_result(result)
        self._show_inspector_tab(self.oms_panel)

        self.viewer.clear_all_analysis_visualizations()

        if self.oms_panel.highlight_all_checkbox.isChecked():
            self.current_analysis_visualization = "oms"
            self._highlight_all_oms(result)
        else:
            self.current_analysis_visualization = None

        self._set_busy(
            False,
            f"OMS detection: {result.oms_count} open metal site(s) "
            f"found ({len(result.sites)} analyzed).",
        )

    def _highlight_all_oms(self, result) -> None:
        """Draw every open site's highlight and adsorption markers,
        tiled to match whatever supercell is currently shown - see
        _display_pore_visualization's docstring for why the marker
        points need this. The atom highlight itself is a separate,
        not-yet-addressed limitation: highlight_group works by atom
        index into the current render model, via
        _render_atom_indices_for_sites. Until round 49 that picked a
        single representative copy of each site, so in a supercell the
        outline appeared once while the markers repeated in every
        cell; it now asks for every drawn copy (all_images=True), so
        outline and markers tile together.
        """
        open_sites = result.open_sites

        atom_indices = self._render_atom_indices_for_sites(
            [site.site_index for site in open_sites], all_images=True
        )
        self.viewer.highlight_group(
            atom_indices, self.viewer.OMS_HIGHLIGHT_COLOR
        )

        marker_points = np.array(
            [
                tuple(
                    p + d * self.OMS_MARKER_DISTANCE
                    for p, d in zip(site.position, mp.direction)
                )
                for site in open_sites
                for mp in site.missing_positions
            ]
        ) if any(site.missing_positions for site in open_sites) else np.zeros(
            (0, 3)
        )

        tiled_markers = tile_points_for_supercell(
            marker_points,
            self.current_structure.lattice.matrix,
            self.supercell,
        )
        self.viewer.show_oms_adsorption_markers(tiled_markers)

    def _on_oms_highlight_all_toggled(self, checked: bool) -> None:
        if self.current_oms_result is None:
            return

        if checked:
            self.current_analysis_visualization = "oms"
            self._highlight_all_oms(self.current_oms_result)
        else:
            self.current_analysis_visualization = None
            self.viewer.clear_group_highlight()
            self.viewer.clear_oms_marker_visualization()

    def _on_oms_site_selected(self, row_index) -> None:
        """OMS inspection mode: selecting a row shows that one site's
        detail and highlights its local coordination sphere (the
        metal plus its coordinating neighbours) and adsorption marker
        in place of the general "every open site" overview - the same
        "selecting a new one replaces the old one" rule every other
        group highlight in this viewer already follows.
        """
        if row_index is None or self.current_oms_result is None:
            self.oms_panel.show_site_detail(None)
            return

        site = self.current_oms_result.sites[row_index]
        self.oms_panel.show_site_detail(site)

        local_site_indices = [site.site_index] + [
            neighbor.site_index
            for neighbor in site.coordination_environment.neighbors
        ]
        atom_indices = self._render_atom_indices_for_sites(
            local_site_indices
        )
        self.viewer.highlight_group(
            atom_indices, self.viewer.OMS_HIGHLIGHT_COLOR
        )

        marker_points = [
            tuple(
                p + d * self.OMS_MARKER_DISTANCE
                for p, d in zip(site.position, mp.direction)
            )
            for mp in site.missing_positions
        ]
        self.viewer.show_oms_adsorption_markers(marker_points)

    def _render_atom_indices_for_sites(
        self, site_indices, *, all_images: bool = False
    ) -> list[int]:
        """Map base-structure site indices to indices into the current
        render model's flat atom list, preferring each site's
        reference-cell (image (0, 0, 0)) copy when more than one drawn
        image exists - the same convention CrystalViewer.
        select_atom_by_site_index() already uses, kept consistent here
        so a coordination sphere highlights the same copy a tree click
        on that site would.

        A site with no drawn copy at all (hidden by a toggle, or
        simply not part of the current view) is silently skipped
        rather than raising - the highlight just includes whichever of
        the environment's atoms are actually visible.

        all_images=True returns *every* drawn copy of each site instead
        of one representative - for an overview highlight (every open
        metal site) that should repeat across a supercell exactly the
        way its markers already do, rather than outlining one cell's
        atoms and leaving the identical copies next door unmarked.
        """
        model = self.current_render_model

        if model is None:
            return []

        indices = []

        for site_index in site_indices:
            candidates = [
                index
                for index, atom in enumerate(model.atoms)
                if atom.site_index == site_index
            ]

            if not candidates:
                continue

            if all_images:
                indices.extend(candidates)
                continue

            chosen = next(
                (
                    index
                    for index in candidates
                    if model.atoms[index].image == (0, 0, 0)
                ),
                candidates[0],
            )
            indices.append(chosen)

        return indices

    def _render_atom_indices_for_site_images(self, site_images) -> list[int]:
        """Map (site_index, image) pairs - from the periodic graph MOF
        analysis built - to indices into the current render model's
        flat atom list.

        Unlike _render_atom_indices_for_sites (which just needs *a*
        drawn copy of a site, and reasonably prefers the reference
        cell), a MOF linker or node is a specific set of periodic
        instances - highlighting a *different* image than the one
        actually in that linker would be visibly wrong, not just a
        reasonable substitute. A site whose exact image isn't part of
        the current 3D view (a different supercell or completion
        setting than MOF analysis used internally) is skipped rather
        than substituted with the wrong copy.
        """
        model = self.current_render_model

        if model is None:
            return []

        lookup = {
            (atom.site_index, atom.image): index
            for index, atom in enumerate(model.atoms)
        }

        return [
            lookup[site_image]
            for site_image in site_images
            if site_image in lookup
        ]

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    @Slot(object)
    def _on_measure_mode_changed(self, action) -> None:
        mode = action.data()

        if mode is None:
            self.viewer.disable_measurement()
            self.measurement_mode = None
            self.statusBar().showMessage("Measurement off", 3000)
            return

        if not self.viewer.enable_measurement(mode):
            # Picking isn't available on this system (no working VTK
            # interactor, or an unsupported picker type). Revert the
            # menu to Off rather than leaving it checked on a mode that
            # silently isn't doing anything.
            if self.measure_off_action is not None:
                self.measure_off_action.setChecked(True)

            self.measurement_mode = None

            QMessageBox.warning(
                self,
                "Measurement unavailable",
                "Atom picking isn't available in this 3D view - your "
                "graphics driver or display setup may not support it.",
            )
            return

        self.measurement_mode = mode

        # Choosing a measurement mode is picking something in the 3D
        # view, the same as Select mode - if the cursor mode toggle was
        # sitting on View, reflect that picking is now genuinely active
        # rather than leaving the toolbar showing "View" while a
        # measurement is actually in progress.
        self.select_mode_action.setChecked(True)

        if mode == "distance":
            self.statusBar().showMessage(
                "Click two atoms in the 3D view to measure the distance "
                "between them."
            )
        elif mode == "torsion":
            self.statusBar().showMessage(
                "Click four atoms A-B-C-D in the 3D view to measure the "
                "torsion angle about the B-C bond."
            )
        else:
            self.statusBar().showMessage(
                "Click three atoms in the 3D view to measure the angle "
                "at the second one."
            )

    def _on_measurement_failed(self, reason: str) -> None:
        """A full set of picks that could not be measured (collinear
        atoms in a torsion, say): say why in the status bar, and that
        picking has restarted, instead of the clicks just vanishing.
        """
        self.statusBar().showMessage(
            f"Measurement not possible: {reason} Picking restarted.", 8000
        )

    def _on_clear_measurements(self) -> None:
        self.viewer.clear_measurements()
        self.measurements_panel.clear()

    def _on_measurement_result(self, result: MeasurementResult) -> None:
        """Called by CrystalViewer, on the GUI thread (VTK/Qt mouse
        interaction callbacks always run there), whenever a distance or
        angle measurement completes. Safe to touch widgets directly.

        The 3D view's own on-screen label only ever shows the numeric
        value, deliberately kept compact - full detail (which atoms,
        which elements) goes to the Measurements panel instead, which
        accumulates one row per measurement rather than being
        overwritten by whatever was measured most recently.
        """
        kind = {
            "distance": "Distance",
            "angle": "Angle",
            "torsion": "Torsion",
        }.get(result.kind, "Measurement")

        self.statusBar().showMessage(f"{kind}: {result.label}", 8000)

        if result.site_indices is None or self.current_structure is None:
            return

        labels_and_elements = [
            self._label_and_element(site_index)
            for site_index in result.site_indices
        ]

        if result.kind == "distance":
            (label_a, element_a), (label_b, element_b) = labels_and_elements

            self.measurements_panel.add_distance(
                label_a=label_a,
                element_a=element_a,
                label_b=label_b,
                element_b=element_b,
                distance_angstrom=result.value,
            )
        elif result.kind == "torsion":
            (
                (label_a, element_a),
                (label_b, element_b),
                (label_c, element_c),
                (label_d, element_d),
            ) = labels_and_elements

            self.measurements_panel.add_torsion(
                label_a=label_a,
                element_a=element_a,
                label_b=label_b,
                element_b=element_b,
                label_c=label_c,
                element_c=element_c,
                label_d=label_d,
                element_d=element_d,
                torsion_degrees=result.value,
            )
        else:
            (
                (label_a, element_a),
                (label_vertex, element_vertex),
                (label_c, element_c),
            ) = labels_and_elements

            self.measurements_panel.add_angle(
                label_a=label_a,
                element_a=element_a,
                label_vertex=label_vertex,
                element_vertex=element_vertex,
                label_c=label_c,
                element_c=element_c,
                angle_degrees=result.value,
            )

        self._show_results_tab(self.measurements_panel)

    def _label_and_element(self, site_index: int) -> tuple[str, str]:
        """(label, element symbol) for a base-structure site index -
        the same lookup the Selected Atom panel uses, factored out
        since the Measurements panel needs it for up to three atoms at
        once.
        """
        label = (
            self.current_site_labels[site_index]
            if site_index < len(self.current_site_labels)
            else f"?{site_index + 1}"
        )

        site = self.current_structure.sites[site_index]
        species = list(site.species.keys())
        element = species[0].symbol if species else "?"

        return label, element

    def _resolve_nearest_image_distance(self, atom_a, atom_b, lattice_matrix):
        """CrystalViewer.resolve_distance hook: nearest-image (minimum
        periodic image) distance between two picked atoms.

        The two atoms a person clicks are whichever periodic copies
        happened to be drawn - a supercell tile, or an image molecule
        completion pulled in - not necessarily the closest copies of
        those two sites to each other. This is the standard
        crystallographic answer to "how far apart are these two atoms"
        regardless of which specific drawn copies were clicked:
        anchored on wherever atom A actually is (its own clicked
        position, not necessarily the reference cell), find whichever
        periodic image of atom B's site is nearest to that point.

        Returns None (falls back to the plain as-clicked distance) if
        there's no structure or lattice to resolve against - should not
        normally happen mid-measurement, but the caller already treats
        None as "use the simple distance instead" for exactly this kind
        of defensive case.
        """
        if self.current_structure is None or lattice_matrix is None:
            return None

        sites = self.current_structure.sites

        if atom_b.site_index >= len(sites):
            return None

        # atom_b's own reference-cell Cartesian position - independent
        # of whichever image was actually rendered and clicked.
        base_position_b = tuple(
            float(v) for v in sites[atom_b.site_index].coords
        )

        position, _image, value = nearest_periodic_image(
            atom_a.position, base_position_b, lattice_matrix
        )

        return position, value

    # ------------------------------------------------------------------
    # Atom selection
    # ------------------------------------------------------------------

    def _on_atom_selected_in_3d(self, atom) -> None:
        """Called by CrystalViewer, on the GUI thread, when a click in
        the 3D view selects an atom.

        Sync direction: 3D -> tree. StructureTree.select_site_index()
        does not re-emit atom_selected, so this cannot loop back into
        this same handler.
        """
        self._show_selected_atom(
            site_index=atom.site_index,
            cartesian_coordinates=atom.position,
        )

        self.tree.select_site_index(atom.site_index)

    @Slot(int)
    def _on_atom_selected_in_tree(self, site_index: int) -> None:
        """Called when the user clicks a row in the structure tree.

        Sync direction: tree -> 3D. CrystalViewer.select_atom_by_site_
        index() does not invoke on_atom_selected, so this cannot loop
        back either. It also tells us the actual drawn Cartesian
        position to show - which may belong to a periodic image rather
        than the site's own coordinates, or may be nothing at all if
        this site isn't part of the current 3D view.
        """
        atom = self.viewer.select_atom_by_site_index(site_index)

        self._show_selected_atom(
            site_index=site_index,
            cartesian_coordinates=atom.position if atom is not None else None,
        )

    def _show_selected_atom(
        self,
        *,
        site_index: int,
        cartesian_coordinates,
    ) -> None:
        """Populate the Selected Atom panel for a site.

        cartesian_coordinates is whichever position the caller already
        resolved - the actual clicked-on 3D position for a 3D pick, the
        highlighted image's position for a tree click that found a
        match in the current view, or None for a tree click on a site
        that isn't currently drawn at all.
        """
        if self.current_structure is None:
            return

        sites = self.current_structure.sites

        if site_index < 0 or site_index >= len(sites):
            return

        # Tracked so Analysis > Coordination Environment can operate on
        # "whatever is currently selected" without needing its own
        # separate selection mechanism.
        self.current_selected_site_index = site_index

        site = sites[site_index]

        species = list(site.species.keys())
        element = species[0].symbol if species else "?"

        label = (
            self.current_site_labels[site_index]
            if site_index < len(self.current_site_labels)
            else f"?{site_index + 1}"
        )

        self.selected_atom_panel.set_atom(
            label=label,
            element=element,
            occupancy=site_occupancy(site),
            fractional_coordinates=tuple(float(v) for v in site.frac_coords),
            cartesian_coordinates=cartesian_coordinates,
        )

        self._show_inspector_tab(self.selected_atom_panel)

    # ------------------------------------------------------------------
    # XRD
    # ------------------------------------------------------------------

    def show_xrd(self) -> None:
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before running the XRD simulation.",
            )
            return

        dialog = XRDSetupDialog(
            self,
            two_theta_range=self.xrd_two_theta_range,
            wavelength_key=self.xrd_wavelength_key,
        )

        if dialog.exec() != XRDSetupDialog.DialogCode.Accepted:
            return

        self.xrd_two_theta_range = dialog.two_theta_range()
        self.xrd_wavelength_key = dialog.wavelength_key()

        self._set_busy(True, "Calculating powder XRD pattern...")

        self._run(
            self.xrd_simulator.simulate,
            self._on_xrd_ready,
            self.current_structure,
            two_theta_range=self.xrd_two_theta_range,
            wavelength=self.xrd_wavelength_key,
        )

    @Slot(object)
    def _on_xrd_ready(self, result) -> None:
        self.current_xrd_result = result
        reduced = result.reduced()

        self._set_busy(
            False,
            f"Powder XRD: {reduced.peak_count} peaks at or above "
            f"{reduced.min_relative_intensity:g}% of the strongest, "
            f"from {result.peak_count} calculated reflections",
        )

        XRDDialog(result, self).exec()

    # ------------------------------------------------------------------
    # Pore size distribution (round 51)
    # ------------------------------------------------------------------

    def compute_psd(self) -> None:
        if self._busy:
            return

        if self.current_structure is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file before computing a pore size "
                "distribution.",
            )
            return

        self._set_busy(True, "Computing pore size distribution...")
        self._run(
            self._run_psd,
            self._on_psd_ready,
            self.current_structure,
            self.viewer.radius_provider,
        )

    def _run_psd(self, structure, radius_provider):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.pore_size_distribution import (
            compute_pore_size_distribution,
        )

        return compute_pore_size_distribution(
            structure, radius_provider=radius_provider
        )

    @Slot(object)
    def _on_psd_ready(self, result) -> None:
        self.current_psd_result = result
        dominant = result.dominant_diameter
        self._set_busy(
            False,
            "Pore size distribution: "
            + (
                f"dominant diameter {dominant:.2f} \u00c5"
                if dominant is not None
                else "no void space"
            ),
        )
        self._show_psd_dialog(result)

    def _show_psd_dialog(self, result) -> None:
        """Separate so tests can skip the modal exec()."""
        title = self.current_info.formula if self.current_info else ""
        PoreSizeDistributionDialog(result, self, title=title).exec()

    # ------------------------------------------------------------------
    # Analysis report (round 51)
    # ------------------------------------------------------------------

    def build_analysis_report(self) -> dict | None:
        """The report for the current structure, from whatever has been
        computed so far - nothing is recalculated."""
        from thevoid_core.services import report as reporting

        if self.current_structure is None:
            return None

        info = self.current_info
        result = reporting.new_report(
            self.current_structure,
            source=info.file_path if info else None,
            space_group=info.space_group if info else None,
        )
        results = result["results"]

        if self.current_topology_result is not None:
            from thevoid_core.services.topology import topology_to_dict

            result["structure"]["topology"] = self.current_topology_result.summary
            results["topology"] = topology_to_dict(self.current_topology_result)

        if self.current_pore_result is not None:
            results["pore"] = reporting.pore_to_dict(self.current_pore_result)
        if self.current_surface_area_result is not None:
            results["surface_area"] = reporting.surface_area_to_dict(
                self.current_surface_area_result
            )
        if self.current_guest_accessibility_result is not None:
            guest = self.current_guest_accessibility_result
            results["guests"] = {
                guest.guest_name: reporting.guest_to_dict(guest)
            }
        if self.current_oms_result is not None:
            results["oms"] = reporting.oms_to_dict(self.current_oms_result)
        if self.current_psd_result is not None:
            results["psd"] = reporting.psd_to_dict(self.current_psd_result)
        if self.current_xrd_result is not None:
            results["xrd"] = reporting.xrd_to_dict(self.current_xrd_result)
        if self.current_raspa_export_summary is not None:
            results["raspa_export"] = self.current_raspa_export_summary
        if self.current_reference_comparison is not None:
            from thevoid_core.services.reference.compare import comparison_to_dict

            record = comparison_to_dict(self.current_reference_comparison)
            record["candidates"] = [
                {"entry": m.entry.id, "match": m.confidence, "matched_by": m.how}
                for m in self.current_reference_matches
                if m.entry.id != self.current_reference_comparison.entry_id
            ]
            results["reference"] = reporting._clean(record)
        if self.current_predictions is not None:
            from thevoid_core.services.ml.predict import predictions_to_dict

            results["predictions"] = predictions_to_dict(self.current_predictions)

        return result

    def export_analysis_report(self, filename: str | None = None) -> bool:
        from thevoid_core.services import report as reporting

        report = self.build_analysis_report()

        if report is None:
            QMessageBox.information(
                self,
                "No Structure Loaded",
                "Open a structure file and run one or more analyses "
                "before exporting a report.",
            )
            return False

        if not filename:
            stem = (
                Path(self.current_info.file_path).stem
                if self.current_info and self.current_info.file_path
                else "report"
            )
            filename, chosen = QFileDialog.getSaveFileName(
                self,
                "Export Analysis Report",
                f"{stem}_report.md",
                "Markdown Report (*.md);;JSON Report (*.json)",
            )
            if not filename:
                return False
            if "json" in chosen.lower() and not filename.lower().endswith(".json"):
                filename += ".json"
            elif "markdown" in chosen.lower() and not filename.lower().endswith(".md"):
                filename += ".md"

        text = (
            reporting.to_json(report)
            if filename.lower().endswith(".json")
            else reporting.to_markdown(report)
        )

        try:
            Path(filename).write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
            return False

        count = len(report["results"])
        self.statusBar().showMessage(
            f"Report saved to {filename} ({count} analysis result(s)).", 8000
        )
        return True

    # ------------------------------------------------------------------
    # Reference databases (round 56)
    # ------------------------------------------------------------------

    def compare_with_coremof(self, entry_id: str | None = None) -> None:
        """Analysis > Compare with CoRE MOF 2024...: find the structure in
        the bundled index and compare (services/reference). Runs in the
        background: the Void's values are recomputed under CoRE's
        convention, which takes as long as a pore plus a surface-area
        analysis."""
        if self._busy:
            return
        if self.current_structure is None:
            QMessageBox.information(
                self, "No Structure Loaded",
                "Open a structure file before comparing it with CoRE MOF 2024.")
            return
        if self._reference_dialog is not None and self._reference_dialog.isVisible():
            self._reference_dialog.set_busy(True)
        self._set_busy(True, "Comparing with CoRE MOF 2024...")
        path = self.current_info.file_path if self.current_info else None
        self._run(self._run_reference_comparison, self._on_reference_comparison_ready,
                  self.current_structure, path, entry_id)

    def _run_reference_comparison(self, structure, path, entry_id):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.reference.compare import compare_with_coremof
        from thevoid_core.services.reference.coremof import find_matches

        matches = find_matches(structure, path=path)
        if entry_id is not None:
            chosen = next((m for m in matches if m.entry.id == entry_id), None)
        else:
            chosen = next((m for m in matches if m.confidence != "name only"), None)
        if chosen is None:
            return matches, None
        return matches, compare_with_coremof(structure, chosen)

    @Slot(object)
    def _on_reference_comparison_ready(self, payload) -> None:
        from thevoid_core.ui.dialogs.reference_comparison_dialog import (
            ReferenceComparisonDialog,
        )

        matches, comparison = payload
        self.current_reference_matches = matches
        if self._reference_dialog is None:
            self._reference_dialog = ReferenceComparisonDialog(self)
            self._reference_dialog.compare_requested.connect(self.compare_with_coremof)
        dialog = self._reference_dialog
        dialog.set_candidates(matches, comparison.entry_id if comparison else None)
        if comparison is None:
            dialog.set_no_match(
                "No CoRE MOF 2024 entry matches this structure (by name, or by "
                "metals, cell contents, density and composition)."
                if not matches else
                "Entries share this structure's name, but none matches its atom "
                "count, density or composition, so no comparison was made. "
                "Choose one to compare anyway."
            )
            message = "CoRE MOF 2024: no matching entry"
        else:
            self.current_reference_comparison = comparison
            dialog.set_comparison(comparison)
            flagged = comparison.flagged
            message = (f"CoRE MOF 2024 {comparison.entry_id} ({comparison.confidence}): "
                       + (f"{len(flagged)} difference(s) flagged" if flagged else "agrees"))
        dialog.show()
        dialog.raise_()
        self._set_busy(False, message)

    # ------------------------------------------------------------------
    # Property models (round 61)
    # ------------------------------------------------------------------

    def predict_properties(self, extra_models=None) -> None:
        """Analysis > Predict Properties...: the bundled models (and any
        model folders added in the dialog), in the background - the
        descriptors take a pore and a surface-area analysis under CoRE
        MOF's convention."""
        if self._busy:
            return
        if self.current_structure is None:
            QMessageBox.information(
                self, "No Structure Loaded",
                "Open a structure file before predicting its properties.")
            return
        from thevoid_core.services.ml.model import _require_sklearn

        try:
            _require_sklearn()
        except ImportError as exc:
            QMessageBox.information(self, "Property Models Need scikit-learn", str(exc))
            return
        if extra_models is None and self._prediction_dialog is not None:
            extra_models = self._prediction_dialog.extra_models
        if self._prediction_dialog is not None and self._prediction_dialog.isVisible():
            self._prediction_dialog.set_busy(True)
        self._set_busy(True, "Predicting properties...")
        self._run(self._run_predictions, self._on_predictions_ready,
                  self.current_structure, list(extra_models or []))

    def _run_predictions(self, structure, extra_models):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.ml.bundled import bundled_names
        from thevoid_core.services.ml.predict import predict_structure

        return predict_structure(structure, bundled_names() + list(extra_models))

    @Slot(object)
    def _on_predictions_ready(self, result) -> None:
        from thevoid_core.ui.dialogs.prediction_dialog import PredictionDialog

        self.current_predictions = result
        if self._prediction_dialog is None:
            self._prediction_dialog = PredictionDialog(self)
            self._prediction_dialog.models_requested.connect(self.predict_properties)
        dialog = self._prediction_dialog
        dialog.set_result(result)
        dialog.show()
        dialog.raise_()
        outside = sum(1 for p in result["predictions"] if not p.in_domain)
        self._set_busy(False, f"Predicted {len(result['predictions'])} properties"
                       + (f" ({outside} outside the applicability domain)" if outside else ""))

    # ------------------------------------------------------------------
    # The shared index (round 66)
    # ------------------------------------------------------------------

    @property
    def shared_index_path(self) -> str:
        return str(self._settings().value("shared_index/path", "") or "")

    def set_shared_index(self, path: str | None) -> None:
        self._settings().setValue("shared_index/path", str(path or ""))
        self._start_index_lookup()

    def choose_shared_index(self, path: str | None = None) -> str | None:
        """File > Shared Index > Set Index Folder...: an existing index, or
        (when the folder chosen is not one) the offer to make one."""
        from thevoid_core.ui.dialogs.shared_index_dialog import create_index, is_index

        if path is None:
            path = QFileDialog.getExistingDirectory(self, "Shared Index Folder",
                                                    self.shared_index_path)
            if not path:
                return None
            if not is_index(path):
                answer = QMessageBox.question(
                    self, "Not an Index",
                    f"{path} is not a shared index (no INDEX.json). Make a new index "
                    "somewhere?")
                if answer != QMessageBox.StandardButton.Yes:
                    return None
                path = create_index(self, path)
                if not path:
                    return None
        if not is_index(path):
            QMessageBox.warning(self, "Not an Index", f"{path} is not a shared index.")
            return None
        self.set_shared_index(path)
        self.statusBar().showMessage(f"Shared index: {path}", 8000)
        return path

    def _start_index_lookup(self) -> None:
        """Look the loaded structure up in the shared index, off the GUI
        thread, like the topology: a background detail of the Structure
        panel that never blocks anything."""
        from thevoid_core.ui.dialogs.shared_index_dialog import is_index

        self._index_token += 1
        self.current_index_matches = None
        path = self.shared_index_path
        source = self.current_info.file_path if self.current_info else None
        if not is_index(path):
            self.info_panel.set_index_unset()
            return
        if self.current_structure is None or not source:
            self.info_panel.index_label.setText("-")
            return
        self.info_panel.set_index_pending()
        worker = _Worker(self._find_in_index, path, source, self._index_token)
        worker.signals.finished.connect(self._on_index_lookup_ready)
        worker.signals.failed.connect(self.info_panel.set_index_failed)
        self._index_worker = worker
        self._pool.start(worker)

    def _find_in_index(self, index_path, source, token):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.index.store import Index

        index = Index.open(index_path)
        return token, index.info.get("name", ""), index.find(source)

    @Slot(object)
    def _on_index_lookup_ready(self, payload) -> None:
        token, name, matches = payload
        if token != self._index_token:
            return  # another structure, or another index, since
        self.current_index_matches = matches
        self.info_panel.set_index_matches(matches, name)

    def _ask_index_add(self, what: str):
        from thevoid_core.ui.dialogs.shared_index_dialog import SharedIndexAddDialog

        settings = self._settings()
        dialog = SharedIndexAddDialog(
            self, what=what, index_path=self.shared_index_path,
            contributor=str(settings.value("shared_index/contributor", "") or ""))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        values = dialog.values()
        settings.setValue("shared_index/contributor", values["contributor"])
        if values["index"] != self.shared_index_path:
            settings.setValue("shared_index/path", values["index"])
        return values

    def add_current_to_index(self, values: dict | None = None) -> None:
        """File > Shared Index > Add Current Structure...: the report of
        what has been run on this structure so far (as File > Export
        Analysis Report... writes it) becomes one entry."""
        if self._busy:
            return
        report = self.build_analysis_report()
        source = self.current_info.file_path if self.current_info else None
        if report is None or not source:
            QMessageBox.information(self, "Nothing to Add",
                                    "Open a structure file and run an analysis first.")
            return
        if not report.get("results"):
            QMessageBox.information(
                self, "Nothing to Add",
                "No analysis has been run on this structure yet. Run one from the "
                "Analysis menu (or add a the-void-analyze results folder from the Batch "
                "Results window).")
            return
        if values is None:
            values = self._ask_index_add(f"The analyses run on {Path(source).name} so far "
                                         f"({', '.join(sorted(report['results']))}).")
            if values is None:
                return
        self._set_busy(True, "Adding to the shared index...")
        self._run(self._run_index_add, self._on_index_add_ready, values, "report",
                  report, source)

    def add_results_to_index(self, folder, only=None, values: dict | None = None) -> None:
        """Batch Results > Add to Shared Index...: rows of a results folder."""
        if self._busy:
            return
        count = len(only) if only is not None else "all"
        if values is None:
            values = self._ask_index_add(f"{count} row(s) of {folder}.")
            if values is None:
                return
        self._set_busy(True, "Adding to the shared index...")
        self._run(self._run_index_add, self._on_index_add_ready, values, "results",
                  str(folder), only)

    def _run_index_add(self, values, kind, what, extra):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.index.store import Index

        index = Index.open(values["index"])
        kwargs = dict(contributor=values["contributor"], note=values.get("note", ""),
                      declared_licence=values.get("declared_licence"))
        if kind == "report":
            return index.add_report(what, extra, **kwargs)
        return index.add_results(what, only=set(extra) if extra is not None else None,
                                 **kwargs)

    @Slot(object)
    def _on_index_add_ready(self, result) -> None:
        from thevoid_core.ui.dialogs.shared_index_dialog import describe_add

        self.last_index_add = result
        self._set_busy(False, f"Shared index: {len(result.added)} added, "
                              f"{len(result.present)} already present")
        self._start_index_lookup()
        QMessageBox.information(self, "Added to the Shared Index", describe_add(result))

    def open_shared_index(self, path: str | None = None) -> None:
        """File > Shared Index > Open Index...: build it (in the
        background: new files may need structure matching) and open
        build/ in the Batch Results window."""
        from thevoid_core.ui.dialogs.shared_index_dialog import is_index

        if self._busy:
            return
        path = path or self.shared_index_path
        if not is_index(path):
            path = self.choose_shared_index()
            if not path:
                return
        self._set_busy(True, "Building the shared index...")
        self._run(self._run_index_build, self._on_index_build_ready, path)

    def _run_index_build(self, path):
        """Worker thread. Must not touch any Qt widget."""
        from thevoid_core.services.index.store import Index

        return Index.open(path).build()

    @Slot(object)
    def _on_index_build_ready(self, result) -> None:
        message = (f"Shared index: {result.rows} entries, {result.structures} structures, "
                   f"{result.determinations} determinations")
        if result.undecided:
            message += f" ({len(result.undecided)} pair(s) not compared)"
        self._set_busy(False, message)
        if result.problems:
            QMessageBox.warning(self, "Shared Index Problems", "\n".join(result.problems[:20]))
        self.open_batch_results(str(result.folder))

    def fetch_reference_structures(self):
        """File > Fetch Reference Structures..."""
        from thevoid_core.ui.dialogs.fetch_reference_dialog import FetchReferenceDialog

        if self._fetch_dialog is None:
            self._fetch_dialog = FetchReferenceDialog(self)
        self._fetch_dialog.show()
        self._fetch_dialog.raise_()
        return self._fetch_dialog

    def functionalize_linkers(self):
        """Build > Functionalise Linkers... (round 59)."""
        if self.current_structure is None:
            QMessageBox.information(
                self, "No Structure Loaded",
                "Open a MOF before making functionalised variants of it.")
            return None
        from thevoid_core.ui.dialogs.functionalize_dialog import FunctionalizeDialog

        source = self.current_info.file_path if self.current_info else None
        dialog = FunctionalizeDialog(self.current_structure, self.current_bonds, source=source,
                                     parent=self, open_file=self._open_file)
        dialog.show()
        self._functionalize_dialog = dialog
        return dialog

    # ------------------------------------------------------------------
    # Help menu (round 51)
    # ------------------------------------------------------------------

    def _create_help_menu(self, menu_bar) -> None:
        help_menu = menu_bar.addMenu("Help")

        self.about_action = help_menu.addAction("About the Void")
        self.about_action.setToolTip(
            "Version, library versions, methods, and the references to "
            "cite when results from this program are published."
        )
        self.about_action.triggered.connect(self.show_about)

        self.open_log_action = help_menu.addAction("Open Log Folder")
        self.open_log_action.setToolTip(
            "Open the folder holding this program's log files - attach "
            "the_void.log to a bug report."
        )
        self.open_log_action.triggered.connect(self.open_log_folder)

        self.copy_diagnostics_action = help_menu.addAction(
            "Copy Diagnostics"
        )
        self.copy_diagnostics_action.setToolTip(
            "Copy the version information, the loaded file, and the "
            "last lines of the log to the clipboard, ready to paste "
            "into a bug report."
        )
        self.copy_diagnostics_action.triggered.connect(self.copy_diagnostics)

    def show_about(self) -> None:
        from thevoid_core.infrastructure.logging import log_file_path

        AboutDialog(self, log_path=str(log_file_path())).exec()

    def open_log_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from thevoid_core.infrastructure.logging import log_directory

        folder = log_directory()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def diagnostics_text(self, log_lines: int = 40) -> str:
        from thevoid_core.infrastructure.logging import log_file_path
        from thevoid_core.ui.dialogs.about_dialog import versions_text

        import platform

        lines = [versions_text(), platform.platform()]

        if self.current_info is not None:
            lines.append(
                f"Loaded: {self.current_info.file_path} "
                f"({self.current_info.formula}, {self.current_info.site_count} sites)"
            )

        path = log_file_path()
        try:
            tail = path.read_text(encoding="utf-8", errors="replace").splitlines()
            lines += ["", f"Last {log_lines} log lines ({path}):"]
            lines += tail[-log_lines:]
        except OSError:
            lines.append(f"No log file at {path}.")

        return "\n".join(lines)

    def copy_diagnostics(self) -> None:
        from PySide6.QtGui import QGuiApplication

        QGuiApplication.clipboard().setText(self.diagnostics_text())
        self.statusBar().showMessage(
            "Diagnostics copied to the clipboard.", 5000
        )

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self._save_layout()
        if getattr(self, "batch_window", None) is not None:
            self.batch_window.close()
        self._pool.waitForDone(2000)
        self.viewer.close_plotter()

        super().closeEvent(event)
