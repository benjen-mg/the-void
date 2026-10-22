from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QVBoxLayout, QWidget

from thevoid_core.data.element_colors import ElementColorProvider
from thevoid_core.data.element_radii import ElementRadiusProvider
from thevoid_core.services.measurement import (
    MeasurementResult,
    angle,
    dihedral,
    distance,
    nearest_atom_index,
)

logger = logging.getLogger(__name__)


class _AtPosition:
    """A picked atom stood in at a different (already-resolved)
    position - lets a torsion's chain resolution hand the resolve hook
    "atom B, but at the periodic image chosen for it" as the anchor
    for resolving C, using the same (position, site_index) interface a
    RenderAtom offers.
    """

    __slots__ = ("site_index", "symbol", "position")

    def __init__(self, atom, position) -> None:
        self.site_index = atom.site_index
        self.symbol = getattr(atom, "symbol", None)
        self.position = tuple(position)


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """"#rrggbb" to an (r, g, b) tuple of floats in [0, 1] - used for
    per-point colour mapping, where pyvista needs plain float RGB
    triples rather than a colour name/hex string per point.
    """
    hex_color = hex_color.lstrip("#")

    return (
        int(hex_color[0:2], 16) / 255.0,
        int(hex_color[2:4], 16) / 255.0,
        int(hex_color[4:6], 16) / 255.0,
    )


def tile_points_for_supercell(points, lattice_matrix, supercell):
    """Replicate a set of Cartesian points across the same range of
    periodic images the 3D view's own atom tiling uses, so an analysis
    overlay (pore points, surface-area samples, guest-accessibility
    channels, OMS markers) visually spans the same extent as whatever
    supercell is currently displayed, instead of staying confined to
    one cell while the atoms around it repeat.

    Matches RenderModelBuilder's own tiling convention exactly - seeds
    every cell at (i, j, k) for i in range(nx), j in range(ny), k in
    range(nz), corner-anchored at the origin cell (0, 0, 0), not
    centred on it - confirmed directly against that code rather than
    assumed, since a different convention here would misalign the
    overlay against the atoms it is meant to sit alongside.

    points with shape (0, 3) (nothing to tile) and supercell (1, 1, 1)
    (nothing to add) both return promptly without doing any wasted
    work.
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return points

    nx, ny, nz = supercell

    if (nx, ny, nz) == (1, 1, 1):
        return points

    lattice = np.asarray(lattice_matrix, dtype=float)

    offsets = np.array(
        [
            i * lattice[0] + j * lattice[1] + k * lattice[2]
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
        ]
    )

    tiled = points[:, None, :] + offsets[None, :, :]

    return tiled.reshape(-1, 3)


class DisplayStyle(str, Enum):
    """How atoms and bonds are drawn."""

    #: Small balls at a fraction of the covalent radius, connected by
    #: rods. The default: shows both the connectivity and the atoms.
    BALL_AND_STICK = "ball_and_stick"

    #: Full van der Waals radius spheres, touching or overlapping,
    #: no bonds drawn. Shows the volume a molecule actually occupies -
    #: what "fills the space" - rather than its skeleton.
    SPACE_FILLING = "space_filling"

    #: Bonds only, drawn as thin lines/rods with no atom spheres.
    #: Shows connectivity with the least visual clutter - useful for a
    #: crowded structure where balls would mostly occlude each other.
    #: Atoms with no bonds at all would otherwise be invisible in this
    #: style, so they still get a small sphere.
    WIREFRAME = "wireframe"


class CrystalViewer(QWidget):
    """Ball-and-stick crystal structure viewer.

    Draws a RenderModel rather than a raw structure, so every bond it
    contains already has an atom at both ends - a molecule that crosses
    a cell face is drawn whole rather than as a fragment with dangling
    bonds.

    Geometry is batched one mesh per element: all atoms of a species
    share one glyphed mesh, all bond halves of a species are merged into
    one mesh, and the unit cell (or supercell grid of cells) is drawn as
    a single line mesh. A structure with hundreds of atoms costs a
    handful of actors rather than one per atom.
    """

    #: Fraction of the covalent radius drawn as a ball, in
    #: DisplayStyle.BALL_AND_STICK.
    ATOM_SCALE = 0.30

    #: Floor on sphere radius so hydrogen stays visible (angstrom),
    #: also used for the isolated-atom spheres in WIREFRAME.
    MIN_ATOM_RADIUS = 0.16

    #: Cylinder radius for bonds in BALL_AND_STICK (angstrom).
    BOND_RADIUS = 0.075

    #: Cylinder radius for bonds in WIREFRAME - thin lines rather than
    #: rods, since the whole point of the style is minimal clutter.
    WIREFRAME_BOND_RADIUS = 0.02

    SPHERE_THETA_RESOLUTION = 24
    SPHERE_PHI_RESOLUTION = 24
    BOND_RESOLUTION = 12

    #: Neither is close to pure white or pure black - both are neutral
    #: mid-tone greys, chosen so no element colour (including white
    #: hydrogen or near-black outlines) disappears into the background.
    BACKGROUND_LIGHT = "#d8d9db"
    BACKGROUND_DARK = "#3a3c40"

    DEFAULT_BACKGROUND = BACKGROUND_LIGHT

    #: The camera orientation used both for the initial view and for
    #: View > Reset View. A plain pyvista/VTK camera position string.
    DEFAULT_CAMERA_POSITION = "iso"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.color_provider = ElementColorProvider()
        self.radius_provider = ElementRadiusProvider()

        self.show_unit_cell = True
        self.background_color = self.DEFAULT_BACKGROUND
        self.display_style = DisplayStyle.BALL_AND_STICK

        #: 1.0 is fully opaque. Mainly useful in SPACE_FILLING, where
        #: overlapping full-radius spheres can otherwise hide whatever
        #: sits inside a cluster.
        self.opacity = 1.0

        #: None, "distance", "angle", or "torsion". Set via
        #: enable_measurement().
        self._measurement_mode = None
        self._measurement_picks: list[int] = []
        self._measurement_actor_names: list[str] = []

        #: The "Select"/"View" cursor mode toggle. True (the default,
        #: matching every previous version's behaviour) means clicking
        #: an atom selects it, as it always has; False means the 3D
        #: view is rotate/pan/zoom only and a click never selects or
        #: measures anything, for someone who just wants to look around
        #: an analysis result without a click-and-drag being mistaken
        #: for a click-to-select. See set_picking_enabled().
        self.picking_enabled = True

        self._current_model = None
        self._lattice_matrix = None

        #: Called with a MeasurementResult each time a distance or
        #: angle completes. The viewer draws the visual marker itself;
        #: this is how a host window finds out to show it elsewhere too
        #: (a status bar message, for instance).
        self.on_measurement = None

        #: Optional: called with a short human-readable reason when a
        #: completed set of picks cannot be measured (three collinear
        #: atoms in a torsion, for instance), so the host can say so
        #: rather than the clicks silently producing nothing.
        self.on_measurement_failed = None

        #: Optional hook: called as resolve_distance(atom_a, atom_b,
        #: lattice_matrix) when a distance measurement (not angle)
        #: completes, where atom_a/atom_b are the two picked RenderAtom
        #: objects. If set and it returns (position, distance) rather
        #: than None, that position and distance are used instead of
        #: the naive straight-line distance between the two literal
        #: picked points - this is how nearest-image (minimum periodic
        #: image) distance gets applied, without this module needing to
        #: know anything about fractional coordinates or pymatgen
        #: itself. Left unset, distance measurement uses the two picked
        #: positions exactly as clicked - correct for a non-periodic
        #: viewer, and a sensible default even for a periodic one until
        #: a host window supplies the fractional-coordinate-aware
        #: resolver (see MainWindow._resolve_nearest_image_distance).
        self.resolve_distance = None

        #: Index into self._current_model.atoms of the one currently
        #: selected atom, or None. Only ever one at a time - selecting
        #: a new atom replaces this outright rather than adding to it.
        self._selected_atom_index = None
        self._selection_actor = None

        #: The one currently highlighted group of atoms (a MOF linker
        #: or metal node, or a coordination sphere), drawn as a single
        #: merged outline. Only one group highlighted at a time,
        #: mirroring atom selection.
        self._group_highlight_actor = None

        #: The current pore-space point cloud actor, if any, and the
        #: data behind it - kept so set_pore_visibility() can toggle
        #: display without needing the caller to re-run analysis just
        #: to show it again.
        self._pore_actor = None
        self._last_pore_points = None
        self._last_pore_color = self.PORE_POINT_COLOR

        #: Accessible surface area visualization: the sampled-point
        #: cloud (colour-mapped by accessibility) and the optional
        #: reconstructed mesh overlay, tracked separately since either
        #: can be toggled independently of the other.
        self._surface_points_actor = None
        self._surface_mesh_actor = None
        self._last_surface_points = None
        self._last_surface_accessible_mask = None

        #: Guest accessibility visualization: accessible channels,
        #: blocked regions, and bottleneck markers, each independently
        #: toggleable.
        self._guest_accessible_actor = None
        self._guest_blocked_actor = None
        self._guest_bottleneck_actor = None
        self._last_guest_accessible_points = None
        self._last_guest_blocked_points = None
        self._last_guest_bottleneck_points = None

        #: OMS adsorption-position markers. Highlighting the OMS atoms
        #: themselves, or one site's local coordination sphere for
        #: inspection, reuses _group_highlight_actor above rather than
        #: a separate actor.
        self._oms_marker_actor = None
        self._last_oms_marker_points = None

        #: Called with a RenderAtom whenever a 3D click selects an atom
        #: (not when selection is driven programmatically from the
        #: structure tree via select_atom_by_site_index, which already
        #: knows what it selected and would just be hearing an echo of
        #: itself).
        self.on_atom_selected = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter)

        self.plotter.set_background(self.background_color)

        try:
            self.plotter.enable_anti_aliasing()
        except Exception:
            logger.debug("Anti-aliasing unavailable.", exc_info=True)

        self.plotter.show_axes()

    def set_background_color(self, color: str) -> None:
        """Change the 3D view's background colour.

        Accepts anything pyvista's set_background does - a hex string
        like '#101822' or a named colour.
        """
        self.background_color = color
        self.plotter.set_background(color)
        self.plotter.render()

    def set_opacity(self, opacity: float) -> None:
        """Change atom opacity. 1.0 is fully opaque, 0.0 invisible.

        Does not itself redraw - the caller rebuilds the render model
        the normal way (opacity, like display style, is a rendering
        choice with nothing to recompute), matching how set_display_
        style-equivalent changes are already handled by callers.
        """
        self.opacity = max(0.0, min(1.0, opacity))

    def reset_view(self) -> None:
        """Restore the default camera orientation and re-fit to bounds.

        Deliberately not just plotter.reset_camera(): that call only
        re-fits zoom and pan to the *current* camera direction, so after
        the user has rotated the scene with the mouse, calling it alone
        keeps the rotated orientation and can look like nothing
        happened at all - which was exactly the bug this method fixes.
        Setting camera_position back to the canonical orientation first
        is what actually makes "reset" mean reset.
        """
        self.plotter.camera_position = self.DEFAULT_CAMERA_POSITION
        self.plotter.reset_camera()

    def export_image(self, filename: str, *, scale: int = 2) -> None:
        """Save the current 3D view as a PNG.

        scale renders at a multiple of the on-screen resolution (2x by
        default) before downsampling, since the window itself is
        usually too small to be a good source resolution for a figure
        or a report.
        """
        self.plotter.screenshot(filename, scale=scale)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        self.plotter.clear()

    def close_plotter(self) -> None:
        """Release the VTK render window.

        pyvistaqt requires an explicit close before the Qt widget is torn
        down; without it the interpreter can segfault on exit.
        """
        try:
            self.plotter.close()
        except Exception:
            logger.debug("Plotter close failed.", exc_info=True)

    def closeEvent(self, event):  # noqa: N802 - Qt naming
        self.close_plotter()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_model(self, model, lattice_matrix=None, reset_camera=True):
        """Draw a RenderModel, optionally with the unit cell outline(s)."""
        self.clear()

        # Kept for measurement picking (nearest_atom_index needs the
        # current atom positions) and re-applied automatically whenever
        # a new structure or supercell is drawn, without the caller
        # needing to do anything measurement-specific.
        self._current_model = model
        self._lattice_matrix = lattice_matrix
        self._measurement_picks = []
        # self.clear() above already wiped every actor including any
        # measurement markers/lines/labels, so just drop the bookkeeping
        # rather than calling _clear_measurement_actors(), which would
        # try to remove actors that are already gone.
        self._measurement_actor_names = []

        # Same reasoning for the selection highlight: self.clear()
        # already removed its actor, so just forget it rather than
        # trying to remove something that's already gone. The selection
        # itself does not survive a redraw - the atom index is only
        # meaningful against the atoms list that was current when it
        # was picked, and a rebuild (a new supercell, a display style
        # change, toggling hydrogens) can change which atoms exist or
        # what index a given one has.
        self._selected_atom_index = None
        self._selection_actor = None
        self._group_highlight_actor = None
        # Pore analysis is tied to the specific structure and supercell
        # it was computed for; a rebuild (a new file, a different
        # supercell) can make cached pore points meaningless, so this
        # follows the same reset-on-redraw policy as everything else
        # above rather than risk showing stale geometry.
        self._pore_actor = None
        self._last_pore_points = None

        # Same reasoning for surface area visualization.
        self._surface_points_actor = None
        self._surface_mesh_actor = None
        self._last_surface_points = None
        self._last_surface_accessible_mask = None

        # Same reasoning for guest accessibility visualization.
        self._guest_accessible_actor = None
        self._guest_blocked_actor = None
        self._guest_bottleneck_actor = None
        self._last_guest_accessible_points = None
        self._last_guest_blocked_points = None
        self._last_guest_bottleneck_points = None

        # Same reasoning for OMS adsorption markers.
        self._oms_marker_actor = None
        self._last_oms_marker_points = None

        if model is not None and model.atoms:
            # == rather than is: DisplayStyle is a str subclass so this
            # is correct either way, but Qt's QAction.data() round-trip
            # silently unwraps a str-Enum to a plain str (confirmed
            # directly - action.data() after setData(DisplayStyle.X)
            # comes back as type(str), not DisplayStyle), which made
            # every "is DisplayStyle.X" comparison here false and
            # rendering fall through to Ball-and-Stick regardless of
            # which style was actually selected in the menu.
            if self.display_style == DisplayStyle.SPACE_FILLING:
                self._render_atoms(model, style=DisplayStyle.SPACE_FILLING)
            elif self.display_style == DisplayStyle.WIREFRAME:
                self._render_bonds(model, radius=self.WIREFRAME_BOND_RADIUS)
                self._render_isolated_atoms(model)
            else:
                self._render_atoms(model, style=DisplayStyle.BALL_AND_STICK)
                self._render_bonds(model, radius=self.BOND_RADIUS)

            # Re-establish whichever picking mode was active before this
            # redraw. Atom selection is the default the moment a
            # structure exists to click on - it is not an opt-in mode
            # the way Distance/Angle measurement is - and if a
            # measurement mode *was* active, it is restored rather than
            # silently kicked back to selection just because something
            # else (a display style toggle, a supercell change) caused
            # a rebuild. picking_enabled (the "Select"/"View" cursor
            # mode toggle) overrides both: in View mode, a rebuild must
            # not silently re-enable picking of any kind, or a person
            # who deliberately switched to View mode specifically to
            # rotate freely without triggering a selection would have
            # it turned back on by an unrelated display change.
            if not self.picking_enabled:
                try:
                    self.plotter.disable_picking()
                except Exception:
                    logger.debug(
                        "disable_picking after redraw in View mode "
                        "failed; likely nothing was enabled to begin "
                        "with.",
                        exc_info=True,
                    )
            elif self._measurement_mode is not None:
                self.enable_measurement(self._measurement_mode)
            else:
                self.enable_selection()

        if self.show_unit_cell and lattice_matrix is not None:
            supercell = model.supercell if model is not None else (1, 1, 1)
            self._render_unit_cell(lattice_matrix, supercell)

        if reset_camera:
            # camera_position first, then reset_camera: reset_camera()
            # alone only re-fits zoom/pan to the *current* orientation,
            # it does not change which direction the camera is facing.
            # Without setting the orientation explicitly here, the very
            # first render's starting angle is whatever VTK's default
            # camera happens to be - not necessarily this isometric
            # view - and after a user has rotated the scene, calling
            # reset_camera() alone visibly does nothing (see reset_view
            # below, which hits the same fix for the button/menu case).
            self.plotter.camera_position = self.DEFAULT_CAMERA_POSITION
            self.plotter.reset_camera()
        else:
            self.plotter.render()

    def _render_atoms(self, model, *, style: DisplayStyle) -> None:
        by_element: dict[str, list] = defaultdict(list)

        for atom in model.atoms:
            by_element[atom.symbol].append(atom.position)

        unit_sphere = pv.Sphere(
            radius=1.0,
            theta_resolution=self.SPHERE_THETA_RESOLUTION,
            phi_resolution=self.SPHERE_PHI_RESOLUTION,
        )

        for symbol, centers in by_element.items():
            radius = self.atom_radius(symbol, style=style)

            cloud = pv.PolyData(np.asarray(centers, dtype=float))

            glyphed = cloud.glyph(
                geom=unit_sphere,
                scale=False,
                orient=False,
                factor=radius,
            )

            self.plotter.add_mesh(
                glyphed,
                color=self.color_provider.get_color(symbol),
                smooth_shading=True,
                ambient=0.15,
                diffuse=0.85,
                specular=0.85,
                specular_power=50,
                opacity=self.opacity,
            )

    def _render_isolated_atoms(self, model) -> None:
        """WIREFRAME-only: give unbonded atoms a small sphere.

        Otherwise a lone counter-ion with no bonds would be completely
        invisible in a style that only draws bonds.
        """
        isolated = model.isolated_atom_indices

        if not isolated:
            return

        by_element: dict[str, list] = defaultdict(list)

        for index in isolated:
            atom = model.atoms[index]
            by_element[atom.symbol].append(atom.position)

        unit_sphere = pv.Sphere(
            radius=1.0,
            theta_resolution=self.SPHERE_THETA_RESOLUTION,
            phi_resolution=self.SPHERE_PHI_RESOLUTION,
        )

        for symbol, centers in by_element.items():
            cloud = pv.PolyData(np.asarray(centers, dtype=float))

            glyphed = cloud.glyph(
                geom=unit_sphere,
                scale=False,
                orient=False,
                factor=self.MIN_ATOM_RADIUS,
            )

            self.plotter.add_mesh(
                glyphed,
                color=self.color_provider.get_color(symbol),
                smooth_shading=True,
                ambient=0.15,
                diffuse=0.85,
                specular=0.85,
                specular_power=50,
            )

    def _render_bonds(self, model, *, radius: float) -> None:
        """Draw each bond as two half-cylinders coloured by their end atom.

        Built with vectorized numpy (_batched_cylinders) rather than one
        pv.Cylinder() call per half-bond merged afterward: constructing
        each cylinder individually costs roughly 1.5ms regardless of
        size, which is unnoticeable for a few dozen bonds but measured
        at ~24s for the ~17,000 half-bonds a 3x3x3 supercell of a
        moderately dense molecular crystal produces - long enough to
        look like a hang. The vectorized path builds the same geometry
        in well under a second at that size.
        """
        by_element_starts: dict[str, list] = defaultdict(list)
        by_element_ends: dict[str, list] = defaultdict(list)

        atoms = model.atoms

        for bond in model.bonds:
            atom_a = atoms[bond.atom_a]
            atom_b = atoms[bond.atom_b]

            start = np.asarray(atom_a.position, dtype=float)
            end = np.asarray(atom_b.position, dtype=float)

            if np.linalg.norm(end - start) < 1e-6:
                continue

            midpoint = (start + end) / 2.0

            by_element_starts[atom_a.symbol].append(start)
            by_element_ends[atom_a.symbol].append(midpoint)

            by_element_starts[atom_b.symbol].append(midpoint)
            by_element_ends[atom_b.symbol].append(end)

        for symbol, starts in by_element_starts.items():
            mesh = self._batched_cylinders(
                np.asarray(starts, dtype=float),
                np.asarray(by_element_ends[symbol], dtype=float),
                radius=radius,
                resolution=self.BOND_RESOLUTION,
            )

            self.plotter.add_mesh(
                mesh,
                color=self.color_provider.get_color(symbol),
                smooth_shading=True,
                ambient=0.10,
                diffuse=0.85,
                specular=0.25,
            )

    def atom_radius(
        self,
        symbol: str,
        *,
        style: DisplayStyle = DisplayStyle.BALL_AND_STICK,
    ) -> float:
        if style == DisplayStyle.SPACE_FILLING:
            return self.radius_provider.get_vdw_radius(symbol)

        return max(
            self.radius_provider.get_radius(symbol) * self.ATOM_SCALE,
            self.MIN_ATOM_RADIUS,
        )

    @staticmethod
    def _batched_cylinders(
        starts: np.ndarray,
        ends: np.ndarray,
        *,
        radius: float,
        resolution: int,
    ) -> pv.PolyData:
        """Build N open-ended cylinders (start[i] to end[i]) in one mesh.

        Vectorized entirely in numpy: for each cylinder, a ring of
        `resolution` points is placed around the start and end centres,
        in a plane perpendicular to that cylinder's own axis, then
        joined into quads along the side. Every cylinder gets the same
        radius regardless of its length - unlike pyvista's glyph()
        vector-scale mode, which scales a template geometry uniformly
        by vector magnitude and would grow the radius along with the
        length (checked directly).

        No end caps, matching the un-capped cylinders this replaces:
        adjoining half-bonds already meet at the shared midpoint, and a
        bond's two outer ends sit inside their atom spheres.
        """
        n = len(starts)

        if n == 0:
            return pv.PolyData()

        axis = ends - starts
        length = np.linalg.norm(axis, axis=1, keepdims=True)
        # Guard divide-by-zero for a (filtered-out upstream, but cheap
        # to be defensive about) zero-length pair; direction is unused
        # for a degenerate cylinder anyway.
        safe_length = np.where(length < 1e-9, 1.0, length)
        direction = axis / safe_length

        # An arbitrary vector not parallel to direction, per cylinder,
        # to build a perpendicular frame from. +Z unless direction is
        # too close to +Z itself, in which case +X.
        reference = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        nearly_parallel = np.abs((direction * reference).sum(axis=1)) > 0.99
        reference[nearly_parallel] = np.array([1.0, 0.0, 0.0])

        perp1 = np.cross(direction, reference)
        perp1 /= np.linalg.norm(perp1, axis=1, keepdims=True)
        perp2 = np.cross(direction, perp1)

        theta = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # (n, resolution, 3): per cylinder, per ring point, the radial
        # offset from that cylinder's own centreline.
        offset = radius * (
            cos_t[None, :, None] * perp1[:, None, :]
            + sin_t[None, :, None] * perp2[:, None, :]
        )

        start_ring = starts[:, None, :] + offset
        end_ring = ends[:, None, :] + offset

        points = np.concatenate([start_ring, end_ring], axis=1).reshape(-1, 3)

        i = np.arange(resolution)
        i_next = (i + 1) % resolution

        # One side quad per (cylinder, ring-edge) pair, indices local
        # to that cylinder's own 2*resolution points.
        quad_local = np.stack(
            [i, i_next, resolution + i_next, resolution + i], axis=1
        )

        base = (np.arange(n) * 2 * resolution)[:, None, None]
        quads = (quad_local[None, :, :] + base).reshape(-1, 4)

        faces = np.empty((quads.shape[0], 5), dtype=np.int64)
        faces[:, 0] = 4
        faces[:, 1:] = quads

        mesh = pv.PolyData(points, faces.reshape(-1))
        mesh.compute_normals(
            cell_normals=False, point_normals=True, inplace=True
        )

        return mesh

    def _render_unit_cell(
        self,
        lattice_matrix,
        supercell: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Draw one wireframe box per cell in the requested tiling.

        All cells are batched into a single PolyData, so a 3x3x3 request
        (27 boxes, 216 points) costs one draw call, not 27. Drawn
        regardless of DisplayStyle - the cell outline is not part of the
        atom/bond style choice.
        """
        a, b, c = np.asarray(lattice_matrix, dtype=float)
        nx, ny, nz = supercell

        corner_offsets = np.array(
            [
                (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
            ],
            dtype=float,
        )

        edges = [
            (0, 1), (0, 2), (0, 3),
            (1, 4), (1, 5),
            (2, 4), (2, 6),
            (3, 5), (3, 6),
            (4, 7), (5, 7), (6, 7),
        ]

        all_points = []
        all_line_cells = []

        cell_index = 0

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    origin = i * a + j * b + k * c

                    corners = origin + corner_offsets @ np.array([a, b, c])
                    all_points.append(corners)

                    base = cell_index * 8

                    for p, q in edges:
                        all_line_cells += [2, base + p, base + q]

                    cell_index += 1

        points = np.vstack(all_points)
        lines = np.array(all_line_cells, dtype=int)

        self.plotter.add_mesh(
            pv.PolyData(points, lines=lines),
            color="black",
            line_width=2,
        )

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    #: Colour used for pick markers, connecting lines, and the value
    #: label - distinct from every element colour and both background
    #: colours so a measurement is always legible.
    MEASUREMENT_COLOR = "#f4c430"

    #: A click farther than this from the nearest atom centre (in
    #: angstrom) is treated as a miss - clicking a bond or empty space
    #: rather than an atom - and ignored, rather than silently snapping
    #: to whatever atom happens to be nearest regardless of how far away
    #: it actually is.
    MEASUREMENT_PICK_TOLERANCE = 1.5

    #: How many atom clicks each measurement kind takes. Torsion (added
    #: in round 49) is the A-B-C-D dihedral about the B-C bond.
    MEASUREMENT_PICKS_NEEDED = {"distance": 2, "angle": 3, "torsion": 4}

    def _set_viewer_cursor(self, shape) -> None:
        """Give the mouse cursor a shape that matches the active
        interaction mode - a crosshair while picking atoms for a
        measurement, a pointing hand while plain atom selection is
        active, and the ordinary arrow otherwise - so the mode is
        visible at a glance instead of only discoverable through the
        menu or toolbar. shape is a Qt.CursorShape, or None to restore
        the default arrow cursor.

        Best-effort only: the widget actually driving the render
        surface differs between the real QtInteractor-backed plotter
        and the plain offscreen Plotter used in tests, and neither
        this cosmetic touch nor its absence should ever be allowed to
        break picking itself.
        """
        target = self.plotter if hasattr(self.plotter, "setCursor") else self

        try:
            if shape is None:
                target.unsetCursor()
            else:
                target.setCursor(QCursor(shape))
        except Exception:
            logger.debug(
                "Could not update the viewer cursor.", exc_info=True
            )

    def enable_measurement(self, kind: str) -> bool:
        """Start picking atoms for a distance or angle measurement.

        kind is "distance" (2 atoms) or "angle" (3 atoms, the second
        pick is the vertex). Left-click an atom to pick it; once enough
        atoms are picked the result is computed, drawn in the scene,
        and (if set) passed to self.on_measurement. Picking then resets
        so another measurement can be taken immediately without
        re-enabling the mode.

        Safe to call while a measurement mode is already active -
        switching straight from Distance to Angle (or vice versa)
        without going through Off first is a normal thing to do from
        the menu, and calling enable_point_picking() a second time on
        top of an already-active picking session is a real, confirmed
        failure mode: it can raise on some VTK/pyvista builds, which
        this method previously did not guard against, only against a
        *fresh* failure with no prior session. disable_picking() is
        therefore always called first, unconditionally, whether or not
        a session looks like it's already running.

        Returns True if picking was actually enabled, False if the
        underlying VTK interactor doesn't support it (no render window
        yet, or a picker type unavailable on this system) - confirmed
        directly to be a real failure mode: enable_point_picking()
        raises AttributeError rather than returning an error when the
        interactor isn't ready. The caller should tell the user rather
        than let that propagate as an unhandled exception.
        """
        if kind not in self.MEASUREMENT_PICKS_NEEDED:
            raise ValueError(
                'kind must be "distance", "angle", or "torsion": '
                f"{kind!r}"
            )

        try:
            self.plotter.disable_picking()
        except Exception:
            logger.debug(
                "disable_picking before re-enabling failed; continuing "
                "anyway, since there may simply be nothing to disable.",
                exc_info=True,
            )

        self._measurement_mode = kind
        self._measurement_picks = []

        try:
            self.plotter.enable_point_picking(
                callback=self._on_measurement_pick,
                left_clicking=True,
                show_message=False,
                show_point=False,
                picker="point",
            )
        except Exception:
            logger.warning(
                "Point picking unavailable; cannot enable measurement "
                "mode.",
                exc_info=True,
            )
            self._measurement_mode = None
            return False

        self.picking_enabled = True
        self._set_viewer_cursor(Qt.CursorShape.CrossCursor)

        return True

    def disable_measurement(self) -> None:
        """Stop picking and return the mouse to normal camera control."""
        self._measurement_mode = None
        self._measurement_picks = []

        try:
            self.plotter.disable_picking()
        except Exception:
            logger.debug("disable_picking unavailable.", exc_info=True)

        self._set_viewer_cursor(None)

    def clear_measurements(self) -> None:
        """Remove every measurement - distance and angle alike -
        markers, lines, and labels drawn so far, without leaving or
        re-entering measurement mode.
        """
        self._measurement_picks = []
        self._clear_measurement_actors()
        self.plotter.render()

    def _clear_measurement_actors(self) -> None:
        for name in self._measurement_actor_names:
            try:
                self.plotter.remove_actor(name)
            except Exception:
                logger.debug(
                    "Could not remove measurement actor %r.",
                    name,
                    exc_info=True,
                )

        self._measurement_actor_names = []

    def _on_measurement_pick(self, picked_point) -> None:
        if self._current_model is None or not self._current_model.atoms:
            return

        atoms = self._current_model.atoms
        positions = [atom.position for atom in atoms]

        index = nearest_atom_index(
            tuple(picked_point),
            positions,
            max_distance=self.MEASUREMENT_PICK_TOLERANCE,
        )

        if index is None:
            # Clicked a bond, the unit cell, or empty space - not close
            # enough to any atom centre to count as picking it.
            return

        if self._measurement_picks and self._measurement_picks[-1] == index:
            # A double-click or an accidental second click on the atom
            # just picked - never a meaningful pick (a zero-length leg),
            # so ignore it rather than let it consume a slot.
            return

        self._measurement_picks.append(index)
        self._add_pick_marker(positions[index])

        needed = self.MEASUREMENT_PICKS_NEEDED[self._measurement_mode]

        if len(self._measurement_picks) < needed:
            return

        picked_atoms = [atoms[i] for i in self._measurement_picks]

        try:
            result = self._finish_measurement(picked_atoms)
        except ValueError as exc:
            self._measurement_picks = []
            logger.info("Measurement could not be completed: %s", exc)

            on_failed = getattr(self, "on_measurement_failed", None)

            if on_failed is not None:
                try:
                    on_failed(str(exc))
                except Exception:
                    logger.debug(
                        "on_measurement_failed callback raised.",
                        exc_info=True,
                    )
            return

        self._measurement_picks = []

        if self.on_measurement is not None:
            try:
                self.on_measurement(result)
            except Exception:
                logger.debug(
                    "on_measurement callback raised.", exc_info=True
                )

    def _finish_measurement(self, picked_atoms) -> MeasurementResult:
        atom_indices = tuple(self._measurement_picks)
        site_indices = tuple(atom.site_index for atom in picked_atoms)

        line_positions = [atom.position for atom in picked_atoms]

        if self._measurement_mode == "distance":
            value = distance(line_positions[0], line_positions[1])

            resolved = self._resolve_distance_pair(
                picked_atoms[0], picked_atoms[1]
            )

            if resolved is not None:
                resolved_position, resolved_value = resolved
                # The line's first endpoint stays exactly where it was
                # clicked; only the second is moved to whichever
                # periodic image the resolver found nearest to it, so
                # the drawn line and the reported distance always agree
                # with each other.
                line_positions[1] = resolved_position
                value = resolved_value
        elif self._measurement_mode == "torsion":
            # Periodic boundary support for torsions: walk the chain,
            # resolving each atom to the periodic image of its site
            # nearest the one before it - B nearest A, C nearest the
            # resolved B, D nearest the resolved C - so A-B-C-D is one
            # connected chain even when the clicked copies straddle a
            # cell boundary. A itself stays exactly where it was
            # clicked.
            for i in range(1, 4):
                resolved = self._resolve_distance_pair(
                    _AtPosition(picked_atoms[i - 1], line_positions[i - 1]),
                    picked_atoms[i],
                )
                if resolved is not None:
                    line_positions[i] = resolved[0]

            value = dihedral(*line_positions)
        else:
            # Periodic boundary support for angles: each outer atom (A
            # and C) is independently resolved to whichever periodic
            # image of its own site sits nearest to the vertex (B) -
            # the vertex itself is never moved, since it is the fixed
            # point the angle is measured *at*. A structure completed
            # across a cell boundary can otherwise show a bond angle
            # between an atom and the wrong periodic copy of one of its
            # neighbours; resolving each leg independently against the
            # shared vertex is what makes A-B-C the same chemically
            # meaningful angle regardless of which specific images were
            # clicked.
            resolved_a = self._resolve_distance_pair(
                picked_atoms[1], picked_atoms[0]
            )
            if resolved_a is not None:
                line_positions[0] = resolved_a[0]

            resolved_c = self._resolve_distance_pair(
                picked_atoms[1], picked_atoms[2]
            )
            if resolved_c is not None:
                line_positions[2] = resolved_c[0]

            value = angle(
                line_positions[0], line_positions[1], line_positions[2]
            )

        result = MeasurementResult(
            kind=self._measurement_mode,
            value=value,
            atom_indices=atom_indices,
            site_indices=site_indices,
        )

        self._draw_measurement(line_positions, result)

        return result

    def _resolve_distance_pair(self, anchor_atom, target_atom):
        """Apply the resolve_distance hook, if set, tolerating failure.

        Despite the name (kept for continuity - this is the same hook
        distance measurement uses), this resolves *any* anchor/target
        pair to the target's nearest periodic image relative to the
        anchor - which is also exactly the sub-problem each leg of an
        angle measurement needs solved, just with the vertex as the
        anchor instead of the first-picked atom.

        A resolver error must not break the measurement outright - it
        just means falling back to the straight-line, as-picked
        position rather than a periodicity-aware one.
        """
        if self.resolve_distance is None:
            return None

        try:
            return self.resolve_distance(
                anchor_atom, target_atom, self._lattice_matrix
            )
        except Exception:
            logger.warning(
                "resolve_distance hook raised; falling back to the "
                "straight-line position between the picked atoms.",
                exc_info=True,
            )
            return None

    def _draw_measurement(
        self, picked_positions, result: MeasurementResult
    ) -> None:
        segments = [
            (picked_positions[i], picked_positions[i + 1])
            for i in range(len(picked_positions) - 1)
        ]

        for start, end in segments:
            name = f"measurement_line_{len(self._measurement_actor_names)}"

            self.plotter.add_mesh(
                pv.Line(start, end),
                color=self.MEASUREMENT_COLOR,
                line_width=3,
                name=name,
            )
            self._measurement_actor_names.append(name)

        # The label sits at the midpoint of the line (distance), at
        # the vertex (angle), or at the midpoint of the central B-C
        # bond (torsion) - the point the value is actually about.
        if result.kind == "distance":
            label_point = tuple(
                (a + b) / 2.0
                for a, b in zip(picked_positions[0], picked_positions[1])
            )
        elif result.kind == "torsion":
            label_point = tuple(
                (a + b) / 2.0
                for a, b in zip(picked_positions[1], picked_positions[2])
            )
        else:
            label_point = picked_positions[1]

        label_name = (
            f"measurement_label_{len(self._measurement_actor_names)}"
        )

        self.plotter.add_point_labels(
            [label_point],
            [result.label],
            name=label_name,
            font_size=16,
            text_color=self.MEASUREMENT_COLOR,
            shape_opacity=0.55,
            show_points=False,
            always_visible=True,
        )
        self._measurement_actor_names.append(label_name)

        self.plotter.render()

    def _add_pick_marker(self, position) -> None:
        """A small sphere at each picked atom, shown immediately so the
        user gets feedback after every click, not only once a
        measurement completes.
        """
        name = f"measurement_marker_{len(self._measurement_actor_names)}"

        marker_radius = max(self.MIN_ATOM_RADIUS * 0.6, 0.08)

        self.plotter.add_mesh(
            pv.Sphere(radius=marker_radius, center=position),
            color=self.MEASUREMENT_COLOR,
            name=name,
            opacity=0.85,
        )
        self._measurement_actor_names.append(name)

        self.plotter.render()

    # ------------------------------------------------------------------
    # Atom selection
    # ------------------------------------------------------------------

    #: Distinct from MEASUREMENT_COLOR so a selection and an in-progress
    #: measurement marker are never confused for each other if both are
    #: visible at once.
    SELECTION_COLOR = "#39d353"

    #: Same reasoning as MEASUREMENT_PICK_TOLERANCE: a click farther
    #: than this from the nearest atom centre is a miss, not a
    #: selection of whatever happens to be nearest regardless of
    #: distance.
    SELECTION_PICK_TOLERANCE = 1.5

    #: How far outside the atom's own drawn radius the highlight ring
    #: sits, in angstrom - a fixed halo margin rather than a
    #: multiplier, so it looks like a consistent "ring around the atom"
    #: at any size rather than growing disproportionately for large
    #: space-filling spheres.
    SELECTION_HIGHLIGHT_MARGIN = 0.15

    def set_picking_enabled(self, enabled: bool) -> bool:
        """The "Select"/"View" cursor mode toggle: switch between
        clicking an atom selecting it (enabled, the historical default)
        and the 3D view being pure rotate/pan/zoom navigation, where a
        click never selects or measures anything (disabled).

        Takes effect immediately, not just on the next redraw - flipping
        this mid-session (the whole point of it existing) must stop or
        start picking right away, not wait for something unrelated like
        a display style change to happen to trigger a rebuild first.

        When re-enabling, restores whichever mode was actually active
        before View mode was switched on - plain atom selection, or
        Distance/Angle measurement if that was in progress - the same
        restoration render_model() already performs after an ordinary
        redraw, reused here for the same reason: switching back to
        Select mode must not silently cancel an in-progress measurement
        by kicking it back to plain selection.

        Returns True if the requested mode was successfully applied,
        False if picking isn't available on this system at all (in
        which case disabling is still applied, since "no picking" is
        trivially satisfiable regardless).
        """
        self.picking_enabled = enabled

        if not enabled:
            try:
                self.plotter.disable_picking()
            except Exception:
                logger.debug(
                    "disable_picking for View mode failed; likely "
                    "nothing was enabled to begin with.",
                    exc_info=True,
                )
            self._set_viewer_cursor(None)
            return True

        if self._measurement_mode is not None:
            applied = self.enable_measurement(self._measurement_mode)
        else:
            applied = self.enable_selection()
        if not applied:
            # Picking is unavailable here, so the window falls back to
            # View mode; say so, rather than leaving Select recorded as
            # the mode in force.
            self.picking_enabled = False
        return applied

    def enable_selection(self) -> bool:
        """Make left-clicking an atom in the 3D view select it.

        This is the default interaction whenever a structure is loaded
        and no measurement mode is active - render_model() calls this
        automatically after every redraw, so nothing needs to call it
        by hand in the ordinary case. Only relevant to call directly
        after explicitly leaving measurement mode (Measure > Off).

        Like enable_measurement(), always tears down any existing
        picking session first (see that method's docstring, and round
        18's fix, for why: enable_point_picking() raises if called
        while a session is already active, which switching between
        measurement and selection would otherwise trigger).

        Returns True if picking was enabled, False if the underlying
        VTK interactor doesn't support it on this system.
        """
        try:
            self.plotter.disable_picking()
        except Exception:
            logger.debug(
                "disable_picking before enabling selection failed; "
                "continuing anyway, since there may simply be nothing "
                "to disable.",
                exc_info=True,
            )

        try:
            self.plotter.enable_point_picking(
                callback=self._on_selection_pick,
                left_clicking=True,
                show_message=False,
                show_point=False,
                picker="point",
            )
        except Exception:
            logger.warning(
                "Point picking unavailable; cannot enable atom "
                "selection.",
                exc_info=True,
            )
            return False

        self.picking_enabled = True
        self._set_viewer_cursor(Qt.CursorShape.PointingHandCursor)

        return True

    def _on_selection_pick(self, picked_point) -> None:
        if self._current_model is None or not self._current_model.atoms:
            return

        positions = [atom.position for atom in self._current_model.atoms]

        index = nearest_atom_index(
            tuple(picked_point),
            positions,
            max_distance=self.SELECTION_PICK_TOLERANCE,
        )

        if index is None:
            # Clicked a bond, the unit cell, or empty space - not a
            # miss that should clear the current selection, just a
            # click that wasn't on an atom. Mirrors how a miss during
            # measurement is silently ignored rather than treated as
            # cancelling the in-progress measurement.
            return

        self._select_atom_index(index, notify=True)

    def _select_atom_index(self, index: int, *, notify: bool) -> None:
        atom = self._current_model.atoms[index]

        self._selected_atom_index = index
        self._draw_selection_highlight(atom)

        if notify and self.on_atom_selected is not None:
            try:
                self.on_atom_selected(atom)
            except Exception:
                logger.debug(
                    "on_atom_selected callback raised.", exc_info=True
                )

        self.plotter.render()

    def select_atom_by_site_index(self, site_index: int):
        """Highlight the drawn atom for a base-structure site index -
        the structure-tree-click direction of selection sync.

        A site can appear as more than one drawn atom at once (a
        supercell tiles it, or molecule completion pulled in a
        periodic image of it); the reference-cell copy (image
        (0, 0, 0)) is preferred when present, for a predictable, stable
        choice rather than whichever happened to be listed first.

        Returns the RenderAtom that was highlighted, or None if this
        site isn't part of the current view at all (hidden by a
        toggle, or simply never drawn) - the caller still has the
        site's own data to show in an info panel even then, there is
        just nothing to highlight in the 3D view for it.

        Does not invoke on_atom_selected: the caller already knows what
        it just selected (that is why it is calling this method), so
        notifying it back would be redundant at best and a feedback
        loop at worst - the same principle
        StructureTree.select_site_index() follows for the opposite
        direction.
        """
        if self._current_model is None:
            self.clear_selection()
            return None

        candidates = [
            (index, atom)
            for index, atom in enumerate(self._current_model.atoms)
            if atom.site_index == site_index
        ]

        if not candidates:
            self.clear_selection()
            return None

        for index, atom in candidates:
            if atom.image == (0, 0, 0):
                self._select_atom_index(index, notify=False)
                return atom

        index, atom = candidates[0]
        self._select_atom_index(index, notify=False)

        return atom

    def clear_selection(self) -> None:
        self._selected_atom_index = None

        if self._selection_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_actor)
            except Exception:
                logger.debug(
                    "Could not remove selection highlight actor.",
                    exc_info=True,
                )

            self._selection_actor = None

        self.plotter.render()

    def _draw_selection_highlight(self, atom) -> None:
        """A thin outline traced around the selected atom's own
        silhouette, rather than a filled shape - a highlight has to
        leave the atom's own colour visible underneath it to still look
        like "this atom, picked out" rather than "a coloured blob here
        now". A wireframe sphere was tried first and rejected after
        rendering it: at a resolution fine enough to look round, the
        grid of wireframe lines was dense enough to look almost solid,
        hiding the atom underneath entirely. add_silhouette() traces
        the actual outline of a mesh - just the ring where a sphere's
        surface reaches its own edge from the camera's point of view -
        which is exactly the "selection ring" look this needs.
        """
        if self._selection_actor is not None:
            try:
                self.plotter.remove_actor(self._selection_actor)
            except Exception:
                logger.debug(
                    "Could not remove previous selection highlight "
                    "actor before drawing a new one.",
                    exc_info=True,
                )
            self._selection_actor = None

        base_radius = self.atom_radius(atom.symbol, style=self.display_style)
        highlight_radius = base_radius + self.SELECTION_HIGHLIGHT_MARGIN

        geometry = pv.Sphere(
            radius=highlight_radius,
            center=atom.position,
            theta_resolution=24,
            phi_resolution=24,
        )

        self._selection_actor = self.plotter.add_silhouette(
            geometry,
            color=self.SELECTION_COLOR,
            line_width=4,
        )

    # ------------------------------------------------------------------
    # Group highlighting (MOF linkers/nodes, coordination spheres)
    # ------------------------------------------------------------------

    #: Extra margin over each atom's own radius for the group outline,
    #: matching the single-atom selection ring's proportions.
    GROUP_HIGHLIGHT_MARGIN = 0.15

    def highlight_group(self, atom_indices, color: str) -> None:
        """Highlight a set of atoms - one MOF linker, one metal node,
        or one coordination sphere - with a single merged outline in
        the given colour.

        Replaces any previous group highlight outright: only one group
        highlighted at a time, the same "selecting a new one replaces
        the old one" rule atom selection already follows, for the same
        reason - a screen covered in every linker's outline at once
        would show nothing useful.

        Silently does nothing if there is no current model or
        atom_indices is empty, rather than raising - a caller passing
        an empty selection (nothing found, or the panel was cleared)
        is a normal, expected case, not an error.
        """
        self._clear_group_highlight_actor()

        if self._current_model is None or not atom_indices:
            self.plotter.render()
            return

        atoms = self._current_model.atoms
        centers = []
        radii = []

        for index in atom_indices:
            if index < 0 or index >= len(atoms):
                continue

            atom = atoms[index]
            centers.append(atom.position)
            radii.append(
                self.atom_radius(atom.symbol, style=self.display_style)
                + self.GROUP_HIGHLIGHT_MARGIN
            )

        if not centers:
            self.plotter.render()
            return

        unit_sphere = pv.Sphere(
            radius=1.0, theta_resolution=16, phi_resolution=16
        )

        cloud = pv.PolyData(np.asarray(centers, dtype=float))
        # Per-atom radius via glyph's scale-by-scalar mode (confirmed
        # directly: scale=<array name> multiplies the template geometry
        # isotropically by that point's scalar value, giving each
        # glyphed sphere its own correct radius rather than one shared
        # factor for every point).
        cloud["radius"] = np.asarray(radii, dtype=float)

        glyphed = cloud.glyph(
            geom=unit_sphere,
            scale="radius",
            orient=False,
            factor=1.0,
        )

        self._group_highlight_actor = self.plotter.add_silhouette(
            glyphed,
            color=color,
            line_width=4,
        )

        self.plotter.render()

    def clear_group_highlight(self) -> None:
        self._clear_group_highlight_actor()
        self.plotter.render()

    def _clear_group_highlight_actor(self) -> None:
        if self._group_highlight_actor is None:
            return

        try:
            self.plotter.remove_actor(self._group_highlight_actor)
        except Exception:
            logger.debug(
                "Could not remove previous group highlight actor.",
                exc_info=True,
            )

        self._group_highlight_actor = None

    # ------------------------------------------------------------------
    # Pore space visualization
    # ------------------------------------------------------------------

    #: A cool, translucent blue reads as "open space" rather than "solid
    #: matter" - distinct from every other highlight colour already in
    #: use in this viewer (selection green, measurement gold, MOF
    #: linker orange, MOF node blue-ish, coordination purple).
    PORE_POINT_COLOR = "#22aaff"

    def show_pore_points(self, points, color: str = PORE_POINT_COLOR) -> None:
        """Render a point cloud representing accessible pore space -
        one point per grid position PoreAnalysisResult.accessible_points
        classified as part of a percolating network.

        A plain point cloud rather than one glyphed sphere per point:
        pore analysis can produce many thousands of points even at a
        modest grid resolution, and a point cloud with points rendered
        as small spheres (render_points_as_spheres, a cheap per-point
        billboard rather than real glyphed geometry) stays responsive
        at that count in a way merging real sphere meshes for every
        point would not.

        Replaces any previous pore visualization outright - only one
        pore rendering shown at a time, matching every other highlight
        in this viewer.
        """
        self._clear_pore_actor()

        self._last_pore_points = points
        self._last_pore_color = color

        if points is None or len(points) == 0:
            self.plotter.render()
            return

        cloud = pv.PolyData(np.asarray(points, dtype=float))

        self._pore_actor = self.plotter.add_mesh(
            cloud,
            color=color,
            point_size=6,
            render_points_as_spheres=True,
            opacity=0.55,
            lighting=False,
        )

        self.plotter.render()

    def clear_pore_visualization(self) -> None:
        self._clear_pore_actor()
        self._last_pore_points = None
        self.plotter.render()

    def set_pore_visibility(self, visible: bool) -> None:
        """Show or hide the current pore visualization without
        recomputing or re-uploading it - the actual "Toggle pore
        visualization on/off" control, distinct from
        show_pore_points/clear_pore_visualization, which respectively
        create and discard the underlying data.
        """
        if self._pore_actor is not None:
            self._pore_actor.SetVisibility(visible)
            self.plotter.render()

    def _clear_pore_actor(self) -> None:
        if self._pore_actor is None:
            return

        try:
            self.plotter.remove_actor(self._pore_actor)
        except Exception:
            logger.debug(
                "Could not remove previous pore visualization actor.",
                exc_info=True,
            )

        self._pore_actor = None

    # ------------------------------------------------------------------
    # Accessible surface area visualization
    # ------------------------------------------------------------------

    #: Distinct from every other colour already in use elsewhere in
    #: this viewer.
    SURFACE_ACCESSIBLE_COLOR = "#33cc66"
    SURFACE_BLOCKED_COLOR = "#cc3333"
    SURFACE_MESH_COLOR = "#33cc66"

    def show_surface_points(
        self,
        points,
        accessible_mask,
        accessible_color: str = SURFACE_ACCESSIBLE_COLOR,
        blocked_color: str = SURFACE_BLOCKED_COLOR,
    ) -> None:
        """Render every Monte Carlo-sampled surface point, colour-mapped
        by accessibility - green for a point a probe could actually
        reach, red for one blocked by a neighbouring atom - satisfying
        "render sampled surface points" and "surface colour mapping"
        together, since they are naturally the same draw call: one
        point cloud, two colours, rather than two overlapping clouds.

        Replaces any previous surface-point visualization outright,
        the same "selecting/running a new one replaces the old one"
        rule every other highlight in this viewer already follows.
        """
        self._clear_surface_points_actor()

        self._last_surface_points = points
        self._last_surface_accessible_mask = accessible_mask

        if points is None or len(points) == 0:
            self.plotter.render()
            return

        points = np.asarray(points, dtype=float)
        accessible_mask = np.asarray(accessible_mask, dtype=bool)

        cloud = pv.PolyData(points)
        colors = np.where(
            accessible_mask[:, None],
            np.array(_hex_to_rgb(accessible_color)),
            np.array(_hex_to_rgb(blocked_color)),
        )
        cloud["colors"] = (colors * 255).astype(np.uint8)

        self._surface_points_actor = self.plotter.add_mesh(
            cloud,
            scalars="colors",
            rgb=True,
            point_size=5,
            render_points_as_spheres=True,
            opacity=0.85,
            lighting=False,
        )

        self.plotter.render()

    def show_surface_mesh(
        self, points, color: str = SURFACE_MESH_COLOR
    ) -> None:
        """Render an approximate reconstructed surface mesh through the
        accessible sample points - a visual aid showing the overall
        shape of the accessible surface, not an independent scientific
        measurement (the accessible surface area itself is already
        fully determined by the Monte Carlo point sampling; this mesh
        is a smoothed visual interpolation through those same points).

        Reconstructs each connected surface patch separately rather
        than fitting one single surface through every point at once.
        This matters because a real structure's accessible surface is
        generally *not* one connected blob - a MOF has many separate
        atom clusters scattered across the cell - and PyVista/VTK's
        reconstruct_surface() is built to fit one continuous implicit
        surface through whatever it is given. Confirmed directly (not
        assumed) to go badly wrong on that kind of input: handed a
        realistic multi-cluster point cloud with default parameters, it
        produced a mesh whose bounds extended more than double the
        actual point cloud's own extent in every direction - large,
        flat, wildly extrapolated sheets connecting and surrounding the
        real clusters rather than a tight skin around each one. Reduced
        but not eliminated by supplying an explicit, data-derived
        sample_spacing instead of trusting VTK's own auto-estimate
        (also confirmed directly to help, on its own, only partially);
        clustering into connected components first and reconstructing
        each separately is what actually removes the artifact, visually
        confirmed by rendering both the broken and fixed versions
        against the same real, multi-cluster test data side by side.

        Two points are treated as part of the same patch if they are
        within 3x their own patch's typical (median) nearest-neighbour
        spacing of each other - generous enough to bridge the small
        gaps Monte Carlo sampling naturally leaves within one continuous
        surface, small enough that genuinely separate atoms (with real
        empty space between them) do not get bridged.

        Silently does nothing if there are too few points to
        reconstruct anything from, or if every cluster turns out too
        small to reconstruct, rather than raising - a caller with a
        very small or fully blocked structure should not have this fail
        the whole visualization.
        """
        self._clear_surface_mesh_actor()

        if points is None or len(points) < 10:
            self.plotter.render()
            return

        points = np.asarray(points, dtype=float)

        try:
            cluster_labels = self._cluster_surface_points(points)
        except Exception:
            logger.warning(
                "Could not cluster sampled points into surface "
                "patches; showing the point cloud only.",
                exc_info=True,
            )
            self.plotter.render()
            return

        meshes = []

        for label in np.unique(cluster_labels):
            cluster_points = points[cluster_labels == label]

            if len(cluster_points) < self.MIN_POINTS_PER_SURFACE_PATCH:
                continue

            try:
                spacing = self._median_nearest_neighbor_distance(
                    cluster_points
                )
                cloud = pv.PolyData(cluster_points)
                mesh = cloud.reconstruct_surface(sample_spacing=spacing)
            except Exception:
                logger.debug(
                    "Could not reconstruct one surface patch; skipping "
                    "just that patch rather than the whole overlay.",
                    exc_info=True,
                )
                continue

            if mesh.n_points > 0:
                meshes.append(mesh)

        if not meshes:
            self.plotter.render()
            return

        combined = meshes[0]
        for mesh in meshes[1:]:
            combined = combined.merge(mesh)

        self._surface_mesh_actor = self.plotter.add_mesh(
            combined,
            color=color,
            opacity=0.35,
            smooth_shading=True,
            specular=0.2,
        )

        self.plotter.render()

    #: Below this many points, a connected patch is too small for
    #: reconstruct_surface() to produce anything meaningful - skipped
    #: rather than attempted and likely failing or producing noise.
    MIN_POINTS_PER_SURFACE_PATCH = 10

    @staticmethod
    def _median_nearest_neighbor_distance(points) -> float:
        """The typical spacing between neighbouring sampled points in
        this specific cluster - used as reconstruct_surface()'s
        sample_spacing instead of trusting VTK's own auto-estimate,
        which is derived from the whole input's bounding box and can be
        a poor match for how densely *this* particular patch was
        actually sampled (larger atoms get the same fixed sample count
        as smaller ones, spread over more surface area, so different
        clusters can legitimately have different natural spacings).
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        distances, _ = tree.query(points, k=2)

        return float(np.median(distances[:, 1]))

    @classmethod
    def _cluster_surface_points(cls, points) -> np.ndarray:
        """Group points into connected surface patches: two points
        link up if they are within 3x the point cloud's own typical
        nearest-neighbour spacing of each other. Returns an integer
        label per point (same length as points), grouping points from
        the same patch under the same label.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        n = len(points)
        spacing = cls._median_nearest_neighbor_distance(points)
        link_radius = spacing * 3.0

        tree = cKDTree(points)
        pairs = tree.query_pairs(r=link_radius, output_type="ndarray")

        if len(pairs) == 0:
            return np.arange(n)

        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(rows))
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))

        _, labels = connected_components(graph, directed=False)

        return labels

    def clear_surface_visualization(self) -> None:
        self._clear_surface_points_actor()
        self._clear_surface_mesh_actor()
        self._last_surface_points = None
        self._last_surface_accessible_mask = None
        self.plotter.render()

    def set_surface_points_visibility(self, visible: bool) -> None:
        if self._surface_points_actor is not None:
            self._surface_points_actor.SetVisibility(visible)
            self.plotter.render()

    def set_surface_mesh_visibility(self, visible: bool) -> None:
        if self._surface_mesh_actor is not None:
            self._surface_mesh_actor.SetVisibility(visible)
            self.plotter.render()

    def _clear_surface_points_actor(self) -> None:
        if self._surface_points_actor is None:
            return

        try:
            self.plotter.remove_actor(self._surface_points_actor)
        except Exception:
            logger.debug(
                "Could not remove previous surface-point actor.",
                exc_info=True,
            )

        self._surface_points_actor = None

    def _clear_surface_mesh_actor(self) -> None:
        if self._surface_mesh_actor is None:
            return

        try:
            self.plotter.remove_actor(self._surface_mesh_actor)
        except Exception:
            logger.debug(
                "Could not remove previous surface-mesh actor.",
                exc_info=True,
            )

        self._surface_mesh_actor = None

    # ------------------------------------------------------------------
    # Guest accessibility visualization
    # ------------------------------------------------------------------

    #: Distinct from every other colour already in use in this viewer.
    #: "Accessible channels" and "diffusion pathways" are the same
    #: underlying data (the connected, percolating network a guest can
    #: actually travel through) and share one colour/toggle rather than
    #: being drawn twice; likewise "inaccessible regions" and "blocking
    #: regions" are the same underlying void-but-too-narrow data.
    GUEST_ACCESSIBLE_COLOR = "#2eb872"
    GUEST_BLOCKED_COLOR = "#d1495b"
    GUEST_BOTTLENECK_COLOR = "#f4a300"

    def show_guest_accessible_channels(
        self, points, color: str = GUEST_ACCESSIBLE_COLOR
    ) -> None:
        """The connected, percolating channel network a guest of the
        analyzed size can actually diffuse through - satisfies both
        "highlight accessible channels" and "show diffusion pathways",
        which are the same network viewed from two angles rather than
        two different things to compute or draw.
        """
        self._clear_guest_accessible_actor()

        self._last_guest_accessible_points = points

        if points is None or len(points) == 0:
            self.plotter.render()
            return

        cloud = pv.PolyData(np.asarray(points, dtype=float))

        self._guest_accessible_actor = self.plotter.add_mesh(
            cloud,
            color=color,
            point_size=6,
            render_points_as_spheres=True,
            opacity=0.6,
            lighting=False,
        )

        self.plotter.render()

    def show_guest_blocked_regions(
        self, points, color: str = GUEST_BLOCKED_COLOR
    ) -> None:
        """Real void space - not inside any atom - that is too narrow
        for the analyzed guest. Satisfies both "highlight inaccessible
        regions" and "show blocking regions": the same set of points
        (void, but under-sized for this guest), not two separate
        computations.
        """
        self._clear_guest_blocked_actor()

        self._last_guest_blocked_points = points

        if points is None or len(points) == 0:
            self.plotter.render()
            return

        cloud = pv.PolyData(np.asarray(points, dtype=float))

        self._guest_blocked_actor = self.plotter.add_mesh(
            cloud,
            color=color,
            point_size=5,
            render_points_as_spheres=True,
            opacity=0.45,
            lighting=False,
        )

        self.plotter.render()

    def show_guest_bottlenecks(
        self, points, color: str = GUEST_BOTTLENECK_COLOR
    ) -> None:
        """Larger, distinct markers at each identified constriction
        point along the accessible network - the specific locations a
        guest has to squeeze through, called out separately from the
        general accessible-channel cloud so they are actually findable
        rather than blending into hundreds of ordinary channel points.
        """
        self._clear_guest_bottleneck_actor()

        self._last_guest_bottleneck_points = points

        if points is None or len(points) == 0:
            self.plotter.render()
            return

        cloud = pv.PolyData(np.asarray(points, dtype=float))

        self._guest_bottleneck_actor = self.plotter.add_mesh(
            cloud,
            color=color,
            point_size=14,
            render_points_as_spheres=True,
            opacity=0.95,
            lighting=False,
        )

        self.plotter.render()

    def clear_guest_accessibility_visualization(self) -> None:
        self._clear_guest_accessible_actor()
        self._clear_guest_blocked_actor()
        self._clear_guest_bottleneck_actor()
        self._last_guest_accessible_points = None
        self._last_guest_blocked_points = None
        self._last_guest_bottleneck_points = None
        self.plotter.render()

    def set_guest_accessible_visibility(self, visible: bool) -> None:
        if self._guest_accessible_actor is not None:
            self._guest_accessible_actor.SetVisibility(visible)
            self.plotter.render()

    def set_guest_blocked_visibility(self, visible: bool) -> None:
        if self._guest_blocked_actor is not None:
            self._guest_blocked_actor.SetVisibility(visible)
            self.plotter.render()

    def set_guest_bottleneck_visibility(self, visible: bool) -> None:
        if self._guest_bottleneck_actor is not None:
            self._guest_bottleneck_actor.SetVisibility(visible)
            self.plotter.render()

    def _clear_guest_accessible_actor(self) -> None:
        if self._guest_accessible_actor is None:
            return

        try:
            self.plotter.remove_actor(self._guest_accessible_actor)
        except Exception:
            logger.debug(
                "Could not remove previous guest-accessible actor.",
                exc_info=True,
            )

        self._guest_accessible_actor = None

    def _clear_guest_blocked_actor(self) -> None:
        if self._guest_blocked_actor is None:
            return

        try:
            self.plotter.remove_actor(self._guest_blocked_actor)
        except Exception:
            logger.debug(
                "Could not remove previous guest-blocked actor.",
                exc_info=True,
            )

        self._guest_blocked_actor = None

    def _clear_guest_bottleneck_actor(self) -> None:
        if self._guest_bottleneck_actor is None:
            return

        try:
            self.plotter.remove_actor(self._guest_bottleneck_actor)
        except Exception:
            logger.debug(
                "Could not remove previous guest-bottleneck actor.",
                exc_info=True,
            )

        self._guest_bottleneck_actor = None

    # ------------------------------------------------------------------
    # Open Metal Site (OMS) visualization
    # ------------------------------------------------------------------
    #
    # "Highlight OMS in viewer" and "show local environment" reuse
    # highlight_group() rather than a dedicated method - the visual
    # need (an outlined set of atoms, one group shown at a time) is
    # identical whether the group is "every detected open metal site"
    # (the overview) or "one site's metal plus its coordinating
    # neighbours" (inspection mode for a single row). Only the marker
    # for a potential adsorption position - a missing coordination
    # direction, which is not an atom at all - needs a new method.

    OMS_HIGHLIGHT_COLOR = "#ff5ea8"
    OMS_ADSORPTION_MARKER_COLOR = "#ffd23f"

    def show_oms_adsorption_markers(
        self, points, color: str = OMS_ADSORPTION_MARKER_COLOR
    ) -> None:
        """Small, distinct markers at each candidate adsorption
        position - the empty coordination direction(s) OMS detection
        identified, offset out from the metal centre so they read as
        "the open spot", not another ligand.
        """
        self._clear_oms_marker_actor()

        self._last_oms_marker_points = points

        if points is None or len(points) == 0:
            self.plotter.render()
            return

        cloud = pv.PolyData(np.asarray(points, dtype=float))

        self._oms_marker_actor = self.plotter.add_mesh(
            cloud,
            color=color,
            point_size=16,
            render_points_as_spheres=True,
            opacity=0.9,
            lighting=False,
        )

        self.plotter.render()

    def clear_oms_marker_visualization(self) -> None:
        self._clear_oms_marker_actor()
        self._last_oms_marker_points = None
        self.plotter.render()

    def set_oms_marker_visibility(self, visible: bool) -> None:
        if self._oms_marker_actor is not None:
            self._oms_marker_actor.SetVisibility(visible)
            self.plotter.render()

    def _clear_oms_marker_actor(self) -> None:
        if self._oms_marker_actor is None:
            return

        try:
            self.plotter.remove_actor(self._oms_marker_actor)
        except Exception:
            logger.debug(
                "Could not remove previous OMS marker actor.",
                exc_info=True,
            )

        self._oms_marker_actor = None

    # ------------------------------------------------------------------
    # Combined analysis-visualization clearing
    # ------------------------------------------------------------------

    def clear_all_analysis_visualizations(self) -> None:
        """Remove every analysis-result overlay from the 3D view in one
        call: the shared group highlight (MOF linkers/nodes,
        coordination spheres, OMS inspection), pore points, surface-area
        points and mesh, and every guest-accessibility and OMS overlay.

        Two callers: MainWindow wires this to an explicit "Clear
        Analysis Visualization" menu action (manual reset), and calls
        it automatically at the start of every analysis's own result
        handler, before that handler draws its own new overlay - so
        running one analysis after another always leaves exactly one
        analysis's results on screen, never an accumulating mix of
        whatever has been run so far in the session.

        Deliberately leaves clear_measurements() and clear_selection()
        alone: manually placed distance/angle measurements and the
        currently selected atom are not "analysis visualization" in
        the sense this method is about, and clearing them as a side
        effect of running an unrelated analysis would be surprising -
        a person measuring a pore's diameter by hand while also running
        Pore Analysis should not have their own measurement wiped out
        by the analysis they are cross-checking it against.
        """
        self.clear_group_highlight()
        self.clear_pore_visualization()
        self.clear_surface_visualization()
        self.clear_guest_accessibility_visualization()
        self.clear_oms_marker_visualization()
