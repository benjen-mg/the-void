from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

WARNING_STYLE = "color: #b3541e; font-weight: bold;"


class StructureInfoPanel(QWidget):
    """Crystallographic summary, plus what the bonding model rejected."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.formula_label = QLabel("-")
        self.spacegroup_label = QLabel("-")
        self.sitecount_label = QLabel("-")
        self.disorder_label = QLabel("-")
        #: Round 53: the underlying net (RCSR code), computed in the
        #: background after each load - see set_topology.
        self.topology_label = QLabel("-")
        self.topology_label.setWordWrap(True)
        #: Round 66: whether the shared index already has this structure,
        #: looked up in the background after each load - see
        #: set_index_matches.
        self.index_label = QLabel("-")
        self.index_label.setWordWrap(True)

        self.bondcount_label = QLabel("-")
        self.bond_min_label = QLabel("-")
        self.bond_max_label = QLabel("-")
        self.bond_avg_label = QLabel("-")
        self.short_contact_label = QLabel("-")

        self.fragment_label = QLabel("-")
        self.fragment_detail_label = QLabel("-")
        self.fragment_detail_label.setWordWrap(True)

        self.a_label = QLabel("-")
        self.b_label = QLabel("-")
        self.c_label = QLabel("-")
        self.alpha_label = QLabel("-")
        self.beta_label = QLabel("-")
        self.gamma_label = QLabel("-")
        self.volume_label = QLabel("-")
        self.density_label = QLabel("-")
        self.density_label.setToolTip(
            "Crystallographic density: the cell's contents (partially "
            "occupied sites counted by their occupancy) over its volume."
        )

        # Round 73: the groups sit in a scroll area, so a short window
        # scrolls them instead of squeezing them, and the element key
        # comes first, where it is seen at a glance.
        content = QWidget()
        layout = QVBoxLayout(content)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        #: Round 73: the element colour key (an ElementLegend, placed by
        #: set_element_legend).
        self.elements_group = QGroupBox("Elements")
        self.elements_group.setToolTip(
            "The colour of each element in the 3D view, and how many sites "
            "of it the cell holds. View > Customize Atom Colors... changes them.")
        self._elements_layout = QVBoxLayout(self.elements_group)
        layout.addWidget(self.elements_group)

        layout.addWidget(
            self._group(
                "Structure Information",
                [
                    ("Formula:", self.formula_label),
                    ("Space Group:", self.spacegroup_label),
                    ("Topology:", self.topology_label),
                    ("Previously Analysed:", self.index_label),
                    ("Sites in Cell:", self.sitecount_label),
                    ("Partial Occupancy:", self.disorder_label),
                ],
            )
        )

        layout.addWidget(
            self._group(
                "Bonding",
                [
                    ("Number of Bonds:", self.bondcount_label),
                    ("Minimum Bond (\u00c5):", self.bond_min_label),
                    ("Maximum Bond (\u00c5):", self.bond_max_label),
                    ("Average Bond (\u00c5):", self.bond_avg_label),
                    ("Short Contacts:", self.short_contact_label),
                ],
            )
        )

        layout.addWidget(
            self._group(
                "Fragments Drawn",
                [
                    ("Count:", self.fragment_label),
                    ("Composition:", self.fragment_detail_label),
                ],
            )
        )

        layout.addWidget(
            self._group(
                "Unit Cell",
                [
                    ("a (\u00c5):", self.a_label),
                    ("b (\u00c5):", self.b_label),
                    ("c (\u00c5):", self.c_label),
                    ("\u03b1 (\u00b0):", self.alpha_label),
                    ("\u03b2 (\u00b0):", self.beta_label),
                    ("\u03b3 (\u00b0):", self.gamma_label),
                    ("Volume (\u00c5\u00b3):", self.volume_label),
                    ("Density (g/cm\u00b3):", self.density_label),
                ],
            )
        )

        layout.addStretch()

    def set_element_legend(self, legend) -> None:
        """Show `legend` (an ElementLegend) in the Elements group."""
        self._elements_layout.addWidget(legend)

    @staticmethod
    def _group(title, rows):
        form = QFormLayout()

        for label, widget in rows:
            form.addRow(label, widget)

        group = QGroupBox(title)
        group.setLayout(form)

        return group

    def set_structure(self, info, stats, diagnostics=None, model=None):
        self.formula_label.setText(info.formula)
        self.spacegroup_label.setText(info.space_group)
        self.sitecount_label.setText(str(info.site_count))

        self._set_disorder(info, diagnostics)

        self.bondcount_label.setText(str(stats["count"]))
        self.bond_min_label.setText(f"{stats['min']:.3f}")
        self.bond_max_label.setText(f"{stats['max']:.3f}")
        self.bond_avg_label.setText(f"{stats['avg']:.3f}")

        self._set_short_contacts(diagnostics)
        self._set_fragments(model)

        self.a_label.setText(f"{info.a:.4f}")
        self.b_label.setText(f"{info.b:.4f}")
        self.c_label.setText(f"{info.c:.4f}")
        self.alpha_label.setText(f"{info.alpha:.2f}")
        self.beta_label.setText(f"{info.beta:.2f}")
        self.gamma_label.setText(f"{info.gamma:.2f}")

        structure = getattr(info, "structure", None)

        if structure is not None and hasattr(structure, "lattice"):
            from thevoid_core.services.report import crystal_density

            self.volume_label.setText(f"{structure.lattice.volume:.2f}")
            self.density_label.setText(f"{crystal_density(structure):.4f}")
        else:
            self.volume_label.setText("-")
            self.density_label.setText("-")

    def set_topology_pending(self) -> None:
        self.topology_label.setText("computing\u2026")
        self.topology_label.setStyleSheet("color: gray;")
        self.topology_label.setToolTip(
            "The underlying net is being worked out in the background."
        )

    def set_topology(self, result) -> None:
        """Show a TopologyResult: the RCSR code when the net was named,
        otherwise why not. The tooltip lists the nodes, how the net was
        derived and what it was matched against."""
        if result is None:
            self.topology_label.setText("-")
            self.topology_label.setStyleSheet("")
            self.topology_label.setToolTip("")
            return

        self.topology_label.setText(result.summary)
        self.topology_label.setStyleSheet(
            "" if result.status == "identified" else "color: gray;"
        )

        lines = []
        if result.status == "identified":
            lines.append(
                f"RCSR net {result.rcsr} ({result.representation} "
                "representation)."
            )
            descriptions = {c.description for c in result.components if c.description}
            lines += [f"Examples: {d}" for d in sorted(descriptions)]
        elif result.message:
            lines.append(result.message)
        if result.vertex_kinds:
            lines.append("")
            lines.append("Nodes (per cell):")
            lines += [
                f"  {k['count']} \u00d7 {k['formula']} ({k['kind']}), "
                f"{k['connectivity']}-connected"
                for k in result.vertex_kinds
            ]
        if result.edge_formulas:
            lines.append(
                "Edges from: "
                + ", ".join(f"{f}" for f in sorted(result.edge_formulas))
            )
        for note in result.notes:
            lines += ["", note]
        lines += ["", f"Matched against: {result.library}."]
        self.topology_label.setToolTip("\n".join(lines))

    # -- round 66: the shared index ----------------------------------

    def set_index_unset(self) -> None:
        self.index_label.setText("no shared index set")
        self.index_label.setStyleSheet("color: gray;")
        self.index_label.setToolTip(
            "File > Shared Index > Set Index Folder... chooses (or makes) the "
            "group's index; each structure you open is then looked up in it.")

    def set_index_pending(self) -> None:
        self.index_label.setText("looking in the shared index\u2026")
        self.index_label.setStyleSheet("color: gray;")
        self.index_label.setToolTip("")

    def set_index_matches(self, matches, index_name: str = "") -> None:
        """matches: [(relation, entry)] from Index.find, closest first."""
        where = f" in {index_name}" if index_name else ""
        if not matches:
            self.index_label.setText(f"not{where}")
            self.index_label.setStyleSheet("")
            self.index_label.setToolTip("No entry of this structure in the shared index.")
            return
        counts = {}
        for relation, _entry in matches:
            counts[relation] = counts.get(relation, 0) + 1
        close = sum(n for r, n in counts.items() if not r.startswith("same key"))
        if close:
            parts = [f"{n} {r}" for r, n in counts.items() if not r.startswith("same key")]
            self.index_label.setText(f"yes{where}: " + ", ".join(parts))
            self.index_label.setStyleSheet("color: #2e7d32;")
        else:
            self.index_label.setText(f"not this structure{where} "
                                     f"({len(matches)} with the same key)")
            self.index_label.setStyleSheet("")
        from thevoid_core.services.index.staleness import load_changes, staleness_of
        from thevoid_core.version import __version__

        changes = load_changes()
        lines = []
        for relation, entry in matches[:15]:
            a = entry["analysis"]
            summary = a.get("summary") or {}
            pld = summary.get("pld_A")
            stale = staleness_of(entry, changes, __version__)
            lines.append(f"{relation}: {entry['structure']['file_name']}, the Void "
                         f"{a['software'].get('the_void', '?')}, {entry['contributor']['name']} "
                         f"{entry['contributor']['added']}"
                         + (f", PLD {pld:.2f} \u00c5" if isinstance(pld, (int, float)) else "")
                         + (f" (stale: {', '.join(sorted(stale.columns))})" if stale.stale
                            else ""))
        if len(matches) > 15:
            lines.append(f"... and {len(matches) - 15} more")
        self.index_label.setToolTip("\n".join(lines))

    def set_index_failed(self, message: str) -> None:
        self.index_label.setText("lookup failed")
        self.index_label.setStyleSheet("color: gray;")
        self.index_label.setToolTip(message)

    def _set_disorder(self, info, diagnostics) -> None:
        partial = getattr(info, "partial_occupancy_sites", 0)

        if not partial and diagnostics is not None:
            partial = diagnostics.partial_occupancy_sites

        if partial:
            self.disorder_label.setText(
                f"{partial} site(s) \u2014 structure is disordered"
            )
            self.disorder_label.setStyleSheet(WARNING_STYLE)
        else:
            self.disorder_label.setText("none")
            self.disorder_label.setStyleSheet("")

    def _set_short_contacts(self, diagnostics) -> None:
        if diagnostics is None or not diagnostics.short_contact_count:
            self.short_contact_label.setText("none")
            self.short_contact_label.setStyleSheet("")
            self.short_contact_label.setToolTip("")
            return

        shortest = diagnostics.shortest_rejected

        self.short_contact_label.setText(
            f"{diagnostics.short_contact_count} rejected "
            f"(shortest {shortest:.3f} \u00c5)"
        )
        self.short_contact_label.setStyleSheet(WARNING_STYLE)

        lines = [
            "Pairs too close to be a bond, so not drawn.",
            "Usually alternative positions of one disordered atom.",
            "",
        ]
        lines += [
            f"{c.symbol1}{c.atom1 + 1}\u2013{c.symbol2}{c.atom2 + 1}: "
            f"{c.distance:.3f} \u00c5"
            for c in sorted(
                diagnostics.short_contacts, key=lambda c: c.distance
            )[:12]
        ]

        self.short_contact_label.setToolTip("\n".join(lines))

    def _set_fragments(self, model) -> None:
        if model is None or not model.fragments:
            self.fragment_label.setText("-")
            self.fragment_detail_label.setText("-")
            return

        self.fragment_label.setText(str(model.fragment_count))

        self.fragment_detail_label.setText(
            "  +  ".join(
                f"{fragment.occurrences} \u00d7 {fragment.formula}"
                for fragment in model.fragments[:8]
            )
        )

        self.fragment_detail_label.setToolTip(
            "Connected components of the drawn structure. Small ones "
            "(counter-ions, solvent) are expected to float free; a large "
            "unattached fragment usually means a bond was missed."
        )
