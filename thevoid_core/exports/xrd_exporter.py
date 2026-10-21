import csv


class XRDExporter:
    """Write an XRDResult out as CSV."""

    @staticmethod
    def export_csv(filename, result) -> None:
        rows = zip(
            result.two_theta,
            result.intensities,
            result.d_spacings,
            getattr(result, "hkls", None) or [None] * len(result.two_theta),
        )

        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)

            for line in XRDExporter._provenance(result):
                writer.writerow([line])

            writer.writerow(
                [
                    "2theta_deg",
                    "rel_intensity_pct",
                    "d_spacing_A",
                    "hkl",
                ]
            )

            for two_theta, intensity, d_spacing, hkl in rows:
                writer.writerow(
                    [
                        two_theta,
                        intensity,
                        d_spacing,
                        XRDExporter.format_hkl(hkl),
                    ]
                )

    @staticmethod
    def export_profile_csv(filename, two_theta, intensities, *, wavelength=None, fwhm=None) -> None:
        """Write a convolved profile trace out as CSV.

        Two plain columns, 2-theta and intensity, one row per sampled
        point - exactly the shape needed to paste into Excel and plot
        as a line chart, which the stick file (one row per discrete
        peak) can't give you: a stick CSV has no points *between* peaks,
        so a line chart of it looks like a jagged mess rather than a
        trace.
        """
        with open(filename, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)

            if wavelength:
                writer.writerow([f"# wavelength_A={wavelength:.6f}"])

            if fwhm:
                writer.writerow([f"# fwhm_deg_2theta={fwhm:g}"])

            writer.writerow([f"# points={len(two_theta)}"])

            writer.writerow(["2theta_deg", "rel_intensity_pct"])

            for two_theta_value, intensity in zip(two_theta, intensities):
                writer.writerow([two_theta_value, intensity])

    @staticmethod
    def format_hkl(hkl) -> str:
        """Render pymatgen's hkl payload as e.g. '(1 1 1)'.

        get_pattern returns a list of dicts per reflection, since several
        symmetry-equivalent planes can share a 2theta. Overlapping entries
        are joined with a comma.
        """
        if not hkl:
            return ""

        entries = hkl if isinstance(hkl, (list, tuple)) else [hkl]

        rendered = []

        for entry in entries:
            indices = entry.get("hkl") if isinstance(entry, dict) else entry

            if indices is None:
                continue

            rendered.append(
                "(" + " ".join(str(int(i)) for i in indices) + ")"
            )

        return ", ".join(rendered)

    @staticmethod
    def _provenance(result) -> list[str]:
        """Header comments recording how the peak list was reduced.

        Without these a filtered export is indistinguishable from a
        complete one, and someone reading the file later has no way to
        know which reflections were left out.
        """
        lines = []

        wavelength = getattr(result, "wavelength", None)

        if wavelength:
            lines.append(f"# wavelength_A={wavelength:.6f}")

        threshold = getattr(result, "min_relative_intensity", 0) or 0

        if threshold > 0:
            lines.append(f"# min_relative_intensity_pct={threshold:g}")

        tolerance = getattr(result, "merge_tolerance", 0) or 0

        if tolerance > 0:
            lines.append(f"# merged_within_deg_2theta={tolerance:g}")

        total = getattr(result, "unfiltered_peak_count", 0) or 0

        if total and total != len(result.two_theta):
            lines.append(
                f"# peaks_listed={len(result.two_theta)} of {total} calculated"
            )

        return lines
