"""Publication figures from a batch (the-void-figures)."""

from __future__ import annotations

from pathlib import Path


class Figure:
    """A drawn figure: `figure` is the matplotlib Figure, `points` and
    `excluded` what was plotted and left out (with reasons)."""

    def __init__(self, figure, data, spec, style, batch):
        self.figure, self._data, self._spec, self._style, self._batch = \
            figure, data, spec, style, batch

    @property
    def points(self) -> list:
        """[(file, x, y, category)]."""
        return list(self._data.points)

    @property
    def excluded(self) -> list:
        """[(file, reason)]."""
        return list(self._data.excluded)

    def save(self, base_path, *, filters=()) -> list:
        """SVG, PDF, 600-dpi PNG and the JSON record, as base_path.<ext>."""
        from thevoid_core.services import figures

        return figures.save(self.figure, self._data, self._spec, self._style, Path(base_path),
                            results=self._batch, filters=filters)

    def _repr_png_(self):  # Jupyter
        import io

        buffer = io.BytesIO()
        self.figure.savefig(buffer, format="png", dpi=150)
        return buffer.getvalue()


def figure(batch, kind: str = "scatter", x: str = "pld_A", y: str | None = "channel_asa_m2_per_g",
           *, colour_by: str | None = None, log_x: bool = False, log_y: bool = False,
           bins: int | None = None, title: str = "", width: str = "single", filters=(),
           save=None) -> Figure:
    """A figure (kind "scatter", "histogram" or "parity") from a
    BatchTable or a results folder, drawn as the-void-figures draws it.
    filters: ("pld_A >= 3.3", "errors is empty", ...). save: a base path
    to write every format and the JSON to."""
    from thevoid_core.figures_cli import parse_filter
    from thevoid_core.services import figures
    from thevoid_core.services.batch_results import load_batch_results, view
    from thevoid.batch import BatchTable

    results = batch.results() if isinstance(batch, BatchTable) else load_batch_results(batch)
    parsed = [parse_filter(f) if isinstance(f, str) else f for f in filters]
    rows = view(results, filters=parsed)
    spec = figures.FigureSpec(kind, x, None if kind == "histogram" else y, colour_by=colour_by,
                              log_x=log_x, log_y=log_y, bins=bins, title=title)
    style = figures.FigureStyle(width=width)
    fig, data = figures.draw(results, rows, spec, style)
    result = Figure(fig, data, spec, style, results)
    if save is not None:
        result.save(save, filters=parsed)
    return result
