"""Figure layout QA: find text artists that collide.

Matplotlib will happily draw a title through a legend, or stack tick labels on
top of an axis label, and nothing in the render pipeline complains.  This module
renders a figure, walks every text artist it can reach, computes window extents
in display space, and reports any pair overlapping by more than a threshold
area.

Two exclusions matter, and without them the checker produces false positives
that drown the real findings:

* axes with ``axison == False`` — their tick labels are not drawn at all;
* tick labels whose locator position falls outside the current view limits —
  matplotlib keeps the artists around but does not render them.

Used as a build gate: :func:`check_all` raises on any real overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# Two labels touching by a few square pixels is antialiasing, not a collision.
MIN_OVERLAP_AREA: float = 60.0


@dataclass(frozen=True)
class Overlap:
    """A detected collision between two text artists.

    Attributes:
        figure: Label of the figure the collision was found in.
        first: Text content of the first artist.
        second: Text content of the second artist.
        area: Overlap area in square display pixels.
    """

    figure: str
    first: str
    second: str
    area: float

    def __str__(self) -> str:
        return (
            f"{self.figure}: {self.area:.0f}px^2 between "
            f"{self.first!r} and {self.second!r}"
        )


def _tick_texts(ax: Axes) -> list[Text]:
    """Return tick labels that are actually rendered on ``ax``.

    Skips invisible axes entirely, and drops tick labels whose data position is
    outside the view limits — matplotlib retains those artists but never draws
    them, so counting them yields phantom collisions off the edge of the plot.
    """
    if not ax.axison:
        return []
    out: list[Text] = []
    for axis, (lo, hi) in (
        (ax.xaxis, sorted(ax.get_xlim())),
        (ax.yaxis, sorted(ax.get_ylim())),
    ):
        for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs()):
            if not lo <= loc <= hi:
                continue
            for label in (tick.label1, tick.label2):
                if label.get_visible() and label.get_text().strip():
                    out.append(label)
    return out


def _texts_of(fig: Figure) -> list[Text]:
    """Collect every candidate text artist on a figure."""
    items: list[Text] = [t for t in fig.texts if t.get_visible() and t.get_text().strip()]
    for ax in fig.axes:
        for candidate in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if candidate.get_visible() and candidate.get_text().strip():
                items.append(candidate)
        items.extend(t for t in ax.texts if t.get_visible() and t.get_text().strip())
        items.extend(_tick_texts(ax))
        legend = ax.get_legend()
        if legend is not None and legend.get_visible():
            items.extend(t for t in legend.get_texts() if t.get_text().strip())
    return items


def _overlap_area(a: Bbox, b: Bbox) -> float:
    """Area of the intersection of two boxes, zero if they do not meet."""
    dx = min(a.x1, b.x1) - max(a.x0, b.x0)
    dy = min(a.y1, b.y1) - max(a.y0, b.y0)
    return dx * dy if dx > 0.0 and dy > 0.0 else 0.0


def _renderer(fig: Figure):
    """Return a renderer that can measure text, whatever backend is active.

    Under a plain ``FigureCanvasBase`` — which is what a figure carries inside
    Quarto's inline-plot machinery — there is no ``get_renderer``.  Attaching an
    Agg canvas gives one.  This mutates ``fig.canvas``, which is harmless
    because the check runs after the figure has already been displayed.
    """
    get = getattr(fig.canvas, "get_renderer", None)
    if get is not None:
        return get()
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    return FigureCanvasAgg(fig).get_renderer()


def check_figure(fig: Figure, label: str) -> list[Overlap]:
    """Report text collisions on a single figure.

    Args:
        fig: The figure to check.  It is drawn first so extents are valid.
        label: Name used in the report.

    Returns:
        A list of overlaps, empty if the layout is clean.
    """
    renderer = _renderer(fig)
    fig.canvas.draw()
    texts = _texts_of(fig)
    boxes = [t.get_window_extent(renderer=renderer) for t in texts]

    found: list[Overlap] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            area = _overlap_area(boxes[i], boxes[j])
            if area > MIN_OVERLAP_AREA:
                found.append(
                    Overlap(label, texts[i].get_text(), texts[j].get_text(), area)
                )
    return found


def check_all(figures: dict[str, Figure], strict: bool = True) -> list[Overlap]:
    """Check every figure and, by default, fail the build on any collision.

    Args:
        figures: Mapping of label to figure.
        strict: Raise on collisions rather than only returning them.

    Returns:
        Every overlap found, across all figures.

    Raises:
        AssertionError: If ``strict`` and any figure has a collision.
    """
    found: list[Overlap] = []
    for label, fig in figures.items():
        found.extend(check_figure(fig, label))
    if found and strict:
        report = "\n".join(f"  {o}" for o in found)
        raise AssertionError(f"{len(found)} text collision(s):\n{report}")
    return found


def report(figures: dict[str, Figure]) -> str:
    """Human-readable summary, for printing at the end of the post.

    Args:
        figures: Mapping of label to figure.

    Returns:
        One line per figure plus a verdict.
    """
    lines: list[str] = []
    total = 0
    for label, fig in figures.items():
        found = check_figure(fig, label)
        total += len(found)
        mark = "ok" if not found else f"{len(found)} OVERLAP"
        lines.append(f"  {label:38s} {mark}")
    lines.append(f"  {'-' * 48}")
    lines.append(f"  {len(figures)} figures checked, {total} collisions")
    return "\n".join(lines)


if __name__ == "__main__":
    # Self-test: a figure built to collide must be caught, a clean one must pass.
    bad, ax = plt.subplots(figsize=(3, 2))
    ax.set_title("a deliberately long title that runs into things")
    ax.text(0.5, 0.98, "a deliberately long title that runs into things",
            ha="center", va="top", transform=ax.transAxes)
    assert check_figure(bad, "deliberate-collision"), "checker missed a real overlap"

    good, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot([0, 1], [0, 1])
    ax2.set_title("clean")
    assert not check_figure(good, "clean"), "checker invented an overlap"
    print("layout_qa self-test passed")
