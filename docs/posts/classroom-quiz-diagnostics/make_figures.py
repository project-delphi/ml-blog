"""Draw the heatmap and print every number the post quotes about the synthetic class.

Run from this folder:
    uv run --no-project --with-requirements requirements.txt python make_figures.py
"""

from __future__ import annotations

import dataclasses
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from diagnostics import (  # noqa: E402
    MASTERY,
    misconceptions,
    objective_cells,
    student_coverage,
)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from synthetic import OBJECTIVES, simulate  # noqa: E402

SEEDS = range(20)
FIGURE_SEED = 0

SURFACE, INK, INK_2, MUTED, HAIRLINE = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9",
)  # fmt: skip
CRITICAL = "#d03b3b"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def every_answer_on_time(c):
    """The counterfactual: every answer arrived before the reveal."""
    return [dataclasses.replace(r, timing="on_time") for r in c.responses] + c.lost


def quoted_numbers() -> dict[str, tuple[float, float]]:
    """Ranges over seeds for each number the post states."""
    runs: dict[str, list[float]] = {
        "mean shift": [],
        "worst cell shift": [],
        "answer rate": [],
        "interval width": [],
        "unclear share": [],
        "usable share, weak": [],
        "usable share, good": [],
        "O4 top misconception share": [],
    }
    for seed in SEEDS:
        c = simulate(seed)
        seen = objective_cells(c.responses, c.items, c.present)
        full = {
            (x.session_id, x.objective_id): x
            for x in objective_cells(every_answer_on_time(c), c.items, c.present)
        }
        shifts = [x.mastery - full[x.session_id, x.objective_id].mastery for x in seen]
        runs["mean shift"].append(statistics.mean(shifts))
        runs["worst cell shift"].append(max(shifts, key=abs))
        runs["answer rate"].append(statistics.mean(x.answer_rate for x in seen))
        runs["interval width"].append(statistics.mean(x.high - x.low for x in seen))
        runs["unclear share"].append(
            sum(x.verdict == "unclear" for x in seen) / len(seen)
        )
        students = {r.player_id for r in c.responses + c.lost}
        coverage = dict(
            student_coverage(c.responses, {s: len(c.present) for s in students})
        )
        runs["usable share, weak"].append(statistics.mean(coverage[s] for s in c.weak))
        runs["usable share, good"].append(
            statistics.mean(coverage[s] for s in students - c.weak)
        )
        tags = misconceptions(c.responses, c.items, "O4")
        runs["O4 top misconception share"].append(tags[0][1] / sum(n for _, n in tags))
    return {k: (min(v), max(v)) for k, v in runs.items()}


def draw(path: str = "heatmap.png") -> None:
    """Objectives by week for one seed, judged against the mastery bar."""
    c = simulate(FIGURE_SEED)
    cells = {
        (x.objective_id, x.session_id): x
        for x in objective_cells(c.responses, c.items, c.present)
    }
    objectives, weeks = list(OBJECTIVES), c.sessions
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)

    fig, ax = plt.subplots(figsize=(8.4, 5.4), dpi=240)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for row, objective in enumerate(objectives):
        for col, week in enumerate(weeks):
            cell = cells.get((objective, week))
            x, y = col, len(objectives) - 1 - row
            if cell is None:
                ax.add_patch(
                    Rectangle(
                        (x + 0.04, y + 0.04), 0.92, 0.92, fill=False,
                        edgecolor=HAIRLINE, linewidth=0.8,
                    )
                )  # fmt: skip
                continue
            fill = cmap(cell.mastery)
            r, g, b, _ = fill
            dark = 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.42  # rough luminance
            ink = "#ffffff" if dark else INK
            ax.add_patch(
                Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, color=fill, linewidth=0)
            )
            ax.text(x + 0.5, y + 0.68, f"{cell.mastery:.2f}", ha="center",
                    va="center", fontsize=13, color=ink, weight="bold")  # fmt: skip
            ax.text(x + 0.5, y + 0.4, f"{cell.low:.2f}\u2013{cell.high:.2f}",
                    ha="center", va="center", fontsize=8.5, color=ink)  # fmt: skip
            ax.text(x + 0.5, y + 0.19, f"{cell.answer_rate:.0%} in",
                    ha="center", va="center", fontsize=8.5, color=ink)  # fmt: skip
            if cell.verdict == "gap":
                ax.add_patch(
                    Rectangle(
                        (x + 0.06, y + 0.06), 0.88, 0.88, fill=False,
                        edgecolor=CRITICAL, linewidth=2.6,
                    )
                )  # fmt: skip
                ax.text(x + 0.11, y + 0.86, "\u25bc", ha="left", va="top",
                        fontsize=9, color=CRITICAL)  # fmt: skip
            elif cell.verdict == "secure":
                ax.text(x + 0.11, y + 0.86, "\u2713", ha="left", va="top",
                        fontsize=10, color=ink, weight="bold")  # fmt: skip

    ax.set_xlim(0, len(weeks))
    ax.set_ylim(0, len(objectives))
    ax.set_xticks([i + 0.5 for i in range(len(weeks))])
    ax.set_xticklabels([f"Week {i + 1}" for i in range(len(weeks))])
    ax.set_yticks([i + 0.5 for i in range(len(objectives))])
    ax.set_yticklabels(list(reversed(objectives)))
    ax.tick_params(length=0, colors=INK_2, labelsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(
        0.5, 0.045,
        "Big number: mean share correct, one vote per student.  Then: 95% Wilson "
        "interval; share of possible answers\nthat arrived before the reveal.   "
        f"\u25bc gap: whole interval under {MASTERY}.   \u2713 secure: whole "
        f"interval over {MASTERY}.   Unmarked: straddles {MASTERY}.",
        ha="center", va="bottom", fontsize=8.5, color=INK_2, linespacing=1.6,
    )  # fmt: skip
    fig.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.16)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    draw()
    for name, (low, high) in quoted_numbers().items():
        print(f"{name:28s} {low:+.3f} .. {high:+.3f}")
