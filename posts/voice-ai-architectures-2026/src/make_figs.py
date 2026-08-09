"""Draw the four figures for the voice-AI-architectures post.

Every number below is transcribed from a primary source and carries that source
inline, so refreshing the post means editing this file and re-running it. The
``self_reported`` flag marks numbers published by the party whose model is being
measured; ``False`` means a third party ran the evaluation.

This post has no venv of its own -- run it with any venv that has matplotlib, or
ephemerally::

    uv run --with matplotlib python posts/voice-ai-architectures-2026/src/make_figs.py

Retrieval date for every leaderboard figure: 2026-08-09.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent

# Categorical palette, validated for colour-vision deficiency and contrast
# against a light surface (worst adjacent pair dE 27.1 protan / 29.9 tritan).
CLOSED = "#4A3AA7"  # proprietary / API-only
OPEN = "#C2570A"  # open weights
# Two steps of the house hue, for the one chart whose two series are stages of
# the same response rather than different kinds of model -- reusing the
# open/closed pair there would overload what orange means elsewhere.
STEP_LIGHT = "#8B7FD4"
INK = "#1F1D24"
MUTED = "#6B6673"
GRID = "#DCDAE2"

WIDTH = 9.0
DPI = 130

# --------------------------------------------------------------------------
# Figure 1 -- latency, cascaded vs native, one harness.
# Source: Full-Duplex-Bench-v3, arXiv:2604.04847, Table 6 (mean seconds).
# All six configurations were driven through LiveKit on identical audio from a
# single fixed server region. Third-party (NTU + NVIDIA), not self-reported.
# --------------------------------------------------------------------------
LATENCY = {
    "source": "Full-Duplex-Bench-v3 (arXiv:2604.04847), Table 6",
    "self_reported": False,
    # label, first-word s, task-completion s, is_cascade
    "rows": [
        ("Ultravox v0.7  (open weights)", 3.88, 8.40, False),
        ("Gemini Live 3.1", 3.95, 4.25, False),
        ("Grok", 5.97, 6.65, False),
        ("GPT-Realtime", 6.36, 6.89, False),
        ("Gemini Live 2.5", 7.03, 7.26, False),
        ("Cascaded: Whisper→GPT-4o→TTS", 8.78, 10.12, True),
    ],
}

# --------------------------------------------------------------------------
# Figure 2 -- English short-form ASR word error rate.
# Source: Hugging Face Open ASR Leaderboard results dataset
# (hf-audio/open-asr-leaderboard-results), "avg cleaned" = macro-average over
# seven English short-form sets (AMI, Earnings22, GigaSpeech, LibriSpeech
# clean/other, SPGISpeech, VoxPopuli) after text normalisation.
# One harness for open weights and commercial APIs alike.
# --------------------------------------------------------------------------
WER = {
    "source": "HF Open ASR Leaderboard results dataset, avg WER (normalised)",
    "self_reported": False,
    # label, avg WER %, is_open
    "rows": [
        ("Microsoft Azure Speech (06-2026)", 4.51, False),
        ("ElevenLabs Scribe v2", 4.65, False),
        ("IBM Granite Speech 4.1 2B", 4.90, True),
        ("Qwen3-ASR 1.7B", 5.02, True),
        ("AssemblyAI Universal-3.5 Pro", 5.03, False),
        ("NVIDIA Canary-Qwen 2.5B", 5.06, True),
        ("Cohere Transcribe (03-2026)", 5.20, True),
        ("NVIDIA Parakeet TDT 0.6B v2", 5.39, True),
        ("Kyutai STT 2.6B", 5.74, True),
        ("Speechmatics Enhanced", 5.90, False),
        ("OpenAI Whisper large-v3", 6.55, True),
    ],
}

# --------------------------------------------------------------------------
# Figure 3 -- text-to-speech listening-preference Elo.
# Source: Artificial Analysis Speech Arena, provider-voice leaderboard.
# Blind pairwise votes from arena listeners; third-party, not self-reported.
# Open-weights flag is the leaderboard's own.
# --------------------------------------------------------------------------
TTS = {
    "source": "Artificial Analysis Speech Arena (provider voices)",
    "self_reported": False,
    # label, Elo, +/- 95% CI, is_open
    "rows": [
        ("Qwen-Audio-3.0-TTS-Plus", 1229, 15, False),
        ("Gemini 3.1 Flash TTS", 1210, 13, False),
        ("Cartesia Sonic 3.5", 1203, 13, False),
        ("Inworld Realtime TTS 1.5 Max", 1194, 13, False),
        ("ElevenLabs Eleven v3", 1171, 12, False),
        ("Fish Audio S2 Pro", 1121, 14, True),
        ("StepFun Step Audio EditX", 1109, 14, True),
        ("OpenAI TTS-1 HD", 1097, 12, False),
        ("Mistral Voxtral TTS", 1067, 14, True),
        ("NVIDIA Magpie-Multilingual 357M", 1065, 14, True),
        ("Kokoro 82M v1.0", 1056, 11, True),
        ("Resemble Chatterbox", 1014, 12, True),
        ("Coqui XTTS v2", 914, 15, True),
    ],
}

# --------------------------------------------------------------------------
# Figure 4 -- the reasoning gap in native speech-to-speech.
# Source: Artificial Analysis Speech to Speech leaderboard.
# Speech Reasoning = Big Bench Audio (1,000 audio questions from Big Bench
# Hard). Conversational Dynamics = a subset of Full-Duplex-Bench v1 and v1.5.
# Both run by Artificial Analysis, not by the model vendors.
# --------------------------------------------------------------------------
S2S = {
    "source": "Artificial Analysis Speech to Speech leaderboard",
    "self_reported": False,
    # label, speech reasoning %, conversational dynamics %, is_open
    "rows": [
        ("Qwen Audio 3.0 Realtime Plus", 99, 98.4, False),
        ("GPT-Realtime-2.1 High", 96, 95.7, False),
        ("Grok Voice Think Fast 2.0", 97, 95.1, False),
        ("Gemini 3.1 Flash (High)", 97, 74.3, False),
        ("Deepslate Opal", 85, 85.7, False),
        ("Qwen3 Omni Flash", 59, 72.7, False),
        ("PersonaPlex", 19, 91.0, True),
        ("Nemotron 3 VoiceChat", 27, 52.9, True),
        ("Freeze-Omni", 33, 58.7, True),
        ("FLM-Audio", 16, 62.0, True),
        ("Moshi", 4, 61.0, True),
    ],
}


def _style(ax):
    """Apply the shared recessive-axis styling to ``ax``."""
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", colors=MUTED, length=0, labelsize=9.5)


def fig_latency() -> None:
    """Grouped bars: first-word and task-completion latency, seconds."""
    rows = LATENCY["rows"]
    labels = [r[0] for r in rows]
    first = [r[1] for r in rows]
    done = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(WIDTH, 4.6), dpi=DPI)
    y = range(len(rows))
    h = 0.36
    ax.barh(
        [i + h / 2 + 0.02 for i in y],
        first,
        height=h,
        color=STEP_LIGHT,
        label="First word",
    )
    ax.barh(
        [i - h / 2 - 0.02 for i in y],
        done,
        height=h,
        color=CLOSED,
        label="Task completion",
    )

    for i, (f, d) in enumerate(zip(first, done)):
        ax.text(f + 0.12, i + h / 2 + 0.02, f"{f:.2f}", va="center", fontsize=8.5, color=INK)
        ax.text(d + 0.12, i - h / 2 - 0.02, f"{d:.2f}", va="center", fontsize=8.5, color=INK)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 12.4)
    ax.set_xlabel("seconds (lower is better)", fontsize=9.5, color=MUTED)
    ax.set_title(
        "Latency on multi-step tool tasks with real, disfluent speech",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    _style(ax)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig-latency.png", facecolor="white")
    plt.close(fig)


def fig_wer() -> None:
    """Bars: average word error rate, open weights vs commercial API."""
    rows = WER["rows"]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.8), dpi=DPI)
    colors = [OPEN if r[2] else CLOSED for r in rows]
    vals = [r[1] for r in rows]
    ax.barh(range(len(rows)), vals, height=0.62, color=colors)

    for i, v in enumerate(vals):
        ax.text(v + 0.06, i, f"{v:.2f}", va="center", fontsize=8.5, color=INK)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 7.6)
    ax.set_xlabel("average word error rate, % (lower is better)", fontsize=9.5, color=MUTED)
    ax.set_title(
        "Speech-to-text accuracy, open weights vs commercial APIs",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    _style(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OPEN),
        plt.Rectangle((0, 0), 1, 1, color=CLOSED),
    ]
    ax.legend(
        handles,
        ["Open weights", "Commercial API"],
        frameon=False,
        fontsize=9.5,
        loc="upper right",
        labelcolor=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig-wer.png", facecolor="white")
    plt.close(fig)


def fig_tts() -> None:
    """Bars with 95% CI: TTS listening-preference Elo."""
    rows = TTS["rows"]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.2), dpi=DPI)
    colors = [OPEN if r[3] else CLOSED for r in rows]
    vals = [r[1] for r in rows]
    errs = [r[2] for r in rows]
    # Elo is an interval scale with no meaningful zero, so this is a dot plot:
    # bar length from an arbitrary baseline would overstate the differences.
    ax.errorbar(
        vals,
        range(len(rows)),
        xerr=errs,
        fmt="none",
        ecolor=MUTED,
        elinewidth=1.4,
        capsize=3,
    )
    ax.scatter(vals, range(len(rows)), s=88, color=colors, zorder=3)

    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 9, i, f"{v:,}", va="center", fontsize=8.5, color=INK)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(880, 1290)
    ax.set_xlabel("Arena Elo, blind listening votes (higher is better)", fontsize=9.5, color=MUTED)
    ax.set_title(
        "Text-to-speech listening preference, 95% confidence intervals",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    _style(ax)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=OPEN),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=CLOSED),
    ]
    ax.legend(
        handles,
        ["Open weights", "Proprietary"],
        frameon=False,
        fontsize=9.5,
        loc="lower right",
        labelcolor=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig-tts.png", facecolor="white")
    plt.close(fig)


def fig_s2s() -> None:
    """Scatter: speech reasoning against conversational dynamics."""
    rows = S2S["rows"]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.4), dpi=DPI)

    # Points above ~80% reasoning sit at the right edge, so their labels go to
    # the left; the two OpenAI/xAI points nearly coincide and are split apart.
    nudge = {
        "GPT-Realtime-2.1 High": (-11, 6, "right"),
        "Grok Voice Think Fast 2.0": (-11, -12, "right"),
    }
    for label, reasoning, dynamics, is_open in rows:
        color = OPEN if is_open else CLOSED
        ax.scatter(
            reasoning,
            dynamics,
            s=96,
            color=color,
            edgecolor="white",
            linewidth=1.6,
            zorder=3,
        )
        if label in nudge:
            dx, dy, ha = nudge[label]
        elif reasoning > 80:
            dx, dy, ha = -11, -3, "right"
        else:
            dx, dy, ha = 9, -3, "left"
        ax.annotate(
            label,
            (reasoning, dynamics),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.6,
            color=INK,
            ha=ha,
        )

    ax.set_xlim(-4, 108)
    ax.set_ylim(45, 105)
    ax.set_xlabel(
        "Speech reasoning — Big Bench Audio, % correct (higher is better)",
        fontsize=9.5,
        color=MUTED,
    )
    ax.set_ylabel(
        "Conversational dynamics — Full-Duplex-Bench, %",
        fontsize=9.5,
        color=MUTED,
    )
    ax.set_title(
        "Native speech-to-speech: open models handle the conversation, not the question",
        fontsize=12.5,
        color=INK,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    _style(ax)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=OPEN),
        plt.Line2D([], [], marker="o", linestyle="", markersize=9, color=CLOSED),
    ]
    ax.legend(
        handles,
        ["Open weights", "Proprietary"],
        frameon=False,
        fontsize=9.5,
        loc="lower right",
        labelcolor=INK,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig-s2s-gap.png", facecolor="white")
    plt.close(fig)


def main() -> int:
    """Write all four figures.

    Returns:
        Process exit code.
    """
    fig_latency()
    fig_wer()
    fig_tts()
    fig_s2s()
    print(f"wrote 4 figures to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
